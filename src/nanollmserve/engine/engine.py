"""Single-request and static-batch generation with Hugging Face KV cache reuse."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from nanollmserve.cache.block_manager import KVBlockManager
from nanollmserve.cache.prefix_cache import PrefixCache, PrefixCacheLookup
from nanollmserve.engine.request import GenerationRequestState
from nanollmserve.engine.scheduler import (
    ContinuousBatchRequest,
    ContinuousBatchScheduler,
    SchedulerStepStats,
)
from nanollmserve.sampling.sampler import sample_next_token


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    text: str
    generated_token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    ttft_seconds: float = 0.0
    tpot_seconds: float = 0.0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    finished: bool = False

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.generated_tokens / self.elapsed_seconds


@dataclass(frozen=True)
class GenerationStep:
    prompt: str
    token_id: int
    text: str
    generated_text: str
    generated_token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    ttft_seconds: float
    tpot_seconds: float
    prefill_seconds: float
    decode_seconds: float
    finished: bool


@dataclass(frozen=True)
class ContinuousBatchGenerationResult:
    request_id: str
    result: GenerationResult
    arrival_step: int
    admitted_step: int
    finished_step: int


@dataclass(frozen=True)
class ContinuousBatchRunResult:
    results: list[ContinuousBatchGenerationResult]
    scheduler_steps: list[SchedulerStepStats]

    @property
    def active_batch_sizes(self) -> list[int]:
        return [step.active_batch_size for step in self.scheduler_steps]


def _model_device(model):
    import torch

    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _move_batch_to_device(batch: dict, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def _eos_token_ids(tokenizer) -> set[int]:
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        return set()
    if isinstance(eos, int):
        return {eos}
    return set(eos)


def generate_one(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    seed: int | None = None,
    kv_block_manager: KVBlockManager | None = None,
    prefix_cache: PrefixCache | None = None,
    request_id: str = "request-0",
) -> GenerationResult:
    """Generate text for one prompt with prefill/decode KV cache reuse."""

    steps = list(
        stream_generate_one(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            kv_block_manager=kv_block_manager,
            prefix_cache=prefix_cache,
            request_id=request_id,
        )
    )
    if not steps:
        raise RuntimeError("generation produced no steps")

    final_step = steps[-1]
    return GenerationResult(
        prompt=prompt,
        text=final_step.generated_text,
        generated_token_ids=final_step.generated_token_ids,
        prompt_tokens=final_step.prompt_tokens,
        generated_tokens=final_step.generated_tokens,
        elapsed_seconds=final_step.elapsed_seconds,
        ttft_seconds=final_step.ttft_seconds,
        tpot_seconds=final_step.tpot_seconds,
        prefill_seconds=final_step.prefill_seconds,
        decode_seconds=final_step.decode_seconds,
        finished=final_step.finished,
    )


def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    seed: int | None = None,
    kv_block_manager: KVBlockManager | None = None,
    request_ids: list[str] | None = None,
) -> list[GenerationResult]:
    """Generate text for a fixed batch in one KV-cache prefill and decode loop.

    This is a teaching-scale static batching path: requests are fixed and all are
    advanced in lock-step until whole-batch completion.
    """

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if not prompts:
        raise ValueError("prompts must contain at least one entry")
    if any(prompt == "" for prompt in prompts):
        raise ValueError("all prompts must be non-empty")
    if request_ids is not None and len(request_ids) != len(prompts):
        raise ValueError("request_ids must match prompts length")

    device = _model_device(model)
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )
    encoded = _move_batch_to_device(encoded, device)
    if "attention_mask" not in encoded:
        # Most tokenizers return this, but keep a defensive path for minimal runners.
        encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"].to(dtype=torch.long)
    prompt_lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
    eos_token_ids = _eos_token_ids(tokenizer)
    eos_token_id = next(iter(eos_token_ids), 0)

    states: list[GenerationRequestState] = []
    block_request_ids = request_ids or [f"batch-{idx}" for idx in range(len(prompts))]
    for idx, prompt in enumerate(prompts):
        length = int(prompt_lengths[idx].item())
        state = GenerationRequestState(
            prompt=prompt,
            prompt_token_ids=[
                int(token_id) for token_id in input_ids[idx, :length].tolist()
            ],
            attention_mask=attention_mask[idx, :length].clone(),
        )
        states.append(state)

    allocated_request_ids: list[str] = []
    try:
        for request_id, state in zip(block_request_ids, states):
            _allocate_prompt_blocks(kv_block_manager, request_id, state.prompt_tokens)
            allocated_request_ids.append(request_id)

        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
        else:
            generator = None

        start = perf_counter()
        model.eval()
        with torch.inference_mode():
            prefill_start = perf_counter()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            prefill_seconds = perf_counter() - prefill_start

            past_key_values = getattr(outputs, "past_key_values", None)
            if past_key_values is None:
                raise RuntimeError("model did not return past_key_values; KV cache decode requires use_cache support")

            next_logits = _select_last_token_logits(outputs.logits, attention_mask)
            next_tokens = _sample_from_logits(
                next_logits,
                temperature=temperature,
                generator=generator,
            )
            input_ids, attention_mask = _append_batch_step(
                input_ids=input_ids,
                attention_mask=attention_mask,
                next_token_ids=next_tokens,
                states=states,
                eos_token_ids=eos_token_ids,
                max_new_tokens=max_new_tokens,
                start=start,
                is_first=True,
                kv_block_manager=kv_block_manager,
                request_ids=block_request_ids,
            )

            for state in states:
                state.prefill_seconds = prefill_seconds

            for _ in range(max_new_tokens - 1):
                if all(state.finished or state.generated_tokens >= max_new_tokens for state in states):
                    break

                decode_start = perf_counter()
                outputs = model(
                    input_ids=next_tokens,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = getattr(outputs, "past_key_values", None)
                if past_key_values is None:
                    raise RuntimeError("model did not return past_key_values during decode")

                sampled_tokens = _sample_from_logits(
                    outputs.logits[:, -1, :],
                    temperature=temperature,
                    generator=generator,
                )
                decode_elapsed = perf_counter() - decode_start
                for state in states:
                    if not state.finished and state.generated_tokens < max_new_tokens:
                        state.decode_seconds += decode_elapsed
                next_tokens = _force_finished_rows(
                    sampled_tokens,
                    states,
                    eos_token_id,
                    max_new_tokens,
                )
                input_ids, attention_mask = _append_batch_step(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    next_token_ids=next_tokens,
                    states=states,
                    eos_token_ids=eos_token_ids,
                    max_new_tokens=max_new_tokens,
                    start=None,
                    is_first=False,
                    kv_block_manager=kv_block_manager,
                    request_ids=block_request_ids,
                )

        elapsed = perf_counter() - start
        return [_finalize_batch_state(state, tokenizer, elapsed=elapsed) for state in states]
    finally:
        _release_blocks(kv_block_manager, allocated_request_ids)


def generate_continuous_batch(
    model,
    tokenizer,
    requests: list[ContinuousBatchRequest],
    *,
    max_batch_size: int | None = None,
    temperature: float = 0.0,
    seed: int | None = None,
    kv_block_manager: KVBlockManager | None = None,
) -> ContinuousBatchRunResult:
    """Generate requests with teaching-scale continuous batching.

    New requests are admitted by ``arrival_step`` while existing requests keep
    decoding. Each scheduler step rebuilds a padded full-token batch from the
    current running set, then removes completed rows before the next step.
    """

    if not requests:
        raise ValueError("requests must contain at least one entry")

    scheduler = ContinuousBatchScheduler(requests=requests, max_batch_size=max_batch_size)
    request_by_id = {request.request_id: request for request in requests}
    states: dict[str, GenerationRequestState] = {}
    admitted_at: dict[str, float] = {}
    finished_at: dict[str, float] = {}
    allocated_request_ids: list[str] = []
    scheduler_steps: list[SchedulerStepStats] = []
    eos_token_ids = _eos_token_ids(tokenizer)

    device = _model_device(model)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    start = perf_counter()
    model.eval()
    step = 0
    try:
        with torch.inference_mode():
            while scheduler.has_work():
                if not scheduler.running and scheduler.next_arrival_step() is not None:
                    step = max(step, scheduler.next_arrival_step())

                admitted = scheduler.admit(step)
                for scheduled in admitted:
                    states[scheduled.request.request_id] = _state_from_prompt(
                        tokenizer,
                        scheduled.request.prompt,
                        device,
                    )
                    _allocate_prompt_blocks(
                        kv_block_manager,
                        scheduled.request.request_id,
                        states[scheduled.request.request_id].prompt_tokens,
                    )
                    allocated_request_ids.append(scheduled.request.request_id)
                    admitted_at[scheduled.request.request_id] = perf_counter()

                running_ids = [state.request.request_id for state in scheduler.running]
                if not running_ids:
                    continue

                batch = _continuous_batch_tensors(states, running_ids, tokenizer, device)
                batch_start = perf_counter()
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                batch_elapsed = perf_counter() - batch_start
                next_logits = _select_last_token_logits(outputs.logits, batch["attention_mask"])
                next_tokens = _sample_from_logits(
                    next_logits,
                    temperature=temperature,
                    generator=generator,
                )

                completed_ids: set[str] = set()
                for index, request_id in enumerate(running_ids):
                    state = states[request_id]
                    request = request_by_id[request_id]
                    token_id = int(next_tokens[index, 0].item())
                    state.generated_token_ids.append(token_id)
                    _append_generated_block_token(kv_block_manager, request_id)
                    state.attention_mask = torch.cat(
                        [
                            state.attention_mask,
                            torch.ones(1, dtype=state.attention_mask.dtype, device=state.attention_mask.device),
                        ],
                        dim=-1,
                    )
                    if state.generated_tokens == 1:
                        state.ttft_seconds = perf_counter() - admitted_at[request_id]
                        state.prefill_seconds += batch_elapsed
                    else:
                        state.decode_seconds += batch_elapsed
                    if token_id in eos_token_ids or state.generated_tokens >= request.max_new_tokens:
                        state.finished = token_id in eos_token_ids
                        completed_ids.add(request_id)
                        finished_at[request_id] = perf_counter()

                completed = scheduler.finish(completed_ids, step)
                for completed_state in completed:
                    if kv_block_manager is not None:
                        kv_block_manager.release(completed_state.request.request_id)
                        allocated_request_ids.remove(completed_state.request.request_id)
                scheduler_steps.append(
                    scheduler.record_step(
                        step=step,
                        admitted=admitted,
                        running_request_ids=running_ids,
                        completed=completed,
                    )
                )
                step += 1
    finally:
        _release_blocks(kv_block_manager, allocated_request_ids)

    results: list[ContinuousBatchGenerationResult] = []
    finished_by_id = {state.request.request_id: state for state in scheduler.finished}
    for request in requests:
        state = states[request.request_id]
        scheduled = finished_by_id[request.request_id]
        elapsed = finished_at[request.request_id] - admitted_at[request.request_id]
        results.append(
            ContinuousBatchGenerationResult(
                request_id=request.request_id,
                result=_finalize_batch_state(state, tokenizer, elapsed=elapsed),
                arrival_step=request.arrival_step,
                admitted_step=scheduled.admitted_step if scheduled.admitted_step is not None else request.arrival_step,
                finished_step=scheduled.finished_step if scheduled.finished_step is not None else step,
            )
        )

    return ContinuousBatchRunResult(results=results, scheduler_steps=scheduler_steps)


def stream_generate_one(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    seed: int | None = None,
    kv_block_manager: KVBlockManager | None = None,
    prefix_cache: PrefixCache | None = None,
    request_id: str = "request-0",
):
    """Yield generated text one token at a time for one prompt.

    The first token is produced by a prefill forward over the whole prompt.
    Later tokens use ``past_key_values`` and pass only the last sampled token
    through the model, which is the smallest teaching-scale version of the
    prefill/decode split used by production serving systems.
    """

    import torch

    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if not prompt:
        raise ValueError("prompt must not be empty")

    device = _model_device(model)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = _move_batch_to_device(encoded, device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    prompt_token_ids = [int(token_id) for token_id in input_ids[0].tolist()]
    state = GenerationRequestState(
        prompt=prompt,
        prompt_token_ids=prompt_token_ids,
        attention_mask=attention_mask,
    )
    _allocate_prompt_blocks(kv_block_manager, request_id, state.prompt_tokens)
    prefix_lookup: PrefixCacheLookup | None = None

    try:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        eos_token_ids = _eos_token_ids(tokenizer)
        start = perf_counter()

        model.eval()
        with torch.inference_mode():
            prefix_lookup = _lookup_prefix_cache(prefix_cache, prompt_token_ids)
            prefix_tokens = prefix_lookup.matched_tokens if prefix_lookup is not None else 0
            prefill_input_ids = input_ids[:, prefix_tokens:]
            if prefill_input_ids.shape[-1] == 0:
                prefix_tokens = 0
                prefill_input_ids = input_ids
                prefix_lookup = None

            prefill_start = perf_counter()
            outputs = model(
                input_ids=prefill_input_ids,
                attention_mask=state.attention_mask,
                past_key_values=prefix_lookup.past_key_values if prefix_lookup is not None else None,
                use_cache=True,
            )
            state.prefill_seconds = perf_counter() - prefill_start
            state.past_key_values = getattr(outputs, "past_key_values", None)
            if state.past_key_values is None:
                raise RuntimeError("model did not return past_key_values; KV cache decode requires use_cache support")
            _store_prompt_prefix_cache(prefix_cache, prompt_token_ids, state.past_key_values)

            next_token = _sample_from_outputs(outputs, temperature=temperature, generator=generator)
            step = _record_step(
                tokenizer,
                state,
                next_token,
                eos_token_ids=eos_token_ids,
                start=start,
                max_new_tokens=max_new_tokens,
                kv_block_manager=kv_block_manager,
                request_id=request_id,
            )
            yield step
            if state.finished:
                return

            for _ in range(max_new_tokens - 1):
                decode_start = perf_counter()
                outputs = model(
                    input_ids=next_token.to(input_ids.device),
                    attention_mask=state.attention_mask,
                    past_key_values=state.past_key_values,
                    use_cache=True,
                )
                state.past_key_values = getattr(outputs, "past_key_values", None)
                if state.past_key_values is None:
                    raise RuntimeError("model did not return past_key_values during decode")

                next_token = _sample_from_outputs(outputs, temperature=temperature, generator=generator)
                state.decode_seconds += perf_counter() - decode_start
                step = _record_step(
                    tokenizer,
                    state,
                    next_token,
                    eos_token_ids=eos_token_ids,
                    start=start,
                    max_new_tokens=max_new_tokens,
                    kv_block_manager=kv_block_manager,
                    request_id=request_id,
                )
                yield step
                if state.finished:
                    return

                if state.generated_tokens >= max_new_tokens:
                    break
    finally:
        _release_prefix_cache(prefix_cache, prefix_lookup)
        _release_blocks(kv_block_manager, [request_id])


def _finalize_batch_state(
    state: GenerationRequestState,
    tokenizer,
    *,
    elapsed: float,
) -> GenerationResult:
    generated_text = tokenizer.decode(state.generated_token_ids, skip_special_tokens=True)
    return GenerationResult(
        prompt=state.prompt,
        text=generated_text,
        generated_token_ids=state.generated_token_ids,
        prompt_tokens=state.prompt_tokens,
        generated_tokens=state.generated_tokens,
        elapsed_seconds=elapsed,
        ttft_seconds=state.ttft_seconds,
        tpot_seconds=state.tpot_seconds,
        prefill_seconds=state.prefill_seconds,
        decode_seconds=state.decode_seconds,
        finished=state.finished,
    )


def _state_from_prompt(tokenizer, prompt: str, device) -> GenerationRequestState:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = _move_batch_to_device(encoded, device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    prompt_token_ids = [int(token_id) for token_id in input_ids[0].tolist()]
    valid_length = int(attention_mask[0].sum().item())
    return GenerationRequestState(
        prompt=prompt,
        prompt_token_ids=prompt_token_ids[:valid_length],
        attention_mask=attention_mask[0, :valid_length].to(dtype=torch.long).clone(),
    )


def _continuous_batch_tensors(
    states: dict[str, GenerationRequestState],
    running_ids: list[str],
    tokenizer,
    device,
) -> dict:
    sequences: list[list[int]] = []
    max_length = 0
    for request_id in running_ids:
        state = states[request_id]
        sequence = state.prompt_token_ids + state.generated_token_ids
        sequences.append(sequence)
        max_length = max(max_length, len(sequence))

    pad_token_id = _pad_token_id(tokenizer)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for sequence in sequences:
        pad_length = max_length - len(sequence)
        input_rows.append(sequence + [pad_token_id] * pad_length)
        mask_rows.append([1] * len(sequence) + [0] * pad_length)

    return {
        "input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(mask_rows, dtype=torch.long, device=device),
    }


def _pad_token_id(tokenizer) -> int:
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        return eos_token_id
    return 0


def _sample_from_outputs(outputs, *, temperature: float, generator):
    next_logits = outputs.logits[:, -1, :]
    return _sample_from_logits(next_logits, temperature=temperature, generator=generator)


def _sample_from_logits(logits, *, temperature: float, generator):
    return sample_next_token(logits, temperature=temperature, generator=generator)


def _force_finished_rows(
    sampled_token_ids,
    states: list[GenerationRequestState],
    eos_token_id: int,
    max_new_tokens: int,
):
    for idx, state in enumerate(states):
        if state.finished or state.generated_tokens >= max_new_tokens:
            sampled_token_ids[idx] = eos_token_id
    return sampled_token_ids


def _append_batch_step(
    *,
    input_ids,
    attention_mask,
    next_token_ids,
    states: list[GenerationRequestState],
    eos_token_ids: set[int],
    max_new_tokens: int,
    start: float | None,
    is_first: bool,
    kv_block_manager: KVBlockManager | None = None,
    request_ids: list[str] | None = None,
):
    for index, state in enumerate(states):
        token_id = int(next_token_ids[index, 0].item())
        if state.finished or state.generated_tokens >= max_new_tokens:
            continue
        state.generated_token_ids.append(token_id)
        if request_ids is not None:
            _append_generated_block_token(kv_block_manager, request_ids[index])
        state.attention_mask = torch.cat(
            [
                state.attention_mask,
                torch.ones(1, dtype=state.attention_mask.dtype, device=state.attention_mask.device),
            ],
            dim=-1,
        )
        if start is not None and is_first and state.generated_tokens == 1:
            state.ttft_seconds = perf_counter() - start
        if token_id in eos_token_ids:
            state.finished = True

    if is_first and start is None:
        raise ValueError("start is required for first step")

    return (
        torch.cat([input_ids, next_token_ids.to(input_ids.device)], dim=-1),
        torch.cat([attention_mask, torch.ones_like(next_token_ids, dtype=attention_mask.dtype)], dim=-1),
    )


def _select_last_token_logits(logits, attention_mask):
    indices = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
    batch = torch.arange(logits.size(0), device=logits.device)
    return logits[batch, indices, :]


def _record_step(
    tokenizer,
    state: GenerationRequestState,
    next_token,
    *,
    eos_token_ids: set[int],
    start: float,
    max_new_tokens: int,
    kv_block_manager: KVBlockManager | None = None,
    request_id: str | None = None,
) -> GenerationStep:
    import torch

    next_id = int(next_token[0, 0].item())
    state.generated_token_ids.append(next_id)
    if request_id is not None:
        _append_generated_block_token(kv_block_manager, request_id)
    state.attention_mask = torch.cat(
        [state.attention_mask, torch.ones_like(next_token).to(state.attention_mask.device)],
        dim=-1,
    )
    if state.generated_tokens == 1:
        state.ttft_seconds = perf_counter() - start
    state.finished = next_id in eos_token_ids
    elapsed = perf_counter() - start

    return GenerationStep(
        prompt=state.prompt,
        token_id=next_id,
        text=tokenizer.decode([next_id], skip_special_tokens=True),
        generated_text=tokenizer.decode(state.generated_token_ids, skip_special_tokens=True),
        generated_token_ids=list(state.generated_token_ids),
        prompt_tokens=state.prompt_tokens,
        generated_tokens=state.generated_tokens,
        elapsed_seconds=elapsed,
        ttft_seconds=state.ttft_seconds,
        tpot_seconds=state.tpot_seconds,
        prefill_seconds=state.prefill_seconds,
        decode_seconds=state.decode_seconds,
        finished=state.finished,
    )


def _allocate_prompt_blocks(
    kv_block_manager: KVBlockManager | None,
    request_id: str,
    prompt_tokens: int,
) -> None:
    if kv_block_manager is not None:
        kv_block_manager.allocate(request_id, prompt_tokens)


def _append_generated_block_token(
    kv_block_manager: KVBlockManager | None,
    request_id: str,
) -> None:
    if kv_block_manager is not None:
        kv_block_manager.append_tokens(request_id, 1)


def _release_blocks(kv_block_manager: KVBlockManager | None, request_ids: list[str]) -> None:
    if kv_block_manager is None:
        return
    for request_id in reversed(request_ids):
        kv_block_manager.release(request_id)


def _lookup_prefix_cache(
    prefix_cache: PrefixCache | None,
    prompt_token_ids: list[int],
) -> PrefixCacheLookup | None:
    if prefix_cache is None:
        return None
    return prefix_cache.lookup(prompt_token_ids)


def _store_prompt_prefix_cache(
    prefix_cache: PrefixCache | None,
    prompt_token_ids: list[int],
    past_key_values,
) -> None:
    if prefix_cache is not None:
        prefix_cache.store_prompt(prompt_token_ids, past_key_values)


def _release_prefix_cache(
    prefix_cache: PrefixCache | None,
    lookup: PrefixCacheLookup | None,
) -> None:
    if prefix_cache is not None and lookup is not None:
        prefix_cache.release(lookup)

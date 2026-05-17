"""Single-request generation loop with Hugging Face KV cache reuse."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from nanollmserve.engine.request import GenerationRequestState
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


def stream_generate_one(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    seed: int | None = None,
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

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    eos_token_ids = _eos_token_ids(tokenizer)
    start = perf_counter()

    model.eval()
    with torch.inference_mode():
        prefill_start = perf_counter()
        outputs = model(input_ids=input_ids, attention_mask=state.attention_mask, use_cache=True)
        state.prefill_seconds = perf_counter() - prefill_start
        state.past_key_values = getattr(outputs, "past_key_values", None)
        if state.past_key_values is None:
            raise RuntimeError("model did not return past_key_values; KV cache decode requires use_cache support")

        next_token = _sample_from_outputs(outputs, temperature=temperature, generator=generator)
        yield _record_step(
            tokenizer,
            state,
            next_token,
            eos_token_ids=eos_token_ids,
            start=start,
            max_new_tokens=max_new_tokens,
        )
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
            yield _record_step(
                tokenizer,
                state,
                next_token,
                eos_token_ids=eos_token_ids,
                start=start,
                max_new_tokens=max_new_tokens,
                decode_start=decode_start,
            )
            if state.finished:
                break


def _sample_from_outputs(outputs, *, temperature: float, generator):
    next_logits = outputs.logits[:, -1, :]
    return sample_next_token(next_logits, temperature=temperature, generator=generator)


def _record_step(
    tokenizer,
    state: GenerationRequestState,
    next_token,
    *,
    eos_token_ids: set[int],
    start: float,
    max_new_tokens: int,
    decode_start: float | None = None,
) -> GenerationStep:
    import torch

    next_id = int(next_token[0, 0].item())
    if decode_start is not None:
        state.decode_seconds += perf_counter() - decode_start
    state.generated_token_ids.append(next_id)
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

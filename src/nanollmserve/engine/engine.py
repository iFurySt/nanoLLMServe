"""Naive single-request generation loop."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from nanollmserve.sampling.sampler import sample_next_token


@dataclass(frozen=True)
class GenerationResult:
    prompt: str
    text: str
    generated_token_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.generated_tokens / self.elapsed_seconds


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
    """Generate text for one prompt with a deliberately naive decode loop."""

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

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    eos_token_ids = _eos_token_ids(tokenizer)
    generated: list[int] = []
    start = perf_counter()

    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_logits = outputs.logits[:, -1, :]
            next_token = sample_next_token(next_logits, temperature=temperature, generator=generator)
            next_id = int(next_token[0, 0].item())
            generated.append(next_id)

            input_ids = torch.cat([input_ids, next_token.to(input_ids.device)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

            if next_id in eos_token_ids:
                break

    elapsed = perf_counter() - start
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return GenerationResult(
        prompt=prompt,
        text=text,
        generated_token_ids=generated,
        prompt_tokens=int(encoded["input_ids"].shape[-1]),
        generated_tokens=len(generated),
        elapsed_seconds=elapsed,
    )

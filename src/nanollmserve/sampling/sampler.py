"""Token sampling helpers for the naive single-request path."""

from __future__ import annotations

import math


def sample_next_token(logits, *, temperature: float = 0.0, generator=None):
    """Return one next-token tensor from a `[batch, vocab]` logits tensor.

    `temperature <= 0` means greedy decoding. Positive temperatures sample from
    the softmax distribution, matching the first serving concept this milestone
    needs without adding top-k/top-p policy surface yet.
    """

    import torch

    if logits.ndim != 2:
        raise ValueError(f"expected logits with shape [batch, vocab], got {tuple(logits.shape)}")
    if not math.isfinite(temperature):
        raise ValueError("temperature must be finite")

    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)

"""Request state and lifecycle contracts for single-request generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationRequestState:
    """Mutable state for one teaching-scale generation request."""

    prompt: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int] = field(default_factory=list)
    attention_mask: Any | None = None
    past_key_values: Any | None = None
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    ttft_seconds: float = 0.0
    finished: bool = False

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def tpot_seconds(self) -> float:
        decode_tokens = max(self.generated_tokens - 1, 0)
        if decode_tokens == 0:
            return 0.0
        return self.decode_seconds / decode_tokens

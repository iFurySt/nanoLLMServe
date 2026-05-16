"""Minimal benchmark for the v0.0 single-request path."""

from __future__ import annotations

import argparse
import json
from statistics import mean

from nanollmserve.engine.engine import generate_one
from nanollmserve.model.hf_runner import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naive single-request generation.")
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local model directory.")
    parser.add_argument("--prompt", default="Explain KV cache in one sentence.", help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    loaded = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )

    for _ in range(args.warmup):
        generate_one(
            loaded.model,
            loaded.tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

    results = [
        generate_one(
            loaded.model,
            loaded.tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        for _ in range(args.runs)
    ]
    payload = {
        "model": args.model,
        "device": loaded.device,
        "dtype": loaded.dtype,
        "runs": args.runs,
        "warmup": args.warmup,
        "prompt_tokens": results[-1].prompt_tokens,
        "generated_tokens": [item.generated_tokens for item in results],
        "mean_elapsed_seconds": mean(item.elapsed_seconds for item in results),
        "mean_tokens_per_second": mean(item.tokens_per_second for item in results),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

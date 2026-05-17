"""Command-line entry point for v0.0 generation."""

from __future__ import annotations

import argparse
import sys

from nanollmserve.engine.engine import generate_one
from nanollmserve.model.hf_runner import load_model_and_tokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate text for one prompt with a causal LM.")
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local model directory.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 or lower uses greedy decoding.")
    parser.add_argument("--seed", type=int, default=None, help="Optional sampling seed.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--dtype", default="auto", help="auto, float32, float16, or bfloat16.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load files that already exist on disk.",
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Print generation timing and token counts to stderr.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    loaded = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    result = generate_one(
        loaded.model,
        loaded.tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    print(result.text)
    if args.show_stats:
        print(
            "prompt_tokens={prompt_tokens} generated_tokens={generated_tokens} "
            "elapsed_seconds={elapsed_seconds:.3f} tokens_per_second={tokens_per_second:.2f} "
            "ttft_seconds={ttft_seconds:.3f} tpot_seconds={tpot_seconds:.3f} "
            "device={device} dtype={dtype}".format(
                prompt_tokens=result.prompt_tokens,
                generated_tokens=result.generated_tokens,
                elapsed_seconds=result.elapsed_seconds,
                tokens_per_second=result.tokens_per_second,
                ttft_seconds=result.ttft_seconds,
                tpot_seconds=result.tpot_seconds,
                device=loaded.device,
                dtype=loaded.dtype,
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

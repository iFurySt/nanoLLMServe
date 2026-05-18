"""Minimal benchmark for single-request generation."""

from __future__ import annotations

import argparse
import json
from statistics import mean
from time import perf_counter

from nanollmserve.engine.engine import generate_batch, generate_one
from nanollmserve.model.hf_runner import load_model_and_tokenizer
from nanollmserve.sampling.sampler import sample_next_token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark single-request generation.")
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local model directory.")
    parser.add_argument("--prompt", default="Explain KV cache in one sentence.", help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--skip-naive-baseline",
        action="store_true",
        help="Only benchmark the current KV-cache path; omit the v0.0-style baseline comparison.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")

    loaded = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )

    batch_prompts = [args.prompt] * args.batch_size

    for _ in range(args.warmup):
        generate_one(
            loaded.model,
            loaded.tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        if args.batch_size > 1:
            _ = generate_batch(
                loaded.model,
                loaded.tokenizer,
                batch_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        if not args.skip_naive_baseline:
            _generate_one_naive(
                loaded.model,
                loaded.tokenizer,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

    kv_results = [
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
        "batch_size": args.batch_size,
        "prompt_tokens": kv_results[-1].prompt_tokens,
        "kv_cache_decode": _summarize(kv_results),
    }

    if args.batch_size > 1:
        batch_results = [
            generate_batch(
                loaded.model,
                loaded.tokenizer,
                batch_prompts,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            for _ in range(args.runs)
        ]
        payload["static_batch"] = _summarize_batch(batch_results)

    if not args.skip_naive_baseline:
        naive_results = [
            _generate_one_naive(
                loaded.model,
                loaded.tokenizer,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            for _ in range(args.runs)
        ]
        naive_summary = _summarize(naive_results)
        kv_summary = payload["kv_cache_decode"]
        payload["v0_0_naive_baseline"] = naive_summary
        payload["comparison"] = {
            "elapsed_speedup": _ratio(
                naive_summary["mean_elapsed_seconds"],
                kv_summary["mean_elapsed_seconds"],
            ),
            "tpot_speedup": _ratio(
                naive_summary["mean_tpot_seconds"],
                kv_summary["mean_tpot_seconds"],
            ),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _summarize(results) -> dict:
    return {
        "generated_tokens": [item.generated_tokens for item in results],
        "mean_elapsed_seconds": mean(item.elapsed_seconds for item in results),
        "mean_tokens_per_second": mean(item.tokens_per_second for item in results),
        "mean_ttft_seconds": mean(item.ttft_seconds for item in results),
        "mean_tpot_seconds": mean(item.tpot_seconds for item in results),
        "mean_prefill_seconds": mean(item.prefill_seconds for item in results),
        "mean_decode_seconds": mean(item.decode_seconds for item in results),
    }


def _summarize_batch(batch_runs) -> dict:
    if not batch_runs:
        return {}

    batch_elapsed_seconds = [run[0].elapsed_seconds for run in batch_runs]
    all_generated = [result.generated_tokens for run in batch_runs for result in run]
    all_prompt_tokens = [result.prompt_tokens for run in batch_runs for result in run]
    all_ttft = [result.ttft_seconds for run in batch_runs for result in run if result.generated_tokens > 0]
    all_tpot = [result.tpot_seconds for run in batch_runs for result in run if result.generated_tokens > 1]
    all_prefill = [result.prefill_seconds for run in batch_runs for result in run]
    all_decode = [result.decode_seconds for run in batch_runs for result in run]

    return {
        "batch_size": len(batch_runs[0]),
        "generated_tokens": [result.generated_tokens for run in batch_runs for result in run],
        "mean_batch_elapsed_seconds": mean(batch_elapsed_seconds),
        "mean_batch_tokens_per_second": mean(
            [
                sum(result.generated_tokens for result in run) / run[0].elapsed_seconds
                for run in batch_runs
            ]
        ),
        "mean_prompt_tokens": mean(all_prompt_tokens),
        "mean_generated_tokens": mean(all_generated),
        "mean_ttft_seconds": mean(all_ttft),
        "mean_tpot_seconds": mean(all_tpot) if all_tpot else 0.0,
        "mean_prefill_seconds": mean(all_prefill),
        "mean_decode_seconds": mean(all_decode),
    }


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _generate_one_naive(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
):
    """Run the v0.0-style full-sequence decode loop for comparison."""

    import torch

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)

    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        eos_token_ids = set()
    elif isinstance(eos, int):
        eos_token_ids = {eos}
    else:
        eos_token_ids = set(eos)

    generated: list[int] = []
    ttft_seconds = 0.0
    start = perf_counter()
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            next_token = sample_next_token(
                outputs.logits[:, -1, :],
                temperature=temperature,
            )
            if not generated:
                ttft_seconds = perf_counter() - start
            next_id = int(next_token[0, 0].item())
            generated.append(next_id)
            input_ids = torch.cat([input_ids, next_token.to(input_ids.device)], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            if next_id in eos_token_ids:
                break

    elapsed = perf_counter() - start
    generated_tokens = len(generated)
    tpot_seconds = (elapsed - ttft_seconds) / max(generated_tokens - 1, 1) if generated_tokens > 1 else 0.0
    return _BenchmarkResult(
        prompt_tokens=int(encoded["input_ids"].shape[-1]),
        generated_tokens=generated_tokens,
        elapsed_seconds=elapsed,
        ttft_seconds=ttft_seconds,
        tpot_seconds=tpot_seconds,
        prefill_seconds=0.0,
        decode_seconds=max(elapsed - ttft_seconds, 0.0),
    )


class _BenchmarkResult:
    def __init__(
        self,
        *,
        prompt_tokens: int,
        generated_tokens: int,
        elapsed_seconds: float,
        ttft_seconds: float,
        tpot_seconds: float,
        prefill_seconds: float,
        decode_seconds: float,
    ):
        self.prompt_tokens = prompt_tokens
        self.generated_tokens = generated_tokens
        self.elapsed_seconds = elapsed_seconds
        self.ttft_seconds = ttft_seconds
        self.tpot_seconds = tpot_seconds
        self.prefill_seconds = prefill_seconds
        self.decode_seconds = decode_seconds

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.generated_tokens / self.elapsed_seconds


if __name__ == "__main__":
    raise SystemExit(main())

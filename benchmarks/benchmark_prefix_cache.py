"""Benchmark repeated-prefix generation with and without PrefixCache."""

from __future__ import annotations

import argparse
import json
from statistics import mean

from nanollmserve.cache.prefix_cache import PrefixCache
from nanollmserve.engine.engine import generate_one
from nanollmserve.model.hf_runner import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark prefix-cache TTFT for repeated prompt prefixes.")
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local model directory.")
    parser.add_argument(
        "--prefix",
        default="You are a concise assistant. Answer in one sentence. Context: KV cache reuse matters because",
    )
    parser.add_argument(
        "--suffixes",
        default="system prompts repeat,RAG documents repeat,agent loops repeat",
        help="Comma-separated suffixes appended to the shared prefix.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prefix-block-size", type=int, default=16)
    parser.add_argument("--prefix-cache-entries", type=int, default=64)
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
    suffixes = _parse_suffixes(args.suffixes)
    prompts = [f"{args.prefix} {suffix}" for suffix in suffixes]

    loaded = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )

    for _ in range(args.warmup):
        _run_without_cache(loaded.model, loaded.tokenizer, prompts, args.max_new_tokens, args.temperature)
        _run_with_cache(
            loaded.model,
            loaded.tokenizer,
            prompts,
            args.max_new_tokens,
            args.temperature,
            block_size=args.prefix_block_size,
            max_entries=args.prefix_cache_entries,
        )

    no_cache_runs = [
        _run_without_cache(loaded.model, loaded.tokenizer, prompts, args.max_new_tokens, args.temperature)
        for _ in range(args.runs)
    ]
    cache_runs = [
        _run_with_cache(
            loaded.model,
            loaded.tokenizer,
            prompts,
            args.max_new_tokens,
            args.temperature,
            block_size=args.prefix_block_size,
            max_entries=args.prefix_cache_entries,
        )
        for _ in range(args.runs)
    ]
    no_cache_summary = _summarize(no_cache_runs)
    cache_summary = _summarize(cache_runs)
    payload = {
        "model": args.model,
        "device": loaded.device,
        "dtype": loaded.dtype,
        "runs": args.runs,
        "warmup": args.warmup,
        "prompt_count": len(prompts),
        "prefix_block_size": args.prefix_block_size,
        "no_prefix_cache": no_cache_summary,
        "prefix_cache": cache_summary,
        "comparison": {
            "ttft_speedup": _ratio(
                no_cache_summary["mean_ttft_seconds"],
                cache_summary["mean_ttft_seconds"],
            ),
            "prefill_speedup": _ratio(
                no_cache_summary["mean_prefill_seconds"],
                cache_summary["mean_prefill_seconds"],
            ),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_without_cache(model, tokenizer, prompts: list[str], max_new_tokens: int, temperature: float) -> dict:
    results = [
        generate_one(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        for prompt in prompts
    ]
    return {
        "results": results,
        "cache_stats": None,
    }


def _run_with_cache(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    *,
    block_size: int,
    max_entries: int,
) -> dict:
    prefix_cache = PrefixCache(max_entries=max_entries, block_size=block_size)
    results = [
        generate_one(
            model,
            tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            prefix_cache=prefix_cache,
        )
        for prompt in prompts
    ]
    return {
        "results": results,
        "cache_stats": prefix_cache.stats(),
    }


def _summarize(runs: list[dict]) -> dict:
    results = [result for run in runs for result in run["results"]]
    summary = {
        "mean_ttft_seconds": mean(result.ttft_seconds for result in results),
        "mean_prefill_seconds": mean(result.prefill_seconds for result in results),
        "mean_elapsed_seconds": mean(result.elapsed_seconds for result in results),
        "mean_tokens_per_second": mean(result.tokens_per_second for result in results),
    }
    stats = [run["cache_stats"] for run in runs if run["cache_stats"] is not None]
    if stats:
        summary["cache_hits"] = [item.hits for item in stats]
        summary["cache_misses"] = [item.misses for item in stats]
        summary["cache_evictions"] = [item.evictions for item in stats]
        summary["cache_entries"] = [item.entries for item in stats]
    return summary


def _parse_suffixes(raw: str) -> list[str]:
    suffixes = [item.strip() for item in raw.split(",") if item.strip()]
    if len(suffixes) < 2:
        raise ValueError("--suffixes must contain at least two comma-separated entries")
    return suffixes


def _ratio(baseline: float, candidate: float) -> float:
    if candidate <= 0:
        return 0.0
    return baseline / candidate


if __name__ == "__main__":
    raise SystemExit(main())

"""Benchmark mixed long/short requests with chunked prefill."""

from __future__ import annotations

import argparse
import json
from time import perf_counter

from nanollmserve.engine.engine import generate_chunked_prefill_batch, generate_one
from nanollmserve.engine.scheduler import ContinuousBatchRequest
from nanollmserve.model.hf_runner import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark chunked prefill on mixed long/short prompts.")
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local model directory.")
    parser.add_argument(
        "--long-prompt",
        default=("Explain KV cache reuse in detail. " * 32).strip(),
    )
    parser.add_argument(
        "--short-prompt",
        default="Define TTFT in one sentence.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-prefill-tokens-per-step", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    loaded = load_model_and_tokenizer(
        args.model,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    requests = [
        ContinuousBatchRequest("long-0", args.long_prompt, max_new_tokens=args.max_new_tokens, arrival_step=0),
        ContinuousBatchRequest("short-1", args.short_prompt, max_new_tokens=args.max_new_tokens, arrival_step=1),
    ]
    monolithic = _run_monolithic_baseline(
        loaded.model,
        loaded.tokenizer,
        requests,
        temperature=args.temperature,
    )
    chunked = generate_chunked_prefill_batch(
        loaded.model,
        loaded.tokenizer,
        requests,
        max_prefill_tokens_per_step=args.max_prefill_tokens_per_step,
        temperature=args.temperature,
    )
    monolithic_short = _find_result(monolithic["results"], "short-1")
    chunked_short = _find_result(chunked.results, "short-1")
    payload = {
        "model": args.model,
        "device": loaded.device,
        "dtype": loaded.dtype,
        "max_prefill_tokens_per_step": args.max_prefill_tokens_per_step,
        "monolithic_prefill_baseline": monolithic,
        "chunked_prefill": _summarize_chunked(chunked),
        "comparison": {
            "short_time_to_first_token_speedup": _ratio(
                monolithic_short["time_to_first_token_seconds"],
                chunked_short.time_to_first_token_seconds,
            ),
            "short_first_token_step": chunked_short.first_token_step,
            "long_first_token_step": _find_result(chunked.results, "long-0").first_token_step,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _run_monolithic_baseline(model, tokenizer, requests: list[ContinuousBatchRequest], *, temperature: float) -> dict:
    started_at = perf_counter()
    results = []
    for request in sorted(requests, key=lambda item: (item.arrival_step, item.request_id)):
        call_started_at = perf_counter()
        result = generate_one(
            model,
            tokenizer,
            request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=temperature,
            request_id=request.request_id,
        )
        results.append(
            {
                "request_id": request.request_id,
                "prompt_tokens": result.prompt_tokens,
                "generated_tokens": result.generated_tokens,
                "time_to_first_token_seconds": (call_started_at - started_at) + result.ttft_seconds,
                "elapsed_seconds": perf_counter() - call_started_at,
            }
        )
    return {
        "policy": "arrival_order_full_prefill",
        "results": results,
    }


def _summarize_chunked(run) -> dict:
    return {
        "policy": "decode_first_shortest_prefill_first",
        "prefill_tokens_per_step": run.prefill_tokens_per_step,
        "results": [
            {
                "request_id": result.request_id,
                "prompt_tokens": result.result.prompt_tokens,
                "generated_tokens": result.result.generated_tokens,
                "first_token_step": result.first_token_step,
                "finished_step": result.finished_step,
                "time_to_first_token_seconds": result.time_to_first_token_seconds,
                "elapsed_seconds": result.result.elapsed_seconds,
            }
            for result in run.results
        ],
    }


def _find_result(results, request_id: str):
    for result in results:
        if isinstance(result, dict) and result["request_id"] == request_id:
            return result
        if getattr(result, "request_id", None) == request_id:
            return result
    raise KeyError(request_id)


def _ratio(baseline: float, candidate: float) -> float:
    if candidate <= 0:
        return 0.0
    return baseline / candidate


if __name__ == "__main__":
    raise SystemExit(main())

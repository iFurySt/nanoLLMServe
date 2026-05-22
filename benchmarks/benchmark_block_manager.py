"""Show block KV cache fragmentation with a small synthetic workload."""

from __future__ import annotations

import argparse
import json

from nanollmserve.cache.block_manager import KVBlockManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark block KV cache allocation metadata.")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--total-blocks", type=int, default=64)
    parser.add_argument(
        "--request-tokens",
        default="9,17,33,5,41,12",
        help="Comma-separated final token counts for synthetic requests.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token_counts = _parse_token_counts(args.request_tokens)
    payload = summarize_fragmentation(
        token_counts,
        block_size=args.block_size,
        total_blocks=args.total_blocks,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def summarize_fragmentation(token_counts: list[int], *, block_size: int, total_blocks: int) -> dict:
    if not token_counts:
        raise ValueError("token_counts must contain at least one entry")
    if any(token_count < 1 for token_count in token_counts):
        raise ValueError("token counts must be positive")

    manager = KVBlockManager(total_blocks=total_blocks, block_size=block_size)
    for index, token_count in enumerate(token_counts):
        manager.allocate(f"req-{index}", token_count)

    block_usage = manager.usage()
    contiguous_reserved_tokens = len(token_counts) * max(token_counts)
    contiguous_fragmentation_tokens = contiguous_reserved_tokens - sum(token_counts)
    return {
        "request_tokens": token_counts,
        "block_size": block_size,
        "block_usage": {
            "used_blocks": block_usage.used_blocks,
            "free_blocks": block_usage.free_blocks,
            "allocated_tokens": block_usage.allocated_tokens,
            "reserved_tokens": block_usage.reserved_tokens,
            "internal_fragmentation_tokens": block_usage.internal_fragmentation_tokens,
            "block_utilization": block_usage.block_utilization,
        },
        "contiguous_fixed_slot_baseline": {
            "reserved_tokens": contiguous_reserved_tokens,
            "internal_fragmentation_tokens": contiguous_fragmentation_tokens,
            "utilization": sum(token_counts) / contiguous_reserved_tokens,
        },
        "fragmentation_tokens_saved_vs_contiguous": (
            contiguous_fragmentation_tokens - block_usage.internal_fragmentation_tokens
        ),
    }


def _parse_token_counts(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_block_manager import _parse_token_counts, summarize_fragmentation


def test_summarize_fragmentation_compares_block_manager_with_contiguous_slots():
    summary = summarize_fragmentation([5, 9, 16], block_size=8, total_blocks=8)

    assert summary["block_usage"]["used_blocks"] == 5
    assert summary["block_usage"]["allocated_tokens"] == 30
    assert summary["block_usage"]["reserved_tokens"] == 40
    assert summary["block_usage"]["internal_fragmentation_tokens"] == 10
    assert summary["contiguous_fixed_slot_baseline"]["reserved_tokens"] == 48
    assert summary["contiguous_fixed_slot_baseline"]["internal_fragmentation_tokens"] == 18
    assert summary["fragmentation_tokens_saved_vs_contiguous"] == 8


def test_summarize_fragmentation_rejects_empty_workload():
    with pytest.raises(ValueError, match="token_counts"):
        summarize_fragmentation([], block_size=8, total_blocks=8)


def test_parse_token_counts():
    assert _parse_token_counts("1, 2,3") == [1, 2, 3]

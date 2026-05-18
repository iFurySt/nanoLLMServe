from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch")

from benchmarks.benchmark_generate import _ratio, _summarize, _summarize_batch


def test_benchmark_summarize_reports_ttft_tpot_and_decode_metrics():
    results = [
        SimpleNamespace(
            generated_tokens=2,
            elapsed_seconds=1.0,
            tokens_per_second=2.0,
            ttft_seconds=0.2,
            tpot_seconds=0.8,
            prefill_seconds=0.15,
            decode_seconds=0.8,
        ),
        SimpleNamespace(
            generated_tokens=4,
            elapsed_seconds=2.0,
            tokens_per_second=2.0,
            ttft_seconds=0.4,
            tpot_seconds=0.5,
            prefill_seconds=0.25,
            decode_seconds=1.5,
        ),
    ]

    summary = _summarize(results)

    assert summary["generated_tokens"] == [2, 4]
    assert summary["mean_elapsed_seconds"] == 1.5
    assert summary["mean_tokens_per_second"] == 2.0
    assert summary["mean_ttft_seconds"] == pytest.approx(0.3)
    assert summary["mean_tpot_seconds"] == pytest.approx(0.65)
    assert summary["mean_prefill_seconds"] == pytest.approx(0.2)
    assert summary["mean_decode_seconds"] == pytest.approx(1.15)


def test_ratio_handles_zero_denominator():
    assert _ratio(10.0, 0.0) == 0.0
    assert _ratio(10.0, 2.0) == 5.0


def test_summarize_batch_aggregates_per_request_rows():
    run1 = [
        SimpleNamespace(generated_tokens=2, elapsed_seconds=1.0, ttft_seconds=0.2, tpot_seconds=0.4, prompt_tokens=3, prefill_seconds=0.1, decode_seconds=0.7),
        SimpleNamespace(generated_tokens=3, elapsed_seconds=1.0, ttft_seconds=0.1, tpot_seconds=0.3, prompt_tokens=3, prefill_seconds=0.1, decode_seconds=0.6),
    ]
    run2 = [
        SimpleNamespace(generated_tokens=1, elapsed_seconds=0.8, ttft_seconds=0.25, tpot_seconds=0.0, prompt_tokens=3, prefill_seconds=0.1, decode_seconds=0.7),
        SimpleNamespace(generated_tokens=2, elapsed_seconds=0.8, ttft_seconds=0.12, tpot_seconds=0.32, prompt_tokens=3, prefill_seconds=0.1, decode_seconds=0.68),
    ]

    summary = _summarize_batch([run1, run2])

    assert summary["batch_size"] == 2
    assert summary["generated_tokens"] == [2, 3, 1, 2]
    assert summary["mean_batch_elapsed_seconds"] == 0.9
    assert summary["mean_batch_tokens_per_second"] == ((2 + 3) / 1.0 + (1 + 2) / 0.8) / 2

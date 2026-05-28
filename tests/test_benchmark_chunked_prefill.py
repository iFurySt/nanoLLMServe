from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_chunked_prefill import _find_result, _ratio, _summarize_chunked


def test_find_result_supports_dicts_and_objects():
    assert _find_result([{"request_id": "a", "value": 1}], "a")["value"] == 1
    assert _find_result([SimpleNamespace(request_id="b", value=2)], "b").value == 2

    with pytest.raises(KeyError):
        _find_result([], "missing")


def test_summarize_chunked_reports_steps_and_request_latency():
    run = SimpleNamespace(
        prefill_tokens_per_step=[2, 2, 0],
        results=[
            SimpleNamespace(
                request_id="short-1",
                result=SimpleNamespace(prompt_tokens=2, generated_tokens=2, elapsed_seconds=0.3),
                first_token_step=1,
                finished_step=2,
                time_to_first_token_seconds=0.2,
            )
        ],
    )

    summary = _summarize_chunked(run)

    assert summary["policy"] == "decode_first_shortest_prefill_first"
    assert summary["prefill_tokens_per_step"] == [2, 2, 0]
    assert summary["results"][0]["first_token_step"] == 1
    assert summary["results"][0]["time_to_first_token_seconds"] == 0.2


def test_ratio_handles_zero_candidate():
    assert _ratio(4.0, 2.0) == 2.0
    assert _ratio(4.0, 0.0) == 0.0

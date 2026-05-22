from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_prefix_cache import _parse_suffixes, _ratio


def test_parse_suffixes_requires_multiple_entries():
    assert _parse_suffixes("alpha, beta,gamma") == ["alpha", "beta", "gamma"]

    with pytest.raises(ValueError, match="suffixes"):
        _parse_suffixes("alpha")


def test_ratio_handles_zero_candidate():
    assert _ratio(2.0, 1.0) == 2.0
    assert _ratio(2.0, 0.0) == 0.0

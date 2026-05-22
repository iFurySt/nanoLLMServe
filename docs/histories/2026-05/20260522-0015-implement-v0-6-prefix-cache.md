## [2026-05-22 00:15] | Task: Implement v0.6 prefix cache

### Execution Context

- Agent ID: `Codex`
- Base Model: `gpt-5`
- Runtime: `Python 3`

### User Query

> 现在我们继续推进v0.6

### Changes Overview

- Area:
  - Prompt-prefix KV cache reuse.
- Key actions:
  - Added `PrefixCache`, prefix token hashing, cache entries, lookup results,
    stats, ref counting, and LRU eviction.
  - Wired optional `PrefixCache` lookup into `generate_one` and
    `stream_generate_one` so shared-prefix requests prefill only uncached suffix
    tokens.
  - Added a repeated-prefix benchmark comparing no-cache and prefix-cache TTFT
    and prefill timing.
  - Added tests for key generation, longest-prefix lookup, ref counts, LRU
    eviction, benchmark helpers, and engine-level suffix-only prefill.
  - Bumped package/runtime version metadata to `0.6.0`.
  - Updated architecture, roadmap, release notes, README, and quality notes.

### Design Intent

v0.6 introduces the core serving idea behind prefix caching without hiding it
behind a distributed cache or custom kernel. The implementation stores
block-aligned prompt prefixes and sliced Hugging Face `past_key_values`, then
reuses the longest strict prefix for later single-request generation. Exact
full-prompt reuse and batch-aware prefix reuse are left for later milestones
because they require additional logits/state contracts and scheduler work.

### Files Modified

- `benchmarks/benchmark_prefix_cache.py`
- `docs/ARCHITECTURE.md`
- `docs/QUALITY_SCORE.md`
- `docs/exec-plans/active/milestones/v0.6-prefix-cache.md`
- `docs/exec-plans/active/nano-llm-serve-roadmap.md`
- `docs/releases/feature-release-notes.md`
- `pyproject.toml`
- `README.md`
- `README.zh-CN.md`
- `src/nanollmserve/__init__.py`
- `src/nanollmserve/api/openai_server.py`
- `src/nanollmserve/cache/__init__.py`
- `src/nanollmserve/cache/prefix_cache.py`
- `src/nanollmserve/engine/engine.py`
- `tests/test_benchmark_prefix_cache.py`
- `tests/test_engine.py`
- `tests/test_prefix_cache.py`

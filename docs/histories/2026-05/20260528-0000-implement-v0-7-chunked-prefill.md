## [2026-05-28 00:00] | Task: Implement v0.7 chunked prefill

### Execution Context

- Agent ID: `Codex`
- Base Model: `gpt-5`
- Runtime: `Python 3`

### User Query

> 我们继续推进v0.7

### Changes Overview

- Area:
  - Chunked prefill scheduling for mixed long/short prompt workloads.
- Key actions:
  - Added `generate_chunked_prefill_batch` using `ContinuousBatchRequest` rows,
    partial prefill state, and a `max_prefill_tokens_per_step` budget.
  - Added chunked prefill result and per-step stats dataclasses.
  - Implemented a decode-first, shortest-prefill-first teaching policy so ready
    decode work runs before more prefill and shorter remaining prompts can run
    before long prompts consume another prefill chunk.
  - Added `benchmark_chunked_prefill.py` comparing an arrival-order monolithic
    prefill baseline with chunked prefill short-request first-token behavior.
  - Added tests for chunked prefill scheduling, invalid budgets, and benchmark
    summarization helpers.
  - Bumped package/runtime version metadata to `0.7.0`.
  - Updated architecture, roadmap, release notes, README, and quality notes.

### Design Intent

Chunked prefill demonstrates why long prompts should not monopolize the engine
loop. The implementation keeps one readable Python scheduler loop and advances
real Hugging Face `past_key_values` across prompt chunks before decode. It does
not try to implement a production policy matrix, fused kernels, or distributed
fairness. The policy is intentionally explicit: decode-ready requests first,
then spend the bounded prefill budget on the shortest remaining prompt chunks.

### Files Modified

- `benchmarks/benchmark_chunked_prefill.py`
- `docs/ARCHITECTURE.md`
- `docs/QUALITY_SCORE.md`
- `docs/exec-plans/active/milestones/v0.7-chunked-prefill.md`
- `docs/exec-plans/active/nano-llm-serve-roadmap.md`
- `docs/releases/feature-release-notes.md`
- `pyproject.toml`
- `README.md`
- `README.zh-CN.md`
- `src/nanollmserve/__init__.py`
- `src/nanollmserve/api/openai_server.py`
- `src/nanollmserve/engine/engine.py`
- `tests/test_benchmark_chunked_prefill.py`
- `tests/test_engine.py`

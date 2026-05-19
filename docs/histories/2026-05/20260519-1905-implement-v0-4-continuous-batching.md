## [2026-05-19 19:05] | Task: Implement v0.4 continuous batching

### Execution Context

- Agent ID: `Codex`
- Base Model: `gpt-5`
- Runtime: `Python 3`

### User Query

> 现在我们来实现v0.4

### Changes Overview

- Area:
  - Scheduler-level continuous batching and benchmark visibility.
- Key actions:
  - Added `ContinuousBatchScheduler` with waiting, running, and finished request lifecycle state.
  - Added `ContinuousBatchRequest`, `SchedulerStepStats`, and `generate_continuous_batch`.
  - Implemented per-step active batch rebuilds, mid-run request admission, completed-row removal, and active batch size metrics.
  - Extended the benchmark to emit `continuous_batch` metrics when `--batch-size > 1`.
  - Added tests for continuous admission, max batch size backpressure, finished-row removal, and benchmark summary metrics.
  - Measured continuous-batch TTFT from per-request admission time.
  - Bumped package/runtime version metadata to `0.4.0`.
  - Updated architecture, roadmap, release notes, README, and quality notes.

### Design Intent

v0.4 introduces the scheduler concept before adding paged/block KV cache. The implementation rebuilds a padded full-token active batch each scheduler step, which keeps the control flow readable while demonstrating the core continuous batching lifecycle: requests can enter while others decode, completed rows leave the running set, and the active batch size changes over time.

### Files Modified

- `benchmarks/benchmark_generate.py`
- `pyproject.toml`
- `src/nanollmserve/__init__.py`
- `src/nanollmserve/api/openai_server.py`
- `src/nanollmserve/engine/__init__.py`
- `src/nanollmserve/engine/engine.py`
- `src/nanollmserve/engine/scheduler.py`
- `tests/test_benchmark_generate.py`
- `tests/test_engine.py`
- `docs/ARCHITECTURE.md`
- `docs/QUALITY_SCORE.md`
- `docs/exec-plans/active/milestones/v0.4-continuous-batching.md`
- `docs/exec-plans/active/nano-llm-serve-roadmap.md`
- `docs/releases/feature-release-notes.md`
- `README.md`
- `README.zh-CN.md`

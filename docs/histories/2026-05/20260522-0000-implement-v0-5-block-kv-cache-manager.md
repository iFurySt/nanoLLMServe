## [2026-05-22 00:00] | Task: Implement v0.5 block KV cache manager

### Execution Context

- Agent ID: `Codex`
- Base Model: `gpt-5`
- Runtime: `Python 3`

### User Query

> 我们继续推进v0.5。做完提交推送（tag和branch都要有，main上也要有）

### Changes Overview

- Area:
  - KV cache block allocation metadata and fragmentation visibility.
- Key actions:
  - Added `KVBlockManager`, `KVBlock`, `RequestBlockTable`, `KVBlockUsage`, and allocation errors.
  - Implemented FIFO free block pool, request-to-block tables, append-token growth, release, and usage metrics.
  - Wired optional block allocation hooks into `generate_one`, `generate_batch`, and `generate_continuous_batch`.
  - Added a synthetic `benchmark_block_manager.py` to compare block allocation fragmentation with a contiguous fixed-slot baseline.
  - Added tests for block allocation, release, over-allocation, usage metrics, generation lifecycle hooks, and fragmentation benchmark output.
  - Bumped package/runtime version metadata to `0.5.0`.
  - Updated architecture, roadmap, release notes, README, and quality notes.

### Design Intent

v0.5 introduces the allocator side of PagedAttention at teaching scale. The implementation tracks fixed-size token blocks and request ownership tables while leaving tensor-level paging, custom kernels, eviction, and prefix reuse to later milestones. This keeps block lifecycle and fragmentation behavior explicit before optimizing execution.

### Files Modified

- `benchmarks/benchmark_block_manager.py`
- `docs/ARCHITECTURE.md`
- `docs/QUALITY_SCORE.md`
- `docs/exec-plans/active/milestones/v0.5-block-kv-cache-manager.md`
- `docs/exec-plans/active/nano-llm-serve-roadmap.md`
- `docs/releases/feature-release-notes.md`
- `pyproject.toml`
- `README.md`
- `README.zh-CN.md`
- `src/nanollmserve/__init__.py`
- `src/nanollmserve/api/openai_server.py`
- `src/nanollmserve/cache/__init__.py`
- `src/nanollmserve/cache/block_manager.py`
- `src/nanollmserve/engine/engine.py`
- `tests/test_benchmark_block_manager.py`
- `tests/test_block_manager.py`
- `tests/test_engine.py`

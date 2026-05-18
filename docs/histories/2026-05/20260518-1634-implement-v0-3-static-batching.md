## [2026-05-18 16:34] | Task: Implement v0.3 static batching

### Execution Context

- Agent ID: `Codex`
- Base Model: `gpt-5`
- Runtime: `Python 3`

### User Query

> 完成 v0.3 静态批处理里程碑：确认目录与接口后，继续推进并提交 milestone 级改动，更新文档并可追溯地打标签与分支。

### Changes Overview

- Area:
  - Engine generation path and batching support.
- Key actions:
  - Added static batch generation API `engine.generate_batch(...)` with padded tokenize+attention mask handling, one prefill over full batch, lock-step decode with `past_key_values`.
  - Exported `generate_batch` from `nanollmserve.engine`.
  - Added batch-focused tests in `tests/test_engine.py` and batch benchmark summary coverage in `tests/test_benchmark_generate.py`.
  - Extended benchmark script to accept `--batch-size`, run fixed-size batch benchmarks, and report `static_batch` metrics.
  - Updated architecture and roadmap docs to record static batch runtime shape and current limitations.
  - Updated release notes and quality score entries for v0.3.

### Design Intent

v0.3 is implemented as a teaching-scale static batching path: all prompts are known before decode starts and always advance together. This preserves correctness and observability with minimal extra complexity while matching the vLLM/SGLang milestone shape (fixed batch prefill + lock-step decode) before introducing queueing or continuous scheduling in later milestones.

### Files Modified

- `src/nanollmserve/engine/engine.py`
- `src/nanollmserve/engine/__init__.py`
- `tests/test_engine.py`
- `benchmarks/benchmark_generate.py`
- `tests/test_benchmark_generate.py`
- `docs/ARCHITECTURE.md`
- `docs/exec-plans/active/milestones/v0.3-static-batching.md`
- `docs/QUALITY_SCORE.md`
- `docs/releases/feature-release-notes.md`
- `README.md`
- `README.zh-CN.md`

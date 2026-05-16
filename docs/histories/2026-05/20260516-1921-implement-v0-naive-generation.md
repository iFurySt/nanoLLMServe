## [2026-05-16 19:21] | Task: implement v0.0 naive generation

### Execution Context

- Agent ID: `codex`
- Base Model: `GPT-5`
- Runtime: `local CLI plus gpu-a100-80g validation`

### User Query

> Look at the first milestone and implement it; uploading to the GPU machine
> for validation is allowed.

### Changes Overview

- Area: v0.0 inference path
- Key actions:
  - Added the `nanollmserve` Python package with Hugging Face model loading,
    a naive single-request generation loop, greedy sampling, temperature
    sampling, and a `nanollm-generate` CLI.
  - Added a minimal benchmark script for single-prompt generation throughput.
  - Added unit tests for CLI parsing, sampling, and decode-loop behavior.
  - Updated README quick starts, quality scoring, release notes, and the v0.0
    milestone validation log.

### Design Intent

Keep v0.0 intentionally simple and inspectable: one process, one prompt, one
model, full-sequence recomputation each step, and no HTTP or batching surface.
That gives later KV cache and batching milestones a concrete baseline to improve
without hiding the first inference loop behind framework helpers.

### Files Modified

- `pyproject.toml`
- `src/nanollmserve/`
- `benchmarks/benchmark_generate.py`
- `tests/`
- `Makefile`
- `README.md`
- `README.zh-CN.md`
- `docs/QUALITY_SCORE.md`
- `docs/exec-plans/active/milestones/v0.0-naive-single-request.md`
- `docs/releases/feature-release-notes.md`

## [2026-05-17 20:31] | Task: add KV cache decode

### Execution Context

- Agent ID: `Codex`
- Base Model: `GPT-5`
- Runtime: `Codex CLI`

### User Query

> Continue v0.2 by studying the local vLLM/SGLang references, implement KV cache
> decode, verify the tests are complete, then commit, push, tag, and create a
> release branch.

### Changes Overview

- Area: single-request generation runtime, CLI stats, benchmark, docs.
- Key actions:
  - Added `GenerationRequestState` to track prompt tokens, generated tokens,
    attention mask, `past_key_values`, TTFT, TPOT, and completion state.
  - Changed generation to run prefill once over the prompt, then decode later
    tokens with only the previous sampled token and cached KV state.
  - Extended CLI stats and benchmark output with TTFT/TPOT.
  - Added a benchmark-side v0.0-style naive full-sequence baseline and speedup
    comparison.
  - Updated roadmap, architecture, README, release notes, quality score, and
    milestone validation notes.

### Design Intent

This milestone introduces the smallest production-shaped KV cache concept while
keeping the implementation readable. vLLM's per-request computed-token state and
SGLang's prefill/decode split informed the design, but this repo deliberately
uses Hugging Face `past_key_values` directly instead of adding paged KV cache,
block management, prefix reuse, or batching before those concepts have their own
milestones.

### Files Modified

- `src/nanollmserve/engine/request.py`
- `src/nanollmserve/engine/engine.py`
- `src/nanollmserve/cli/generate.py`
- `benchmarks/benchmark_generate.py`
- `tests/test_engine.py`
- `tests/test_cli.py`
- `tests/test_request_state.py`
- `tests/test_benchmark_generate.py`
- `pyproject.toml`
- `src/nanollmserve/__init__.py`
- `src/nanollmserve/api/openai_server.py`
- `README.md`
- `README.zh-CN.md`
- `docs/ARCHITECTURE.md`
- `docs/QUALITY_SCORE.md`
- `docs/exec-plans/active/nano-llm-serve-roadmap.md`
- `docs/exec-plans/active/milestones/v0.2-kv-cache-decode.md`
- `docs/releases/feature-release-notes.md`

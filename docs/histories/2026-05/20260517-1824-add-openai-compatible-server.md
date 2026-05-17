## [2026-05-17 18:24] | Task: Add OpenAI-compatible server milestone

### Execution Context

- Agent ID: `codex`
- Base Model: `GPT-5`
- Runtime: `local macOS workspace plus A100 validation host`

### User Query

> Continue with the next milestone.

### Changes Overview

- Area: v0.1 OpenAI-compatible HTTP serving.
- Key actions:
  - Added protocol models for `/v1/models`, `/v1/responses`,
    `/v1/completions`, and `/v1/chat/completions`.
  - Added a FastAPI server and `nanollm-serve` CLI entry point for one loaded
    causal LM.
  - Added a text-only Responses API subset with `output_text`, usage, SSE
    deltas, and lightweight in-memory retrieval.
  - Expanded Responses endpoint tests to cover unknown models, unknown
    previous responses, and unknown retrieval IDs.
  - Added a future milestone for full Responses API lifecycle, tool-call, and
    event parity so the v0.1 text-only subset is not mistaken for the final
    shape.
  - Added incremental token streaming over server-sent events.
  - Extended the engine with a streaming generation iterator shared by the HTTP
    streaming path and final-response generation.
  - Added protocol, endpoint, SSE, and engine streaming tests.
  - Aligned completion streaming chunk object names and single-model request
    validation with vLLM/SGLang behavior.
  - Updated README quick starts, architecture notes, roadmap status, release
    notes, and quality score.

### Design Intent

The milestone exposes the existing naive single-request generator through a
stable OpenAI-compatible adapter without moving scheduling or model execution
policy into the API layer. Responses is the preferred new-project surface, while
Chat Completions and Completions remain compatibility endpoints. This keeps the
external interface useful while leaving KV cache, batching, routing, tools,
multimodal input, background runs, and auth to later milestones.

### Files Modified

- `pyproject.toml`
- `README.md`
- `README.zh-CN.md`
- `src/nanollmserve/api/openai_server.py`
- `src/nanollmserve/api/protocol.py`
- `src/nanollmserve/engine/engine.py`
- `tests/test_openai_server.py`
- `tests/test_protocol.py`
- `tests/test_engine.py`
- `docs/ARCHITECTURE.md`
- `docs/QUALITY_SCORE.md`
- `docs/exec-plans/active/nano-llm-serve-roadmap.md`
- `docs/exec-plans/active/milestones/v0.1-openai-compatible-server.md`
- `docs/releases/feature-release-notes.md`

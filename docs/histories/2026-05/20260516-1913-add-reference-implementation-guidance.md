## [2026-05-16 19:13] | Task: Add reference implementation guidance

### Execution Context

- Agent ID: `Codex`
- Base Model: `GPT-5`
- Runtime: `Codex CLI`

### User Query

> Add guidance that implementation work may reference `./tmp/vllm` and
> `./tmp/sglang` to compare against mature serving systems and extract useful
> ideas while still building this repository's own implementation.

### Changes Overview

- Area: Architecture documentation.
- Key actions: Added an implementation reference policy for local vLLM and
  SGLang checkouts.

### Design Intent

The guidance belongs in `docs/ARCHITECTURE.md` because it affects how agents
design serving features and package boundaries. `AGENTS.md` remains a short
routing document that points agents to the architecture docs as the source of
record.

### Files Modified

- `docs/ARCHITECTURE.md`
- `docs/histories/2026-05/20260516-1913-add-reference-implementation-guidance.md`

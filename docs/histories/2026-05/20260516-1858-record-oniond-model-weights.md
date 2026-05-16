## [2026-05-16 18:58] | Task: record oniond model weights

### Execution Context

- Agent ID: `codex`
- Base Model: `GPT-5`
- Runtime: `local CLI with SSH and Open Browser Use`

### User Query

> Investigate the A100 harness, Hugging Face mirror, and internal Quick Start notes;
> pull the recommended sub-10B Qwen3 model weights with `oniond`; record model
> locations and `oniond` usage in `.harness`.

### Changes Overview

- Area: harness documentation
- Key actions:
  - Verified `oniond` availability on `gpu-A100-05`.
  - Downloaded Qwen3 0.6B, 1.7B, 4B, and 8B weights to `/data2/nanoLLMServe/models`.
  - Recorded local model paths, sizes, `from_pretrained` usage, and `oniond` commands.
  - Noted the global Conda NumPy/sklearn/scipy ABI issue observed during lightweight validation.

### Design Intent

Keep machine-specific operational knowledge in `.harness` so future agents can
use local model paths without rediscovering download steps or relying on chat
context. Prefer cached `oniond` downloads on the A100 machine and reserve
personal Hugging Face tokens as a fallback for uncached models.

### Files Modified

- `.harness/gpu-a100-80g.md`
- `docs/histories/2026-05/20260516-1858-record-oniond-model-weights.md`

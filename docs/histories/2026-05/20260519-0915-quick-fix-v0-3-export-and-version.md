## [2026-05-19 09:15] | Task: Fix v0.3 API export and version alignment

### Execution Context

- Agent ID: `Codex`
- Base Model: `gpt-5`
- Runtime: `Python 3`

### User Query

> 复核 v0.3 改动并直接修正发现的问题。

### Changes Overview

- Area:
  - Top-level package API and release metadata consistency.
- Key actions:
  - Exported `generate_batch` from `nanollmserve/__init__.py` to match v0.3 milestone API.
  - Bumped package version metadata to `0.3.1` in `pyproject.toml`.
  - Bumped runtime-reported version string in `openai_server.py` to `0.3.1`.

### Design Intent

v0.3 静态批处理是里程碑级能力，顶层 API 可见性和版本一致性应与功能发布一致，避免教学演示和下游调用层出现版本/入口歧义。

### Files Modified

- `pyproject.toml`
- `src/nanollmserve/__init__.py`
- `src/nanollmserve/api/openai_server.py`

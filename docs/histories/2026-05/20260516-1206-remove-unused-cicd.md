## [2026-05-16 12:06] | Task: Remove unused CI/CD

### Execution Context

- Agent ID: `Codex`
- Base Model: `GPT-5`
- Runtime: `Codex CLI`

### User Query

> 清理一下无用的CICD，我们目前都不需要

### Changes Overview

- Area: CI/CD scaffolding and repository validation docs.
- Key actions: removed unused GitHub Actions workflows, release packaging, dependency review config, and action pinning checks; kept local repository validation through `make check-repo`.

### Design Intent

The repository should not carry template CI/CD automation before the project has a real stack, release artifact, or deployment target. Local checks remain available so contributors and agents can still validate repository hygiene without implying remote automation exists.

### Files Modified

- `.github/workflows/ci.yml`
- `.github/workflows/release.yml`
- `.github/workflows/supply-chain-security.yml`
- `.github/dependency-review-config.yml`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `Makefile`
- `AGENTS.md`
- `CONTRIBUTING.md`
- `scripts/check-action-pinning.sh`
- `scripts/ci.sh`
- `scripts/release-package.sh`
- `scripts/check-repo-hygiene.sh`
- `docs/CICD.md`
- `docs/SUPPLY_CHAIN_SECURITY.md`
- `docs/SECURITY.md`
- `docs/RELIABILITY.md`
- `docs/REPO_COLLAB_GUIDE.md`

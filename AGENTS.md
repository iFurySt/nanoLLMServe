# nanoLLMServe

This repository builds a tiny, readable LLM serving engine.

The project is teaching-first, but the implementation standard is production-minded. Readability is not an excuse for vague correctness or weak performance. On single-machine and single-GPU paths, features should aim to match the behavior and performance shape of serious serving systems, even when this repo cannot cover large-scale distributed production scenarios.

`AGENTS.md` stays short on purpose. Treat it as a map, not the encyclopedia. Repository-local markdown under `docs/` is the system of record.

If a code or workflow change makes a doc stale, update the doc in the same task.

## Read At The Start Of Each Task

- `docs/REPO_COLLAB_GUIDE.md`: repository-wide collaboration, commit, documentation, and testing expectations.
- `docs/ARCHITECTURE.md`: top-level architecture map and intended package boundaries.
- `docs/design-docs/core-beliefs.md`: agent-first operating principles and repository design intent.

## Read Before Finishing A Code Change

- `docs/HISTORY_GUIDE.md`: when to record code changes, naming rules, and redaction rules.
- `docs/QUALITY_SCORE.md`: current quality targets and gaps by area.

## Read When The Task Needs It

- `docs/PLANS_GUIDE.md`: when to create an execution plan and how to maintain it.
- `docs/PRODUCT_SENSE.md`: user value, product constraints, and feature prioritization heuristics.
- `docs/RELIABILITY.md`: runtime guardrails, observability expectations, and operational readiness.
- `docs/SECURITY.md`: secure defaults for auth, data handling, secrets, and external integrations.
- `docs/SUPPLY_CHAIN_SECURITY.md`: dependency and repository supply-chain security posture.
- `docs/CICD.md`: current CI/CD posture and when to reintroduce automation.
- `docs/FRONTEND.md`: UI/system guidance if the repo includes a frontend surface.
- `CONTRIBUTING.md`: pull request expectations and default review checklist.
- `docs/releases/README.md`: how to maintain user-facing release notes.
- `docs/references/README.md`: curated external references copied into the repo for agent use.

## Working Rules

- Prefer small, explicit, repository-legible abstractions.
- Treat each feature as a production concept implemented at teaching scale.
- Add or update tests for repeated correctness checks before calling work complete.
- Include functional validation and performance validation for serving changes.
- Do not hand-wave performance-sensitive paths; benchmark them and record the result.
- For single-machine or single-GPU behavior, keep the bar close to production serving expectations.
- Keep prompts, policies, and architectural rules versioned in-repo.
- For complex work, create an execution plan instead of relying on long chat context.
- Record finished code changes in `docs/histories/`.

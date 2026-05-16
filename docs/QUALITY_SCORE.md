# Quality Score

Track quality by product area and architectural layer so agents can prioritize the weakest parts of the system.

## Suggested Scale

- `A`: strong coverage, stable behavior, clear docs, low operational risk.
- `B`: acceptable but still has known gaps.
- `C`: works but needs targeted hardening.
- `D`: fragile or underspecified.

## Initial Template

| Area | Score | Why | Next Step |
| --- | --- | --- | --- |
| Product surface | C | v0.0 defines a CLI path for one prompt and one causal LM. | Add the OpenAI-compatible HTTP API in v0.1. |
| Architecture docs | B | Package boundaries and v0.0 runtime flow are documented. | Revisit when HTTP serving changes the topology. |
| Testing | C | Unit tests cover CLI parsing/main flow, sampling edge cases, naive decode loop boundaries, and HF loader argument/dependency behavior; local no-torch environments skip torch-specific checks. | Add real-model smoke tests once CI has an appropriate runner or cached tiny model. |
| Observability | D | No local stack or conventions yet. | Document logs, metrics, traces, and local access. |
| Security | C | Defaults are documented, implementation is pending. | Add real auth, secret, and dependency rules. |

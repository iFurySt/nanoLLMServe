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
| Product surface | B | v0.1 exposes one causal LM through CLI plus OpenAI-compatible Responses, Chat Completions, Completions, and Models endpoints. | Add KV cache decode in v0.2 so the server has a real serving-performance concept. |
| Architecture docs | B | Package boundaries and v0.1 CLI/HTTP runtime flows are documented. | Revisit when KV cache changes engine/model boundaries. |
| Testing | B | Unit tests cover CLI parsing/main flow, protocol models, FastAPI endpoints including Responses, SSE chunking, sampling edge cases, naive decode loop boundaries, and HF loader behavior; local no-torch or no-FastAPI environments skip dependency-specific checks. | Add real-model HTTP smoke tests once CI has an appropriate runner or cached tiny model. |
| Observability | D | No local stack or conventions yet. | Document logs, metrics, traces, and local access. |
| Security | C | Defaults are documented, implementation is pending. | Add real auth, secret, and dependency rules. |

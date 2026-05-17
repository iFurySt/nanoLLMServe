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
| Product surface | B | v0.2 exposes one causal LM through CLI plus OpenAI-compatible HTTP endpoints and now uses KV-cache prefill/decode instead of full-sequence recompute. | Add static batching in v0.3 so multiple prompts can share one model step. |
| Architecture docs | B | Package boundaries and v0.2 CLI/HTTP runtime flows are documented, including request state and direct Hugging Face `past_key_values` reuse. | Revisit when static batching changes request lifecycle and scheduler boundaries. |
| Testing | B | Unit tests cover CLI parsing/main flow, protocol models, FastAPI endpoints including Responses, SSE chunking, sampling edge cases, KV-cache decode behavior, and HF loader behavior; local no-torch or no-FastAPI environments skip dependency-specific checks. | Add automated real-model HTTP smoke tests once CI has an appropriate runner or cached tiny model. |
| Observability | D | No local stack or conventions yet. | Document logs, metrics, traces, and local access. |
| Security | C | Defaults are documented, implementation is pending. | Add real auth, secret, and dependency rules. |

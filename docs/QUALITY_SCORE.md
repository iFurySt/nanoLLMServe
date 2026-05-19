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
| Product surface | B | v0.4 adds scheduler-level continuous batching with waiting/running/finished state, active batch metrics, and benchmark output; HTTP endpoints still use the single-request path. | Add paged/block KV cache so dynamic batches can keep per-row KV state instead of rebuilding full-token rows. |
| Architecture docs | B | Package boundaries and runtime flow now document `generate_one`, static `generate_batch`, and scheduler-level `generate_continuous_batch`. | Revisit when v0.5 block KV cache changes cache ownership and scheduler/cache contracts. |
| Testing | B | Unit tests cover static-batch generation behavior, continuous batching admission/removal and active batch sizes, response lifecycle, response cancellation, continuation, and response SSE sequencing. | Add GPU benchmark validation for real model continuous batching and later paged-KV behavior. |
| Observability | D | No local stack or conventions yet. | Document logs, metrics, traces, and local access. |
| Security | C | Defaults are documented, implementation is pending. | Add real auth, secret, and dependency rules. |

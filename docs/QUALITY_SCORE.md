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
| Product surface | B | v0.5 adds block KV allocation metadata, generation lifecycle hooks, and fragmentation benchmark output; HTTP endpoints still use the single-request path. | Add prefix cache and later tensor-level paged KV execution so block tables affect real KV tensor reuse. |
| Architecture docs | B | Package boundaries and runtime flow now document `generate_one`, static `generate_batch`, scheduler-level `generate_continuous_batch`, and `KVBlockManager`. | Revisit when v0.6 prefix cache introduces cross-request cache lookup and eviction. |
| Testing | B | Unit tests cover block allocation/release/fragmentation metrics, generation block lifecycle hooks, static batching, continuous batching admission/removal, and response lifecycle behavior. | Add GPU benchmark validation for real model block metadata and later tensor-level paged-KV behavior. |
| Observability | D | No local stack or conventions yet. | Document logs, metrics, traces, and local access. |
| Security | C | Defaults are documented, implementation is pending. | Add real auth, secret, and dependency rules. |

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
| Product surface | B | v0.6 adds single-request prefix-cache reuse, visible cache stats, and repeated-prefix benchmark output; batch prefix reuse and tensor-level paged KV remain future work. | Add chunked prefill and later connect prefix reuse to batch scheduling and tensor block ownership. |
| Architecture docs | B | Package boundaries and runtime flow now document `generate_one`, static `generate_batch`, scheduler-level `generate_continuous_batch`, `KVBlockManager`, and `PrefixCache`. | Revisit when v0.7 chunked prefill changes long-prompt admission and prefill scheduling. |
| Testing | B | Unit tests cover prefix hashing, longest-prefix lookup, ref counts, LRU eviction, single-request prefix prefill skipping, block allocation/release, batching, and response lifecycle behavior. | Add GPU benchmark validation for real model prefix-cache TTFT and later tensor-level paged-KV behavior. |
| Observability | D | No local stack or conventions yet. | Document logs, metrics, traces, and local access. |
| Security | C | Defaults are documented, implementation is pending. | Add real auth, secret, and dependency rules. |

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
| Product surface | B | v0.7 adds chunked prefill for mixed long/short prompt workloads with visible scheduler steps and a benchmark; HTTP endpoints still use the single-request path. | Add observability so scheduler, cache, and latency counters are exposed outside benchmark JSON. |
| Architecture docs | B | Package boundaries and runtime flow now document `generate_one`, static `generate_batch`, continuous batching, `KVBlockManager`, `PrefixCache`, and chunked prefill scheduling. | Revisit when v0.8 introduces metrics/exporter boundaries. |
| Testing | B | Unit tests cover chunked prefill scheduling, short-prompt prefill priority, prefix cache, block allocation/release, batching, and response lifecycle behavior. | Add GPU benchmark validation for real-model chunked prefill and later production-like mixed prefill/decode batching. |
| Observability | D | No local stack or conventions yet. | Document logs, metrics, traces, and local access. |
| Security | C | Defaults are documented, implementation is pending. | Add real auth, secret, and dependency rules. |

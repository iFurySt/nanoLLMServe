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
| Product surface | B | v0.2 exposes one causal LM through CLI plus OpenAI-compatible HTTP endpoints with KV-cache prefill/decode; v0.3 static batching adds fixed-batch throughput path via benchmark path. | Add Responses text-only continuation/cancel/stateful SSE work and continue toward tool/capability parity. |
| Architecture docs | B | Package boundaries and runtime flow are documented with `generate_one` and new `generate_batch` static-batch semantics. | Revisit when batch scheduling boundaries shift toward continuous batching in v0.4. |
| Testing | B | Unit tests cover static-batch generation behavior, response lifecycle, response cancellation, continuation, and response SSE sequencing; some model/tokenizer tests remain dependency-skipped without local torch/transformer runtime. | Add tests for background execution and tool-call structured output as those runtimes are added. |
| Observability | D | No local stack or conventions yet. | Document logs, metrics, traces, and local access. |
| Security | C | Defaults are documented, implementation is pending. | Add real auth, secret, and dependency rules. |

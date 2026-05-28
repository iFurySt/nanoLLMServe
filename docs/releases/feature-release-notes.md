# Feature Release Notes

## 2026-05

| Date | Area | User Impact | Change Summary |
| --- | --- | --- | --- |
| 2026-05-28 | v0.7 Chunked prefill | Users can benchmark mixed long/short prompt workloads where long prefill is split across scheduler steps so shorter requests can reach first token earlier. | Added `generate_chunked_prefill_batch`, chunked prefill step stats, decode-first/shortest-prefill-first policy, `benchmark_chunked_prefill.py`, tests, and docs. |
| 2026-05-22 | v0.6 Prefix cache | Users can reuse shared prompt-prefix KV state on the single-request path and benchmark repeated-prefix TTFT behavior. | Added `PrefixCache`, prefix token hashing, longest-prefix lookup, ref counts, LRU eviction, `generate_one` prefix-cache hooks, hit/miss stats, and `benchmark_prefix_cache.py`. |
| 2026-05-22 | v0.5 Block KV cache manager | Users can inspect KV block allocation, release, utilization, and fragmentation behavior through tests and a benchmark. | Added `KVBlockManager`, request block tables, free block pool, generation lifecycle hooks, usage metrics, and `benchmark_block_manager.py`. |
| 2026-05-19 | v0.4 Continuous batching | Users can benchmark scheduler-level continuous batching where requests arrive and finish across scheduler steps. | Added `ContinuousBatchScheduler`, `generate_continuous_batch`, per-step active batch metrics, benchmark `continuous_batch` output, and tests for admission/removal behavior. |
| 2026-05-18 | v0.3 Responses API | Users can cancel running responses, continue from prior turns, and parse deterministic response SSE ordering. | Added `/v1/responses/{response_id}/cancel`, `previous_response_id` continuation for text-only prompts, in-memory response state transitions, and ordered SSE `sequence_number` events. |
| 2026-05-18 | v0.3 Static batching | Users can benchmark fixed-size batching for multiple prompts in one model forward path. | Added `engine.generate_batch` with padded prompt tokenization, lock-step prefill/decode over fixed batches, new batch benchmark output (`--batch-size`, `static_batch`), and tests for batch behavior/coverage. |
| 2026-05-17 | v0.2 KV cache decode | Users get a readable prefill/decode generation path that reuses Hugging Face KV cache and reports TTFT/TPOT benchmark metrics. | Added request generation state, `past_key_values` reuse, decode-only token forwards after prefill, CLI timing stats, naive-vs-KV benchmark comparison, tests, and docs. |
| 2026-05-17 | v0.1 HTTP API | Users can call one loaded causal LM through OpenAI-compatible Responses, chat completions, completions, models, and streaming endpoints. | Added FastAPI server, protocol models, `nanollm-serve`, Responses support, SSE chunking, endpoint tests, and curl documentation. |
| 2026-05-16 | v0.0 CLI | Users can load a Hugging Face causal LM and generate text for one prompt from the command line. | Added the `nanollm-generate` CLI, naive decode loop, greedy and temperature sampling, tests, and a minimal benchmark. |

## 2026-04

| Date | Area | User Impact | Change Summary |
| --- | --- | --- | --- |
| 2026-04-08 | Template | Introduced the base harness repository template for future services and products. | Added agent entry docs, execution-plan scaffolding, change-history templates, and docs checks. |

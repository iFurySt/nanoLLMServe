# Architecture

This file is the top-level map for the repository. It fixes the intended
package boundaries before later milestones add serving complexity.

## Current Repository Shape

- `src/nanollmserve/`: the teaching-scale serving engine package.
  - `api/`: external API surfaces and wire protocols. `v0.1` starts here with
    an OpenAI-compatible server and request/response models.
  - `cli/`: command-line tools. CLI modules should parse arguments and delegate
    immediately to package APIs.
  - `engine/`: request lifecycle orchestration. The engine owns admission,
    scheduling calls, decode-step sequencing, and output assembly.
  - `model/`: model loading and model-runner code. Hugging Face integration
    lives here; model runners should not own scheduling policy.
  - `sampling/`: sampling parameters, logits processing, and token selection.
  - `cache/`: KV cache metadata, block allocation, prefix cache, and radix-tree
    data structures.
  - `worker/`: local execution workers, initially single-process/single-GPU.
  - `distributed/`: routers and worker coordination for multi-process or
    multi-node serving.
  - `metrics/`: internal stats structs and exporters such as Prometheus.
  - `structured_output/`: grammar- or schema-constrained decoding.
- `benchmarks/`: runnable local benchmarks for serving behavior.
- `tests/`: focused correctness checks for package behavior.
- `infra/`: deployment, infrastructure, and environment definitions.
- `scripts/`: repository automation that agents can run directly.
- `docs/`: the repository knowledge base and system of record.

The runtime now includes:

- `v0.2` single-request path: tokenizes one prompt, runs a prefill forward over
  the prompt, then decodes later tokens by passing only the last generated token
  with `past_key_values`.
- `v0.3` static batching path: tokenizes a fixed list of prompts with
  padding/masks, runs one prefill pass over the batch, then performs lock-step
  decode and finalizes when all rows finish or reach `max_new_tokens`.
- `v0.4` continuous batching path: admits requests by scheduler step, keeps
  waiting/running/finished request lifecycle state, rebuilds a padded active
  batch each decode step, records active batch size metrics, and removes
  completed rows without stopping the scheduler.
- `v0.5` block KV cache manager: tracks fixed-size token blocks, a free block
  pool, request-to-block tables, allocation/release, and fragmentation metrics.
  The manager is allocator metadata and is wired into generation lifecycle; it
  does not replace Hugging Face `past_key_values` with GPU block tensors yet.

The CLI and HTTP server both use the single-request path. `generate_batch` and
`generate_continuous_batch` are used by benchmarked batching scenarios while
regular endpoints still call the single-request path by default.

## Target Package Layout

The layout is intentionally closer to vLLM v1's explicit `engine`, `core`,
`worker`, `sample`, and `metrics` boundaries than to SGLang's broad `managers`
bucket. SGLang's `mem_cache` and scheduler ideas still inform `cache/` and
`engine/scheduler.py`, but this repository keeps names direct so each serving
concept has an obvious home.

```text
src/nanollmserve/
  api/
    openai_server.py
    protocol.py
  cli/
    generate.py
  engine/
    engine.py
    request.py
    scheduler.py
  model/
    hf_runner.py
  sampling/
    params.py
    sampler.py
  cache/
    kv_cache.py
    block_manager.py
    prefix_cache.py
    radix_tree.py
  worker/
    gpu_worker.py
  distributed/
    router.py
    worker.py
  metrics/
    stats.py
    prometheus.py
  structured_output/
```

Do not add empty subpackages beyond this map just to reserve names. Create a new
subpackage when a milestone needs an implementation boundary that cannot be
expressed cleanly in one of the directories above.

## Boundary Rules

- Put serving logic in `src/nanollmserve/` before adding application wrappers.
- Keep CLI code thin; argument parsing belongs in `cli/`, generation behavior
  belongs in `engine/`, model execution belongs in `model/`, and sampling policy
  belongs in `sampling/`.
- `api/` may depend on `engine/` contracts, but `engine/` must not depend on
  `api/` protocols. OpenAI compatibility is an adapter, not the core contract.
- `engine/` may coordinate `model/`, `sampling/`, `cache/`, `worker/`, and
  `metrics/`. Those lower-level packages should avoid importing concrete API or
  CLI modules.
- `cache/` owns allocation and prefix-reuse data structures; `model/` owns
  tensors and forward calls; `engine/` decides when a request receives cache
  blocks.
- `worker/` is for local process/device execution. `distributed/` is only for
  coordination across processes or machines.
- `metrics/stats.py` should hold plain data structures that are easy to test.
  Exporter-specific code belongs in files such as `metrics/prometheus.py`.
- Keep benchmarks as runnable scripts that import the package instead of copying
  model logic.
- Keep infrastructure and runtime orchestration explicit and versioned.
- Avoid hidden cross-package coupling; document allowed dependency directions once the stack is real.
- When the architecture changes, update this file in the same task.

## Implementation References

When implementing serving features, agents may inspect `./tmp/vllm` and
`./tmp/sglang` as local reference checkouts. Use them to understand production
behavior, performance shape, naming, tests, and failure handling before designing
the teaching-scale version here.

Reference those projects comparatively rather than copying them wholesale:

- Prefer extracting the smallest transferable serving idea that fits
  `nanoLLMServe`'s current milestone.
- Preserve this repository's readable package boundaries and explicit control
  flow even when the reference implementation is more optimized or distributed.
- If a behavior is intentionally different from vLLM or SGLang, document the
  reason in the relevant plan, architecture note, test, benchmark, or history
  entry.
- For performance-sensitive features, compare against the reference projects'
  behavior where practical, then record the local benchmark result.

## Runtime Flow

```text
CLI args
  -> load_model_and_tokenizer()
  -> tokenizer(prompt)
  -> engine.generate_one()
       -> create GenerationRequestState
       -> prefill: model(prompt input_ids, attention_mask, use_cache=True)
       -> sample first token
       -> decode: model(last token, attention_mask, past_key_values, use_cache=True)
       -> sample and append each later token
  -> tokenizer.decode(generated_token_ids)

HTTP JSON
  -> api.protocol request models
  -> api.openai_server FastAPI endpoint
  -> engine.generate_one() or engine.stream_generate_one()
  -> api.protocol response models or SSE chunks
  
Batch CLI path
  -> CLI args or test harness
  -> tokenizer(prompts, return_tensors="pt", padding=True)
  -> engine.generate_batch()
      -> prefill over full padded batch
      -> sample first token from each row's last valid position
      -> lock-step decode with past_key_values and growing attention masks
      -> stop when all rows are finished or reached max_new_tokens
      -> return row-wise GenerationResult

Continuous batch benchmark path
  -> benchmark creates ContinuousBatchRequest rows with arrival_step
  -> engine.generate_continuous_batch()
      -> ContinuousBatchScheduler moves rows waiting -> running -> finished
      -> each scheduler step rebuilds a padded full-token active batch
      -> model(active input_ids, attention_mask, use_cache=False)
      -> sample one token per active row
      -> completed rows leave the running set before the next step
      -> return per-request results plus active_batch_size step metrics

Block KV cache manager path
  -> KVBlockManager(total_blocks, block_size)
  -> generation prefill allocates prompt token blocks per request
  -> each generated token appends capacity and may allocate a new block
  -> completion releases request block tables back to the free pool
  -> benchmarks/benchmark_block_manager.py reports reserved tokens,
     internal fragmentation, utilization, and a contiguous-slot comparison
```

## Known Gaps

- KV cache reuse uses Hugging Face `past_key_values` directly. v0.5 adds block
  allocation metadata and lifecycle hooks, but it does not implement a
  PagedAttention kernel, tensor paging, eviction, or prefix reuse yet.
- Static batching exists for a fixed batch size in `engine.generate_batch`.
  Continuous batching exists at scheduler level in `generate_continuous_batch`,
  but it still rebuilds full-token active batches; v0.5 exposes block ownership
  metadata before v0.6 prefix cache and later tensor-level paging work.
- The OpenAI-compatible server covers the common `/v1/models`,
  `/v1/responses`, `/v1/chat/completions`, and `/v1/completions` shapes, but it
  does not implement auth, tools, background Responses runs, Responses cancel,
  multi-choice generation, logprobs, multimodal messages, or model routing.
  Full Responses lifecycle and tool parity is tracked in
  `docs/exec-plans/active/milestones/v0.10-responses-api-parity.md`.

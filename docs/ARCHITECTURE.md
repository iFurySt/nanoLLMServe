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

The v0.0 runtime is intentionally a single-process Python CLI. It loads one
Hugging Face causal LM, tokenizes one prompt, repeatedly runs the model over the
full growing sequence, samples one token, and decodes only the generated tokens.
This path is deliberately naive so later milestones can make the performance
reason for KV cache decode and batching visible.

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
       -> model(input_ids, attention_mask)
       -> sampling.sample_next_token()
       -> append token
  -> tokenizer.decode(generated_token_ids)
```

## Known Gaps

- No HTTP API yet; that begins in `v0.1`.
- No explicit KV cache handling yet; `v0.0` relies on default model behavior and
  recomputes the growing sequence to keep the baseline readable.
- No batching yet; all APIs are intentionally single prompt / single request.

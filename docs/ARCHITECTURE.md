# Architecture

This file is the top-level map for the repository. Replace the placeholders below as the real project takes shape.

## Current Repository Shape

- `src/nanollmserve/`: the teaching-scale serving engine package.
  - `engine.py`: the v0.0 naive single-request decode loop.
  - `sampling.py`: token selection policies for greedy and temperature sampling.
  - `modeling.py`: Hugging Face model and tokenizer loading.
  - `cli.py`: command-line generation entry point.
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

## Boundary Rules

- Put serving logic in `src/nanollmserve/` before adding application wrappers.
- Keep CLI code thin; argument parsing belongs in `cli.py`, generation behavior
  belongs in `engine.py`, and sampling policy belongs in `sampling.py`.
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
  -> generate_one()
       -> model(input_ids, attention_mask)
       -> sample_next_token()
       -> append token
  -> tokenizer.decode(generated_token_ids)
```

## Known Gaps

- No HTTP API yet; that begins in `v0.1`.
- No explicit KV cache handling yet; `v0.0` relies on default model behavior and
  recomputes the growing sequence to keep the baseline readable.
- No batching yet; all APIs are intentionally single prompt / single request.

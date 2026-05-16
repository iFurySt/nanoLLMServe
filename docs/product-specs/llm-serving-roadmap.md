# LLM Serving Roadmap

`nanoLLMServe` is a teaching-first LLM serving engine.

The product promise is simple: every version runs, every version introduces one
serving concept, and every concept maps to a real production problem.

## Audience

- Engineers who want to understand vLLM/SGLang-style serving.
- Builders who prefer readable code before custom kernels.
- Agents and contributors who need a repo-level plan that survives chat context.

## Principles

- Run a real small model in every milestone.
- Keep the first implementation simple before optimizing it.
- Add benchmarks when a feature claims performance value.
- Document why each feature exists before expanding its scope.
- Prefer Python, PyTorch, Transformers, and FastAPI until the serving concepts are clear.

## Milestone Order

1. `v0.0` naive single request
2. `v0.1` OpenAI-compatible server
3. `v0.2` KV cache decode
4. `v0.3` static batching
5. `v0.4` continuous batching
6. `v0.5` block KV cache manager
7. `v0.6` prefix cache
8. `v0.7` chunked prefill
9. `v0.8` observability
10. `v0.9` structured output
11. `v1.0` speculative decoding
12. `v1.1` LoRA serving
13. `v1.2` quantization
14. `v1.3` multi-GPU minimal
15. `v1.4` tensor parallel toy version

Execution files live in [`../exec-plans/active/milestones`](../exec-plans/active/milestones).

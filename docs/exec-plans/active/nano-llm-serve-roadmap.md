# nanoLLMServe Roadmap

## Goal

Build a readable LLM serving engine through small, runnable milestones.

## Scope

- In scope: core serving concepts from naive inference through batching, KV cache,
  prefix reuse, chunked prefill, observability, constrained decoding, speculative
  decoding, LoRA, quantization, and minimal distributed serving.
- Out of scope: beating vLLM/SGLang on performance, production hardening before
  the concept is readable, or custom kernels before the Python path is clear.

## Context

- Product spec: [`../../product-specs/llm-serving-roadmap.md`](../../product-specs/llm-serving-roadmap.md)
- Milestones: [`./milestones`](./milestones)
- Expected first model: Qwen2.5-0.5B, Llama-3.2-1B, or another small HF causal LM.

## Milestones

- [x] [`v0.0` naive single request](./milestones/v0.0-naive-single-request.md)
- [x] [`v0.1` OpenAI-compatible server](./milestones/v0.1-openai-compatible-server.md)
- [x] [`v0.2` KV cache decode](./milestones/v0.2-kv-cache-decode.md)
- [x] [`v0.3` static batching](./milestones/v0.3-static-batching.md)
- [x] [`v0.4` continuous batching](./milestones/v0.4-continuous-batching.md)
- [x] [`v0.5` block KV cache manager](./milestones/v0.5-block-kv-cache-manager.md)
- [ ] [`v0.6` prefix cache](./milestones/v0.6-prefix-cache.md)
- [ ] [`v0.7` chunked prefill](./milestones/v0.7-chunked-prefill.md)
- [ ] [`v0.8` observability](./milestones/v0.8-observability.md)
- [ ] [`v0.9` structured output](./milestones/v0.9-structured-output.md)
- [ ] [`v0.10` Responses API parity](./milestones/v0.10-responses-api-parity.md)
- [ ] [`v1.0` speculative decoding](./milestones/v1.0-speculative-decoding.md)
- [ ] [`v1.1` LoRA serving](./milestones/v1.1-lora-serving.md)
- [ ] [`v1.2` quantization](./milestones/v1.2-quantization.md)
- [ ] [`v1.3` multi-GPU minimal](./milestones/v1.3-multi-gpu-minimal.md)
- [ ] [`v1.4` tensor parallel toy version](./milestones/v1.4-tensor-parallel-toy-version.md)

## Validation Standard

Each milestone should include:

- A runnable demo.
- A small benchmark.
- Tests for the new behavior where practical.
- A short doc explaining why the feature exists.
- Before/after metrics when the feature is performance-related.

## Decision Log

- 2026-05-16: Use `nanoLLMServe` as the project name and track roadmap milestones as repo files.
- 2026-05-17: Keep the v0.1 Responses API as a text-only teaching subset and
  schedule full Responses lifecycle/tool parity as `v0.10`, after structured
  output and before speculative decoding.

# nanoLLMServe

[![English](https://img.shields.io/badge/English-Click-yellow)](./README.md)
[![简体中文](https://img.shields.io/badge/简体中文-点击查看-orange)](./README.zh-CN.md)

`nanoLLMServe` is a tiny, readable LLM serving engine.

The goal is to build a small vLLM/SGLang-style system that can actually run,
while keeping each production serving idea easy to inspect:

- OpenAI-compatible API
- KV cache decode
- batching and continuous batching
- block KV cache management
- prefix cache
- chunked prefill
- metrics, structured output, speculative decoding, LoRA, quantization, and distributed serving

It is not trying to be faster than vLLM. It is trying to make the serving stack
understandable.

## Roadmap

Milestones live in [`docs/exec-plans/active/milestones`](./docs/exec-plans/active/milestones).

Start with [`v0.0-naive-single-request`](./docs/exec-plans/active/milestones/v0.0-naive-single-request.md).

## License

[MIT](LICENSE)

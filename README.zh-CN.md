# nanoLLMServe

[![English](https://img.shields.io/badge/English-Click-yellow)](./README.md)
[![简体中文](https://img.shields.io/badge/简体中文-点击查看-orange)](./README.zh-CN.md)

`nanoLLMServe` 是一个小而可读的 LLM 推理服务引擎。

目标是做一个能跑起来的 vLLM/SGLang 风格系统，把生产级 serving 里的核心概念拆开讲清楚：

- OpenAI-compatible API
- KV cache decode
- batching 和 continuous batching
- block KV cache manager
- prefix cache
- chunked prefill
- metrics、structured output、speculative decoding、LoRA、quantization 和 distributed serving

它不是为了比 vLLM 更快，而是为了让 LLM serving stack 更容易理解。

## Roadmap

Milestone 放在 [`docs/exec-plans/active/milestones`](./docs/exec-plans/active/milestones)。

从 [`v0.0-naive-single-request`](./docs/exec-plans/active/milestones/v0.0-naive-single-request.md) 开始。

## License

[MIT](LICENSE)

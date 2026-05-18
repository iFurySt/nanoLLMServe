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

从 [`v0.0-naive-single-request`](./docs/exec-plans/active/milestones/v0.0-naive-single-request.md) 开始，
再进入 [`v0.1 OpenAI-compatible server`](./docs/exec-plans/active/milestones/v0.1-openai-compatible-server.md)，
然后切到 [`v0.2 KV cache decode`](./docs/exec-plans/active/milestones/v0.2-kv-cache-decode.md)。

## Quick Start

先创建隔离环境并安装：

```bash
uv venv --python python3
source .venv/bin/activate
uv pip install -e ".[dev]"
```

对单个 prompt 生成文本：

```bash
nanollm-generate \
  --model Qwen/Qwen3-1.7B \
  --prompt "Explain KV cache in one sentence." \
  --max-new-tokens 32 \
  --temperature 0 \
  --show-stats
```

A100 harness 机器上优先使用已缓存的本地权重：

```bash
nanollm-generate \
  --model /data2/nanoLLMServe/models/Qwen3-1.7B \
  --local-files-only \
  --prompt "Explain KV cache in one sentence." \
  --max-new-tokens 32 \
  --temperature 0 \
  --show-stats
```

## v0.2 KV Cache Decode

默认生成路径现在会先对 prompt 做一次 prefill forward，然后复用 Hugging Face
`past_key_values`，后续每个 decode step 只喂最后一个生成 token。

运行 benchmark，并和 v0.0 风格的 naive 全量重算路径对比：

```bash
python benchmarks/benchmark_generate.py \
  --model /data2/nanoLLMServe/models/Qwen3-1.7B \
  --local-files-only \
  --runs 3 \
  --warmup 1
```

JSON 输出会包含 `kv_cache_decode.mean_ttft_seconds`、
`kv_cache_decode.mean_tpot_seconds`，以及 `comparison` 里的 elapsed 和 TPOT
speedup。

## v0.3 静态批处理（教学规模）

固定批大小静态批处理通过 `--batch-size` 开启（1 即退化为单请求）：

```bash
python benchmarks/benchmark_generate.py \
  --model /data2/nanoLLMServe/models/Qwen3-1.7B \
  --local-files-only \
  --batch-size 4 \
  --runs 5 \
  --warmup 2
```

JSON 结果会新增 `static_batch`，包含固定批次下的耗时、TTFT、TPOT 与行级吞吐统计。

## v0.1 OpenAI-Compatible Server

启动一个只服务单个本地或 Hugging Face causal LM 的 HTTP server：

```bash
nanollm-serve \
  --model /data2/nanoLLMServe/models/Qwen3-1.7B \
  --served-model-name Qwen3-1.7B \
  --local-files-only \
  --host 127.0.0.1 \
  --port 8000
```

查看模型列表：

```bash
curl http://127.0.0.1:8000/v1/models
```

调用推荐的 Responses endpoint：

```bash
curl http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-1.7B",
    "instructions": "Answer in one sentence.",
    "input": "Explain KV cache.",
    "max_output_tokens": 32,
    "temperature": 0
  }'
```

调用 chat completions：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-1.7B",
    "messages": [{"role": "user", "content": "Explain KV cache in one sentence."}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

Streaming 使用 OpenAI 风格的 server-sent events：

```bash
curl -N http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3-1.7B",
    "input": "KV cache is",
    "max_output_tokens": 16,
    "temperature": 0,
    "stream": true
  }'
```

## License

[MIT](LICENSE)

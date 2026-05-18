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

Start with [`v0.0-naive-single-request`](./docs/exec-plans/active/milestones/v0.0-naive-single-request.md),
expose it through the [`v0.1 OpenAI-compatible server`](./docs/exec-plans/active/milestones/v0.1-openai-compatible-server.md),
then switch generation to [`v0.2 KV cache decode`](./docs/exec-plans/active/milestones/v0.2-kv-cache-decode.md).

## Quick Start

Install the package in a virtual environment:

```bash
uv venv --python python3
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Generate text for one prompt:

```bash
nanollm-generate \
  --model Qwen/Qwen3-1.7B \
  --prompt "Explain KV cache in one sentence." \
  --max-new-tokens 32 \
  --temperature 0 \
  --show-stats
```

On the A100 harness machine, use the cached local weights:

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

The default generation path now runs one prefill pass over the prompt and then
uses Hugging Face `past_key_values` so each later decode step only receives the
last generated token.

Run the benchmark with a v0.0-style naive baseline comparison:

```bash
python benchmarks/benchmark_generate.py \
  --model /data2/nanoLLMServe/models/Qwen3-1.7B \
  --local-files-only \
  --runs 3 \
  --warmup 1
```

The JSON output includes `kv_cache_decode.mean_ttft_seconds`,
`kv_cache_decode.mean_tpot_seconds`, and a `comparison` section with elapsed and
TPOT speedup against the naive full-sequence loop.

## v0.3 Static Batching (Teaching-Scale)

You can benchmark fixed-size static batching with `--batch-size`:

```bash
python benchmarks/benchmark_generate.py \
  --model /data2/nanoLLMServe/models/Qwen3-1.7B \
  --local-files-only \
  --batch-size 4 \
  --runs 5 \
  --warmup 2
```

The output will include `static_batch`, including batch elapsed time and mean
row-level token throughput for the fixed-size group.

## v0.1 OpenAI-Compatible Server

Serve one local or Hugging Face causal LM:

```bash
nanollm-serve \
  --model /data2/nanoLLMServe/models/Qwen3-1.7B \
  --served-model-name Qwen3-1.7B \
  --local-files-only \
  --host 127.0.0.1 \
  --port 8000
```

List models:

```bash
curl http://127.0.0.1:8000/v1/models
```

Call the recommended Responses endpoint:

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

Call the chat completions endpoint:

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

Streaming uses OpenAI-style server-sent events:

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

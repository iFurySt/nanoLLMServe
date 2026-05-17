# Feature Release Notes

## 2026-05

| Date | Area | User Impact | Change Summary |
| --- | --- | --- | --- |
| 2026-05-17 | v0.1 HTTP API | Users can call one loaded causal LM through OpenAI-compatible Responses, chat completions, completions, models, and streaming endpoints. | Added FastAPI server, protocol models, `nanollm-serve`, Responses support, SSE chunking, endpoint tests, and curl documentation. |
| 2026-05-16 | v0.0 CLI | Users can load a Hugging Face causal LM and generate text for one prompt from the command line. | Added the `nanollm-generate` CLI, naive decode loop, greedy and temperature sampling, tests, and a minimal benchmark. |

## 2026-04

| Date | Area | User Impact | Change Summary |
| --- | --- | --- | --- |
| 2026-04-08 | Template | Introduced the base harness repository template for future services and products. | Added agent entry docs, execution-plan scaffolding, change-history templates, and docs checks. |

# Tech Debt Tracker

Track known debt that is real enough to preserve but not urgent enough to block the current task.

| Date | Area | Debt | Why It Exists | Planned Follow-Up |
| --- | --- | --- | --- | --- |
| 2026-05-17 | Responses API | v0.1 only implements a text-only `POST /v1/responses` and in-memory retrieve subset. It does not yet support cancel, durable state, tool calls, multimodal inputs, or full vLLM/SGLang-style event parity. | v0.1 needed a readable OpenAI-compatible server before the repository has scheduler, structured output, tools, or multimodal support. | Complete [`v0.10 Responses API parity`](./active/milestones/v0.10-responses-api-parity.md). |

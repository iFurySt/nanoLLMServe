"""OpenAI-compatible HTTP server for the naive generation engine."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable

from nanollmserve.api.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ListModelsResponse,
    ModelCard,
    ResponsesRequest,
    ResponsesResponse,
    build_chat_completion_response,
    build_completion_response,
    build_responses_response,
    chat_messages_to_prompt,
    finish_reason,
    new_response_id,
    now_timestamp,
    responses_input_to_prompt,
)
from nanollmserve.engine.engine import generate_one, stream_generate_one
from nanollmserve.model.hf_runner import LoadedModel, load_model_and_tokenizer


GenerateFn = Callable[..., object]
StreamGenerateFn = Callable[..., object]


@dataclass(frozen=True)
class ServerConfig:
    served_model_name: str
    loaded: LoadedModel
    generate_fn: GenerateFn = generate_one
    stream_generate_fn: StreamGenerateFn = stream_generate_one
    response_store: dict[str, ResponsesResponse] | None = None


def create_app(
    *,
    model_path: str | None = None,
    served_model_name: str | None = None,
    device: str = "auto",
    dtype: str = "auto",
    local_files_only: bool = False,
    loaded: LoadedModel | None = None,
    generate_fn: GenerateFn = generate_one,
    stream_generate_fn: StreamGenerateFn = stream_generate_one,
):
    """Create a FastAPI app around one loaded causal LM."""

    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
    except ImportError as exc:
        raise RuntimeError("FastAPI server dependencies are not installed") from exc

    if loaded is None:
        if model_path is None:
            raise ValueError("model_path is required when loaded is not provided")
        loaded = load_model_and_tokenizer(
            model_path,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )

    model_name = served_model_name or model_path or "nanollmserve-model"
    config = ServerConfig(
        served_model_name=model_name,
        loaded=loaded,
        generate_fn=generate_fn,
        stream_generate_fn=stream_generate_fn,
        response_store={},
    )

    app = FastAPI(title="nanoLLMServe", version="0.2.0")
    app.state.nanollmserve = config

    @app.get("/v1/models", response_model=ListModelsResponse)
    def list_models() -> ListModelsResponse:
        return ListModelsResponse(data=[ModelCard(id=config.served_model_name)])

    @app.post("/v1/completions")
    def completions(request: CompletionRequest):
        model_error = _validate_requested_model(config, request.model)
        if model_error is not None:
            return model_error
        if request.stream:
            return StreamingResponse(
                _completion_stream(config, request),
                media_type="text/event-stream",
            )

        result = config.generate_fn(
            config.loaded.model,
            config.loaded.tokenizer,
            request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
        )
        return build_completion_response(request, result)

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatCompletionRequest):
        model_error = _validate_requested_model(config, request.model)
        if model_error is not None:
            return model_error
        prompt = chat_messages_to_prompt(request.messages)
        if request.stream:
            return StreamingResponse(
                _chat_completion_stream(config, request, prompt),
                media_type="text/event-stream",
            )

        result = config.generate_fn(
            config.loaded.model,
            config.loaded.tokenizer,
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
        )
        return build_chat_completion_response(request, result)

    @app.post("/v1/responses")
    def responses(request: ResponsesRequest):
        model_error = _validate_requested_model(config, request.model)
        if model_error is not None:
            return model_error
        unsupported_error = _validate_responses_request(config, request)
        if unsupported_error is not None:
            return unsupported_error

        prompt = responses_input_to_prompt(request)
        if request.stream:
            return StreamingResponse(
                _responses_stream(config, request, prompt),
                media_type="text/event-stream",
            )

        result = config.generate_fn(
            config.loaded.model,
            config.loaded.tokenizer,
            prompt,
            max_new_tokens=request.max_output_tokens,
            temperature=request.temperature,
            seed=request.seed,
        )
        response = build_responses_response(request, result)
        _store_response(config, response)
        return response

    @app.get("/v1/responses/{response_id}")
    def retrieve_response(response_id: str):
        stored = (config.response_store or {}).get(response_id)
        if stored is None:
            return _response_not_found(response_id)
        return stored

    return app


def _validate_requested_model(config: ServerConfig, requested_model: str):
    try:
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise RuntimeError("FastAPI server dependencies are not installed") from exc

    if requested_model == config.served_model_name:
        return None

    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"The model '{requested_model}' does not exist.",
                "type": "invalid_request_error",
                "param": "model",
                "code": "model_not_found",
            }
        },
    )


def _validate_responses_request(config: ServerConfig, request: ResponsesRequest):
    try:
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise RuntimeError("FastAPI server dependencies are not installed") from exc

    if request.tools:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "tools are not supported by nanoLLMServe Responses API yet.",
                    "type": "invalid_request_error",
                    "param": "tools",
                    "code": "unsupported_feature",
                }
            },
        )
    if request.previous_response_id and request.previous_response_id not in (config.response_store or {}):
        return _response_not_found(request.previous_response_id)
    return None


def _response_not_found(response_id: str):
    try:
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise RuntimeError("FastAPI server dependencies are not installed") from exc

    return JSONResponse(
        status_code=404,
        content={
            "error": {
                "message": f"The response '{response_id}' does not exist.",
                "type": "invalid_request_error",
                "param": "response_id",
                "code": "response_not_found",
            }
        },
    )


def _store_response(config: ServerConfig, response: ResponsesResponse) -> None:
    if response.store and config.response_store is not None:
        config.response_store[response.id] = response


def _completion_stream(config: ServerConfig, request: CompletionRequest):
    response_id = new_response_id("cmpl")
    created = now_timestamp()
    for step in config.stream_generate_fn(
        config.loaded.model,
        config.loaded.tokenizer,
        request.prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        seed=request.seed,
    ):
        payload = {
            "id": response_id,
            "object": "text_completion",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "text": step.text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": finish_reason(
                        step.generated_tokens,
                        request.max_tokens,
                        step.finished,
                    )
                    if step.finished or step.generated_tokens >= request.max_tokens
                    else None,
                }
            ],
        }
        yield _sse_event(payload)
    yield "data: [DONE]\n\n"


def _chat_completion_stream(config: ServerConfig, request: ChatCompletionRequest, prompt: str):
    response_id = new_response_id("chatcmpl")
    created = now_timestamp()
    for step in config.stream_generate_fn(
        config.loaded.model,
        config.loaded.tokenizer,
        prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        seed=request.seed,
    ):
        payload = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": step.text},
                    "finish_reason": finish_reason(
                        step.generated_tokens,
                        request.max_tokens,
                        step.finished,
                    )
                    if step.finished or step.generated_tokens >= request.max_tokens
                    else None,
                }
            ],
        }
        yield _sse_event(payload)
    yield "data: [DONE]\n\n"


def _responses_stream(config: ServerConfig, request: ResponsesRequest, prompt: str):
    response_id = new_response_id("resp")
    created = now_timestamp()
    skeleton = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "in_progress",
        "model": request.model,
        "output": [],
        "output_text": "",
    }
    yield _sse_event(
        {"type": "response.created", "response": skeleton},
        event="response.created",
    )
    yield _sse_event(
        {"type": "response.in_progress", "response": skeleton},
        event="response.in_progress",
    )

    steps = []
    item_id = new_response_id("msg")
    yielded_content_part = False
    for step in config.stream_generate_fn(
        config.loaded.model,
        config.loaded.tokenizer,
        prompt,
        max_new_tokens=request.max_output_tokens,
        temperature=request.temperature,
        seed=request.seed,
    ):
        steps.append(step)
        if not yielded_content_part:
            yielded_content_part = True
            yield _sse_event(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                },
                event="response.output_item.added",
            )
            yield _sse_event(
                {
                    "type": "response.content_part.added",
                    "output_index": 0,
                    "content_index": 0,
                    "item_id": item_id,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
                event="response.content_part.added",
            )
        yield _sse_event(
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "content_index": 0,
                "item_id": item_id,
                "delta": step.text,
            },
            event="response.output_text.delta",
        )

    final_step = steps[-1]
    result = SimpleNamespace(
        text=final_step.generated_text,
        prompt_tokens=final_step.prompt_tokens,
        generated_tokens=final_step.generated_tokens,
        generated_token_ids=final_step.generated_token_ids,
        elapsed_seconds=final_step.elapsed_seconds,
        finished=final_step.finished,
    )
    response = build_responses_response(
        request,
        result,
        response_id=response_id,
        created_at=created,
    )
    _store_response(config, response)
    yield _sse_event(
        {
            "type": "response.output_text.done",
            "output_index": 0,
            "content_index": 0,
            "item_id": item_id,
            "text": final_step.generated_text,
        },
        event="response.output_text.done",
    )
    yield _sse_event(
        {"type": "response.completed", "response": response.model_dump()},
        event="response.completed",
    )


def _sse_event(payload: dict, *, event: str | None = None) -> str:
    data = "data: " + json.dumps(payload, separators=(",", ":")) + "\n\n"
    if event is None:
        return data
    return f"event: {event}\n{data}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve one causal LM with OpenAI-compatible endpoints.")
    parser.add_argument("--model", required=True, help="Hugging Face repo id or local model directory.")
    parser.add_argument("--served-model-name", default=None, help="Model id returned by /v1/models.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for uvicorn.")
    parser.add_argument("--port", type=int, default=8000, help="Port for uvicorn.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps.")
    parser.add_argument("--dtype", default="auto", help="auto, float32, float16, or bfloat16.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load files that already exist on disk.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run nanollm-serve") from exc

    app = create_app(
        model_path=args.model,
        served_model_name=args.served_model_name,
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

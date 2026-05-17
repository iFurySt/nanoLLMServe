"""OpenAI-compatible request and response models for the teaching server."""

from __future__ import annotations

from typing import Literal
from time import time
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "nanoLLMServe"


class ListModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class CompletionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = 0.0
    stream: bool = False
    seed: int | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = 0.0
    stream: bool = False
    seed: int | None = None


class ResponseInputMessage(BaseModel):
    role: str
    content: str


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    input: str | list[ResponseInputMessage]
    instructions: str | None = None
    max_output_tokens: int = Field(default=16, ge=1)
    temperature: float = 0.0
    stream: bool = False
    seed: int | None = None
    store: bool = True
    previous_response_id: str | None = None
    tools: list[dict] = Field(default_factory=list)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: None = None
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class ChatCompletionResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list = Field(default_factory=list)


class ResponseOutputMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    status: Literal["completed", "in_progress", "incomplete"] = "completed"
    role: Literal["assistant"] = "assistant"
    content: list[ResponseOutputText]


class ResponsesResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: Literal["completed", "in_progress", "incomplete"]
    model: str
    output: list[ResponseOutputMessage]
    output_text: str
    usage: ResponseUsage | None = None
    instructions: str | None = None
    previous_response_id: str | None = None
    store: bool = True


def now_timestamp() -> int:
    return int(time())


def new_response_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def finish_reason(generated_tokens: int, max_tokens: int, finished: bool = False) -> str:
    if finished:
        return "stop"
    if generated_tokens >= max_tokens:
        return "length"
    return "stop"


def usage_from_result(result) -> Usage:
    return Usage(
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.generated_tokens,
        total_tokens=result.prompt_tokens + result.generated_tokens,
    )


def chat_messages_to_prompt(messages: list[ChatMessage]) -> str:
    lines = [f"{message.role}: {message.content}" for message in messages]
    lines.append("assistant:")
    return "\n".join(lines)


def responses_input_to_prompt(request: ResponsesRequest) -> str:
    if isinstance(request.input, str):
        parts = [request.input]
    else:
        parts = [f"{message.role}: {message.content}" for message in request.input]
        parts.append("assistant:")

    if request.instructions:
        return request.instructions + "\n" + "\n".join(parts)
    return "\n".join(parts)


def build_completion_response(request: CompletionRequest, result) -> CompletionResponse:
    return CompletionResponse(
        id=new_response_id("cmpl"),
        created=now_timestamp(),
        model=request.model,
        choices=[
            CompletionChoice(
                text=result.text,
                finish_reason=finish_reason(
                    result.generated_tokens,
                    request.max_tokens,
                    getattr(result, "finished", False),
                ),
            )
        ],
        usage=usage_from_result(result),
    )


def build_chat_completion_response(request: ChatCompletionRequest, result) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=new_response_id("chatcmpl"),
        created=now_timestamp(),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionResponseMessage(content=result.text),
                finish_reason=finish_reason(
                    result.generated_tokens,
                    request.max_tokens,
                    getattr(result, "finished", False),
                ),
            )
        ],
        usage=usage_from_result(result),
    )


def build_responses_response(
    request: ResponsesRequest,
    result,
    *,
    response_id: str | None = None,
    created_at: int | None = None,
) -> ResponsesResponse:
    response_id = response_id or new_response_id("resp")
    created_at = created_at or now_timestamp()
    status = "completed" if getattr(result, "finished", False) else "incomplete"
    text = result.text
    return ResponsesResponse(
        id=response_id,
        created_at=created_at,
        status=status,
        model=request.model,
        output=[
            ResponseOutputMessage(
                id=new_response_id("msg"),
                status=status,
                content=[ResponseOutputText(text=text)],
            )
        ],
        output_text=text,
        usage=ResponseUsage(
            input_tokens=result.prompt_tokens,
            output_tokens=result.generated_tokens,
            total_tokens=result.prompt_tokens + result.generated_tokens,
        ),
        instructions=request.instructions,
        previous_response_id=request.previous_response_id,
        store=request.store,
    )

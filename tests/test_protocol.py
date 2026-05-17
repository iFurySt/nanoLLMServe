from types import SimpleNamespace

from nanollmserve.api.protocol import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    ResponseInputMessage,
    ResponsesRequest,
    build_chat_completion_response,
    build_completion_response,
    build_responses_response,
    chat_messages_to_prompt,
    responses_input_to_prompt,
)


def test_completion_response_matches_openai_shape():
    request = CompletionRequest(model="toy-model", prompt="hello", max_tokens=4)
    result = SimpleNamespace(
        text=" world",
        prompt_tokens=2,
        generated_tokens=2,
        generated_token_ids=[10, 11],
        finished=True,
    )

    response = build_completion_response(request, result)

    assert response.object == "text_completion"
    assert response.model == "toy-model"
    assert response.choices[0].text == " world"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens == 4


def test_chat_messages_become_a_single_prompt():
    prompt = chat_messages_to_prompt(
        [
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="hello"),
        ]
    )

    assert prompt == "system: be brief\nuser: hello\nassistant:"


def test_chat_completion_response_matches_openai_shape():
    request = ChatCompletionRequest(
        model="toy-model",
        messages=[ChatMessage(role="user", content="hello")],
        max_tokens=1,
    )
    result = SimpleNamespace(
        text="world",
        prompt_tokens=3,
        generated_tokens=1,
        generated_token_ids=[10],
        finished=False,
    )

    response = build_chat_completion_response(request, result)

    assert response.object == "chat.completion"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].message.content == "world"
    assert response.choices[0].finish_reason == "length"
    assert response.usage.completion_tokens == 1


def test_responses_input_string_uses_instructions_as_prefix():
    request = ResponsesRequest(
        model="toy-model",
        instructions="be brief",
        input="hello",
    )

    assert responses_input_to_prompt(request) == "be brief\nhello"


def test_responses_input_messages_become_a_single_prompt():
    request = ResponsesRequest(
        model="toy-model",
        input=[
            ResponseInputMessage(role="system", content="be brief"),
            ResponseInputMessage(role="user", content="hello"),
        ],
    )

    assert responses_input_to_prompt(request) == "system: be brief\nuser: hello\nassistant:"


def test_responses_response_matches_minimal_openai_shape():
    request = ResponsesRequest(model="toy-model", input="hello", max_output_tokens=4)
    result = SimpleNamespace(
        text="world",
        prompt_tokens=1,
        generated_tokens=2,
        generated_token_ids=[10, 11],
        finished=True,
    )

    response = build_responses_response(request, result)

    assert response.object == "response"
    assert response.id.startswith("resp-")
    assert response.status == "completed"
    assert response.output_text == "world"
    assert response.output[0].type == "message"
    assert response.output[0].content[0].type == "output_text"
    assert response.output[0].content[0].text == "world"
    assert response.usage.total_tokens == 3

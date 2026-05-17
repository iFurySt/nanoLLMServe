from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from nanollmserve.api.openai_server import build_parser, create_app
from nanollmserve.model.hf_runner import LoadedModel


def _loaded_model():
    return LoadedModel(model=object(), tokenizer=object(), device="cpu", dtype="float32")


def test_models_endpoint_returns_served_model_name():
    app = create_app(loaded=_loaded_model(), served_model_name="toy-model")
    client = TestClient(app)

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "toy-model"


def test_completions_endpoint_calls_generation_function():
    calls = []

    def fake_generate(model, tokenizer, prompt, *, max_new_tokens, temperature, seed):
        calls.append((model, tokenizer, prompt, max_new_tokens, temperature, seed))
        return SimpleNamespace(
            text=" world",
            prompt_tokens=1,
            generated_tokens=2,
            generated_token_ids=[10, 11],
            elapsed_seconds=0.01,
            finished=True,
        )

    loaded = _loaded_model()
    app = create_app(loaded=loaded, served_model_name="toy-model", generate_fn=fake_generate)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "toy-model",
            "prompt": "hello",
            "max_tokens": 4,
            "temperature": 0.5,
            "seed": 7,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["text"] == " world"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 3
    assert calls == [(loaded.model, loaded.tokenizer, "hello", 4, 0.5, 7)]


def test_completions_endpoint_rejects_unknown_model():
    def fake_generate(*args, **kwargs):
        raise AssertionError("generation should not run for an unknown model")

    app = create_app(loaded=_loaded_model(), served_model_name="toy-model", generate_fn=fake_generate)
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={
            "model": "other-model",
            "prompt": "hello",
            "max_tokens": 4,
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_chat_completions_endpoint_wraps_messages_as_prompt():
    calls = []

    def fake_generate(model, tokenizer, prompt, *, max_new_tokens, temperature, seed):
        calls.append(prompt)
        return SimpleNamespace(
            text="world",
            prompt_tokens=4,
            generated_tokens=1,
            generated_token_ids=[10],
            elapsed_seconds=0.01,
            finished=True,
        )

    app = create_app(loaded=_loaded_model(), served_model_name="toy-model", generate_fn=fake_generate)
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "toy-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "world"}
    assert calls == ["user: hello\nassistant:"]


def test_completions_stream_emits_sse_chunks():
    def fake_stream(model, tokenizer, prompt, *, max_new_tokens, temperature, seed):
        assert prompt == "hello"
        return iter(
            [
                SimpleNamespace(text="A", generated_tokens=1, finished=False),
                SimpleNamespace(text="B", generated_tokens=2, finished=False),
            ]
        )

    app = create_app(
        loaded=_loaded_model(),
        served_model_name="toy-model",
        stream_generate_fn=fake_stream,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/completions",
        json={"model": "toy-model", "prompt": "hello", "max_tokens": 2, "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"object":"text_completion"' in response.text
    assert "text_completion.chunk" not in response.text
    assert '"text":"A"' in response.text
    assert '"text":"B"' in response.text
    assert '"finish_reason":"length"' in response.text
    assert "data: [DONE]" in response.text


def test_chat_completions_stream_uses_chat_chunk_object():
    def fake_stream(model, tokenizer, prompt, *, max_new_tokens, temperature, seed):
        return iter([SimpleNamespace(text="A", generated_tokens=1, finished=True)])

    app = create_app(
        loaded=_loaded_model(),
        served_model_name="toy-model",
        stream_generate_fn=fake_stream,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "toy-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 2,
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert '"object":"chat.completion.chunk"' in response.text


def test_responses_endpoint_returns_output_text_and_can_be_retrieved():
    calls = []

    def fake_generate(model, tokenizer, prompt, *, max_new_tokens, temperature, seed):
        calls.append((prompt, max_new_tokens, temperature, seed))
        return SimpleNamespace(
            text="world",
            prompt_tokens=2,
            generated_tokens=1,
            generated_token_ids=[10],
            elapsed_seconds=0.01,
            finished=True,
        )

    app = create_app(loaded=_loaded_model(), served_model_name="toy-model", generate_fn=fake_generate)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "toy-model",
            "instructions": "be brief",
            "input": "hello",
            "max_output_tokens": 3,
            "temperature": 0.2,
            "seed": 9,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["output_text"] == "world"
    assert body["output"][0]["content"][0]["type"] == "output_text"
    assert body["usage"] == {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
    assert calls == [("be brief\nhello", 3, 0.2, 9)]

    retrieved = client.get(f"/v1/responses/{body['id']}")
    assert retrieved.status_code == 200
    assert retrieved.json()["id"] == body["id"]


def test_responses_endpoint_supports_message_list_input():
    calls = []

    def fake_generate(model, tokenizer, prompt, *, max_new_tokens, temperature, seed):
        calls.append(prompt)
        return SimpleNamespace(
            text="world",
            prompt_tokens=4,
            generated_tokens=1,
            generated_token_ids=[10],
            elapsed_seconds=0.01,
            finished=True,
        )

    app = create_app(loaded=_loaded_model(), served_model_name="toy-model", generate_fn=fake_generate)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "toy-model",
            "input": [{"role": "user", "content": "hello"}],
            "max_output_tokens": 2,
        },
    )

    assert response.status_code == 200
    assert response.json()["output_text"] == "world"
    assert calls == ["user: hello\nassistant:"]


def test_responses_endpoint_rejects_unknown_model():
    def fake_generate(*args, **kwargs):
        raise AssertionError("generation should not run for an unknown model")

    app = create_app(loaded=_loaded_model(), served_model_name="toy-model", generate_fn=fake_generate)
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={"model": "other-model", "input": "hello", "max_output_tokens": 2},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_responses_endpoint_rejects_unknown_previous_response_id():
    app = create_app(loaded=_loaded_model(), served_model_name="toy-model")
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "toy-model",
            "input": "hello",
            "previous_response_id": "resp-missing",
        },
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "response_not_found"


def test_retrieve_response_rejects_unknown_id():
    app = create_app(loaded=_loaded_model(), served_model_name="toy-model")
    client = TestClient(app)

    response = client.get("/v1/responses/resp-missing")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "response_not_found"


def test_responses_endpoint_rejects_tools_for_now():
    app = create_app(loaded=_loaded_model(), served_model_name="toy-model")
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={
            "model": "toy-model",
            "input": "hello",
            "tools": [{"type": "web_search_preview"}],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_feature"


def test_responses_stream_emits_response_events():
    def fake_stream(model, tokenizer, prompt, *, max_new_tokens, temperature, seed):
        assert prompt == "hello"
        return iter(
            [
                SimpleNamespace(
                    text="A",
                    generated_text="A",
                    generated_tokens=1,
                    generated_token_ids=[10],
                    prompt_tokens=1,
                    elapsed_seconds=0.01,
                    finished=False,
                ),
                SimpleNamespace(
                    text="B",
                    generated_text="AB",
                    generated_tokens=2,
                    generated_token_ids=[10, 11],
                    prompt_tokens=1,
                    elapsed_seconds=0.02,
                    finished=True,
                ),
            ]
        )

    app = create_app(
        loaded=_loaded_model(),
        served_model_name="toy-model",
        stream_generate_fn=fake_stream,
    )
    client = TestClient(app)

    response = client.post(
        "/v1/responses",
        json={"model": "toy-model", "input": "hello", "max_output_tokens": 2, "stream": True},
    )

    assert response.status_code == 200
    assert "event: response.created" in response.text
    assert "event: response.output_text.delta" in response.text
    assert '"delta":"A"' in response.text
    assert '"delta":"B"' in response.text
    assert "event: response.completed" in response.text
    assert '"output_text":"AB"' in response.text


def test_serve_parser_accepts_minimal_args():
    args = build_parser().parse_args(["--model", "/models/Qwen3-1.7B"])

    assert args.model == "/models/Qwen3-1.7B"
    assert args.host == "127.0.0.1"
    assert args.port == 8000

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nanollmserve.engine.engine import generate_one, stream_generate_one


class FakeTokenizer:
    eos_token_id = 9

    def __init__(self, *, include_attention_mask=True):
        self.include_attention_mask = include_attention_mask

    def __call__(self, prompt, return_tensors):
        assert prompt == "hello"
        assert return_tensors == "pt"
        batch = {
            "input_ids": torch.tensor([[5, 1]], dtype=torch.long),
        }
        if self.include_attention_mask:
            batch["attention_mask"] = torch.tensor([[1, 1]], dtype=torch.long)
        return batch

    def decode(self, token_ids, skip_special_tokens=True):
        values = [item for item in token_ids if not skip_special_tokens or item != self.eos_token_id]
        return "".join({2: "A", 3: "B"}.get(item, "") for item in values)


class FakeModel:
    def __init__(self):
        self.attention_masks = []
        self.input_ids = []
        self.past_key_values = []
        self.use_cache_values = []
        self.call_count = 0

    def parameters(self):
        return iter(())

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, past_key_values=None, use_cache=False):
        self.call_count += 1
        self.input_ids.append(input_ids.clone())
        self.attention_masks.append(attention_mask.clone())
        self.past_key_values.append(past_key_values)
        self.use_cache_values.append(use_cache)
        vocab_size = 10
        logits = torch.full((*input_ids.shape, vocab_size), -100.0)
        next_id = {1: 2, 2: 3}.get(self.call_count, 9)
        logits[:, -1, next_id] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=(f"kv-{self.call_count}",))


def test_generate_one_runs_until_eos():
    result = generate_one(
        FakeModel(),
        FakeTokenizer(),
        "hello",
        max_new_tokens=8,
        temperature=0.0,
    )

    assert result.text == "AB"
    assert result.generated_token_ids == [2, 3, 9]
    assert result.prompt_tokens == 2
    assert result.generated_tokens == 3
    assert result.elapsed_seconds >= 0
    assert result.finished is True


def test_generate_one_rejects_empty_prompt():
    with pytest.raises(ValueError, match="prompt"):
        generate_one(FakeModel(), FakeTokenizer(), "", max_new_tokens=1)


def test_generate_one_rejects_non_positive_max_new_tokens():
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate_one(FakeModel(), FakeTokenizer(), "hello", max_new_tokens=0)


def test_generate_one_stops_at_max_new_tokens_without_eos():
    class NoEosTokenizer(FakeTokenizer):
        eos_token_id = None

    result = generate_one(
        FakeModel(),
        NoEosTokenizer(),
        "hello",
        max_new_tokens=2,
        temperature=0.0,
    )

    assert result.generated_token_ids == [2, 3]
    assert result.generated_tokens == 2
    assert result.text == "AB"
    assert result.finished is False


def test_generate_one_creates_attention_mask_when_missing():
    model = FakeModel()

    generate_one(
        model,
        FakeTokenizer(include_attention_mask=False),
        "hello",
        max_new_tokens=2,
        temperature=0.0,
    )

    assert [mask.tolist() for mask in model.attention_masks] == [
        [[1, 1]],
        [[1, 1, 1]],
    ]


def test_generate_one_reuses_kv_cache_after_prefill():
    model = FakeModel()

    result = generate_one(
        model,
        FakeTokenizer(),
        "hello",
        max_new_tokens=3,
        temperature=0.0,
    )

    assert [ids.tolist() for ids in model.input_ids] == [
        [[5, 1]],
        [[2]],
        [[3]],
    ]
    assert model.past_key_values == [None, ("kv-1",), ("kv-2",)]
    assert model.use_cache_values == [True, True, True]
    assert result.ttft_seconds >= 0
    assert result.decode_seconds >= 0
    assert result.tpot_seconds >= 0


def test_stream_generate_one_yields_incremental_tokens():
    steps = list(
        stream_generate_one(
            FakeModel(),
            FakeTokenizer(),
            "hello",
            max_new_tokens=3,
            temperature=0.0,
        )
    )

    assert [step.text for step in steps] == ["A", "B", ""]
    assert [step.generated_text for step in steps] == ["A", "AB", "AB"]
    assert [step.generated_token_ids for step in steps] == [[2], [2, 3], [2, 3, 9]]
    assert steps[-1].finished is True

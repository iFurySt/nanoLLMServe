from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nanollmserve.engine.engine import generate_batch, generate_one, stream_generate_one


class FakeTokenizer:
    eos_token_id = 9

    def __init__(self, *, include_attention_mask=True):
        self.include_attention_mask = include_attention_mask

    def __call__(self, prompt, return_tensors):
        assert isinstance(prompt, str)
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


class FakeBatchTokenizer:
    eos_token_id = 9

    def __call__(self, prompts, return_tensors, padding=False, **kwargs):
        assert isinstance(prompts, list)
        assert return_tensors == "pt"
        assert padding is True
        assert len(prompts) == 2
        assert prompts == ["hello", "world"]
        return {
            "input_ids": torch.tensor(
                [
                    [5, 1, 0],
                    [8, 2, 0],
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                ],
                dtype=torch.long,
            ),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        output = {
            2: "A",
            3: "B",
            4: "C",
            5: "D",
            7: "E",
            8: "F",
        }
        filtered = [value for value in token_ids if not skip_special_tokens or value != self.eos_token_id]
        return "".join(output.get(item, "") for item in filtered)


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
        logits = torch.full((*input_ids.shape, vocab_size), -100.0, device=input_ids.device)
        next_id = {1: 2, 2: 3}.get(self.call_count, 9)
        logits[:, -1, next_id] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=(f"kv-{self.call_count}",))


class FakeBatchModel:
    def __init__(self, next_token_schedule):
        self.next_token_schedule = next_token_schedule
        self.call_count = 0
        self.call_inputs = []
        self.call_attention_masks = []
        self.call_use_cache = []
        self.call_past_key_values = []

    def parameters(self):
        return iter(())

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, past_key_values=None, use_cache=False):
        self.call_count += 1
        self.call_inputs.append(input_ids.clone())
        self.call_attention_masks.append(attention_mask.clone())
        self.call_use_cache.append(use_cache)
        self.call_past_key_values.append(past_key_values)

        vocab_size = 10
        logits = torch.full((*input_ids.shape, vocab_size), -100.0, device=input_ids.device)
        schedule_index = min(self.call_count - 1, len(self.next_token_schedule) - 1)
        next_ids = self.next_token_schedule[schedule_index]
        for idx, token_id in enumerate(next_ids):
            last_index = int(attention_mask[idx].sum().item()) - 1 if past_key_values is None else -1
            logits[idx, last_index, int(token_id)] = 100.0
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


def test_generate_batch_prefill_and_decode_all_requests_together():
    model = FakeBatchModel(
        next_token_schedule=[
            [2, 4],
            [3, 5],
            [9, 6],
        ]
    )

    results = generate_batch(
        model,
        FakeBatchTokenizer(),
        ["hello", "world"],
        max_new_tokens=3,
        temperature=0.0,
    )

    assert [result.generated_token_ids for result in results] == [[2, 3, 9], [4, 5, 6]]
    assert [result.generated_tokens for result in results] == [3, 3]
    assert [result.finished for result in results] == [True, False]
    assert model.call_inputs[0].shape == (2, 3)
    assert model.call_inputs[1].shape == (2, 1)
    assert model.call_inputs[2].shape == (2, 1)
    assert model.call_attention_masks[1].shape == (2, 4)
    assert model.call_attention_masks[2].shape == (2, 5)
    assert model.call_use_cache == [True, True, True]


def test_generate_batch_keeps_finished_request_from_growing():
    model = FakeBatchModel(
        next_token_schedule=[
            [2, 4],
            [9, 5],
            [9, 6],
            [9, 7],
        ]
    )

    results = generate_batch(
        model,
        FakeBatchTokenizer(),
        ["hello", "world"],
        max_new_tokens=4,
        temperature=0.0,
    )

    assert results[0].generated_token_ids == [2, 9]
    assert results[0].text == "A"
    assert results[0].finished is True
    assert results[1].generated_token_ids == [4, 5, 6, 7]


def test_generate_batch_rejects_empty_prompts():
    with pytest.raises(ValueError, match="prompts"):
        generate_batch(
            FakeBatchModel([[2]]),
            FakeBatchTokenizer(),
            [],
            max_new_tokens=1,
        )


def test_generate_batch_rejects_empty_prompt_entries():
    with pytest.raises(ValueError, match="non-empty"):
        generate_batch(
            FakeBatchModel([[2, 2]]),
            FakeBatchTokenizer(),
            ["", "world"],
            max_new_tokens=1,
        )

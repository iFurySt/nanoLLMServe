from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nanollmserve.cache.block_manager import KVBlockManager
from nanollmserve.engine.engine import (
    generate_batch,
    generate_continuous_batch,
    generate_one,
    stream_generate_one,
)
from nanollmserve.engine.scheduler import ContinuousBatchRequest


class TrackingBlockManager(KVBlockManager):
    def __post_init__(self):
        super().__post_init__()
        self.events = []

    def allocate(self, request_id, token_count):
        table = super().allocate(request_id, token_count)
        self.events.append(("allocate", request_id, self.usage().used_blocks, self.usage().allocated_tokens))
        return table

    def append_tokens(self, request_id, token_count=1):
        table = super().append_tokens(request_id, token_count)
        self.events.append(("append", request_id, self.usage().used_blocks, self.usage().allocated_tokens))
        return table

    def release(self, request_id):
        released = super().release(request_id)
        self.events.append(("release", request_id, self.usage().used_blocks, self.usage().allocated_tokens))
        return released


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


class FakeContinuousTokenizer:
    eos_token_id = 9
    pad_token_id = 0

    prompt_tokens = {
        "short": [1, 1],
        "late": [4, 4],
    }

    def __call__(self, prompt, return_tensors):
        assert isinstance(prompt, str)
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([self.prompt_tokens[prompt]], dtype=torch.long),
            "attention_mask": torch.tensor([[1] * len(self.prompt_tokens[prompt])], dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=True):
        output = {
            2: "A",
            3: "B",
            5: "C",
            6: "D",
        }
        filtered = [value for value in token_ids if not skip_special_tokens or value != self.eos_token_id]
        return "".join(output.get(item, "") for item in filtered)


class FakeContinuousModel:
    prompt_lengths = {
        1: 2,
        4: 2,
    }
    token_schedule = {
        1: [2, 9],
        4: [5, 6, 9],
    }

    def __init__(self):
        self.call_inputs = []
        self.call_attention_masks = []
        self.call_use_cache = []

    def parameters(self):
        return iter(())

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask, use_cache=False, **kwargs):
        self.call_inputs.append(input_ids.clone())
        self.call_attention_masks.append(attention_mask.clone())
        self.call_use_cache.append(use_cache)

        vocab_size = 10
        logits = torch.full((*input_ids.shape, vocab_size), -100.0, device=input_ids.device)
        for row in range(input_ids.shape[0]):
            prompt_id = int(input_ids[row, 0].item())
            valid_length = int(attention_mask[row].sum().item())
            generated_count = valid_length - self.prompt_lengths[prompt_id]
            next_token = self.token_schedule[prompt_id][generated_count]
            logits[row, valid_length - 1, next_token] = 100.0
        return SimpleNamespace(logits=logits)


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


def test_generate_one_allocates_and_releases_kv_blocks():
    manager = TrackingBlockManager(total_blocks=4, block_size=2)

    generate_one(
        FakeModel(),
        FakeTokenizer(),
        "hello",
        max_new_tokens=3,
        temperature=0.0,
        kv_block_manager=manager,
        request_id="req-a",
    )

    assert [event[:2] for event in manager.events] == [
        ("allocate", "req-a"),
        ("append", "req-a"),
        ("append", "req-a"),
        ("append", "req-a"),
        ("release", "req-a"),
    ]
    assert manager.events[0] == ("allocate", "req-a", 1, 2)
    assert manager.events[-1] == ("release", "req-a", 0, 0)


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


def test_generate_batch_allocates_and_releases_kv_blocks_per_request():
    manager = TrackingBlockManager(total_blocks=8, block_size=2)

    generate_batch(
        FakeBatchModel(
            next_token_schedule=[
                [2, 4],
                [3, 5],
                [9, 6],
            ]
        ),
        FakeBatchTokenizer(),
        ["hello", "world"],
        max_new_tokens=3,
        temperature=0.0,
        kv_block_manager=manager,
        request_ids=["req-a", "req-b"],
    )

    assert ("allocate", "req-a", 1, 2) in manager.events
    assert ("allocate", "req-b", 2, 4) in manager.events
    assert manager.events[-2:] == [
        ("release", "req-b", 3, 5),
        ("release", "req-a", 0, 0),
    ]


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


def test_generate_continuous_batch_admits_new_request_while_decoding():
    model = FakeContinuousModel()

    run = generate_continuous_batch(
        model,
        FakeContinuousTokenizer(),
        [
            ContinuousBatchRequest("short-0", "short", max_new_tokens=4, arrival_step=0),
            ContinuousBatchRequest("late-1", "late", max_new_tokens=4, arrival_step=1),
        ],
        temperature=0.0,
    )

    assert [result.request_id for result in run.results] == ["short-0", "late-1"]
    assert [result.result.generated_token_ids for result in run.results] == [[2, 9], [5, 6, 9]]
    assert [result.result.text for result in run.results] == ["A", "CD"]
    assert run.active_batch_sizes == [1, 2, 1, 1]
    assert run.scheduler_steps[1].admitted_request_ids == ["late-1"]
    assert run.scheduler_steps[1].completed_request_ids == ["short-0"]
    assert [tuple(input_ids.shape) for input_ids in model.call_inputs] == [
        (1, 2),
        (2, 3),
        (1, 3),
        (1, 4),
    ]
    assert model.call_use_cache == [False, False, False, False]


def test_generate_continuous_batch_releases_completed_request_blocks_mid_run():
    manager = TrackingBlockManager(total_blocks=8, block_size=2)

    generate_continuous_batch(
        FakeContinuousModel(),
        FakeContinuousTokenizer(),
        [
            ContinuousBatchRequest("short-0", "short", max_new_tokens=4, arrival_step=0),
            ContinuousBatchRequest("late-1", "late", max_new_tokens=4, arrival_step=1),
        ],
        temperature=0.0,
        kv_block_manager=manager,
    )

    assert ("release", "short-0", 2, 3) in manager.events
    assert manager.events[-1] == ("release", "late-1", 0, 0)


def test_generate_continuous_batch_respects_max_batch_size():
    run = generate_continuous_batch(
        FakeContinuousModel(),
        FakeContinuousTokenizer(),
        [
            ContinuousBatchRequest("short-0", "short", max_new_tokens=2, arrival_step=0),
            ContinuousBatchRequest("late-1", "late", max_new_tokens=3, arrival_step=0),
        ],
        max_batch_size=1,
        temperature=0.0,
    )

    assert run.scheduler_steps[0].admitted_request_ids == ["short-0"]
    assert run.scheduler_steps[2].admitted_request_ids == ["late-1"]
    assert [result.admitted_step for result in run.results] == [0, 2]


def test_generate_continuous_batch_rejects_empty_requests():
    with pytest.raises(ValueError, match="requests"):
        generate_continuous_batch(
            FakeContinuousModel(),
            FakeContinuousTokenizer(),
            [],
        )

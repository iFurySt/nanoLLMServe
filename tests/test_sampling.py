import pytest

torch = pytest.importorskip("torch")

from nanollmserve.sampling.sampler import sample_next_token


def test_greedy_sampling_picks_highest_logit():
    logits = torch.tensor([[0.1, 4.0, 1.5]])

    next_token = sample_next_token(logits, temperature=0.0)

    assert next_token.tolist() == [[1]]


def test_temperature_sampling_accepts_seeded_generator():
    logits = torch.tensor([[0.1, 4.0, 1.5]])
    generator_a = torch.Generator(device="cpu").manual_seed(7)
    generator_b = torch.Generator(device="cpu").manual_seed(7)

    first = sample_next_token(logits, temperature=0.8, generator=generator_a)
    second = sample_next_token(logits, temperature=0.8, generator=generator_b)

    assert first.tolist() == second.tolist()


def test_sampling_rejects_non_batch_logits():
    with pytest.raises(ValueError, match="expected logits"):
        sample_next_token(torch.tensor([0.1, 4.0, 1.5]), temperature=0.0)


def test_sampling_rejects_non_finite_temperature():
    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(torch.tensor([[0.1, 4.0, 1.5]]), temperature=float("nan"))

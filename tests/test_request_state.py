from nanollmserve.engine.request import GenerationRequestState


def test_generation_request_state_tracks_token_counts_and_tpot():
    state = GenerationRequestState(prompt="hello", prompt_token_ids=[5, 1])
    state.generated_token_ids.extend([2, 3, 4])
    state.decode_seconds = 0.4

    assert state.prompt_tokens == 2
    assert state.generated_tokens == 3
    assert state.tpot_seconds == 0.2


def test_generation_request_state_tpot_is_zero_until_decode_tokens_exist():
    state = GenerationRequestState(prompt="hello", prompt_token_ids=[5, 1])
    state.generated_token_ids.append(2)
    state.decode_seconds = 0.4

    assert state.tpot_seconds == 0.0

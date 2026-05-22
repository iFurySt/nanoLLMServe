import pytest

torch = pytest.importorskip("torch")

from nanollmserve.cache.prefix_cache import (
    PrefixCache,
    clone_past_key_values,
    make_prefix_key,
    slice_past_key_values,
)


def _past_key_values(token_count: int):
    key = torch.arange(token_count * 2, dtype=torch.float32).reshape(1, 1, token_count, 2)
    value = key + 100
    return ((key, value),)


def test_prefix_key_depends_on_tokens_and_length():
    assert make_prefix_key([1, 2, 3]) == make_prefix_key([1, 2, 3])
    assert make_prefix_key([1, 2, 3]) != make_prefix_key([1, 2])
    assert make_prefix_key([1, 2, 3]) != make_prefix_key([1, 2, 4])


def test_prefix_cache_returns_longest_strict_prefix_and_tracks_refs():
    cache = PrefixCache(max_entries=8, block_size=2)
    cache.store_prompt([1, 2, 3, 4], _past_key_values(4))

    lookup = cache.lookup([1, 2, 3, 4, 9])

    assert lookup.matched_tokens == 4
    assert lookup.block_ids == (0, 1)
    assert lookup.past_key_values[0][0].shape[-2] == 4
    assert cache.stats().hits == 1
    assert cache.stats().referenced_entries == 1

    cache.release(lookup)

    assert cache.stats().referenced_entries == 0


def test_prefix_cache_records_miss_for_unknown_prefix():
    cache = PrefixCache(max_entries=8, block_size=2)
    cache.store_prompt([1, 2, 3, 4], _past_key_values(4))

    lookup = cache.lookup([8, 2, 3, 4, 9])

    assert lookup.entry is None
    assert cache.stats().hits == 0
    assert cache.stats().misses == 1


def test_prefix_cache_evicts_lru_unref_entries():
    cache = PrefixCache(max_entries=2, block_size=2)
    cache.store_prompt([1, 1], _past_key_values(2))
    pinned = cache.lookup([1, 1, 9])
    cache.store_prompt([2, 2], _past_key_values(2))
    cache.store_prompt([3, 3], _past_key_values(2))

    stats = cache.stats()

    assert stats.entries == 2
    assert stats.evictions == 1
    assert cache.lookup([1, 1, 8]).matched_tokens == 2

    cache.release(pinned)


def test_slice_past_key_values_preserves_cache_object_type_without_mutating_original():
    class CacheLike:
        def __init__(self):
            self.crop_called = False
            self.token_count = 4

        def crop(self, token_count):
            self.crop_called = True
            self.token_count = token_count

    cache_like = CacheLike()

    sliced = slice_past_key_values(cache_like, 2)

    assert cache_like.crop_called is False
    assert cache_like.token_count == 4
    assert isinstance(sliced, CacheLike)
    assert sliced.crop_called is True
    assert sliced.token_count == 2


def test_prefix_cache_lookup_returns_request_local_past_key_values_copy():
    cache = PrefixCache(max_entries=8, block_size=2)
    cache.store_prompt([1, 2], _past_key_values(2))

    first = cache.lookup([1, 2, 3])
    second = cache.lookup([1, 2, 4])

    assert first.past_key_values is not second.past_key_values
    assert first.past_key_values[0][0] is not second.past_key_values[0][0]


def test_clone_past_key_values_copies_tensor_storage():
    original = _past_key_values(2)

    cloned = clone_past_key_values(original)

    assert cloned[0][0] is not original[0][0]
    cloned[0][0][..., 0, 0] = -1
    assert original[0][0][..., 0, 0].item() == 0

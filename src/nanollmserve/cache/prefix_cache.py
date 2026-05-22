"""Teaching-scale prefix cache for reusable prompt KV state.

The cache stores block-aligned token prefixes and the corresponding
``past_key_values`` slice. It is intentionally small and in-process: the goal is
to make lookup, ref counting, and eviction visible before later milestones add
tensor-level paged KV execution.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from struct import pack
from typing import Any


@dataclass(frozen=True)
class PrefixCacheKey:
    token_hash: str
    token_count: int


@dataclass
class PrefixCacheEntry:
    key: PrefixCacheKey
    token_ids: tuple[int, ...]
    block_ids: tuple[int, ...]
    past_key_values: Any
    ref_count: int = 0
    hits: int = 0


@dataclass(frozen=True)
class PrefixCacheLookup:
    entry: PrefixCacheEntry | None

    @property
    def matched_tokens(self) -> int:
        if self.entry is None:
            return 0
        return self.entry.key.token_count

    @property
    def block_ids(self) -> tuple[int, ...]:
        if self.entry is None:
            return ()
        return self.entry.block_ids

    @property
    def past_key_values(self) -> Any | None:
        if self.entry is None:
            return None
        return self.entry.past_key_values


@dataclass(frozen=True)
class PrefixCacheStats:
    hits: int
    misses: int
    evictions: int
    entries: int
    tokens: int
    referenced_entries: int


class PrefixCache:
    """LRU cache for strict prompt-prefix KV reuse."""

    def __init__(self, *, max_entries: int = 64, block_size: int = 16) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        self.max_entries = max_entries
        self.block_size = block_size
        self._entries: OrderedDict[PrefixCacheKey, PrefixCacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def lookup(self, token_ids: list[int] | tuple[int, ...]) -> PrefixCacheLookup:
        """Return the longest cached strict prefix for ``token_ids``.

        Exact full-prompt hits are deliberately excluded because this milestone
        stores reusable KV state, not the final prompt logits needed to generate
        the first new token without any suffix prefill.
        """

        tokens = tuple(token_ids)
        for prefix_length in self._candidate_lengths(len(tokens)):
            key = make_prefix_key(tokens[:prefix_length])
            entry = self._entries.get(key)
            if entry is None or entry.token_ids != tokens[:prefix_length]:
                continue
            entry.ref_count += 1
            entry.hits += 1
            self._hits += 1
            self._entries.move_to_end(key)
            return PrefixCacheLookup(entry=entry)

        self._misses += 1
        return PrefixCacheLookup(entry=None)

    def release(self, lookup: PrefixCacheLookup) -> None:
        if lookup.entry is None:
            return
        lookup.entry.ref_count = max(lookup.entry.ref_count - 1, 0)
        self._evict_if_needed()

    def store_prompt(self, token_ids: list[int] | tuple[int, ...], past_key_values: Any) -> list[PrefixCacheKey]:
        """Store block-aligned prefixes sliced from a full-prompt KV tensor."""

        tokens = tuple(token_ids)
        stored: list[PrefixCacheKey] = []
        for prefix_length in self._stored_lengths(len(tokens)):
            prefix_tokens = tokens[:prefix_length]
            key = make_prefix_key(prefix_tokens)
            if key in self._entries:
                self._entries.move_to_end(key)
                continue
            entry = PrefixCacheEntry(
                key=key,
                token_ids=prefix_tokens,
                block_ids=tuple(range(_block_count(prefix_length, self.block_size))),
                past_key_values=slice_past_key_values(past_key_values, prefix_length),
            )
            self._entries[key] = entry
            stored.append(key)
            self._evict_if_needed()
        return stored

    def stats(self) -> PrefixCacheStats:
        return PrefixCacheStats(
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
            entries=len(self._entries),
            tokens=sum(entry.key.token_count for entry in self._entries.values()),
            referenced_entries=sum(1 for entry in self._entries.values() if entry.ref_count > 0),
        )

    def _candidate_lengths(self, token_count: int) -> list[int]:
        max_prefix = token_count - 1
        if max_prefix < self.block_size:
            return []
        lengths = list(range(self.block_size, max_prefix + 1, self.block_size))
        return list(reversed(lengths))

    def _stored_lengths(self, token_count: int) -> list[int]:
        if token_count < self.block_size:
            return []
        lengths = list(range(self.block_size, token_count + 1, self.block_size))
        if lengths[-1] != token_count:
            lengths.append(token_count)
        return lengths

    def _evict_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            evicted = False
            for key, entry in list(self._entries.items()):
                if entry.ref_count == 0:
                    del self._entries[key]
                    self._evictions += 1
                    evicted = True
                    break
            if not evicted:
                return


def make_prefix_key(token_ids: list[int] | tuple[int, ...]) -> PrefixCacheKey:
    digest = sha256()
    count = 0
    for token_id in token_ids:
        if token_id < 0:
            raise ValueError("token ids must be non-negative")
        digest.update(pack(">Q", int(token_id)))
        count += 1
    return PrefixCacheKey(token_hash=digest.hexdigest(), token_count=count)


def slice_past_key_values(past_key_values: Any, token_count: int) -> Any:
    """Slice Hugging Face-style KV tensors to ``token_count`` sequence slots."""

    if past_key_values is None:
        return None
    if hasattr(past_key_values, "to_legacy_cache"):
        return slice_past_key_values(past_key_values.to_legacy_cache(), token_count)
    if isinstance(past_key_values, tuple):
        return tuple(slice_past_key_values(item, token_count) for item in past_key_values)
    if isinstance(past_key_values, list):
        return [slice_past_key_values(item, token_count) for item in past_key_values]
    if hasattr(past_key_values, "shape") and hasattr(past_key_values, "__getitem__"):
        return past_key_values[..., :token_count, :].detach()
    return past_key_values


def _block_count(token_count: int, block_size: int) -> int:
    return (token_count + block_size - 1) // block_size

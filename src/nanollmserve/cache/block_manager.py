"""Block-based KV cache allocation metadata.

This module models the allocator side of paged KV cache. It does not own GPU
tensors or implement a PagedAttention kernel; it tracks which fixed-size token
blocks each request would hold so scheduler behavior and fragmentation are
visible in tests and benchmarks.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import ceil


class BlockAllocationError(RuntimeError):
    """Raised when the KV block pool cannot satisfy an allocation."""


@dataclass(frozen=True)
class KVBlock:
    block_id: int
    capacity_tokens: int


@dataclass
class RequestBlockTable:
    request_id: str
    block_ids: list[int] = field(default_factory=list)
    token_count: int = 0


@dataclass(frozen=True)
class KVBlockUsage:
    block_size: int
    total_blocks: int
    used_blocks: int
    free_blocks: int
    request_count: int
    allocated_tokens: int
    reserved_tokens: int

    @property
    def internal_fragmentation_tokens(self) -> int:
        return self.reserved_tokens - self.allocated_tokens

    @property
    def block_utilization(self) -> float:
        if self.reserved_tokens == 0:
            return 0.0
        return self.allocated_tokens / self.reserved_tokens


@dataclass
class KVBlockManager:
    total_blocks: int
    block_size: int = 16

    def __post_init__(self) -> None:
        if self.total_blocks < 1:
            raise ValueError("total_blocks must be at least 1")
        if self.block_size < 1:
            raise ValueError("block_size must be at least 1")
        self.blocks = [KVBlock(block_id=index, capacity_tokens=self.block_size) for index in range(self.total_blocks)]
        self.free_block_ids: deque[int] = deque(block.block_id for block in self.blocks)
        self.request_tables: dict[str, RequestBlockTable] = {}

    def allocate(self, request_id: str, token_count: int) -> RequestBlockTable:
        if request_id in self.request_tables:
            raise ValueError(f"request already has allocated blocks: {request_id}")
        if token_count < 0:
            raise ValueError("token_count must be non-negative")

        needed_blocks = self._blocks_for_tokens(token_count)
        self._ensure_free_blocks(needed_blocks)
        table = RequestBlockTable(
            request_id=request_id,
            block_ids=self._take_blocks(needed_blocks),
            token_count=token_count,
        )
        self.request_tables[request_id] = table
        return self.snapshot_request(request_id)

    def append_tokens(self, request_id: str, token_count: int = 1) -> RequestBlockTable:
        if token_count < 0:
            raise ValueError("token_count must be non-negative")
        if request_id not in self.request_tables:
            raise KeyError(f"request has no allocated blocks: {request_id}")
        if token_count == 0:
            return self.snapshot_request(request_id)

        table = self.request_tables[request_id]
        old_blocks = self._blocks_for_tokens(table.token_count)
        new_token_count = table.token_count + token_count
        new_blocks = self._blocks_for_tokens(new_token_count)
        additional_blocks = new_blocks - old_blocks
        self._ensure_free_blocks(additional_blocks)
        table.block_ids.extend(self._take_blocks(additional_blocks))
        table.token_count = new_token_count
        return self.snapshot_request(request_id)

    def release(self, request_id: str) -> list[int]:
        table = self.request_tables.pop(request_id, None)
        if table is None:
            raise KeyError(f"request has no allocated blocks: {request_id}")
        released = list(table.block_ids)
        self.free_block_ids.extend(released)
        return released

    def snapshot_request(self, request_id: str) -> RequestBlockTable:
        table = self.request_tables[request_id]
        return RequestBlockTable(
            request_id=table.request_id,
            block_ids=list(table.block_ids),
            token_count=table.token_count,
        )

    def usage(self) -> KVBlockUsage:
        used_blocks = sum(len(table.block_ids) for table in self.request_tables.values())
        allocated_tokens = sum(table.token_count for table in self.request_tables.values())
        return KVBlockUsage(
            block_size=self.block_size,
            total_blocks=self.total_blocks,
            used_blocks=used_blocks,
            free_blocks=len(self.free_block_ids),
            request_count=len(self.request_tables),
            allocated_tokens=allocated_tokens,
            reserved_tokens=used_blocks * self.block_size,
        )

    def _blocks_for_tokens(self, token_count: int) -> int:
        if token_count == 0:
            return 0
        return ceil(token_count / self.block_size)

    def _ensure_free_blocks(self, count: int) -> None:
        if count > len(self.free_block_ids):
            raise BlockAllocationError(
                f"not enough free KV blocks: needed={count} free={len(self.free_block_ids)}"
            )

    def _take_blocks(self, count: int) -> list[int]:
        return [self.free_block_ids.popleft() for _ in range(count)]

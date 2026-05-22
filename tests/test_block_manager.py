import pytest

from nanollmserve.cache.block_manager import BlockAllocationError, KVBlockManager


def test_block_manager_allocates_appends_and_releases_blocks():
    manager = KVBlockManager(total_blocks=4, block_size=4)

    table = manager.allocate("req-a", 5)
    assert table.block_ids == [0, 1]
    assert table.token_count == 5
    assert manager.usage().used_blocks == 2
    assert manager.usage().internal_fragmentation_tokens == 3

    table = manager.append_tokens("req-a", 3)
    assert table.block_ids == [0, 1]
    assert table.token_count == 8

    table = manager.append_tokens("req-a", 1)
    assert table.block_ids == [0, 1, 2]
    assert table.token_count == 9

    released = manager.release("req-a")
    assert released == [0, 1, 2]
    assert manager.usage().used_blocks == 0
    assert manager.usage().free_blocks == 4


def test_block_manager_reuses_released_blocks_fifo():
    manager = KVBlockManager(total_blocks=3, block_size=2)

    manager.allocate("req-a", 3)
    manager.release("req-a")
    table = manager.allocate("req-b", 2)

    assert table.block_ids == [2]


def test_block_manager_rejects_over_allocation_without_mutation():
    manager = KVBlockManager(total_blocks=1, block_size=4)
    manager.allocate("req-a", 4)

    with pytest.raises(BlockAllocationError):
        manager.allocate("req-b", 1)

    assert manager.usage().used_blocks == 1
    assert list(manager.request_tables) == ["req-a"]


def test_block_manager_usage_reports_utilization():
    manager = KVBlockManager(total_blocks=4, block_size=8)
    manager.allocate("req-a", 9)

    usage = manager.usage()

    assert usage.total_blocks == 4
    assert usage.used_blocks == 2
    assert usage.free_blocks == 2
    assert usage.request_count == 1
    assert usage.allocated_tokens == 9
    assert usage.reserved_tokens == 16
    assert usage.internal_fragmentation_tokens == 7
    assert usage.block_utilization == pytest.approx(9 / 16)

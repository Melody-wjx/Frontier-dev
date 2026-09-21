from __future__ import annotations

import pytest

from frontier.scheduler.replica_scheduler.base_replica_scheduler import (
    BaseReplicaScheduler,
)


@pytest.mark.parametrize(
    ("max_tokens", "block_size", "expected_blocks"),
    [
        (1, 16, 1),
        (4096, 16, 256),
        (4097, 16, 257),
    ],
)
def test_max_blocks_per_sequence_rounds_up(
    max_tokens: int, block_size: int, expected_blocks: int
) -> None:
    assert (
        BaseReplicaScheduler._calculate_max_blocks_per_sequence(
            max_tokens, block_size
        )
        == expected_blocks
    )


def test_max_blocks_per_sequence_rejects_nonpositive_block_size() -> None:
    with pytest.raises(ValueError, match="block_size must be > 0"):
        BaseReplicaScheduler._calculate_max_blocks_per_sequence(4096, 0)

from __future__ import annotations

import pytest

from frontier.entities.batch import Batch
from frontier.entities.request import Request


def _request(*, layer: int, completed: bool = False) -> Request:
    request = Request(
        arrived_at=0.0,
        num_prefill_tokens=16,
        num_decode_tokens=4,
        num_processed_tokens=16,
    )
    request._is_prefill_complete = True
    request._completed_layer_count = layer
    request._completed = completed
    return request


def _batch(requests: list[Request]) -> Batch:
    return Batch(
        replica_id=0,
        requests=requests,
        num_tokens=[1] * len(requests),
        is_moe=False,
    )


def test_af_inflight_layer_count_rejects_mismatch_after_first_two_requests() -> None:
    batch = _batch(
        [
            _request(layer=3),
            _request(layer=3),
            _request(layer=4),
        ]
    )

    with pytest.raises(AssertionError, match="Layer count mismatch"):
        _ = batch.af_inflight_layer_count


def test_af_inflight_layer_count_accepts_all_active_requests_at_same_layer() -> None:
    batch = _batch([_request(layer=7), _request(layer=7), _request(layer=7)])

    assert batch.af_inflight_layer_count == 7


def test_af_inflight_layer_count_ignores_completed_request_stale_layer() -> None:
    batch = _batch(
        [
            _request(layer=31, completed=True),
            _request(layer=5),
            _request(layer=5),
        ]
    )

    assert batch.af_inflight_layer_count == 5

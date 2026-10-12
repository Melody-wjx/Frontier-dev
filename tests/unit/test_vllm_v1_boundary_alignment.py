"""Default-off vLLM 0.10.2 boundary fixes, without GPU or timing fitting."""
from dataclasses import fields
from types import SimpleNamespace as NS

import pytest

from frontier.config.config import BaseExecutionTimePredictorConfig, VllmV1SchedulerConfig
from frontier.entities.request import Request
from frontier.kv_cache.replica_kv_cache_manager import ReplicaKVCacheManager
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler as Scheduler,
)
from frontier.types import ClusterType

FLAGS = ("align_fcfs_preemption_victim", "align_prefix_cache_computed_tokens",
         "preserve_skipped_waiting_order")


def request(i, prompt, computed=0, output=0, hashes=None):
    r = Request(0., prompt, 20000,
                num_processed_tokens=prompt + output if output else computed,
                block_hash_ids=hashes or [])
    r._id = i
    r._is_prefill_complete = output > 0
    return r


def scheduler(running=(), waiting=(), blocks=100, budget=16, chunked=True, **flags):
    s = object.__new__(Scheduler)
    s._cluster_type = ClusterType.MONOLITHIC
    s._num_stages = 1
    s._replica_id = s._dp_id = 0
    s._max_num_running_reqs = 16
    s._max_num_scheduled_tokens = budget
    s._max_model_len = 32768
    s._enable_preemption = True
    s._enable_chunked_prefill = chunked
    s._long_prefill_token_threshold = 8192 if chunked else 0
    s._scheduling_policy = 'fcfs'
    s._current_schedule_time = 1.
    s._watermark_blocks = 0
    s._final_prefill_reserved_slots = s._final_prefill_reserved_tokens = 0
    s._enable_phase_aware_thinking_profile = False
    s._monolithic_pp_terminal_release_followup_poll_pending = False
    s._spec_decode_enabled = False
    s._preempted_requests = []
    s._scheduled_num_computed_tokens_by_request = {}
    s._config = NS(num_blocks=blocks, block_size=16,
                   scheduling_policy='fcfs', enable_thinking_round_priority=False,
                   **{name: flags.get(name, False) for name in FLAGS})
    s._kv_cache_manager = ReplicaKVCacheManager(16, blocks, False, 'builtin', 0)
    s._running_requests = list(running)
    s._request_queue = list(waiting)
    for r in running:
        computed = r.num_processed_tokens - int(r.is_prefill_complete)
        s._scheduled_num_computed_tokens_by_request[r.id] = computed
        assert s._kv_cache_manager.allocate_slots(r, computed, num_computed_tokens=0) is not None
    s._allocation_map = {r.id: s._kv_cache_manager.get_num_blocks_for_request(r) for r in running}
    s._num_allocated_blocks = s._kv_cache_manager.num_used_blocks
    s._create_batch = lambda rs, ts: NS(requests=rs, num_tokens=ts)
    s._validate_boundary_alignment_scope()
    return s


def test_defaults_keep_baseline():
    assert all(not getattr(VllmV1SchedulerConfig(), flag) for flag in FLAGS)


def test_retired_prefill_limiter_is_not_a_config_option():
    for config_type in (BaseExecutionTimePredictorConfig, VllmV1SchedulerConfig):
        assert "saturated_prefill_interleave_chunk_tokens" not in {
            field.name for field in fields(config_type)
        }


@pytest.mark.parametrize("budget", [16, 256, 512, 8192])
def test_waiting_prefill_uses_available_budget_without_extra_limiter(budget):
    decode = request(0, 16, 31, 16)
    prefill = request(1, 1024)
    s = scheduler([decode], [prefill], blocks=128, budget=budget)
    batch = s._schedule_two_phase()
    tokens = {r.id: n for r, n in zip(batch.requests, batch.num_tokens)}
    assert tokens == {0: 1, 1: min(1024, budget - 1)}


@pytest.mark.parametrize('flag', FLAGS)
@pytest.mark.parametrize('field,value', [('_num_stages', 2), ('_spec_decode_enabled', True),
                                        ('_cluster_type', ClusterType.PREFILL)])
def test_scope_guard(flag, field, value):
    s = scheduler(**{flag: True})
    setattr(s, field, value)
    with pytest.raises(ValueError, match='MONOLITHIC, PP=1, non-speculative'):
        s._validate_boundary_alignment_scope()
    setattr(s._config, flag, False)
    s._validate_boundary_alignment_scope()


@pytest.mark.parametrize('enabled', [False, True])
def test_fcfs_tail_can_preempt_itself(enabled):
    a, b = request(0, 16, 30, 15), request(1, 48, 16)
    s = scheduler([a, b], blocks=3, budget=8192, align_fcfs_preemption_victim=enabled)
    batch = s._schedule_two_phase()
    assert [r.id for r in batch.requests] == ([0] if enabled else [1])
    assert [r.id for r in s._running_requests] == ([0] if enabled else [1])
    assert [r.id for r in (a, b) if r._preempted] == ([1] if enabled else [0])
    assert [r.id for r in s._get_sorted_waiting_queue()] == ([1] if enabled else [0])


@pytest.mark.parametrize('enabled', [False, True])
def test_skipped_head_preserves_fcfs_order(enabled):
    s = scheduler([request(0, 16, 31, 16)],
                  [request(1, 32), request(2, 48), request(3, 15), request(4, 1)],
                  chunked=False, preserve_skipped_waiting_order=enabled)
    b = s._schedule_two_phase()
    assert {r.id: n for r, n in zip(b.requests, b.num_tokens)} == {0: 1, 3: 15}
    assert [r.id for r in s._get_sorted_waiting_queue()] == ([1, 2, 4] if enabled else [4, 1, 2])


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('computed', [30, 31, 32, 46, 47, 48])
def test_allocator_and_scheduler_use_same_frontier(enabled, computed):
    r = request(0, 16, computed, computed - 16 + 1)
    s = scheduler([r], blocks=20, align_prefix_cache_computed_tokens=enabled)
    original = r.num_processed_tokens
    assert s._can_allocate_request(r, 1)
    s._allocate_request(r, 1)
    accounted = computed if enabled else computed + 1
    assert s._kv_cache_manager.get_num_blocks_for_request(r) == (accounted + 1 + 15) // 16
    assert r.num_processed_tokens == original


def test_full_pool_last_token_does_not_require_extra_block():
    r = request(0, 16, 31, 16)
    s = scheduler([r], blocks=2, align_prefix_cache_computed_tokens=True)
    b = s._schedule_two_phase()
    assert [r.id for r in b.requests] == [0]
    assert not s._preempted_requests


def test_cache_commit_does_not_publish_uncomputed_full_block():
    k = ReplicaKVCacheManager(16, 10, True, 'builtin', 0)
    r = request(0, 16, 30, 15, hashes=[101, 102, 103])
    assert k.allocate_slots(r, 30, num_computed_tokens=0) is not None
    assert k.allocate_slots(r, 1, num_computed_tokens=30) is not None
    assert k.num_cached_blocks[r.id] == 1
    assert k.block_pool.get_cached_block(102) is None
    assert k.allocate_slots(r, 1, num_computed_tokens=31) is not None
    assert k.num_cached_blocks[r.id] == 2
    assert k.block_pool.get_cached_block(102) is not None
    assert k.get_num_blocks_for_request(r) == 2


def test_new_prefix_blocks_counted_once_and_preflight_matches_commit():
    k = ReplicaKVCacheManager(16, 3, True, 'builtin', 0)
    a = request(0, 32, hashes=[101, 102])
    assert k.allocate_slots(a, 32, num_computed_tokens=0) is not None
    k.free(a)
    b = request(1, 48, hashes=[101, 102, 103])
    hits, tokens = k.get_computed_blocks(b)
    assert tokens == 32
    assert k.can_allocate_slots(b, 16, hits, num_computed_tokens=0)
    assert k.allocate_slots(b, 16, hits, num_computed_tokens=0) is not None
    assert k.get_num_blocks_for_request(b) == k.num_cached_blocks[b.id] == 3
    assert k.block_pool.get_num_free_blocks() == 0


def test_legacy_api_unchanged_and_negative_frontier_rejected():
    r = request(0, 16, 31, 16)
    legacy = ReplicaKVCacheManager(16, 3, False, 'builtin', 0)
    assert legacy.allocate_slots(r, 1) is not None
    assert legacy.get_num_blocks_for_request(r) == 3
    with pytest.raises(ValueError, match='num_computed_tokens must be >= 0'):
        legacy.allocate_slots(r, 1, num_computed_tokens=-1)

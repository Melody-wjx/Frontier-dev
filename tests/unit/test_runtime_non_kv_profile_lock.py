"""Host-wide file lock around uncached runtime non-KV overhead profiling.

The profiler measures device-wide free-memory deltas, so two simulator
processes profiling concurrently on the same host corrupt each other
(observed as a spurious negative non_torch_increase_bytes). The exclusive
flock in runtime_estimator must serialize those windows across processes.
"""

import contextlib
import fcntl
from types import SimpleNamespace

import pytest

from frontier.profiling.non_kv_cache_overhead import runtime_estimator


def test_lock_file_default_and_env_override(monkeypatch, tmp_path):
    monkeypatch.delenv(runtime_estimator._RUNTIME_NON_KV_PROFILE_LOCK_FILE_ENV, raising=False)
    assert runtime_estimator._resolve_runtime_non_kv_profile_lock_file().endswith(
        "frontier_runtime_non_kv_profile.lock"
    )
    override = tmp_path / "custom.lock"
    monkeypatch.setenv(runtime_estimator._RUNTIME_NON_KV_PROFILE_LOCK_FILE_ENV, str(override))
    assert runtime_estimator._resolve_runtime_non_kv_profile_lock_file() == str(override)


def test_file_lock_is_exclusive_while_held(monkeypatch, tmp_path):
    lock_path = tmp_path / "profile.lock"
    monkeypatch.setenv(runtime_estimator._RUNTIME_NON_KV_PROFILE_LOCK_FILE_ENV, str(lock_path))
    with runtime_estimator._runtime_non_kv_profile_file_lock():
        assert lock_path.exists()
        with open(lock_path) as probe:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    with open(lock_path) as probe:
        fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_uncached_profiling_runs_under_file_lock(monkeypatch, tmp_path):
    events = []
    lock_path = tmp_path / "profile.lock"
    monkeypatch.setenv(runtime_estimator._RUNTIME_NON_KV_PROFILE_LOCK_FILE_ENV, str(lock_path))
    real_lock = runtime_estimator._runtime_non_kv_profile_file_lock

    @contextlib.contextmanager
    def recording_lock():
        events.append("lock_enter")
        with real_lock():
            with open(lock_path) as probe:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        events.append("lock_exit")

    monkeypatch.setattr(runtime_estimator, "_runtime_non_kv_profile_file_lock", recording_lock)
    monkeypatch.setattr(runtime_estimator, "_build_cache_key", lambda **kwargs: ("key",))

    def fake_profile(**kwargs):
        events.append("profiled")
        return "raw"

    fake_result = runtime_estimator.RuntimeNonKVProfileResult(
        input_weights_memory_bytes=1,
        measured_weights_memory_bytes=1,
        overhead_bytes=0,
        non_kv_cache_memory_bytes=1,
        torch_peak_increase_bytes=0,
        non_torch_increase_bytes=0,
    )
    fake_replica = SimpleNamespace(
        attn_tensor_parallel_size=2,
        model_config=SimpleNamespace(get_name=lambda: "fake-model"),
    )
    fake_cluster = SimpleNamespace(name="MONOLITHIC")

    monkeypatch.setattr(
        runtime_estimator, "_profile_non_kv_cache_overhead_bytes_uncached", fake_profile
    )
    monkeypatch.setattr(
        runtime_estimator, "_normalize_profile_result", lambda **kwargs: fake_result
    )
    runtime_estimator.clear_runtime_non_kv_cache_overhead_cache()
    try:
        result = runtime_estimator.estimate_non_kv_cache_profile(
            replica_config=fake_replica,
            cluster_type=fake_cluster,
            max_num_batched_tokens=8192,
            weights_memory_bytes=1,
            weights_memory_source="param_counter",
        )
        assert result is fake_result
        assert events == ["lock_enter", "profiled", "lock_exit"]
        # cached path must not re-enter the lock or the profiler
        assert (
            runtime_estimator.estimate_non_kv_cache_profile(
                replica_config=fake_replica,
                cluster_type=fake_cluster,
                max_num_batched_tokens=8192,
                weights_memory_bytes=1,
                weights_memory_source="param_counter",
            )
            is fake_result
        )
        assert events == ["lock_enter", "profiled", "lock_exit"]
    finally:
        runtime_estimator.clear_runtime_non_kv_cache_overhead_cache()

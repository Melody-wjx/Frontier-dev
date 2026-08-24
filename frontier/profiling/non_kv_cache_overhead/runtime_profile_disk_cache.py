"""Cross-process disk cache for runtime non-KV memory profile results.

The runtime non-KV overhead measurement (see runtime_estimator.py) is a pure
function of a handful of static config inputs, but triggering it requires
importing the vLLM-backed profiling stack (~seconds of import time) and a GPU
measurement. This module lets the scheduler resolve the result from a small
JSON file instead, skipping both the heavy import and the measurement on warm
runs.

The key intentionally mirrors the fields of
runtime_estimator._build_cache_key; keep them in sync if that function
changes. The cache only needs to be self-consistent: identical inputs must
map to identical keys, and differing inputs must not collide.

Enable by setting FRONTIER_SIM_RUNTIME_PROFILE_CACHE_DIR to a writable
directory; set FRONTIER_SIM_RUNTIME_PROFILE_NO_CACHE=1 to bypass.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Dict, Optional

import torch

from frontier.profiling.non_kv_cache_overhead.nccl_buffer_estimator import (
    get_effective_nccl_buffer_config,
)

_CACHE_FILE_NAME = "runtime_non_kv_profile_cache.json"


def _cuda_total_memory_key() -> object:
    """Mirror runtime_estimator._get_current_cuda_total_memory_cache_key."""
    try:
        if not torch.cuda.is_available():
            return "cuda_unavailable"
        current_device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(current_device)
        return int(properties.total_memory)
    except Exception:
        return "cuda_total_memory_unknown"


def build_runtime_profile_disk_cache_key(
    *,
    replica_config,
    cluster_type,
    max_num_batched_tokens: int,
    weights_memory_bytes: int,
    weights_memory_source: str,
    nccl_buffer_config=None,
) -> str:
    """Build a stable hex key from the same inputs as the in-process cache."""
    model_config = replica_config.model_config
    fields = [
        cluster_type.name,
        model_config.get_name(),
        int(model_config.num_layers),
        int(model_config.num_q_heads),
        int(model_config.num_kv_heads),
        int(model_config.embedding_dim),
        int(model_config.mlp_hidden_dim),
        int(model_config.get_head_dim()),
        int(replica_config.attn_tensor_parallel_size),
        int(getattr(replica_config, "attn_data_parallel_size", 1)),
        int(getattr(replica_config, "moe_tensor_parallel_size", 1)),
        int(getattr(replica_config, "moe_expert_parallel_size", 1)),
        int(replica_config.num_pipeline_stages),
        int(max_num_batched_tokens),
        int(weights_memory_bytes),
        str(weights_memory_source),
        str(getattr(model_config, "torch_dtype", getattr(model_config, "dtype", "float16"))),
        str(getattr(getattr(replica_config, "speculative_decoding_config", None), "method", "")),
        str(getattr(getattr(replica_config, "speculative_decoding_config", None), "spec_model_name", "")),
        int(getattr(getattr(replica_config, "speculative_decoding_config", None), "mtp_n_predict", 0)),
        int(getattr(getattr(replica_config, "speculative_decoding_config", None), "mtp_num_layers", 0)),
        str(_cuda_total_memory_key()),
        [str(item) for item in get_effective_nccl_buffer_config(nccl_buffer_config).cache_fingerprint()],
    ]
    payload = json.dumps(fields, sort_keys=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _cache_file_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _CACHE_FILE_NAME)


def _read_store(cache_dir: str) -> Dict[str, dict]:
    cache_file = _cache_file_path(cache_dir)
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def load_runtime_profile_cache(cache_dir: str, key: str) -> Optional[dict]:
    """Return the cached result dict for key, or None on miss/error."""
    entry = _read_store(cache_dir).get(key)
    if not isinstance(entry, dict):
        return None
    required = (
        "input_weights_memory_bytes",
        "measured_weights_memory_bytes",
        "overhead_bytes",
    )
    if any(field not in entry for field in required):
        return None
    return entry


def store_runtime_profile_cache(cache_dir: str, key: str, result: dict) -> None:
    """Best-effort atomic upsert of a profile result; never raises."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        data = _read_store(cache_dir)
        data[key] = result
        fd, tmp_path = tempfile.mkstemp(
            dir=cache_dir, prefix=".runtime_non_kv_profile_", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, _cache_file_path(cache_dir))
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    except Exception:
        pass

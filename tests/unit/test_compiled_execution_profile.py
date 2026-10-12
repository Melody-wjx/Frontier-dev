"""Pure CPU execution-contract/ledger tests; no vLLM import or GPU required."""
from copy import deepcopy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from frontier.profiling.compiled_forward import make_plan
from frontier.profiling.compiled_forward_worker import _ForwardTap, _json_value
from frontier.profiling.execution_contract import (
    SCHEMA, fingerprint, load_profile, make_contract, require_compatible, summarize_graph_trace,
)


def runtime():
    return dict(vllm="0.10.2", torch="2.8", triton="3.4", cuda="12.8", device="A100",
                capability=[8, 0], dtype="bfloat16", model_config_sha256="test", tp=2, pp=1,
                compilation={"level":3, "custom_ops":[], "use_inductor":True},
                attention_backend="FLASH_ATTN")


def trace():
    return {"traceEvents": [
        dict(cat="user_annotation", name="frontier_compiled_forward", ts=100, dur=20, pid=10, tid=11),
        dict(cat="cuda_runtime", name="cudaGraphLaunch", ts=101, dur=5, pid=10, tid=11, args={"correlation":42}),
        dict(cat="kernel", name="fused_norm", ts=200, dur=5, pid=0, tid=7, args={"correlation":42}),
        dict(cat="kernel", name="fused_norm", ts=210, dur=7, pid=0, tid=7, args={"correlation":42}),
        dict(cat="kernel", name="gemm", ts=220, dur=11, pid=0, tid=7, args={"correlation":42}),
        dict(cat="kernel", name="prime", ts=50, dur=3, pid=0, tid=7, args={"correlation":9}),
    ]}


def test_graph_one_to_many_correlation_not_first_kernel_only():
    result = summarize_graph_trace(trace())
    assert result["activity_count"] == 3
    assert result["gpu_activity_sum_ms"] == pytest.approx(.023)
    assert next(r for r in result["ledger"] if r["implementation"]=="fused_norm")["calls"] == 2


def test_gpu_stream_id_is_not_cpu_thread_id():
    data = trace(); data["traceEvents"][2]["tid"] = 99
    assert summarize_graph_trace(data)["activity_count"] == 3


def test_unrelated_scope_and_launch_do_not_pollute_graph():
    data = trace()
    data["traceEvents"].extend([
        dict(cat="cuda_runtime", name="cudaGraphLaunch", ts=102, dur=2, pid=10, tid=90, args={"correlation":43}),
        dict(cat="kernel", name="other", ts=230, dur=30, pid=0, tid=7, args={"correlation":43})])
    assert summarize_graph_trace(data)["gpu_activity_sum_ms"] == pytest.approx(.023)


def test_duplicate_and_missing_gpu_activity_fail():
    data = trace(); data["traceEvents"].append(deepcopy(data["traceEvents"][2]))
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_graph_trace(data)
    data = trace(); data["traceEvents"] = data["traceEvents"][:2]
    with pytest.raises(ValueError, match="Missing"):
        summarize_graph_trace(data)


@pytest.mark.parametrize("index", [0,1])
def test_missing_graph_boundary_fails(index):
    data = trace(); data["traceEvents"].pop(index)
    with pytest.raises(ValueError):
        summarize_graph_trace(data)


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_invalid_duration(duration):
    data = trace(); data["traceEvents"][2]["dur"] = duration
    with pytest.raises(ValueError, match="Invalid"):
        summarize_graph_trace(data)


def test_contract_boundary_is_forward_not_blocks_or_executor():
    contract = make_contract(runtime())
    assert "tp_collectives" in contract["includes"]
    assert "final_norm" in contract["includes"]
    assert "sampling" in contract["excludes"]
    assert "rpc" in contract["excludes"]
    require_compatible(contract, deepcopy(contract))
    for field, value in [("unit", "block_ops"), ("graph_mode", "NONE"), ("includes", [])]:
        candidate = deepcopy(contract); candidate[field] = value
        with pytest.raises(ValueError, match="mismatch"):
            require_compatible(candidate, contract)


@pytest.mark.parametrize("field", list(runtime()))
def test_runtime_identity_mismatch_cannot_silently_reuse(field):
    contract = make_contract(runtime())
    changed = deepcopy(contract); changed["runtime"][field] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        require_compatible(changed, contract)


@pytest.mark.parametrize("field", list(runtime()))
def test_missing_runtime_identity_is_rejected(field):
    data = runtime(); data.pop(field)
    with pytest.raises(ValueError, match="Incomplete"):
        make_contract(data)


@pytest.mark.parametrize("batches,prompts,limit", [([], [1], 20), ([1], [], 20),
    ([0],[1],20), ([True],[1],20), ([1],[0],20), ([1],[18],20)])
def test_invalid_plan(batches, prompts, limit):
    with pytest.raises(ValueError):
        make_plan(batches,prompts,limit)


def test_grid_does_not_take_workload_latencies():
    assert len(make_plan([1,3,8,16,32], [512,4096],32768)) == 10
    assert make_plan([3,1,3], [512],32768)[0]["batch_size"] == 1


def test_checksum_and_boundary_checked_by_reader(tmp_path):
    contract = make_contract(runtime())
    data = dict(schema=SCHEMA, status="complete", rows=[dict(contract=contract,
                output_equivalent=True, trace_repetitions=[summarize_graph_trace(trace())])])
    data["payload_sha256"] = fingerprint(data)
    path = tmp_path / "profile.json"; path.write_text(json.dumps(data))
    assert load_profile(path, contract)["status"] == "complete"
    other = deepcopy(contract); other["unit"] = "legacy_linear_ops"
    with pytest.raises(ValueError, match="mismatch"):
        load_profile(path, other)
    data["rows"][0]["output_equivalent"] = False
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum"):
        load_profile(path, contract)


def test_json_configuration_stable():
    assert _json_value({"capture_sizes":(1,2,4)}) == {"capture_sizes":[1,2,4]}


def test_callable_pass_instance_has_stable_identity_not_address():
    class Pass:
        def __call__(self):
            pass
    a, b = _json_value(Pass()), _json_value(Pass())
    assert a == b
    assert a["callable"].endswith("Pass")
    assert a["source_sha256"]


@pytest.mark.parametrize("mode,query,active,expected_calls", [
    ("FULL", 1, 3, 1), ("FULL", 1, 2, 0), ("NONE", 1, 3, 0), ("FULL", 2, 3, 0)])
def test_live_batch_tap_is_matching_and_one_shot(monkeypatch, mode, query, active, expected_calls):
    metadata = type("FlashAttentionMetadata", (), {})()
    metadata.max_query_len, metadata.num_actual_tokens = query, active
    context = SimpleNamespace(cudagraph_runtime_mode=SimpleNamespace(name=mode),
                              attn_metadata={"layer":metadata}, batch_descriptor="capture4")
    module = ModuleType("vllm.forward_context")
    module.get_forward_context = lambda: context
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.forward_context", module)
    output, calls = object(), []
    tap = _ForwardTap(lambda *args, **kwargs: output)
    tap.expected_active = 3
    def measure():
        assert tap.callback is None
        assert tap.last == ("capture4", metadata, output)
        calls.append(True)
        return {"measured":True}
    tap.callback = measure
    assert tap() is output
    assert tap() is output
    assert len(calls) == expected_calls
    assert tap.result == ({"measured":True} if expected_calls else None)


def test_callable_compiler_cache_identity_is_preserved():
    class Pass:
        def __call__(self):
            pass
        def uuid(self):
            return b"versioned-pass-config"
    assert _json_value(Pass())["compiler_cache_uuid"] == {"bytes_hex":b"versioned-pass-config".hex()}

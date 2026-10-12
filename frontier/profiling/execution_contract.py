"""Versioned execution identity and exclusive GPU ledger for compiled profiles.

No torch dependency; importing this module never changes legacy predictors.
A native Forward profile is deliberately NOT a linear-op training CSV.
"""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "frontier.compiled-forward.v1"
INCLUDES = ["embedding", "transformer_blocks", "kv_cache_write", "tp_collectives", "final_norm"]
EXCLUDES = ["scheduler", "input_preparation", "logits", "sampling", "output_processing", "rpc"]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def make_contract(runtime):
    required = {"vllm", "torch", "triton", "cuda", "device", "capability", "dtype",
                "model_config_sha256", "tp", "pp", "compilation", "attention_backend"}
    if not required <= runtime.keys() or any(runtime[k] is None for k in required):
        raise ValueError("Incomplete execution identity")
    if runtime["vllm"] != "0.10.2" or runtime["pp"] != 1:
        raise ValueError("Audited producer supports vLLM 0.10.2 PP=1 only")
    return dict(runtime=runtime, unit="native_captured_forward", graph_mode="FULL",
                includes=INCLUDES.copy(), excludes=EXCLUDES.copy(),
                stream_measurement="cuda_event_elapsed_ms",
                kernel_measurement="sum_of_unique_correlated_gpu_activity_ms",
                composition="whole_forward_exclusive; never add included components again",
                sampling="frozen_real_buffers_replay; not serving_step_wall_time")


def require_compatible(actual, expected):
    """Compare the complete declared contract, not just shape/dtype/model name."""
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ValueError("Both complete execution contracts are required")
    required = {"runtime", "unit", "graph_mode", "includes", "excludes", "stream_measurement",
                "kernel_measurement", "composition", "sampling"}
    if not required <= actual.keys() or not required <= expected.keys():
        raise ValueError("Incomplete execution contract")
    if fingerprint(actual) != fingerprint(expected):
        keys = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
        raise ValueError("Execution contract mismatch: " + ", ".join(keys))


def summarize_graph_trace(trace, scope_name="frontier_compiled_forward"):
    """Attribute ALL graph activities, not the first correlated kernel.

    This is a new, graph-only producer, NOT a change to RecordFunctionTracer.
    Scope ownership is CPU pid/tid; GPU stream/thread IDs are separate domains.
    Fail closed if CUPTI did not export graph-launch correlation. Unknown kernel
    names stay intact; no name-based redistribution into unfused operators.
    """
    events = trace["traceEvents"]
    scopes = [e for e in events if e.get("cat") == "user_annotation" and e.get("name") == scope_name]
    if len(scopes) != 1:
        raise ValueError("Expected exactly one native graph scope")
    scope = scopes[0]
    launches = [e for e in events if e.get("cat") in {"cuda_runtime", "cuda_driver"}
                and "GraphLaunch" in e.get("name", "") and "dur" in e
                and (e.get("pid"), e.get("tid")) == (scope.get("pid"), scope.get("tid"))
                and scope["ts"] <= e["ts"] and e["ts"]+e["dur"] <= scope["ts"]+scope["dur"]]
    correlations = {e.get("args", {}).get("correlation") for e in launches}
    correlations.discard(None)
    if len(launches) != 1 or len(correlations) != 1:
        raise ValueError("Expected one correlated cudaGraphLaunch per capture")
    selected = [e for e in events if e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
                and e.get("args", {}).get("correlation") in correlations]
    if not selected or not any(e.get("cat") == "kernel" for e in selected):
        raise ValueError("Missing correlated graph kernels; refusing zero/partial profile")
    grouped = defaultdict(list)
    seen = set()
    for e in selected:
        args = e.get("args", {})
        key = (e.get("pid"), e.get("tid"), e["ts"], e["dur"], e["name"],
               args.get("correlation"), args.get("graph node id"))
        if key in seen:
            raise ValueError("Duplicate GPU activity in graph trace")
        seen.add(key)
        duration = float(e["dur"]) / 1000
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid GPU duration")
        grouped[(e["cat"], e["name"])].append(duration)
    ledger = [dict(category=category, implementation=name, calls=len(values),
                   sum_ms=sum(values), durations_ms=values)
              for (category, name), values in sorted(grouped.items())]
    return dict(ledger=ledger, activity_count=len(selected),
                gpu_activity_sum_ms=sum(row["sum_ms"] for row in ledger),
                implementation_signature=fingerprint([(r["category"], r["implementation"], r["calls"]) for r in ledger]))


def load_profile(path, expected_contract):
    """Explicit reader: never silently interpret Forward as model_time/GEMMs."""
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA or data.get("status") != "complete":
        raise ValueError("Unsupported or incomplete compiled execution profile")
    if data.get("payload_sha256") != fingerprint({k:v for k,v in data.items() if k != "payload_sha256"}):
        raise ValueError("Profile checksum mismatch")
    if not data.get("rows"):
        raise ValueError("Empty compiled execution profile")
    for row in data["rows"]:
        require_compatible(row["contract"], expected_contract)
        if not row.get("output_equivalent") or not row.get("trace_repetitions"):
            raise ValueError("Unverified compiled execution row")
    return data

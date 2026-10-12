"""CPU-only tests for disjoint accounting, exact coverage and legacy fallback."""
import ast
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.execution_time_predictor.compiled_activation_profile import (
    BOUNDARY, SCHEMA, CompiledActivationProfile, activation_symbols, summarize_samples,
)
from frontier.profiling.execution_contract import fingerprint


def fixture_data():
    contract = dict(runtime={"compilation":{"cudagraph_capture_sizes":[1,2,4,8,16,32,64]}})
    return dict(schema=SCHEMA, boundary=BOUNDARY, execution_contract=contract,
        contract_sha256=fingerprint(contract), compiler_evidence=[{"sha256":"source"}],
        identity=dict(model_name="test", device_key="a100", num_layers=28,
                      embedding_dim=3584, mlp_hidden_dim=18944, tp=2, dtype="BF16"),
        points=[dict(active=n, capture=cap, samples_ms=[.002]*6, median_ms=.002, relative_mad=0.)
                for n,cap in [(1,1),(3,4),(8,8)]])


def write_data(tmp_path, data):
    data["payload_sha256"] = fingerprint({k:v for k,v in data.items() if k!="payload_sha256"})
    path=tmp_path/"profile.json"
    path.write_text(json.dumps(data))
    return path


def table(tmp_path):
    d=fixture_data()
    return CompiledActivationProfile(write_data(tmp_path,d), d["identity"], d["contract_sha256"])


@pytest.mark.parametrize("active,capture,expected", [(1,1,.002),(3,4,.002),(8,8,.002),
    (2,2,.009),(4,4,.009),(5,8,.009),(16,16,.009),(64,64,.009),(3,8,.009)])
def test_exact_shapes_only_no_clipping_or_interpolation(tmp_path, active, capture, expected):
    t=table(tmp_path)
    assert t.predict(active,capture,.009)==expected
    assert t.stats["hit" if expected==.002 else "missing_shape"]==1


@pytest.mark.parametrize("field", list(fixture_data()["identity"]))
def test_model_hardware_identity_mismatch(tmp_path, field):
    d=fixture_data(); expected=deepcopy(d["identity"]);expected[field]="different"
    with pytest.raises(ValueError,match="identity"):
        CompiledActivationProfile(write_data(tmp_path,d),expected,d["contract_sha256"])


@pytest.mark.parametrize("fingerprint_value", [None,"","wrong"])
def test_explicit_contract_required(tmp_path, fingerprint_value):
    d=fixture_data()
    with pytest.raises(ValueError,match="contract"):
        CompiledActivationProfile(write_data(tmp_path,d),d["identity"],fingerprint_value)


@pytest.mark.parametrize("mutation", ["boundary","checksum","empty","duplicate","median","shape","evidence"])
def test_invalid_payload(tmp_path, mutation):
    d=fixture_data()
    if mutation=="boundary":d["boundary"]="whole_forward_ms"
    if mutation=="empty":d["points"]=[]
    if mutation=="duplicate":d["points"].append(deepcopy(d["points"][0]))
    if mutation=="median":d["points"][0]["median_ms"]+=.01
    if mutation=="shape":d["points"][0]["active"]=2
    if mutation=="evidence":d["compiler_evidence"]=[]
    path=write_data(tmp_path,d)
    if mutation=="checksum":
        d["points"][0]["median_ms"]+=.01
        path.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        CompiledActivationProfile(path,d["identity"],d["contract_sha256"])


@pytest.mark.parametrize("samples", [[0,1,1],[-1,1,1],[float('nan'),1,1],[float('inf'),1,1],[1,2],[1,2,3]])
def test_invalid_or_unstable_samples(samples):
    with pytest.raises(ValueError):summarize_samples(samples)


@pytest.mark.parametrize("ops,arguments,accepted", [
    ("aten.silu, aten.mul","in_ptr0, out_ptr0, xnumel, XBLOCK",True),
    ("aten.silu, aten.mul, aten.mm","in_ptr0, out_ptr0, xnumel, XBLOCK",False),
    ("aten.silu, aten.mul","in_ptr0, residual, out_ptr0, xnumel, XBLOCK",False),
    ("aten.mul","in_ptr0, out_ptr0, xnumel, XBLOCK",False)])
def test_kernel_name_alone_is_not_sufficient(ops,arguments,accepted):
    source=f"# Topologically Sorted Source Nodes: [silu, mul], Original ATen: [{ops}]\n"
    source+=f"k = async_compile.triton('triton_silu', '''def k({arguments}):\n    pass\n''')\n"
    assert activation_symbols(source)==({"triton_silu"} if accepted else set())


def getter():
    source=Path(__file__).resolve().parents[2]/"frontier/execution_time_predictor/sklearn_execution_time_predictor.py"
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="SklearnExecutionTimePredictor")
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_get_mlp_layer_act_execution_time")
    env={"Batch":object,"MeasurementType":SimpleNamespace(KERNEL_ONLY="kernel")}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(source),"exec"),env)
    return env[method.name]


@pytest.mark.parametrize("prefill", [0,1])
@pytest.mark.parametrize("mode", ["FULL","NONE"])
@pytest.mark.parametrize("family", ["kernel","eager"])
@pytest.mark.parametrize("enabled", [False,True])
def test_runtime_disjointness_and_default_parity(tmp_path,prefill,mode,family,enabled):
    t=table(tmp_path)
    obj=SimpleNamespace(_supports_operation=lambda x:True,_cluster_type="mono",
        _predictions={"mlp_act":{(4,):.009}},_active_measurement_type=family,
        _compiled_decode_activation=t if enabled else None)
    b=SimpleNamespace(num_prefill_tokens=prefill,num_decode_tokens=3,requests=[1,2,3],
        get_effective_total_tokens_rounded=lambda x:4,get_decode_cuda_graph_runtime_mode=lambda:mode)
    before=deepcopy(obj._predictions)
    expected=.002 if enabled and not prefill and mode=="FULL" and family=="kernel" else .009
    assert getter()(obj,b)==expected
    assert obj._predictions==before  # Never pollute trained/prediction caches.


def test_speculative_multi_token_batch_falls_back(tmp_path):
    t=table(tmp_path)
    obj=SimpleNamespace(_supports_operation=lambda x:True,_cluster_type="mono",
        _predictions={"mlp_act":{(4,):.009}},_active_measurement_type="kernel",_compiled_decode_activation=t)
    b=SimpleNamespace(num_prefill_tokens=0,num_decode_tokens=4,requests=[1,2,3],
        get_effective_total_tokens_rounded=lambda x:4,get_decode_cuda_graph_runtime_mode=lambda:"FULL")
    assert getter()(obj,b)==.009
    assert t.stats==Counter(ineligible_runtime=1)

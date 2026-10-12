from copy import deepcopy
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from frontier.entities.execution_time import ExecutionTime
from frontier.execution_time_predictor.serving_forward_profile import (
    ServingForwardProfile, aggregate, batch_shape, features, fit, validate_shape,
)
from frontier.profiling.execution_contract import INCLUDES, EXCLUDES, fingerprint
from frontier.profiling.serving_forward import make_plan


def raw_data():
    contract=dict(runtime=dict(tp=2),includes=INCLUDES,excludes=EXCLUDES)
    rows=[]
    for mode in ["FULL","NONE"]:
        for i in range(1,17):
            q=[1]*i if mode=="FULL" else [32*i]
            shape=dict(query=q,history=[64*i]*len(q),input_tokens=sum(q),mode=mode,
                       phase="decode" if mode=="FULL" else "prefill")
            for rank in [0,1]:
                rows.append(dict(case=dict(id=f"{mode}_{i}",split="holdout" if i==16 else "train"),
                    rank=rank,contract=contract,output_equivalent=True,shape=shape,
                    stream_samples_ms=[1.+sum(q)/100+rank/10]*3))
    d=dict(schema="frontier.serving-forward.v1",status="complete",rows=rows)
    d["payload_sha256"]=fingerprint(d)
    return d


def test_training_excludes_holdout():
    a=raw_data();b=deepcopy(a)
    for row in b["rows"]:
        if row["case"]["split"]=="holdout":row["stream_samples_ms"]=[999.]*3
    b["payload_sha256"]=fingerprint({k:v for k,v in b.items() if k!="payload_sha256"})
    assert fit(a,{})["models"]==fit(b,{})["models"]


def test_native_padding_and_sampled_token_boundary():
    b=SimpleNamespace(num_tokens=[7,1],requests=[SimpleNamespace(num_processed_tokens=512,is_prefill_complete=False),
        SimpleNamespace(num_processed_tokens=1025,is_prefill_complete=True)])
    s=batch_shape(b,[1,2,4,8,16,32,64])
    assert s["history"]==[512,1024] and s["mode"]=="NONE"
    b.num_tokens=[8,1]
    assert batch_shape(b,[1,2,4,8,16,32,64])["input_tokens"]==16
    b.num_tokens=[1,1]
    assert batch_shape(b,[1,2,4,8,16,32,64])["mode"]=="FULL"


def test_ranks_are_max_not_sum():
    _,points=aggregate(raw_data())
    assert points[0]["median_ms"]==pytest.approx(1.11)


def test_work_preserves_query_history_association():
    a=dict(query=[1024,1],history=[0,8192],mode="NONE",input_tokens=1025)
    b=dict(a,history=[8192,0])
    assert features(a)[-4]==0
    assert features(b)[-4]==pytest.approx(1024*8192/1e6)
    assert features(a)[-3]>features(b)[-3]


@pytest.mark.parametrize("q,h,t,mode", [([],[],0,"NONE"),([0],[0],0,"NONE"),
    ([1],[-1],1,"FULL"),([2],[0],2,"FULL"),([1],[32768],1,"FULL"),
    ([9000],[0],9000,"NONE"),([2],[0],1,"NONE"),([1],[0],1,"PIECEWISE"),
    ([1,1],[0],2,"FULL"),([1.0],[0],1,"FULL")])
def test_invalid_shape(q,h,t,mode):
    with pytest.raises(ValueError):validate_shape(dict(query=q,history=h,input_tokens=t,mode=mode))


@pytest.mark.parametrize("mutation",["checksum","contract","identity","boundary","coef"])
def test_contract_guards(tmp_path,mutation):
    d=fit(raw_data(),{"model":"test"});expected=d["contract_sha256"]
    if mutation=="contract":expected="wrong"
    if mutation=="identity":d["identity"]={}
    if mutation=="boundary":
        d["contract"]["includes"]=[];expected=fingerprint(d["contract"]);d["contract_sha256"]=expected
    if mutation=="coef":d["models"]["FULL"]["coefficients"][0]=-1
    d["payload_sha256"]=fingerprint({k:v for k,v in d.items() if k!="payload_sha256"})
    if mutation=="checksum":d["payload_sha256"]="bad"
    path=tmp_path/"model.json";path.write_text(json.dumps(d))
    with pytest.raises(ValueError):ServingForwardProfile(path,{"model":"test"},expected)


def test_extrapolation_report_not_clipping(tmp_path):
    d=fit(raw_data(),{});path=tmp_path/"model.json";path.write_text(json.dumps(d))
    p=ServingForwardProfile(path,{},d["contract_sha256"])
    s=dict(query=[1]*32,history=[30000]*32,input_tokens=32,mode="FULL",phase="decode")
    p.predict(s)
    assert p.stats["extrapolated"]==1 and p.stats["replaced_steps"]==1


@pytest.mark.parametrize("layers",[1,2,28,32,80])
def test_whole_forward_is_once_and_cpu_unchanged(layers):
    e=ExecutionTime.from_serving_forward(10.,layers,schedule_time=1.,sampler_e2e_time=2.,
        prepare_inputs_e2e_time=3.,process_model_outputs_time=4.,ray_comm_time=5.)
    assert e.model_time==.010 and e.total_time==pytest.approx(.025)
    assert e.attention_all_reduce_time==e.mlp_all_reduce_time==0
    assert e.attention_pre_proj_time==e.mlp_layer_act_execution_time==0


def test_double_charging_rejected():
    with pytest.raises(ValueError,match="exclusive"):
        ExecutionTime.from_serving_forward(10.,28,schedule_time=0.,sampler_e2e_time=0.,
            prepare_inputs_e2e_time=0.,process_model_outputs_time=0.,ray_comm_time=0.,
            mlp_layer_act_execution_time=1.)


def test_ledger_sums_to_total():
    from frontier.metrics.metrics_store import MetricsStore
    e=ExecutionTime.from_serving_forward(10.,28,schedule_time=1.,sampler_e2e_time=2.,
        prepare_inputs_e2e_time=3.,process_model_outputs_time=4.,ray_comm_time=5.)
    ledger=MetricsStore._build_frontier_stage_batch_component_ledger(None,e)
    assert ledger["serving_forward_time"]==10.
    assert sum(ledger.values())==pytest.approx(e.total_time*1000)


def test_plan_independent_and_all_phases():
    plan=make_plan()
    assert {c["phase"] for c in plan}=={"decode","prefill","mixed"}
    assert len({c["id"] for c in plan})==len(plan)
    assert any(c["split"]=="holdout" for c in plan)
    assert max(c["n"] for c in plan)==64
    assert {c["phase"] for c in make_plan(True)}=={"decode","prefill","mixed"}


def test_enabled_by_default_with_explicit_legacy_optout():
    from frontier.config.config import (
        BaseExecutionTimePredictorConfig, RandomForrestExecutionTimePredictorConfig,
    )
    assert BaseExecutionTimePredictorConfig.__dataclass_fields__["serving_forward_enabled"].default is True
    assert RandomForrestExecutionTimePredictorConfig().serving_forward_enabled is True
    assert RandomForrestExecutionTimePredictorConfig(serving_forward_enabled=False).serving_forward_enabled is False


@pytest.mark.parametrize("flags,expected", [([],True),
    (["--origin_profiler"],False),
    (["--no-random_forrest_execution_time_predictor_config_serving_forward_enabled"],False),
    (["--random_forrest_execution_time_predictor_config_serving_forward_enabled"],True)])
def test_cli_default_and_explicit_toggle(monkeypatch,flags,expected):
    import sys
    from frontier.config.config import SimulationConfig
    from frontier.config.flat_dataclass import create_flat_dataclass
    monkeypatch.setattr(sys,"argv",["frontier.main",*flags])
    flat=create_flat_dataclass(SimulationConfig).create_from_cli_args()
    assert flat.random_forrest_execution_time_predictor_config_serving_forward_enabled is expected


@pytest.mark.parametrize("flags", [
    ["--origin_profiler","--random_forrest_execution_time_predictor_config_serving_forward_enabled"],
    ["--random_forrest_execution_time_predictor_config_serving_forward_enabled","--origin_profiler"],
    ["--origin_profiler","--origin_profiler"],
])
def test_origin_profiler_disables_all_predictor_variants(monkeypatch,flags):
    import sys
    from dataclasses import fields
    from frontier.config.config import SimulationConfig
    from frontier.config.flat_dataclass import create_flat_dataclass
    monkeypatch.setattr(sys,"argv",["frontier.main",*flags])
    flat=create_flat_dataclass(SimulationConfig).create_from_cli_args()
    names=[f.name for f in fields(flat) if f.name.endswith("_serving_forward_enabled")]
    assert names and all(getattr(flat,name) is False for name in names)
    assert not any(name.startswith("__disable_alias_") for name in vars(flat))


def test_origin_profiler_in_help_and_previous_name_not_supported(monkeypatch,capsys):
    import sys
    from frontier.config.config import SimulationConfig
    from frontier.config.flat_dataclass import create_flat_dataclass
    flat=create_flat_dataclass(SimulationConfig)
    monkeypatch.setattr(sys,"argv",["frontier.main","--help"])
    with pytest.raises(SystemExit) as exc:flat.create_from_cli_args()
    assert exc.value.code==0 and "--origin_profiler" in capsys.readouterr().out
    monkeypatch.setattr(sys,"argv",["frontier.main","--previous_profiler"])
    with pytest.raises(SystemExit) as exc:flat.create_from_cli_args()
    assert exc.value.code==2


@pytest.mark.parametrize("enabled,dummy", [(False,False),(False,True),(True,True)])
def test_disabled_or_dummy_never_loads_profile(enabled,dummy):
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
    # Deliberately no model, paths or output directory: these must not be read.
    obj=SimpleNamespace(_config=SimpleNamespace(serving_forward_enabled=enabled),
                        _enable_dummy_mode=dummy)
    SklearnExecutionTimePredictor._load_serving_forward(obj,None)
    assert obj._serving_forward is None


@pytest.mark.parametrize("filename,contract", [(None,None),("profile.json",None),(None,"hash")])
def test_default_on_requires_explicit_profile_and_contract(monkeypatch,filename,contract):
    from frontier.config.config import RandomForrestExecutionTimePredictorConfig
    from frontier.execution_time_predictor import sklearn_execution_time_predictor as module
    config=RandomForrestExecutionTimePredictorConfig(
        serving_forward_profile=filename,serving_forward_contract_sha256=contract)
    obj=SimpleNamespace(_config=config,_enable_dummy_mode=False,
        _model_config=SimpleNamespace(is_moe=False,get_default_precision=lambda:SimpleNamespace(name="BF16"),
                                     quantization_config=None),
        _replica_config=SimpleNamespace(num_pipeline_stages=1,attn_data_parallel_size=1,
                                       speculative_decoding_config=SimpleNamespace(enabled=False)),
        _cluster_type=module.ClusterType.MONOLITHIC)
    monkeypatch.setattr(module.global_vars,"get_decode_cuda_graph_mode",lambda:"full_decode_only")
    with pytest.raises(ValueError,match="provide both serving_forward_profile and serving_forward_contract_sha256"):
        module.SklearnExecutionTimePredictor._load_serving_forward(obj,None)


def test_default_on_loads_matching_profile_without_enable_flag(tmp_path,monkeypatch):
    from frontier.config.config import RandomForrestExecutionTimePredictorConfig
    from frontier.execution_time_predictor import sklearn_execution_time_predictor as module
    captures=[1,2,4,8,16,32,64]
    raw=raw_data()
    for row in raw["rows"]:
        row["contract"]["runtime"].update(vllm="0.10.2",attention_backend="test.FlashInferMetadata",
                                         compilation=dict(cudagraph_capture_sizes=captures))
    raw["payload_sha256"]=fingerprint({k:v for k,v in raw.items() if k!="payload_sha256"})
    identity=dict(model_name="test",device_key="test_gpu",num_layers=28,
                  embedding_dim=128,mlp_hidden_dim=256,tp=2,dtype="BF16")
    data=fit(raw,identity)
    path=tmp_path/"profile.json";path.write_text(json.dumps(data))
    config=RandomForrestExecutionTimePredictorConfig(
        serving_forward_profile=str(path),serving_forward_contract_sha256=data["contract_sha256"])
    obj=SimpleNamespace(_config=config,_enable_dummy_mode=False,
        _model_config=SimpleNamespace(is_moe=False,get_default_precision=lambda:SimpleNamespace(name="BF16"),
            quantization_config=None,get_name=lambda:"test",num_layers=28,embedding_dim=128,mlp_hidden_dim=256),
        _replica_config=SimpleNamespace(num_pipeline_stages=1,attn_data_parallel_size=1,
            speculative_decoding_config=SimpleNamespace(enabled=False),device="test_gpu",attn_tensor_parallel_size=2),
        _cluster_type=module.ClusterType.MONOLITHIC,_uses_unified_attention_profile=lambda:False)
    monkeypatch.setattr(module.global_vars,"get_decode_cuda_graph_mode",lambda:"full_decode_only")
    monkeypatch.setattr(module.global_vars,"get_cudagraph_capture_sizes",lambda:captures)
    monkeypatch.setattr(module.atexit,"register",lambda *args:None)
    module.SklearnExecutionTimePredictor._load_serving_forward(obj,SimpleNamespace(output_dir=str(tmp_path)))
    assert obj._serving_forward is not None
    shape=dict(query=[1]*4,history=[256]*4,input_tokens=4,mode="FULL",phase="decode")
    explicitly_loaded=ServingForwardProfile(path,identity,data["contract_sha256"])
    assert obj._serving_forward.predict(shape)==explicitly_loaded.predict(shape)


@pytest.mark.parametrize("calibrated",[False,True])
def test_native_entry_cpu_composition(calibrated):
    source=Path(__file__).resolve().parents[2]/"frontier/execution_time_predictor/sklearn_execution_time_predictor.py"
    cls=next(n for n in ast.parse(source.read_text()).body
             if isinstance(n,ast.ClassDef) and n.name=="SklearnExecutionTimePredictor")
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=="_predict_serving_forward")
    env=dict(ExecutionTime=ExecutionTime,global_vars=SimpleNamespace(get_cudagraph_capture_sizes=lambda:[1,2,4,8,16,32,64]))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),"exec"),env)
    obj=SimpleNamespace(_model_config=SimpleNamespace(num_layers=28),
        _serving_forward=SimpleNamespace(predict=lambda shape:10.),
        _get_schedule_time=lambda b:1.,_get_sampler_e2e_time=lambda b:2.,
        _get_prepare_inputs_e2e_time=lambda b:3.,_get_process_model_outputs_time=lambda b:4.,
        _get_ray_comm_time=lambda b:5.,_step_calibration_enabled=lambda:calibrated,
        _get_step_calibration_scale=lambda b:.5,_validate_prediction_value=lambda v,*a:v)
    batch=SimpleNamespace(num_tokens=[1],requests=[SimpleNamespace(num_processed_tokens=513,is_prefill_complete=True)])
    result=env[fn.name](obj,batch,0,28)
    assert result.model_time_ms==10.
    assert result.total_time*1000==pytest.approx(17.5 if calibrated else 25.)
    with pytest.raises(ValueError,match="split"):
        env[fn.name](obj,batch,0,1)

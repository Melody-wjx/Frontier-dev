"""Opt-in FULL graph shape semantics, bounded interpolation, and old fallback."""
import json
from types import SimpleNamespace as NS

import pytest

from frontier.config import RandomForrestExecutionTimePredictorConfig, SimulationConfig
from frontier.execution_time_predictor.full_graph_decode_profile import FullGraphDecodeProfile
from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
from frontier.profiling.attention.full_graph_decode import make_plan
from frontier.types import MeasurementType


class Predictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):return {}
    def _get_estimator(self):return None


def document():
    return {'metadata':dict(schema_version=1,backend='FLASH_ATTN',vllm_version='0.10.2',fa_version=2,
        dtype='bfloat16',distribution='uniform',measurement='attention_subgraph_stream_ms',
        context_semantics='past_kv_tokens; visible_sequence_length=context_tokens+1',
        cache_write_included=False,cpu_included=False,max_model_len=256,capture_sizes=[1,2,4,8]),
        'rows':[dict(active=3,capture=4,context_tokens=k,split='train',median_ms=v,
                     sample_count=9,samples_ms=[v]*9,relative_mad=0)
                for k,v in [(64,1.0),(128,2.0)]]}


def load(tmp_path,data=None,expected=None):
    path=tmp_path/'profile.json';path.write_text(json.dumps(document() if data is None else data))
    return FullGraphDecodeProfile(path,expected or {'max_model_len':256})


def test_default_off_without_files():
    config=RandomForrestExecutionTimePredictorConfig()
    assert config.full_graph_decode_attention_enabled is False
    p=object.__new__(Predictor);p._config=config
    p._load_full_graph_decode_attention(None)
    assert p._full_graph_decode_attention is None
    assert p._get_full_graph_decode_attention_time(None,1,1) is None


def test_exact_and_interpolated_without_axis_substitution(tmp_path):
    table=load(tmp_path)
    assert table.predict(3,4,64)==1
    assert table.predict(3,4,96)==1.5
    assert table.predict(3,4,128)==2
    assert table.predict(4,4,64) is None
    assert table.predict(3,8,64) is None
    assert table.stats==dict(exact=2,interpolated=1,missing_shape=2)


@pytest.mark.parametrize('context',[-1,0,63,129,256,float('inf'),float('nan')])
def test_ood_falls_back_not_clip_or_extrapolate(tmp_path,context):
    table=load(tmp_path)
    assert table.predict(3,4,context) is None


@pytest.mark.parametrize('key,value',[
    ('backend','FLASHINFER'),('vllm_version','0.11.0'),('fa_version',3),('dtype','float16'),
    ('distribution','bimodal'),('measurement','executor_wall_ms'),('cpu_included',True),
    ('cache_write_included',True),('schema_version',2),('max_model_len',512),
    ('context_semantics','visible_seq_len'),
])
def test_profile_identity_mismatch(tmp_path,key,value):
    d=document();d['metadata'][key]=value
    with pytest.raises(ValueError,match='mismatch'):load(tmp_path,d)


@pytest.mark.parametrize('key,value',[
    ('active',0),('active',5),('active',True),('capture',3),('context_tokens',-1),
    ('context_tokens',256),('context_tokens',64.5),('split','heldout'),('sample_count',8),
    ('samples_ms',[0]*9),('samples_ms',[float('nan')]*9),('median_ms',2),('relative_mad',.5),
])
def test_reject_invalid_training_rows(tmp_path,key,value):
    d=document();d['rows'][0][key]=value
    with pytest.raises(ValueError):load(tmp_path,d)


def test_duplicate_empty_and_unstable_rows(tmp_path):
    d=document();d['rows'].append(d['rows'][0])
    with pytest.raises(ValueError,match='Duplicate'):load(tmp_path,d)
    d=document();d['rows']=[]
    with pytest.raises(ValueError,match='Empty'):load(tmp_path,d)
    d=document();d['rows'][0].update(samples_ms=[.5]*4+[1]+[1.5]*4,relative_mad=.5)
    with pytest.raises(ValueError,match='Unstable'):load(tmp_path,d)


def fixture(tmp_path):
    p=object.__new__(Predictor)
    p._full_graph_decode_attention=load(tmp_path)
    p._active_measurement_type=MeasurementType.KERNEL_ONLY
    p._get_batch_decode_attention_params=lambda batch:(4,64)
    p._supports_operation=lambda op:True
    p._predictions={'attn_decode':{(4,64):2.0}}
    p._attention_decode_batching_overhead_fraction=.1
    p._get_late_decode_only_calibration_scale=lambda *args:None
    p._get_calibration_scale=lambda *args:1
    batch=NS(requests=[None]*3,num_prefill_tokens=0,num_decode_tokens=3,
             get_decode_cuda_graph_runtime_mode=lambda:'FULL')
    return p,batch


def test_only_attention_replaced_and_original_uplift_kept(tmp_path):
    p,b=fixture(tmp_path)
    assert p._get_attention_decode_execution_time(b)==pytest.approx(1.1)
    assert p._predictions=={'attn_decode':{(4,64):2.0}}
    p._full_graph_decode_attention=None
    assert p._get_attention_decode_execution_time(b)==pytest.approx(2.2)


@pytest.mark.parametrize('mode',['NONE','PIECEWISE','FULL_AND_PIECEWISE'])
def test_non_full_routing_preserves_legacy(tmp_path,mode):
    p,b=fixture(tmp_path);b.get_decode_cuda_graph_runtime_mode=lambda:mode
    assert p._get_attention_decode_execution_time(b)==pytest.approx(2.2)


@pytest.mark.parametrize('kind',['mixed','spec','eager','missing_active','missing_context'])
def test_unsupported_or_missing_shape_keeps_old_route(tmp_path,kind):
    p,b=fixture(tmp_path)
    if kind=='mixed':b.num_prefill_tokens=64
    if kind=='spec':b.num_decode_tokens=6
    if kind=='eager':p._active_measurement_type=MeasurementType.CUDA_EVENT
    if kind=='missing_active':b.requests=[None]*4;b.num_decode_tokens=4
    context=200 if kind=='missing_context' else 64
    assert p._get_full_graph_decode_attention_time(b,4,context) is None


def test_new_switch_does_not_invalidate_old_rf_identity():
    p=object.__new__(Predictor);p._active_measurement_type=MeasurementType.KERNEL_ONLY
    p.to_dict=lambda:{'base':'same'}
    original=p._get_model_hash('attn_decode')
    p.to_dict=lambda:{'base':'same','full_graph_decode_attention_enabled':True,
                     'full_graph_decode_attention_profile':'different.json',
                     'full_graph_decode_attention_max_model_len':32768}
    assert p._get_model_hash('attn_decode')==original


def test_planning_has_no_heldout_leakage():
    plan=make_plan(32,[1,2,4,8,16,32,64],32768)
    train={(r['active'],r['capture'],r['context_tokens']) for r in plan if r['split']=='train'}
    test={(r['active'],r['capture'],r['context_tokens']) for r in plan if r['split']=='heldout'}
    assert len(train)==288 and len(test)==49 and not train&test
    assert all(r['active']<=r['capture'] and 0<=r['context_tokens']<32768 for r in plan)


@pytest.mark.parametrize('limit',[0,256,32768])
def test_capture_bound_must_match_serving_bound(limit):
    config=NS(sys_arch='co-location',request_generator_config=NS(max_tokens=32768),
        cluster_config=NS(execution_time_predictor_config=NS(full_graph_decode_attention_enabled=True,
            full_graph_decode_attention_profile='profile.json',full_graph_decode_attention_max_model_len=limit)))
    if limit==32768:
        SimulationConfig._validate_full_graph_decode_attention_config(config)
    else:
        with pytest.raises(ValueError,match='matching'):
            SimulationConfig._validate_full_graph_decode_attention_config(config)

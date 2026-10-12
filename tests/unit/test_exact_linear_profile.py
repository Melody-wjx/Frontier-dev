"""Scheme B: exact stable GEMM points, unchanged fallback and cache identities."""
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from frontier.config import RandomForrestExecutionTimePredictorConfig, PrecisionType
from frontier.execution_time_predictor.exact_linear_profile import (
    LINEAR_OPERATIONS, measured_linear_points, overlay_exact_points,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
from frontier.types import ClusterType, MeasurementType


class Predictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        return None


def row(tokens=8, value=2.0, count=20, std=0.02, **extra):
    r = {"num_tokens": tokens}
    for op in LINEAR_OPERATIONS:
        prefix = "time_stats." + op + "."
        r.update({prefix + "median": value, prefix + "mean": value,
                  prefix + "count": count, prefix + "std": std})
    r.update(extra)
    return r


def test_default_off():
    assert RandomForrestExecutionTimePredictorConfig().prefer_profiled_linear is False
    p = object.__new__(Predictor)
    p._config = SimpleNamespace()
    # No input files, GPU or other initialized attributes are needed when off.
    p._apply_profiled_linear_overrides(None)


@pytest.mark.parametrize("op", LINEAR_OPERATIONS)
def test_exact_median_not_rf(op):
    points, audit = measured_linear_points(pd.DataFrame([row()]), op)
    assert points == {(8,): 2.0}
    original = {(8,): 2.7, (9,): 2.8, (800,): 8.0}
    result, changes = overlay_exact_points(original, points)
    assert result == {(8,): 2.0, (9,): 2.8, (800,): 8.0}
    assert original[(8,)] == 2.7
    assert changes[0]["delta_ms"] == pytest.approx(-0.7)
    assert audit["accepted_shapes"] == 1


@pytest.mark.parametrize("kwargs,reason", [
    ({"count": 1}, "insufficient_samples"),
    ({"count": 20.5}, "insufficient_samples"),
    ({"std": 0.21}, "unstable_samples"),
    ({"std": -0.1}, "invalid_statistics"),
    ({"std": float('nan')}, "nonfinite_statistics"),
    ({"value": 0}, "invalid_statistics"),
    ({"value": float('inf')}, "nonfinite_statistics"),
    ({"tokens": 0}, "invalid_shape"),
    ({"tokens": 1.5}, "invalid_shape"),
    ({"tokens": float('nan')}, "invalid_shape"),
])
def test_reject_incomplete_unstable_or_invalid(kwargs, reason):
    kwargs = {"value": 1.0, **kwargs}
    points, audit = measured_linear_points(pd.DataFrame([row(**kwargs)]), "mlp_up_proj")
    assert not points
    assert audit["rejected"][0]["reason"] == reason


def test_missing_summary_does_not_trust_median_alone():
    frame = pd.DataFrame([row()]).drop(columns=['time_stats.mlp_up_proj.std'])
    points, audit = measured_linear_points(frame, "mlp_up_proj")
    assert points == {}
    assert audit['missing_columns'] == ['time_stats.mlp_up_proj.std']


def test_duplicate_aggregation_and_deduplication():
    frame = pd.DataFrame([row(value=2.0), row(value=2.0), row(value=2.1)])
    points, audit = measured_linear_points(frame, "mlp_up_proj")
    assert points[(8,)] == pytest.approx(2.05)
    assert audit['points'][0]['summary_rows'] == 2


@pytest.mark.parametrize("second", [row(value=3.0), row(count=1), row(std=1.0)])
def test_bad_duplicate_rejects_whole_shape(second):
    points, _ = measured_linear_points(pd.DataFrame([row(), second]), "mlp_up_proj")
    assert points == {}


def test_missing_op_row_is_not_an_observation():
    frame = pd.DataFrame([row(), row(**{'time_stats.mlp_up_proj.median': np.nan})])
    points, audit = measured_linear_points(frame, "mlp_up_proj")
    assert points == {(8,): 2.0}
    assert audit['points'][0]['summary_rows'] == 1


def test_never_inserts_unrepresented_or_clipped_keys():
    result, changes = overlay_exact_points({(99,): 7.0}, {(8,): 2.0})
    assert result == {(99,): 7.0}
    assert changes == []


@pytest.mark.parametrize('op', ['attn_decode', 'attn_rope', 'schedule', 'mlp_act'])
def test_scope_excludes_attention_cpu_and_non_gemm(op):
    with pytest.raises(ValueError, match='Unsupported'):
        measured_linear_points(pd.DataFrame([row()]), op)


def predictor(tmp_path):
    p = object.__new__(Predictor)
    p._config = SimpleNamespace(prefer_profiled_linear=True)
    p._enable_dummy_mode = False
    p._model_config = SimpleNamespace(is_moe=False, quantization_config=SimpleNamespace(quant_method=None),
                                     get_default_precision=lambda: PrecisionType.BF16)
    p._cluster_type = ClusterType.MONOLITHIC
    p._replica_config = SimpleNamespace(model_name='fixture', device='a100')
    for family in ['eager', 'kernel_only']:
        path = tmp_path / f'{family}.csv'
        pd.DataFrame([row(value=2.0 if family == 'eager' else 1.0)]).to_csv(path, index=False)
        setattr(p, '_compute_input_file_' + family, str(path))
        setattr(p, '_attention_input_file_' + family, '')
        setattr(p, '_moe_input_file_' + family, '')
        setattr(p, '_models_' + family, {})
        setattr(p, '_predictions_' + family, {'mlp_up_proj': {(8,): 3.0, (9,): 4.0},
                                            'attn_decode': {(8, 16): 5.0}, 'schedule': {(8,): 7.0}})
    p._get_linear_op_tp_key = lambda op: 2
    p._load_compute_df = lambda path, tensor_parallel_size: pd.read_csv(path)
    p._get_profiling_metadata = lambda *_: SimpleNamespace(profiling_precision=PrecisionType.BF16, quant_signature='none')
    p._activate_measurement_type(MeasurementType.KERNEL_ONLY)
    return p


def test_family_separation_unchanged_other_ops_and_owned_tables(tmp_path):
    p = predictor(tmp_path)
    original_eager = p._predictions_eager['mlp_up_proj']
    p._apply_profiled_linear_overrides(SimpleNamespace(output_dir=str(tmp_path)))
    assert original_eager == {(8,): 3.0, (9,): 4.0}
    assert p._predictions_eager['mlp_up_proj'] == {(8,): 2.0, (9,): 4.0}
    assert p._predictions_kernel_only['mlp_up_proj'] == {(8,): 1.0, (9,): 4.0}
    assert p._predictions is p._predictions_kernel_only
    for tables in [p._predictions_eager, p._predictions_kernel_only]:
        assert tables['attn_decode'] == {(8, 16): 5.0}
        assert tables['schedule'] == {(8,): 7.0}
    report = json.loads(next(tmp_path.glob('exact_linear_profile.*.json')).read_text())
    assert set(report['families']) == {'CUDA_EVENT', 'KERNEL_ONLY'}
    assert report['families']['KERNEL_ONLY']['operations']['mlp_up_proj']['overridden_keys'] == 1


@pytest.mark.parametrize('setting', ['dummy', 'moe', 'quantized', 'precision', 'quant_data'])
def test_unsupported_configuration_rejected(tmp_path, setting):
    p = predictor(tmp_path)
    if setting == 'dummy':
        p._enable_dummy_mode = True
    elif setting == 'moe':
        p._model_config.is_moe = True
    elif setting == 'quantized':
        p._model_config.quantization_config.quant_method = 'gptq'
    elif setting == 'precision':
        p._get_profiling_metadata = lambda *_: SimpleNamespace(profiling_precision=None, quant_signature='none')
    else:
        p._get_profiling_metadata = lambda *_: SimpleNamespace(profiling_precision=PrecisionType.BF16, quant_signature='other')
    with pytest.raises(ValueError):
        p._apply_profiled_linear_overrides(SimpleNamespace(output_dir=str(tmp_path)))

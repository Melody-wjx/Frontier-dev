"""Optional mixed-prefill geometry, compatibility, CLI and cache contracts."""
from types import SimpleNamespace

import pytest

from frontier.config import RandomForrestExecutionTimePredictorConfig, SimulationConfig
from frontier.config.flat_dataclass import create_flat_dataclass
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import MeasurementType


class Predictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        raise AssertionError('Unit tests must not train estimators')

    def _get_estimator(self):
        raise AssertionError('Unit tests must not train estimators')


def predictor(enabled=False, table=None):
    p = object.__new__(Predictor)
    p._config = SimpleNamespace(mixed_prefill_work_preserving_kv=enabled,
                               kv_cache_prediction_granularity=64)
    p._attention_prefill_batching_overhead_fraction = 0.1
    p._predictions = {p._dense_attention_prefill_op_name():
                     ({(640, 250000): 10.0, (512, 250000): 4.0} if table is None else table)}
    return p


def batch():
    # Rounded historical KV: 100 -> 128, 500 -> 512. Decode is excluded.
    return SimpleNamespace(requests=[
        SimpleNamespace(_is_prefill_complete=False, num_processed_tokens=100),
        SimpleNamespace(_is_prefill_complete=False, num_processed_tokens=500),
        SimpleNamespace(_is_prefill_complete=True, num_processed_tokens=2000)],
        num_tokens=[300, 400, 1])


def test_default_and_absent_field_keep_original_key():
    assert RandomForrestExecutionTimePredictorConfig().mixed_prefill_work_preserving_kv is False
    p = predictor()
    assert p._get_decomposed_prefill_attention_time(batch()) == 11.0
    del p._config.mixed_prefill_work_preserving_kv
    assert p._get_decomposed_prefill_attention_time(batch()) == 11.0


def test_work_weighting_uses_own_rounded_history_and_keeps_uplift():
    p = predictor(True)
    b = batch()
    assert p._get_batch_prefill_attention_params(b) == [(128, 300), (512, 400)]
    # q_eff=500; history work=243200; effective KV=486.4 rounded up to 512.
    assert p._get_decomposed_prefill_attention_time(b) == pytest.approx(4.4)


def test_toggle_does_not_mutate_shared_table_or_cached_batch_parameters():
    p = predictor()
    b = batch()
    table = p._predictions[p._dense_attention_prefill_op_name()]
    original_table = dict(table)
    assert p._get_decomposed_prefill_attention_time(b) == 11.0
    p._config.mixed_prefill_work_preserving_kv = True
    assert p._get_decomposed_prefill_attention_time(b) == pytest.approx(4.4)
    p._config.mixed_prefill_work_preserving_kv = False
    assert p._get_decomposed_prefill_attention_time(b) == 11.0
    assert table == original_table
    assert b._prefill_params == [(128, 300), (512, 400)]


def test_missing_replacement_key_falls_back_to_original_value():
    p = predictor(True, {(640, 250000): 10.0})
    assert p._get_decomposed_prefill_attention_time(batch()) == 11.0


def test_missing_original_key_keeps_original_zero_even_if_replacement_exists():
    p = predictor(True, {(512, 250000): 4.0})
    assert p._get_decomposed_prefill_attention_time(batch()) == 0.0


def test_missing_operator_and_no_prefill_keep_original_zero():
    p = predictor(True)
    p._predictions = {}
    assert p._get_decomposed_prefill_attention_time(batch()) == 0.0
    assert p._get_decomposed_prefill_attention_time(SimpleNamespace(_prefill_params=[])) == 0.0


def test_single_prefill_keeps_original_key_without_batching_uplift():
    p = predictor(True, {(256, 4096): 10.0, (128, 4096): 4.0})
    assert p._get_decomposed_prefill_attention_time(
        SimpleNamespace(_prefill_params=[(256, 64)])) == 10.0


def test_no_history_preserves_the_same_lookup():
    p = predictor(True, {(0, 250000): 10.0})
    assert p._get_decomposed_prefill_attention_time(
        SimpleNamespace(_prefill_params=[(0, 300), (0, 400)])) == 11.0


def test_zero_query_work_keeps_legacy_result_without_division():
    p = predictor(True, {(640, 0): 10.0})
    assert p._get_decomposed_prefill_attention_time(
        SimpleNamespace(_prefill_params=[(128, 0), (512, 0)])) == 11.0


@pytest.mark.parametrize('family', [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY])
def test_switch_preserves_estimator_and_prediction_cache_identity(family):
    p = predictor()
    p._active_measurement_type = family
    p._prediction_ood_policy = 'clip'
    model = SimpleNamespace(_frontier_model_hash='frozen-profile-estimator')
    p.to_dict = lambda: {'stable_config': 'same'}
    original_model = p._get_model_hash('attn_prefill')
    original_grid = p._get_prediction_cache_hash('attn_prefill', model)
    for enabled in [False, True]:
        p.to_dict = lambda: {'stable_config': 'same', 'mixed_prefill_work_preserving_kv': enabled}
        assert p._get_model_hash('attn_prefill') == original_model
        assert p._get_prediction_cache_hash('attn_prefill', model) == original_grid


@pytest.fixture(scope='module')
def flat_config():
    return create_flat_dataclass(SimulationConfig)


@pytest.mark.parametrize('flags,expected', [
    ([], False),
    (['--random_forrest_execution_time_predictor_config_mixed_prefill_work_preserving_kv'], True),
    (['--no-random_forrest_execution_time_predictor_config_mixed_prefill_work_preserving_kv'], False),
])
def test_native_cli_default_enable_and_disable(monkeypatch, flat_config, flags, expected):
    monkeypatch.setattr('sys.argv', ['frontier.main', *flags])
    parsed = flat_config.create_from_cli_args()
    assert parsed.random_forrest_execution_time_predictor_config_mixed_prefill_work_preserving_kv is expected

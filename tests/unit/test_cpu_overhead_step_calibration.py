"""Tests for closed-loop engine-step CPU-overhead calibration.

The calibration rescales the five named CPU-overhead components
(schedule, sampler, prepare_inputs, process_model_outputs, ray_comm) so their
sum matches the measured non-model engine-step wall residual
(engine_step_wall - model_execution_e2e_wall) recorded in the profile
calibration file, instead of over-counting thread CPU that overlaps GPU
execution.
"""
import json
from types import SimpleNamespace

import pytest

from frontier.config import RandomForrestExecutionTimePredictorConfig
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import ClusterType


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        raise AssertionError("estimator construction is not used in these tests")

    def _get_grid_search_params(self):
        raise AssertionError("grid search is not used in these tests")


def _measurement_record(batch_size, num_prefill, num_decode, residual, envelope):
    return {
        "identity": {
            "shape": {
                "batch_size": batch_size,
                "num_prefill_tokens": num_prefill,
                "num_decode_tokens": num_decode,
            }
        },
        "engine_boundaries_ms": {
            "engine_step_wall": envelope + residual,
            "model_execution_e2e_wall": envelope,
        },
        "components_ms": {"mapper_step_cpu_envelope": envelope},
    }


def _write_calibration_file(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return str(path)


def _build_predictor(tmp_path, enabled, calibration_file=""):
    predictor = object.__new__(_ConcretePredictor)
    predictor._config = SimpleNamespace(
        skip_cpu_overhead_modeling=False,
        cpu_overhead_ray_comm_policy="charge",
        calibration_enabled=enabled,
        calibration_file=calibration_file,
    )
    predictor._replica_config = SimpleNamespace(num_pipeline_stages=1)
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._step_calibration_by_shape = {}
    predictor._step_calibration_loaded = False
    return predictor


_BATCH = SimpleNamespace(
    id=1,
    size=32,
    num_prefill_tokens=0,
    num_decode_tokens=32,
)


def test_config_defaults_disabled() -> None:
    config = RandomForrestExecutionTimePredictorConfig()
    assert config.calibration_enabled is False
    assert config.calibration_file == ""


def test_disabled_calibration_returns_unit_scale(tmp_path) -> None:
    predictor = _build_predictor(tmp_path, enabled=False)
    predictor._load_step_calibration_table()

    assert predictor._step_calibration_loaded is False
    assert predictor._get_step_calibration_scale(_BATCH) == 1.0


def test_enabled_without_file_warns_and_stays_uncalibrated(tmp_path, caplog) -> None:
    predictor = _build_predictor(tmp_path, enabled=True, calibration_file="")
    with caplog.at_level("WARNING"):
        predictor._load_step_calibration_table()

    assert predictor._step_calibration_loaded is False
    assert "no calibration_file" in caplog.text


def test_enabled_with_missing_file_warns(tmp_path, caplog) -> None:
    missing = str(tmp_path / "does_not_exist.jsonl")
    predictor = _build_predictor(tmp_path, enabled=True, calibration_file=missing)
    with caplog.at_level("WARNING"):
        predictor._load_step_calibration_table()

    assert predictor._step_calibration_loaded is False


def test_loads_median_residual_and_computes_scale(tmp_path) -> None:
    records = [
        # shape (32, 0, 32): residual 1.4ms over envelope 2.5ms -> k=0.56
        _measurement_record(32, 0, 32, residual=1.3, envelope=2.5),
        _measurement_record(32, 0, 32, residual=1.5, envelope=2.5),
    ]
    path = _write_calibration_file(tmp_path / "calib.jsonl", records)
    predictor = _build_predictor(tmp_path, enabled=True, calibration_file=path)
    predictor._load_step_calibration_table()

    assert predictor._step_calibration_loaded is True
    assert (32, 0, 32) in predictor._step_calibration_by_shape
    assert predictor._get_step_calibration_scale(_BATCH) == pytest.approx(0.56)


def test_unknown_shape_returns_unit_scale(tmp_path) -> None:
    records = [_measurement_record(8, 0, 8, residual=0.7, envelope=1.7)]
    path = _write_calibration_file(tmp_path / "calib.jsonl", records)
    predictor = _build_predictor(tmp_path, enabled=True, calibration_file=path)
    predictor._load_step_calibration_table()

    # Batch shape (32, 0, 32) is not in the table -> no rescale.
    assert predictor._get_step_calibration_scale(_BATCH) == 1.0


def test_negative_or_zero_residual_skipped(tmp_path) -> None:
    records = [
        _measurement_record(32, 0, 32, residual=-0.1, envelope=2.5),
        _measurement_record(32, 0, 32, residual=0.0, envelope=2.5),
    ]
    path = _write_calibration_file(tmp_path / "calib.jsonl", records)
    predictor = _build_predictor(tmp_path, enabled=True, calibration_file=path)
    predictor._load_step_calibration_table()

    # Both records rejected (negative residual and zero residual), so the
    # shape must be absent and fall back to k=1.
    assert (32, 0, 32) not in predictor._step_calibration_by_shape
    assert predictor._get_step_calibration_scale(_BATCH) == 1.0


def test_scale_is_clipped_to_sane_range(tmp_path) -> None:
    records = [_measurement_record(32, 0, 32, residual=10.0, envelope=1.0)]
    path = _write_calibration_file(tmp_path / "calib.jsonl", records)
    predictor = _build_predictor(tmp_path, enabled=True, calibration_file=path)
    predictor._load_step_calibration_table()

    assert predictor._get_step_calibration_scale(_BATCH) == pytest.approx(2.0)


def test_malformed_file_is_tolerated(tmp_path, caplog) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json}\n", encoding="utf-8")
    predictor = _build_predictor(tmp_path, enabled=True, calibration_file=str(path))
    with caplog.at_level("WARNING"):
        predictor._load_step_calibration_table()

    assert predictor._step_calibration_loaded is False
    assert "Failed to load" in caplog.text

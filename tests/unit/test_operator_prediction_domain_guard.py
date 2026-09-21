from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from frontier.config import RandomForrestExecutionTimePredictorConfig
from frontier.execution_time_predictor.prediction_domain import compute_feature_bounds
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.types import MeasurementType


class _ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        raise AssertionError("estimator construction is not used in these tests")

    def _get_grid_search_params(self):
        raise AssertionError("grid search is not used in these tests")


class _RecordingModel:
    n_features_in_ = 1
    _frontier_feature_names = ["num_tokens"]
    _frontier_feature_bounds = {"num_tokens": (1.0, 4.0)}

    def __init__(self) -> None:
        self.seen: list[float] = []

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        self.seen.extend(float(value) for value in features["num_tokens"])
        return features["num_tokens"].to_numpy(dtype=float)


def _build_on_demand_predictor(policy: str) -> tuple[_ConcretePredictor, _RecordingModel]:
    predictor = object.__new__(_ConcretePredictor)
    predictor._config = SimpleNamespace(prediction_ood_policy=policy)
    predictor._prediction_ood_policy = policy
    predictor._prediction_domain_stats = {}
    predictor._prediction_domain_warning_cache = set()
    predictor._active_measurement_type = None
    predictor._measurement_family_name = lambda _measurement_type: "eager"
    predictor._runtime_cache = {"eager": {"linear": {}}}
    predictor._persisted_on_demand_cache = {}
    predictor._rf_fast_trees = {}

    model = _RecordingModel()
    predictor._predictions = {
        "linear": {
            "_on_demand_prediction": True,
            "_feature_names": ["num_tokens"],
            "_model": model,
        }
    }
    return predictor, model


def test_prediction_ood_policy_defaults_to_clip() -> None:
    config = RandomForrestExecutionTimePredictorConfig()
    assert config.prediction_ood_policy == "clip"


def test_missing_policy_metadata_falls_back_to_clip() -> None:
    predictor, model = _build_on_demand_predictor("clip")
    predictor._config = SimpleNamespace()
    del predictor._prediction_ood_policy

    prediction = predictor._get_on_demand_prediction(
        "linear", {"num_tokens": 8.0}
    )

    assert prediction == pytest.approx(4.0)
    assert model.seen == [4.0]


def test_prediction_ood_policy_rejects_unknown_value() -> None:
    predictor = object.__new__(_ConcretePredictor)
    predictor._config = SimpleNamespace(prediction_ood_policy="extrapolate")

    with pytest.raises(ValueError, match="prediction_ood_policy must be one of"):
        predictor._resolve_prediction_ood_policy()


def test_on_demand_prediction_fails_before_using_cached_ood_value() -> None:
    predictor, model = _build_on_demand_predictor("error")
    predictor._runtime_cache["eager"]["linear"][(8.0,)] = 123.0

    with pytest.raises(
        ValueError,
        match=r"outside the profiling domain.*num_tokens: query=\[8, 8\].*profile=\[1, 4\]",
    ):
        predictor._get_on_demand_prediction("linear", {"num_tokens": 8.0})

    assert model.seen == []


def test_on_demand_clip_uses_boundary_and_records_coverage() -> None:
    predictor, model = _build_on_demand_predictor("clip")

    prediction = predictor._get_on_demand_prediction(
        "linear", {"num_tokens": 8.0}
    )

    assert prediction == pytest.approx(4.0)
    assert model.seen == [4.0]
    stats = predictor.get_prediction_domain_stats()["eager:linear:runtime"]
    assert stats["total_queries"] == 1
    assert stats["out_of_domain_queries"] == 1
    assert stats["clipped_queries"] == 1
    assert stats["feature_violations"]["num_tokens"]["above_count"] == 1


def test_precomputed_prediction_fails_on_ood_grid() -> None:
    predictor, model = _build_on_demand_predictor("error")
    predictor._models = {"linear": model}

    with pytest.raises(ValueError, match="source=precompute"):
        predictor._get_model_prediction(
            "linear",
            model,
            pd.DataFrame({"num_tokens": [1.0, 8.0]}),
        )

    assert model.seen == []


def test_precomputed_clip_keeps_original_lookup_keys(tmp_path) -> None:
    predictor, model = _build_on_demand_predictor("clip")
    predictor._models = {"linear": model}
    predictor._cache_dir = str(tmp_path)
    predictor._get_prediction_cache_hash = lambda _name, _model: "hash"
    predictor._load_model_predication_cache = lambda _name, _hash: None
    predictor._store_model_predication_cache = lambda _name, _hash, _values: None

    predictions = predictor._get_model_prediction(
        "linear",
        model,
        pd.DataFrame({"num_tokens": [1.0, 8.0]}),
    )

    assert predictions == {(1.0,): 1.0, (8.0,): 4.0}
    assert model.seen == [1.0, 4.0]


def test_training_domain_rejects_non_finite_profile_features() -> None:
    profile = pd.DataFrame({"num_tokens": [1.0, np.inf]})

    with pytest.raises(ValueError, match="contains non-finite values"):
        compute_feature_bounds(
            profile,
            ["num_tokens"],
            context="test operator",
        )


def test_train_model_passes_profile_bounds_to_lazy_cached_model() -> None:
    predictor = object.__new__(_ConcretePredictor)
    predictor._get_model_hash = lambda _name, _df: "model-hash"
    captured: dict[str, object] = {}
    cached_model = SimpleNamespace()

    def _load_cached(model_name, model_hash, feature_cols, feature_bounds):
        captured.update(
            model_name=model_name,
            model_hash=model_hash,
            feature_cols=feature_cols,
            feature_bounds=feature_bounds,
        )
        return cached_model

    predictor._load_model_from_cache_lazy = _load_cached
    result = predictor._train_model(
        "linear",
        pd.DataFrame(
            {
                "num_tokens": [1.0, 4.0],
                "time_stats.linear.median": [0.1, 0.4],
            }
        ),
        ["num_tokens"],
        "time_stats.linear.median",
    )

    assert result is cached_model
    assert captured["feature_bounds"] == {"num_tokens": (1.0, 4.0)}


def test_shared_model_manager_restores_bounds_on_cached_model() -> None:
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._get_profiling_precision_from_df = lambda _df: "BF16"
    manager._validate_active_measurement_type = (
        lambda _df: MeasurementType.CUDA_EVENT
    )
    manager._get_model_hash = lambda *_args: "model-hash"
    cached_model = SimpleNamespace()
    manager._load_model_from_cache = lambda _name, _hash: cached_model
    manager._store_model_precision = lambda *_args: None

    result = manager._train_single_model(
        model_name="linear",
        df=pd.DataFrame(
            {
                "num_tokens": [1.0, 4.0],
                "time_stats.linear.median": [0.1, 0.4],
            }
        ),
        feature_cols=["num_tokens"],
        target_col="time_stats.linear.median",
        execution_time_predictor_config=SimpleNamespace(),
    )

    assert result is cached_model
    assert cached_model._frontier_feature_names == ["num_tokens"]
    assert cached_model._frontier_feature_bounds == {"num_tokens": (1.0, 4.0)}

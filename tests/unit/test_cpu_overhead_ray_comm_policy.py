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


def _write_cpu_overhead_csv(
    path,
    *,
    v2_schema: bool,
    scheduling_mode: str = "sync",
    cpu_overhead_source: str = "vllm_replay",
    omit_identity_column: str | None = None,
) -> str:
    if v2_schema:
        fields = {
            "model_name": "test-model",
            "batch_size": "8",
            "tensor_parallel_degree": "1",
            "num_prefill_tokens": "0",
            "num_decode_tokens": "8",
            "scheduling_mode": scheduling_mode,
            "cpu_overhead_source": cpu_overhead_source,
            "ray_comm_time_mean": "0.4",
            "profiling_precision": "BF16",
        }
        if omit_identity_column is not None:
            fields.pop(omit_identity_column)
        path.write_text(",".join(fields) + "\n" + ",".join(fields.values()) + "\n")
    else:
        # Legacy schema-v1 (sarathi) CSV: no scheduling_mode /
        # cpu_overhead_source identity columns.
        path.write_text(
            "model_name,batch_size,tensor_parallel_degree,"
            "ray_comm_time_mean,profiling_precision\n"
            "test-model,8,1,0.4,BF16\n"
        )
    return str(path)


def _build_predictor(
    tmp_path,
    policy: str,
    *,
    num_pipeline_stages: int = 1,
    cluster_type=ClusterType.MONOLITHIC,
    v2_schema: bool = True,
    scheduling_mode: str = "sync",
    cpu_overhead_source: str = "vllm_replay",
    omit_identity_column: str | None = None,
) -> _ConcretePredictor:
    predictor = object.__new__(_ConcretePredictor)
    predictor._config = SimpleNamespace(
        skip_cpu_overhead_modeling=False,
        cpu_overhead_ray_comm_policy=policy,
    )
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=num_pipeline_stages
    )
    predictor._cluster_type = cluster_type
    predictor._cpu_overhead_input_file = _write_cpu_overhead_csv(
        tmp_path / f"cpu_overhead_{'v2' if v2_schema else 'v1'}.csv",
        v2_schema=v2_schema,
        scheduling_mode=scheduling_mode,
        cpu_overhead_source=cpu_overhead_source,
        omit_identity_column=omit_identity_column,
    )
    predictor._predictions = {
        "ray_comm_time": {(8,): 0.4},
        "schedule": {(8,): 0.5},
    }
    return predictor


_BATCH = SimpleNamespace(id=1, size=8)


def test_cpu_overhead_ray_comm_policy_defaults_to_charge() -> None:
    config = RandomForrestExecutionTimePredictorConfig()
    assert config.cpu_overhead_ray_comm_policy == "charge"


def test_cpu_overhead_ray_comm_policy_rejects_unknown_value() -> None:
    predictor = object.__new__(_ConcretePredictor)
    predictor._config = SimpleNamespace(cpu_overhead_ray_comm_policy="bill")

    with pytest.raises(
        ValueError, match="cpu_overhead_ray_comm_policy must be one of"
    ):
        predictor._resolve_cpu_overhead_ray_comm_policy()


def test_auto_charges_vllm_replay_residual_for_pp1_monolithic(tmp_path) -> None:
    predictor = _build_predictor(tmp_path, "auto")

    assert predictor._get_ray_comm_time(_BATCH) == pytest.approx(0.4)
    # Other CPU overhead components are unaffected by the policy.
    assert predictor._get_schedule_time(_BATCH) == pytest.approx(0.5)


def test_auto_charges_ray_comm_when_pp_greater_than_1(tmp_path) -> None:
    predictor = _build_predictor(tmp_path, "auto", num_pipeline_stages=2)

    assert predictor._get_ray_comm_time(_BATCH) == pytest.approx(0.4)


def test_auto_charges_ray_comm_for_non_monolithic_cluster(tmp_path) -> None:
    predictor = _build_predictor(
        tmp_path, "auto", cluster_type=ClusterType.PREFILL
    )

    assert predictor._get_ray_comm_time(_BATCH) == pytest.approx(0.4)


def test_auto_charges_ray_comm_for_legacy_schema_v1_csv(tmp_path) -> None:
    predictor = _build_predictor(tmp_path, "auto", v2_schema=False)

    assert predictor._get_ray_comm_time(_BATCH) == pytest.approx(0.4)


def test_auto_charges_ray_comm_for_async_vllm_replay(tmp_path) -> None:
    predictor = _build_predictor(tmp_path, "auto", scheduling_mode="async")

    assert predictor._get_ray_comm_time(_BATCH) == pytest.approx(0.4)


def test_auto_charges_ray_comm_for_native_measured_profile(tmp_path) -> None:
    predictor = _build_predictor(
        tmp_path, "auto", cpu_overhead_source="measured"
    )

    assert predictor._get_ray_comm_time(_BATCH) == pytest.approx(0.4)


@pytest.mark.parametrize(
    "missing_column", ["scheduling_mode", "cpu_overhead_source"]
)
def test_auto_charges_when_identity_column_is_missing(tmp_path, missing_column) -> None:
    predictor = _build_predictor(
        tmp_path, "auto", omit_identity_column=missing_column
    )

    assert predictor._get_ray_comm_time(_BATCH) == pytest.approx(0.4)


def test_charge_policy_bills_ray_comm_everywhere(tmp_path) -> None:
    predictor = _build_predictor(tmp_path, "charge")

    assert predictor._get_ray_comm_time(_BATCH) == pytest.approx(0.4)


def test_zero_policy_zeroes_ray_comm_everywhere(tmp_path) -> None:
    predictor = _build_predictor(
        tmp_path, "zero", num_pipeline_stages=2, v2_schema=False
    )

    assert predictor._get_ray_comm_time(_BATCH) == 0.0


def test_auto_conservative_charge_logs_once(tmp_path, caplog) -> None:
    predictor = _build_predictor(tmp_path, "auto")

    with caplog.at_level("INFO"):
        predictor._get_ray_comm_time(_BATCH)
        predictor._get_ray_comm_time(_BATCH)

    matches = [
        record
        for record in caplog.records
        if "cpu_overhead_ray_comm_policy=auto" in record.getMessage()
    ]
    assert len(matches) == 1
    assert "charging ray_comm_time conservatively" in matches[0].getMessage()


def _build_cache_key_predictor(policy: str) -> _ConcretePredictor:
    predictor = object.__new__(_ConcretePredictor)
    predictor._config = SimpleNamespace(cpu_overhead_ray_comm_policy=policy)
    predictor._cpu_overhead_ray_comm_policy = policy
    predictor._prediction_ood_policy = "clip"
    predictor._get_model_hash = lambda _name, df=None: "config-hash"
    return predictor


def test_policy_does_not_enter_estimator_model_hash() -> None:
    def build(policy: str, zeroed: bool) -> _ConcretePredictor:
        predictor = object.__new__(_ConcretePredictor)
        predictor._active_measurement_type = SimpleNamespace(value="CUDA_EVENT")
        predictor.to_dict = lambda: {
            "stable_training_config": "same",
            "prediction_ood_policy": "clip",
            "cpu_overhead_ray_comm_policy": policy,
            "ray_comm_time_zeroed": zeroed,
        }
        return predictor

    assert build("charge", False)._get_model_hash("ray_comm_time") == build(
        "zero", True
    )._get_model_hash("ray_comm_time")


def test_policy_does_not_enter_precomputed_prediction_cache_key() -> None:
    model = SimpleNamespace(_frontier_model_hash="model-hash")
    auto = _build_cache_key_predictor("auto")
    charge = _build_cache_key_predictor("charge")

    auto_hash = auto._get_prediction_cache_hash("linear", model)
    charge_hash = charge._get_prediction_cache_hash("linear", model)

    assert auto_hash == charge_hash
    assert auto_hash == auto._get_prediction_cache_hash("linear", model)


def test_policy_does_not_enter_persisted_on_demand_cache_key() -> None:
    model = SimpleNamespace(_frontier_model_hash="model-hash")
    predictor = _build_cache_key_predictor("charge")
    predictor._persisted_on_demand_cache = {}

    predictor._persist_on_demand_value(
        "eager", "ray_comm_time", model, (8.0, 0.0, 8.0), 0.4
    )

    assert list(predictor._persisted_on_demand_cache) == [
        ("eager", "ray_comm_time", "model-hash", "clip")
    ]
    assert (
        predictor._lookup_persisted_on_demand(
            "eager", "ray_comm_time", model, (8.0, 0.0, 8.0)
        )
        == pytest.approx(0.4)
    )

    # Billing happens after prediction, so another policy reuses the value.
    predictor._cpu_overhead_ray_comm_policy = "auto"
    assert (
        predictor._lookup_persisted_on_demand(
            "eager", "ray_comm_time", model, (8.0, 0.0, 8.0)
        )
        == pytest.approx(0.4)
    )

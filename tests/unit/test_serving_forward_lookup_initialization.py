"""CPU-only lookup initialization must not change the native timing model."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from frontier.config import RandomForrestExecutionTimePredictorConfig
from frontier.execution_time_predictor import sklearn_execution_time_predictor as module
from frontier.types import ClusterType, MeasurementType


class Predictor(module.SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise AssertionError("This unit test must not train an estimator")


def bare_predictor():
    return Predictor.__new__(Predictor)


@pytest.mark.parametrize("loaded", [False, True])
@pytest.mark.parametrize("shared", [False, True])
def test_only_validated_independent_forward_uses_cpu_only_lookups(loaded, shared):
    predictor = bare_predictor()
    predictor._serving_forward = object() if loaded else None
    predictor._model_manager = object() if shared else None
    assert predictor._uses_serving_forward_cpu_only_lookups() is (loaded and not shared)


@pytest.mark.parametrize("loaded,shared", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("skip_cpu", [False, True])
def test_lookup_builders_preserve_cpu_and_legacy_routing(loaded, shared, skip_cpu):
    predictor = bare_predictor()
    predictor._serving_forward = object() if loaded else None
    predictor._model_manager = object() if shared else None
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(skip_cpu_overhead_modeling=skip_cpu)
    calls = []

    def compute():
        assert not (loaded and not shared), "Unused GPU lookup was built"
        calls.append("compute")
        return {"gpu": 1.0}

    def attention():
        assert not (loaded and not shared), "Unused attention lookup was built"
        calls.append("attention")
        return {"attention": 2.0}

    def cpu():
        calls.append("cpu")
        return {} if skip_cpu else {"schedule": 3.0}

    predictor._predict_for_compute_models = compute
    predictor._predict_for_attention_layer_models = attention
    predictor._predict_for_cpu_overhead_models = cpu
    actual = predictor._predict_from_models()
    if loaded and not shared:
        assert calls == ["cpu"]
        assert actual == ({} if skip_cpu else {"schedule": 3.0})
    else:
        assert calls == (["compute", "attention"] if skip_cpu else ["compute", "cpu", "attention"])
        assert actual == ({"gpu": 1.0, "attention": 2.0} if skip_cpu else
                          {"gpu": 1.0, "schedule": 3.0, "attention": 2.0})


def initialization_spies(monkeypatch, tmp_path, *, enabled=True, dummy=False,
                         shared=False, invalid=False, skip_cpu=False):
    calls = []
    config = RandomForrestExecutionTimePredictorConfig(
        serving_forward_enabled=enabled, enable_dummy_mode=dummy,
        skip_cpu_overhead_modeling=skip_cpu, no_cache=False,
    )
    model = SimpleNamespace(num_layers=28, num_q_heads=32, num_kv_heads=8)
    replica = SimpleNamespace(model_config=model, num_pipeline_stages=1,
                              attn_tensor_parallel_size=2,
                              node_config=SimpleNamespace(num_devices_per_node=8))
    scheduler = SimpleNamespace(get_type=lambda: "vllm_v1", block_size=16,
                                max_tokens_in_batch=8192)
    metrics = SimpleNamespace(cache_dir=str(tmp_path), output_dir=str(tmp_path))

    def initialize_paths(self, _):
        for kind in ("compute", "attention", "moe"):
            for family in ("eager", "kernel_only"):
                setattr(self, f"_{kind}_input_file_{family}", f"{family}-{kind}")

    monkeypatch.setattr(Predictor, "_initialize_file_paths", initialize_paths)
    for name in ("_initialize_pp_stage_boundary_lookup", "_initialize_pp_receiver_head_lookup",
                 "_initialize_pp_producer_send_path_lookup", "_initialize_pp_prefill_consumer_active_lookup",
                 "_log_quantization_details"):
        monkeypatch.setattr(Predictor, name, lambda self: None)
    monkeypatch.setattr(module, "get_quantization_manager", lambda: SimpleNamespace(
        configure_from_model_config=lambda _: None))
    monkeypatch.setattr(Predictor, "_get_calibration_scale", lambda *args: 1.0)
    monkeypatch.setattr(Predictor, "_get_operator_calibration_scale", lambda *args: 1.0)
    monkeypatch.setattr(Predictor, "_get_default_measurement_type_for_cluster",
                        lambda self: MeasurementType.CUDA_EVENT)
    monkeypatch.setattr(Predictor, "_should_enable_measurement_family", lambda *args: True)

    def train(self, family):
        calls.append(("train_and_validate", family))
        return {"schedule": object(), "gpu": object()}

    def load(self, _):
        calls.append(("load_forward", None))
        if invalid:
            raise ValueError("invalid execution contract")
        self._serving_forward = object() if enabled and not dummy else None

    def cpu(self):
        calls.append(("cpu_lookup", self._active_measurement_type))
        return {} if skip_cpu else {"schedule": {(1,): 3.0}}

    def compute(self):
        calls.append(("gpu_lookup", self._active_measurement_type))
        return {"gpu": {(1,): 10.0}}

    monkeypatch.setattr(Predictor, "_train_models_for_family", train)
    monkeypatch.setattr(Predictor, "_load_serving_forward", load)
    monkeypatch.setattr(Predictor, "_predict_for_cpu_overhead_models", cpu)
    monkeypatch.setattr(Predictor, "_predict_for_compute_models", compute)
    monkeypatch.setattr(Predictor, "_predict_for_attention_layer_models", lambda self: {})
    for name in ("_load_persisted_on_demand_cache", "_load_step_calibration_table",
                 "_register_profiling_metadata_from_files"):
        monkeypatch.setattr(Predictor, name, lambda self, name=name: calls.append((name, None)))
    for name in ("_apply_profiled_linear_overrides", "_load_full_graph_decode_attention",
                 "_load_compiled_decode_activation"):
        monkeypatch.setattr(Predictor, name, lambda self, _, name=name: calls.append((name, None)))
    monkeypatch.setattr(module.atexit, "register", lambda *args: None)
    manager = None
    if shared:
        manager = SimpleNamespace(get_models_for_cluster=lambda _: {
            "eager": {"schedule": object(), "gpu": object()},
            "kernel_only": {"schedule": object(), "gpu": object()},
        })
    return calls, lambda: Predictor(config, replica, scheduler, metrics,
                                    model_manager=manager, cluster_type=ClusterType.MONOLITHIC)


@pytest.mark.parametrize("skip_cpu", [False, True])
def test_initialization_keeps_training_validation_before_profile_and_lookups(monkeypatch, tmp_path, skip_cpu):
    calls, create = initialization_spies(monkeypatch, tmp_path, skip_cpu=skip_cpu)
    predictor = create()
    assert calls[:3] == [("train_and_validate", MeasurementType.CUDA_EVENT),
                         ("train_and_validate", MeasurementType.KERNEL_ONLY),
                         ("load_forward", None)]
    assert [item for item in calls if item[0] == "cpu_lookup"] == [
        ("cpu_lookup", MeasurementType.CUDA_EVENT),
        ("cpu_lookup", MeasurementType.KERNEL_ONLY),
    ]
    assert not any(item[0] == "gpu_lookup" for item in calls)
    assert ("_load_persisted_on_demand_cache", None) in calls
    assert ("_load_step_calibration_table", None) in calls
    assert "gpu" in predictor._models_eager and "gpu" in predictor._models_kernel_only
    assert predictor._predictions_eager == ({} if skip_cpu else {"schedule": {(1,): 3.0}})


def test_invalid_profile_cannot_skip_into_lookup_generation(monkeypatch, tmp_path):
    calls, create = initialization_spies(monkeypatch, tmp_path, invalid=True)
    with pytest.raises(ValueError, match="invalid execution contract"):
        create()
    assert [name for name, _ in calls] == ["train_and_validate", "train_and_validate", "load_forward"]


@pytest.mark.parametrize("enabled,dummy,shared", [(False, False, False), (True, True, False),
                                                 (False, False, True), (True, False, True)])
def test_legacy_dummy_and_manager_initialization_remain_unpruned(monkeypatch, tmp_path,
                                                               enabled, dummy, shared):
    calls, create = initialization_spies(monkeypatch, tmp_path, enabled=enabled,
                                        dummy=dummy, shared=shared)
    predictor = create()
    if dummy:
        assert not any(name in {"train_and_validate", "gpu_lookup", "cpu_lookup"} for name, _ in calls)
        assert predictor._serving_forward is None
    else:
        assert len([name for name, _ in calls if name == "gpu_lookup"]) == 2
        if shared:
            assert calls[-1] == ("load_forward", None)
            assert ("_register_profiling_metadata_from_files", None) in calls


@pytest.mark.parametrize("query,history", [([1, 1], [256, 512]), ([32], [0]), ([32, 1], [128, 256])])
@pytest.mark.parametrize("skip_cpu", [False, True])
@pytest.mark.parametrize("zero_ray", [False, True])
@pytest.mark.parametrize("calibrated", [False, True])
def test_stage_numeric_parity_with_and_without_legacy_gpu_lookups(monkeypatch, query, history,
                                                                skip_cpu, zero_ray, calibrated):
    monkeypatch.setattr(module.global_vars, "get_cudagraph_capture_sizes", lambda: [1, 2, 4, 8, 16, 32, 64])
    monkeypatch.setattr(module.global_vars, "get_sys_arch", lambda: "co-location")
    predictor = bare_predictor()
    predictor._model_manager = None
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(skip_cpu_overhead_modeling=skip_cpu)
    predictor._model_config = SimpleNamespace(num_layers=28)
    predictor._serving_forward = SimpleNamespace(predict=lambda shape: 12.5)
    predictor._models_eager = predictor._models_kernel_only = {}
    predictor._predictions_eager = {} if skip_cpu else {
        name: {(len(query),): value} for name, value in
        [("schedule", 1.0), ("sampler_e2e", 2.0), ("prepare_inputs_e2e", 3.0),
         ("process_model_outputs", 4.0), ("ray_comm_time", 5.0)]
    }
    predictor._predictions_kernel_only = deepcopy(predictor._predictions_eager)
    for kind in ("compute", "attention", "moe"):
        for family in ("eager", "kernel_only"):
            setattr(predictor, f"_{kind}_input_file_{family}", f"{family}-{kind}")
    predictor._should_zero_ray_comm_time = lambda: zero_ray
    predictor._step_calibration_enabled = lambda: calibrated
    predictor._get_step_calibration_scale = lambda _: 0.5
    predictor._validate_prediction_value = lambda value, *args: value
    family = MeasurementType.KERNEL_ONLY if all(value == 1 for value in query) else MeasurementType.CUDA_EVENT
    predictor._select_measurement_type_for_batch = lambda _: family
    batch = SimpleNamespace(id=1, size=len(query), num_tokens=query,
        requests=[SimpleNamespace(num_processed_tokens=h + (q == 1), is_prefill_complete=q == 1)
                  for q, h in zip(query, history)])
    fast = predictor.predict_stage_execution_time(batch, 0, ClusterType.MONOLITHIC, 28)

    # Re-create the previous entry condition: legacy GPU tables make each
    # family nonempty, but their values must never contribute to Forward.
    predictor._uses_serving_forward_cpu_only_lookups = lambda: False
    predictor._predictions_eager["unused_gpu"] = {(1,): 999999.0}
    predictor._predictions_kernel_only["unused_gpu"] = {(1,): 999999.0}
    baseline = predictor.predict_stage_execution_time(batch, 0, ClusterType.MONOLITHIC, 28)
    assert {key: value for key, value in vars(fast).items() if key != "_id"} == {
        key: value for key, value in vars(baseline).items() if key != "_id"
    }
    cpu_ms = 0.0 if skip_cpu else (10.0 if zero_ray else 15.0) * (0.5 if calibrated else 1.0)
    assert fast.total_time == pytest.approx((12.5 + cpu_ms) / 1000.0)
    assert predictor._active_measurement_type == family


@pytest.mark.parametrize("shared", [False, True])
def test_empty_predictions_still_rejected_outside_cpu_only_forward(shared):
    predictor = bare_predictor()
    predictor._model_manager = object() if shared else None
    predictor._serving_forward = object() if shared else None
    predictor._enable_dummy_mode = False
    predictor._select_measurement_type_for_batch = lambda _: MeasurementType.CUDA_EVENT
    predictor._predictions_eager = predictor._predictions_kernel_only = {}
    predictor._cluster_type = ClusterType.MONOLITHIC
    batch = SimpleNamespace(id=1, size=1, num_tokens=[1])
    with pytest.raises(ValueError, match="No eager predictor family"):
        predictor.predict_stage_execution_time(batch, 0, ClusterType.MONOLITHIC, 28)


def test_exact_linear_override_is_explicitly_inactive_without_reading_old_csv(tmp_path, monkeypatch):
    predictor = bare_predictor()
    predictor._serving_forward = object()
    predictor._model_manager = None
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._config = SimpleNamespace(prefer_profiled_linear=True)
    predictor._model_config = SimpleNamespace(is_moe=False, quantization_config=None)
    predictor._replica_config = SimpleNamespace(model_name="test", device="test_gpu")
    predictor._predictions_eager = predictor._predictions_kernel_only = {"schedule": {(1,): 1.0}}
    predictor._compute_input_file = "/intentionally/missing/legacy.csv"
    monkeypatch.setattr(predictor, "_load_compute_df", lambda *args, **kwargs: pytest.fail("unused CSV was read"))
    logs = []
    monkeypatch.setattr(module.logger, "info", lambda *args: logs.append(args[0]))
    predictor._apply_profiled_linear_overrides(SimpleNamespace(output_dir=str(tmp_path)))
    report = json.loads(next(tmp_path.glob("exact_linear_profile.*.json")).read_text())
    assert report["enabled"] is True and report["effective"] is False
    assert report["status"] == "superseded_by_whole_forward" and report["families"] == {}
    assert any("inactive" in message for message in logs)
    assert not any("Applied" in message for message in logs)


def test_family_models_and_persistent_cache_are_not_filtered(monkeypatch):
    predictor = bare_predictor()
    predictor._serving_forward = object()
    predictor._model_manager = None
    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    predictor._config = SimpleNamespace(skip_cpu_overhead_modeling=False, prediction_max_batch_size=8)
    predictor._models_eager = {}
    predictor._models_kernel_only = {}
    for family, models in [("eager", predictor._models_eager), ("kernel_only", predictor._models_kernel_only)]:
        models["schedule"] = SimpleNamespace(n_features_in_=3,
            _frontier_feature_names=["batch_size", "num_prefill_tokens", "num_decode_tokens"],
            _frontier_model_hash=family, _frontier_exact_lookup={(1., 0., 1.): 1.0})
        models["unused_gpu"] = object()
        for kind in ("compute", "attention", "moe"):
            setattr(predictor, f"_{kind}_input_file_{family}", f"{family}-{kind}")
    predictor._predictions_eager = predictor._predictions_kernel_only = {}
    predictor._models = predictor._models_eager
    predictor._predictions = {}
    predictor._persisted_on_demand_cache = {
        ("eager", "unused_gpu", "hash", "clip"): {(1,): 42.0},
        ("kernel_only", "schedule", "hash", "clip"): {(1, 0, 1): 3.0},
    }
    saved_cache = deepcopy(predictor._persisted_on_demand_cache)
    eager = predictor._predict_from_models_for_family(MeasurementType.CUDA_EVENT, predictor._models_eager)
    kernel = predictor._predict_from_models_for_family(MeasurementType.KERNEL_ONLY, predictor._models_kernel_only)
    assert eager["schedule"]["_model"] is predictor._models_eager["schedule"]
    assert kernel["schedule"]["_model"] is predictor._models_kernel_only["schedule"]
    assert eager["schedule"]["_model"]._frontier_model_hash != kernel["schedule"]["_model"]._frontier_model_hash
    assert predictor._persisted_on_demand_cache == saved_cache
    assert "unused_gpu" in predictor._models_eager and "unused_gpu" in predictor._models_kernel_only

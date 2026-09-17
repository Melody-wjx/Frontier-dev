"""Legacy paths stay opt-in; unified FlashAttention costs are not doubled."""
from types import SimpleNamespace
import pytest

from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
from frontier.cc_backend.backends.vidur_cc_backend import VidurCCBackend
from frontier.profiling.attention.backends import AttentionBackend, set_attention_backend


class ConcretePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        raise AssertionError("not used")

    def _get_grid_search_params(self):
        raise AssertionError("not used")


@pytest.mark.parametrize("backend,expected", [("FLASH_ATTN", True), ("AttentionBackend.FLASH_ATTN", True),
    ("FLASHINFER", False), ("AttentionBackend.FLASHINFER", False), ("", False)])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
def test_profile_semantics_are_data_selected(tmp_path, backend, expected, batch_size):
    profile = tmp_path / "attention.csv"
    profile.write_text(f"attention_backend,batch_size\n{backend},{batch_size}\n")
    predictor = object.__new__(ConcretePredictor)
    predictor._attention_input_file = str(profile)
    assert predictor._uses_unified_attention_profile() == expected
    assert predictor._uses_unified_attention_profile() == expected  # cached model load too


@pytest.mark.parametrize("size", [2, 4, 8, 16, 32, 64])
def test_unified_mixed_prefill_is_not_charged_twice(size):
    predictor = object.__new__(ConcretePredictor)
    predictor._uses_unified_attention_profile = lambda: True
    assert predictor._get_attention_prefill_execution_time(
        SimpleNamespace(num_prefill_tokens=256, num_decode_tokens=size - 1)) == 0


@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 64])
def test_legacy_collectives_keep_exact_path(mode, size):
    backend = object.__new__(VidurCCBackend)
    backend._models = {}
    calls = []
    def predict(**kwargs):
        calls.append(kwargs)
        return 123.456
    backend.predict_allreduce = predict
    assert backend.predict_allreduce_for_execution_mode(
        execution_mode=mode, data_size_bytes=size * 7168, num_devices=2) == 123.456
    assert calls == [{"data_size_bytes": size * 7168, "num_devices": 2}]


@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("cached", [False, True])
def test_mode_tagged_collective_graph_selection(size, cached):
    backend = object.__new__(VidurCCBackend)
    backend._models = {"all_reduce_graph": object()}
    tokens = size * 3584
    backend._predictions = {"all_reduce_graph": {(tokens,): .125} if cached else {}}
    extensions = []
    def extension(name, count):
        extensions.append((name, count))
        return .125
    backend._predict_out_of_grid = extension
    backend.predict_allreduce = lambda **kwargs: .75
    assert backend.predict_allreduce_for_execution_mode(
        execution_mode="graph", data_size_bytes=tokens * 2, num_devices=2) == .125
    assert extensions == ([] if cached else [("all_reduce_graph", tokens)])
    assert backend.predict_allreduce_for_execution_mode(
        execution_mode="eager", data_size_bytes=tokens * 2, num_devices=2) == .75


def test_backend_enum_is_explicit():
    assert AttentionBackend.FLASH_ATTN.value == "FLASH_ATTN"
    set_attention_backend("FLASH_ATTN")
    set_attention_backend("NO_OP")


@pytest.mark.parametrize("runtime_mode,expected", [("FULL", "graph"), ("NONE", "eager"),
    ("PIECEWISE", "eager")])
@pytest.mark.parametrize("size", [1, 2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("tagged", [False, True])
def test_live_communication_operator_selects_measured_mode(runtime_mode, expected, size, tagged):
    predictor = object.__new__(ConcretePredictor)
    predictor._model_config = SimpleNamespace()
    predictor._replica_config = SimpleNamespace()
    predictor._cluster_type = "test_cluster"
    calls = []
    def predict(**kwargs):
        calls.append(kwargs)
        return 1.25
    predictor._cc_backend = SimpleNamespace(predict_allreduce_for_execution_mode=predict) if tagged else None
    predictor.predict_allreduce_time = predict
    batch = SimpleNamespace(get_decode_cuda_graph_runtime_mode=lambda: runtime_mode)
    operator = SimpleNamespace(name="attn_tensor_parallel_allreduce", collective_alias="allreduce",
        comm_domain="ATTN_TP", apply_allreduce_launch_overhead_strip=False,
        build_payload_bytes=lambda ctx: size * 7168, num_devices=lambda ctx: 2)
    assert predictor._predict_comm_operator(operator, batch) == 1.25
    expected_call = dict(data_size_bytes=size * 7168, num_devices=2,
        cluster_type="test_cluster", comm_domain="ATTN_TP")
    if tagged:
        expected_call["execution_mode"] = expected
    assert calls == [expected_call]

"""Algorithm tests use synthetic durations, not fitted vLLM phase timings."""
from dataclasses import replace
import math

import pytest

from frontier.execution_time_predictor.step_dependency import StepDependencyModel, StepPhase


def example():
    # Host submits Forward, then does independent CPU work. It must wait for
    # GPU sampling before bookkeeping; update cannot overlap the executor.
    return [
        StepPhase("schedule", "controller", (), "exclusive_host_service", 1., ("schedule",)),
        StepPhase("prepare", "worker", ("schedule",), "exclusive_host_service", 2., ("prepare",)),
        StepPhase("forward", "gpu", ("prepare",), "predicted_forward", None, ("forward",)),
        StepPhase("host_tail", "worker", ("prepare",), "exclusive_host_service", 3., ("host_tail",)),
        StepPhase("sample", "gpu", ("forward", "host_tail"), "device_service", 1., ("sample",)),
        StepPhase("wait", "worker", ("sample",), "dependency_fence", 0., ()),
        StepPhase("bookkeep", "worker", (), "exclusive_host_service", 1., ("bookkeep",)),
        StepPhase("update", "controller", ("bookkeep",), "exclusive_host_service", 1., ("update",)),
    ]


def test_overlap_depends_on_forward_and_preserves_outer_cpu():
    model = StepDependencyModel(example())
    result = model.evaluate(5.)
    assert result.total_ms == 11.
    assert result.service_sum_ms == 14.
    assert result.overlap_ms == 3.
    assert result.starts_ms["forward"] == result.starts_ms["host_tail"] == 3.
    assert result.starts_ms["bookkeep"] == 9.
    assert result.critical_path == ("schedule", "prepare", "forward", "sample", "wait", "bookkeep", "update")
    short_forward = model.evaluate(1.)
    assert short_forward.total_ms == 9.
    assert short_forward.overlap_ms == 1.


@pytest.mark.parametrize("forward", [0., .01, .1, 1., 3., 5., 10., 100.])
def test_waits_and_serialization(forward):
    nodes = example()
    result = StepDependencyModel(nodes).evaluate(forward)
    assert result.total_ms == pytest.approx(6. + max(3., forward))
    assert result.total_ms >= forward
    assert result.total_ms <= result.service_sum_ms + 1e-10
    for node in nodes:
        for dep in node.predecessors:
            assert result.starts_ms[node.name] >= result.finishes_ms[dep]
    # Turning the host tail into a dependent phase removes overlap.
    nodes[3] = replace(nodes[3], predecessors=("forward",))
    serial = StepDependencyModel(nodes).evaluate(forward)
    assert serial.overlap_ms == pytest.approx(0.)


@pytest.mark.parametrize("duration", [-1., math.nan, math.inf, -math.inf, True, "1"])
def test_invalid_duration(duration):
    nodes = example()
    nodes[0] = replace(nodes[0], duration_ms=duration)
    with pytest.raises(ValueError):
        StepDependencyModel(nodes)
    with pytest.raises(ValueError):
        StepDependencyModel(example()).evaluate(duration)


@pytest.mark.parametrize("kind", ["thread_cpu", "inclusive_wall", "poll_suffix", "residual", "engine_step"])
def test_cannot_relabel_legacy_measurements(kind):
    nodes = example()
    nodes[0] = replace(nodes[0], duration_kind=kind)
    with pytest.raises(ValueError, match="not exclusive"):
        StepDependencyModel(nodes)


def test_duplicate_parent_child_charge():
    nodes = example()
    nodes[0] = replace(nodes[0], work_units=("schedule", "prepare"))
    with pytest.raises(ValueError, match="more than once"):
        StepDependencyModel(nodes)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "cycle", "measured_forward", "no_forward", "wait_cost"])
def test_invalid_structure(mutation):
    nodes = example()
    if mutation == "duplicate":
        nodes.append(nodes[0])
    elif mutation == "missing":
        nodes[0] = replace(nodes[0], predecessors=("absent",))
    elif mutation == "cycle":
        nodes[0] = replace(nodes[0], predecessors=("update",))
    elif mutation == "measured_forward":
        nodes[2] = replace(nodes[2], duration_ms=5.)
    elif mutation == "no_forward":
        nodes[2] = replace(nodes[2], duration_kind="device_service", duration_ms=5.)
    else:
        nodes[5] = replace(nodes[5], duration_ms=1.)
    with pytest.raises(ValueError):
        StepDependencyModel(nodes)


def test_no_global_max_shortcut():
    nodes = example()
    result = StepDependencyModel(nodes).evaluate(5.)
    # max(total CPU, total GPU)=max(8,6)=8 is wrong: schedule/prepare
    # precede the GPU; bookkeeping/update depend on its result.
    assert result.total_ms != max(8., 6.)

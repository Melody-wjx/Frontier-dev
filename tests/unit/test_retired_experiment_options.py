"""Retired experiments must not remain available as silent runtime switches."""
from dataclasses import fields

import pytest

from frontier.config import SimulationConfig, TraceRequestGeneratorConfig
from frontier.config import global_vars
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


@pytest.mark.parametrize("config_type,field_name", [
    (SimulationConfig, "decode_step_wall_table_file"),
    (TraceRequestGeneratorConfig, "max_concurrent_sessions"),
])
def test_retired_fields_are_not_in_generated_cli(config_type, field_name):
    assert field_name not in {field.name for field in fields(config_type)}


def test_session_gate_constructor_is_rejected():
    with pytest.raises(TypeError, match="max_concurrent_sessions"):
        TraceRequestGeneratorConfig(max_concurrent_sessions=8)


@pytest.mark.parametrize("name", [
    "DECODE_STEP_WALL_TABLE_FILE",
    "set_decode_step_wall_table_file",
    "get_decode_step_wall_table_file",
])
def test_retired_global_timing_switch_is_absent(name):
    assert not hasattr(global_vars, name)


@pytest.mark.parametrize("name", [
    "_get_decode_step_wall_table", "_interp_step_wall",
    "_maybe_decode_step_wall_execution_time",
])
def test_whole_step_override_is_absent(name):
    assert not hasattr(SklearnExecutionTimePredictor, name)


@pytest.mark.parametrize("name", [
    "set_pending_closed_loop_requests", "release_next_closed_loop_request",
    "num_pending_closed_loop_requests",
])
def test_scheduler_has_no_session_admission_gate(name):
    assert not hasattr(BaseGlobalScheduler, name)

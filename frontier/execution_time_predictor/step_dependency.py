"""Offline single-step dependency evaluator, not a new simulator default.

This core deliberately has no adapter for legacy CPU-calibration rows. Their
inclusive scopes/active-thread clocks cannot identify exclusive service time.
Callers must supply mutually exclusive work and explicit resource order. GPU
Forward is supplied by the operator predictor, never a target-workload RPC.
The chronological host/device release points are part of the input contract;
this evaluator cannot infer them from aggregate phase medians.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable


@dataclass(frozen=True)
class StepPhase:
    name: str
    resource: str | None
    predecessors: tuple[str, ...]
    duration_kind: str
    duration_ms: float | None
    work_units: tuple[str, ...]


@dataclass(frozen=True)
class StepTimeline:
    starts_ms: dict[str, float]
    finishes_ms: dict[str, float]
    total_ms: float
    service_sum_ms: float
    overlap_ms: float
    critical_path: tuple[str, ...]


class StepDependencyModel:
    """Compile a fixed-resource-order DAG once; evaluate it in O(nodes+edges).

    Input order defines serialization on each resource. Independent resources
    overlap only if dependency edges allow it. A zero-cost dependency fence
    reserves its resource while waiting (e.g. a synchronous worker thread).
    A host call spanning GPU work must be split at its submission/wait boundary
    before being supplied here; aggregate lexical wall time is not admissible.
    """

    def __init__(self, phases: Iterable[StepPhase]):
        self.phases = tuple(phases)
        if not self.phases:
            raise ValueError("step dependency model requires phases")
        names, charged, last_resource = set(), set(), {}
        dependencies = {}
        forward_count = 0
        for phase in self.phases:
            if not isinstance(phase.name, str) or not phase.name or phase.name in names:
                raise ValueError("phase names must be nonempty and unique")
            names.add(phase.name)
            if phase.resource is not None and (not isinstance(phase.resource, str) or not phase.resource):
                raise ValueError("invalid resource")
            if any(not isinstance(x, str) or not x for x in phase.predecessors):
                raise ValueError("invalid predecessor")
            if len(set(phase.predecessors)) != len(phase.predecessors):
                raise ValueError("duplicate predecessor")
            if phase.duration_kind not in ("exclusive_host_service", "device_service", "predicted_forward", "dependency_fence"):
                raise ValueError("inclusive walls, thread CPU and residuals are not exclusive service times")
            if phase.duration_kind == "predicted_forward":
                forward_count += 1
                if phase.duration_ms is not None:
                    raise ValueError("Forward must come from the predictor, not an observed duration")
            elif (isinstance(phase.duration_ms, bool) or not isinstance(phase.duration_ms, (int, float))
                  or not math.isfinite(phase.duration_ms) or phase.duration_ms < 0):
                raise ValueError("service duration must be finite and nonnegative")
            if phase.duration_kind == "dependency_fence":
                if phase.duration_ms != 0 or phase.work_units:
                    raise ValueError("wait fences have zero service cost and no charged work")
            elif phase.resource is None or not phase.work_units:
                raise ValueError("service phases require a resource and explicit work units")
            if any(not isinstance(x, str) or not x for x in phase.work_units):
                raise ValueError("invalid work unit")
            if len(set(phase.work_units)) != len(phase.work_units) or charged.intersection(phase.work_units):
                raise ValueError("work is charged more than once (parent/child overlap)")
            charged.update(phase.work_units)
            deps = set(phase.predecessors)
            if phase.resource is not None:
                if phase.resource in last_resource:
                    deps.add(last_resource[phase.resource])
                last_resource[phase.resource] = phase.name
            dependencies[phase.name] = deps
        if forward_count != 1:
            raise ValueError("exactly one predicted Forward is required")
        if any(not deps <= names for deps in dependencies.values()):
            raise ValueError("unknown predecessor")
        self._by_name = {p.name: p for p in self.phases}
        self._dependencies = {name: tuple(sorted(deps)) for name, deps in dependencies.items()}
        successors = {name: [] for name in names}
        indegree = {name: len(deps) for name, deps in dependencies.items()}
        for name, deps in dependencies.items():
            for dependency in deps:
                successors[dependency].append(name)
        ready = [name for name in names if indegree[name] == 0]
        heapq.heapify(ready)
        order = []
        while ready:
            name = heapq.heappop(ready)
            order.append(name)
            for successor in successors[name]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    heapq.heappush(ready, successor)
        if len(order) != len(names):
            raise ValueError("cyclic dependencies or incompatible resource order")
        self._order = tuple(order)

    def evaluate(self, model_forward_ms: float) -> StepTimeline:
        if (isinstance(model_forward_ms, bool) or not isinstance(model_forward_ms, (int, float))
                or not math.isfinite(model_forward_ms) or model_forward_ms < 0):
            raise ValueError("model Forward must be finite and nonnegative")
        starts, finishes, previous = {}, {}, {}
        service_sum = 0.0
        for name in self._order:
            phase = self._by_name[name]
            deps = self._dependencies[name]
            parent = max(deps, key=lambda x: (finishes[x], x)) if deps else None
            start = finishes[parent] if parent else 0.0
            duration = model_forward_ms if phase.duration_kind == "predicted_forward" else phase.duration_ms
            starts[name], finishes[name], previous[name] = start, start + duration, parent
            service_sum += duration
        terminal = max(self._order, key=lambda x: (finishes[x], x))
        total = finishes[terminal]
        critical = []
        node = terminal
        while node is not None:
            critical.append(node)
            node = previous[node]
        return StepTimeline(starts, finishes, total, service_sum,
                            max(0.0, service_sum - total), tuple(reversed(critical)))

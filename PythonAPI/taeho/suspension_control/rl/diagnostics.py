"""Diagnostic helpers shared by RL dry-run and route logging."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def flatten_state(state: Any, prefix: str = "state_") -> Dict[str, Any]:
    if state is None:
        return {}
    values = state.as_dict() if hasattr(state, "as_dict") else dict(
        getattr(state, "__dict__", {}))
    return {prefix + key: value for key, value in values.items()}


def command_scales(command: Any, prefix: str) -> Dict[str, Any]:
    wheels = tuple(getattr(command, "wheels", ()) or ())
    result: Dict[str, Any] = {}
    spring_values = []
    damper_values = []
    for index, wheel in enumerate(wheels):
        spring = float(getattr(wheel, "spring_scale", 1.0))
        damper = float(getattr(wheel, "damper_scale", 1.0))
        spring_values.append(spring)
        damper_values.append(damper)
        result["%sspring_%d" % (prefix, index)] = spring
        result["%sdamper_%d" % (prefix, index)] = damper
    if spring_values:
        result["%smean_spring" % prefix] = sum(spring_values) / float(len(spring_values))
    if damper_values:
        result["%smean_damper" % prefix] = sum(damper_values) / float(len(damper_values))
    return result


def indexed_values(prefix: str, values: Sequence[float]) -> Dict[str, Any]:
    return {"%s%d" % (prefix, index): value for index, value in enumerate(values)}


def planning_summary(planning: Any) -> Dict[str, Any]:
    if planning is None:
        return {"planning_available": 0.0}
    if hasattr(planning, "preview_summary"):
        try:
            result = dict(planning.preview_summary())
            result["planning_source"] = getattr(planning, "source", "")
            result["planning_frame"] = getattr(planning, "frame", -1)
            return result
        except Exception:
            return {"planning_available": 0.0}
    points = tuple(getattr(planning, "points", ()) or ())
    return {
        "planning_available": 1.0 if points else 0.0,
        "planning_source": getattr(planning, "source", ""),
    }


def merge_prefixed(prefix: str, values: Mapping[str, Any]) -> Dict[str, Any]:
    return {prefix + key: value for key, value in values.items()}

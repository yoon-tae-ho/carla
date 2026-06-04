"""Diagnostic helpers shared by RL dry-run and route logging."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


WHEEL_LABELS = ("fl", "fr", "rl", "rr")


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
        label = WHEEL_LABELS[index] if index < len(WHEEL_LABELS) else str(index)
        spring_values.append(spring)
        damper_values.append(damper)
        result["%sspring_%d" % (prefix, index)] = spring
        result["%sdamper_%d" % (prefix, index)] = damper
        result["%sspring_%s" % (prefix, label)] = spring
        result["%sdamper_%s" % (prefix, label)] = damper
    if spring_values:
        result["%smean_spring" % prefix] = sum(spring_values) / float(len(spring_values))
    if damper_values:
        result["%smean_damper" % prefix] = sum(damper_values) / float(len(damper_values))
    return result


def indexed_values(prefix: str, values: Sequence[float]) -> Dict[str, Any]:
    result = {"%s%d" % (prefix, index): value for index, value in enumerate(values)}
    for index, value in enumerate(values):
        if index < len(WHEEL_LABELS):
            result["%s%s" % (prefix, WHEEL_LABELS[index])] = value
    return result


def wheel_values_from_mapping(
    values: Mapping[str, Any],
    prefix: str,
    wheel_count: int = 4,
) -> tuple:
    """Read wheel-label values from a diagnostics mapping."""

    result = []
    for index in range(max(0, int(wheel_count))):
        label = WHEEL_LABELS[index] if index < len(WHEEL_LABELS) else str(index)
        candidates = (
            "%s%s" % (prefix, label),
            "%s%d" % (prefix, index),
        )
        found = False
        for key in candidates:
            if key not in values:
                continue
            try:
                result.append(float(values.get(key)))
            except (TypeError, ValueError):
                return ()
            found = True
            break
        if not found:
            return ()
    return tuple(result)


def planning_summary(
    planning: Any,
    current_frame: Optional[int] = None,
) -> Dict[str, Any]:
    if planning is None:
        return {
            "planning_available": 0.0,
            "planning_source": "empty",
            "planning_stale": 0,
            "planning_age_frames": "",
        }
    if hasattr(planning, "preview_summary"):
        try:
            result = dict(planning.preview_summary())
            source = getattr(planning, "source", "") or "empty"
            frame = getattr(planning, "frame", -1)
            age = _planning_age(frame, current_frame)
            stale = _planning_stale(planning, age)
            if source == "empty":
                result["planning_available"] = 0.0
                stale = 0
                age = ""
            elif stale:
                result["planning_available"] = 0.0
            result["planning_source"] = source
            result["planning_frame"] = frame
            result["planning_stale"] = int(stale)
            result["planning_age_frames"] = age
            return result
        except Exception:
            return {
                "planning_available": 0.0,
                "planning_source": "invalid",
                "planning_stale": 1,
                "planning_age_frames": "",
            }
    points = tuple(getattr(planning, "points", ()) or ())
    source = getattr(planning, "source", "") or "empty"
    frame = getattr(planning, "frame", -1)
    age = _planning_age(frame, current_frame)
    stale = _planning_stale(planning, age)
    return {
        "planning_available": 1.0 if points and not stale else 0.0,
        "planning_source": source,
        "planning_frame": frame,
        "planning_stale": int(stale),
        "planning_age_frames": "" if source == "empty" else age,
    }


def merge_prefixed(prefix: str, values: Mapping[str, Any]) -> Dict[str, Any]:
    return {prefix + key: value for key, value in values.items()}


def _planning_age(frame: Any, current_frame: Optional[int]) -> Any:
    if current_frame is None:
        return ""
    try:
        frame_value = int(frame)
        current = int(current_frame)
    except (TypeError, ValueError):
        return ""
    if frame_value < 0:
        return ""
    return max(0, current - frame_value)


def _planning_stale(planning: Any, age: Any) -> bool:
    metadata = dict(getattr(planning, "metadata", {}) or {})
    extra = dict(getattr(planning, "extra", {}) or {})
    for values in (metadata, extra):
        if values.get("planning_stale") or values.get("stale"):
            return True
    try:
        return int(age) > int(extra.get("max_age_frames", 1))
    except (TypeError, ValueError):
        return False

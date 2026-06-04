"""Task-info normalization and rollout summary helpers for suspension RL."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_TASK_INFO: Dict[str, Any] = {
    "route_progress_available": 0.0,
    "route_progress_fraction": 0.0,
    "route_progress_monotonic_fraction": 0.0,
    "route_progress_m": 0.0,
    "route_progress_delta_m": 0.0,
    "route_delta_progress_m": 0.0,
    "route_progress_rate_mps": 0.0,
    "route_distance_to_end_m": 0.0,
    "route_completion_proxy": 0.0,
    "delta_progress": 0.0,
    "target_speed": 0.0,
    "target_speed_source": "missing",
    "speed_error": 0.0,
    "abs_speed_error": 0.0,
    "target_speed_error": 0.0,
    "planned_stop": 0.0,
    "route_deviation_m": 0.0,
    "route_deviation": 0.0,
    "route_deviation_valid": 0.0,
    "lane_invasion_count": 0.0,
    "collision_count": 0.0,
    "red_light_count": 0.0,
    "blocked_vehicle": 0.0,
    "route_timeout": 0.0,
    "low_speed_not_planned": 0.0,
    "low_speed_not_planned_severity": 0.0,
    "route_progress_stall": 0.0,
    "progress_stall": 0.0,
    "progress_stall_count": 0.0,
    "negative_progress": 0.0,
    "route_failed": 0.0,
}


@dataclass(frozen=True)
class TaskInfoBuilderConfig:
    nominal_target_speed: float = 8.0
    reward_low_speed_threshold_mps: float = 0.5
    reward_min_progress_rate_mps: float = 0.5
    reward_planned_stop_brake_threshold: float = 0.25
    progress_stall_steps: int = 10
    progress_stall_epsilon_m: float = 0.1


class TaskInfoBuilder:
    """Build a conservative route-task mapping for reward diagnostics."""

    def __init__(self, config: Optional[TaskInfoBuilderConfig] = None):
        self.config = config or TaskInfoBuilderConfig()
        self._stall_counts: Dict[str, int] = {}

    def reset(self, actor_id: Any = None) -> None:
        if actor_id is None:
            self._stall_counts.clear()
        else:
            self._stall_counts.pop(str(actor_id), None)

    def build(
        self,
        *,
        state: Any,
        previous_state: Any = None,
        route_progress: Optional[Mapping[str, Any]] = None,
        planning: Any = None,
        live_task_info: Optional[Mapping[str, Any]] = None,
        actor_id: Any = "hero",
    ) -> Dict[str, Any]:
        del previous_state
        cfg = self.config
        route = dict(route_progress or {})
        live = dict(live_task_info or {})
        progress_available = _first_float(route, (
            "route_progress_available",
            "progress_available",
        ))
        progress_fraction = _clamp01(_first_float(route, (
            "route_progress_fraction",
            "route_progress",
            "progress",
        )))
        monotonic_fraction = _clamp01(_first_float(route, (
            "route_progress_monotonic_fraction",
            "route_completion_proxy",
        )))
        if monotonic_fraction <= 0.0:
            monotonic_fraction = progress_fraction
        route_progress_m = _first_float(route, ("route_progress_m",))
        route_progress_delta_m = _first_float(route, (
            "route_progress_delta_m",
            "progress_delta",
            "route_progress_delta",
            "delta_progress",
        ))
        raw_delta_progress_m = _first_float_or_none(route, (
            "route_delta_progress_m",
            "route_progress_raw_delta_m",
        ))
        if raw_delta_progress_m is None:
            raw_delta_progress_m = route_progress_delta_m
        progress_rate_mps = _first_float(route, (
            "route_progress_rate_mps",
            "progress_rate_mps",
        ))
        route_distance_to_end_m = _first_float(route, (
            "route_distance_to_end_m",
            "distance_to_end",
        ))
        route_deviation = _first_float(route, (
            "route_deviation_m",
            "route_progress_raw_deviation_m",
            "route_deviation",
        ))
        if route_deviation == 0.0 and planning is not None:
            route_deviation = _finite_float(getattr(planning, "route_deviation", 0.0))
        route_deviation_valid = _first_float(route, (
            "route_progress_deviation_valid",
            "route_deviation_valid",
        ))

        target_speed, target_source = _target_speed(planning, cfg.nominal_target_speed)
        speed = _finite_float(_object_get(state, "speed", 0.0))
        speed_error = target_speed - speed if target_source != "missing" else 0.0
        abs_speed_error = abs(speed_error)
        brake = _finite_float(_object_get(state, "brake", 0.0))

        planned_stop = _planned_stop(
            planning=planning,
            live_task_info=live,
            brake=brake,
            target_speed=target_speed,
            low_speed_threshold_mps=cfg.reward_low_speed_threshold_mps,
            brake_threshold=cfg.reward_planned_stop_brake_threshold)
        route_complete = monotonic_fraction >= 0.98
        low_speed = (
            target_source != "missing" and
            not planned_stop and
            speed < cfg.reward_low_speed_threshold_mps and
            not route_complete and
            progress_rate_mps < cfg.reward_min_progress_rate_mps and
            brake < cfg.reward_planned_stop_brake_threshold)
        speed_severity = _clamp01(
            (cfg.reward_low_speed_threshold_mps - speed) /
            max(cfg.reward_low_speed_threshold_mps, 1.0e-9))
        rate_severity = _clamp01(
            (cfg.reward_min_progress_rate_mps - progress_rate_mps) /
            max(cfg.reward_min_progress_rate_mps, 1.0e-9))
        low_speed_severity = (
            max(speed_severity, rate_severity) if low_speed else 0.0)

        actor_key = str(actor_id)
        if (
                not route_complete and
                route_progress_delta_m < cfg.progress_stall_epsilon_m and
                not planned_stop):
            self._stall_counts[actor_key] = self._stall_counts.get(actor_key, 0) + 1
        else:
            self._stall_counts[actor_key] = 0
        progress_stall_count = self._stall_counts.get(actor_key, 0)
        progress_stall = (
            1.0
            if progress_stall_count >= max(1, int(cfg.progress_stall_steps))
            else 0.0)
        negative_progress = 1.0 if (
            raw_delta_progress_m < -1.0e-9 or
            _first_float(route, ("route_progress_negative_raw",)) > 0.0
        ) else 0.0

        info: Dict[str, Any] = dict(DEFAULT_TASK_INFO)
        info.update({
            "route_progress_available": progress_available,
            "route_deviation": route_deviation,
            "route_deviation_m": route_deviation,
            "route_deviation_valid": route_deviation_valid,
            "route_progress_fraction": progress_fraction,
            "route_progress_monotonic_fraction": monotonic_fraction,
            "route_progress_m": route_progress_m,
            "route_progress_delta_m": route_progress_delta_m,
            "route_delta_progress_m": raw_delta_progress_m,
            "route_progress_rate_mps": progress_rate_mps,
            "route_distance_to_end_m": route_distance_to_end_m,
            "route_completion_proxy": (
                1.0 if monotonic_fraction >= 0.99 else monotonic_fraction),
            "delta_progress": route_progress_delta_m,
            "target_speed": target_speed,
            "target_speed_source": target_source,
            "speed_error": speed_error,
            "abs_speed_error": abs_speed_error,
            "target_speed_error": speed_error,
            "planned_stop": 1.0 if planned_stop else 0.0,
            "low_speed_not_planned": 1.0 if low_speed else 0.0,
            "low_speed_not_planned_severity": low_speed_severity,
            "route_progress_stall": progress_stall,
            "progress_stall": progress_stall,
            "progress_stall_count": float(progress_stall_count),
            "negative_progress": negative_progress,
            "collision_count": _first_float(live, (
                "collision_count",
                "collisions_layout",
                "collisions_pedestrian",
                "collisions_vehicle",
            )),
            "lane_invasion_count": _first_float(live, (
                "lane_invasion_count",
                "lane_invasion",
                "outside_route_lanes",
            )),
            "red_light_count": _first_float(live, (
                "red_light_count",
                "red_light",
                "red_light_infractions",
            )),
            "blocked_vehicle": _first_float(live, (
                "blocked_vehicle",
                "vehicle_blocked",
                "blocked",
            )),
            "route_timeout": _first_float(live, (
                "route_timeout",
                "scenario_timeouts",
                "timeout",
            )),
            "route_failed": _first_float(live, (
                "route_failed",
                "failed",
                "route_failure",
            )),
        })
        return info


def normalize_task_info(
    raw: Mapping[str, Any],
    *,
    require_non_empty: bool,
) -> Dict[str, Any]:
    """Normalize backend task metrics into the reward contract."""

    if not raw:
        if require_non_empty:
            raise RuntimeError("backend task_info is empty in online CARLA RL env")
        return dict(DEFAULT_TASK_INFO)

    source = dict(raw or {})
    normalized = dict(DEFAULT_TASK_INFO)
    normalized["route_progress_available"] = _first_float(source, (
        "route_progress_available",
        "progress_available",
    ))
    normalized["route_progress_fraction"] = _first_float(source, (
        "route_progress_fraction",
        "route_completion",
        "route_progress",
        "progress",
        "score_route_fraction",
    ))
    normalized["route_progress_monotonic_fraction"] = _first_float(source, (
        "route_progress_monotonic_fraction",
        "route_completion_proxy",
    ))
    if normalized["route_progress_monotonic_fraction"] == 0.0:
        normalized["route_progress_monotonic_fraction"] = normalized[
            "route_progress_fraction"]
    normalized["route_progress_m"] = _first_float(source, (
        "route_progress_m",
        "progress_m",
    ))
    monotonic_delta = _first_float_or_none(source, (
        "route_progress_delta_m",
        "progress_delta",
        "route_progress_delta",
        "delta_progress",
    ))
    raw_delta = _first_float_or_none(source, (
        "route_delta_progress_m",
        "route_progress_raw_delta_m",
    ))
    if monotonic_delta is None:
        monotonic_delta = raw_delta if raw_delta is not None else 0.0
    if raw_delta is None:
        raw_delta = monotonic_delta
    normalized["route_progress_delta_m"] = monotonic_delta
    normalized["route_delta_progress_m"] = raw_delta
    normalized["delta_progress"] = monotonic_delta
    normalized["route_progress_rate_mps"] = _first_float(source, (
        "route_progress_rate_mps",
        "progress_rate_mps",
    ))
    normalized["route_distance_to_end_m"] = _first_float(source, (
        "route_distance_to_end_m",
        "distance_to_end",
    ))
    normalized["route_completion_proxy"] = _first_float(source, (
        "route_completion_proxy",
        "route_completion",
    ))
    if normalized["route_completion_proxy"] == 0.0:
        normalized["route_completion_proxy"] = (
            1.0
            if normalized["route_progress_monotonic_fraction"] >= 0.99
            else normalized["route_progress_monotonic_fraction"])
    normalized["target_speed"] = _first_float(source, (
        "target_speed",
        "desired_speed",
        "speed_limit",
    ))
    normalized["target_speed_source"] = str(
        source.get("target_speed_source", normalized.get("target_speed_source", "missing"))
        or "missing")
    normalized["speed_error"] = _first_float(source, (
        "speed_error",
        "target_speed_error",
    ))
    normalized["abs_speed_error"] = _first_float(source, (
        "abs_speed_error",
        "target_speed_error_abs",
    ))
    normalized["target_speed_error"] = _first_float(source, (
        "target_speed_error",
        "speed_error",
    ))
    if normalized["abs_speed_error"] == 0.0:
        normalized["abs_speed_error"] = abs(_finite_float(
            normalized.get("target_speed_error", 0.0)))
    normalized["planned_stop"] = _first_float(source, (
        "planned_stop",
        "route_planned_stop",
    ))
    route_deviation = _first_float(source, (
        "route_deviation_m",
        "route_progress_raw_deviation_m",
        "route_deviation",
        "route_dev",
        "outside_route_lanes",
    ))
    normalized["route_deviation_m"] = route_deviation
    normalized["route_deviation"] = route_deviation
    normalized["route_deviation_valid"] = _first_float(source, (
        "route_progress_deviation_valid",
        "route_deviation_valid",
    ))
    normalized["lane_invasion_count"] = _first_float(source, (
        "lane_invasion_count",
        "lane_invasion",
        "outside_route_lanes",
    ))
    normalized["collision_count"] = _sum_float(source, (
        "collision_count",
        "collisions_layout",
        "collisions_pedestrian",
        "collisions_vehicle",
    ))
    normalized["red_light_count"] = _first_float(source, (
        "red_light_count",
        "red_light",
        "red_light_infractions",
    ))
    normalized["blocked_vehicle"] = _first_float(source, (
        "blocked_vehicle",
        "vehicle_blocked",
        "blocked",
    ))
    normalized["route_timeout"] = _first_float(source, (
        "route_timeout",
        "scenario_timeouts",
        "timeout",
    ))
    normalized["low_speed_not_planned"] = _first_float(source, (
        "low_speed_not_planned",
        "min_speed_infractions",
        "low_speed",
    ))
    normalized["low_speed_not_planned_severity"] = _first_float(source, (
        "low_speed_not_planned_severity",
        "low_speed_severity",
    ))
    progress_stall = _first_float(source, (
        "route_progress_stall",
        "progress_stall",
    ))
    normalized["route_progress_stall"] = progress_stall
    normalized["progress_stall"] = progress_stall
    normalized["progress_stall_count"] = _first_float(source, (
        "progress_stall_count",
        "route_progress_stall_count",
    ))
    normalized["negative_progress"] = _first_float(source, (
        "negative_progress",
        "route_negative_progress",
        "route_progress_negative_raw",
    ))
    if normalized["negative_progress"] == 0.0 and raw_delta < 0.0:
        normalized["negative_progress"] = 1.0
    normalized["route_failed"] = _first_float(source, (
        "route_failed",
        "failed",
        "route_failure",
    ))
    for key in tuple(normalized):
        if key == "target_speed_source":
            continue
        value = normalized[key]
        normalized[key] = value if math.isfinite(value) else 0.0
    return normalized


def task_done_reason(task_info: Mapping[str, Any]) -> str:
    info = normalize_task_info(task_info, require_non_empty=False)
    if info["collision_count"] > 0.0:
        return "collision"
    if info["route_failed"] > 0.0:
        return "route_failed"
    if info["route_timeout"] > 0.0:
        return "route_timeout"
    if info["blocked_vehicle"] > 0.0:
        return "blocked_vehicle"
    if (
            info["route_progress_monotonic_fraction"] >= 0.999 or
            info["route_progress_fraction"] >= 0.999):
        return "route_completed"
    return ""


def task_summary_fields(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute action, projection, reward, and task summary fields."""

    rows = list(rows or [])
    count = len(rows)
    summary: Dict[str, Any] = {
        "diagnostic_rows": count,
        "policy_available_ratio": "",
        "fallback_ratio": "",
        "effective_control_ratio": "",
        "low_speed_mask_ratio": "",
        "hard_safety_gate_ratio": "",
        "soft_safety_gain_ratio": "",
        "observation_clip_ratio": "",
        "mean_abs_action": "",
        "mean_abs_residual_damper": "",
        "mean_reward_total": "",
        "mean_reward_comfort": "",
        "mean_reward_stability": "",
        "mean_reward_task": "",
        "mean_reward_action": "",
        "mean_reward_safety": "",
        "route_progress_fraction_max": "",
        "route_completion_proxy": "",
        "collision_count": 0.0,
        "lane_invasion_count": 0.0,
        "red_light_count": 0.0,
        "blocked_vehicle_count": 0.0,
        "route_timeout_count": 0.0,
    }
    if not rows:
        return summary

    policy_available = 0
    fallback = 0
    nonzero_residual = 0
    low_speed_mask = 0
    hard_safety = 0
    soft_safety = 0
    observation_clipped = 0
    abs_actions = []
    abs_residuals = []
    reward_total = []
    reward_comfort = []
    reward_stability = []
    reward_task = []
    reward_action = []
    reward_safety = []
    route_progress = []
    collisions = []
    lanes = []
    reds = []
    blocked = []
    timeouts = []

    for row in rows:
        if _positive(row.get("rl_policy_available", 1.0)):
            policy_available += 1
        if str(row.get("rl_fallback_reason", "") or ""):
            fallback += 1
        if _positive(row.get("rl_mean_abs_residual_damper")):
            nonzero_residual += 1
        reasons = set(
            item for item in str(row.get("rl_safety_gate_reason", "")).split(";")
            if item)
        safety_gain = _safe_float(row.get("rl_safety_gain"))
        gate_active = _positive(row.get("rl_safety_gate_active"))
        low_speed = (
            _positive(row.get("rl_safety_gate_speed_limit")) or
            "speed_below_min" in reasons)
        hard = (
            _positive(row.get("rl_safety_gate_nonfinite_obs")) or
            _positive(row.get("rl_safety_gate_action_invalid")) or
            "nonfinite_obs" in reasons or
            "action_invalid" in reasons)
        if low_speed:
            low_speed_mask += 1
        if (
                not low_speed and
                (hard or (
                    gate_active and
                    safety_gain is not None and
                    safety_gain <= 1.0e-12))):
            hard_safety += 1
        if (
                not low_speed and
                safety_gain is not None and
                safety_gain > 1.0e-12 and
                safety_gain < 1.0 - 1.0e-12):
            soft_safety += 1
        if _positive(row.get("rl_observation_clip_count")):
            observation_clipped += 1

        _append(abs_actions, row.get("rl_mean_abs_action"))
        _append(abs_residuals, row.get("rl_mean_abs_residual_damper"))
        _append(reward_total, row.get("reward_total"))
        _append(reward_comfort, row.get("reward_comfort"))
        _append(reward_stability, row.get("reward_stability"))
        _append(reward_task, row.get("reward_task"))
        _append(reward_action, row.get("reward_action"))
        _append(reward_safety, row.get("reward_safety"))
        _append(route_progress, row.get("route_progress_fraction"))
        _append(collisions, row.get("collision_count"))
        _append(lanes, row.get("lane_invasion_count"))
        _append(reds, row.get("red_light_count"))
        _append(blocked, row.get("blocked_vehicle"))
        _append(timeouts, row.get("route_timeout"))

    progress_max = max(route_progress) if route_progress else ""
    summary.update({
        "policy_available_ratio": policy_available / float(count),
        "fallback_ratio": fallback / float(count),
        "effective_control_ratio": nonzero_residual / float(count),
        "low_speed_mask_ratio": low_speed_mask / float(count),
        "hard_safety_gate_ratio": hard_safety / float(count),
        "soft_safety_gain_ratio": soft_safety / float(count),
        "observation_clip_ratio": observation_clipped / float(count),
        "mean_abs_action": _mean_or_empty(abs_actions),
        "mean_abs_residual_damper": _mean_or_empty(abs_residuals),
        "mean_reward_total": _mean_or_empty(reward_total),
        "mean_reward_comfort": _mean_or_empty(reward_comfort),
        "mean_reward_stability": _mean_or_empty(reward_stability),
        "mean_reward_task": _mean_or_empty(reward_task),
        "mean_reward_action": _mean_or_empty(reward_action),
        "mean_reward_safety": _mean_or_empty(reward_safety),
        "route_progress_fraction_max": progress_max,
        "route_completion_proxy": (
            1.0 if progress_max != "" and progress_max >= 0.99 else 0.0),
        "collision_count": max(collisions) if collisions else 0.0,
        "lane_invasion_count": max(lanes) if lanes else 0.0,
        "red_light_count": max(reds) if reds else 0.0,
        "blocked_vehicle_count": max(blocked) if blocked else 0.0,
        "route_timeout_count": max(timeouts) if timeouts else 0.0,
    })
    return summary


def _target_speed(planning: Any, nominal_target_speed: float) -> tuple:
    planning_speed = _planning_target_speed(planning)
    if planning_speed is not None:
        return planning_speed, "planning"
    nominal = _safe_float(nominal_target_speed)
    if nominal is not None:
        return max(0.0, nominal), "nominal"
    return 0.0, "missing"


def _planning_target_speed(planning: Any) -> Optional[float]:
    if planning is None:
        return None
    values = []
    if isinstance(planning, Mapping):
        raw = planning.get("target_speed", ())
        values.extend(_sequence_values(raw))
        raw_points = planning.get("points", ())
    else:
        values.extend(_sequence_values(getattr(planning, "target_speed", ())))
        raw_points = getattr(planning, "points", ())
    for point in raw_points or ():
        if isinstance(point, Mapping):
            values.append(point.get("speed"))
        else:
            values.append(getattr(point, "speed", None))
    for value in values:
        number = _safe_float(value)
        if number is not None:
            return max(0.0, number)
    return None


def _planned_stop(
    *,
    planning: Any,
    live_task_info: Mapping[str, Any],
    brake: float,
    target_speed: float,
    low_speed_threshold_mps: float,
    brake_threshold: float,
) -> bool:
    if brake >= brake_threshold:
        return True
    if target_speed <= low_speed_threshold_mps:
        return True
    if _first_float(live_task_info, (
            "planned_stop",
            "route_planned_stop",
            "stop_planned",
            "hard_brake_planned")) > 0.0:
        return True
    if planning is None:
        return False

    for mapping in _planning_mappings(planning):
        for name in (
                "planned_stop",
                "route_planned_stop",
                "stop_planned",
                "hard_brake",
                "hard_brake_planned"):
            if name in mapping and _truthy(mapping.get(name)):
                return True

    if any(value >= brake_threshold for value in _planning_brake_values(planning)):
        return True
    if any(value <= low_speed_threshold_mps for value in _planning_speed_values(planning)):
        return True
    return False


def _planning_mappings(planning: Any) -> Sequence[Mapping[str, Any]]:
    mappings = []
    if isinstance(planning, Mapping):
        mappings.append(planning)
        for name in ("metadata", "extra"):
            value = planning.get(name)
            if isinstance(value, Mapping):
                mappings.append(value)
        return tuple(mappings)
    for name in ("metadata", "extra"):
        value = getattr(planning, name, None)
        if isinstance(value, Mapping):
            mappings.append(value)
    return tuple(mappings)


def _planning_brake_values(planning: Any) -> Sequence[float]:
    if planning is None:
        return ()
    raw = planning.get("brake", ()) if isinstance(planning, Mapping) else getattr(
        planning,
        "brake",
        ())
    return tuple(
        value for value in (_safe_float(item) for item in _sequence_values(raw))
        if value is not None)


def _planning_speed_values(planning: Any) -> Sequence[float]:
    if planning is None:
        return ()
    values = []
    if isinstance(planning, Mapping):
        values.extend(_sequence_values(planning.get("target_speed", ())))
        raw_points = planning.get("points", ())
    else:
        values.extend(_sequence_values(getattr(planning, "target_speed", ())))
        raw_points = getattr(planning, "points", ())
    for point in raw_points or ():
        if isinstance(point, Mapping):
            values.append(point.get("speed"))
        else:
            values.append(getattr(point, "speed", None))
    return tuple(
        value for value in (_safe_float(item) for item in values)
        if value is not None)


def _sequence_values(values: Any) -> Sequence[Any]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        return (values,)
    try:
        return tuple(values)
    except TypeError:
        return (values,)


def _object_get(obj: Any, name: str, default: Any = 0.0) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _clamp01(value: Any) -> float:
    return max(0.0, min(1.0, _finite_float(value)))


def _first_float(source: Mapping[str, Any], names: Sequence[str]) -> float:
    for name in names:
        if name in source:
            return _finite_float(source.get(name))
    return 0.0


def _first_float_or_none(
    source: Mapping[str, Any],
    names: Sequence[str],
) -> Optional[float]:
    for name in names:
        if name in source:
            value = _safe_float(source.get(name))
            if value is not None:
                return value
            return 0.0
    return None


def _sum_float(source: Mapping[str, Any], names: Sequence[str]) -> float:
    found = False
    total = 0.0
    for name in names:
        if name in source:
            found = True
            total += max(0.0, _finite_float(source.get(name)))
    return total if found else 0.0


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _safe_float(value: Any):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    number = _safe_float(value)
    return number is not None and number > 0.0


def _positive(value: Any) -> bool:
    number = _safe_float(value)
    return number is not None and number > 0.0


def _append(values, value: Any) -> None:
    number = _safe_float(value)
    if number is not None:
        values.append(number)


def _mean_or_empty(values: Sequence[float]) -> Any:
    return sum(values) / float(len(values)) if values else ""

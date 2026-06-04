"""Reward decomposition for residual-RL suspension control."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .task_info import normalize_task_info


@dataclass(frozen=True)
class SuspensionRewardConfig:
    alive_bonus: float = 1.0
    comfort_weight: float = 1.0
    stability_weight: float = 1.0
    task_weight: float = 1.0
    action_weight: float = 0.25
    safety_weight: float = 4.0

    acc_scale: float = 10.0
    jerk_scale: float = 50.0
    angle_scale: float = 10.0
    angle_rate_scale: float = 60.0
    yaw_rate_scale: float = 60.0
    action_scale: float = 1.0
    damper_scale: float = 1.0
    route_deviation_scale: float = 8.0
    delta_progress_scale: float = 0.01
    min_delta_progress: float = 0.001
    target_speed_error_scale: float = 8.0
    t_route_deviation: float = 0.0
    t_low_speed_not_planned: float = 0.50
    t_progress_stall: float = 0.5
    t_negative_progress: float = 0.0
    t_abs_speed_error: float = 0.05
    t_lane_invasion: float = 2.0
    t_collision: float = 10.0
    t_red_light: float = 5.0
    t_route_timeout: float = 5.0
    t_blocked_vehicle: float = 2.0
    speed_error_scale: float = 8.0
    progress_rate_scale: float = 5.0
    reward_route_deviation_enabled: bool = False
    reward_route_deviation_valid_threshold_m: float = 8.0
    reward_task_abs_max_without_infraction: float = 2.0

    c_abs_az: float = 1.00
    c_abs_lat_acc: float = 0.25
    c_abs_long_acc: float = 0.15
    c_jerk_z: float = 0.45
    c_jerk_y: float = 0.20
    c_jerk_x: float = 0.10
    c_roll_rate: float = 0.25
    c_pitch_rate: float = 0.25

    s_roll: float = 0.35
    s_pitch: float = 0.25
    s_yaw_rate: float = 0.20
    s_lat_acc: float = 0.20
    s_body_activity: float = 0.30
    s_corner_vz: float = 0.20

    a_action_mag: float = 0.20
    a_action_rate: float = 0.30
    a_damper_rate: float = 0.20
    a_baseline_dev: float = 0.25
    a_residual_damper_mag: float = 0.35
    a_residual_damper_rate: float = 0.20
    a_final_damper_rate: float = 0.20
    a_spring_dev: float = 1.00

    safety_gate_active_penalty: float = 0.25
    safety_gain_loss_penalty: float = 0.50
    observation_clip_penalty: float = 0.05

    terminal_collision_penalty: float = 20.0
    terminal_route_failure_penalty: float = 20.0
    terminal_nonfinite_penalty: float = 20.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SuspensionRewardConfig":
        allowed = {item.name for item in fields(cls)}
        kwargs = {key: values[key] for key in values if key in allowed}
        return cls(**kwargs)


@dataclass(frozen=True)
class RewardTransition:
    state: Any
    previous_state: Optional[Any] = None
    action: Sequence[float] = ()
    previous_action: Sequence[float] = ()
    previous_final_damper_scales: Sequence[float] = ()
    final_command: Any = None
    baseline_command: Any = None
    task_info: Mapping[str, Any] = None
    diagnostics: Mapping[str, Any] = None
    terminal: bool = False
    dt: float = 0.05


class SuspensionReward:
    """Dense, decomposed reward for dry-run and future CARLA training."""

    def __init__(self, config: Optional[SuspensionRewardConfig] = None):
        self.config = config or SuspensionRewardConfig()

    def compute(self, transition: RewardTransition) -> Tuple[float, Dict[str, float]]:
        cfg = self.config
        comfort_terms = self._comfort_terms(transition)
        stability_terms = self._stability_terms(transition)
        task_terms = self._task_terms(transition)
        action_terms = self._action_terms(transition)
        safety_terms = self._safety_terms(transition)

        comfort = _sum_reward_terms(comfort_terms)
        stability = _sum_reward_terms(stability_terms)
        task = _sum_reward_terms(task_terms)
        action = _sum_reward_terms(action_terms)
        safety = _sum_reward_terms(safety_terms)
        reward = (
            cfg.alive_bonus -
            cfg.comfort_weight * comfort -
            cfg.stability_weight * stability -
            cfg.task_weight * task -
            cfg.action_weight * action -
            cfg.safety_weight * safety)

        diagnostics: Dict[str, float] = {
            "reward_total": reward,
            "reward_alive_bonus": cfg.alive_bonus,
            "reward_comfort": -cfg.comfort_weight * comfort,
            "reward_stability": -cfg.stability_weight * stability,
            "reward_task": -cfg.task_weight * task,
            "reward_action": -cfg.action_weight * action,
            "reward_safety": -cfg.safety_weight * safety,
            "reward_cost_comfort": comfort,
            "reward_cost_stability": stability,
            "reward_cost_task": task,
            "reward_cost_action": action,
            "reward_cost_safety": safety,
        }
        diagnostics.update(comfort_terms)
        diagnostics.update(stability_terms)
        diagnostics.update(task_terms)
        diagnostics.update(action_terms)
        diagnostics.update(safety_terms)
        diagnostics.update(_action_diagnostics(transition))
        return reward, diagnostics

    def _comfort_terms(self, transition: RewardTransition) -> Dict[str, float]:
        cfg = self.config
        state = transition.state
        previous = transition.previous_state
        dt = max(float(transition.dt or 0.0), 1.0e-9)
        jerk_x = _jerk(state, previous, "local_ax", dt)
        jerk_y = _jerk(state, previous, "local_ay", dt)
        jerk_z = _jerk(state, previous, "az", dt)
        return {
            "reward_term_abs_az": cfg.c_abs_az * _norm_abs(
                _get(state, "az"),
                cfg.acc_scale),
            "reward_term_abs_lat_acc": cfg.c_abs_lat_acc * _norm_abs(
                _get(state, "local_ay"),
                cfg.acc_scale),
            "reward_term_abs_long_acc": cfg.c_abs_long_acc * _norm_abs(
                _get(state, "local_ax"),
                cfg.acc_scale),
            "reward_term_jerk_z": cfg.c_jerk_z * _norm_abs(jerk_z, cfg.jerk_scale),
            "reward_term_jerk_y": cfg.c_jerk_y * _norm_abs(jerk_y, cfg.jerk_scale),
            "reward_term_jerk_x": cfg.c_jerk_x * _norm_abs(jerk_x, cfg.jerk_scale),
            "reward_term_roll_rate": cfg.c_roll_rate * _norm_abs(
                _get(state, "roll_rate"),
                cfg.angle_rate_scale),
            "reward_term_pitch_rate": cfg.c_pitch_rate * _norm_abs(
                _get(state, "pitch_rate"),
                cfg.angle_rate_scale),
        }

    def _stability_terms(self, transition: RewardTransition) -> Dict[str, float]:
        cfg = self.config
        state = transition.state
        diagnostics = dict(transition.diagnostics or {})
        return {
            "reward_term_roll": cfg.s_roll * _norm_abs(
                _get(state, "roll"),
                cfg.angle_scale),
            "reward_term_pitch": cfg.s_pitch * _norm_abs(
                _get(state, "pitch"),
                cfg.angle_scale),
            "reward_term_yaw_rate": cfg.s_yaw_rate * _norm_abs(
                _get(state, "yaw_rate"),
                cfg.yaw_rate_scale),
            "reward_term_stability_lat_acc": cfg.s_lat_acc * _norm_abs(
                _get(state, "local_ay"),
                cfg.acc_scale),
            "reward_term_body_activity": cfg.s_body_activity * _nonnegative(
                diagnostics.get("body_activity", _body_activity(state))),
            "reward_term_corner_vz": cfg.s_corner_vz * _norm_abs(
                diagnostics.get("mean_abs_corner_vertical_velocity",
                                diagnostics.get("skyhook_mean_abs_corner_vz", 0.0)),
                5.0),
        }

    def _task_terms(self, transition: RewardTransition) -> Dict[str, float]:
        cfg = self.config
        info = normalize_task_info(
            transition.task_info or {},
            require_non_empty=False)
        target_speed = _nonnegative(info.get("target_speed", 0.0))
        planned_stop = (
            _nonnegative(info.get("planned_stop", 0.0)) > 0.0 or
            target_speed <= 0.5)
        delta_progress = _float(info.get("route_progress_delta_m",
                                         info.get("delta_progress", 0.0)))
        abs_speed_error = _nonnegative(info.get("abs_speed_error", 0.0))
        if abs_speed_error <= 0.0:
            abs_speed_error = abs(_float(info.get("target_speed_error", 0.0)))
        target_speed_error_term = (
            0.0
            if planned_stop
            else cfg.t_abs_speed_error * abs_speed_error /
            max(abs(cfg.target_speed_error_scale), 1.0e-9))
        low_speed_severity = _nonnegative(
            info.get("low_speed_not_planned_severity",
                     info.get("low_speed_not_planned", 0.0)))
        if low_speed_severity <= 0.0:
            low_speed_severity = _nonnegative(info.get("low_speed_not_planned", 0.0))
        route_deviation = _float(info.get("route_deviation_m",
                                          info.get("route_deviation", 0.0)))
        route_deviation_valid = (
            _nonnegative(info.get("route_deviation_valid", 0.0)) > 0.0 and
            abs(route_deviation) <=
            max(0.0, cfg.reward_route_deviation_valid_threshold_m))
        route_deviation_enabled = bool(cfg.reward_route_deviation_enabled)
        route_deviation_used = route_deviation_enabled and route_deviation_valid
        negative_progress_term = (
            cfg.t_negative_progress *
            _nonnegative(info.get("negative_progress", 0.0)))
        terms = {
            "reward_term_negative_progress": negative_progress_term,
            "reward_term_insufficient_progress": (
                _norm_positive(
                    cfg.min_delta_progress - delta_progress,
                    cfg.delta_progress_scale)
                if not planned_stop else 0.0),
            "reward_term_target_speed_error": target_speed_error_term,
            "reward_term_abs_speed_error": target_speed_error_term,
            "reward_term_route_deviation": (
                cfg.t_route_deviation *
                _norm_abs(route_deviation, cfg.route_deviation_scale)
                if route_deviation_used else 0.0),
            "reward_term_lane_invasion": (
                cfg.t_lane_invasion *
                _nonnegative(info.get("lane_invasion_count", 0.0))),
            "reward_term_collision": (
                cfg.t_collision *
                _nonnegative(info.get("collision_count", 0.0))),
            "reward_term_red_light": (
                cfg.t_red_light *
                _nonnegative(info.get("red_light_count", 0.0))),
            "reward_term_blocked_vehicle": (
                cfg.t_blocked_vehicle *
                _nonnegative(info.get("blocked_vehicle", 0.0))),
            "reward_term_low_speed_not_planned": (
                cfg.t_low_speed_not_planned * low_speed_severity),
            "reward_term_progress_stall": (
                cfg.t_progress_stall *
                _nonnegative(info.get("progress_stall", 0.0))),
            "reward_term_route_timeout": (
                cfg.t_route_timeout *
                _nonnegative(info.get("route_timeout", 0.0))),
            "reward_route_deviation_used": 1.0 if route_deviation_used else 0.0,
            "reward_route_deviation_enabled": (
                1.0 if route_deviation_enabled else 0.0),
            "reward_route_deviation_valid": (
                1.0 if route_deviation_valid else 0.0),
            "reward_task_capped_without_infraction": 0.0,
            "reward_task_abs_max_without_infraction": (
                cfg.reward_task_abs_max_without_infraction),
        }
        if not _has_task_infraction(info):
            cap = max(0.0, cfg.reward_task_abs_max_without_infraction)
            total = _sum_reward_terms(terms)
            if cap > 0.0 and total > cap:
                scale = cap / max(total, 1.0e-9)
                for key in tuple(terms):
                    if key.startswith("reward_term_"):
                        terms[key] *= scale
                terms["reward_task_capped_without_infraction"] = 1.0
        return terms

    def _action_terms(self, transition: RewardTransition) -> Dict[str, float]:
        cfg = self.config
        diagnostics = dict(transition.diagnostics or {})
        action = _float_list(transition.action)
        previous_action = _float_list(transition.previous_action)
        final_dampers = _command_values(transition.final_command, "damper_scale")
        baseline_dampers = _command_values(transition.baseline_command, "damper_scale")
        final_springs = _command_values(transition.final_command, "spring_scale")
        previous_final_dampers = _float_list(transition.previous_final_damper_scales)
        action_rate = _mean_abs_diff(action, previous_action)
        baseline_deviation = _mean_abs_diff(final_dampers, baseline_dampers)
        residual_mag = _nonnegative(
            diagnostics.get("rl_mean_abs_residual_damper", baseline_deviation))
        residual_rate = _mean_abs(_diagnostic_wheel_values(
            diagnostics,
            "rl_rate_limited_residual_damper_"))
        final_damper_rate = (
            _mean_abs_diff(final_dampers, previous_final_dampers)
            if previous_final_dampers
            else 0.0)
        return {
            "reward_term_action_mag": cfg.a_action_mag * _mean_abs(action),
            "reward_term_action_rate": cfg.a_action_rate * action_rate,
            "reward_term_damper_rate": cfg.a_damper_rate * _mean_abs_diff(
                final_dampers,
                baseline_dampers),
            "reward_term_baseline_dev": cfg.a_baseline_dev * baseline_deviation,
            "reward_term_residual_damper_mag": (
                cfg.a_residual_damper_mag * residual_mag),
            "reward_term_residual_damper_rate": (
                cfg.a_residual_damper_rate * residual_rate),
            "reward_term_final_damper_rate": (
                cfg.a_final_damper_rate * final_damper_rate),
            "reward_term_spring_dev": cfg.a_spring_dev * _mean_abs_diff(
                final_springs,
                [1.0 for _ in final_springs]),
        }

    def _safety_terms(self, transition: RewardTransition) -> Dict[str, float]:
        info = normalize_task_info(
            transition.task_info or {},
            require_non_empty=False)
        diagnostics = dict(transition.diagnostics or {})
        state_values = getattr(transition.state, "as_dict", lambda: {})().values()
        nonfinite = any(
            isinstance(value, (int, float)) and not math.isfinite(float(value))
            for value in state_values)
        return {
            "reward_term_safety_gate_active": (
                self.config.safety_gate_active_penalty
                if _nonnegative(diagnostics.get("rl_safety_gate_active", 0.0)) > 0.0
                else 0.0),
            "reward_term_safety_gain_loss": (
                self.config.safety_gain_loss_penalty *
                max(0.0, 1.0 - _float(diagnostics.get("rl_safety_gain", 1.0)))),
            "reward_term_observation_clip_count": (
                self.config.observation_clip_penalty *
                _nonnegative(diagnostics.get("observation_clip_count",
                                             diagnostics.get("rl_observation_clip_count", 0.0)))),
            "reward_term_terminal_collision": (
                self.config.terminal_collision_penalty
                if transition.terminal and _nonnegative(info.get("collision_count", 0.0)) > 0.0
                else 0.0),
            "reward_term_terminal_route_failure": (
                self.config.terminal_route_failure_penalty
                if transition.terminal and _nonnegative(info.get("route_failed", 0.0)) > 0.0
                else 0.0),
            "reward_term_nonfinite": (
                self.config.terminal_nonfinite_penalty if nonfinite else 0.0),
        }


def _sum_reward_terms(terms: Mapping[str, Any]) -> float:
    return sum(
        _float(value)
        for key, value in terms.items()
        if str(key).startswith("reward_term_"))


def _has_task_infraction(info: Mapping[str, Any]) -> bool:
    return any(
        _nonnegative(info.get(name, 0.0)) > 0.0
        for name in (
            "lane_invasion_count",
            "collision_count",
            "red_light_count",
            "blocked_vehicle",
            "route_timeout",
            "route_failed",
        ))


def _get(obj: Any, name: str, default: float = 0.0) -> float:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return _float(obj.get(name, default))
    return _float(getattr(obj, name, default))


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _float_list(values: Sequence[float]) -> Tuple[float, ...]:
    return tuple(_float(value) for value in (values or ()))


def _norm_abs(value: Any, scale: float) -> float:
    return abs(_float(value)) / max(abs(float(scale)), 1.0e-9)


def _norm_positive(value: Any, scale: float) -> float:
    return max(0.0, _float(value)) / max(abs(float(scale)), 1.0e-9)


def _nonnegative(value: Any) -> float:
    return max(0.0, _float(value))


def _jerk(state: Any, previous: Any, name: str, dt: float) -> float:
    if previous is None:
        return 0.0
    return (_get(state, name) - _get(previous, name)) / dt


def _body_activity(state: Any) -> float:
    components = (
        _norm_abs(_get(state, "roll"), 10.0),
        _norm_abs(_get(state, "pitch"), 10.0),
        _norm_abs(_get(state, "roll_rate"), 60.0),
        _norm_abs(_get(state, "pitch_rate"), 60.0),
        _norm_abs(_get(state, "local_ay"), 10.0),
        _norm_abs(_get(state, "az"), 10.0),
    )
    return sum(components) / float(len(components))


def _command_values(command: Any, name: str) -> Tuple[float, ...]:
    wheels = tuple(getattr(command, "wheels", ()) or ())
    return tuple(_float(getattr(wheel, name, 1.0)) for wheel in wheels)


def _action_diagnostics(transition: RewardTransition) -> Dict[str, float]:
    action = _float_list(transition.action)
    previous_action = _float_list(transition.previous_action)
    final_dampers = _command_values(transition.final_command, "damper_scale")
    previous_final_dampers = _float_list(transition.previous_final_damper_scales)
    return {
        "rl_action_rate": _mean_abs_diff(action, previous_action),
        "rl_final_damper_rate": (
            _mean_abs_diff(final_dampers, previous_final_dampers)
            if previous_final_dampers else 0.0),
    }


def _diagnostic_wheel_values(
    diagnostics: Mapping[str, Any],
    prefix: str,
) -> Tuple[float, ...]:
    labels = ("fl", "fr", "rl", "rr")
    values = []
    for label in labels:
        key = prefix + label
        if key in diagnostics:
            values.append(_float(diagnostics.get(key)))
    return tuple(values)


def _mean_abs(values: Sequence[float]) -> float:
    values = tuple(values or ())
    if not values:
        return 0.0
    return sum(abs(_float(value)) for value in values) / float(len(values))


def _mean_abs_diff(left: Sequence[float], right: Sequence[float]) -> float:
    left = tuple(left or ())
    right = tuple(right or ())
    if not left and not right:
        return 0.0
    count = max(len(left), len(right))
    padded_left = list(left) + [0.0 for _ in range(count - len(left))]
    padded_right = list(right) + [0.0 for _ in range(count - len(right))]
    return _mean_abs([
        _float(a) - _float(b)
        for a, b in zip(padded_left, padded_right)
    ])

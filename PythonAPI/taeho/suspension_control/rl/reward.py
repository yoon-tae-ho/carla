"""Reward decomposition for residual-RL suspension control."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class SuspensionRewardConfig:
    alive_bonus: float = 1.0
    comfort_weight: float = 1.0
    stability_weight: float = 1.0
    task_weight: float = 2.0
    action_weight: float = 0.25
    safety_weight: float = 4.0

    acc_scale: float = 10.0
    jerk_scale: float = 50.0
    angle_scale: float = 10.0
    angle_rate_scale: float = 60.0
    yaw_rate_scale: float = 60.0
    action_scale: float = 1.0
    damper_scale: float = 1.0
    route_deviation_scale: float = 2.0

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
    a_spring_dev: float = 1.00

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

        comfort = sum(comfort_terms.values())
        stability = sum(stability_terms.values())
        task = sum(task_terms.values())
        action = sum(action_terms.values())
        safety = sum(safety_terms.values())
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
        info = dict(transition.task_info or {})
        return {
            "reward_term_route_deviation": _norm_abs(
                info.get("route_deviation", 0.0),
                cfg.route_deviation_scale),
            "reward_term_lane_invasion": _nonnegative(
                info.get("lane_invasion_count", 0.0)),
            "reward_term_collision": _nonnegative(
                info.get("collision_count", 0.0)),
            "reward_term_red_light": _nonnegative(
                info.get("red_light_count", 0.0)),
            "reward_term_blocked_vehicle": _nonnegative(
                info.get("blocked_vehicle", 0.0)),
            "reward_term_low_speed_not_planned": _nonnegative(
                info.get("low_speed_not_planned", 0.0)),
            "reward_term_route_timeout": _nonnegative(
                info.get("route_timeout", 0.0)),
        }

    def _action_terms(self, transition: RewardTransition) -> Dict[str, float]:
        cfg = self.config
        action = _float_list(transition.action)
        previous_action = _float_list(transition.previous_action)
        final_dampers = _command_values(transition.final_command, "damper_scale")
        baseline_dampers = _command_values(transition.baseline_command, "damper_scale")
        final_springs = _command_values(transition.final_command, "spring_scale")
        return {
            "reward_term_action_mag": cfg.a_action_mag * _mean_abs(action),
            "reward_term_action_rate": cfg.a_action_rate * _mean_abs_diff(
                action,
                previous_action),
            "reward_term_damper_rate": cfg.a_damper_rate * _mean_abs_diff(
                final_dampers,
                baseline_dampers),
            "reward_term_baseline_dev": cfg.a_baseline_dev * _mean_abs_diff(
                final_dampers,
                baseline_dampers),
            "reward_term_spring_dev": cfg.a_spring_dev * _mean_abs_diff(
                final_springs,
                [1.0 for _ in final_springs]),
        }

    def _safety_terms(self, transition: RewardTransition) -> Dict[str, float]:
        info = dict(transition.task_info or {})
        state_values = getattr(transition.state, "as_dict", lambda: {})().values()
        nonfinite = any(
            isinstance(value, (int, float)) and not math.isfinite(float(value))
            for value in state_values)
        return {
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

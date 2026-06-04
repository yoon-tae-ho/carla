"""Gymnasium-compatible scaffold for residual-RL suspension training."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

from ..controllers.base import (
    ControllerContext,
    SuspensionCommand,
    VehicleState,
    WheelScale,
    clamp,
)
from ..controllers.skyhook import SkyhookController
from .observations import ObservationBuilder
from .reward import RewardTransition, SuspensionReward, SuspensionRewardConfig


try:  # Optional dependency.
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - depends on environment.
    gym = None
    spaces = None
    np = None


_BaseEnv = gym.Env if gym is not None else object


class SuspensionCarlaEnv(_BaseEnv):
    """Dry-run environment plus placeholder for future real-CARLA training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        dry_run: bool = True,
        max_steps: int = 100,
        dt: float = 0.05,
        reward_config: Optional[SuspensionRewardConfig] = None,
        residual_scale: float = 0.08,
        wheel_count: int = 4,
        initial_transition_skip_seconds: float = 3.0,
    ):
        if not dry_run:
            raise NotImplementedError(
                "real CARLA training is a scaffold; use dry_run=True")
        self.dry_run = bool(dry_run)
        self.max_steps = int(max_steps)
        self.dt = float(dt)
        self.residual_scale = float(residual_scale)
        self.wheel_count = int(wheel_count)
        self.initial_transition_skip_seconds = max(
            0.0,
            float(initial_transition_skip_seconds))
        self.reward = SuspensionReward(reward_config)
        self.observation_builder = ObservationBuilder(wheel_count=self.wheel_count)
        self.baseline_controller = SkyhookController()
        self.step_index = 0
        self.previous_state = None
        self.state = None
        self.previous_action = [0.0 for _ in range(self.wheel_count)]
        self.previous_damper_scales = None
        spec = self.observation_builder.spec()
        if spaces is not None and np is not None:
            self.observation_space = spaces.Box(
                low=-5.0,
                high=5.0,
                shape=(len(spec.feature_names),),
                dtype=np.float32)
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.wheel_count,),
                dtype=np.float32)
        else:
            self.observation_space = None
            self.action_space = None

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        del seed, options
        self.step_index = 0
        self.previous_state = None
        self.state = self._fake_state(0, speed=8.0)
        self.previous_action = [0.0 for _ in range(self.wheel_count)]
        self.previous_damper_scales = None
        self.baseline_controller.reset()
        observation, info = self._observation()
        return _array(observation), info

    def step(self, action: Sequence[float]):
        if self.state is None:
            self.reset()
        action_values = _action_values(action, self.wheel_count)
        assert self.state is not None
        context = ControllerContext(
            state=self.state,
            previous_state=self.previous_state,
            step=self.step_index,
            dt=self.dt)
        baseline_output = self.baseline_controller.compute(context)
        baseline_command = baseline_output.command
        final_command = _apply_residual(
            baseline_command,
            action_values,
            self.residual_scale)
        next_state = self._advance_state(self.state, action_values)
        transition = RewardTransition(
            state=next_state,
            previous_state=self.state,
            action=action_values,
            previous_action=self.previous_action,
            final_command=final_command,
            baseline_command=baseline_command,
            task_info={},
            diagnostics=baseline_output.diagnostics,
            terminal=(self.step_index + 1) >= self.max_steps,
            dt=self.dt)
        reward_value, reward_diag = self.reward.compute(transition)
        reward_value = self._mask_initial_reward(next_state, reward_value, reward_diag)

        self.previous_state = self.state
        self.state = next_state
        self.previous_action = list(action_values)
        self.previous_damper_scales = [
            wheel.damper_scale for wheel in final_command.wheels
        ]
        self.step_index += 1
        observation, obs_info = self._observation()
        terminated = self.step_index >= self.max_steps
        truncated = False
        info: Dict[str, Any] = {
            "baseline_command": baseline_command,
            "final_command": final_command,
            "reward_diagnostics": reward_diag,
            "observation_diagnostics": obs_info,
            "controller_diagnostics": baseline_output.diagnostics,
        }
        return _array(observation), float(reward_value), terminated, truncated, info

    def close(self) -> None:
        return None

    def _observation(self):
        assert self.state is not None
        context = ControllerContext(
            state=self.state,
            previous_state=self.previous_state,
            step=self.step_index,
            dt=self.dt)
        baseline_output = self.baseline_controller.compute(context)
        return self.observation_builder.build(
            context,
            baseline_output,
            self.previous_action,
            self.previous_damper_scales)

    def _fake_state(self, step: int, speed: float) -> VehicleState:
        time_seconds = step * self.dt
        roll = 1.5 * math.sin(0.7 * time_seconds)
        pitch = 0.8 * math.sin(0.4 * time_seconds)
        lateral = 1.2 * math.sin(0.9 * time_seconds)
        vertical = 0.6 * math.sin(1.7 * time_seconds)
        return VehicleState(
            step=step,
            frame=step,
            elapsed_seconds=time_seconds,
            dt=self.dt,
            x=speed * time_seconds,
            y=0.2 * math.sin(0.2 * time_seconds),
            z=0.0,
            vx=speed,
            vy=0.0,
            vz=0.2 * math.sin(1.2 * time_seconds),
            speed=speed,
            local_vx=speed,
            local_vy=0.0,
            ax=0.0,
            ay=lateral,
            az=vertical,
            local_ax=0.0,
            local_ay=lateral,
            roll=roll,
            pitch=pitch,
            yaw=0.0,
            roll_rate=1.05 * math.cos(0.7 * time_seconds),
            pitch_rate=0.32 * math.cos(0.4 * time_seconds),
            yaw_rate=2.0 * math.sin(0.3 * time_seconds),
            throttle=0.3,
            brake=0.0,
            steer=0.05 * math.sin(0.5 * time_seconds))

    def _advance_state(
        self,
        state: VehicleState,
        action: Sequence[float],
    ) -> VehicleState:
        mean_abs_action = sum(abs(value) for value in action) / float(len(action))
        speed = max(0.0, state.speed + 0.02 - 0.03 * mean_abs_action)
        return self._fake_state(self.step_index + 1, speed=speed)

    def _mask_initial_reward(
        self,
        state: VehicleState,
        reward_value: float,
        reward_diag: Dict[str, float],
    ) -> float:
        if state.elapsed_seconds < self.initial_transition_skip_seconds:
            reward_diag["reward_total_unmasked"] = float(reward_value)
            reward_diag["reward_total"] = 0.0
            reward_diag["reward_initial_masked"] = 1.0
            reward_diag["reward_initial_skip_seconds"] = (
                self.initial_transition_skip_seconds)
            return 0.0
        reward_diag["reward_total_unmasked"] = float(reward_value)
        reward_diag["reward_initial_masked"] = 0.0
        reward_diag["reward_initial_skip_seconds"] = (
            self.initial_transition_skip_seconds)
        return reward_value


def _action_values(action: Sequence[float], wheel_count: int) -> Tuple[float, ...]:
    values = list(action or ())
    result = []
    for value in values[:wheel_count]:
        try:
            result.append(clamp(float(value), -1.0, 1.0))
        except (TypeError, ValueError):
            result.append(0.0)
    while len(result) < wheel_count:
        result.append(0.0)
    return tuple(result)


def _apply_residual(
    baseline_command: SuspensionCommand,
    action: Sequence[float],
    residual_scale: float,
) -> SuspensionCommand:
    wheels = []
    for wheel, action_value in zip(baseline_command.wheels, action):
        wheels.append(WheelScale(
            spring_scale=1.0,
            damper_scale=clamp(
                wheel.damper_scale + float(action_value) * residual_scale,
                0.80,
                1.30)))
    return SuspensionCommand(tuple(wheels)).validate(
        expected_wheels=len(baseline_command.wheels))


def _array(values):
    if np is None:
        return list(values)
    return np.asarray(values, dtype=np.float32)


SuspensionSyntheticEnv = SuspensionCarlaEnv

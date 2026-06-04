"""Backend-driven Gymnasium environment for online suspension RL."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..controllers.base import ControllerContext, PlanningInfo
from ..controllers.pid import FeedbackPIDController
from ..controllers.skyhook import SkyhookController
from .action_projection import (
    ResidualActionProjector,
    ResidualActionProjectorConfig,
)
from .carla_backend import CarlaSuspensionBackend
from .observations import ObservationBuilder
from .reward import RewardTransition, SuspensionReward, SuspensionRewardConfig
from .rollout_logger import build_rollout_row
from .task_info import normalize_task_info, task_done_reason


try:  # Optional dependency.
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - depends on optional packages.
    gym = None
    spaces = None
    np = None


_BaseEnv = gym.Env if gym is not None else object


class CarlaSuspensionEnv(_BaseEnv):
    """CARLA-online suspension RL env backed by a pluggable backend."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: CarlaSuspensionBackend,
        baseline: str = "skyhook",
        planning_provider: str = "empty",
        reward_config: Optional[SuspensionRewardConfig] = None,
        projector_config: Optional[ResidualActionProjectorConfig] = None,
        warmup_seconds: float = 3.0,
        max_episode_steps: Optional[int] = None,
        verify_suspension: bool = False,
        verify_every: Optional[int] = None,
        route_id: str = "00",
        seed: int = 0,
    ):
        self.backend = backend
        self.baseline_name = str(baseline or "skyhook").strip().lower()
        self.planning_provider = str(planning_provider or "empty")
        self.reward = SuspensionReward(reward_config)
        self.projector = ResidualActionProjector(projector_config)
        self.wheel_count = self.projector.config.wheel_count
        self.observation_builder = ObservationBuilder(wheel_count=self.wheel_count)
        self.baseline_controller = self._make_baseline_controller()
        self.warmup_seconds = max(0.0, float(warmup_seconds))
        self.max_episode_steps = (
            max(0, int(max_episode_steps))
            if max_episode_steps is not None
            else 0)
        self.verify_suspension = bool(verify_suspension)
        self.verify_every = (
            max(0, int(verify_every))
            if verify_every is not None
            else 1 if self.verify_suspension else 0)
        self.route_id = str(route_id)
        self.seed_value = int(seed)

        self.state = None
        self.previous_state = None
        self.planning: Optional[PlanningInfo] = PlanningInfo.empty()
        self.task_info: Mapping[str, Any] = {}
        self.previous_action = [0.0 for _ in range(self.wheel_count)]
        self.previous_damper_scales = None
        self.episode_step = 0
        self.warmup_skipped_steps = 0
        self.suspension_apply_count = 0
        self.last_backend_info: Mapping[str, Any] = {}

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

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        options = dict(options or {})
        if seed is not None:
            self.seed_value = int(seed)
        route_id = str(options.get("route_id", self.route_id))
        reset_result = self.backend.reset(self.seed_value, route_id)
        self.route_id = route_id
        self.state = reset_result.state
        self.previous_state = None
        self.planning = self._effective_planning(reset_result.planning)
        self.task_info = self._require_task_info(reset_result.task_info)
        self.last_backend_info = dict(reset_result.info or {})
        self.baseline_controller.reset(self.backend.get_native_suspension())
        self.projector.reset()
        self.previous_action = [0.0 for _ in range(self.wheel_count)]
        self.previous_damper_scales = None
        self.episode_step = 0
        self.warmup_skipped_steps = 0
        self.suspension_apply_count = 0

        self._run_warmup()
        observation, obs_info = self._build_observation()
        info = {
            "task_info": dict(self.task_info),
            "observation_diagnostics": obs_info,
            "backend_info": dict(self.last_backend_info or {}),
            "warmup_skipped_steps": self.warmup_skipped_steps,
            "suspension_apply_count": self.suspension_apply_count,
            "verify_every": self.verify_every,
            "max_episode_steps": self.max_episode_steps,
            "route_id": self.route_id,
            "seed": self.seed_value,
        }
        return _array(observation), info

    def step(self, action: Sequence[float]):
        if self.state is None:
            self.reset()
        assert self.state is not None
        current_state = self.state
        previous_action = list(self.previous_action)

        context = self._context(current_state, self.previous_state, self.planning)
        baseline_output = self.baseline_controller.compute(context)
        observation, obs_diag = self.observation_builder.build(
            context,
            baseline_output,
            self.previous_action,
            self.previous_damper_scales)
        del observation
        projection = self.projector.project(
            baseline_output.command,
            action,
            current_state,
            observation_valid=bool(obs_diag.get("observation_valid", 0)))
        verify = self._next_apply_verify()
        apply_info = self.backend.apply_suspension(
            projection.command,
            verify=verify)
        control_info = self.backend.apply_autonomous_control()
        step_result = self.backend.tick()
        task_info = self._require_task_info(step_result.task_info)
        next_state = step_result.state
        reward_value, reward_diag = self.reward.compute(RewardTransition(
            state=next_state,
            previous_state=current_state,
            action=projection.sanitized_action,
            previous_action=previous_action,
            previous_final_damper_scales=self.previous_damper_scales or (),
            final_command=projection.command,
            baseline_command=baseline_output.command,
            task_info=task_info,
            diagnostics=_merged_diagnostics(
                baseline_output.diagnostics,
                projection.diagnostics,
                obs_diag),
            terminal=bool(step_result.terminated),
            dt=max(float(getattr(next_state, "dt", 0.05) or 0.05), 1.0e-9)))

        self.previous_state = current_state
        self.state = next_state
        self.planning = self._effective_planning(step_result.planning)
        self.task_info = task_info
        self.last_backend_info = dict(step_result.info or {})
        self.previous_action = (
            [0.0 for _ in range(self.wheel_count)]
            if projection.fallback_reason
            else list(projection.sanitized_action))
        self.previous_damper_scales = list(projection.final_damper_per_wheel)
        self.episode_step += 1

        next_observation, next_obs_info = self._build_observation()
        terminated = bool(step_result.terminated)
        max_step_truncated = (
            self.max_episode_steps > 0 and
            self.episode_step >= self.max_episode_steps and
            not terminated)
        truncated = bool(step_result.truncated or max_step_truncated)
        done_reason = task_done_reason(task_info) if terminated else ""
        truncated_reason = (
            "max_episode_steps"
            if max_step_truncated else
            str(step_result.info.get("truncated_reason", "truncated"))
            if step_result.truncated else "")
        backend_info = _merged_diagnostics(
            apply_info,
            control_info,
            step_result.info)
        info = {
            "task_info": dict(task_info),
            "baseline_command": baseline_output.command,
            "final_command": projection.command,
            "projection_diagnostics": dict(projection.diagnostics),
            "reward_diagnostics": reward_diag,
            "observation_diagnostics": next_obs_info,
            "controller_diagnostics": _merged_diagnostics(
                baseline_output.diagnostics,
                projection.diagnostics,
                obs_diag),
            "backend_info": backend_info,
            "warmup_skipped_steps": self.warmup_skipped_steps,
            "suspension_apply_count": self.suspension_apply_count,
            "verify_every": self.verify_every,
            "max_episode_steps": self.max_episode_steps,
            "route_id": self.route_id,
            "seed": self.seed_value,
            "done_reason": done_reason,
            "truncated_reason": truncated_reason,
        }
        info["rollout_row"] = build_rollout_row(
            episode_id="%s_seed%d" % (self.route_id, self.seed_value),
            step=self.episode_step - 1,
            state=next_state,
            planning=self.planning,
            baseline_command=baseline_output.command,
            action=projection.sanitized_action,
            final_command=projection.command,
            reward_diagnostics=reward_diag,
            controller_diagnostics=info["controller_diagnostics"],
            extra=_merged_diagnostics(
                task_info,
                backend_info,
                {
                    "route_id": self.route_id,
                    "seed": self.seed_value,
                    "terminated": int(terminated),
                    "truncated": int(truncated),
                    "done_reason": done_reason,
                    "truncated_reason": truncated_reason,
                }))
        return (
            _array(next_observation),
            float(reward_value),
            terminated,
            truncated,
            info)

    def close(self) -> None:
        self.backend.close()

    def _run_warmup(self) -> None:
        elapsed = 0.0
        zero_action = [0.0 for _ in range(self.wheel_count)]
        while elapsed < self.warmup_seconds - 1.0e-12:
            if self.state is None:
                break
            current = self.state
            dt = max(float(getattr(current, "dt", 0.05) or 0.05), 1.0e-9)
            context = self._context(current, self.previous_state, self.planning)
            baseline_output = self.baseline_controller.compute(context)
            _, obs_diag = self.observation_builder.build(
                context,
                baseline_output,
                self.previous_action,
                self.previous_damper_scales)
            projection = self.projector.project(
                baseline_output.command,
                zero_action,
                current,
                observation_valid=bool(obs_diag.get("observation_valid", 0)))
            verify = self._next_apply_verify()
            self.backend.apply_suspension(
                projection.command,
                verify=verify)
            self.backend.apply_autonomous_control()
            step_result = self.backend.tick()
            self.previous_state = current
            self.state = step_result.state
            self.planning = self._effective_planning(step_result.planning)
            self.task_info = self._require_task_info(step_result.task_info)
            self.last_backend_info = dict(step_result.info or {})
            self.previous_action = list(projection.sanitized_action)
            self.previous_damper_scales = list(projection.final_damper_per_wheel)
            self.warmup_skipped_steps += 1
            elapsed += dt
            if step_result.terminated or step_result.truncated:
                break

    def _next_apply_verify(self) -> bool:
        self.suspension_apply_count += 1
        return (
            self.verify_every > 0 and
            self.suspension_apply_count % self.verify_every == 0)

    def _build_observation(self) -> Tuple[Sequence[float], Dict[str, Any]]:
        assert self.state is not None
        context = self._context(self.state, self.previous_state, self.planning)
        baseline_output = self.baseline_controller.compute(context)
        return self.observation_builder.build(
            context,
            baseline_output,
            self.previous_action,
            self.previous_damper_scales)

    def _context(
        self,
        state: Any,
        previous_state: Any,
        planning: Optional[PlanningInfo],
    ) -> ControllerContext:
        return ControllerContext(
            state=state,
            previous_state=previous_state,
            planning=planning or PlanningInfo.empty(),
            native_suspension=self.backend.get_native_suspension(),
            current_suspension=self.backend.get_current_suspension(),
            step=getattr(state, "step", self.episode_step),
            dt=max(float(getattr(state, "dt", 0.05) or 0.05), 1.0e-9))

    def _effective_planning(
        self,
        planning: Optional[PlanningInfo],
    ) -> PlanningInfo:
        if self.planning_provider == "empty":
            return PlanningInfo.empty()
        return planning or PlanningInfo.empty()

    def _require_task_info(self, task_info: Mapping[str, Any]) -> Mapping[str, Any]:
        return normalize_task_info(task_info, require_non_empty=True)

    def _make_baseline_controller(self):
        if self.baseline_name == "skyhook":
            return SkyhookController()
        if self.baseline_name == "pid":
            return FeedbackPIDController()
        raise ValueError("unknown baseline %s" % self.baseline_name)


def _merged_diagnostics(*items: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in items:
        result.update(dict(item or {}))
    return result


def _array(values):
    if np is None:
        return list(values)
    return np.asarray(values, dtype=np.float32)

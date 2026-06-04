"""Backend interfaces for online CARLA suspension RL environments."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from ..controllers.base import (
    PlanningInfo,
    SuspensionCommand,
    VehicleState,
)
from ..runtime.carla_adapter import (
    apply_suspension_command,
    import_carla,
    read_suspension_scale_summary,
    read_vehicle_state,
    validate_suspension_control,
)
from ..runtime.planning_provider import (
    ControlHistoryPlanningInfoProvider,
    EmptyPlanningInfoProvider,
    JsonlPlanningInfoProvider,
    PlanningInfoProvider,
)
from ..runtime.route_progress import RouteProgressTracker, unavailable_route_progress
from .task_info import TaskInfoBuilder


LIVE_BACKEND_UNAVAILABLE_MESSAGE = (
    "LiveRouteProcessBackend dependencies are unavailable.\n"
    "This code path must not fall back to synthetic training.\n"
    "Run inside the Docker runner with CARLA, scenario_runner, leaderboard, "
    "and the route runner environment loaded before --env carla training.")


@dataclass(frozen=True)
class BackendResetResult:
    state: VehicleState
    planning: Optional[PlanningInfo]
    task_info: Mapping[str, Any]
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendStepResult:
    state: VehicleState
    planning: Optional[PlanningInfo]
    task_info: Mapping[str, Any]
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


class CarlaSuspensionBackend(Protocol):
    def reset(self, seed: int, route_id: str) -> BackendResetResult:
        ...

    def get_state(self) -> VehicleState:
        ...

    def get_planning(self) -> Optional[PlanningInfo]:
        ...

    def get_task_info(self) -> Mapping[str, Any]:
        ...

    def get_native_suspension(self) -> Any:
        ...

    def get_current_suspension(self) -> Any:
        ...

    def apply_suspension(
        self,
        command: SuspensionCommand,
        verify: bool,
    ) -> Mapping[str, Any]:
        ...

    def apply_autonomous_control(self) -> Mapping[str, Any]:
        ...

    def tick(self) -> BackendStepResult:
        ...

    def close(self) -> None:
        ...


BaseCarlaRouteBackend = CarlaSuspensionBackend


class FakeCarlaSuspensionBackend:
    """Deterministic CARLA-free backend for unit and smoke tests."""

    def __init__(
        self,
        dt: float = 0.05,
        max_steps: int = 200,
        target_speed: float = 8.0,
        route_length: float = 100.0,
        planning_provider: str = "empty",
    ):
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.target_speed = float(target_speed)
        self.route_length = max(float(route_length), 1.0)
        self.planning_provider = str(planning_provider or "empty")
        self.seed = 0
        self.route_id = "00"
        self.step_index = 0
        self.progress = 0.0
        self.last_progress = 0.0
        self.speed = self.target_speed
        self.last_command = SuspensionCommand.identity()
        self.state = self._state()
        self.task_info = self._task_info(delta_progress_m=0.0)
        self.closed = False

    def reset(self, seed: int, route_id: str) -> BackendResetResult:
        self.seed = int(seed)
        self.route_id = str(route_id)
        self.step_index = 0
        self.progress = 0.0
        self.last_progress = 0.0
        self.speed = self.target_speed
        self.last_command = SuspensionCommand.identity()
        self.state = self._state()
        self.task_info = self._task_info(delta_progress_m=0.0)
        return BackendResetResult(
            state=self.state,
            planning=self.get_planning(),
            task_info=self.task_info,
            info={
                "backend": "fake",
                "real_backend_used": 0,
                "fake_backend_used": 1,
                "carla_connected": 0,
                "route_process_started": 0,
                "hero_attached": 0,
                "route_process_alive": 0,
                "route_id": self.route_id,
                "seed": self.seed,
            })

    def get_state(self) -> VehicleState:
        return self.state

    def get_planning(self) -> Optional[PlanningInfo]:
        if self.planning_provider == "empty":
            return PlanningInfo.empty()
        return PlanningInfo(
            available=True,
            source=self.planning_provider,
            frame=self.state.frame,
            target_speed=(self.target_speed,) * 5)

    def get_task_info(self) -> Mapping[str, Any]:
        return dict(self.task_info)

    def get_native_suspension(self) -> Any:
        return {"spring_scale": 1.0, "damper_scale": 1.0}

    def get_current_suspension(self) -> Any:
        return self.last_command

    def apply_suspension(
        self,
        command: SuspensionCommand,
        verify: bool,
    ) -> Mapping[str, Any]:
        self.last_command = command.validate(expected_wheels=len(command.wheels))
        dampers = [wheel.damper_scale for wheel in self.last_command.wheels]
        springs = [wheel.spring_scale for wheel in self.last_command.wheels]
        return {
            "backend_apply_suspension": 1,
            "backend_verify_requested": int(bool(verify)),
            "readback_mean_damper_scale": _mean(dampers),
            "readback_mean_spring_scale": _mean(springs),
        }

    def apply_autonomous_control(self) -> Mapping[str, Any]:
        speed_error = self.target_speed - self.speed
        throttle = max(0.0, min(1.0, 0.25 + 0.04 * speed_error))
        brake = max(0.0, min(1.0, -0.05 * speed_error))
        return {
            "backend_apply_autonomous_control": 1,
            "backend_throttle": throttle,
            "backend_brake": brake,
        }

    def tick(self) -> BackendStepResult:
        self.step_index += 1
        self.last_progress = self.progress
        damping = _mean([wheel.damper_scale for wheel in self.last_command.wheels])
        damper_deviation = abs(damping - 1.0)
        speed_error = self.target_speed - self.speed
        self.speed = max(
            0.0,
            self.speed + 0.04 * speed_error - 0.02 * damper_deviation)
        delta_m = self.speed * self.dt
        delta_fraction = delta_m / self.route_length
        self.progress = min(1.0, self.progress + delta_fraction)
        self.state = self._state()
        self.task_info = self._task_info(delta_progress_m=delta_m)
        terminated = self.step_index >= self.max_steps or self.progress >= 1.0
        return BackendStepResult(
            state=self.state,
            planning=self.get_planning(),
            task_info=self.task_info,
            terminated=terminated,
            truncated=False,
            info={
                "backend": "fake",
                "real_backend_used": 0,
                "fake_backend_used": 1,
                "carla_connected": 0,
                "route_process_started": 0,
                "hero_attached": 0,
                "route_process_alive": 0,
                "step_index": self.step_index,
                "mean_damper_scale": damping,
            })

    def close(self) -> None:
        self.closed = True

    def _state(self) -> VehicleState:
        time_seconds = self.step_index * self.dt
        damping = _mean([wheel.damper_scale for wheel in self.last_command.wheels])
        damping_effect = max(-0.5, min(0.5, damping - 1.0))
        lateral = (1.1 - 0.15 * damping_effect) * math.sin(0.9 * time_seconds)
        vertical = (0.55 - 0.20 * damping_effect) * math.sin(1.7 * time_seconds)
        return VehicleState(
            step=self.step_index,
            frame=self.step_index,
            elapsed_seconds=time_seconds,
            dt=self.dt,
            x=self.progress * self.route_length,
            y=0.2 * math.sin(0.2 * time_seconds),
            z=0.0,
            vx=self.speed,
            vy=0.0,
            vz=0.15 * math.sin(1.2 * time_seconds),
            speed=self.speed,
            local_vx=self.speed,
            local_vy=0.0,
            ax=0.0,
            ay=lateral,
            az=vertical,
            local_ax=0.0,
            local_ay=lateral,
            roll=1.2 * math.sin(0.7 * time_seconds),
            pitch=0.7 * math.sin(0.4 * time_seconds),
            yaw=0.0,
            roll_rate=0.84 * math.cos(0.7 * time_seconds),
            pitch_rate=0.28 * math.cos(0.4 * time_seconds),
            yaw_rate=1.5 * math.sin(0.3 * time_seconds),
            throttle=0.3,
            brake=0.0,
            steer=0.04 * math.sin(0.5 * time_seconds))

    def _task_info(self, delta_progress_m: float) -> Dict[str, float]:
        target_speed_error = self.target_speed - self.speed
        return {
            "route_progress_available": 1.0,
            "route_progress_fraction": self.progress,
            "route_progress_monotonic_fraction": self.progress,
            "route_progress_m": self.progress * self.route_length,
            "route_progress_delta_m": delta_progress_m,
            "route_delta_progress_m": delta_progress_m,
            "route_progress_rate_mps": self.speed,
            "route_distance_to_end_m": max(
                0.0,
                self.route_length - self.progress * self.route_length),
            "route_completion_proxy": self.progress,
            "delta_progress": delta_progress_m,
            "target_speed": self.target_speed,
            "target_speed_source": "fake",
            "speed_error": target_speed_error,
            "abs_speed_error": abs(target_speed_error),
            "target_speed_error": target_speed_error,
            "route_deviation": abs(self.state.y) if hasattr(self, "state") else 0.0,
            "route_deviation_m": abs(self.state.y) if hasattr(self, "state") else 0.0,
            "route_deviation_valid": 1.0,
            "lane_invasion_count": 0.0,
            "collision_count": 0.0,
            "red_light_count": 0.0,
            "blocked_vehicle": 0.0,
            "route_timeout": 0.0,
            "low_speed_not_planned": 1.0 if self.speed < 0.5 else 0.0,
            "route_failed": 0.0,
        }


FakeCarlaRouteBackend = FakeCarlaSuspensionBackend


class LiveCarlaSuspensionBackend:
    """Passive external-clock backend for route-runner owned CARLA episodes."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2000,
        timeout: float = 10.0,
        routes: str = "",
        routes_subset: str = "00",
        route_script: str = "",
        output_dir: str = "",
        team_agent: str = "",
        team_config: str = "",
        baseline: str = "skyhook",
        planning_provider: str = "empty",
        planning_preview_jsonl: str = "",
        traffic_manager_port: int = 8000,
        fixed_delta_seconds: float = 0.05,
        seed: int = 0,
        role_name: str = "hero",
        actor_id: str = "",
        verify_suspension: bool = False,
        readback_tolerance: float = 1.0e-4,
        debug: int = 0,
        repetitions: int = 1,
        connect_retry_seconds: float = 0.25,
        hero_timeout_seconds: float = 300.0,
        route_wait_timeout_seconds: float = 10.0,
        restore_native_on_close: bool = True,
        required_route_modules: Sequence[str] = ("srunner", "leaderboard"),
        carla_module: Any = None,
        carla_importer: Optional[Callable[[], Any]] = None,
        dependency_importer: Optional[Callable[[str], Any]] = None,
        popen_factory: Optional[Callable[..., Any]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        time_fn: Callable[[], float] = time.monotonic,
        **extra: Any,
    ):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)
        self.routes = os.path.abspath(os.path.expanduser(
            str(routes or _default_routes_path())))
        self.routes_subset = str(routes_subset or "")
        self.route_script = os.path.abspath(os.path.expanduser(
            str(route_script or _default_route_script_path())))
        self.output_dir = os.path.abspath(os.path.expanduser(
            str(output_dir or os.path.join(
                "/tmp",
                "rl_suspension_phase4_live_route"))))
        self.team_agent = str(team_agent or "")
        self.team_config = str(team_config or "")
        self.baseline = str(baseline or "skyhook")
        self.planning_provider = str(planning_provider or "empty")
        self.planning_preview_jsonl = str(planning_preview_jsonl or "")
        self.traffic_manager_port = int(traffic_manager_port)
        self.fixed_delta_seconds = float(fixed_delta_seconds)
        self.seed = int(seed)
        self.role_name = str(role_name or "hero")
        self.actor_id = str(actor_id or "")
        self.verify_suspension = bool(verify_suspension)
        self.readback_tolerance = float(readback_tolerance)
        self.debug = int(debug)
        self.repetitions = int(repetitions)
        self.connect_retry_seconds = max(0.0, float(connect_retry_seconds))
        self.hero_timeout_seconds = max(0.0, float(hero_timeout_seconds))
        self.route_wait_timeout_seconds = max(0.0, float(route_wait_timeout_seconds))
        self.restore_native_on_close = bool(restore_native_on_close)
        self.required_route_modules = tuple(required_route_modules or ())
        self.carla = carla_module
        self._carla_importer = carla_importer
        self._dependency_importer = dependency_importer or __import__
        self._popen_factory = popen_factory or subprocess.Popen
        self._sleep = sleep_fn
        self._time = time_fn
        self.extra = dict(extra or {})
        self.live_backend_available = True

        self.route_output_dir = os.path.join(self.output_dir, "tfpp")
        self.route_stdout_log = os.path.join(self.output_dir, "route_stdout.log")
        self.route_result_path = os.path.join(self.route_output_dir, "result.json")
        self.route_debug_result_path = os.path.join(
            self.route_output_dir,
            "live_results.txt")
        self.route_process = None
        self._route_log_file = None
        self.client = None
        self.world = None
        self.vehicle = None
        self.native_suspension = None
        self.previous_state: Optional[VehicleState] = None
        self.current_state: Optional[VehicleState] = None
        self.current_planning: PlanningInfo = PlanningInfo.empty()
        self.current_task_info: Mapping[str, Any] = {}
        self.step_index = 0
        self.route_progress_tracker: Optional[RouteProgressTracker] = None
        self.task_info_builder = TaskInfoBuilder()
        self.planning_info_provider = self._make_planning_provider()
        self.unavailable_message = self._dependency_unavailable_message()
        if self.unavailable_message:
            self.live_backend_available = False

    def config_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "timeout": self.timeout,
            "routes": self.routes,
            "routes_subset": self.routes_subset,
            "route_script": self.route_script,
            "output_dir": self.output_dir,
            "route_output_dir": self.route_output_dir,
            "route_stdout_log": self.route_stdout_log,
            "route_result_path": self.route_result_path,
            "route_debug_result_path": self.route_debug_result_path,
            "team_agent": self.team_agent,
            "team_config": self.team_config,
            "baseline": self.baseline,
            "planning_provider": self.planning_provider,
            "planning_preview_jsonl": self.planning_preview_jsonl,
            "traffic_manager_port": self.traffic_manager_port,
            "fixed_delta_seconds": self.fixed_delta_seconds,
            "seed": self.seed,
            "role_name": self.role_name,
            "actor_id": self.actor_id,
            "verify_suspension": self.verify_suspension,
            "readback_tolerance": self.readback_tolerance,
            "debug": self.debug,
            "repetitions": self.repetitions,
            "connect_retry_seconds": self.connect_retry_seconds,
            "hero_timeout_seconds": self.hero_timeout_seconds,
            "route_wait_timeout_seconds": self.route_wait_timeout_seconds,
            "live_backend_available": self.live_backend_available,
            "unavailable_message": self.unavailable_message,
        }

    def reset(self, seed: int, route_id: str) -> BackendResetResult:
        self._ensure_available()
        self.seed = int(seed)
        self.routes_subset = str(route_id or self.routes_subset)
        self.step_index = 0
        self.previous_state = None
        self.current_state = None
        self.current_planning = PlanningInfo.empty()
        self.current_task_info = {}
        self.task_info_builder.reset(self._actor_key())
        self.route_progress_tracker = RouteProgressTracker(
            route_xml_path=self.routes,
            route_id=self.routes_subset)
        self.start_route_process(seed=self.seed, route_id=self.routes_subset)
        self._connect_world()
        self.vehicle = self.wait_for_hero()
        self.native_suspension = self.vehicle.get_suspension_physics_control()
        validate_suspension_control(self.native_suspension)
        self.current_state = self.get_state()
        self.current_planning = self.get_planning()
        self.current_task_info = self.get_task_info()
        return BackendResetResult(
            state=self.current_state,
            planning=self.current_planning,
            task_info=self.current_task_info,
            info=self.backend_info())

    def get_state(self) -> VehicleState:
        self._ensure_attached()
        control = (
            self.vehicle.get_control()
            if hasattr(self.vehicle, "get_control")
            else None)
        state = read_vehicle_state(
            self.world,
            self.vehicle,
            step=self.step_index,
            previous_state=self.previous_state,
            control=control)
        return state

    def get_planning(self) -> Optional[PlanningInfo]:
        state = self.current_state or self.get_state()
        return self.planning_info_provider.get(
            frame=getattr(state, "frame", -1),
            state=state,
            previous_state=self.previous_state)

    def get_task_info(self) -> Mapping[str, Any]:
        state = self.current_state or self.get_state()
        planning = self.current_planning or PlanningInfo.empty()
        route_progress = (
            self.route_progress_tracker.update(
                state,
                actor_id=self._actor_key(),
                frame=getattr(state, "frame", None))
            if self.route_progress_tracker is not None
            else unavailable_route_progress("route_tracker_unavailable"))
        return self.task_info_builder.build(
            state=state,
            previous_state=self.previous_state,
            route_progress=route_progress,
            planning=planning,
            live_task_info={},
            actor_id=self._actor_key())

    def get_native_suspension(self) -> Any:
        self._ensure_attached()
        return self.native_suspension

    def get_current_suspension(self) -> Any:
        self._ensure_attached()
        return self.vehicle.get_suspension_physics_control()

    def apply_suspension(
        self,
        command: SuspensionCommand,
        verify: bool,
    ) -> Mapping[str, Any]:
        self._ensure_attached()
        result = apply_suspension_command(
            self.vehicle,
            self.native_suspension,
            command,
            verify_readback=bool(verify),
            readback_tolerance=self.readback_tolerance,
            carla_module=self.carla)
        readback = result.get("readback")
        if readback is None:
            readback = self.vehicle.get_suspension_physics_control()
        summary = result.get("readback_summary")
        if summary is None:
            summary = read_suspension_scale_summary(
                self.native_suspension,
                readback)
        return {
            "backend": "live",
            "backend_apply_suspension": 1,
            "backend_verify_requested": int(bool(verify)),
            "readback_mean_damper_scale": summary.get("mean_damper_scale", ""),
            "readback_mean_spring_scale": summary.get("mean_spring_scale", ""),
            "readback_min_damper_scale": summary.get("min_damper_scale", ""),
            "readback_max_damper_scale": summary.get("max_damper_scale", ""),
            "route_process_alive": int(self.route_process_alive()),
        }

    def apply_autonomous_control(self) -> Mapping[str, Any]:
        # The leaderboard/TransFuser++ route process owns vehicle control.
        return {
            "backend": "live",
            "backend_apply_autonomous_control": 0,
            "route_process_alive": int(self.route_process_alive()),
        }

    def tick(self) -> BackendStepResult:
        self._ensure_attached()
        before = self.current_state
        self.wait_next_frame()
        self.step_index += 1
        self.previous_state = before
        self.current_state = self.get_state()
        self.current_planning = self.get_planning()
        self.current_task_info = self.get_task_info()
        route_alive = self.route_process_alive()
        route_returncode = self.route_process_returncode()
        terminated = self.route_process is not None and not route_alive
        info = self.backend_info()
        if terminated:
            info.update({
                "route_process_returncode": route_returncode,
                "route_result_available": int(os.path.isfile(self.route_result_path)),
                "route_result": self._read_route_result(),
            })
        return BackendStepResult(
            state=self.current_state,
            planning=self.current_planning,
            task_info=self.current_task_info,
            terminated=terminated,
            truncated=False,
            info=info)

    def close(self) -> None:
        if (
                self.restore_native_on_close and
                self.vehicle is not None and
                self.native_suspension is not None):
            try:
                self.vehicle.apply_suspension_physics_control(
                    self.native_suspension)
            except Exception:
                pass
        if self.route_process is not None and self.route_process_alive():
            try:
                self.route_process.terminate()
                self.route_process.wait(timeout=10.0)
            except Exception:
                try:
                    self.route_process.kill()
                except Exception:
                    pass
        if self._route_log_file is not None:
            try:
                self._route_log_file.close()
            finally:
                self._route_log_file = None

    def build_route_command(self) -> Tuple[str, ...]:
        return ("bash", self.route_script)

    def build_route_env(
        self,
        seed: Optional[int] = None,
        route_id: Optional[str] = None,
    ) -> Dict[str, str]:
        env = dict(os.environ)
        env["PORT"] = str(self.port)
        env["DEBUG"] = str(self.debug)
        env["ROUTES"] = self.routes
        env["ROUTES_SUBSET"] = str(
            self.routes_subset if route_id is None else route_id)
        env["REPETITIONS"] = str(self.repetitions)
        env["TRAFFIC_MANAGER_SEED"] = str(
            self.seed if seed is None else int(seed))
        env["TRAFFIC_MANAGER_PORT"] = str(self.traffic_manager_port)
        env["OUT_DIR"] = self.route_output_dir
        if self.team_agent:
            env["TEAM_AGENT"] = self.team_agent
        if self.team_config:
            env["TEAM_CONFIG"] = self.team_config
        if self.planning_preview_jsonl:
            env["SUSPENSION_PLANNING_PREVIEW_JSONL"] = (
                self.planning_preview_jsonl)
        return env

    def start_route_process(self, seed: int, route_id: str) -> Any:
        self._ensure_available()
        if not os.path.isfile(self.route_script):
            raise RuntimeError(
                "route runner script missing: %s" % self.route_script)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.route_output_dir, exist_ok=True)
        if self._route_log_file is not None:
            self._route_log_file.close()
        self._route_log_file = open(self.route_stdout_log, "w")
        self.route_process = self._popen_factory(
            list(self.build_route_command()),
            stdout=self._route_log_file,
            stderr=subprocess.STDOUT,
            env=self.build_route_env(seed=seed, route_id=route_id),
            cwd=os.path.dirname(self.route_script))
        return self.route_process

    def wait_next_frame(self) -> None:
        if self.world is None:
            raise RuntimeError("CARLA world is not attached")
        if hasattr(self.world, "wait_for_tick"):
            self.world.wait_for_tick(self.route_wait_timeout_seconds)
            return
        self._sleep(min(self.route_wait_timeout_seconds, 0.05))

    def wait_for_hero(self) -> Any:
        if self.world is None:
            raise RuntimeError("CARLA world is not attached")
        deadline = self._time() + self.hero_timeout_seconds
        last_wait_error = ""
        while self._time() <= deadline:
            vehicle = self._find_hero_vehicle()
            if vehicle is not None:
                return vehicle
            if self.route_process is not None and not self.route_process_alive():
                raise RuntimeError(
                    "route process exited before hero attach: returncode=%s" %
                    self.route_process_returncode())
            try:
                self.wait_next_frame()
            except Exception as error:
                last_wait_error = str(error)
                self._sleep(min(max(self.connect_retry_seconds, 0.05), 1.0))
        raise RuntimeError(
            "timed out waiting for hero vehicle role_name=%s actor_id=%s%s" %
            (
                self.role_name,
                self.actor_id,
                " last_wait_error=%s" % last_wait_error
                if last_wait_error else ""))

    def route_process_alive(self) -> bool:
        return self.route_process is not None and self.route_process.poll() is None

    def route_process_returncode(self) -> Any:
        if self.route_process is None:
            return ""
        return self.route_process.poll()

    def backend_info(self) -> Dict[str, Any]:
        return {
            "backend": "live",
            "real_backend_used": 1,
            "fake_backend_used": 0,
            "carla_connected": int(self.world is not None),
            "route_process_started": int(self.route_process is not None),
            "hero_attached": int(self.vehicle is not None),
            "route_process_alive": int(self.route_process_alive()),
            "route_process_pid": getattr(self.route_process, "pid", ""),
            "route_process_returncode": self.route_process_returncode(),
            "route_script": self.route_script,
            "route_output_dir": self.route_output_dir,
            "route_stdout_log": self.route_stdout_log,
            "route_result_path": self.route_result_path,
            "route_debug_result_path": self.route_debug_result_path,
        }

    def _dependency_unavailable_message(self) -> str:
        errors = []
        if self.carla is None:
            try:
                self._load_carla_module()
            except Exception as error:
                errors.append("carla import failed: %s" % error)
        if not os.path.isfile(self.route_script):
            errors.append("route runner script missing: %s" % self.route_script)
        for module_name in self.required_route_modules:
            try:
                self._dependency_importer(module_name)
            except Exception as error:
                errors.append("%s import failed: %s" % (module_name, error))
        if not errors:
            return ""
        return "%s\n- %s" % (
            LIVE_BACKEND_UNAVAILABLE_MESSAGE,
            "\n- ".join(errors))

    def _load_carla_module(self) -> Any:
        if self.carla is None:
            if self._carla_importer is not None:
                self.carla = self._carla_importer()
            else:
                self.carla = import_carla()
        return self.carla

    def _connect_world(self) -> None:
        carla = self._load_carla_module()
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()

    def _find_hero_vehicle(self) -> Any:
        actors = self.world.get_actors()
        vehicles = actors.filter("vehicle.*") if hasattr(actors, "filter") else actors
        actor_id = str(self.actor_id or "")
        for actor in vehicles:
            if actor_id and str(getattr(actor, "id", "")) != actor_id:
                continue
            role_name = getattr(actor, "attributes", {}).get("role_name", "")
            if not actor_id and role_name != self.role_name:
                continue
            if not hasattr(actor, "get_suspension_physics_control"):
                continue
            if not hasattr(actor, "apply_suspension_physics_control"):
                continue
            return actor
        return None

    def _make_planning_provider(self) -> PlanningInfoProvider:
        provider = self.planning_provider.strip().lower()
        if provider == "empty":
            return EmptyPlanningInfoProvider()
        if provider == "jsonl":
            return JsonlPlanningInfoProvider(path=self.planning_preview_jsonl)
        if provider == "control_history":
            return ControlHistoryPlanningInfoProvider()
        raise ValueError("unknown planning provider %s" % self.planning_provider)

    def _actor_key(self) -> str:
        return str(self.actor_id or self.role_name or "hero")

    def _read_route_result(self) -> Mapping[str, Any]:
        if not os.path.isfile(self.route_result_path):
            return {}
        try:
            with open(self.route_result_path) as json_file:
                return json.load(json_file)
        except Exception:
            return {}

    def _ensure_available(self) -> None:
        if self.unavailable_message:
            raise RuntimeError(self.unavailable_message)

    def _ensure_attached(self) -> None:
        self._ensure_available()
        if self.world is None or self.vehicle is None:
            raise RuntimeError("live route backend is not attached to a hero vehicle")


def _mean(values: Any) -> float:
    values = list(values or [])
    return sum(float(value) for value in values) / float(len(values)) if values else 0.0


def _sim_root_from_here() -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        ".."))


def _default_route_script_path() -> str:
    explicit = os.environ.get("PHASE4_ROUTE_SCRIPT", "")
    if explicit:
        return os.path.expanduser(explicit)
    candidates = (
        "/workspace/e2e_models/scripts/run_tfpp_debug_route.sh",
        os.path.join(
            _sim_root_from_here(),
            "e2e_models",
            "scripts",
            "run_tfpp_debug_route.sh"),
    )
    for path in candidates:
        if path and os.path.isfile(os.path.expanduser(path)):
            return os.path.expanduser(path)
    return candidates[1]


def _default_routes_path() -> str:
    explicit = os.environ.get("ROUTES", "")
    if explicit:
        return os.path.expanduser(explicit)
    garage_root = os.environ.get("GARAGE_ROOT", "")
    candidates = (
        os.path.join(
            garage_root,
            "leaderboard",
            "data",
            "suspension_town13_short.xml") if garage_root else "",
        "/workspace/e2e_models/carla_garage/leaderboard/data/suspension_town13_short.xml",
        os.path.join(
            _sim_root_from_here(),
            "e2e_models",
            "carla_garage",
            "leaderboard",
            "data",
            "suspension_town13_short.xml"),
    )
    for path in candidates:
        if path and os.path.isfile(os.path.expanduser(path)):
            return os.path.expanduser(path)
    return candidates[2]

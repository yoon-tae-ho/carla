#!/usr/bin/env python

"""Run TransFuser++ route experiments with suspension-control metrics.

This script wraps the existing ``run_tfpp_debug_route.sh`` command and starts a
CARLA sidecar client for each scenario. The sidecar never ticks the world; it
only waits for leaderboard ticks, records the hero vehicle profile, and
optionally applies a suspension controller command for the next simulator tick.

Default scenarios:

  S0_stock:    no suspension API call, profile recording only
  S1_identity: apply native suspension every tick as an API sanity check
  S2_pid:      apply controllers.pid.FeedbackPIDController every tick

Optional scenario:

  S3_skyhook:  apply controllers.skyhook.SkyhookController every tick
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from suspension_control.controllers.base import ControllerContext, PlanningInfo
from suspension_control.controllers.identity import IdentityController
from suspension_control.controllers.pid import FeedbackPIDConfig, FeedbackPIDController
from suspension_control.controllers.skyhook import SkyhookConfig, SkyhookController
from suspension_control.metrics.comfort import comfort_metrics
from suspension_control.metrics.stability import stability_metrics
from suspension_control.runtime.carla_adapter import (
    apply_suspension_command,
    import_carla,
    read_suspension_scale_summary,
    read_vehicle_state,
    validate_suspension_control,
)


SIM_ROOT = os.environ.get("SIM_ROOT", os.path.expanduser("~/sim"))
DEFAULT_ROUTE_SCRIPT = os.path.join(
    SIM_ROOT, "e2e_models", "scripts", "run_tfpp_debug_route.sh")
DEFAULT_ROUTES = os.path.join(
    SIM_ROOT,
    "e2e_models",
    "carla_garage",
    "leaderboard",
    "data",
    "suspension_town13_short.xml",
)
DEFAULT_TFPP_OUTPUT_ROOT = os.path.join(
    SIM_ROOT, "e2e_models", "outputs", "transfuserpp")
DEFAULT_PID_CONFIG = os.path.join(
    SCRIPT_DIR, "suspension_control", "configs", "pid.yaml")
DEFAULT_SKYHOOK_CONFIG = os.path.join(
    SCRIPT_DIR, "suspension_control", "configs", "skyhook.yaml")


SCENARIOS = OrderedDict((
    ("stock", {
        "name": "S0_stock",
        "label": "S0 stock",
        "controller": "stock",
        "uses_suspension_api": False,
    }),
    ("identity", {
        "name": "S1_identity",
        "label": "S1 identity",
        "controller": "identity",
        "uses_suspension_api": True,
    }),
    ("pid", {
        "name": "S2_pid",
        "label": "S2 feedback PID",
        "controller": "pid",
        "uses_suspension_api": True,
    }),
    ("skyhook", {
        "name": "S3_skyhook",
        "label": "S3 skyhook damping",
        "controller": "skyhook",
        "uses_suspension_api": True,
    }),
))


STATE_FIELD_NAMES = (
    "step",
    "frame",
    "elapsed_seconds",
    "dt",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "speed",
    "local_vx",
    "local_vy",
    "ax",
    "ay",
    "az",
    "local_ax",
    "local_ay",
    "roll",
    "pitch",
    "yaw",
    "roll_rate",
    "pitch_rate",
    "yaw_rate",
    "throttle",
    "brake",
    "steer",
)

PROFILE_FIELDS = (
    "wall_time",
    "scenario",
    "label",
    "controller",
    "seed",
    "episode_index",
    "actor_id",
    "type_id",
    "role_name",
) + STATE_FIELD_NAMES

EVENT_FIELDS = (
    "wall_time",
    "scenario",
    "label",
    "controller",
    "seed",
    "event",
    "frame",
    "episode_index",
    "actor_id",
    "type_id",
    "role_name",
    "message",
)

DIAGNOSTIC_FIELDS = (
    "wall_time",
    "scenario",
    "label",
    "controller",
    "seed",
    "episode_index",
    "actor_id",
    "frame",
    "elapsed_seconds",
    "step",
    "command_applied",
    "apply_count",
    "verify_count",
    "spring_scale",
    "damper_scale",
    "mean_spring_scale_readback",
    "mean_damper_scale_readback",
    "min_spring_scale_readback",
    "max_spring_scale_readback",
    "min_damper_scale_readback",
    "max_damper_scale_readback",
    "activity",
    "activity_error",
    "integral_error",
    "derivative_error",
    "p_term",
    "i_term",
    "d_term",
    "activity_roll_angle",
    "activity_pitch_angle",
    "activity_roll_rate",
    "activity_pitch_rate",
    "activity_lateral_acc",
    "activity_vertical_acc",
    "skyhook_corner_vz_fl",
    "skyhook_corner_vz_fr",
    "skyhook_corner_vz_rl",
    "skyhook_corner_vz_rr",
    "skyhook_activity_fl",
    "skyhook_activity_fr",
    "skyhook_activity_rl",
    "skyhook_activity_rr",
    "skyhook_damper_scale_fl",
    "skyhook_damper_scale_fr",
    "skyhook_damper_scale_rl",
    "skyhook_damper_scale_rr",
    "skyhook_mean_abs_corner_vz",
    "skyhook_max_activity",
    "skyhook_roll_rate_rad",
    "skyhook_pitch_rate_rad",
)

ROUTE_METRIC_PREFIXES = (
    "comfort_",
    "stability_",
    "warmup_excluded_comfort_",
    "warmup_excluded_stability_",
)


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in ("", "null", "None", "~"):
        return None
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    try:
        integer = int(value, 10)
        if str(integer) == value:
            return integer
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def read_flat_yaml(path: str) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    if not path:
        return values
    with open(path) as yaml_file:
        for raw_line in yaml_file:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            if key:
                values[key] = parse_scalar(value)
    return values


def build_pid_config(path: str) -> FeedbackPIDConfig:
    if path and os.path.isfile(path):
        return FeedbackPIDConfig.from_mapping(read_flat_yaml(path))
    return FeedbackPIDConfig()


def build_skyhook_config(path: str) -> SkyhookConfig:
    if path and os.path.isfile(path):
        return SkyhookConfig.from_mapping(read_flat_yaml(path))
    return SkyhookConfig()


def make_controller(controller_name: str, args: argparse.Namespace):
    if controller_name == "stock":
        return None
    if controller_name == "identity":
        return IdentityController()
    if controller_name == "pid":
        return FeedbackPIDController(build_pid_config(args.pid_config))
    if controller_name == "skyhook":
        return SkyhookController(build_skyhook_config(args.skyhook_config))
    raise ValueError("unknown controller %s" % controller_name)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def output_dir_path(path: str) -> str:
    if path:
        return expand_path(path)
    return os.path.join(
        DEFAULT_TFPP_OUTPUT_ROOT,
        "suspension_pid_routes_validation_%s" % now_stamp())


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> List[int]:
    seeds = []
    for item in parse_csv_list(value):
        seeds.append(int(item, 10))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def selected_scenarios(value: str) -> List[Dict[str, Any]]:
    selected = []
    for name in parse_csv_list(value):
        key = name
        if key.startswith("S0"):
            key = "stock"
        elif key.startswith("S1"):
            key = "identity"
        elif key.startswith("S2"):
            key = "pid"
        elif key.startswith("S3"):
            key = "skyhook"
        if key not in SCENARIOS:
            raise ValueError(
                "unknown scenario %s; choose from %s" %
                (name, ",".join(SCENARIOS.keys())))
        selected.append(dict(SCENARIOS[key]))
    if not selected:
        raise ValueError("at least one scenario is required")
    return selected


def safe_float(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        return "%0.9g" % value
    return value


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def field_order(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> List[str]:
    fields = list(preferred)
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def write_csv_rows(
    path: str,
    rows: Sequence[Mapping[str, Any]],
    preferred_fields: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = field_order(rows, preferred_fields)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def get_actor_control(actor: Any):
    try:
        return actor.get_control()
    except Exception:
        return None


class SuspensionExperimentSidecar(threading.Thread):
    """CARLA client sidecar used by one scenario run."""

    def __init__(
        self,
        args: argparse.Namespace,
        scenario: Mapping[str, Any],
        seed: int,
        profile_path: str,
        diagnostics_path: str,
        events_path: str,
    ):
        threading.Thread.__init__(self)
        self.daemon = True
        self.args = args
        self.scenario = dict(scenario)
        self.seed = seed
        self.profile_path = profile_path
        self.diagnostics_path = diagnostics_path
        self.events_path = events_path
        self.stop_event = threading.Event()
        self.error: Optional[BaseException] = None

        self.native_by_actor_id: Dict[int, Any] = {}
        self.controller_by_actor_id: Dict[int, Any] = {}
        self.previous_state_by_actor_id: Dict[int, Any] = {}
        self.episode_by_actor_id: Dict[int, int] = {}
        self.step_by_actor_id: Dict[int, int] = {}
        self.apply_count_by_actor_id: Dict[int, int] = {}
        self.verify_count_by_actor_id: Dict[int, int] = {}
        self.last_frame_by_actor_id: Dict[int, int] = {}
        self.next_episode_index = 0

    def stop(self) -> None:
        self.stop_event.set()

    @property
    def controller_name(self) -> str:
        return str(self.scenario["controller"])

    def base_row(self, actor: Any = None) -> Dict[str, Any]:
        actor_id = getattr(actor, "id", "")
        return {
            "wall_time": "%0.6f" % time.time(),
            "scenario": self.scenario["name"],
            "label": self.scenario["label"],
            "controller": self.controller_name,
            "seed": self.seed,
            "actor_id": actor_id,
            "type_id": getattr(actor, "type_id", ""),
            "role_name": (
                actor.attributes.get("role_name", "")
                if actor is not None else ""),
        }

    def log_event(
        self,
        writer: csv.DictWriter,
        event: str,
        frame: Any = "",
        actor: Any = None,
        message: str = "",
        episode_index: Any = "",
    ) -> None:
        row = self.base_row(actor)
        row.update({
            "event": event,
            "frame": frame,
            "episode_index": episode_index,
            "message": message,
        })
        writer.writerow({field: row.get(field, "") for field in EVENT_FIELDS})

    def connect_world(self, event_writer: csv.DictWriter):
        carla = import_carla()
        while not self.stop_event.is_set():
            try:
                client = carla.Client(self.args.host, self.args.port)
                client.set_timeout(self.args.timeout)
                world = client.get_world()
                self.log_event(
                    event_writer,
                    "connected",
                    message="connected to CARLA")
                return world
            except RuntimeError as error:
                self.log_event(
                    event_writer,
                    "connect_wait",
                    message=str(error))
                time.sleep(self.args.connect_retry_seconds)
        return None

    def target_vehicles(self, world: Any) -> List[Any]:
        vehicles = []
        for actor in world.get_actors().filter("vehicle.*"):
            if self.args.actor_id is not None and actor.id != self.args.actor_id:
                continue
            role_name = actor.attributes.get("role_name", "")
            if self.args.actor_id is None and role_name != self.args.role_name:
                continue
            if self.scenario["uses_suspension_api"]:
                if not hasattr(actor, "get_suspension_physics_control"):
                    continue
                if not hasattr(actor, "apply_suspension_physics_control"):
                    continue
            vehicles.append(actor)
        return vehicles

    def remove_missing_actors(
        self,
        event_writer: csv.DictWriter,
        active_actor_ids: Iterable[int],
        frame: int,
    ) -> None:
        active = set(active_actor_ids)
        for actor_id in list(self.episode_by_actor_id.keys()):
            if actor_id in active:
                continue
            episode_index = self.episode_by_actor_id.get(actor_id, "")
            self.native_by_actor_id.pop(actor_id, None)
            self.controller_by_actor_id.pop(actor_id, None)
            self.previous_state_by_actor_id.pop(actor_id, None)
            self.episode_by_actor_id.pop(actor_id, None)
            self.step_by_actor_id.pop(actor_id, None)
            self.apply_count_by_actor_id.pop(actor_id, None)
            self.verify_count_by_actor_id.pop(actor_id, None)
            self.last_frame_by_actor_id.pop(actor_id, None)
            self.log_event(
                event_writer,
                "actor_removed",
                frame=frame,
                message="actor %s disappeared" % actor_id,
                episode_index=episode_index)

    def capture_if_needed(
        self,
        event_writer: csv.DictWriter,
        actor: Any,
        frame: int,
    ) -> None:
        if actor.id in self.episode_by_actor_id:
            return

        episode_index = self.next_episode_index
        self.next_episode_index += 1
        self.episode_by_actor_id[actor.id] = episode_index
        self.step_by_actor_id[actor.id] = 0
        self.apply_count_by_actor_id[actor.id] = 0
        self.verify_count_by_actor_id[actor.id] = 0

        if self.scenario["uses_suspension_api"]:
            native = actor.get_suspension_physics_control()
            validate_suspension_control(native)
            controller = make_controller(self.controller_name, self.args)
            controller.reset(native)
            self.native_by_actor_id[actor.id] = native
            self.controller_by_actor_id[actor.id] = controller
            self.log_event(
                event_writer,
                "native_captured",
                frame=frame,
                actor=actor,
                episode_index=episode_index,
                message="captured native suspension")
            self.log_event(
                event_writer,
                "controller_reset",
                frame=frame,
                actor=actor,
                episode_index=episode_index,
                message="controller=%s" % self.controller_name)
        else:
            self.log_event(
                event_writer,
                "observer_attached",
                frame=frame,
                actor=actor,
                episode_index=episode_index,
                message="stock observer attached")

    def write_profile_row(
        self,
        profile_writer: csv.DictWriter,
        actor: Any,
        state: Any,
    ) -> None:
        row = self.base_row(actor)
        row["episode_index"] = self.episode_by_actor_id[actor.id]
        row.update(state.as_dict())
        profile_writer.writerow({field: row.get(field, "") for field in PROFILE_FIELDS})

    def should_verify(self, actor_id: int, next_apply_count: int) -> bool:
        return self.args.verify_every > 0 and next_apply_count % self.args.verify_every == 0

    def should_readback(self, actor_id: int, next_apply_count: int) -> bool:
        return (
            self.args.readback_every > 0 and
            next_apply_count % self.args.readback_every == 0)

    def process_controller(
        self,
        diagnostic_writer: csv.DictWriter,
        event_writer: csv.DictWriter,
        actor: Any,
        state: Any,
    ) -> None:
        actor_id = actor.id
        native = self.native_by_actor_id[actor_id]
        controller = self.controller_by_actor_id[actor_id]
        previous_state = self.previous_state_by_actor_id.get(actor_id)
        current_suspension = actor.get_suspension_physics_control()
        dt = state.dt if state.dt > 0.0 else self.args.default_dt
        context = ControllerContext(
            state=state,
            previous_state=previous_state,
            planning=PlanningInfo.empty(),
            native_suspension=native,
            current_suspension=current_suspension,
            step=state.step,
            dt=dt)
        output = controller.compute(context)
        output.command.validate(expected_wheels=len(native.wheels))

        next_apply_count = self.apply_count_by_actor_id[actor_id] + 1
        verify = self.should_verify(actor_id, next_apply_count)
        apply_suspension_command(
            actor,
            native,
            output.command,
            verify_readback=verify,
            readback_tolerance=self.args.readback_tolerance)
        self.apply_count_by_actor_id[actor_id] = next_apply_count

        readback_summary: Mapping[str, Any] = {}
        if verify:
            self.verify_count_by_actor_id[actor_id] += 1
            self.log_event(
                event_writer,
                "command_verified",
                frame=state.frame,
                actor=actor,
                episode_index=self.episode_by_actor_id[actor_id],
                message="readback matched command")
            readback_summary = read_suspension_scale_summary(
                native,
                actor.get_suspension_physics_control())
        elif self.should_readback(actor_id, next_apply_count):
            readback_summary = read_suspension_scale_summary(
                native,
                actor.get_suspension_physics_control())

        scales = output.command.as_scale_lists()
        spring_scales = scales["spring_scales"]
        damper_scales = scales["damper_scales"]

        row: Dict[str, Any] = self.base_row(actor)
        row.update({
            "episode_index": self.episode_by_actor_id[actor_id],
            "frame": state.frame,
            "elapsed_seconds": state.elapsed_seconds,
            "step": state.step,
            "command_applied": 1,
            "apply_count": self.apply_count_by_actor_id[actor_id],
            "verify_count": self.verify_count_by_actor_id[actor_id],
            "spring_scale": (
                sum(spring_scales) / float(len(spring_scales))
                if spring_scales else ""),
            "damper_scale": (
                sum(damper_scales) / float(len(damper_scales))
                if damper_scales else ""),
            "mean_spring_scale_readback": readback_summary.get(
                "mean_spring_scale", ""),
            "mean_damper_scale_readback": readback_summary.get(
                "mean_damper_scale", ""),
            "min_spring_scale_readback": readback_summary.get(
                "min_spring_scale", ""),
            "max_spring_scale_readback": readback_summary.get(
                "max_spring_scale", ""),
            "min_damper_scale_readback": readback_summary.get(
                "min_damper_scale", ""),
            "max_damper_scale_readback": readback_summary.get(
                "max_damper_scale", ""),
        })
        row.update(output.diagnostics)
        diagnostic_writer.writerow({
            field: format_value(row.get(field, ""))
            for field in DIAGNOSTIC_FIELDS
        })

    def process_actor(
        self,
        world: Any,
        profile_writer: csv.DictWriter,
        diagnostic_writer: csv.DictWriter,
        event_writer: csv.DictWriter,
        actor: Any,
        frame: int,
    ) -> None:
        last_frame = self.last_frame_by_actor_id.get(actor.id)
        if last_frame == frame:
            return
        self.last_frame_by_actor_id[actor.id] = frame

        step = self.step_by_actor_id[actor.id]
        previous_state = self.previous_state_by_actor_id.get(actor.id)
        state = read_vehicle_state(
            world,
            actor,
            step=step,
            previous_state=previous_state,
            control=get_actor_control(actor))

        self.write_profile_row(profile_writer, actor, state)
        if self.scenario["uses_suspension_api"]:
            self.process_controller(diagnostic_writer, event_writer, actor, state)

        self.previous_state_by_actor_id[actor.id] = state
        self.step_by_actor_id[actor.id] = step + 1

    def run(self) -> None:
        os.makedirs(os.path.dirname(self.profile_path), exist_ok=True)
        try:
            with open(self.profile_path, "w", newline="") as profile_file, \
                    open(self.diagnostics_path, "w", newline="") as diagnostic_file, \
                    open(self.events_path, "w", newline="") as events_file:
                profile_writer = csv.DictWriter(profile_file, fieldnames=PROFILE_FIELDS)
                diagnostic_writer = csv.DictWriter(
                    diagnostic_file,
                    fieldnames=DIAGNOSTIC_FIELDS)
                event_writer = csv.DictWriter(events_file, fieldnames=EVENT_FIELDS)
                profile_writer.writeheader()
                diagnostic_writer.writeheader()
                event_writer.writeheader()

                world = self.connect_world(event_writer)
                if world is None:
                    return

                last_wait_log = 0.0
                while not self.stop_event.is_set():
                    try:
                        snapshot = world.get_snapshot()
                        frame = int(snapshot.frame)
                        vehicles = self.target_vehicles(world)
                        active_ids = [actor.id for actor in vehicles]
                        self.remove_missing_actors(event_writer, active_ids, frame)

                        if not vehicles:
                            now = time.time()
                            if now - last_wait_log >= self.args.wait_log_seconds:
                                self.log_event(
                                    event_writer,
                                    "waiting_for_vehicle",
                                    frame=frame,
                                    message="no matching vehicle found")
                                last_wait_log = now
                        else:
                            for actor in vehicles:
                                self.capture_if_needed(event_writer, actor, frame)
                                self.process_actor(
                                    world,
                                    profile_writer,
                                    diagnostic_writer,
                                    event_writer,
                                    actor,
                                    frame)

                        profile_file.flush()
                        diagnostic_file.flush()
                        events_file.flush()
                        if self.args.sidecar_tick_mode == "wait":
                            world.wait_for_tick(self.args.tick_wait_timeout)
                        else:
                            time.sleep(self.args.poll_seconds)
                    except RuntimeError as error:
                        self.log_event(
                            event_writer,
                            "runtime_error",
                            message=str(error))
                        events_file.flush()
                        time.sleep(self.args.connect_retry_seconds)
                    except Exception as error:
                        self.error = error
                        self.log_event(
                            event_writer,
                            "fatal_error",
                            message=str(error))
                        events_file.flush()
                        return

                self.log_event(event_writer, "stopped", message="sidecar stopped")
                events_file.flush()
        except Exception as error:
            self.error = error


def normalized_command(args: argparse.Namespace) -> List[str]:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if command:
        return command
    return [expand_path(args.route_script)]


def run_env(
    args: argparse.Namespace,
    seed: int,
    scenario_dir: str,
) -> Dict[str, str]:
    env = os.environ.copy()
    sim_root = env.get("SIM_ROOT", SIM_ROOT)
    carla_root = env.get("CARLA_ROOT", os.path.join(sim_root, "carla-0.9.15"))
    e2e_root = env.get("E2E_ROOT", os.path.join(sim_root, "e2e_models"))
    garage_root = env.get("GARAGE_ROOT", os.path.join(e2e_root, "carla_garage"))
    e2e_output_root = env.get("E2E_OUTPUT_ROOT", os.path.join(e2e_root, "outputs"))
    e2e_log_root = env.get("E2E_LOG_ROOT", os.path.join(e2e_root, "logs"))
    e2e_ckpt_root = env.get("E2E_CKPT_ROOT", os.path.join(e2e_root, "checkpoints"))

    env.setdefault("SIM_ROOT", sim_root)
    env.setdefault("CARLA_ROOT", carla_root)
    env.setdefault("CARLA_SOURCE_ROOT", carla_root)
    env.setdefault("CARLA_SERVER_ROOT", carla_root)
    env.setdefault("E2E_ROOT", e2e_root)
    env.setdefault("E2E_OUTPUT_ROOT", e2e_output_root)
    env.setdefault("E2E_LOG_ROOT", e2e_log_root)
    env.setdefault("E2E_CKPT_ROOT", e2e_ckpt_root)
    env.setdefault("GARAGE_ROOT", garage_root)
    env.setdefault("WORK_DIR", garage_root)
    env.setdefault("SCENARIO_RUNNER_ROOT", os.path.join(garage_root, "scenario_runner"))
    env.setdefault("LEADERBOARD_ROOT", os.path.join(garage_root, "leaderboard"))
    env.setdefault("TFPP_CKPT_ROOT", os.path.join(e2e_ckpt_root, "transfuserpp"))
    env.setdefault("TFPP_OUTPUT_ROOT", os.path.join(e2e_output_root, "transfuserpp"))
    env.setdefault("TFPP_LOG_ROOT", os.path.join(e2e_log_root, "transfuserpp"))

    env["PORT"] = str(args.port)
    env["DEBUG"] = str(args.debug)
    env["ROUTES"] = expand_path(args.routes)
    env["REPETITIONS"] = str(args.repetitions)
    env["ROUTES_SUBSET"] = args.routes_subset
    env["TRAFFIC_MANAGER_SEED"] = str(seed)
    env["TRAFFIC_MANAGER_PORT"] = str(args.traffic_manager_port)
    env["OUT_DIR"] = os.path.join(scenario_dir, "tfpp")
    if args.team_config:
        env["TEAM_CONFIG"] = expand_path(args.team_config)
    if args.team_agent:
        env["TEAM_AGENT"] = expand_path(args.team_agent)
    return env


def run_route_command(
    command: Sequence[str],
    log_path: str,
    env: Mapping[str, str],
) -> Tuple[int, List[str]]:
    print("Running route command:")
    print("  %s" % " ".join(command))
    output_lines: List[str] = []
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=dict(env))
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                output_lines.append(line.rstrip("\n"))
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                return_code = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()
    return return_code, output_lines


def newest_result_from_output(lines: Sequence[str]) -> str:
    for line in reversed(lines):
        candidate = line.strip()
        if candidate.startswith("CHECKPOINT="):
            candidate = candidate.split("=", 1)[1].strip()
        if candidate.endswith("result.json") and os.path.isfile(candidate):
            return candidate
    return ""


def read_result_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as json_file:
        return json.load(json_file)


def extract_result_metrics(path: str) -> Dict[str, Any]:
    data = read_result_json(path)
    if not data:
        return {}

    checkpoint = data.get("_checkpoint", {})
    global_record = checkpoint.get("global_record", {})
    scores = global_record.get("scores_mean", {})
    infractions = global_record.get("infractions", {})
    meta = global_record.get("meta", {})
    records = checkpoint.get("records", [])

    route_statuses = ";".join(
        "%s:%s" % (record.get("route_id", ""), record.get("status", ""))
        for record in records)
    route_scores = ";".join(
        "%s:%s" % (
            record.get("route_id", ""),
            record.get("scores", {}).get("score_composed", ""))
        for record in records)

    return {
        "entry_status": data.get("entry_status", ""),
        "eligible": data.get("eligible", ""),
        "progress": "%s/%s" % tuple(checkpoint.get("progress", ["", ""])),
        "global_status": global_record.get("status", ""),
        "score_composed": scores.get("score_composed", ""),
        "score_route": scores.get("score_route", ""),
        "score_penalty": scores.get("score_penalty", ""),
        "collisions_layout": infractions.get("collisions_layout", ""),
        "collisions_pedestrian": infractions.get("collisions_pedestrian", ""),
        "collisions_vehicle": infractions.get("collisions_vehicle", ""),
        "red_light": infractions.get("red_light", ""),
        "stop_infraction": infractions.get("stop_infraction", ""),
        "outside_route_lanes": infractions.get("outside_route_lanes", ""),
        "route_dev": infractions.get("route_dev", ""),
        "vehicle_blocked": infractions.get("vehicle_blocked", ""),
        "route_timeout": infractions.get("route_timeout", ""),
        "scenario_timeouts": infractions.get("scenario_timeouts", ""),
        "min_speed_infractions": infractions.get("min_speed_infractions", ""),
        "duration_game": meta.get("duration_game", ""),
        "duration_system": meta.get("duration_system", ""),
        "route_statuses": route_statuses,
        "route_scores": route_scores,
    }


def result_finished(path: str) -> bool:
    return extract_result_metrics(path).get("entry_status") == "Finished"


def summarize_events_csv(path: str) -> Dict[str, Any]:
    summary = {
        "sidecar_profile_rows": 0,
        "sidecar_diagnostic_rows": 0,
        "sidecar_native_captures": 0,
        "sidecar_observer_attaches": 0,
        "sidecar_command_verifies": 0,
        "sidecar_actor_ids": "",
        "sidecar_fatal_errors": 0,
        "sidecar_runtime_errors": 0,
    }
    actor_ids = set()
    if os.path.isfile(path):
        with open(path) as csv_file:
            for row in csv.DictReader(csv_file):
                actor_id = row.get("actor_id", "")
                if actor_id:
                    actor_ids.add(actor_id)
                event = row.get("event", "")
                if event == "native_captured":
                    summary["sidecar_native_captures"] += 1
                elif event == "observer_attached":
                    summary["sidecar_observer_attaches"] += 1
                elif event == "command_verified":
                    summary["sidecar_command_verifies"] += 1
                elif event == "fatal_error":
                    summary["sidecar_fatal_errors"] += 1
                elif event == "runtime_error":
                    summary["sidecar_runtime_errors"] += 1
    summary["sidecar_actor_ids"] = ";".join(sorted(actor_ids))
    return summary


def count_csv_rows(path: str) -> int:
    if not os.path.isfile(path):
        return 0
    with open(path) as csv_file:
        reader = csv.DictReader(csv_file)
        return sum(1 for _ in reader)


def prefixed_metrics(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {prefix + key: value for key, value in metrics.items()}


def read_profile_groups(path: str) -> Tuple[List[Dict[str, Any]], OrderedDict]:
    all_rows: List[Dict[str, Any]] = []
    groups: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    if not os.path.isfile(path):
        return all_rows, groups
    with open(path) as csv_file:
        for row in csv.DictReader(csv_file):
            all_rows.append(row)
            episode = row.get("episode_index", "")
            groups.setdefault(episode, []).append(row)
    return all_rows, groups


def drop_warmup_rows(
    rows: Sequence[Mapping[str, Any]],
    warmup_seconds: float,
) -> List[Mapping[str, Any]]:
    if warmup_seconds <= 0.0 or not rows:
        return list(rows)
    first_elapsed = safe_float(rows[0].get("elapsed_seconds"))
    if first_elapsed is None:
        return list(rows)
    cutoff = first_elapsed + warmup_seconds
    filtered = []
    for row in rows:
        elapsed = safe_float(row.get("elapsed_seconds"))
        if elapsed is not None and elapsed >= cutoff:
            filtered.append(row)
    return filtered if filtered else list(rows)


def update_warmup_excluded_metrics(
    metrics: Dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    warmup_seconds: float,
    default_dt: float,
    steady_fraction: float,
) -> None:
    filtered_rows = drop_warmup_rows(rows, warmup_seconds)
    metrics["warmup_excluded_seconds"] = max(0.0, warmup_seconds)
    metrics["warmup_excluded_profile_rows"] = len(filtered_rows)
    metrics["warmup_excluded_start_elapsed_seconds"] = (
        filtered_rows[0].get("elapsed_seconds", "") if filtered_rows else "")
    metrics.update(prefixed_metrics(
        "warmup_excluded_comfort_",
        comfort_metrics(filtered_rows, default_dt=default_dt)))
    metrics.update(prefixed_metrics(
        "warmup_excluded_stability_",
        stability_metrics(filtered_rows, steady_fraction=steady_fraction)))


def compute_profile_metrics(
    scenario: Mapping[str, Any],
    seed: int,
    profile_path: str,
    metrics_path: str,
    default_dt: float,
    steady_fraction: float,
    metric_warmup_seconds: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    all_rows, groups = read_profile_groups(profile_path)
    episode_rows: List[Dict[str, Any]] = []

    for episode, rows in groups.items():
        metric_row: Dict[str, Any] = {
            "scenario": scenario["name"],
            "label": scenario["label"],
            "controller": scenario["controller"],
            "seed": seed,
            "episode_index": episode,
            "actor_id": rows[0].get("actor_id", "") if rows else "",
            "profile_rows": len(rows),
            "start_frame": rows[0].get("frame", "") if rows else "",
            "end_frame": rows[-1].get("frame", "") if rows else "",
            "start_elapsed_seconds": rows[0].get("elapsed_seconds", "") if rows else "",
            "end_elapsed_seconds": rows[-1].get("elapsed_seconds", "") if rows else "",
        }
        metric_row.update(prefixed_metrics(
            "comfort_",
            comfort_metrics(rows, default_dt=default_dt)))
        metric_row.update(prefixed_metrics(
            "stability_",
            stability_metrics(rows, steady_fraction=steady_fraction)))
        update_warmup_excluded_metrics(
            metric_row,
            rows,
            metric_warmup_seconds,
            default_dt,
            steady_fraction)
        episode_rows.append(metric_row)

    preferred = (
        "scenario",
        "label",
        "controller",
        "seed",
        "episode_index",
        "actor_id",
        "profile_rows",
        "start_frame",
        "end_frame",
        "start_elapsed_seconds",
        "end_elapsed_seconds",
        "warmup_excluded_seconds",
        "warmup_excluded_profile_rows",
        "warmup_excluded_start_elapsed_seconds",
    )
    write_csv_rows(metrics_path, episode_rows, preferred)

    run_metrics: Dict[str, Any] = {"profile_rows": len(all_rows)}
    if all_rows:
        run_metrics.update(prefixed_metrics(
            "comfort_",
            comfort_metrics(all_rows, default_dt=default_dt)))
        run_metrics.update(prefixed_metrics(
            "stability_",
            stability_metrics(all_rows, steady_fraction=steady_fraction)))
        update_warmup_excluded_metrics(
            run_metrics,
            all_rows,
            metric_warmup_seconds,
            default_dt,
            steady_fraction)
    return run_metrics, episode_rows


def write_comparison_summary(
    output_dir: str,
    summary_rows: Sequence[Mapping[str, Any]],
    baseline_scenario: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    by_seed_and_scenario: Dict[Tuple[Any, str], Mapping[str, Any]] = {}
    for row in summary_rows:
        by_seed_and_scenario[(row.get("seed"), row.get("scenario"))] = row

    comparison_rows: List[Dict[str, Any]] = []
    lower_is_better_prefixes = ROUTE_METRIC_PREFIXES
    score_metrics = ("score_composed", "score_route", "score_penalty")

    metric_names = []
    for row in summary_rows:
        for key in row:
            if key.startswith(lower_is_better_prefixes) or key in score_metrics:
                if key not in metric_names:
                    metric_names.append(key)

    for row in summary_rows:
        seed = row.get("seed")
        scenario = row.get("scenario")
        if scenario == baseline_scenario:
            continue
        baseline = by_seed_and_scenario.get((seed, baseline_scenario))
        if not baseline:
            continue
        for metric in metric_names:
            base_value = safe_float(baseline.get(metric))
            candidate_value = safe_float(row.get(metric))
            if base_value is None or candidate_value is None:
                continue
            lower_is_better = metric.startswith(lower_is_better_prefixes)
            delta = candidate_value - base_value
            if lower_is_better:
                improvement_pct = (
                    100.0 * (base_value - candidate_value) / base_value
                    if abs(base_value) > 1.0e-12 else "")
            else:
                improvement_pct = (
                    100.0 * (candidate_value - base_value) / base_value
                    if abs(base_value) > 1.0e-12 else "")
            comparison_rows.append({
                "seed": seed,
                "baseline_scenario": baseline_scenario,
                "candidate_scenario": scenario,
                "metric": metric,
                "baseline_value": base_value,
                "candidate_value": candidate_value,
                "delta_candidate_minus_baseline": delta,
                "lower_is_better": int(lower_is_better),
                "improvement_pct": improvement_pct,
            })

    path = os.path.join(output_dir, "suite_comparison.csv")
    write_csv_rows(
        path,
        comparison_rows,
        (
            "seed",
            "baseline_scenario",
            "candidate_scenario",
            "metric",
            "baseline_value",
            "candidate_value",
            "delta_candidate_minus_baseline",
            "lower_is_better",
            "improvement_pct",
        ))
    return path, comparison_rows


def run_scenario(
    args: argparse.Namespace,
    scenario: Mapping[str, Any],
    seed: int,
    command: Sequence[str],
    output_dir: str,
) -> Dict[str, Any]:
    run_name = "seed%d_%s" % (seed, scenario["name"])
    scenario_dir = os.path.join(output_dir, run_name)
    os.makedirs(scenario_dir, exist_ok=True)
    profile_path = os.path.join(scenario_dir, "profile.csv")
    diagnostics_path = os.path.join(scenario_dir, "controller_diagnostics.csv")
    events_path = os.path.join(scenario_dir, "sidecar_events.csv")
    metrics_path = os.path.join(scenario_dir, "metrics_by_episode.csv")
    route_log_path = os.path.join(scenario_dir, "route_stdout.log")
    expected_result_path = os.path.join(scenario_dir, "tfpp", "result.json")

    print("")
    print("=== seed %d / %s ===" % (seed, scenario["label"]))
    start_time = time.time()

    sidecar = SuspensionExperimentSidecar(
        args,
        scenario,
        seed,
        profile_path,
        diagnostics_path,
        events_path)
    sidecar.start()
    print("Started suspension sidecar:")
    print("  %s" % events_path)

    env = run_env(args, seed, scenario_dir)
    return_code: Optional[int] = None
    output_lines: List[str] = []
    status = "ok"
    result_path = ""

    try:
        return_code, output_lines = run_route_command(command, route_log_path, env)
        result_path = expected_result_path if os.path.isfile(expected_result_path) else ""
        if not result_path:
            result_path = newest_result_from_output(output_lines)
        if not result_path:
            status = "missing_result_json"
        elif return_code != 0:
            if result_finished(result_path):
                status = "finished_return_nonzero"
            else:
                status = "route_command_failed"
    finally:
        sidecar.stop()
        sidecar.join(timeout=5.0)
        if sidecar.error is not None:
            status = "sidecar_failed"

    end_time = time.time()

    result_metrics = extract_result_metrics(result_path)
    event_summary = summarize_events_csv(events_path)
    event_summary["sidecar_profile_rows"] = count_csv_rows(profile_path)
    event_summary["sidecar_diagnostic_rows"] = count_csv_rows(diagnostics_path)
    profile_metrics, _ = compute_profile_metrics(
        scenario,
        seed,
        profile_path,
        metrics_path,
        args.default_dt,
        args.steady_fraction,
        args.metric_warmup_seconds)

    if event_summary["sidecar_profile_rows"] == 0 and status == "ok":
        status = "sidecar_no_profile"
    if scenario["uses_suspension_api"] and status == "ok":
        if event_summary["sidecar_native_captures"] == 0:
            status = "sidecar_no_target_vehicle"
        elif event_summary["sidecar_diagnostic_rows"] == 0:
            status = "sidecar_no_commands"
        elif args.require_verification and event_summary["sidecar_command_verifies"] == 0:
            status = "sidecar_not_verified"

    row: Dict[str, Any] = {
        "seed": seed,
        "scenario": scenario["name"],
        "label": scenario["label"],
        "controller": scenario["controller"],
        "uses_suspension_api": int(bool(scenario["uses_suspension_api"])),
        "status": status,
        "return_code": return_code,
        "result_json": result_path,
        "scenario_dir": scenario_dir,
        "profile_csv": profile_path,
        "controller_diagnostics_csv": diagnostics_path,
        "metrics_by_episode_csv": metrics_path,
        "sidecar_events_csv": events_path,
        "route_stdout_log": route_log_path,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": end_time - start_time,
    }
    row.update(result_metrics)
    row.update(event_summary)
    row.update(profile_metrics)

    write_json(os.path.join(scenario_dir, "summary.json"), row)
    print("Scenario result:")
    print("  status=%s return_code=%s" % (status, return_code))
    print("  result_json=%s" % result_path)
    print("  profile_csv=%s" % profile_path)
    print("  metrics_by_episode_csv=%s" % metrics_path)

    if status in ("route_command_failed", "sidecar_failed") and args.stop_on_failure:
        raise RuntimeError("%s failed with status %s" % (run_name, status))

    return row


def write_suite_summary(output_dir: str, rows: Sequence[Mapping[str, Any]]) -> Tuple[str, str]:
    preferred = (
        "seed",
        "scenario",
        "label",
        "controller",
        "uses_suspension_api",
        "status",
        "return_code",
        "entry_status",
        "global_status",
        "progress",
        "score_composed",
        "score_route",
        "score_penalty",
        "sidecar_profile_rows",
        "sidecar_diagnostic_rows",
        "sidecar_native_captures",
        "sidecar_observer_attaches",
        "sidecar_command_verifies",
        "sidecar_fatal_errors",
        "sidecar_runtime_errors",
        "comfort_comfort_score",
        "comfort_rms_vertical_acc",
        "comfort_rms_lateral_acc",
        "comfort_rms_longitudinal_acc",
        "comfort_rms_vertical_jerk",
        "comfort_rms_lateral_jerk",
        "comfort_peak_abs_vertical_acc",
        "comfort_peak_abs_vertical_jerk",
        "warmup_excluded_seconds",
        "warmup_excluded_profile_rows",
        "warmup_excluded_comfort_comfort_score",
        "warmup_excluded_comfort_rms_vertical_acc",
        "warmup_excluded_comfort_rms_lateral_acc",
        "warmup_excluded_comfort_rms_longitudinal_acc",
        "warmup_excluded_comfort_rms_vertical_jerk",
        "warmup_excluded_comfort_rms_lateral_jerk",
        "warmup_excluded_comfort_peak_abs_vertical_acc",
        "warmup_excluded_comfort_peak_abs_vertical_jerk",
        "stability_peak_abs_roll",
        "stability_peak_abs_pitch",
        "stability_peak_abs_yaw_rate",
        "stability_peak_abs_lateral_acc",
        "stability_rms_roll",
        "stability_rms_pitch",
        "stability_rms_yaw_rate",
        "stability_rms_lateral_acc",
        "warmup_excluded_stability_peak_abs_roll",
        "warmup_excluded_stability_peak_abs_pitch",
        "warmup_excluded_stability_peak_abs_yaw_rate",
        "warmup_excluded_stability_peak_abs_lateral_acc",
        "warmup_excluded_stability_rms_roll",
        "warmup_excluded_stability_rms_pitch",
        "warmup_excluded_stability_rms_yaw_rate",
        "warmup_excluded_stability_rms_lateral_acc",
        "duration_game",
        "duration_system",
        "duration_seconds",
        "result_json",
        "profile_csv",
        "controller_diagnostics_csv",
        "metrics_by_episode_csv",
        "sidecar_events_csv",
        "route_stdout_log",
    )
    csv_path = os.path.join(output_dir, "suite_summary.csv")
    json_path = os.path.join(output_dir, "suite_summary.json")
    write_csv_rows(csv_path, rows, preferred)
    write_json(json_path, list(rows))
    return csv_path, json_path


def print_suite_table(rows: Sequence[Mapping[str, Any]]) -> None:
    print("")
    print("TransFuser++ suspension-control suite summary:")
    print("%-8s %-14s %-24s %9s %9s %9s %10s %10s %8s" % (
        "seed",
        "scenario",
        "status",
        "score",
        "comfort",
        "roll_rms",
        "profiles",
        "commands",
        "rc"))
    for row in rows:
        print("%-8s %-14s %-24s %9s %9s %9s %10s %10s %8s" % (
            row.get("seed", ""),
            row.get("scenario", ""),
            row.get("status", ""),
            format_value(row.get("score_composed", "")),
            format_value(row.get("comfort_comfort_score", "")),
            format_value(row.get("stability_rms_roll", "")),
            row.get("sidecar_profile_rows", ""),
            row.get("sidecar_diagnostic_rows", ""),
            row.get("return_code", "")))


def main(args: argparse.Namespace) -> None:
    output_dir = output_dir_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    scenarios = selected_scenarios(args.scenarios)
    seeds = parse_seeds(args.seeds)
    command = normalized_command(args)

    write_json(os.path.join(output_dir, "suite_config.json"), {
        "created_at": time.time(),
        "host": args.host,
        "port": args.port,
        "routes": expand_path(args.routes),
        "routes_subset": args.routes_subset,
        "repetitions": args.repetitions,
        "metric_warmup_seconds": args.metric_warmup_seconds,
        "steady_fraction": args.steady_fraction,
        "seeds": seeds,
        "scenarios": scenarios,
        "pid_config": expand_path(args.pid_config) if args.pid_config else "",
        "command": command,
    })

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for index, scenario in enumerate(scenarios):
            if rows and args.pause_between_runs > 0.0:
                time.sleep(args.pause_between_runs)
            rows.append(run_scenario(args, scenario, seed, command, output_dir))

    print_suite_table(rows)
    csv_path, json_path = write_suite_summary(output_dir, rows)
    comparison_path, _ = write_comparison_summary(
        output_dir,
        rows,
        baseline_scenario=args.baseline_scenario)

    print("")
    print("Wrote suite outputs:")
    print("  %s" % csv_path)
    print("  %s" % json_path)
    print("  %s" % comparison_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="CARLA host (default: 127.0.0.1)")
    parser.add_argument(
        "-p",
        "--port",
        default=2000,
        type=int,
        help="CARLA port and route PORT env (default: 2000)")
    parser.add_argument(
        "--timeout",
        default=30.0,
        type=float,
        help="CARLA client timeout in seconds (default: 30.0)")
    parser.add_argument(
        "--debug",
        default=1,
        type=int,
        help="DEBUG env for run_tfpp_debug_route.sh (default: 1)")
    parser.add_argument(
        "--routes",
        default=DEFAULT_ROUTES,
        help="routes XML path (default: suspension_town13_short.xml)")
    parser.add_argument(
        "--routes-subset",
        default="",
        help="leaderboard routes subset, for example 0 or 0-2 (default: all)")
    parser.add_argument(
        "--repetitions",
        default=1,
        type=int,
        help="leaderboard repetitions per route (default: 1)")
    parser.add_argument(
        "--seeds",
        default="100",
        help="comma-separated traffic manager seeds (default: 100)")
    parser.add_argument(
        "--traffic-manager-port",
        default=8000,
        type=int,
        help="TRAFFIC_MANAGER_PORT env for route script (default: 8000)")
    parser.add_argument(
        "--scenarios",
        default="stock,identity,pid",
        help="comma-separated scenarios: stock,identity,pid,skyhook "
        "(default: stock,identity,pid)")
    parser.add_argument(
        "--baseline-scenario",
        default="S0_stock",
        help="scenario name used in suite_comparison.csv (default: S0_stock)")
    parser.add_argument(
        "--role-name",
        default="hero",
        help="vehicle role_name to observe/control when actor-id is not set")
    parser.add_argument(
        "--actor-id",
        default=None,
        type=int,
        help="specific vehicle actor id to observe/control instead of role-name")
    parser.add_argument(
        "--pid-config",
        default=DEFAULT_PID_CONFIG,
        help="flat YAML PID config path")
    parser.add_argument(
        "--skyhook-config",
        default=DEFAULT_SKYHOOK_CONFIG,
        help="flat YAML skyhook config path")
    parser.add_argument(
        "--default-dt",
        default=0.05,
        type=float,
        help="fallback dt for metrics/controller (default: 0.05)")
    parser.add_argument(
        "--steady-fraction",
        default=0.30,
        type=float,
        help="tail fraction for stability tail metrics (default: 0.30)")
    parser.add_argument(
        "--metric-warmup-seconds",
        default=2.0,
        type=float,
        help="initial episode seconds excluded from warmup_excluded metrics "
        "(default: 2.0; 0 disables exclusion)")
    parser.add_argument(
        "--verify-every",
        default=50,
        type=int,
        help="verify suspension readback every N applies; 0 disables")
    parser.add_argument(
        "--readback-every",
        default=10,
        type=int,
        help="record readback summary every N applies; 0 disables")
    parser.add_argument(
        "--readback-tolerance",
        default=1.0e-4,
        type=float,
        help="readback scale tolerance (default: 1e-4)")
    parser.add_argument(
        "--require-verification",
        action="store_true",
        help="mark API scenarios failed when no verification row is written")
    parser.add_argument(
        "--tick-wait-timeout",
        default=5.0,
        type=float,
        help="seconds to wait for each world tick in wait mode (default: 5.0)")
    parser.add_argument(
        "--sidecar-tick-mode",
        choices=("poll", "wait"),
        default="poll",
        help="sidecar timing mode; poll avoids an extra CARLA tick stream (default: poll)")
    parser.add_argument(
        "--poll-seconds",
        default=0.01,
        type=float,
        help="wall-clock sleep between sidecar polls in poll mode (default: 0.01)")
    parser.add_argument(
        "--connect-retry-seconds",
        default=1.0,
        type=float,
        help="seconds between CARLA reconnect attempts (default: 1.0)")
    parser.add_argument(
        "--wait-log-seconds",
        default=5.0,
        type=float,
        help="seconds between no-vehicle wait logs (default: 5.0)")
    parser.add_argument(
        "--pause-between-runs",
        default=5.0,
        type=float,
        help="seconds to pause between scenario runs (default: 5.0)")
    parser.add_argument(
        "--route-script",
        default=DEFAULT_ROUTE_SCRIPT,
        help="run_tfpp_debug_route.sh path")
    parser.add_argument(
        "--team-config",
        default="",
        help="optional TEAM_CONFIG override")
    parser.add_argument(
        "--team-agent",
        default="",
        help="optional TEAM_AGENT override")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="stop the suite on route/sidecar failure")
    parser.add_argument(
        "--output-dir",
        default="",
        help="suite output directory")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="optional route command after --")
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

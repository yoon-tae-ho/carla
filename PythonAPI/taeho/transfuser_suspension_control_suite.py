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

from suspension_control.controllers.base import (
    ControllerContext,
    PlanningInfo,
    SuspensionCommand,
    WheelScale,
)
from suspension_control.controllers.identity import IdentityController
from suspension_control.controllers.pid import FeedbackPIDConfig, FeedbackPIDController
from suspension_control.controllers.rl_residual import (
    ResidualRLConfig,
    ResidualRLController,
)
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
from suspension_control.runtime.planning_provider import (
    make_planning_provider,
    planning_diagnostics,
)
from suspension_control.runtime.route_progress import RouteProgressTracker
from suspension_control.rl.diagnostics import wheel_values_from_mapping
from suspension_control.rl.reward import RewardTransition, SuspensionReward
from suspension_control.rl.reward_calibration import (
    write_phase3b_reward_calibration_report,
)
from suspension_control.rl.reward_sanity import write_phase3_reward_sanity_report
from suspension_control.rl.task_info import TaskInfoBuilder, TaskInfoBuilderConfig


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
DEFAULT_RL_RESIDUAL_CONFIG = os.path.join(
    SCRIPT_DIR, "suspension_control", "configs", "rl_residual.yaml")


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
    ("rl_residual_skyhook", {
        "name": "S4_rl_residual_skyhook",
        "label": "S4 residual RL over skyhook",
        "controller": "rl_residual",
        "uses_suspension_api": True,
    }),
    ("rl_residual_pid", {
        "name": "S5_rl_residual_pid",
        "label": "S5 residual RL over PID",
        "controller": "rl_residual_pid",
        "uses_suspension_api": True,
    }),
    ("rl_residual_skyhook_no_planning", {
        "name": "S6_rl_residual_skyhook_no_planning",
        "label": "S6 residual RL over skyhook without planning",
        "controller": "rl_residual_skyhook_no_planning",
        "uses_suspension_api": True,
        "force_empty_planning": True,
    }),
    ("rl_zero_residual_pid", {
        "name": "S7_rl_zero_residual_pid",
        "label": "S7 zero-residual RL over PID",
        "controller": "rl_zero_residual_pid",
        "uses_suspension_api": True,
        "phase2_baseline_scenario": "S2_pid",
    }),
    ("rl_zero_residual_skyhook", {
        "name": "S8_rl_zero_residual_skyhook",
        "label": "S8 zero-residual RL over skyhook",
        "controller": "rl_zero_residual_skyhook",
        "uses_suspension_api": True,
        "phase2_baseline_scenario": "S3_skyhook",
    }),
    ("rl_const_plus_0p02_skyhook", {
        "name": "S9_rl_const_plus_0p02_skyhook",
        "label": "S9 const action +0.02 RL over skyhook",
        "controller": "rl_const_plus_0p02_skyhook",
        "uses_suspension_api": True,
        "phase2c_expected_action": 0.02,
        "phase2c_role": "constant_positive_canary",
    }),
    ("rl_const_minus_0p02_skyhook", {
        "name": "S10_rl_const_minus_0p02_skyhook",
        "label": "S10 const action -0.02 RL over skyhook",
        "controller": "rl_const_minus_0p02_skyhook",
        "uses_suspension_api": True,
        "phase2c_expected_action": -0.02,
        "phase2c_role": "constant_negative_canary",
    }),
    ("rl_random_small_skyhook", {
        "name": "S11_rl_random_small_skyhook",
        "label": "S11 random action 0.02 RL over skyhook",
        "controller": "rl_random_small_skyhook",
        "uses_suspension_api": True,
        "phase2c_expected_action": "",
        "phase2c_role": "random_small_canary",
    }),
    ("rl_const_action_plus_0p25_skyhook", {
        "name": "S12_rl_const_action_plus_0p25_skyhook",
        "label": "S12 const action +0.25 RL over skyhook",
        "controller": "rl_const_action_plus_0p25_skyhook",
        "uses_suspension_api": True,
        "phase2c_expected_action": 0.25,
        "phase2c_role": "constant_positive_canary",
    }),
    ("rl_const_action_minus_0p25_skyhook", {
        "name": "S13_rl_const_action_minus_0p25_skyhook",
        "label": "S13 const action -0.25 RL over skyhook",
        "controller": "rl_const_action_minus_0p25_skyhook",
        "uses_suspension_api": True,
        "phase2c_expected_action": -0.25,
        "phase2c_role": "constant_negative_canary",
    }),
    ("rl_random_action_0p10_skyhook", {
        "name": "S14_rl_random_action_0p10_skyhook",
        "label": "S14 random action 0.10 RL over skyhook",
        "controller": "rl_random_action_0p10_skyhook",
        "uses_suspension_api": True,
        "phase2c_expected_action": "",
        "phase2c_role": "random_action_canary",
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
    "speed",
    "roll",
    "local_ay",
    "yaw_rate",
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
    "planning_available",
    "planning_source",
    "planning_frame",
    "planning_age_frames",
    "preview_curvature_now",
    "preview_curvature_mean",
    "preview_curvature_max_abs",
    "preview_curvature_signed_peak",
    "preview_target_speed_now",
    "preview_target_speed_mean",
    "preview_target_speed_min",
    "preview_target_speed_max",
    "preview_longitudinal_acc_mean",
    "preview_longitudinal_acc_max_abs",
    "preview_lateral_acc_mean",
    "preview_lateral_acc_max_abs",
    "preview_steer_now",
    "preview_steer_mean",
    "preview_steer_max_abs",
    "preview_throttle_mean",
    "preview_brake_mean",
    "preview_brake_max",
    "time_to_hard_brake",
    "time_to_sharp_turn",
    "time_to_high_lateral_acc",
    "planning_jsonl_malformed_lines",
    "planning_jsonl_rejected_messages",
    "planning_jsonl_read_errors",
    "rl_baseline",
    "rl_residual_mode",
    "rl_action_scale",
    "rl_residual_gain",
    "rl_scripted_residual_kind",
    "rl_scripted_residual_value",
    "rl_policy_available",
    "rl_policy_status",
    "rl_policy_builtin_id",
    "rl_policy_alias_deprecated",
    "rl_observation_valid",
    "rl_safety_gain",
    "rl_safety_gate_active",
    "rl_safety_gate_reason",
    "rl_safety_gate_speed_limit",
    "rl_safety_gate_roll_limit",
    "rl_safety_gate_pitch_limit",
    "rl_safety_gate_lateral_acc_limit",
    "rl_safety_gate_yaw_rate_limit",
    "rl_safety_gate_nonfinite_obs",
    "rl_safety_gate_action_invalid",
    "rl_fallback_reason",
    "rl_action_fl",
    "rl_action_fr",
    "rl_action_rl",
    "rl_action_rr",
    "rl_raw_residual_damper_fl",
    "rl_raw_residual_damper_fr",
    "rl_raw_residual_damper_rl",
    "rl_raw_residual_damper_rr",
    "rl_scaled_residual_damper_fl",
    "rl_scaled_residual_damper_fr",
    "rl_scaled_residual_damper_rl",
    "rl_scaled_residual_damper_rr",
    "rl_scale_clipped_residual_damper_fl",
    "rl_scale_clipped_residual_damper_fr",
    "rl_scale_clipped_residual_damper_rl",
    "rl_scale_clipped_residual_damper_rr",
    "rl_safety_scaled_residual_damper_fl",
    "rl_safety_scaled_residual_damper_fr",
    "rl_safety_scaled_residual_damper_rl",
    "rl_safety_scaled_residual_damper_rr",
    "rl_rate_limited_residual_damper_fl",
    "rl_rate_limited_residual_damper_fr",
    "rl_rate_limited_residual_damper_rl",
    "rl_rate_limited_residual_damper_rr",
    "rl_residual_damper_fl",
    "rl_residual_damper_fr",
    "rl_residual_damper_rl",
    "rl_residual_damper_rr",
    "rl_final_residual_damper_fl",
    "rl_final_residual_damper_fr",
    "rl_final_residual_damper_rl",
    "rl_final_residual_damper_rr",
    "rl_baseline_damper_fl",
    "rl_baseline_damper_fr",
    "rl_baseline_damper_rl",
    "rl_baseline_damper_rr",
    "rl_final_damper_fl",
    "rl_final_damper_fr",
    "rl_final_damper_rl",
    "rl_final_damper_rr",
    "rl_planning_available",
    "rl_observation_size",
    "rl_observation_clip_count",
    "rl_mean_action",
    "rl_mean_abs_action",
    "rl_mean_raw_residual_damper",
    "rl_mean_abs_raw_residual_damper",
    "rl_mean_scaled_residual_damper",
    "rl_mean_abs_scaled_residual_damper",
    "rl_mean_scale_clipped_residual_damper",
    "rl_mean_abs_scale_clipped_residual_damper",
    "rl_mean_residual_damper",
    "rl_mean_abs_residual_damper",
    "rl_mean_final_residual_damper",
    "rl_mean_abs_final_residual_damper",
    "rl_residual_scale_clip",
    "rl_damper_final_clamp",
    "rl_residual_saturation",
    "reward_mode",
    "reward_baseline_command_source",
    "reward_total",
    "reward_alive_bonus",
    "reward_comfort",
    "reward_stability",
    "reward_task",
    "reward_action",
    "reward_safety",
    "reward_cost_comfort",
    "reward_cost_stability",
    "reward_cost_task",
    "reward_cost_action",
    "reward_cost_safety",
    "reward_term_abs_az",
    "reward_term_abs_lat_acc",
    "reward_term_abs_long_acc",
    "reward_term_jerk_z",
    "reward_term_jerk_y",
    "reward_term_jerk_x",
    "reward_term_roll_rate",
    "reward_term_pitch_rate",
    "reward_term_roll",
    "reward_term_pitch",
    "reward_term_yaw_rate",
    "reward_term_stability_lat_acc",
    "reward_term_body_activity",
    "reward_term_corner_vz",
    "reward_term_route_deviation",
    "reward_term_low_speed_not_planned",
    "reward_term_progress_stall",
    "reward_term_negative_progress",
    "reward_term_insufficient_progress",
    "reward_term_target_speed_error",
    "reward_term_abs_speed_error",
    "reward_term_lane_invasion",
    "reward_term_collision",
    "reward_term_red_light",
    "reward_term_blocked_vehicle",
    "reward_term_route_timeout",
    "reward_term_action_mag",
    "reward_term_action_rate",
    "reward_term_damper_rate",
    "reward_term_baseline_dev",
    "reward_term_residual_damper_mag",
    "reward_term_residual_damper_rate",
    "reward_term_final_damper_rate",
    "reward_term_spring_dev",
    "reward_term_safety_gate_active",
    "reward_term_safety_gain_loss",
    "reward_term_observation_clip_count",
    "reward_term_terminal_collision",
    "reward_term_terminal_route_failure",
    "reward_term_nonfinite",
    "reward_route_deviation_used",
    "reward_route_deviation_enabled",
    "reward_route_deviation_valid",
    "reward_task_capped_without_infraction",
    "reward_task_abs_max_without_infraction",
    "rl_action_rate",
    "rl_final_damper_rate",
    "route_progress_available",
    "route_progress_raw_m",
    "route_progress_m",
    "route_progress_fraction",
    "route_progress_monotonic_m",
    "route_progress_monotonic_fraction",
    "route_progress_total_length_m",
    "route_total_length_m",
    "route_progress_delta_m",
    "route_delta_progress_m",
    "route_progress_rate_mps",
    "route_progress_raw_negative_delta_m",
    "route_progress_negative_raw",
    "route_progress_raw_deviation_m",
    "route_progress_deviation_valid",
    "route_deviation_m",
    "route_deviation_valid",
    "route_deviation",
    "route_distance_to_end_m",
    "route_progress_nearest_segment_index",
    "route_progress_projection_segment",
    "route_progress_global_research_used",
    "route_progress_search_window_start",
    "route_progress_search_window_end",
    "route_progress_tracker_status",
    "route_progress_error",
    "route_progress_stall",
    "progress_stall_count",
    "delta_progress",
    "target_speed",
    "target_speed_source",
    "target_speed_error",
    "speed_error",
    "abs_speed_error",
    "planned_stop",
    "low_speed_not_planned",
    "low_speed_not_planned_severity",
    "progress_stall",
    "negative_progress",
    "route_completion_proxy",
    "collision_count",
    "lane_invasion_count",
    "red_light_count",
    "blocked_vehicle",
    "route_timeout",
    "route_failed",
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


def build_rl_residual_config(
    args: argparse.Namespace,
    baseline_override: str = "",
) -> ResidualRLConfig:
    values: Dict[str, Any] = {}
    if args.rl_residual_config and os.path.isfile(args.rl_residual_config):
        values.update(read_flat_yaml(args.rl_residual_config))
    if args.rl_policy:
        values["policy_path"] = expand_path(args.rl_policy)
    if args.rl_normalizer:
        values["normalizer_path"] = expand_path(args.rl_normalizer)
    if baseline_override:
        values["baseline"] = baseline_override
    return ResidualRLConfig.from_mapping(values)


def build_zero_residual_config(
    args: argparse.Namespace,
    baseline: str,
) -> ResidualRLConfig:
    values: Dict[str, Any] = {}
    if args.rl_residual_config and os.path.isfile(args.rl_residual_config):
        values.update(read_flat_yaml(args.rl_residual_config))
    if args.rl_normalizer:
        values["normalizer_path"] = expand_path(args.rl_normalizer)
    values.update({
        "baseline": baseline,
        "policy_path": "dummy_zero",
        "allow_untrained_policy": False,
        "deterministic_policy": True,
    })
    return ResidualRLConfig.from_mapping(values)


def build_builtin_policy_residual_config(
    args: argparse.Namespace,
    baseline: str,
    policy_path: str,
) -> ResidualRLConfig:
    values: Dict[str, Any] = {}
    if args.rl_residual_config and os.path.isfile(args.rl_residual_config):
        values.update(read_flat_yaml(args.rl_residual_config))
    if args.rl_normalizer:
        values["normalizer_path"] = expand_path(args.rl_normalizer)
    values.update({
        "baseline": baseline,
        "policy_path": policy_path,
        "allow_untrained_policy": False,
        "deterministic_policy": True,
    })
    return ResidualRLConfig.from_mapping(values)


def make_controller(controller_name: str, args: argparse.Namespace):
    if controller_name == "stock":
        return None
    if controller_name == "identity":
        return IdentityController()
    if controller_name == "pid":
        return FeedbackPIDController(build_pid_config(args.pid_config))
    if controller_name == "skyhook":
        return SkyhookController(build_skyhook_config(args.skyhook_config))
    if controller_name == "rl_residual":
        return ResidualRLController(build_rl_residual_config(args))
    if controller_name == "rl_residual_pid":
        return ResidualRLController(build_rl_residual_config(args, "pid"))
    if controller_name == "rl_residual_skyhook_no_planning":
        return ResidualRLController(build_rl_residual_config(args, "skyhook"))
    if controller_name == "rl_zero_residual_pid":
        return ResidualRLController(build_zero_residual_config(args, "pid"))
    if controller_name == "rl_zero_residual_skyhook":
        return ResidualRLController(build_zero_residual_config(args, "skyhook"))
    if controller_name == "rl_const_plus_0p02_skyhook":
        return ResidualRLController(build_builtin_policy_residual_config(
            args,
            "skyhook",
            "const_action_plus_0p02"))
    if controller_name == "rl_const_minus_0p02_skyhook":
        return ResidualRLController(build_builtin_policy_residual_config(
            args,
            "skyhook",
            "const_action_minus_0p02"))
    if controller_name == "rl_random_small_skyhook":
        return ResidualRLController(build_builtin_policy_residual_config(
            args,
            "skyhook",
            "random_action_0p02"))
    if controller_name == "rl_const_action_plus_0p25_skyhook":
        return ResidualRLController(build_builtin_policy_residual_config(
            args,
            "skyhook",
            "const_action_plus_0p25"))
    if controller_name == "rl_const_action_minus_0p25_skyhook":
        return ResidualRLController(build_builtin_policy_residual_config(
            args,
            "skyhook",
            "const_action_minus_0p25"))
    if controller_name == "rl_random_action_0p10_skyhook":
        return ResidualRLController(build_builtin_policy_residual_config(
            args,
            "skyhook",
            "random_action_0p10"))
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
    scenario_name_to_key = {
        scenario["name"]: key
        for key, scenario in SCENARIOS.items()
    }
    scenario_code_to_key = {
        scenario["name"].split("_", 1)[0]: key
        for key, scenario in SCENARIOS.items()
    }
    selected = []
    for name in parse_csv_list(value):
        key = name
        if key not in SCENARIOS:
            scenario_code = name if "_" not in name else ""
            key = scenario_name_to_key.get(
                name,
                scenario_code_to_key.get(scenario_code, key))
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


def route_progress_id_from_subset(value: Any) -> str:
    parts = parse_csv_list(str(value or ""))
    if not parts:
        return ""
    first = parts[0]
    if "-" in first:
        first = first.split("-", 1)[0]
    return first.strip()


def reward_action_from_diagnostics(
    diagnostics: Mapping[str, Any],
    wheel_count: int,
) -> Tuple[float, ...]:
    values = wheel_values_from_mapping(diagnostics, "rl_action_", wheel_count)
    if values:
        return values
    return tuple(0.0 for _ in range(wheel_count))


def damper_scales_from_command(command: Any) -> Tuple[float, ...]:
    return tuple(
        float(getattr(wheel, "damper_scale", 1.0))
        for wheel in tuple(getattr(command, "wheels", ()) or ()))


def baseline_command_for_reward(
    final_command: SuspensionCommand,
    diagnostics: Mapping[str, Any],
) -> Tuple[SuspensionCommand, str]:
    wheels = tuple(final_command.wheels)
    baseline_dampers = wheel_values_from_mapping(
        diagnostics,
        "rl_baseline_damper_",
        len(wheels))
    if baseline_dampers:
        return SuspensionCommand(tuple(
            WheelScale(
                spring_scale=float(getattr(wheel, "spring_scale", 1.0)),
                damper_scale=float(damper))
            for wheel, damper in zip(wheels, baseline_dampers)
        )), "rl_baseline_diagnostics"
    return final_command, (
        "output_command_fallback"
        if diagnostics.get("rl_policy_available", "") != ""
        else "output_command")


def task_info_from_planning(planning: Any) -> Dict[str, Any]:
    if planning is None:
        return {}
    if hasattr(planning, "as_dict"):
        return dict(planning.as_dict())
    if isinstance(planning, Mapping):
        return dict(planning)
    return {}


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
        self.previous_action_by_actor_id: Dict[int, Tuple[float, ...]] = {}
        self.previous_final_damper_by_actor_id: Dict[int, Tuple[float, ...]] = {}
        self.next_episode_index = 0
        self.planning_provider = make_planning_provider(args)
        self.route_progress_tracker = RouteProgressTracker(
            expand_path(args.routes),
            route_id=route_progress_id_from_subset(args.routes_subset))
        self.task_info_builder = TaskInfoBuilder(TaskInfoBuilderConfig(
            nominal_target_speed=getattr(
                args,
                "reward_nominal_target_speed",
                TaskInfoBuilderConfig.nominal_target_speed),
            reward_low_speed_threshold_mps=getattr(
                args,
                "reward_low_speed_threshold_mps",
                TaskInfoBuilderConfig.reward_low_speed_threshold_mps),
            reward_min_progress_rate_mps=getattr(
                args,
                "reward_min_progress_rate_mps",
                TaskInfoBuilderConfig.reward_min_progress_rate_mps),
            reward_planned_stop_brake_threshold=getattr(
                args,
                "reward_planned_stop_brake_threshold",
                TaskInfoBuilderConfig.reward_planned_stop_brake_threshold),
            progress_stall_steps=getattr(
                args,
                "reward_progress_stall_steps",
                TaskInfoBuilderConfig.progress_stall_steps),
            progress_stall_epsilon_m=getattr(
                args,
                "reward_progress_stall_epsilon_m",
                TaskInfoBuilderConfig.progress_stall_epsilon_m)))
        self.reward = SuspensionReward()

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
        try:
            carla = import_carla()
        except Exception as error:
            self.error = error
            self.log_event(
                event_writer,
                "fatal_error",
                message="import_carla failed: %s: %s" % (
                    error.__class__.__name__,
                    error))
            return None
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
            self.previous_action_by_actor_id.pop(actor_id, None)
            self.previous_final_damper_by_actor_id.pop(actor_id, None)
            self.task_info_builder.reset(actor_id)
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
        if self.scenario.get("force_empty_planning", False):
            planning = PlanningInfo.empty()
        else:
            planning = self.planning_provider.get(state.frame, state, previous_state)
        context = ControllerContext(
            state=state,
            previous_state=previous_state,
            planning=planning,
            native_suspension=native,
            current_suspension=current_suspension,
            step=state.step,
            dt=dt)
        output = controller.compute(context)
        output.command.validate(expected_wheels=len(native.wheels))
        output_diagnostics = dict(output.diagnostics or {})
        route_progress = self.route_progress_tracker.update(
            state,
            actor_id=actor_id,
            dt=dt)
        task_info = self.task_info_builder.build(
            state=state,
            previous_state=previous_state,
            route_progress=route_progress,
            planning=planning,
            live_task_info=task_info_from_planning(planning),
            actor_id=actor_id)
        action = reward_action_from_diagnostics(
            output_diagnostics,
            len(output.command.wheels))
        previous_action = self.previous_action_by_actor_id.get(
            actor_id,
            tuple(0.0 for _ in range(len(action))))
        previous_final_dampers = self.previous_final_damper_by_actor_id.get(
            actor_id,
            ())
        baseline_command, baseline_source = baseline_command_for_reward(
            output.command,
            output_diagnostics)
        _, reward_diagnostics = self.reward.compute(RewardTransition(
            state=state,
            previous_state=previous_state,
            action=action,
            previous_action=previous_action,
            previous_final_damper_scales=previous_final_dampers,
            final_command=output.command,
            baseline_command=baseline_command,
            task_info=task_info,
            diagnostics=output_diagnostics,
            terminal=False,
            dt=dt))
        reward_diagnostics["reward_mode"] = "instantaneous_route_diagnostic"
        reward_diagnostics["reward_baseline_command_source"] = baseline_source

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
            "speed": state.speed,
            "roll": state.roll,
            "local_ay": state.local_ay,
            "yaw_rate": state.yaw_rate,
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
        row.update(planning_diagnostics(planning, current_frame=state.frame))
        row.update(output_diagnostics)
        row.update(route_progress)
        row.update(task_info)
        row.update(reward_diagnostics)
        diagnostic_writer.writerow({
            field: format_value(row.get(field, ""))
            for field in DIAGNOSTIC_FIELDS
        })
        self.previous_action_by_actor_id[actor_id] = tuple(action)
        self.previous_final_damper_by_actor_id[actor_id] = damper_scales_from_command(
            output.command)

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
    try:
        leaderboard_timeout = max(300.0, float(args.timeout))
    except (TypeError, ValueError):
        leaderboard_timeout = 300.0
    env["LEADERBOARD_TIMEOUT"] = str(leaderboard_timeout)
    env["OUT_DIR"] = os.path.join(scenario_dir, "tfpp")
    if args.planning_provider == "jsonl" and args.planning_preview_jsonl:
        env["SUSPENSION_PLANNING_PREVIEW_JSONL"] = expand_path(
            args.planning_preview_jsonl)
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


def summarize_diagnostics_csv(path: str) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "diagnostic_rows": 0,
        "rl_diagnostic_rows": 0,
        "rl_policy_available_rows": 0,
        "rl_policy_available_ratio": "",
        "rl_fallback_rows": 0,
        "rl_fallback_reasons": "",
        "rl_zero_action_rows": 0,
        "rl_nonzero_action_rows": 0,
        "rl_zero_residual_rows": 0,
        "rl_nonzero_residual_rows": 0,
        "rl_mean_safety_gain": "",
        "rl_residual_modes": "",
        "rl_action_scale_mean": "",
        "rl_residual_gain_mean": "",
        "rl_scripted_residual_kinds": "",
        "rl_mean_action": "",
        "rl_mean_abs_action": "",
        "rl_mean_raw_residual_damper": "",
        "rl_mean_abs_raw_residual_damper": "",
        "rl_mean_scaled_residual_damper": "",
        "rl_mean_abs_scaled_residual_damper": "",
        "rl_mean_scale_clipped_residual_damper": "",
        "rl_mean_abs_scale_clipped_residual_damper": "",
        "rl_mean_residual_damper": "",
        "rl_mean_abs_residual_damper": "",
        "rl_mean_final_residual_damper": "",
        "rl_mean_abs_final_residual_damper": "",
        "residual_scale_clip_rows": 0,
        "residual_scale_clip_ratio": "",
        "damper_final_clamp_rows": 0,
        "damper_final_clamp_ratio": "",
        "residual_saturation_rows": 0,
        "residual_saturation_ratio": "",
        "rl_observation_clip_count_max": "",
        "low_speed_mask_rows": 0,
        "low_speed_mask_ratio": "",
        "hard_safety_gate_rows": 0,
        "hard_safety_gate_ratio": "",
        "soft_safety_gain_rows": 0,
        "soft_safety_gain_ratio": "",
        "effective_control_ratio": "",
        "safety_gate_rows": 0,
        "safety_gate_active_rows": 0,
        "safety_gate_active_ratio": "",
        "safety_gate_reason_counts": "",
        "safety_gate_active_speed_min": "",
        "safety_gate_active_speed_mean": "",
        "safety_gate_active_speed_max": "",
        "safety_gate_active_roll_min": "",
        "safety_gate_active_roll_mean": "",
        "safety_gate_active_roll_max": "",
        "safety_gate_active_abs_roll_mean": "",
        "safety_gate_active_abs_roll_max": "",
        "safety_gate_active_lateral_acc_min": "",
        "safety_gate_active_lateral_acc_mean": "",
        "safety_gate_active_lateral_acc_max": "",
        "safety_gate_active_abs_lateral_acc_mean": "",
        "safety_gate_active_abs_lateral_acc_max": "",
        "safety_gate_active_yaw_rate_min": "",
        "safety_gate_active_yaw_rate_mean": "",
        "safety_gate_active_yaw_rate_max": "",
        "safety_gate_active_abs_yaw_rate_mean": "",
        "safety_gate_active_abs_yaw_rate_max": "",
        "planning_available_rows": 0,
        "planning_sources": "",
        "policy_available_ratio": "",
        "fallback_ratio": "",
        "observation_clip_ratio": "",
        "mean_abs_action": "",
        "raw_mean_abs_action": "",
        "raw_mean_abs_residual_damper": "",
        "scaled_mean_abs_residual_damper": "",
        "final_mean_abs_residual_damper": "",
        "mean_abs_residual_damper": "",
        "mean_reward_total": "",
        "mean_reward_comfort": "",
        "mean_reward_stability": "",
        "mean_reward_task": "",
        "mean_reward_action": "",
        "mean_reward_safety": "",
        "reward_rows": 0,
        "reward_row_ratio": "",
        "mean_reward_cost_comfort": "",
        "mean_reward_cost_stability": "",
        "mean_reward_cost_task": "",
        "mean_reward_cost_action": "",
        "mean_reward_cost_safety": "",
        "mean_reward_term_action_mag": "",
        "mean_reward_term_action_rate": "",
        "mean_reward_term_damper_rate": "",
        "mean_reward_term_baseline_dev": "",
        "mean_reward_term_low_speed_not_planned": "",
        "mean_reward_term_progress_stall": "",
        "mean_reward_term_route_deviation": "",
        "mean_reward_term_abs_speed_error": "",
        "mean_reward_route_deviation_used_ratio": "",
        "reward_route_deviation_used_rows": 0,
        "reward_route_deviation_enabled_rows": 0,
        "reward_route_deviation_valid_rows": 0,
        "reward_task_capped_without_infraction_rows": 0,
        "reward_task_capped_without_infraction_ratio": "",
        "route_progress_available_rows": 0,
        "route_progress_available_ratio": "",
        "route_progress_fraction_max": "",
        "route_progress_monotonic_fraction_max": "",
        "route_completion_proxy": "",
        "route_progress_delta_negative_raw_rows": 0,
        "route_progress_delta_negative_raw_ratio": "",
        "route_progress_raw_negative_rows": 0,
        "route_progress_raw_negative_ratio": "",
        "route_progress_monotonic_negative_rows": 0,
        "route_progress_monotonic_negative_ratio": "",
        "route_deviation_m_mean": "",
        "route_deviation_m_max": "",
        "route_deviation_valid_rows": 0,
        "route_deviation_valid_ratio": "",
        "planned_stop_rows": 0,
        "planned_stop_ratio": "",
        "low_speed_not_planned_rows": 0,
        "low_speed_not_planned_ratio": "",
        "progress_stall_rows": 0,
        "progress_stall_ratio": "",
        "route_progress_stall_rows": 0,
        "route_progress_stall_ratio": "",
        "progress_stall_count_max": "",
        "negative_progress_rows": 0,
        "negative_progress_ratio": "",
        "mean_route_progress_rate_mps": "",
        "mean_abs_speed_error": "",
        "collision_count": 0,
        "lane_invasion_count": 0,
        "red_light_count": 0,
        "blocked_vehicle_count": 0,
        "route_timeout_count": 0,
    }
    if not os.path.isfile(path):
        return summary

    fallback_reasons = set()
    planning_sources = set()
    residual_modes = set()
    scripted_residual_kinds = set()
    action_scales: List[float] = []
    residual_gains: List[float] = []
    safety_gains = []
    signed_actions = []
    abs_actions = []
    signed_raw_residuals = []
    abs_raw_residuals = []
    signed_scaled_residuals = []
    abs_scaled_residuals = []
    signed_scale_clipped_residuals = []
    abs_scale_clipped_residuals = []
    signed_residuals = []
    abs_residuals = []
    signed_final_residuals = []
    abs_final_residuals = []
    clip_counts = []
    gate_reason_counts: Dict[str, int] = {}
    gate_active_speeds: List[float] = []
    gate_active_rolls: List[float] = []
    gate_active_lateral_accs: List[float] = []
    gate_active_yaw_rates: List[float] = []
    reward_total: List[float] = []
    reward_comfort: List[float] = []
    reward_stability: List[float] = []
    reward_task: List[float] = []
    reward_action: List[float] = []
    reward_safety: List[float] = []
    reward_cost_comfort: List[float] = []
    reward_cost_stability: List[float] = []
    reward_cost_task: List[float] = []
    reward_cost_action: List[float] = []
    reward_cost_safety: List[float] = []
    reward_term_action_mag: List[float] = []
    reward_term_action_rate: List[float] = []
    reward_term_damper_rate: List[float] = []
    reward_term_baseline_dev: List[float] = []
    reward_term_low_speed_not_planned: List[float] = []
    reward_term_progress_stall: List[float] = []
    reward_term_route_deviation: List[float] = []
    reward_term_abs_speed_error: List[float] = []
    reward_route_deviation_used: List[float] = []
    route_progress_values: List[float] = []
    route_progress_monotonic_values: List[float] = []
    route_completion_values: List[float] = []
    route_deviation_values: List[float] = []
    progress_stall_counts: List[float] = []
    route_progress_rates: List[float] = []
    abs_speed_errors: List[float] = []
    collision_counts: List[float] = []
    lane_counts: List[float] = []
    red_light_counts: List[float] = []
    blocked_counts: List[float] = []
    timeout_counts: List[float] = []

    with open(path) as csv_file:
        for row in csv.DictReader(csv_file):
            summary["diagnostic_rows"] += 1
            has_rl = row.get("rl_policy_available", "") != ""
            if has_rl:
                summary["rl_diagnostic_rows"] += 1
            residual_mode = row.get("rl_residual_mode", "")
            if residual_mode:
                residual_modes.add(residual_mode)
            scripted_kind = row.get("rl_scripted_residual_kind", "")
            if scripted_kind:
                scripted_residual_kinds.add(scripted_kind)
            _append_float(action_scales, row.get("rl_action_scale"))
            _append_float(residual_gains, row.get("rl_residual_gain"))
            policy_available = safe_float(row.get("rl_policy_available"))
            if policy_available is not None and policy_available > 0.0:
                summary["rl_policy_available_rows"] += 1
            reason = row.get("rl_fallback_reason", "")
            if reason:
                summary["rl_fallback_rows"] += 1
                fallback_reasons.add(reason)
            planning_available = safe_float(row.get("planning_available"))
            if planning_available is not None and planning_available > 0.0:
                summary["planning_available_rows"] += 1
            planning_source = row.get("planning_source", "")
            if planning_source:
                planning_sources.add(planning_source)
            safety_gain = safe_float(row.get("rl_safety_gain"))
            if safety_gain is not None:
                safety_gains.append(safety_gain)
            _append_float(signed_actions, row.get("rl_mean_action"))
            action = safe_float(row.get("rl_mean_abs_action"))
            if action is not None:
                abs_actions.append(action)
                if abs(action) <= 1.0e-12:
                    summary["rl_zero_action_rows"] += 1
                else:
                    summary["rl_nonzero_action_rows"] += 1
            _append_float(signed_raw_residuals, row.get("rl_mean_raw_residual_damper"))
            _append_float(abs_raw_residuals, row.get("rl_mean_abs_raw_residual_damper"))
            _append_float(signed_scaled_residuals, row.get("rl_mean_scaled_residual_damper"))
            _append_float(abs_scaled_residuals, row.get("rl_mean_abs_scaled_residual_damper"))
            _append_float(
                signed_scale_clipped_residuals,
                row.get("rl_mean_scale_clipped_residual_damper"))
            _append_float(
                abs_scale_clipped_residuals,
                row.get("rl_mean_abs_scale_clipped_residual_damper"))
            _append_float(signed_residuals, row.get("rl_mean_residual_damper"))
            residual = safe_float(row.get("rl_mean_abs_residual_damper"))
            if residual is not None:
                abs_residuals.append(residual)
                if abs(residual) <= 1.0e-12:
                    summary["rl_zero_residual_rows"] += 1
                else:
                    summary["rl_nonzero_residual_rows"] += 1
            _append_float(signed_final_residuals, row.get("rl_mean_final_residual_damper"))
            _append_float(abs_final_residuals, row.get("rl_mean_abs_final_residual_damper"))
            if _positive_number(row.get("rl_residual_scale_clip")):
                summary["residual_scale_clip_rows"] += 1
            if _positive_number(row.get("rl_damper_final_clamp")):
                summary["damper_final_clamp_rows"] += 1
            if _positive_number(row.get("rl_residual_saturation")):
                summary["residual_saturation_rows"] += 1
            _append_float(clip_counts, row.get("rl_observation_clip_count"))
            _append_float(reward_total, row.get("reward_total"))
            _append_float(reward_comfort, row.get("reward_comfort"))
            _append_float(reward_stability, row.get("reward_stability"))
            _append_float(reward_task, row.get("reward_task"))
            _append_float(reward_action, row.get("reward_action"))
            _append_float(reward_safety, row.get("reward_safety"))
            _append_float(reward_cost_comfort, row.get("reward_cost_comfort"))
            _append_float(reward_cost_stability, row.get("reward_cost_stability"))
            _append_float(reward_cost_task, row.get("reward_cost_task"))
            _append_float(reward_cost_action, row.get("reward_cost_action"))
            _append_float(reward_cost_safety, row.get("reward_cost_safety"))
            _append_float(reward_term_action_mag, row.get("reward_term_action_mag"))
            _append_float(reward_term_action_rate, row.get("reward_term_action_rate"))
            _append_float(reward_term_damper_rate, row.get("reward_term_damper_rate"))
            _append_float(reward_term_baseline_dev, row.get("reward_term_baseline_dev"))
            _append_float(
                reward_term_low_speed_not_planned,
                row.get("reward_term_low_speed_not_planned"))
            _append_float(
                reward_term_progress_stall,
                row.get("reward_term_progress_stall"))
            _append_float(
                reward_term_route_deviation,
                row.get("reward_term_route_deviation"))
            _append_float(
                reward_term_abs_speed_error,
                row.get("reward_term_abs_speed_error"))
            if safe_float(row.get("reward_total")) is not None:
                summary["reward_rows"] += 1
            route_deviation_used = safe_float(row.get("reward_route_deviation_used"))
            if route_deviation_used is not None:
                reward_route_deviation_used.append(route_deviation_used)
                if route_deviation_used > 0.0:
                    summary["reward_route_deviation_used_rows"] += 1
            if _positive_number(row.get("reward_route_deviation_enabled")):
                summary["reward_route_deviation_enabled_rows"] += 1
            if _positive_number(row.get("reward_route_deviation_valid")):
                summary["reward_route_deviation_valid_rows"] += 1
            if _positive_number(row.get("reward_task_capped_without_infraction")):
                summary["reward_task_capped_without_infraction_rows"] += 1
            route_progress_available = safe_float(row.get("route_progress_available"))
            if route_progress_available is not None and route_progress_available > 0.0:
                summary["route_progress_available_rows"] += 1
            _append_float(route_progress_values, row.get("route_progress_fraction"))
            _append_float(
                route_progress_monotonic_values,
                row.get("route_progress_monotonic_fraction"))
            _append_float(route_completion_values, row.get("route_completion_proxy"))
            route_deviation = row.get(
                "route_deviation_m",
                row.get("route_progress_raw_deviation_m", row.get("route_deviation")))
            _append_float(route_deviation_values, route_deviation)
            if (
                    _positive_number(row.get("route_progress_deviation_valid")) or
                    _positive_number(row.get("route_deviation_valid"))):
                summary["route_deviation_valid_rows"] += 1
            _append_float(route_progress_rates, row.get("route_progress_rate_mps"))
            _append_float(abs_speed_errors, row.get("abs_speed_error"))
            if _raw_negative_progress(row):
                summary["route_progress_delta_negative_raw_rows"] += 1
                summary["route_progress_raw_negative_rows"] += 1
            if _negative_number(row.get("route_progress_delta_m")):
                summary["route_progress_monotonic_negative_rows"] += 1
            if _positive_number(row.get("planned_stop")):
                summary["planned_stop_rows"] += 1
            if _positive_number(row.get("low_speed_not_planned")):
                summary["low_speed_not_planned_rows"] += 1
            if _positive_number(row.get("route_progress_stall")):
                summary["route_progress_stall_rows"] += 1
            if _positive_number(row.get("progress_stall")):
                summary["progress_stall_rows"] += 1
            _append_float(progress_stall_counts, row.get("progress_stall_count"))
            if _positive_number(row.get("negative_progress")):
                summary["negative_progress_rows"] += 1
            _append_float(collision_counts, row.get("collision_count"))
            _append_float(lane_counts, row.get("lane_invasion_count"))
            _append_float(red_light_counts, row.get("red_light_count"))
            _append_float(blocked_counts, row.get("blocked_vehicle"))
            _append_float(blocked_counts, row.get("vehicle_blocked"))
            _append_float(timeout_counts, row.get("route_timeout"))
            gate_active = safe_float(row.get("rl_safety_gate_active"))
            if gate_active is not None:
                summary["safety_gate_rows"] += 1
                if gate_active > 0.0:
                    summary["safety_gate_active_rows"] += 1
                    _append_float(gate_active_speeds, row.get("speed"))
                    _append_float(gate_active_rolls, row.get("roll"))
                    _append_float(gate_active_lateral_accs, row.get("local_ay"))
                    _append_float(gate_active_yaw_rates, row.get("yaw_rate"))
            reasons = row.get("rl_safety_gate_reason", "")
            for reason in [item for item in reasons.split(";") if item]:
                gate_reason_counts[reason] = gate_reason_counts.get(reason, 0) + 1
            reason_set = set(item for item in reasons.split(";") if item)
            low_speed_mask = (
                _positive_number(row.get("rl_safety_gate_speed_limit")) or
                "speed_below_min" in reason_set)
            hard_condition = (
                _positive_number(row.get("rl_safety_gate_nonfinite_obs")) or
                _positive_number(row.get("rl_safety_gate_action_invalid")) or
                "nonfinite_obs" in reason_set or
                "action_invalid" in reason_set)
            if low_speed_mask:
                summary["low_speed_mask_rows"] += 1
            if (
                    not low_speed_mask and
                    (hard_condition or (
                        safety_gain is not None and
                        safety_gain <= 1.0e-12 and
                        gate_active is not None and
                        gate_active > 0.0))):
                summary["hard_safety_gate_rows"] += 1
            if (
                    not low_speed_mask and
                    safety_gain is not None and
                    safety_gain > 1.0e-12 and
                    safety_gain < 1.0 - 1.0e-12):
                summary["soft_safety_gain_rows"] += 1

    rl_rows = summary["rl_diagnostic_rows"]
    summary["rl_policy_available_ratio"] = (
        float(summary["rl_policy_available_rows"]) / float(rl_rows)
        if rl_rows else "")
    summary["policy_available_ratio"] = summary["rl_policy_available_ratio"]
    summary["fallback_ratio"] = _ratio_or_empty(
        summary["rl_fallback_rows"],
        rl_rows)
    summary["low_speed_mask_ratio"] = _ratio_or_empty(
        summary["low_speed_mask_rows"],
        rl_rows)
    summary["hard_safety_gate_ratio"] = _ratio_or_empty(
        summary["hard_safety_gate_rows"],
        rl_rows)
    summary["soft_safety_gain_ratio"] = _ratio_or_empty(
        summary["soft_safety_gain_rows"],
        rl_rows)
    summary["effective_control_ratio"] = _ratio_or_empty(
        summary["rl_nonzero_residual_rows"],
        rl_rows)
    summary["observation_clip_ratio"] = _ratio_or_empty(
        sum(1 for value in clip_counts if value > 0.0),
        rl_rows)
    gate_rows = summary["safety_gate_rows"]
    summary["safety_gate_active_ratio"] = (
        float(summary["safety_gate_active_rows"]) / float(gate_rows)
        if gate_rows else "")
    summary["safety_gate_reason_counts"] = ";".join(
        "%s=%d" % (reason, gate_reason_counts[reason])
        for reason in sorted(gate_reason_counts))
    summary["rl_fallback_reasons"] = ";".join(sorted(fallback_reasons))
    summary["planning_sources"] = ";".join(sorted(planning_sources))
    summary["rl_mean_safety_gain"] = _mean_or_empty(safety_gains)
    summary["rl_residual_modes"] = ";".join(sorted(residual_modes))
    summary["rl_action_scale_mean"] = _mean_or_empty(action_scales)
    summary["rl_residual_gain_mean"] = _mean_or_empty(residual_gains)
    summary["rl_scripted_residual_kinds"] = ";".join(sorted(scripted_residual_kinds))
    summary["rl_mean_action"] = _mean_or_empty(signed_actions)
    summary["rl_mean_abs_action"] = _mean_or_empty(abs_actions)
    summary["rl_mean_raw_residual_damper"] = _mean_or_empty(signed_raw_residuals)
    summary["rl_mean_abs_raw_residual_damper"] = _mean_or_empty(abs_raw_residuals)
    summary["rl_mean_scaled_residual_damper"] = _mean_or_empty(signed_scaled_residuals)
    summary["rl_mean_abs_scaled_residual_damper"] = _mean_or_empty(abs_scaled_residuals)
    summary["rl_mean_scale_clipped_residual_damper"] = _mean_or_empty(signed_scale_clipped_residuals)
    summary["rl_mean_abs_scale_clipped_residual_damper"] = _mean_or_empty(abs_scale_clipped_residuals)
    summary["rl_mean_residual_damper"] = _mean_or_empty(signed_residuals)
    summary["rl_mean_abs_residual_damper"] = _mean_or_empty(abs_residuals)
    summary["rl_mean_final_residual_damper"] = _mean_or_empty(signed_final_residuals)
    summary["rl_mean_abs_final_residual_damper"] = _mean_or_empty(abs_final_residuals)
    summary["mean_abs_action"] = summary["rl_mean_abs_action"]
    summary["raw_mean_abs_action"] = summary["rl_mean_abs_action"]
    summary["raw_mean_abs_residual_damper"] = summary["rl_mean_abs_raw_residual_damper"]
    summary["scaled_mean_abs_residual_damper"] = summary["rl_mean_abs_scaled_residual_damper"]
    summary["final_mean_abs_residual_damper"] = summary["rl_mean_abs_final_residual_damper"]
    summary["mean_abs_residual_damper"] = summary["rl_mean_abs_residual_damper"]
    summary["residual_scale_clip_ratio"] = _ratio_or_empty(
        summary["residual_scale_clip_rows"],
        rl_rows)
    summary["damper_final_clamp_ratio"] = _ratio_or_empty(
        summary["damper_final_clamp_rows"],
        rl_rows)
    summary["residual_saturation_ratio"] = _ratio_or_empty(
        summary["residual_saturation_rows"],
        rl_rows)
    summary["mean_reward_total"] = _mean_or_empty(reward_total)
    summary["mean_reward_comfort"] = _mean_or_empty(reward_comfort)
    summary["mean_reward_stability"] = _mean_or_empty(reward_stability)
    summary["mean_reward_task"] = _mean_or_empty(reward_task)
    summary["mean_reward_action"] = _mean_or_empty(reward_action)
    summary["mean_reward_safety"] = _mean_or_empty(reward_safety)
    diagnostic_rows = summary["diagnostic_rows"]
    summary["reward_row_ratio"] = _ratio_or_empty(
        summary["reward_rows"],
        diagnostic_rows)
    summary["mean_reward_cost_comfort"] = _mean_or_empty(reward_cost_comfort)
    summary["mean_reward_cost_stability"] = _mean_or_empty(reward_cost_stability)
    summary["mean_reward_cost_task"] = _mean_or_empty(reward_cost_task)
    summary["mean_reward_cost_action"] = _mean_or_empty(reward_cost_action)
    summary["mean_reward_cost_safety"] = _mean_or_empty(reward_cost_safety)
    summary["mean_reward_term_action_mag"] = _mean_or_empty(
        reward_term_action_mag)
    summary["mean_reward_term_action_rate"] = _mean_or_empty(
        reward_term_action_rate)
    summary["mean_reward_term_damper_rate"] = _mean_or_empty(
        reward_term_damper_rate)
    summary["mean_reward_term_baseline_dev"] = _mean_or_empty(
        reward_term_baseline_dev)
    summary["mean_reward_term_low_speed_not_planned"] = _mean_or_empty(
        reward_term_low_speed_not_planned)
    summary["mean_reward_term_progress_stall"] = _mean_or_empty(
        reward_term_progress_stall)
    summary["mean_reward_term_route_deviation"] = _mean_or_empty(
        reward_term_route_deviation)
    summary["mean_reward_term_abs_speed_error"] = _mean_or_empty(
        reward_term_abs_speed_error)
    summary["mean_reward_route_deviation_used_ratio"] = _ratio_or_empty(
        summary["reward_route_deviation_used_rows"],
        diagnostic_rows)
    summary["reward_task_capped_without_infraction_ratio"] = _ratio_or_empty(
        summary["reward_task_capped_without_infraction_rows"],
        diagnostic_rows)
    summary["route_progress_available_ratio"] = _ratio_or_empty(
        summary["route_progress_available_rows"],
        diagnostic_rows)
    if route_progress_values:
        summary["route_progress_fraction_max"] = max(route_progress_values)
    if route_progress_monotonic_values:
        summary["route_progress_monotonic_fraction_max"] = max(
            route_progress_monotonic_values)
    if route_completion_values:
        summary["route_completion_proxy"] = max(route_completion_values)
    elif route_progress_values:
        summary["route_completion_proxy"] = (
            1 if max(route_progress_values) >= 0.99 else 0)
    if route_deviation_values:
        summary["route_deviation_m_mean"] = _mean_or_empty(route_deviation_values)
        summary["route_deviation_m_max"] = max(route_deviation_values)
    summary["route_progress_delta_negative_raw_ratio"] = _ratio_or_empty(
        summary["route_progress_delta_negative_raw_rows"],
        diagnostic_rows)
    summary["route_progress_raw_negative_ratio"] = _ratio_or_empty(
        summary["route_progress_raw_negative_rows"],
        diagnostic_rows)
    summary["route_progress_monotonic_negative_ratio"] = _ratio_or_empty(
        summary["route_progress_monotonic_negative_rows"],
        diagnostic_rows)
    summary["route_deviation_valid_ratio"] = _ratio_or_empty(
        summary["route_deviation_valid_rows"],
        diagnostic_rows)
    summary["planned_stop_ratio"] = _ratio_or_empty(
        summary["planned_stop_rows"],
        diagnostic_rows)
    summary["low_speed_not_planned_ratio"] = _ratio_or_empty(
        summary["low_speed_not_planned_rows"],
        diagnostic_rows)
    summary["progress_stall_ratio"] = _ratio_or_empty(
        summary["progress_stall_rows"],
        diagnostic_rows)
    summary["route_progress_stall_ratio"] = _ratio_or_empty(
        summary["route_progress_stall_rows"],
        diagnostic_rows)
    summary["progress_stall_count_max"] = (
        max(progress_stall_counts) if progress_stall_counts else "")
    summary["negative_progress_ratio"] = _ratio_or_empty(
        summary["negative_progress_rows"],
        diagnostic_rows)
    summary["mean_route_progress_rate_mps"] = _mean_or_empty(route_progress_rates)
    summary["mean_abs_speed_error"] = _mean_or_empty(abs_speed_errors)
    summary["collision_count"] = max(collision_counts) if collision_counts else 0
    summary["lane_invasion_count"] = max(lane_counts) if lane_counts else 0
    summary["red_light_count"] = max(red_light_counts) if red_light_counts else 0
    summary["blocked_vehicle_count"] = max(blocked_counts) if blocked_counts else 0
    summary["route_timeout_count"] = max(timeout_counts) if timeout_counts else 0
    summary["rl_observation_clip_count_max"] = (
        max(clip_counts) if clip_counts else "")
    _update_min_mean_max(summary, "safety_gate_active_speed", gate_active_speeds)
    _update_min_mean_max(summary, "safety_gate_active_roll", gate_active_rolls)
    _update_min_mean_max(
        summary,
        "safety_gate_active_lateral_acc",
        gate_active_lateral_accs)
    _update_min_mean_max(
        summary,
        "safety_gate_active_yaw_rate",
        gate_active_yaw_rates)
    _update_abs_mean_max(
        summary,
        "safety_gate_active_abs_roll",
        gate_active_rolls)
    _update_abs_mean_max(
        summary,
        "safety_gate_active_abs_lateral_acc",
        gate_active_lateral_accs)
    _update_abs_mean_max(
        summary,
        "safety_gate_active_abs_yaw_rate",
        gate_active_yaw_rates)
    return summary


def _append_float(values: List[float], value: Any) -> None:
    number = safe_float(value)
    if number is not None:
        values.append(number)


def _mean_or_empty(values: Sequence[float]) -> Any:
    return sum(values) / float(len(values)) if values else ""


def _ratio_or_empty(numerator: Any, denominator: Any) -> Any:
    numerator_value = safe_float(numerator)
    denominator_value = safe_float(denominator)
    if (
            numerator_value is None or
            denominator_value is None or
            denominator_value <= 0.0):
        return ""
    return numerator_value / denominator_value


def _update_min_mean_max(
    summary: Dict[str, Any],
    prefix: str,
    values: Sequence[float],
) -> None:
    if not values:
        return
    summary["%s_min" % prefix] = min(values)
    summary["%s_mean" % prefix] = _mean_or_empty(values)
    summary["%s_max" % prefix] = max(values)


def _update_abs_mean_max(
    summary: Dict[str, Any],
    prefix: str,
    values: Sequence[float],
) -> None:
    if not values:
        return
    abs_values = [abs(value) for value in values]
    summary["%s_mean" % prefix] = _mean_or_empty(abs_values)
    summary["%s_max" % prefix] = max(abs_values)


def prefixed_metrics(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {prefix + key: value for key, value in metrics.items()}


ProfileGroupKey = Tuple[str, str]


def read_profile_groups(
    path: str,
) -> Tuple[List[Dict[str, Any]], OrderedDict, OrderedDict]:
    all_rows: List[Dict[str, Any]] = []
    episode_groups: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()
    actor_groups: OrderedDict[ProfileGroupKey, List[Dict[str, Any]]] = OrderedDict()
    if not os.path.isfile(path):
        return all_rows, episode_groups, actor_groups
    with open(path) as csv_file:
        for row in csv.DictReader(csv_file):
            all_rows.append(row)
            episode = row.get("episode_index", "")
            actor = row.get("actor_id", "")
            episode_groups.setdefault(episode, []).append(row)
            actor_groups.setdefault((episode, actor), []).append(row)
    return all_rows, episode_groups, actor_groups


def select_main_profile_group(
    groups: Mapping[ProfileGroupKey, Sequence[Mapping[str, Any]]],
) -> Tuple[ProfileGroupKey, List[Mapping[str, Any]]]:
    if not groups:
        return ("", ""), []
    key, rows = max(groups.items(), key=lambda item: len(item[1]))
    return key, list(rows)


def elapsed_reset_detected(rows: Sequence[Mapping[str, Any]]) -> bool:
    previous: Optional[float] = None
    for row in rows:
        elapsed = safe_float(row.get("elapsed_seconds"))
        if elapsed is None:
            continue
        if previous is not None and elapsed < previous:
            return True
        previous = elapsed
    return False


def add_metric_warning(metrics: Dict[str, Any], reason: str) -> None:
    existing = str(metrics.get("metric_warning", "") or "")
    reasons = [part for part in existing.split(";") if part]
    if reason not in reasons:
        reasons.append(reason)
    metrics["metric_warning"] = ";".join(reasons)


def drop_warmup_rows(
    rows: Sequence[Mapping[str, Any]],
    warmup_seconds: float,
) -> List[Mapping[str, Any]]:
    if warmup_seconds <= 0.0 or not rows:
        return list(rows)
    filtered = []
    for row in rows:
        elapsed = safe_float(row.get("elapsed_seconds"))
        if elapsed is not None and elapsed >= warmup_seconds:
            filtered.append(row)
    return filtered


def elapsed_range(rows: Sequence[Mapping[str, Any]]) -> Tuple[Any, Any]:
    elapsed_values = [
        value for value in (safe_float(row.get("elapsed_seconds")) for row in rows)
        if value is not None
    ]
    if not elapsed_values:
        return "", ""
    return min(elapsed_values), max(elapsed_values)


def update_warmup_excluded_metrics(
    metrics: Dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
    warmup_seconds: float,
    default_dt: float,
    steady_fraction: float,
) -> None:
    filtered_rows = drop_warmup_rows(rows, warmup_seconds)
    elapsed_min, elapsed_max = elapsed_range(filtered_rows)
    metrics["warmup_excluded_seconds"] = max(0.0, warmup_seconds)
    metrics["warmup_excluded_profile_rows"] = len(filtered_rows)
    metrics["warmup_excluded_rows"] = len(filtered_rows)
    metrics["warmup_excluded_valid"] = int(bool(filtered_rows))
    metrics["warmup_excluded_start_elapsed_seconds"] = (
        filtered_rows[0].get("elapsed_seconds", "") if filtered_rows else "")
    metrics["warmup_excluded_elapsed_min"] = elapsed_min
    metrics["warmup_excluded_elapsed_max"] = elapsed_max
    metrics["metric_invalid"] = int(not bool(filtered_rows))
    metrics.setdefault("metric_warning", "")
    if not filtered_rows:
        add_metric_warning(metrics, "warmup_filter_empty")
        for key in comfort_metrics([], default_dt=default_dt):
            metrics["warmup_excluded_comfort_" + key] = ""
        for key in stability_metrics([], steady_fraction=steady_fraction):
            metrics["warmup_excluded_stability_" + key] = ""
        return
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
    all_rows, _episode_groups, actor_groups = read_profile_groups(profile_path)
    main_group, main_rows = select_main_profile_group(actor_groups)
    main_episode, main_actor = main_group
    actor_switch = len(actor_groups) > 1
    elapsed_reset = elapsed_reset_detected(all_rows)
    episode_rows: List[Dict[str, Any]] = []

    for (episode, actor), rows in actor_groups.items():
        metric_row: Dict[str, Any] = {
            "scenario": scenario["name"],
            "label": scenario["label"],
            "controller": scenario["controller"],
            "seed": seed,
            "episode_index": episode,
            "actor_id": actor,
            "profile_rows": len(rows),
            "start_frame": rows[0].get("frame", "") if rows else "",
            "end_frame": rows[-1].get("frame", "") if rows else "",
            "start_elapsed_seconds": rows[0].get("elapsed_seconds", "") if rows else "",
            "end_elapsed_seconds": rows[-1].get("elapsed_seconds", "") if rows else "",
            "metric_source": "profile_actor_episode_warmup_excluded",
            "metric_invalid": 0,
            "metric_warning": "",
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
        "metric_source",
        "metric_invalid",
        "metric_warning",
        "start_frame",
        "end_frame",
        "start_elapsed_seconds",
        "end_elapsed_seconds",
        "warmup_excluded_seconds",
        "warmup_excluded_profile_rows",
        "warmup_excluded_rows",
        "warmup_excluded_valid",
        "warmup_excluded_start_elapsed_seconds",
        "warmup_excluded_elapsed_min",
        "warmup_excluded_elapsed_max",
    )
    write_csv_rows(metrics_path, episode_rows, preferred)

    main_metric_row = next(
        (
            row for row in episode_rows
            if row.get("episode_index") == main_episode and row.get("actor_id") == main_actor
        ),
        max(episode_rows, key=lambda row: int(row.get("profile_rows") or 0))
        if episode_rows else {},
    )
    run_metrics: Dict[str, Any] = dict(main_metric_row)
    non_main_rows = max(0, len(all_rows) - len(main_rows))
    run_metrics.update({
        "profile_rows": len(main_rows),
        "profile_total_rows": len(all_rows),
        "profile_main_rows": len(main_rows),
        "profile_excluded_stale_rows": non_main_rows,
        "profile_excluded_non_main_rows": non_main_rows,
        "profile_main_episode_index": main_episode,
        "profile_main_actor_id": main_actor,
        "metric_main_episode_index": main_episode,
        "metric_main_actor_id": main_actor,
        "main_episode_index": main_episode,
        "main_actor_id": main_actor,
        "metric_actor_switch_detected": int(actor_switch),
        "metric_elapsed_reset_detected": int(elapsed_reset),
        "metric_source": "metrics_by_episode_main_actor" if main_rows else "",
        "raw_metric_source": "metrics_by_episode_main_actor_raw" if main_rows else "",
    })
    if actor_switch:
        add_metric_warning(run_metrics, "actor_switch_detected")
    if elapsed_reset:
        add_metric_warning(run_metrics, "elapsed_reset_detected")
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


PHASE2_INFRACTION_FIELDS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
    "scenario_timeouts",
    "min_speed_infractions",
)


PHASE2_BASELINE_BY_SCENARIO = {
    "S7_rl_zero_residual_pid": "S2_pid",
    "S8_rl_zero_residual_skyhook": "S3_skyhook",
}

PHASE2C_CANARY_BY_SCENARIO = {
    "S8_rl_zero_residual_skyhook": {
        "role": "zero_reference",
        "expected_mean_action": 0.0,
        "expected_abs_action": 0.0,
        "expect_nonzero_residual": False,
    },
    "S9_rl_const_plus_0p02_skyhook": {
        "role": "constant_positive_canary",
        "expected_mean_action": 0.02,
        "expected_abs_action": 0.02,
        "expect_nonzero_residual": True,
    },
    "S10_rl_const_minus_0p02_skyhook": {
        "role": "constant_negative_canary",
        "expected_mean_action": -0.02,
        "expected_abs_action": 0.02,
        "expect_nonzero_residual": True,
    },
    "S11_rl_random_small_skyhook": {
        "role": "random_small_canary",
        "expected_mean_action": "",
        "expected_abs_action": "",
        "expect_nonzero_residual": True,
    },
    "S12_rl_const_action_plus_0p25_skyhook": {
        "role": "constant_positive_canary",
        "expected_mean_action": 0.25,
        "expected_abs_action": 0.25,
        "expect_nonzero_residual": True,
    },
    "S13_rl_const_action_minus_0p25_skyhook": {
        "role": "constant_negative_canary",
        "expected_mean_action": -0.25,
        "expected_abs_action": 0.25,
        "expect_nonzero_residual": True,
    },
    "S14_rl_random_action_0p10_skyhook": {
        "role": "random_action_canary",
        "expected_mean_action": "",
        "expected_abs_action": "",
        "expect_nonzero_residual": True,
    },
}


def phase2_acceptance_rows(
    summary_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_seed_scenario: Dict[Tuple[Any, str], Mapping[str, Any]] = {
        (row.get("seed"), row.get("scenario")): row
        for row in summary_rows
    }
    rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        scenario_name = str(row.get("scenario", ""))
        controller = str(row.get("controller", ""))
        baseline_scenario = PHASE2_BASELINE_BY_SCENARIO.get(scenario_name, "")
        failed_checks: List[str] = []

        route_score_ok = _score_is_100(row.get("score_route"))
        if not route_score_ok:
            failed_checks.append("route_score")
        composed_score_ok = _score_is_100(row.get("score_composed"))
        if not composed_score_ok:
            failed_checks.append("score_composed")
        infraction_ok = _infractions_are_zero(row)
        if not infraction_ok:
            failed_checks.append("infractions")

        uses_api = _truthy(row.get("uses_suspension_api"))
        verification_ok = (
            not uses_api or
            _positive_number(row.get("sidecar_command_verifies")))
        if not verification_ok:
            failed_checks.append("verification")

        policy_available_ratio = safe_float(row.get("rl_policy_available_ratio"))
        policy_available_ok = ""
        fallback_ok = ""
        zero_action_ok = ""
        zero_residual_ok = ""
        pair_score_match_ok = ""
        paired_baseline_found = ""

        if controller.startswith("rl_zero_residual"):
            policy_available_ok = int(
                policy_available_ratio is not None and
                policy_available_ratio >= 0.99)
            if not policy_available_ok:
                failed_checks.append("policy_available_ratio")
            fallback_ok = int(_zero_number(row.get("rl_fallback_rows")))
            if not fallback_ok:
                failed_checks.append("fallback_rows")
            zero_action_ok = int(
                _zero_number(row.get("rl_mean_abs_action")) and
                _zero_number(row.get("rl_nonzero_action_rows")))
            if not zero_action_ok:
                failed_checks.append("rl_action")
            zero_residual_ok = int(
                _zero_number(row.get("rl_mean_abs_residual_damper")) and
                _zero_number(row.get("rl_nonzero_residual_rows")))
            if not zero_residual_ok:
                failed_checks.append("rl_residual")

            baseline = by_seed_scenario.get((row.get("seed"), baseline_scenario))
            paired_baseline_found = int(baseline is not None)
            if baseline is None:
                pair_score_match_ok = "not_applicable"
            else:
                score_delta = _delta(row.get("score_route"), baseline.get("score_route"))
                penalty_delta = _delta(row.get("score_penalty"), baseline.get("score_penalty"))
                pair_score_match_ok = int(
                    score_delta == 0.0 and penalty_delta == 0.0)
                if not pair_score_match_ok:
                    failed_checks.append("paired_score")
        else:
            baseline = None

        acceptance = {
            "seed": row.get("seed", ""),
            "scenario": scenario_name,
            "label": row.get("label", ""),
            "controller": controller,
            "phase2_role": (
                "rl_zero_residual"
                if controller.startswith("rl_zero_residual")
                else "route_or_api_baseline"),
            "phase2_status": "pass" if not failed_checks else "fail",
            "failed_checks": ";".join(failed_checks),
            "baseline_scenario": baseline_scenario,
            "route_score_ok": int(route_score_ok),
            "score_composed_ok": int(composed_score_ok),
            "infraction_ok": int(infraction_ok),
            "verification_ok": int(verification_ok),
            "sidecar_command_verifies": row.get("sidecar_command_verifies", ""),
            "policy_available_ratio": (
                policy_available_ratio if policy_available_ratio is not None else ""),
            "policy_available_ok": policy_available_ok,
            "rl_fallback_rows": row.get("rl_fallback_rows", ""),
            "fallback_ok": fallback_ok,
            "rl_mean_abs_action": row.get("rl_mean_abs_action", ""),
            "zero_action_ok": zero_action_ok,
            "rl_mean_abs_residual_damper": row.get(
                "rl_mean_abs_residual_damper",
                ""),
            "low_speed_mask_ratio": row.get("low_speed_mask_ratio", ""),
            "hard_safety_gate_ratio": row.get("hard_safety_gate_ratio", ""),
            "soft_safety_gain_ratio": row.get("soft_safety_gain_ratio", ""),
            "effective_control_ratio": row.get("effective_control_ratio", ""),
            "safety_gate_active_ratio": row.get("safety_gate_active_ratio", ""),
            "safety_gate_active_rows": row.get("safety_gate_active_rows", ""),
            "safety_gate_reason_counts": row.get("safety_gate_reason_counts", ""),
            "safety_gate_active_speed_min": row.get(
                "safety_gate_active_speed_min",
                ""),
            "safety_gate_active_speed_mean": row.get(
                "safety_gate_active_speed_mean",
                ""),
            "safety_gate_active_speed_max": row.get(
                "safety_gate_active_speed_max",
                ""),
            "safety_gate_active_roll_min": row.get(
                "safety_gate_active_roll_min",
                ""),
            "safety_gate_active_roll_mean": row.get(
                "safety_gate_active_roll_mean",
                ""),
            "safety_gate_active_roll_max": row.get(
                "safety_gate_active_roll_max",
                ""),
            "safety_gate_active_lateral_acc_min": row.get(
                "safety_gate_active_lateral_acc_min",
                ""),
            "safety_gate_active_lateral_acc_mean": row.get(
                "safety_gate_active_lateral_acc_mean",
                ""),
            "safety_gate_active_lateral_acc_max": row.get(
                "safety_gate_active_lateral_acc_max",
                ""),
            "safety_gate_active_yaw_rate_min": row.get(
                "safety_gate_active_yaw_rate_min",
                ""),
            "safety_gate_active_yaw_rate_mean": row.get(
                "safety_gate_active_yaw_rate_mean",
                ""),
            "safety_gate_active_yaw_rate_max": row.get(
                "safety_gate_active_yaw_rate_max",
                ""),
            "zero_residual_ok": zero_residual_ok,
            "paired_baseline_found": paired_baseline_found,
            "pair_score_match_ok": pair_score_match_ok,
        }
        if baseline is not None:
            acceptance.update({
                "score_route_delta_vs_baseline": _delta(
                    row.get("score_route"),
                    baseline.get("score_route")),
                "score_composed_delta_vs_baseline": _delta(
                    row.get("score_composed"),
                    baseline.get("score_composed")),
                "score_penalty_delta_vs_baseline": _delta(
                    row.get("score_penalty"),
                    baseline.get("score_penalty")),
                "warmup_excluded_comfort_score_delta_vs_baseline": _delta(
                    row.get("warmup_excluded_comfort_comfort_score"),
                    baseline.get("warmup_excluded_comfort_comfort_score")),
                "warmup_excluded_rms_roll_delta_vs_baseline": _delta(
                    row.get("warmup_excluded_stability_rms_roll"),
                    baseline.get("warmup_excluded_stability_rms_roll")),
            })
        rows.append(acceptance)
    return rows


def phase2c_acceptance_rows(
    summary_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in summary_rows:
        scenario_name = str(row.get("scenario", ""))
        expectation = PHASE2C_CANARY_BY_SCENARIO.get(scenario_name)
        if not expectation:
            continue

        failed_checks: List[str] = []
        route_score_ok = _score_is_100(row.get("score_route"))
        if not route_score_ok:
            failed_checks.append("route_score")
        composed_score_ok = _score_is_100(row.get("score_composed"))
        if not composed_score_ok:
            failed_checks.append("score_composed")
        infraction_ok = _infractions_are_zero(row)
        if not infraction_ok:
            failed_checks.append("infractions")
        verification_ok = _positive_number(row.get("sidecar_command_verifies"))
        if not verification_ok:
            failed_checks.append("verification")

        policy_available_ratio = safe_float(row.get("rl_policy_available_ratio"))
        policy_available_ok = int(
            policy_available_ratio is not None and
            policy_available_ratio >= 0.99)
        if not policy_available_ok:
            failed_checks.append("policy_available_ratio")

        fallback_reason_ok = int(_fallback_reasons_are_allowed(
            row.get("rl_fallback_reasons", ""),
            ("safety_gate_zero",)))
        if not fallback_reason_ok:
            failed_checks.append("fallback_reason")

        expected_mean_action = expectation["expected_mean_action"]
        expected_abs_action = expectation["expected_abs_action"]
        mean_action = safe_float(row.get("rl_mean_action"))
        mean_abs_action = safe_float(row.get("rl_mean_abs_action"))
        diagnostic_rows = safe_float(row.get("rl_diagnostic_rows")) or 0.0
        nonzero_action_rows = safe_float(row.get("rl_nonzero_action_rows")) or 0.0
        nonzero_residual_rows = safe_float(row.get("rl_nonzero_residual_rows")) or 0.0
        nonzero_action_ratio = (
            nonzero_action_rows / diagnostic_rows
            if diagnostic_rows > 0.0 else "")

        if expected_mean_action == "":
            action_path_ok = int(
                mean_abs_action is not None and
                mean_abs_action > 1.0e-3 and
                nonzero_action_rows > 0.0)
        else:
            action_path_ok = int(
                _near_number(mean_action, float(expected_mean_action), 1.0e-6) and
                _near_number(mean_abs_action, float(expected_abs_action), 1.0e-6))
        if not action_path_ok:
            failed_checks.append("action_path")

        mean_abs_residual = safe_float(row.get("rl_mean_abs_residual_damper"))
        if expectation["expect_nonzero_residual"]:
            residual_path_ok = int(
                mean_abs_residual is not None and
                mean_abs_residual > 1.0e-9 and
                nonzero_residual_rows > 0.0)
        else:
            residual_path_ok = int(
                _zero_number(row.get("rl_nonzero_residual_rows")) and
                _zero_number(row.get("rl_mean_abs_residual_damper")))
        if not residual_path_ok:
            failed_checks.append("residual_path")

        rows.append({
            "seed": row.get("seed", ""),
            "scenario": scenario_name,
            "label": row.get("label", ""),
            "controller": row.get("controller", ""),
            "phase2c_role": expectation["role"],
            "phase2c_status": "pass" if not failed_checks else "fail",
            "failed_checks": ";".join(failed_checks),
            "route_score_ok": int(route_score_ok),
            "score_composed_ok": int(composed_score_ok),
            "infraction_ok": int(infraction_ok),
            "verification_ok": int(verification_ok),
            "policy_available_ok": policy_available_ok,
            "fallback_reason_ok": fallback_reason_ok,
            "action_path_ok": action_path_ok,
            "residual_path_ok": residual_path_ok,
            "expected_mean_action": expected_mean_action,
            "expected_abs_action": expected_abs_action,
            "rl_diagnostic_rows": row.get("rl_diagnostic_rows", ""),
            "rl_policy_available_ratio": row.get(
                "rl_policy_available_ratio",
                ""),
            "rl_fallback_rows": row.get("rl_fallback_rows", ""),
            "rl_fallback_reasons": row.get("rl_fallback_reasons", ""),
            "rl_nonzero_action_rows": row.get("rl_nonzero_action_rows", ""),
            "rl_nonzero_action_ratio": nonzero_action_ratio,
            "rl_mean_action": row.get("rl_mean_action", ""),
            "rl_mean_abs_action": row.get("rl_mean_abs_action", ""),
            "rl_nonzero_residual_rows": row.get(
                "rl_nonzero_residual_rows",
                ""),
            "rl_mean_residual_damper": row.get(
                "rl_mean_residual_damper",
                ""),
            "rl_mean_abs_residual_damper": row.get(
                "rl_mean_abs_residual_damper",
                ""),
            "sidecar_command_verifies": row.get("sidecar_command_verifies", ""),
            "low_speed_mask_ratio": row.get("low_speed_mask_ratio", ""),
            "hard_safety_gate_ratio": row.get("hard_safety_gate_ratio", ""),
            "soft_safety_gain_ratio": row.get("soft_safety_gain_ratio", ""),
            "effective_control_ratio": row.get("effective_control_ratio", ""),
            "safety_gate_active_ratio": row.get("safety_gate_active_ratio", ""),
            "safety_gate_active_rows": row.get("safety_gate_active_rows", ""),
            "safety_gate_reason_counts": row.get(
                "safety_gate_reason_counts",
                ""),
        })
    return rows


def write_phase2_acceptance_summary(
    output_dir: str,
    summary_rows: Sequence[Mapping[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    rows = phase2_acceptance_rows(summary_rows)
    path = os.path.join(output_dir, "phase2_online_dummy_acceptance.csv")
    write_csv_rows(
        path,
        rows,
        (
            "seed",
            "scenario",
            "label",
            "controller",
            "phase2_role",
            "phase2_status",
            "failed_checks",
            "baseline_scenario",
            "route_score_ok",
            "score_composed_ok",
            "infraction_ok",
            "verification_ok",
            "sidecar_command_verifies",
            "policy_available_ratio",
            "policy_available_ok",
            "rl_fallback_rows",
            "fallback_ok",
            "rl_mean_abs_action",
            "zero_action_ok",
            "rl_mean_abs_residual_damper",
            "low_speed_mask_ratio",
            "hard_safety_gate_ratio",
            "soft_safety_gain_ratio",
            "effective_control_ratio",
            "safety_gate_active_ratio",
            "safety_gate_active_rows",
            "safety_gate_reason_counts",
            "safety_gate_active_speed_min",
            "safety_gate_active_speed_mean",
            "safety_gate_active_speed_max",
            "safety_gate_active_roll_min",
            "safety_gate_active_roll_mean",
            "safety_gate_active_roll_max",
            "safety_gate_active_lateral_acc_min",
            "safety_gate_active_lateral_acc_mean",
            "safety_gate_active_lateral_acc_max",
            "safety_gate_active_yaw_rate_min",
            "safety_gate_active_yaw_rate_mean",
            "safety_gate_active_yaw_rate_max",
            "zero_residual_ok",
            "paired_baseline_found",
            "pair_score_match_ok",
        ))
    return path, rows


def write_phase2c_acceptance_summary(
    output_dir: str,
    summary_rows: Sequence[Mapping[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    rows = phase2c_acceptance_rows(summary_rows)
    path = os.path.join(output_dir, "phase2c_nonzero_action_acceptance.csv")
    write_csv_rows(
        path,
        rows,
        (
            "seed",
            "scenario",
            "label",
            "controller",
            "phase2c_role",
            "phase2c_status",
            "failed_checks",
            "route_score_ok",
            "score_composed_ok",
            "infraction_ok",
            "verification_ok",
            "policy_available_ok",
            "fallback_reason_ok",
            "action_path_ok",
            "residual_path_ok",
            "expected_mean_action",
            "expected_abs_action",
            "rl_diagnostic_rows",
            "rl_policy_available_ratio",
            "rl_fallback_rows",
            "rl_fallback_reasons",
            "rl_nonzero_action_rows",
            "rl_nonzero_action_ratio",
            "rl_mean_action",
            "rl_mean_abs_action",
            "rl_nonzero_residual_rows",
            "rl_mean_residual_damper",
            "rl_mean_abs_residual_damper",
            "sidecar_command_verifies",
            "low_speed_mask_ratio",
            "hard_safety_gate_ratio",
            "soft_safety_gain_ratio",
            "effective_control_ratio",
            "safety_gate_active_ratio",
            "safety_gate_active_rows",
            "safety_gate_reason_counts",
        ))
    return path, rows


def _score_is_100(value: Any) -> bool:
    number = safe_float(value)
    return number is not None and abs(number - 100.0) <= 1.0e-9


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("", None):
        return False
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "none", "no")
    return bool(value)


def _positive_number(value: Any) -> bool:
    number = safe_float(value)
    return number is not None and number > 0.0


def _negative_number(value: Any) -> bool:
    number = safe_float(value)
    return number is not None and number < -1.0e-9


def _raw_negative_progress(row: Mapping[str, Any]) -> bool:
    return (
        _positive_number(row.get("route_progress_negative_raw")) or
        _negative_number(row.get("route_progress_raw_negative_delta_m")) or
        _negative_number(row.get("route_delta_progress_m")))


def _zero_number(value: Any) -> bool:
    number = safe_float(value)
    return number is not None and abs(number) <= 1.0e-12


def _near_number(value: Any, expected: float, tolerance: float) -> bool:
    number = safe_float(value)
    return (
        number is not None and
        abs(number - expected) <= abs(float(tolerance)))


def _fallback_reasons_are_allowed(
    value: Any,
    allowed_reasons: Sequence[str],
) -> bool:
    reasons = [
        item
        for item in str(value or "").split(";")
        if item
    ]
    allowed = set(allowed_reasons)
    return all(reason in allowed for reason in reasons)


def _infractions_are_zero(row: Mapping[str, Any]) -> bool:
    for field in PHASE2_INFRACTION_FIELDS:
        value = row.get(field, "")
        if value in ("", None, "[]", "{}", "0", "0.0"):
            continue
        number = safe_float(value)
        if number is not None and abs(number) <= 1.0e-12:
            continue
        return False
    return True


def _delta(left: Any, right: Any) -> Any:
    left_value = safe_float(left)
    right_value = safe_float(right)
    if left_value is None or right_value is None:
        return ""
    return left_value - right_value


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
    diagnostic_summary = summarize_diagnostics_csv(diagnostics_path)
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
    row.update(diagnostic_summary)
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
        "profile_total_rows",
        "profile_main_rows",
        "profile_excluded_stale_rows",
        "profile_excluded_non_main_rows",
        "profile_main_episode_index",
        "profile_main_actor_id",
        "metric_main_episode_index",
        "metric_main_actor_id",
        "main_episode_index",
        "main_actor_id",
        "metric_actor_switch_detected",
        "metric_elapsed_reset_detected",
        "metric_invalid",
        "metric_warning",
        "metric_source",
        "raw_metric_source",
        "rl_diagnostic_rows",
        "rl_policy_available_rows",
        "rl_policy_available_ratio",
        "rl_fallback_rows",
        "rl_fallback_reasons",
        "rl_zero_action_rows",
        "rl_nonzero_action_rows",
        "rl_zero_residual_rows",
        "rl_nonzero_residual_rows",
        "rl_mean_safety_gain",
        "rl_residual_modes",
        "rl_action_scale_mean",
        "rl_residual_gain_mean",
        "rl_scripted_residual_kinds",
        "rl_mean_action",
        "rl_mean_abs_action",
        "rl_mean_raw_residual_damper",
        "rl_mean_abs_raw_residual_damper",
        "rl_mean_scaled_residual_damper",
        "rl_mean_abs_scaled_residual_damper",
        "rl_mean_scale_clipped_residual_damper",
        "rl_mean_abs_scale_clipped_residual_damper",
        "rl_mean_residual_damper",
        "rl_mean_abs_residual_damper",
        "rl_mean_final_residual_damper",
        "rl_mean_abs_final_residual_damper",
        "residual_scale_clip_rows",
        "residual_scale_clip_ratio",
        "damper_final_clamp_rows",
        "damper_final_clamp_ratio",
        "residual_saturation_rows",
        "residual_saturation_ratio",
        "rl_observation_clip_count_max",
        "low_speed_mask_ratio",
        "low_speed_mask_rows",
        "hard_safety_gate_ratio",
        "hard_safety_gate_rows",
        "soft_safety_gain_ratio",
        "soft_safety_gain_rows",
        "effective_control_ratio",
        "safety_gate_active_ratio",
        "safety_gate_active_rows",
        "safety_gate_reason_counts",
        "safety_gate_active_speed_min",
        "safety_gate_active_speed_mean",
        "safety_gate_active_speed_max",
        "safety_gate_active_roll_min",
        "safety_gate_active_roll_mean",
        "safety_gate_active_roll_max",
        "safety_gate_active_abs_roll_mean",
        "safety_gate_active_abs_roll_max",
        "safety_gate_active_lateral_acc_min",
        "safety_gate_active_lateral_acc_mean",
        "safety_gate_active_lateral_acc_max",
        "safety_gate_active_abs_lateral_acc_mean",
        "safety_gate_active_abs_lateral_acc_max",
        "safety_gate_active_yaw_rate_min",
        "safety_gate_active_yaw_rate_mean",
        "safety_gate_active_yaw_rate_max",
        "safety_gate_active_abs_yaw_rate_mean",
        "safety_gate_active_abs_yaw_rate_max",
        "planning_available_rows",
        "planning_sources",
        "policy_available_ratio",
        "fallback_ratio",
        "observation_clip_ratio",
        "mean_abs_action",
        "raw_mean_abs_action",
        "raw_mean_abs_residual_damper",
        "scaled_mean_abs_residual_damper",
        "final_mean_abs_residual_damper",
        "mean_abs_residual_damper",
        "mean_reward_total",
        "mean_reward_comfort",
        "mean_reward_stability",
        "mean_reward_task",
        "mean_reward_action",
        "mean_reward_safety",
        "reward_rows",
        "reward_row_ratio",
        "mean_reward_cost_comfort",
        "mean_reward_cost_stability",
        "mean_reward_cost_task",
        "mean_reward_cost_action",
        "mean_reward_cost_safety",
        "mean_reward_term_action_mag",
        "mean_reward_term_action_rate",
        "mean_reward_term_damper_rate",
        "mean_reward_term_baseline_dev",
        "mean_reward_term_low_speed_not_planned",
        "mean_reward_term_progress_stall",
        "mean_reward_term_route_deviation",
        "mean_reward_term_abs_speed_error",
        "mean_reward_route_deviation_used_ratio",
        "reward_route_deviation_used_rows",
        "reward_route_deviation_enabled_rows",
        "reward_route_deviation_valid_rows",
        "reward_task_capped_without_infraction_rows",
        "reward_task_capped_without_infraction_ratio",
        "route_progress_available_rows",
        "route_progress_available_ratio",
        "route_progress_fraction_max",
        "route_progress_monotonic_fraction_max",
        "route_completion_proxy",
        "route_progress_delta_negative_raw_rows",
        "route_progress_delta_negative_raw_ratio",
        "route_progress_raw_negative_rows",
        "route_progress_raw_negative_ratio",
        "route_progress_monotonic_negative_rows",
        "route_progress_monotonic_negative_ratio",
        "route_deviation_m_mean",
        "route_deviation_m_max",
        "route_deviation_valid_rows",
        "route_deviation_valid_ratio",
        "planned_stop_rows",
        "planned_stop_ratio",
        "low_speed_not_planned_rows",
        "low_speed_not_planned_ratio",
        "progress_stall_rows",
        "progress_stall_ratio",
        "route_progress_stall_rows",
        "route_progress_stall_ratio",
        "progress_stall_count_max",
        "negative_progress_rows",
        "negative_progress_ratio",
        "mean_route_progress_rate_mps",
        "mean_abs_speed_error",
        "collision_count",
        "lane_invasion_count",
        "red_light_count",
        "blocked_vehicle_count",
        "route_timeout_count",
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
        "warmup_excluded_rows",
        "warmup_excluded_valid",
        "warmup_excluded_start_elapsed_seconds",
        "warmup_excluded_elapsed_min",
        "warmup_excluded_elapsed_max",
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
        "planning_provider": args.planning_provider,
        "planning_preview_jsonl": (
            expand_path(args.planning_preview_jsonl)
            if args.planning_preview_jsonl else ""),
        "planning_max_frame_lag": args.planning_max_frame_lag,
        "planning_horizon_dt": args.planning_horizon_dt,
        "reward_nominal_target_speed": args.reward_nominal_target_speed,
        "reward_low_speed_threshold_mps": args.reward_low_speed_threshold_mps,
        "reward_min_progress_rate_mps": args.reward_min_progress_rate_mps,
        "reward_planned_stop_brake_threshold": (
            args.reward_planned_stop_brake_threshold),
        "reward_progress_stall_steps": args.reward_progress_stall_steps,
        "reward_progress_stall_epsilon_m": args.reward_progress_stall_epsilon_m,
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
    phase2_path, _ = write_phase2_acceptance_summary(output_dir, rows)
    phase2c_path, phase2c_rows = write_phase2c_acceptance_summary(
        output_dir,
        rows)
    phase3_csv_path, phase3_json_path, phase3_rows = (
        write_phase3_reward_sanity_report(output_dir, rows))
    phase3b_csv_path, phase3b_json_path, phase3b_rows = (
        write_phase3b_reward_calibration_report(output_dir, rows))

    print("")
    print("Wrote suite outputs:")
    print("  %s" % csv_path)
    print("  %s" % json_path)
    print("  %s" % comparison_path)
    print("  %s" % phase2_path)
    if phase2c_rows:
        print("  %s" % phase2c_path)
    if phase3_rows:
        print("  %s" % phase3_csv_path)
        print("  %s" % phase3_json_path)
    if phase3b_rows:
        print("  %s" % phase3b_csv_path)
        print("  %s" % phase3b_json_path)


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
        help="comma-separated scenarios: stock,identity,pid,skyhook,"
        "rl_residual_skyhook,rl_residual_pid,"
        "rl_residual_skyhook_no_planning,"
        "rl_zero_residual_pid,rl_zero_residual_skyhook,"
        "rl_const_plus_0p02_skyhook,"
        "rl_const_minus_0p02_skyhook,"
        "rl_random_small_skyhook,"
        "rl_const_action_plus_0p25_skyhook,"
        "rl_const_action_minus_0p25_skyhook,"
        "rl_random_action_0p10_skyhook "
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
        "--rl-residual-config",
        default=DEFAULT_RL_RESIDUAL_CONFIG,
        help="flat YAML residual-RL config path")
    parser.add_argument(
        "--rl-policy",
        default="",
        help="optional residual-RL policy path override")
    parser.add_argument(
        "--rl-normalizer",
        default="",
        help="optional residual-RL normalizer JSON path override")
    parser.add_argument(
        "--planning-provider",
        choices=("empty", "jsonl", "control_history"),
        default="empty",
        help="planning preview source for controller context (default: empty)")
    parser.add_argument(
        "--planning-preview-jsonl",
        default="",
        help="JSONL path used when --planning-provider=jsonl")
    parser.add_argument(
        "--planning-max-frame-lag",
        default=5,
        type=int,
        help="maximum accepted stale planning frame lag (default: 5)")
    parser.add_argument(
        "--planning-horizon-dt",
        default=0.1,
        type=float,
        help="fallback planning preview horizon dt in seconds (default: 0.1)")
    parser.add_argument(
        "--reward-nominal-target-speed",
        default=8.0,
        type=float,
        help="nominal target speed used when planning preview is empty (default: 8.0)")
    parser.add_argument(
        "--reward-low-speed-threshold-mps",
        default=0.5,
        type=float,
        help="speed threshold for low-speed-not-planned task diagnostics (default: 0.5)")
    parser.add_argument(
        "--reward-min-progress-rate-mps",
        default=0.5,
        type=float,
        help="minimum progress rate before low-speed/stall diagnostics (default: 0.5)")
    parser.add_argument(
        "--reward-planned-stop-brake-threshold",
        default=0.25,
        type=float,
        help="brake threshold that suppresses low-speed-not-planned (default: 0.25)")
    parser.add_argument(
        "--reward-progress-stall-steps",
        default=10,
        type=int,
        help="consecutive flat-progress steps before progress_stall=1 (default: 10)")
    parser.add_argument(
        "--reward-progress-stall-epsilon-m",
        default=0.1,
        type=float,
        help="progress delta tolerance for progress_stall detection in meters (default: 0.1)")
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
        default=3.0,
        type=float,
        help="initial episode seconds excluded from warmup_excluded metrics "
        "(default: 3.0; 0 disables exclusion)")
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

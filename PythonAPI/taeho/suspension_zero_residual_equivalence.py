#!/usr/bin/env python

"""Offline zero-residual equivalence replay for suspension controllers.

The script reads an existing sidecar ``profile.csv`` and replays the exact same
VehicleState sequence through a classical baseline and the residual-RL wrapper
with a deterministic zero policy. It does not connect to CARLA.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from suspension_control.controllers.base import (  # noqa: E402
    ControllerContext,
    PlanningInfo,
    VehicleState,
)
from suspension_control.controllers.pid import (  # noqa: E402
    FeedbackPIDConfig,
    FeedbackPIDController,
)
from suspension_control.controllers.rl_residual import (  # noqa: E402
    ResidualRLConfig,
    ResidualRLController,
)
from suspension_control.controllers.skyhook import (  # noqa: E402
    SkyhookConfig,
    SkyhookController,
)


DEFAULT_PID_CONFIG = os.path.join(
    SCRIPT_DIR, "suspension_control", "configs", "pid.yaml")
DEFAULT_SKYHOOK_CONFIG = os.path.join(
    SCRIPT_DIR, "suspension_control", "configs", "skyhook.yaml")
DEFAULT_RL_RESIDUAL_CONFIG = os.path.join(
    SCRIPT_DIR, "suspension_control", "configs", "rl_residual.yaml")

WHEEL_LABELS = ("fl", "fr", "rl", "rr")
SUMMARY_FIELDS = (
    "baseline",
    "status",
    "profile_csv",
    "detail_csv",
    "profile_rows",
    "episodes",
    "max_abs_damper_diff",
    "max_abs_spring_diff",
    "max_rl_mean_abs_action",
    "max_rl_mean_abs_residual_damper",
    "damper_tolerance",
    "spring_tolerance",
    "action_tolerance",
    "residual_tolerance",
    "first_failure_episode",
    "first_failure_frame",
    "first_failure_step",
    "first_failure_reason",
)
DETAIL_FIELDS = (
    "baseline",
    "episode_index",
    "profile_index",
    "step",
    "frame",
    "elapsed_seconds",
    "max_abs_damper_diff",
    "max_abs_spring_diff",
    "rl_mean_abs_action",
    "rl_mean_abs_residual_damper",
    "rl_fallback_reason",
    "passed",
)


class ZeroPolicy:
    """PolicyAdapter-compatible deterministic zero policy."""

    is_available = True
    status = "zero"

    def __init__(self, wheel_count: int):
        self.wheel_count = int(wheel_count)

    def predict(self, observation: Sequence[float], deterministic: bool = True):
        del observation, deterministic
        return [0.0 for _ in range(self.wheel_count)]


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
    if not os.path.isfile(path):
        raise RuntimeError("config path does not exist: %s" % path)
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


def build_skyhook_config(path: str) -> SkyhookConfig:
    return SkyhookConfig.from_mapping(read_flat_yaml(path)) if path else SkyhookConfig()


def build_pid_config(path: str) -> FeedbackPIDConfig:
    return FeedbackPIDConfig.from_mapping(read_flat_yaml(path)) if path else FeedbackPIDConfig()


def build_rl_residual_config(
    path: str,
    baseline: str,
    wheel_count: int,
    default_dt: float,
) -> ResidualRLConfig:
    values = read_flat_yaml(path) if path else {}
    values.update({
        "baseline": baseline,
        "wheel_count": wheel_count,
        "default_dt": default_dt,
        "allow_untrained_policy": True,
        "policy_path": "",
        "normalizer_path": "",
        "deterministic_policy": True,
    })
    return ResidualRLConfig.from_mapping(values)


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_int(value: Any, default: int = 0) -> int:
    number = safe_float(value)
    return int(number) if number is not None else default


def resolve_profile_path(path: str) -> str:
    profile_path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(profile_path):
        profile_path = os.path.join(profile_path, "profile.csv")
    if not os.path.isfile(profile_path):
        raise RuntimeError("profile.csv not found: %s" % profile_path)
    return profile_path


def read_profile_rows(path: str) -> "OrderedDict[str, List[Dict[str, Any]]]":
    profile_path = resolve_profile_path(path)
    groups: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    with open(profile_path) as csv_file:
        for index, row in enumerate(csv.DictReader(csv_file)):
            row["_profile_index"] = index
            episode = row.get("episode_index", "")
            groups.setdefault(str(episode), []).append(row)
    if not groups:
        raise RuntimeError("profile has no rows: %s" % profile_path)
    return groups


def state_from_profile_row(
    row: Mapping[str, Any],
    previous_state: Optional[VehicleState],
    default_dt: float,
) -> VehicleState:
    elapsed = safe_float(row.get("elapsed_seconds"), 0.0) or 0.0
    dt = safe_float(row.get("dt"))
    if dt is None or dt <= 0.0:
        if previous_state is not None:
            dt = max(0.0, elapsed - previous_state.elapsed_seconds)
        else:
            dt = max(0.0, float(default_dt))
    speed = safe_float(row.get("speed"), 0.0) or 0.0
    return VehicleState(
        step=safe_int(row.get("step"), safe_int(row.get("_profile_index"), 0)),
        frame=safe_int(row.get("frame"), 0),
        elapsed_seconds=elapsed,
        dt=dt,
        x=safe_float(row.get("x"), 0.0) or 0.0,
        y=safe_float(row.get("y"), 0.0) or 0.0,
        z=safe_float(row.get("z"), 0.0) or 0.0,
        vx=safe_float(row.get("vx"), 0.0) or 0.0,
        vy=safe_float(row.get("vy"), 0.0) or 0.0,
        vz=safe_float(row.get("vz"), 0.0) or 0.0,
        speed=speed,
        local_vx=safe_float(row.get("local_vx"), speed) or 0.0,
        local_vy=safe_float(row.get("local_vy"), 0.0) or 0.0,
        ax=safe_float(row.get("ax"), 0.0) or 0.0,
        ay=safe_float(row.get("ay"), 0.0) or 0.0,
        az=safe_float(row.get("az"), 0.0) or 0.0,
        local_ax=safe_float(row.get("local_ax"), safe_float(row.get("ax"), 0.0)) or 0.0,
        local_ay=safe_float(row.get("local_ay"), safe_float(row.get("ay"), 0.0)) or 0.0,
        roll=safe_float(row.get("roll"), 0.0) or 0.0,
        pitch=safe_float(row.get("pitch"), 0.0) or 0.0,
        yaw=safe_float(row.get("yaw"), 0.0) or 0.0,
        roll_rate=safe_float(
            row.get("roll_rate"),
            safe_float(row.get("angular_velocity_x"), 0.0)) or 0.0,
        pitch_rate=safe_float(
            row.get("pitch_rate"),
            safe_float(row.get("angular_velocity_y"), 0.0)) or 0.0,
        yaw_rate=safe_float(
            row.get("yaw_rate"),
            safe_float(row.get("angular_velocity_z"), 0.0)) or 0.0,
        throttle=safe_float(row.get("throttle"), 0.0) or 0.0,
        brake=safe_float(row.get("brake"), 0.0) or 0.0,
        steer=safe_float(row.get("steer"), 0.0) or 0.0,
    )


def command_values(command: Any, field: str) -> Tuple[float, ...]:
    return tuple(float(getattr(wheel, field)) for wheel in command.wheels)


def max_abs_diff(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise RuntimeError("command wheel count mismatch: %d vs %d" % (
            len(left),
            len(right)))
    if not left:
        return 0.0
    return max(abs(a - b) for a, b in zip(left, right))


def make_baseline_controller(name: str, args: argparse.Namespace):
    if name == "skyhook":
        return SkyhookController(build_skyhook_config(args.skyhook_config))
    if name == "pid":
        return FeedbackPIDController(build_pid_config(args.pid_config))
    raise ValueError("unsupported baseline %s" % name)


def make_residual_controller(
    name: str,
    args: argparse.Namespace,
    wheel_count: int,
):
    cfg = build_rl_residual_config(
        args.rl_residual_config,
        baseline=name,
        wheel_count=wheel_count,
        default_dt=args.default_dt)
    return ResidualRLController(cfg, policy=ZeroPolicy(wheel_count))


def replay_equivalence_for_baseline(
    baseline_name: str,
    groups: "OrderedDict[str, List[Dict[str, Any]]]",
    args: argparse.Namespace,
    output_dir: str,
    profile_path: str,
) -> Dict[str, Any]:
    baseline_controller = make_baseline_controller(baseline_name, args)
    wheel_count = int(getattr(baseline_controller.config, "wheel_count", 4))
    residual_controller = make_residual_controller(baseline_name, args, wheel_count)

    detail_rows: List[Dict[str, Any]] = []
    max_damper_diff = 0.0
    max_spring_diff = 0.0
    max_action = 0.0
    max_residual = 0.0
    first_failure: Optional[Dict[str, Any]] = None

    for episode, rows in groups.items():
        baseline_controller.reset()
        residual_controller.reset()
        previous_state = None
        for row in rows:
            state = state_from_profile_row(row, previous_state, args.default_dt)
            context = ControllerContext(
                state=state,
                previous_state=previous_state,
                planning=PlanningInfo.empty(),
                step=state.step,
                dt=state.dt if state.dt > 0.0 else args.default_dt)
            baseline_output = baseline_controller.compute(context)
            residual_output = residual_controller.compute(context)

            baseline_command = baseline_output.command.validate(
                expected_wheels=wheel_count)
            residual_command = residual_output.command.validate(
                expected_wheels=wheel_count)
            damper_diff = max_abs_diff(
                command_values(baseline_command, "damper_scale"),
                command_values(residual_command, "damper_scale"))
            spring_diff = max_abs_diff(
                command_values(baseline_command, "spring_scale"),
                command_values(residual_command, "spring_scale"))
            diagnostics = dict(residual_output.diagnostics or {})
            mean_action = safe_float(diagnostics.get("rl_mean_abs_action"), 0.0) or 0.0
            mean_residual = safe_float(
                diagnostics.get("rl_mean_abs_residual_damper"),
                0.0) or 0.0

            max_damper_diff = max(max_damper_diff, damper_diff)
            max_spring_diff = max(max_spring_diff, spring_diff)
            max_action = max(max_action, abs(mean_action))
            max_residual = max(max_residual, abs(mean_residual))

            passed = (
                damper_diff <= args.damper_tolerance and
                spring_diff <= args.spring_tolerance and
                abs(mean_action) <= args.action_tolerance and
                abs(mean_residual) <= args.residual_tolerance)
            detail = {
                "baseline": baseline_name,
                "episode_index": episode,
                "profile_index": row.get("_profile_index", ""),
                "step": state.step,
                "frame": state.frame,
                "elapsed_seconds": state.elapsed_seconds,
                "max_abs_damper_diff": damper_diff,
                "max_abs_spring_diff": spring_diff,
                "rl_mean_abs_action": mean_action,
                "rl_mean_abs_residual_damper": mean_residual,
                "rl_fallback_reason": diagnostics.get("rl_fallback_reason", ""),
                "passed": int(passed),
            }
            for label, value in zip(
                    WHEEL_LABELS,
                    command_values(baseline_command, "damper_scale")):
                detail["baseline_damper_%s" % label] = value
            for label, value in zip(
                    WHEEL_LABELS,
                    command_values(residual_command, "damper_scale")):
                detail["residual_damper_%s" % label] = value
            for label, value in zip(
                    WHEEL_LABELS,
                    command_values(baseline_command, "spring_scale")):
                detail["baseline_spring_%s" % label] = value
            for label, value in zip(
                    WHEEL_LABELS,
                    command_values(residual_command, "spring_scale")):
                detail["residual_spring_%s" % label] = value
            detail_rows.append(detail)

            if not passed and first_failure is None:
                reasons = []
                if damper_diff > args.damper_tolerance:
                    reasons.append("damper_diff")
                if spring_diff > args.spring_tolerance:
                    reasons.append("spring_diff")
                if abs(mean_action) > args.action_tolerance:
                    reasons.append("rl_action")
                if abs(mean_residual) > args.residual_tolerance:
                    reasons.append("rl_residual")
                first_failure = dict(detail)
                first_failure["reason"] = ",".join(reasons)
            previous_state = state

    detail_path = os.path.join(
        output_dir,
        "zero_residual_equivalence_%s.csv" % baseline_name)
    write_csv_rows(detail_path, detail_rows, DETAIL_FIELDS)

    status = "pass" if first_failure is None else "fail"
    return {
        "baseline": baseline_name,
        "status": status,
        "profile_csv": profile_path,
        "detail_csv": detail_path,
        "profile_rows": sum(len(rows) for rows in groups.values()),
        "episodes": len(groups),
        "max_abs_damper_diff": max_damper_diff,
        "max_abs_spring_diff": max_spring_diff,
        "max_rl_mean_abs_action": max_action,
        "max_rl_mean_abs_residual_damper": max_residual,
        "damper_tolerance": args.damper_tolerance,
        "spring_tolerance": args.spring_tolerance,
        "action_tolerance": args.action_tolerance,
        "residual_tolerance": args.residual_tolerance,
        "first_failure_episode": (
            first_failure.get("episode_index", "") if first_failure else ""),
        "first_failure_frame": (
            first_failure.get("frame", "") if first_failure else ""),
        "first_failure_step": (
            first_failure.get("step", "") if first_failure else ""),
        "first_failure_reason": (
            first_failure.get("reason", "") if first_failure else ""),
    }


def write_csv_rows(
    path: str,
    rows: Sequence[Mapping[str, Any]],
    preferred_fields: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(preferred_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: str, values: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as json_file:
        json.dump(values, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def parse_baselines(value: str) -> Tuple[str, ...]:
    baselines = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not baselines:
        raise ValueError("at least one baseline is required")
    for baseline in baselines:
        if baseline not in ("skyhook", "pid"):
            raise ValueError("unsupported baseline %s" % baseline)
    return baselines


def output_dir_for(profile_path: str, output_dir: str) -> str:
    if output_dir:
        return os.path.abspath(os.path.expanduser(output_dir))
    return os.path.join(os.path.dirname(profile_path), "zero_residual_equivalence")


def run_equivalence(
    profile: str,
    output_dir: str = "",
    baselines: Sequence[str] = ("skyhook", "pid"),
    pid_config: str = DEFAULT_PID_CONFIG,
    skyhook_config: str = DEFAULT_SKYHOOK_CONFIG,
    rl_residual_config: str = DEFAULT_RL_RESIDUAL_CONFIG,
    default_dt: float = 0.05,
    damper_tolerance: float = 1.0e-6,
    spring_tolerance: float = 0.0,
    action_tolerance: float = 0.0,
    residual_tolerance: float = 0.0,
) -> Tuple[List[Dict[str, Any]], str]:
    profile_path = resolve_profile_path(profile)
    groups = read_profile_rows(profile_path)
    args = argparse.Namespace(
        pid_config=pid_config,
        skyhook_config=skyhook_config,
        rl_residual_config=rl_residual_config,
        default_dt=default_dt,
        damper_tolerance=damper_tolerance,
        spring_tolerance=spring_tolerance,
        action_tolerance=action_tolerance,
        residual_tolerance=residual_tolerance)
    resolved_output_dir = output_dir_for(profile_path, output_dir)
    os.makedirs(resolved_output_dir, exist_ok=True)

    summary_rows = [
        replay_equivalence_for_baseline(
            baseline,
            groups,
            args,
            resolved_output_dir,
            profile_path)
        for baseline in baselines
    ]
    summary_csv = os.path.join(
        resolved_output_dir,
        "zero_residual_equivalence_summary.csv")
    summary_json = os.path.join(
        resolved_output_dir,
        "zero_residual_equivalence_summary.json")
    write_csv_rows(summary_csv, summary_rows, SUMMARY_FIELDS)
    write_json(summary_json, {
        "profile_csv": profile_path,
        "summary_csv": summary_csv,
        "summary_rows": summary_rows,
    })
    return summary_rows, summary_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        required=True,
        help="profile.csv path or scenario directory containing profile.csv")
    parser.add_argument(
        "--output-dir",
        default="",
        help="output directory; default: <profile_dir>/zero_residual_equivalence")
    parser.add_argument(
        "--baselines",
        default="skyhook,pid",
        help="comma-separated baselines to test: skyhook,pid")
    parser.add_argument("--pid-config", default=DEFAULT_PID_CONFIG)
    parser.add_argument("--skyhook-config", default=DEFAULT_SKYHOOK_CONFIG)
    parser.add_argument("--rl-residual-config", default=DEFAULT_RL_RESIDUAL_CONFIG)
    parser.add_argument("--default-dt", default=0.05, type=float)
    parser.add_argument("--damper-tolerance", default=1.0e-6, type=float)
    parser.add_argument("--spring-tolerance", default=0.0, type=float)
    parser.add_argument("--action-tolerance", default=0.0, type=float)
    parser.add_argument("--residual-tolerance", default=0.0, type=float)
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="write reports but return exit code 0 even when equivalence fails")
    return parser


def print_summary(summary_rows: Sequence[Mapping[str, Any]], summary_csv: str) -> None:
    print("zero-residual equivalence summary: %s" % summary_csv)
    for row in summary_rows:
        print(
            "  {baseline}: {status} "
            "max_abs_damper_diff={damper:.9g} "
            "max_abs_spring_diff={spring:.9g} "
            "max_rl_mean_abs_action={action:.9g} "
            "max_rl_mean_abs_residual_damper={residual:.9g}".format(
                baseline=row.get("baseline"),
                status=row.get("status"),
                damper=float(row.get("max_abs_damper_diff", 0.0)),
                spring=float(row.get("max_abs_spring_diff", 0.0)),
                action=float(row.get("max_rl_mean_abs_action", 0.0)),
                residual=float(row.get("max_rl_mean_abs_residual_damper", 0.0))))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary_rows, summary_csv = run_equivalence(
        profile=args.profile,
        output_dir=args.output_dir,
        baselines=parse_baselines(args.baselines),
        pid_config=os.path.abspath(os.path.expanduser(args.pid_config)),
        skyhook_config=os.path.abspath(os.path.expanduser(args.skyhook_config)),
        rl_residual_config=os.path.abspath(os.path.expanduser(args.rl_residual_config)),
        default_dt=args.default_dt,
        damper_tolerance=args.damper_tolerance,
        spring_tolerance=args.spring_tolerance,
        action_tolerance=args.action_tolerance,
        residual_tolerance=args.residual_tolerance)
    print_summary(summary_rows, summary_csv)
    failed = any(row.get("status") != "pass" for row in summary_rows)
    return 1 if failed and not args.no_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

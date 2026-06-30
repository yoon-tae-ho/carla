#!/usr/bin/env python

"""Run LEAD/TFv6 routes with the suspension-control sidecar.

This is a thin LEAD-specific wrapper around
``transfuser_suspension_control_suite.py``. It keeps the existing sidecar,
controller, metrics, and comparison code, but swaps the route command defaults
and exports the environment variables consumed by ``run_lead_debug_route.sh``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Mapping, Sequence

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import transfuser_suspension_control_suite as suite


def _detect_sim_root() -> str:
    if os.path.isdir("/workspace/e2e_models"):
        return "/workspace"
    return os.environ.get("SIM_ROOT", os.path.expanduser("~/sim"))


SIM_ROOT = _detect_sim_root()
E2E_ROOT = os.environ.get("E2E_ROOT", os.path.join(SIM_ROOT, "e2e_models"))
DEFAULT_LEAD_ROUTE_SCRIPT = os.path.join(
    E2E_ROOT, "scripts", "run_lead_debug_route.sh")
DEFAULT_LEAD_ROUTES = os.path.join(
    E2E_ROOT,
    "carla_garage",
    "leaderboard",
    "data",
    "suspension_routes",
    "suspension_town04_fig8_route18_noscenario.xml",
)
DEFAULT_LEAD_SUITE_ROOT = os.path.join(
    E2E_ROOT, "outputs", "lead", "suspension_suite")
DEFAULT_LEAD_PROJECT_ROOT = os.path.join(E2E_ROOT, "lead")
DEFAULT_LEAD_CHECKPOINT = os.environ.get(
    "LEAD_DEFAULT_CHECKPOINT",
    os.path.join(
        DEFAULT_LEAD_PROJECT_ROOT,
        "outputs",
        "checkpoints",
        "tfv6_regnety032",
    ))


_ORIGINAL_RUN_ENV = suite.run_env
_ORIGINAL_OUTPUT_DIR_PATH = suite.output_dir_path
_ORIGINAL_NEWEST_RESULT_FROM_OUTPUT = suite.newest_result_from_output


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _checkpoint_in_dir(path: str) -> str:
    if not path:
        return ""
    candidate = os.path.join(path, "checkpoint_endpoint.json")
    return candidate if os.path.isfile(candidate) else ""


def _lead_output_dirs_from_lines(lines: Sequence[str]) -> List[str]:
    output_dirs: List[str] = []
    for line in lines:
        text = line.strip()
        if "LEAD_OUTPUT_DIR=" in text:
            output_dirs.append(text.split("LEAD_OUTPUT_DIR=", 1)[1].strip())
        elif "Output Dir:" in text:
            output_dirs.append(text.split("Output Dir:", 1)[1].strip())
    return output_dirs


def newest_lead_result_from_output(lines: Sequence[str]) -> str:
    for output_dir in reversed(_lead_output_dirs_from_lines(lines)):
        checkpoint = _checkpoint_in_dir(output_dir)
        if checkpoint:
            return checkpoint

    for line in reversed(lines):
        candidate = line.strip()
        if candidate.startswith("CHECKPOINT="):
            candidate = candidate.split("=", 1)[1].strip()
        if candidate.endswith("checkpoint_endpoint.json") and os.path.isfile(candidate):
            return candidate

    return _ORIGINAL_NEWEST_RESULT_FROM_OUTPUT(lines)


def lead_output_dir_path(path: str) -> str:
    if path:
        return _expand_path(path)
    return os.path.join(DEFAULT_LEAD_SUITE_ROOT, time.strftime("%Y%m%d_%H%M%S"))


def lead_run_env(
    args: argparse.Namespace,
    seed: int,
    scenario_dir: str,
) -> Dict[str, str]:
    env = _ORIGINAL_RUN_ENV(args, seed, scenario_dir)
    e2e_root = env.get("E2E_ROOT", E2E_ROOT)
    sim_root = env.get("SIM_ROOT", SIM_ROOT)
    lead_project_root = env.get(
        "LEAD_PROJECT_ROOT",
        os.path.join(e2e_root, "lead"))
    lead_output_dir = os.path.join(scenario_dir, "lead")

    env["SIM_ROOT"] = sim_root
    env["E2E_ROOT"] = e2e_root
    env["LEAD_PROJECT_ROOT"] = lead_project_root
    lead_checkpoint = getattr(args, "lead_checkpoint", "") or env.get(
        "LEAD_CHECKPOINT",
        DEFAULT_LEAD_CHECKPOINT)
    env["LEAD_CHECKPOINT"] = _expand_path(lead_checkpoint)
    env["LEAD_ROUTE"] = _expand_path(args.routes)
    env["LEAD_OUTPUT_DIR"] = lead_output_dir
    env["LEAD_PORT"] = str(args.port)
    env["CARLA_PORT"] = str(args.port)
    env["LEAD_TM_PORT"] = str(args.traffic_manager_port)
    env["LEAD_TRAFFIC_MANAGER_SEED"] = str(seed)
    env["LEAD_REPETITIONS"] = str(args.repetitions)
    env["LEAD_DEBUG"] = str(args.debug)
    env["LEAD_TIMEOUT"] = env.get("LEADERBOARD_TIMEOUT", str(max(300.0, args.timeout)))
    env.setdefault("CARLA_USE_EGG", "0")

    if args.planning_provider == "jsonl" and args.planning_preview_jsonl:
        env["TAEHO_LEAD_PLANNING_JSONL"] = _expand_path(
            args.planning_preview_jsonl)

    if "bench2drive" in env["LEAD_ROUTE"].lower():
        env.setdefault("LEAD_BENCH2DRIVE", "1")
    return env


def print_lead_suite_table(rows: Sequence[Mapping[str, object]]) -> None:
    print("")
    print("LEAD suspension-control suite summary:")
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
            suite.format_value(row.get("score_composed", "")),
            suite.format_value(row.get("comfort_comfort_score", "")),
            suite.format_value(row.get("stability_rms_roll", "")),
            row.get("sidecar_profile_rows", ""),
            row.get("sidecar_diagnostic_rows", ""),
            row.get("return_code", "")))


def configure_suite_defaults() -> None:
    suite.SIM_ROOT = SIM_ROOT
    suite.DEFAULT_ROUTE_SCRIPT = DEFAULT_LEAD_ROUTE_SCRIPT
    suite.DEFAULT_ROUTES = DEFAULT_LEAD_ROUTES
    suite.DEFAULT_TFPP_OUTPUT_ROOT = os.path.join(E2E_ROOT, "outputs", "lead")
    suite.run_env = lead_run_env
    suite.output_dir_path = lead_output_dir_path
    suite.newest_result_from_output = newest_lead_result_from_output
    suite.print_suite_table = print_lead_suite_table


def preflight_sidecar_carla_import() -> None:
    try:
        carla = suite.import_carla()
    except Exception as error:
        raise RuntimeError(
            "LEAD sidecar CARLA import preflight failed: %s: %s" % (
                error.__class__.__name__,
                error)) from error

    missing = [
        name for name in (
            "get_suspension_physics_control",
            "apply_suspension_physics_control",
        )
        if not hasattr(carla.Vehicle, name)
    ]
    if missing:
        raise RuntimeError(
            "LEAD sidecar CARLA import preflight missing patched Vehicle "
            "methods: %s" % ", ".join(missing))
    print(
        "[lead_suspension_control_suite] sidecar CARLA import OK: %s" %
        getattr(carla, "__file__", "<unknown>"))


def build_arg_parser() -> argparse.ArgumentParser:
    configure_suite_defaults()
    parser = suite.build_arg_parser()
    parser.description = __doc__
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run LEAD sidecar CARLA import preflight and exit")
    parser.add_argument(
        "--lead-checkpoint",
        default=DEFAULT_LEAD_CHECKPOINT,
        help="LEAD checkpoint directory (default: tfv6_regnety032)")
    parser.set_defaults(
        debug=0,
        route_script=DEFAULT_LEAD_ROUTE_SCRIPT,
        routes=DEFAULT_LEAD_ROUTES,
        scenarios="stock,identity,skyhook",
    )
    for action in parser._actions:
        if action.dest == "debug":
            action.default = 0
            action.help = "DEBUG/LEAD_DEBUG env for run_lead_debug_route.sh (default: 0)"
        elif action.dest == "route_script":
            action.default = DEFAULT_LEAD_ROUTE_SCRIPT
            action.help = "run_lead_debug_route.sh path"
        elif action.dest == "routes":
            action.default = DEFAULT_LEAD_ROUTES
            action.help = "LEAD routes XML path (default: suspension Town04 route18 no-scenario)"
        elif action.dest == "scenarios":
            action.default = "stock,identity,skyhook"
            action.help = (
                "comma-separated scenarios: stock,identity,pid,skyhook,"
                "rl_residual_skyhook,rl_residual_pid,"
                "rl_residual_skyhook_no_planning,"
                "rl_zero_residual_pid,rl_zero_residual_skyhook,"
                "rl_const_plus_0p02_skyhook,"
                "rl_const_minus_0p02_skyhook,"
                "rl_random_small_skyhook,"
                "rl_const_action_plus_0p25_skyhook,"
                "rl_const_action_minus_0p25_skyhook,"
                "rl_random_action_0p10_skyhook,"
                "constant_damper_1p03,"
                "target_speed_schedule_v0,"
                "target_speed_schedule_shadow,"
                "target_speed_schedule_v0_safe,"
                "skyhook_roll,"
                "skyhook_roll_yaw,"
                "skyhook_estimator_dryrun,"
                "constant_damper_1p02,"
                "pard_v2_shadow,"
                "pard_v2_active_ultra_safe,"
                "pard_v2_active_safe_1p06,"
                "pard_v2_active_aggressive_0p75_1p25 "
                "(default: stock,identity,skyhook)")
    return parser


def main(args: argparse.Namespace) -> None:
    configure_suite_defaults()
    preflight_sidecar_carla_import()
    if getattr(args, "preflight_only", False):
        return
    suite.main(args)


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

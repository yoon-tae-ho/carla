#!/usr/bin/env python3

"""Summarize and gate Step 03 LEAD stock-vs-identity paired repeats."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import summarize_lead_step07_matrix as step07


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Step 03 output root")
    parser.add_argument("--seed", default="", help="expected seed")
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=3,
        help="minimum stock/identity pairs required for pass")
    parser.add_argument(
        "--score-drop-tolerance",
        type=float,
        default=5.0,
        help="allowed identity score_composed drop vs stock")
    parser.add_argument(
        "--route-score-drop-tolerance",
        type=float,
        default=1.0,
        help="allowed identity score_route drop vs stock")
    parser.add_argument(
        "--identity-scale-tolerance",
        type=float,
        default=1.0e-6,
        help="allowed identity command/readback scale error")
    parser.add_argument(
        "--sim-root",
        default=os.environ.get("SIM_ROOT", os.path.expanduser("~/sim")),
        help="host sim root used to translate /workspace paths")
    return parser.parse_args()


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv_rows(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    step07.write_csv_rows(path, rows)


def write_json(path: str, data: Any) -> None:
    step07.write_json(path, data)


def read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path) as json_file:
            data = json.load(json_file)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def format_value(value: Any) -> str:
    value = step07.format_value(value)
    return "" if value is None else str(value)


def float_or_none(value: Any) -> Optional[float]:
    return step07.float_or_none(value)


def int_or_zero(value: Any) -> int:
    return step07.int_or_zero(value)


def infraction_count(value: Any) -> int:
    return step07.infraction_count(value)


def bool_true(value: Any) -> bool:
    if value is True:
        return True
    return str(value).strip().lower() in ("1", "true", "yes")


def csv_value(row: Mapping[str, Any], key: str) -> str:
    return format_value(row.get(key, ""))


def scenario_label(scenario_key: str) -> str:
    return step07.sanitize_label(scenario_key)


def read_child_return_codes(attempt_dir: str) -> Dict[str, str]:
    rows = read_csv_rows(os.path.join(attempt_dir, "step07_runner_results.csv"))
    result: Dict[str, str] = {}
    for row in rows:
        result[row.get("scenario_key", "")] = row.get("return_code", "")
    return result


def read_wrapper_results(root: str) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(os.path.join(root, "equivalence_results.csv"))
    return {row.get("repeat", ""): row for row in rows}


def finite_scale_error(value: Any, expected: float = 1.0) -> Optional[float]:
    number = float_or_none(value)
    if number is None:
        return None
    return abs(number - expected)


def summarize_identity_scales(path: str, tolerance: float) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "identity_command_rows": 0,
        "identity_readback_rows": 0,
        "identity_spring_scale_violations": 0,
        "identity_damper_scale_violations": 0,
        "identity_readback_spring_violations": 0,
        "identity_readback_damper_violations": 0,
        "identity_max_command_scale_error": "",
        "identity_max_readback_scale_error": "",
    }
    if not path or not os.path.isfile(path):
        return summary

    command_errors: List[float] = []
    readback_errors: List[float] = []
    for row in read_csv_rows(path):
        command_seen = False
        for field, violation_key in (
                ("spring_scale", "identity_spring_scale_violations"),
                ("damper_scale", "identity_damper_scale_violations")):
            error = finite_scale_error(row.get(field))
            if error is None:
                continue
            command_seen = True
            command_errors.append(error)
            if error > tolerance:
                summary[violation_key] += 1
        if command_seen:
            summary["identity_command_rows"] += 1

        readback_seen = False
        for field, violation_key in (
                ("mean_spring_scale_readback", "identity_readback_spring_violations"),
                ("min_spring_scale_readback", "identity_readback_spring_violations"),
                ("max_spring_scale_readback", "identity_readback_spring_violations"),
                ("mean_damper_scale_readback", "identity_readback_damper_violations"),
                ("min_damper_scale_readback", "identity_readback_damper_violations"),
                ("max_damper_scale_readback", "identity_readback_damper_violations")):
            error = finite_scale_error(row.get(field))
            if error is None:
                continue
            readback_seen = True
            readback_errors.append(error)
            if error > tolerance:
                summary[violation_key] += 1
        if readback_seen:
            summary["identity_readback_rows"] += 1

    if command_errors:
        summary["identity_max_command_scale_error"] = max(command_errors)
    if readback_errors:
        summary["identity_max_readback_scale_error"] = max(readback_errors)
    return summary


def route_is_complete(row: Mapping[str, Any]) -> bool:
    if not int_or_zero(row.get("checkpoint_exists")):
        return False
    if row.get("entry_status") not in ("", "Finished"):
        return False
    if row.get("eligible") not in ("", True) and not bool_true(row.get("eligible")):
        return False
    if row.get("global_status") not in ("Perfect", "Completed"):
        return False
    duration_game = float_or_none(row.get("duration_game"))
    return bool(duration_game is not None and duration_game > 0.0)


def summarize_one_plan_row(
    plan_row: Mapping[str, str],
    args: argparse.Namespace,
    wrapper_results: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    scenario = plan_row.get("scenario_key", "")
    repeat = plan_row.get("repeat", "")
    attempt_dir = plan_row.get("attempt_dir_host", "")
    run_kind = "sidecar" if scenario == "LEAD_identity" else "direct"
    step07_plan = {
        "seed": plan_row.get("seed", ""),
        "scenario_key": scenario,
        "run_kind": run_kind,
        "run_dir_host": plan_row.get("run_dir_host", ""),
        "planning_jsonl_host": "",
    }
    if run_kind == "sidecar":
        detail = step07.summarize_sidecar_run(
            step07_plan,
            args.sim_root,
            planning_age_limit=2.0)
    else:
        detail = step07.summarize_direct_run(step07_plan, args.sim_root)

    child_rc = read_child_return_codes(attempt_dir).get(scenario, "")
    wrapper_row = wrapper_results.get(repeat, {})
    row: Dict[str, Any] = {
        "repeat": repeat,
        "repeat_index": plan_row.get("repeat_index", ""),
        "seed": plan_row.get("seed", ""),
        "scenario_key": scenario,
        "run_kind": run_kind,
        "attempt_dir": attempt_dir,
        "run_dir": plan_row.get("run_dir_host", ""),
        "port": plan_row.get("port", ""),
        "traffic_manager_port": plan_row.get("traffic_manager_port", ""),
        "compose_project_name": plan_row.get("compose_project_name", ""),
        "runner_return_code": child_rc,
        "wrapper_return_code": wrapper_row.get("return_code", ""),
        "attempt_stdout": wrapper_row.get("attempt_stdout", ""),
    }
    row.update(detail)
    row["major_infraction_count"] = sum(
        infraction_count(row.get(field))
        for field in step07.MAJOR_INFRACTION_FIELDS)
    row["collision_count"] = sum(
        infraction_count(row.get(field))
        for field in (
            "collisions_layout",
            "collisions_pedestrian",
            "collisions_vehicle"))
    row["min_speed_count"] = infraction_count(row.get("min_speed_infractions"))
    row["route_complete"] = int(route_is_complete(row))
    if scenario == "LEAD_identity":
        row.update(summarize_identity_scales(
            str(row.get("controller_diagnostics_csv", "")),
            args.identity_scale_tolerance))
    return row


def number_delta(left: Any, right: Any) -> Any:
    left_value = float_or_none(left)
    right_value = float_or_none(right)
    if left_value is None or right_value is None:
        return ""
    return left_value - right_value


def valid_return_code(value: Any) -> bool:
    return str(value) in ("0", "0.0")


def evaluate_pair(
    repeat: str,
    stock: Optional[Mapping[str, Any]],
    identity: Optional[Mapping[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    failures: List[str] = []
    warnings: List[str] = []
    if stock is None:
        failures.append("missing_stock")
    if identity is None:
        failures.append("missing_identity")
    if stock is None or identity is None:
        return {
            "repeat": repeat,
            "seed": args.seed,
            "gate_status": "fail",
            "failed_checks": ";".join(failures),
            "warnings": "",
        }

    if not valid_return_code(stock.get("runner_return_code")):
        failures.append("stock_runner_return_code")
    if not valid_return_code(identity.get("runner_return_code")):
        failures.append("identity_runner_return_code")
    if not route_is_complete(stock):
        failures.append("stock_route_incomplete")
    if not route_is_complete(identity):
        failures.append("identity_route_incomplete")
    if int_or_zero(stock.get("major_infraction_count")) != 0:
        failures.append("stock_major_infractions")
    if int_or_zero(identity.get("major_infraction_count")) != 0:
        failures.append("identity_major_infractions")
    if int_or_zero(identity.get("collision_count")) > int_or_zero(stock.get("collision_count")):
        failures.append("identity_new_collision")

    score_delta = number_delta(identity.get("score_composed"), stock.get("score_composed"))
    route_score_delta = number_delta(identity.get("score_route"), stock.get("score_route"))
    duration_game_delta = number_delta(identity.get("duration_game"), stock.get("duration_game"))
    duration_system_delta = number_delta(identity.get("duration_system"), stock.get("duration_system"))
    if isinstance(score_delta, float) and score_delta < -args.score_drop_tolerance:
        failures.append("score_composed_drop")
    if isinstance(route_score_delta, float) and route_score_delta < -args.route_score_drop_tolerance:
        failures.append("route_score_drop")

    if int_or_zero(stock.get("min_speed_count")):
        warnings.append("stock_min_speed_caveat")
    if int_or_zero(identity.get("min_speed_count")):
        warnings.append("identity_min_speed_caveat")
    if int_or_zero(identity.get("verify_warning_count")):
        warnings.append("identity_verify_warning")

    for field, failure in (
            ("controller_diagnostic_rows", "identity_missing_diagnostics"),
            ("identity_command_rows", "identity_missing_command_rows"),
            ("identity_readback_rows", "identity_missing_readback_rows")):
        if int_or_zero(identity.get(field)) == 0:
            failures.append(failure)
    for field, failure in (
            ("verify_hard_error_count", "identity_verify_hard_error"),
            ("sidecar_runtime_hard_error_count", "identity_sidecar_runtime_hard_error"),
            ("identity_spring_scale_violations", "identity_spring_scale"),
            ("identity_damper_scale_violations", "identity_damper_scale"),
            ("identity_readback_spring_violations", "identity_readback_spring_scale"),
            ("identity_readback_damper_violations", "identity_readback_damper_scale")):
        if int_or_zero(identity.get(field)) != 0:
            failures.append(failure)

    return {
        "repeat": repeat,
        "seed": stock.get("seed", args.seed),
        "gate_status": "fail" if failures else "pass",
        "failed_checks": ";".join(sorted(set(failures))),
        "warnings": ";".join(sorted(set(warnings))),
        "stock_score_composed": stock.get("score_composed", ""),
        "identity_score_composed": identity.get("score_composed", ""),
        "score_composed_delta_identity_minus_stock": score_delta,
        "stock_score_route": stock.get("score_route", ""),
        "identity_score_route": identity.get("score_route", ""),
        "score_route_delta_identity_minus_stock": route_score_delta,
        "stock_global_status": stock.get("global_status", ""),
        "identity_global_status": identity.get("global_status", ""),
        "stock_major_infraction_count": stock.get("major_infraction_count", ""),
        "identity_major_infraction_count": identity.get("major_infraction_count", ""),
        "stock_collision_count": stock.get("collision_count", ""),
        "identity_collision_count": identity.get("collision_count", ""),
        "stock_min_speed_count": stock.get("min_speed_count", ""),
        "identity_min_speed_count": identity.get("min_speed_count", ""),
        "stock_duration_game": stock.get("duration_game", ""),
        "identity_duration_game": identity.get("duration_game", ""),
        "duration_game_delta_identity_minus_stock": duration_game_delta,
        "stock_duration_system": stock.get("duration_system", ""),
        "identity_duration_system": identity.get("duration_system", ""),
        "duration_system_delta_identity_minus_stock": duration_system_delta,
        "identity_command_rows": identity.get("identity_command_rows", ""),
        "identity_readback_rows": identity.get("identity_readback_rows", ""),
        "identity_max_command_scale_error": identity.get("identity_max_command_scale_error", ""),
        "identity_max_readback_scale_error": identity.get("identity_max_readback_scale_error", ""),
        "stock_result_json": stock.get("result_json", ""),
        "identity_result_json": identity.get("result_json", ""),
        "identity_controller_diagnostics_csv": identity.get("controller_diagnostics_csv", ""),
        "identity_sidecar_events_csv": identity.get("sidecar_events_csv", ""),
    }


def group_rows_by_repeat(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Mapping[str, Any]]]:
    grouped: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        repeat = str(row.get("repeat", ""))
        scenario = str(row.get("scenario_key", ""))
        grouped.setdefault(repeat, {})[scenario] = row
    return grouped


def write_readme(
    path: str,
    root: str,
    gate: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
) -> None:
    with open(path, "w") as readme:
        readme.write("# Step 03 Stock Identity Equivalence Results\n\n")
        readme.write("Output root:\n\n")
        readme.write("```text\n%s\n```\n\n" % root)
        readme.write("Gate status: `%s`\n\n" % gate.get("status", ""))
        readme.write("Pairs: `%s`, passed: `%s`, failed: `%s`\n\n" % (
            gate.get("pair_count", ""),
            gate.get("passed_pair_count", ""),
            gate.get("failed_pair_count", "")))
        readme.write("Generated files:\n\n")
        for name in (
                "equivalence_summary.csv",
                "equivalence_summary.json",
                "equivalence_pairs.csv",
                "equivalence_gate.json",
                "README_equivalence.md"):
            readme.write("- `%s`\n" % name)
        readme.write("\n")
        failed = [row for row in pair_rows if row.get("gate_status") != "pass"]
        if failed:
            readme.write("Failed pairs:\n\n")
            for row in failed:
                readme.write("- repeat `%s`: `%s`\n" % (
                    row.get("repeat", ""),
                    row.get("failed_checks", "")))
            readme.write("\n")
        readme.write(
            "This gate treats MinSpeedTest as a recorded caveat rather than a "
            "critical major-infraction failure; collisions, route timeout, "
            "route deviation, outside-route, vehicle-blocked, scenario-timeout, "
            "and other major counters still fail the pair.\n")


def main(args: argparse.Namespace) -> None:
    root = os.path.abspath(os.path.expanduser(args.root))
    manifest = read_json(os.path.join(root, "equivalence_manifest.json"))
    plan_rows = read_csv_rows(os.path.join(root, "equivalence_plan.csv"))
    if args.seed:
        plan_rows = [row for row in plan_rows if row.get("seed", "") == args.seed]
    wrapper_results = read_wrapper_results(root)

    summary_rows = [
        summarize_one_plan_row(row, args, wrapper_results)
        for row in plan_rows
    ]
    grouped = group_rows_by_repeat(summary_rows)
    dry_run = bool(manifest.get("dry_run")) or manifest.get("status") == "dry_run"
    sorted_repeats = sorted(grouped, key=lambda item: int(item) if item.isdigit() else item)
    if dry_run:
        pair_rows = [
            {
                "repeat": repeat,
                "seed": args.seed or manifest.get("seed", ""),
                "gate_status": "dry_run",
                "failed_checks": "",
                "warnings": "dry_run_no_route_result",
            }
            for repeat in sorted_repeats
        ]
    else:
        pair_rows = [
            evaluate_pair(
                repeat,
                grouped[repeat].get("LEAD_stock"),
                grouped[repeat].get("LEAD_identity"),
                args)
            for repeat in sorted_repeats
        ]

    failed_pairs = [row for row in pair_rows if row.get("gate_status") == "fail"]
    passed_pairs = [row for row in pair_rows if row.get("gate_status") == "pass"]
    status = "dry_run" if dry_run else ("pass" if not failed_pairs else "fail")
    if not dry_run and len(pair_rows) < args.min_pairs:
        status = "fail"
        failed_pairs = list(failed_pairs)
    gate_path = os.path.join(root, "equivalence_gate.json")
    summary_csv = os.path.join(root, "equivalence_summary.csv")
    summary_json = os.path.join(root, "equivalence_summary.json")
    pairs_csv = os.path.join(root, "equivalence_pairs.csv")
    readme_md = os.path.join(root, "README_equivalence.md")

    gate: Dict[str, Any] = {
        "status": status,
        "root": root,
        "seed": args.seed or manifest.get("seed", ""),
        "pair_count": len(pair_rows),
        "passed_pair_count": len(passed_pairs),
        "failed_pair_count": len(failed_pairs),
        "min_pairs": args.min_pairs,
        "score_drop_tolerance": args.score_drop_tolerance,
        "route_score_drop_tolerance": args.route_score_drop_tolerance,
        "identity_scale_tolerance": args.identity_scale_tolerance,
        "dry_run": dry_run,
        "failed_pairs": failed_pairs,
        "files": {
            "equivalence_summary_csv": summary_csv,
            "equivalence_summary_json": summary_json,
            "equivalence_pairs_csv": pairs_csv,
            "equivalence_gate_json": gate_path,
            "readme_equivalence_md": readme_md,
        },
        "caveats": [
            "seed 101 is accepted as a collision-free stock candidate, not a strict all-infractions-zero perfect seed",
            "Town04 route18 removes scenario actors, so traffic-manager seed effects may be limited",
            "MinSpeedTest is recorded as a caveat and warning, not a critical major-infraction failure",
        ],
    }
    if not dry_run and len(pair_rows) < args.min_pairs:
        gate["status"] = "fail"
        gate["failed_pairs"] = list(failed_pairs) + [{
            "repeat": "",
            "seed": gate["seed"],
            "gate_status": "fail",
            "failed_checks": "insufficient_pairs",
            "warnings": "",
        }]
        gate["failed_pair_count"] = len(gate["failed_pairs"])

    write_csv_rows(summary_csv, summary_rows)
    write_json(summary_json, summary_rows)
    write_csv_rows(pairs_csv, pair_rows)
    write_json(gate_path, gate)
    write_readme(readme_md, root, gate, pair_rows)

    print("Step 03 stock-vs-identity equivalence summary complete")
    print("  root=%s" % root)
    print("  gate_status=%s" % gate["status"])
    print("  pairs=%d passed=%d failed=%d" % (
        len(pair_rows),
        len(passed_pairs),
        gate["failed_pair_count"]))
    print("  summary_csv=%s" % summary_csv)
    print("  gate_report=%s" % gate_path)


if __name__ == "__main__":
    main(parse_args())

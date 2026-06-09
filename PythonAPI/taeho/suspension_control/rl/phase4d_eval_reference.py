"""Acceptance and manifest writer for Phase 4-D S8 reference eval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


S8_SCENARIO = "S8_rl_zero_residual_skyhook"
S8_CONTROLLER = "rl_zero_residual_skyhook"


PHASE4D_EVAL_S8_ACCEPTANCE_FIELDS: Tuple[str, ...] = (
    "phase4d_eval_s8_status",
    "reference_status",
    "failed_checks",
    "warnings",
    "workflow_role",
    "scenario",
    "controller",
    "baseline",
    "planning_provider",
    "routes_subset",
    "eval_seed",
    "output_dir",
    "suite_summary_path",
    "route_return_code",
    "scenario_status",
    "scenario_return_code",
    "entry_status",
    "global_status",
    "score_route",
    "score_composed",
    "score_penalty",
    "route_completion_proxy",
    "sidecar_command_verifies",
    "policy_available_ratio",
    "fallback_ratio",
    "low_speed_mask_ratio",
    "mean_abs_action",
    "mean_abs_residual_damper",
    "effective_control_ratio",
    "hard_safety_gate_ratio",
    "soft_safety_gain_ratio",
    "collision_count",
    "lane_invasion_count",
    "red_light_count",
    "route_timeout_count",
    "blocked_vehicle_count",
    "sidecar_fatal_errors",
    "sidecar_runtime_errors",
    "result_json",
    "profile_csv",
    "controller_diagnostics_csv",
    "metrics_by_episode_csv",
    "sidecar_events_csv",
    "route_stdout_log",
)


PHASE4D_EVAL_S8_MANIFEST_FIELDS: Tuple[str, ...] = (
    "artifact_id",
    "workflow_role",
    "reference_status",
    "phase",
    "scenario",
    "controller",
    "baseline",
    "planning_provider",
    "routes_subset",
    "eval_seed",
    "eval_dir",
    "suite_summary_path",
    "summary_path",
    "result_json",
    "profile_csv",
    "controller_diagnostics_csv",
    "metrics_by_episode_csv",
    "sidecar_events_csv",
    "route_stdout_log",
    "acceptance_path",
    "created_at",
    "phase4d_eval_s8_status",
    "failed_checks",
    "warnings",
)


def write_phase4d_eval_s8_acceptance(
    *,
    output_dir: str,
    suite_summary_path: str = "",
    routes_subset: str = "00",
    seed: str = "100",
    route_return_code: int = 0,
) -> Tuple[str, str, str, Dict[str, Any], Dict[str, Any]]:
    output_dir = _abs_path(output_dir)
    suite_path = _resolve_path(
        suite_summary_path or os.path.join(output_dir, "suite_summary.csv"),
        output_dir)
    source = _select_s8_row(_load_suite_rows(suite_path))
    failed = _eval_failed_checks(source, route_return_code)
    warnings = _eval_warnings(source)
    row = _acceptance_row(
        output_dir=output_dir,
        suite_summary_path=suite_path,
        routes_subset=routes_subset,
        seed=seed,
        route_return_code=route_return_code,
        source=source or {},
        failed_checks=failed,
        warnings=warnings)
    csv_path, json_path = _write_acceptance_outputs(output_dir, row)
    manifest = _manifest_row(
        output_dir=output_dir,
        suite_summary_path=suite_path,
        acceptance=row)
    manifest_json, _manifest_csv = _write_manifest_outputs(output_dir, manifest)
    return csv_path, json_path, manifest_json, row, manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suite-summary", default="")
    parser.add_argument("--routes-subset", default="00")
    parser.add_argument("--seed", default="100")
    parser.add_argument("--route-return-code", type=int, default=0)
    parser.add_argument("--fail-on-reject", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    csv_path, json_path, manifest_json, row, _manifest = write_phase4d_eval_s8_acceptance(
        output_dir=args.output_dir,
        suite_summary_path=args.suite_summary,
        routes_subset=args.routes_subset,
        seed=args.seed,
        route_return_code=args.route_return_code)
    print("phase4d eval s8 acceptance: %s" % csv_path)
    print("phase4d eval s8 acceptance json: %s" % json_path)
    print("phase4d eval s8 manifest: %s" % manifest_json)
    print("phase4d eval s8 status: %s reference_status=%s failed_checks=%s" % (
        row.get("phase4d_eval_s8_status"),
        row.get("reference_status"),
        row.get("failed_checks")))
    if args.fail_on_reject and row.get("phase4d_eval_s8_status") != "pass":
        return 1
    return 0


def _acceptance_row(
    *,
    output_dir: str,
    suite_summary_path: str,
    routes_subset: str,
    seed: str,
    route_return_code: int,
    source: Mapping[str, Any],
    failed_checks: Sequence[str],
    warnings: Sequence[str],
) -> Dict[str, Any]:
    status = "pass" if not failed_checks else "fail"
    policy_ratio = _float_first(source, ("rl_policy_available_ratio", "policy_available_ratio"))
    action = _float_first(source, ("rl_mean_abs_action", "mean_abs_action"))
    residual = _float_first(
        source,
        ("rl_mean_abs_residual_damper", "mean_abs_residual_damper"))
    return _ordered_row({
        "phase4d_eval_s8_status": status,
        "reference_status": "accepted_reference" if status == "pass" else "invalid_reference",
        "failed_checks": ";".join(_dedupe(failed_checks)),
        "warnings": ";".join(_dedupe(warnings)),
        "workflow_role": "eval_s8_reference",
        "scenario": source.get("scenario", S8_SCENARIO),
        "controller": source.get("controller", S8_CONTROLLER),
        "baseline": "skyhook",
        "planning_provider": "empty",
        "routes_subset": routes_subset,
        "eval_seed": seed,
        "output_dir": output_dir,
        "suite_summary_path": suite_summary_path,
        "route_return_code": int(route_return_code),
        "scenario_status": source.get("status", ""),
        "scenario_return_code": source.get("return_code", ""),
        "entry_status": source.get("entry_status", ""),
        "global_status": source.get("global_status", ""),
        "score_route": source.get("score_route", ""),
        "score_composed": source.get("score_composed", ""),
        "score_penalty": source.get("score_penalty", ""),
        "route_completion_proxy": source.get("route_completion_proxy", ""),
        "sidecar_command_verifies": source.get("sidecar_command_verifies", ""),
        "policy_available_ratio": policy_ratio if policy_ratio is not None else "",
        "fallback_ratio": source.get("fallback_ratio", ""),
        "low_speed_mask_ratio": source.get("low_speed_mask_ratio", ""),
        "mean_abs_action": action if action is not None else "",
        "mean_abs_residual_damper": residual if residual is not None else "",
        "effective_control_ratio": source.get("effective_control_ratio", ""),
        "hard_safety_gate_ratio": source.get("hard_safety_gate_ratio", ""),
        "soft_safety_gain_ratio": source.get("soft_safety_gain_ratio", ""),
        "collision_count": source.get("collision_count", ""),
        "lane_invasion_count": source.get("lane_invasion_count", ""),
        "red_light_count": source.get("red_light_count", ""),
        "route_timeout_count": source.get("route_timeout_count", ""),
        "blocked_vehicle_count": source.get("blocked_vehicle_count", ""),
        "sidecar_fatal_errors": source.get("sidecar_fatal_errors", ""),
        "sidecar_runtime_errors": source.get("sidecar_runtime_errors", ""),
        "result_json": source.get("result_json", ""),
        "profile_csv": source.get("profile_csv", ""),
        "controller_diagnostics_csv": source.get("controller_diagnostics_csv", ""),
        "metrics_by_episode_csv": source.get("metrics_by_episode_csv", ""),
        "sidecar_events_csv": source.get("sidecar_events_csv", ""),
        "route_stdout_log": source.get("route_stdout_log", ""),
    }, PHASE4D_EVAL_S8_ACCEPTANCE_FIELDS)


def _eval_failed_checks(
    source: Optional[Mapping[str, Any]],
    route_return_code: int,
) -> List[str]:
    failed: List[str] = []
    if int(route_return_code) != 0:
        failed.append("route_return_code")
    if source is None:
        failed.append("suite_summary")
        return failed
    if str(source.get("status", "")) != "ok":
        failed.append("scenario_status")
    if _int_value(source.get("return_code")) != 0:
        failed.append("scenario_return_code")
    global_status = str(source.get("global_status", "")).strip()
    if global_status and global_status not in (
            "Perfect",
            "SUCCESS",
            "Finished",
            "Completed",
            "success",
            "finished"):
        failed.append("global_status")
    if not _float_close(_float_first(source, ("score_route",)), 100.0):
        failed.append("score_route")
    if not _float_close(_float_first(source, ("score_composed",)), 100.0):
        failed.append("score_composed")
    if (_float_first(source, ("sidecar_command_verifies",)) or 0.0) <= 0.0:
        failed.append("verification")
    if _float_missing_or_below(_float_first(source, ("rl_policy_available_ratio", "policy_available_ratio")), 0.99):
        failed.append("policy_available_ratio")
    if not _float_close(_float_first(source, ("rl_mean_abs_action", "mean_abs_action")), 0.0):
        failed.append("mean_abs_action")
    if not _float_close(
            _float_first(
                source,
                ("rl_mean_abs_residual_damper", "mean_abs_residual_damper")),
            0.0):
        failed.append("mean_abs_residual_damper")
    if not _float_close(_float_first(source, ("effective_control_ratio",)), 0.0):
        failed.append("effective_control_ratio")
    if not _float_close(_float_first(source, ("hard_safety_gate_ratio",)), 0.0):
        failed.append("hard_safety_gate_ratio")
    for field in (
            "collision_count",
            "lane_invasion_count",
            "red_light_count",
            "route_timeout_count",
            "blocked_vehicle_count"):
        if (_float_first(source, (field,)) or 0.0) > 0.0:
            failed.append(field)
    if (_float_first(source, ("sidecar_fatal_errors",)) or 0.0) > 0.0:
        failed.append("sidecar_fatal_errors")
    return _dedupe(failed)


def _eval_warnings(source: Optional[Mapping[str, Any]]) -> List[str]:
    if source is None:
        return []
    warnings: List[str] = []
    for field in (
            "sidecar_runtime_errors",
            "fallback_ratio",
            "low_speed_mask_ratio",
            "soft_safety_gain_ratio"):
        if (_float_first(source, (field,)) or 0.0) > 0.0:
            warnings.append(field)
    return _dedupe(warnings)


def _manifest_row(
    *,
    output_dir: str,
    suite_summary_path: str,
    acceptance: Mapping[str, Any],
) -> Dict[str, Any]:
    identity = "|".join((
        str(acceptance.get("scenario", "")),
        str(acceptance.get("routes_subset", "")),
        str(acceptance.get("eval_seed", "")),
        str(output_dir),
        str(acceptance.get("result_json", "")),
    ))
    artifact_id = "phase4d-eval-s8-%s" % hashlib.sha256(
        identity.encode("utf-8")).hexdigest()[:12]
    return _ordered_row({
        "artifact_id": artifact_id,
        "workflow_role": "eval_s8_reference",
        "reference_status": acceptance.get("reference_status", ""),
        "phase": "phase4d",
        "scenario": acceptance.get("scenario", S8_SCENARIO),
        "controller": acceptance.get("controller", S8_CONTROLLER),
        "baseline": "skyhook",
        "planning_provider": "empty",
        "routes_subset": acceptance.get("routes_subset", ""),
        "eval_seed": acceptance.get("eval_seed", ""),
        "eval_dir": output_dir,
        "suite_summary_path": suite_summary_path,
        "summary_path": _scenario_summary_path(
            output_dir,
            acceptance.get("eval_seed", ""),
            acceptance.get("scenario", S8_SCENARIO)),
        "result_json": acceptance.get("result_json", ""),
        "profile_csv": acceptance.get("profile_csv", ""),
        "controller_diagnostics_csv": acceptance.get("controller_diagnostics_csv", ""),
        "metrics_by_episode_csv": acceptance.get("metrics_by_episode_csv", ""),
        "sidecar_events_csv": acceptance.get("sidecar_events_csv", ""),
        "route_stdout_log": acceptance.get("route_stdout_log", ""),
        "acceptance_path": os.path.join(output_dir, "phase4d_eval_s8_acceptance.json"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase4d_eval_s8_status": acceptance.get("phase4d_eval_s8_status", ""),
        "failed_checks": acceptance.get("failed_checks", ""),
        "warnings": acceptance.get("warnings", ""),
    }, PHASE4D_EVAL_S8_MANIFEST_FIELDS)


def _scenario_summary_path(output_dir: str, seed: Any, scenario: Any) -> str:
    seed_text = str(seed or "")
    scenario_text = str(scenario or S8_SCENARIO)
    if seed_text:
        return os.path.join(output_dir, "seed%s_%s" % (seed_text, scenario_text), "summary.json")
    return ""


def _write_acceptance_outputs(output_dir: str, row: Mapping[str, Any]) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "phase4d_eval_s8_acceptance.csv")
    json_path = os.path.join(output_dir, "phase4d_eval_s8_acceptance.json")
    _write_single_row_csv(csv_path, row, PHASE4D_EVAL_S8_ACCEPTANCE_FIELDS)
    _write_json(json_path, row)
    return csv_path, json_path


def _write_manifest_outputs(output_dir: str, row: Mapping[str, Any]) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "phase4d_eval_s8_manifest.json")
    csv_path = os.path.join(output_dir, "phase4d_eval_s8_manifest.csv")
    _write_json(json_path, row)
    _write_single_row_csv(csv_path, row, PHASE4D_EVAL_S8_MANIFEST_FIELDS)
    return json_path, csv_path


def _load_suite_rows(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    if path.endswith(".csv"):
        with open(path, newline="") as csv_file:
            return [dict(row) for row in csv.DictReader(csv_file)]
    try:
        with open(path) as json_file:
            value = json.load(json_file)
    except Exception:
        return []
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, Mapping)]
    if isinstance(value, Mapping):
        return [dict(value)]
    return []


def _select_s8_row(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for row in rows:
        scenario = str(row.get("scenario", ""))
        if scenario == S8_SCENARIO or scenario == "rl_zero_residual_skyhook":
            return row
    return rows[0] if len(rows) == 1 else None


def _write_single_row_csv(path: str, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def _write_json(path: str, value: Mapping[str, Any]) -> None:
    with open(path, "w") as json_file:
        json.dump(value, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _ordered_row(row: Mapping[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    return {key: row.get(key, "") for key in fields}


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path or ".")))


def _resolve_path(path: Any, base_dir: str) -> str:
    value = os.path.expanduser(str(path or ""))
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(base_dir, value))


def _float_first(source: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in source:
            value = _float_value(source.get(key))
            if value is not None:
                return value
    return None


def _float_value(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_value(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float_close(value: Optional[float], expected: float, tol: float = 1.0e-9) -> bool:
    return value is not None and abs(value - expected) <= tol


def _float_missing_or_below(value: Optional[float], threshold: float) -> bool:
    return value is None or value < threshold


def _dedupe(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

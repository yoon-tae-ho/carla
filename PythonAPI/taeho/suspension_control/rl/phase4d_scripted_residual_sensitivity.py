"""CARLA-free planning and aggregation CLI for scripted residual sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .phase4d_authority_sensitivity_common import (
    DEFAULT_SCRIPTED_RESIDUALS,
    SCRIPTED_RESIDUAL_VALUES,
    parse_bool,
    parse_scripted_residuals,
    write_scripted_residual_plan,
)


STATUS_PASS = "SCRIPTED_SENSITIVITY_PASS"
STATUS_PASS_WARN = "SCRIPTED_SENSITIVITY_PASS_WITH_WARNINGS"
STATUS_FAIL_SAFETY = "SCRIPTED_SENSITIVITY_FAIL_SAFETY"
STATUS_INCOMPLETE = "SCRIPTED_SENSITIVITY_INCOMPLETE"

SUITE_SCENARIO = "S4_rl_residual_skyhook"

SUMMARY_FIELDS: Tuple[str, ...] = (
    "eval_seed",
    "scripted_residual",
    "scripted_residual_value",
    "planned_scenario",
    "suite_scenario",
    "scripted_row_status",
    "scenario_status",
    "route_return_code",
    "global_status",
    "score_composed",
    "zero_reference_status",
    "zero_score_composed",
    "raw_mean_abs_action",
    "raw_mean_abs_residual_damper",
    "scaled_mean_abs_residual_damper",
    "final_mean_abs_residual_damper",
    "effective_control_ratio",
    "fallback_ratio",
    "low_speed_mask_ratio",
    "soft_safety_gain_ratio",
    "hard_safety_gate_ratio",
    "residual_scale_clip_ratio",
    "damper_final_clamp_ratio",
    "residual_saturation_ratio",
    "sidecar_runtime_errors",
    "sidecar_fatal_errors",
    "sidecar_command_verifies",
    "collision_count",
    "lane_invasion_count",
    "red_light_count",
    "route_timeout_count",
    "blocked_vehicle_count",
    "delta_comfort_score_vs_zero",
    "delta_rms_vertical_acc_vs_zero",
    "delta_rms_vertical_jerk_vs_zero",
    "delta_peak_abs_vertical_acc_vs_zero",
    "metric_effect_above_noise",
    "stageA_noise_metric",
    "stageA_noise_std",
    "failed_checks",
    "warnings",
    "residual_output_dir",
    "zero_output_dir",
    "compare_output_dir",
)

ACCEPTANCE_FIELDS: Tuple[str, ...] = (
    "phase4d_scripted_residual_status",
    "aggregate_ok",
    "failed_checks",
    "warnings",
    "workflow_role",
    "eval_seed_count",
    "scripted_residual_count",
    "completed_scenario_count",
    "passing_scenario_count",
    "warning_scenario_count",
    "failed_scenario_count",
    "incomplete_scenario_count",
    "zero_failed_seed_count",
    "scripted_effect_above_noise",
    "signed_residual_direction_consistency",
    "magnitude_response_consistency",
    "SAFETY_AUTHORITY_STATUS",
    "SENSITIVITY_DIAGNOSIS",
    "recommended_next_action",
    "summary_csv",
    "summary_json",
    "report_json",
    "manifest_json",
    "output_dir",
)

METRIC_DELTAS: Tuple[Tuple[str, str, str], ...] = (
    (
        "delta_comfort_score_vs_zero",
        "warmup_excluded_comfort_comfort_score",
        "delta_comfort_score",
    ),
    (
        "delta_rms_vertical_acc_vs_zero",
        "warmup_excluded_comfort_rms_vertical_acc",
        "delta_rms_vertical_acc",
    ),
    (
        "delta_rms_vertical_jerk_vs_zero",
        "warmup_excluded_comfort_rms_vertical_jerk",
        "delta_rms_vertical_jerk",
    ),
    (
        "delta_peak_abs_vertical_acc_vs_zero",
        "warmup_excluded_comfort_peak_abs_vertical_acc",
        "delta_peak_abs_vertical_acc",
    ),
)


def write_scripted_residual_aggregate(
    *,
    output_dir: str,
) -> Tuple[str, str, str, str, str, Dict[str, Any]]:
    """Aggregate scripted residual sensitivity outputs without invoking CARLA."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plan_path = os.path.join(output_dir, "phase4d_scripted_residual_plan.json")
    manifest_path = os.path.join(output_dir, "phase4d_scripted_residual_manifest.json")
    plan = _read_json_dict(plan_path)
    manifest = _read_json_dict(manifest_path)
    stageA_report = _load_stageA_report(plan)
    rows: List[Dict[str, Any]] = []
    zero_failed_seeds: List[str] = []

    for seed_plan in _list_value(plan.get("seed_plans")):
        seed = str(seed_plan.get("eval_seed", ""))
        zero_dir = str(seed_plan.get("zero_reference_output_dir", ""))
        zero_summary = _select_summary_row(
            os.path.join(zero_dir, "suite_summary.csv"),
            SUITE_SCENARIO)
        zero_failed = _scenario_failed_checks(zero_summary)
        if zero_failed:
            zero_failed_seeds.append(seed)
        for residual_plan in _list_value(seed_plan.get("residual_plans")):
            row = _summary_row(
                seed=seed,
                residual_plan=residual_plan,
                zero_dir=zero_dir,
                zero_summary=zero_summary,
                zero_failed=zero_failed,
                stageA_report=stageA_report)
            rows.append(row)
            _write_compare_outputs(row)

    report = _scripted_report(
        output_dir=output_dir,
        plan=plan,
        manifest=manifest,
        rows=rows,
        zero_failed_seeds=zero_failed_seeds,
        stageA_report=stageA_report)
    summary_csv = os.path.join(output_dir, "phase4d_scripted_residual_summary.csv")
    summary_json = os.path.join(output_dir, "phase4d_scripted_residual_summary.json")
    report_json = os.path.join(output_dir, "phase4d_scripted_residual_report.json")
    acceptance_json = os.path.join(output_dir, "phase4d_scripted_residual_acceptance.json")
    acceptance_csv = os.path.join(output_dir, "phase4d_scripted_residual_acceptance.csv")
    _write_rows_csv(summary_csv, rows, SUMMARY_FIELDS)
    _write_json(summary_json, {
        "phase4d_scripted_residual_status": report["phase4d_scripted_residual_status"],
        "rows": rows,
        "counts": report["counts"],
        "diagnosis": report["diagnosis"],
    })
    _write_json(report_json, report)
    _write_json(acceptance_json, report["acceptance"])
    _write_single_row_csv(acceptance_csv, report["acceptance"], ACCEPTANCE_FIELDS)
    return summary_csv, summary_json, acceptance_csv, acceptance_json, report_json, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "aggregate"), default="plan")
    parser.add_argument("--eval-seeds", default="100,101,102")
    parser.add_argument("--routes-subset", default="00")
    parser.add_argument("--baseline", default="skyhook")
    parser.add_argument("--planning-provider", default="empty")
    parser.add_argument("--scripted-residuals", default=",".join(DEFAULT_SCRIPTED_RESIDUALS))
    parser.add_argument("--stageA-report", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-existing-evals", default="false")
    parser.add_argument("--compare", default="true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-preflight", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.action == "aggregate":
        summary_csv, summary_json, acceptance_csv, acceptance_json, report_json, report = (
            write_scripted_residual_aggregate(output_dir=args.output_dir))
        print("phase4d scripted residual summary: %s" % summary_csv)
        print("phase4d scripted residual summary json: %s" % summary_json)
        print("phase4d scripted residual acceptance: %s" % acceptance_csv)
        print("phase4d scripted residual acceptance json: %s" % acceptance_json)
        print("phase4d scripted residual report: %s" % report_json)
        print("phase4d scripted residual status: %s failed_checks=%s" % (
            report["phase4d_scripted_residual_status"],
            report["acceptance"].get("failed_checks", "")))
        return 0 if report["acceptance"].get("aggregate_ok") else 1
    plan_path, acceptance_json, manifest_json, result = write_scripted_residual_plan(
        output_dir=args.output_dir,
        eval_seeds=args.eval_seeds,
        routes_subset=args.routes_subset,
        baseline=args.baseline,
        planning_provider=args.planning_provider,
        scripted_residuals=parse_scripted_residuals(args.scripted_residuals),
        stageA_report_path=args.stageA_report,
        compare=parse_bool(args.compare),
        dry_run=args.dry_run)
    acceptance = result["acceptance"]
    print("phase4d scripted residual plan: %s" % plan_path)
    print("phase4d scripted residual acceptance: %s" % acceptance_json)
    print("phase4d scripted residual manifest: %s" % manifest_json)
    print("phase4d scripted residual status: %s failed_checks=%s" % (
        acceptance.get("phase4d_scripted_residual_status", ""),
        acceptance.get("failed_checks", "")))
    if args.fail_on_preflight and not int(acceptance.get("preflight_ok") or 0):
        return 1
    return 0


def _summary_row(
    *,
    seed: str,
    residual_plan: Mapping[str, Any],
    zero_dir: str,
    zero_summary: Mapping[str, Any],
    zero_failed: Sequence[str],
    stageA_report: Mapping[str, Any],
) -> Dict[str, Any]:
    residual = str(residual_plan.get("scripted_residual", ""))
    residual_dir = str(residual_plan.get("output_dir", ""))
    compare_dir = str(residual_plan.get("compare_output_dir", ""))
    row = _select_summary_row(os.path.join(residual_dir, "suite_summary.csv"), SUITE_SCENARIO)
    failed = _scenario_failed_checks(row)
    warnings = _scenario_warnings(row)
    if zero_failed:
        failed.extend("zero_%s" % item for item in zero_failed)
    metric_effect, noise_metric, noise_std = _metric_effect_above_noise(row, zero_summary, stageA_report)
    zero_missing = any(item == "suite_summary_missing" for item in zero_failed)
    if not row or zero_missing:
        row_status = "incomplete"
    elif failed:
        row_status = "fail"
    elif warnings:
        row_status = "pass_with_warnings"
    else:
        row_status = "pass"
    result: Dict[str, Any] = {
        "eval_seed": seed,
        "scripted_residual": residual,
        "scripted_residual_value": residual_plan.get("scripted_residual_value", ""),
        "planned_scenario": residual_plan.get("scenario", ""),
        "suite_scenario": row.get("scenario", ""),
        "scripted_row_status": row_status,
        "scenario_status": row.get("status", ""),
        "route_return_code": row.get("return_code", ""),
        "global_status": row.get("global_status", ""),
        "score_composed": row.get("score_composed", ""),
        "zero_reference_status": "fail" if zero_failed else "pass",
        "zero_score_composed": zero_summary.get("score_composed", ""),
        "raw_mean_abs_action": _first_nonempty(
            row.get("raw_mean_abs_action"),
            row.get("mean_abs_action"),
            row.get("rl_mean_abs_action")),
        "raw_mean_abs_residual_damper": _first_nonempty(
            row.get("raw_mean_abs_residual_damper"),
            row.get("rl_mean_abs_raw_residual_damper")),
        "scaled_mean_abs_residual_damper": _first_nonempty(
            row.get("scaled_mean_abs_residual_damper"),
            row.get("rl_mean_abs_scaled_residual_damper")),
        "final_mean_abs_residual_damper": _final_residual_value(row),
        "effective_control_ratio": row.get("effective_control_ratio", ""),
        "fallback_ratio": row.get("fallback_ratio", ""),
        "low_speed_mask_ratio": row.get("low_speed_mask_ratio", ""),
        "soft_safety_gain_ratio": row.get("soft_safety_gain_ratio", ""),
        "hard_safety_gate_ratio": row.get("hard_safety_gate_ratio", ""),
        "residual_scale_clip_ratio": row.get("residual_scale_clip_ratio", ""),
        "damper_final_clamp_ratio": row.get("damper_final_clamp_ratio", ""),
        "residual_saturation_ratio": row.get("residual_saturation_ratio", ""),
        "sidecar_runtime_errors": row.get("sidecar_runtime_errors", ""),
        "sidecar_fatal_errors": row.get("sidecar_fatal_errors", ""),
        "sidecar_command_verifies": row.get("sidecar_command_verifies", ""),
        "collision_count": row.get("collision_count", ""),
        "lane_invasion_count": _first_nonempty(
            row.get("lane_invasion_count"),
            row.get("outside_route_lanes")),
        "red_light_count": row.get("red_light_count", ""),
        "route_timeout_count": row.get("route_timeout_count", ""),
        "blocked_vehicle_count": row.get("blocked_vehicle_count", ""),
        "metric_effect_above_noise": _blank_if_none(metric_effect),
        "stageA_noise_metric": noise_metric,
        "stageA_noise_std": _blank_if_none(noise_std),
        "failed_checks": ";".join(_dedupe(failed)),
        "warnings": ";".join(_dedupe(warnings)),
        "residual_output_dir": residual_dir,
        "zero_output_dir": zero_dir,
        "compare_output_dir": compare_dir,
    }
    for output_key, metric_key, _noise_key in METRIC_DELTAS:
        result[output_key] = _metric_delta(row, zero_summary, metric_key)
    return _ordered_row(result, SUMMARY_FIELDS)


def _scripted_report(
    *,
    output_dir: str,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    zero_failed_seeds: Sequence[str],
    stageA_report: Mapping[str, Any],
) -> Dict[str, Any]:
    del stageA_report
    row_statuses = [str(row.get("scripted_row_status", "")) for row in rows]
    failed_rows = [row for row in rows if row.get("scripted_row_status") == "fail"]
    incomplete_rows = [row for row in rows if row.get("scripted_row_status") == "incomplete"]
    warning_rows = [row for row in rows if row.get("scripted_row_status") == "pass_with_warnings"]
    passing_rows = [row for row in rows if row.get("scripted_row_status") in ("pass", "pass_with_warnings")]
    signed_ok = _signed_direction_consistency(rows)
    magnitude_ok = _magnitude_response_consistency(rows)
    effect_above_noise = any(int(_float_value(row.get("metric_effect_above_noise")) or 0) for row in rows)
    safety_limited = bool(zero_failed_seeds or failed_rows)
    failed_checks: List[str] = []
    warnings: List[str] = []
    if not rows or len(passing_rows) == 0 or incomplete_rows:
        failed_checks.append("incomplete_outputs")
        status = STATUS_INCOMPLETE
    elif zero_failed_seeds:
        failed_checks.append("zero_reference_safety")
        status = STATUS_FAIL_SAFETY
    elif failed_rows:
        failed_checks.append("scripted_residual_safety")
        status = STATUS_FAIL_SAFETY
    elif warning_rows or not signed_ok or not magnitude_ok:
        status = STATUS_PASS_WARN
    else:
        status = STATUS_PASS
    if warning_rows:
        warnings.append("scenario_warnings")
    if not signed_ok:
        warnings.append("signed_residual_direction")
    if not magnitude_ok:
        warnings.append("magnitude_response")
    if not effect_above_noise:
        warnings.append("metric_effect_not_above_stageA_noise")
    diagnosis = {
        "SENSITIVITY_DIAGNOSIS": _diagnosis_label(
            status=status,
            effect_above_noise=effect_above_noise,
            signed_ok=signed_ok,
            magnitude_ok=magnitude_ok,
            safety_limited=safety_limited),
        "SAFETY_AUTHORITY_STATUS": "safety_limited" if safety_limited else "safe",
        "scripted_effect_above_noise": int(effect_above_noise),
        "signed_residual_direction_consistency": int(signed_ok),
        "magnitude_response_consistency": int(magnitude_ok),
        "recommended_next_action": _recommended_next_action(
            status=status,
            effect_above_noise=effect_above_noise,
            signed_ok=signed_ok,
            magnitude_ok=magnitude_ok,
            safety_limited=safety_limited),
    }
    counts = {
        "eval_seed_count": len(_list_value(plan.get("eval_seeds"))),
        "scripted_residual_count": len(_list_value(plan.get("scripted_residuals"))),
        "completed_scenario_count": len([status for status in row_statuses if status != "incomplete"]),
        "passing_scenario_count": len(passing_rows),
        "warning_scenario_count": len(warning_rows),
        "failed_scenario_count": len(failed_rows),
        "incomplete_scenario_count": len(incomplete_rows),
        "zero_failed_seed_count": len(zero_failed_seeds),
    }
    acceptance = _ordered_row({
        "phase4d_scripted_residual_status": status,
        "aggregate_ok": int(status in (STATUS_PASS, STATUS_PASS_WARN)),
        "failed_checks": ";".join(_dedupe(failed_checks)),
        "warnings": ";".join(_dedupe(warnings)),
        "workflow_role": "phase4d_scripted_residual_sensitivity",
        **counts,
        **diagnosis,
        "summary_csv": os.path.join(output_dir, "phase4d_scripted_residual_summary.csv"),
        "summary_json": os.path.join(output_dir, "phase4d_scripted_residual_summary.json"),
        "report_json": os.path.join(output_dir, "phase4d_scripted_residual_report.json"),
        "manifest_json": os.path.join(output_dir, "phase4d_scripted_residual_manifest.json"),
        "output_dir": output_dir,
    }, ACCEPTANCE_FIELDS)
    return {
        "phase4d_scripted_residual_status": status,
        "aggregate_ok": acceptance["aggregate_ok"],
        "failed_checks": _dedupe(failed_checks),
        "warnings": _dedupe(warnings),
        "counts": counts,
        "diagnosis": diagnosis,
        "acceptance": acceptance,
        "rows": list(rows),
        "manifest": manifest,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_claim_warning": (
            "This scripted residual sweep is an eval-only sensitivity "
            "diagnostic, not a paper-level performance claim."),
    }


def _write_compare_outputs(row: Mapping[str, Any]) -> None:
    compare_dir = str(row.get("compare_output_dir", ""))
    if not compare_dir:
        return
    os.makedirs(compare_dir, exist_ok=True)
    compare = {
        "eval_seed": row.get("eval_seed", ""),
        "scripted_residual": row.get("scripted_residual", ""),
        "scripted_residual_value": row.get("scripted_residual_value", ""),
        "planned_scenario": row.get("planned_scenario", ""),
        "zero_reference_status": row.get("zero_reference_status", ""),
        "scripted_row_status": row.get("scripted_row_status", ""),
        "failed_checks": row.get("failed_checks", ""),
        "warnings": row.get("warnings", ""),
        "metric_effect_above_noise": row.get("metric_effect_above_noise", ""),
        "delta_comfort_score_vs_zero": row.get("delta_comfort_score_vs_zero", ""),
        "delta_rms_vertical_acc_vs_zero": row.get("delta_rms_vertical_acc_vs_zero", ""),
        "delta_rms_vertical_jerk_vs_zero": row.get("delta_rms_vertical_jerk_vs_zero", ""),
        "delta_peak_abs_vertical_acc_vs_zero": row.get("delta_peak_abs_vertical_acc_vs_zero", ""),
    }
    _write_json(os.path.join(compare_dir, "phase4d_scripted_residual_compare.json"), compare)
    _write_single_row_csv(
        os.path.join(compare_dir, "phase4d_scripted_residual_compare.csv"),
        compare,
        tuple(compare.keys()))


def _scenario_failed_checks(row: Mapping[str, Any]) -> List[str]:
    failed: List[str] = []
    if not row:
        return ["suite_summary_missing"]
    if str(row.get("status", "")) != "ok":
        failed.append("scenario_status")
    if _int_value(row.get("return_code")) != 0:
        failed.append("route_return_code")
    score = _float_value(row.get("score_composed"))
    if score is None or score < 100.0:
        failed.append("score_composed")
    for key in (
            "collision_count",
            "lane_invasion_count",
            "red_light_count",
            "route_timeout_count",
            "blocked_vehicle_count",
            "sidecar_fatal_errors"):
        value = _float_value(row.get(key))
        if value is not None and value > 0.0:
            failed.append(key)
    return _dedupe(failed)


def _scenario_warnings(row: Mapping[str, Any]) -> List[str]:
    warnings: List[str] = []
    if not row:
        return warnings
    for key in (
            "sidecar_runtime_errors",
            "fallback_ratio",
            "low_speed_mask_ratio",
            "soft_safety_gain_ratio",
            "damper_final_clamp_ratio",
            "residual_scale_clip_ratio"):
        value = _float_value(row.get(key))
        if value is not None and value > 0.0:
            warnings.append(key)
    return _dedupe(warnings)


def _metric_effect_above_noise(
    row: Mapping[str, Any],
    zero_row: Mapping[str, Any],
    stageA_report: Mapping[str, Any],
) -> Tuple[Optional[int], str, Optional[float]]:
    metrics = stageA_report.get("metrics") if isinstance(stageA_report, Mapping) else {}
    if not isinstance(metrics, Mapping) or not metrics:
        return None, "", None
    best_metric = ""
    best_std: Optional[float] = None
    for _output_key, metric_key, noise_key in METRIC_DELTAS:
        delta = _metric_delta(row, zero_row, metric_key)
        noise = metrics.get(noise_key)
        if delta in (None, "") or not isinstance(noise, Mapping):
            continue
        std = _float_value(noise.get("std"))
        delta_value = _float_value(delta)
        if std is None or delta_value is None:
            continue
        threshold = max(2.0 * std, 1.0e-12)
        best_metric = noise_key
        best_std = std
        if abs(delta_value) > threshold:
            return 1, noise_key, std
    if best_metric:
        return 0, best_metric, best_std
    return None, "", None


def _signed_direction_consistency(rows: Sequence[Mapping[str, Any]]) -> bool:
    pairs = (("const_p0p05", "const_m0p05"), ("const_p0p10", "const_m0p10"))
    for positive, negative in pairs:
        pos_values = _metric_values_for_residual(rows, positive, "delta_rms_vertical_acc_vs_zero")
        neg_values = _metric_values_for_residual(rows, negative, "delta_rms_vertical_acc_vs_zero")
        if not pos_values or not neg_values:
            continue
        pos_mean = sum(pos_values) / float(len(pos_values))
        neg_mean = sum(neg_values) / float(len(neg_values))
        if pos_mean == 0.0 or neg_mean == 0.0:
            continue
        if (pos_mean > 0.0 > neg_mean) or (pos_mean < 0.0 < neg_mean):
            return True
    return False


def _magnitude_response_consistency(rows: Sequence[Mapping[str, Any]]) -> bool:
    for sign in ("p", "m"):
        small = _metric_values_for_residual(rows, "const_%s0p05" % sign, "delta_rms_vertical_acc_vs_zero")
        large = _metric_values_for_residual(rows, "const_%s0p10" % sign, "delta_rms_vertical_acc_vs_zero")
        if not small or not large:
            continue
        small_mean = sum(abs(value) for value in small) / float(len(small))
        large_mean = sum(abs(value) for value in large) / float(len(large))
        if large_mean + 1.0e-12 >= small_mean:
            return True
    return False


def _metric_values_for_residual(
    rows: Sequence[Mapping[str, Any]],
    residual: str,
    key: str,
) -> List[float]:
    values: List[float] = []
    for row in rows:
        if row.get("scripted_residual") != residual:
            continue
        value = _float_value(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _recommended_next_action(
    *,
    status: str,
    effect_above_noise: bool,
    signed_ok: bool,
    magnitude_ok: bool,
    safety_limited: bool,
) -> str:
    if status == STATUS_INCOMPLETE:
        return "rerun_missing_scripted_residual_scenarios"
    if safety_limited:
        return "inspect_safety_before_increasing_scripted_residual_authority"
    if not effect_above_noise:
        return "scripted_effect_is_within_stageA_noise_floor"
    if not signed_ok:
        return "inspect_metric_direction_or_sign_semantics"
    if not magnitude_ok:
        return "inspect_residual_magnitude_response_or_clamp_limits"
    return "use_sensitivity_result_to_choose_next_policy_authority_test"


def _diagnosis_label(
    *,
    status: str,
    effect_above_noise: bool,
    signed_ok: bool,
    magnitude_ok: bool,
    safety_limited: bool,
) -> str:
    if status == STATUS_INCOMPLETE:
        return STATUS_INCOMPLETE
    if safety_limited:
        return "SAFETY_LIMITED_SCRIPTED_RESIDUAL"
    if not effect_above_noise:
        return "SCRIPTED_EFFECT_WITHIN_NOISE"
    if not signed_ok:
        return "SIGNED_DIRECTION_INCONSISTENT"
    if not magnitude_ok:
        return "MAGNITUDE_RESPONSE_INCONSISTENT"
    return "SCRIPTED_SENSITIVITY_DETECTED"


def _load_stageA_report(plan: Mapping[str, Any]) -> Dict[str, Any]:
    path = str(plan.get("stageA_report_path", ""))
    if not path:
        return {}
    return _read_json_dict(path)


def _select_summary_row(path: str, scenario: str) -> Dict[str, Any]:
    rows = _read_csv_rows(path)
    for row in rows:
        if row.get("scenario") == scenario:
            return row
    return rows[0] if rows else {}


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    path = _resolve_workspace_path(path)
    if not path or not os.path.isfile(path):
        return []
    with open(path, "r", newline="") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def _read_json_dict(path: str) -> Dict[str, Any]:
    path = _resolve_workspace_path(path)
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_rows_csv(path: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_value(row.get(field, "")) for field in fields})


def _write_single_row_csv(path: str, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    _write_rows_csv(path, [row], fields)


def _write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write("\n")


def _resolve_workspace_path(path: str) -> str:
    if not path:
        return ""
    candidates = [path]
    if path.startswith("/workspace/"):
        candidates.append("/home/yth/sim/" + path[len("/workspace/"):])
    elif path.startswith("/home/yth/sim/"):
        candidates.append("/workspace/" + path[len("/home/yth/sim/"):])
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path


def _metric_delta(row: Mapping[str, Any], zero_row: Mapping[str, Any], key: str) -> Any:
    value = _float_value(row.get(key))
    zero = _float_value(zero_row.get(key))
    if value is None or zero is None:
        return ""
    return value - zero


def _final_residual_value(row: Mapping[str, Any]) -> Any:
    return _first_nonempty(
        row.get("final_mean_abs_residual_damper"),
        row.get("rl_mean_abs_final_residual_damper"),
        row.get("mean_abs_residual_damper"),
        row.get("rl_mean_abs_residual_damper"))


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def _list_value(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


def _float_value(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _int_value(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        return "%0.9g" % value
    return value


def _ordered_row(values: Mapping[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    row = {field: values.get(field, "") for field in fields}
    for key, value in values.items():
        if key not in row:
            row[key] = value
    return row


def _dedupe(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

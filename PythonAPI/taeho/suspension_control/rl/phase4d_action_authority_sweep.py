"""CARLA-free planning and aggregation CLI for Phase 4-D authority sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .phase4d_authority_sensitivity_common import (
    DEFAULT_ACTION_SCALES,
    parse_action_scales,
    parse_bool,
    write_action_authority_plan,
)


STATUS_PASS = "AUTHORITY_SWEEP_PASS"
STATUS_PASS_WARN = "AUTHORITY_SWEEP_PASS_WITH_WARNINGS"
STATUS_FAIL_SAFETY = "AUTHORITY_SWEEP_FAIL_SAFETY"
STATUS_FAIL_PREFLIGHT = "AUTHORITY_SWEEP_FAIL_PREFLIGHT"
STATUS_INCOMPLETE = "AUTHORITY_SWEEP_INCOMPLETE"

S4_SCENARIO = "S4_rl_residual_skyhook"
S8_SCENARIO = "S8_rl_zero_residual_skyhook"

AUTHORITY_SUMMARY_FIELDS: Tuple[str, ...] = (
    "eval_seed",
    "action_scale",
    "residual_gain",
    "action_scale_label",
    "planned_scenario",
    "suite_scenario",
    "authority_row_status",
    "scenario_status",
    "route_return_code",
    "global_status",
    "score_composed",
    "s8_reference_status",
    "s8_score_composed",
    "compare_status",
    "policy_selection_status",
    "policy_available_ratio",
    "mean_abs_action",
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
    "delta_comfort_score",
    "delta_rms_vertical_acc",
    "delta_rms_vertical_jerk",
    "delta_peak_abs_vertical_acc",
    "metric_effect_above_noise",
    "stageA_noise_metric",
    "stageA_noise_std",
    "failed_checks",
    "warnings",
    "s4_output_dir",
    "s8_output_dir",
    "compare_output_dir",
)

AGGREGATE_ACCEPTANCE_FIELDS: Tuple[str, ...] = (
    "phase4d_action_authority_status",
    "aggregate_ok",
    "failed_checks",
    "warnings",
    "workflow_role",
    "source_train_artifact_id",
    "eval_seed_count",
    "action_scale_count",
    "completed_scenario_count",
    "passing_scenario_count",
    "warning_scenario_count",
    "failed_scenario_count",
    "incomplete_scenario_count",
    "s8_failed_seed_count",
    "scale_1p0_failed_count",
    "authority_monotonicity_ok",
    "metric_effect_above_noise",
    "safety_limited_gain",
    "clamp_limited_gain",
    "recommended_next_action",
    "summary_csv",
    "summary_json",
    "report_json",
    "manifest_json",
    "output_dir",
)

COMPARISON_DELTA_KEYS: Tuple[Tuple[str, str, str], ...] = (
    (
        "delta_comfort_score",
        "warmup_excluded_comfort_comfort_score_delta_s4_minus_s8",
        "delta_comfort_score",
    ),
    (
        "delta_rms_vertical_acc",
        "warmup_excluded_comfort_rms_vertical_acc_delta_s4_minus_s8",
        "delta_rms_vertical_acc",
    ),
    (
        "delta_rms_vertical_jerk",
        "warmup_excluded_comfort_rms_vertical_jerk_delta_s4_minus_s8",
        "delta_rms_vertical_jerk",
    ),
    (
        "delta_peak_abs_vertical_acc",
        "warmup_excluded_comfort_peak_abs_vertical_acc_delta_s4_minus_s8",
        "delta_peak_abs_vertical_acc",
    ),
)


def write_action_authority_aggregate(
    *,
    output_dir: str,
) -> Tuple[str, str, str, str, str, Dict[str, Any]]:
    """Aggregate a completed authority sweep without invoking CARLA."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plan_path = os.path.join(output_dir, "phase4d_action_authority_plan.json")
    manifest_path = os.path.join(output_dir, "phase4d_action_authority_manifest.json")
    plan = _read_json_dict(plan_path)
    manifest = _read_json_dict(manifest_path)
    stageA_report = _load_stageA_report(plan)

    rows: List[Dict[str, Any]] = []
    s8_failed_seeds: List[str] = []
    for seed_plan in _list_value(plan.get("seed_plans")):
        seed = str(seed_plan.get("eval_seed", ""))
        s8_dir = str(seed_plan.get("s8_reference_output_dir", ""))
        s8_summary = _select_summary_row(
            os.path.join(s8_dir, "suite_summary.csv"),
            S8_SCENARIO)
        s8_acceptance = _read_json_dict(
            os.path.join(s8_dir, "phase4d_eval_s8_acceptance.json"))
        s8_failed = _reference_failed_checks(s8_summary, s8_acceptance)
        if s8_failed:
            s8_failed_seeds.append(seed)
        for gain_plan in _list_value(seed_plan.get("action_scale_plans")):
            rows.append(_authority_summary_row(
                seed=seed,
                gain_plan=gain_plan,
                s8_dir=s8_dir,
                s8_summary=s8_summary,
                s8_acceptance=s8_acceptance,
                s8_failed=s8_failed,
                stageA_report=stageA_report))

    report = _authority_report(
        output_dir=output_dir,
        plan=plan,
        manifest=manifest,
        rows=rows,
        s8_failed_seeds=s8_failed_seeds,
        stageA_report=stageA_report)
    summary_csv = os.path.join(output_dir, "phase4d_action_authority_summary.csv")
    summary_json = os.path.join(output_dir, "phase4d_action_authority_summary.json")
    report_json = os.path.join(output_dir, "phase4d_action_authority_report.json")
    acceptance_json = os.path.join(output_dir, "phase4d_action_authority_acceptance.json")
    acceptance_csv = os.path.join(output_dir, "phase4d_action_authority_acceptance.csv")
    _write_rows_csv(summary_csv, rows, AUTHORITY_SUMMARY_FIELDS)
    _write_json(summary_json, {
        "phase4d_action_authority_status": report["phase4d_action_authority_status"],
        "rows": rows,
        "counts": report["counts"],
        "diagnosis": report["diagnosis"],
    })
    _write_json(report_json, report)
    _write_json(acceptance_json, report["acceptance"])
    _write_single_row_csv(acceptance_csv, report["acceptance"], AGGREGATE_ACCEPTANCE_FIELDS)
    return summary_csv, summary_json, acceptance_csv, acceptance_json, report_json, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "aggregate"), default="plan")
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--train-dir", default="")
    parser.add_argument("--eval-seeds", default="100,101,102")
    parser.add_argument("--routes-subset", default="00")
    parser.add_argument("--baseline", default="skyhook")
    parser.add_argument("--planning-provider", default="empty")
    parser.add_argument("--action-scales", default="")
    parser.add_argument("--residual-gains", default="")
    parser.add_argument("--stageA-report", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reuse-existing-evals", default="false")
    parser.add_argument("--run-s8-reference", default="true")
    parser.add_argument("--compare", default="true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-preflight", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.action == "aggregate":
        summary_csv, summary_json, acceptance_csv, acceptance_json, report_json, report = (
            write_action_authority_aggregate(output_dir=args.output_dir))
        print("phase4d action authority summary: %s" % summary_csv)
        print("phase4d action authority summary json: %s" % summary_json)
        print("phase4d action authority acceptance: %s" % acceptance_csv)
        print("phase4d action authority acceptance json: %s" % acceptance_json)
        print("phase4d action authority report: %s" % report_json)
        print("phase4d action authority status: %s failed_checks=%s" % (
            report["phase4d_action_authority_status"],
            report["acceptance"].get("failed_checks", "")))
        return 0 if report["acceptance"].get("aggregate_ok") else 1
    if not args.train_manifest:
        raise SystemExit("--action plan requires --train-manifest")
    if not args.train_dir:
        raise SystemExit("--action plan requires --train-dir")
    action_scale_text = (
        args.action_scales or
        args.residual_gains or
        ",".join(str(value) for value in DEFAULT_ACTION_SCALES))
    plan_path, acceptance_json, manifest_json, result = write_action_authority_plan(
        output_dir=args.output_dir,
        train_manifest_path=args.train_manifest,
        train_dir=args.train_dir,
        eval_seeds=args.eval_seeds,
        routes_subset=args.routes_subset,
        baseline=args.baseline,
        planning_provider=args.planning_provider,
        action_scales=parse_action_scales(action_scale_text),
        stageA_report_path=args.stageA_report,
        run_s8_reference=parse_bool(args.run_s8_reference),
        compare=parse_bool(args.compare),
        dry_run=args.dry_run)
    acceptance = result["acceptance"]
    print("phase4d action authority plan: %s" % plan_path)
    print("phase4d action authority acceptance: %s" % acceptance_json)
    print("phase4d action authority manifest: %s" % manifest_json)
    print("phase4d action authority status: %s failed_checks=%s" % (
        acceptance.get("phase4d_action_authority_status", ""),
        acceptance.get("failed_checks", "")))
    if args.fail_on_preflight and not int(acceptance.get("preflight_ok") or 0):
        return 1
    return 0


def _authority_summary_row(
    *,
    seed: str,
    gain_plan: Mapping[str, Any],
    s8_dir: str,
    s8_summary: Mapping[str, Any],
    s8_acceptance: Mapping[str, Any],
    s8_failed: Sequence[str],
    stageA_report: Mapping[str, Any],
) -> Dict[str, Any]:
    del s8_acceptance
    scale = _float_value(gain_plan.get("action_scale"))
    s4_dir = str(gain_plan.get("output_dir", ""))
    compare_dir = str(gain_plan.get("compare_output_dir", ""))
    s4_summary = _select_summary_row(os.path.join(s4_dir, "suite_summary.csv"), S4_SCENARIO)
    s4_acceptance = _read_json_dict(os.path.join(s4_dir, "phase4d_eval_s4_acceptance.json"))
    compare = _read_json_dict(os.path.join(compare_dir, "phase4d_split_comparison.json"))
    s4_failed = _scenario_failed_checks(s4_summary, s4_acceptance, scale)
    warnings = _scenario_warnings(s4_summary, scale)
    if s8_failed:
        s4_failed.extend("s8_%s" % item for item in s8_failed)
    metric_effect, noise_metric, noise_std = _metric_effect_above_noise(compare, stageA_report)
    for output_key, compare_key, _noise_key in COMPARISON_DELTA_KEYS:
        del _noise_key
        if output_key not in compare:
            continue
    if not s4_summary:
        row_status = "incomplete"
    elif s4_failed:
        row_status = "fail"
    elif warnings:
        row_status = "pass_with_warnings"
    else:
        row_status = "pass"
    row: Dict[str, Any] = {
        "eval_seed": seed,
        "action_scale": _blank_if_none(scale),
        "residual_gain": _blank_if_none(scale),
        "action_scale_label": gain_plan.get("action_scale_label", ""),
        "planned_scenario": gain_plan.get("scenario", ""),
        "suite_scenario": s4_summary.get("scenario", ""),
        "authority_row_status": row_status,
        "scenario_status": s4_summary.get("status", ""),
        "route_return_code": _first_nonempty(
            s4_acceptance.get("route_return_code"),
            s4_summary.get("return_code")),
        "global_status": s4_summary.get("global_status", ""),
        "score_composed": s4_summary.get("score_composed", ""),
        "s8_reference_status": "fail" if s8_failed else "pass",
        "s8_score_composed": s8_summary.get("score_composed", ""),
        "compare_status": compare.get("phase4d_compare_status", ""),
        "policy_selection_status": compare.get("policy_selection_status", ""),
        "policy_available_ratio": _first_nonempty(
            s4_summary.get("policy_available_ratio"),
            s4_summary.get("rl_policy_available_ratio")),
        "mean_abs_action": _first_nonempty(
            s4_summary.get("raw_mean_abs_action"),
            s4_summary.get("mean_abs_action"),
            s4_summary.get("rl_mean_abs_action")),
        "raw_mean_abs_residual_damper": _first_nonempty(
            s4_summary.get("raw_mean_abs_residual_damper"),
            s4_summary.get("rl_mean_abs_raw_residual_damper")),
        "scaled_mean_abs_residual_damper": _first_nonempty(
            s4_summary.get("scaled_mean_abs_residual_damper"),
            s4_summary.get("rl_mean_abs_scaled_residual_damper")),
        "final_mean_abs_residual_damper": _final_residual_value(s4_summary),
        "effective_control_ratio": s4_summary.get("effective_control_ratio", ""),
        "fallback_ratio": s4_summary.get("fallback_ratio", ""),
        "low_speed_mask_ratio": s4_summary.get("low_speed_mask_ratio", ""),
        "soft_safety_gain_ratio": s4_summary.get("soft_safety_gain_ratio", ""),
        "hard_safety_gate_ratio": s4_summary.get("hard_safety_gate_ratio", ""),
        "residual_scale_clip_ratio": s4_summary.get("residual_scale_clip_ratio", ""),
        "damper_final_clamp_ratio": s4_summary.get("damper_final_clamp_ratio", ""),
        "residual_saturation_ratio": s4_summary.get("residual_saturation_ratio", ""),
        "sidecar_runtime_errors": s4_summary.get("sidecar_runtime_errors", ""),
        "sidecar_fatal_errors": s4_summary.get("sidecar_fatal_errors", ""),
        "sidecar_command_verifies": s4_summary.get("sidecar_command_verifies", ""),
        "collision_count": s4_summary.get("collision_count", ""),
        "lane_invasion_count": _first_nonempty(
            s4_summary.get("lane_invasion_count"),
            s4_summary.get("outside_route_lanes")),
        "red_light_count": s4_summary.get("red_light_count", ""),
        "route_timeout_count": s4_summary.get("route_timeout_count", ""),
        "blocked_vehicle_count": s4_summary.get("blocked_vehicle_count", ""),
        "metric_effect_above_noise": _blank_if_none(metric_effect),
        "stageA_noise_metric": noise_metric,
        "stageA_noise_std": _blank_if_none(noise_std),
        "failed_checks": ";".join(_dedupe(s4_failed)),
        "warnings": ";".join(_dedupe(warnings)),
        "s4_output_dir": s4_dir,
        "s8_output_dir": s8_dir,
        "compare_output_dir": compare_dir,
    }
    for output_key, compare_key, _noise_key in COMPARISON_DELTA_KEYS:
        row[output_key] = compare.get(compare_key, "")
    return _ordered_row(row, AUTHORITY_SUMMARY_FIELDS)


def _authority_report(
    *,
    output_dir: str,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    s8_failed_seeds: Sequence[str],
    stageA_report: Mapping[str, Any],
) -> Dict[str, Any]:
    preflight_failed = str(plan.get("phase4d_action_authority_status", "")) == "preflight_failed"
    row_statuses = [str(row.get("authority_row_status", "")) for row in rows]
    failed_rows = [row for row in rows if row.get("authority_row_status") == "fail"]
    incomplete_rows = [row for row in rows if row.get("authority_row_status") == "incomplete"]
    warning_rows = [row for row in rows if row.get("authority_row_status") == "pass_with_warnings"]
    passing_rows = [row for row in rows if row.get("authority_row_status") in ("pass", "pass_with_warnings")]
    scale_1p0_failed = [
        row for row in rows
        if abs((_float_value(row.get("action_scale")) or 0.0) - 1.0) <= 1.0e-9
        and row.get("authority_row_status") == "fail"
    ]
    safety_limited = bool(s8_failed_seeds or failed_rows or _any_positive(rows, "hard_safety_gate_ratio"))
    clamp_limited = bool(
        _any_positive(rows, "residual_scale_clip_ratio") or
        _any_positive(rows, "damper_final_clamp_ratio") or
        _any_positive(rows, "residual_saturation_ratio"))
    monotonic_ok = _residual_monotonicity_ok(rows)
    metric_effect = any(int(_float_value(row.get("metric_effect_above_noise")) or 0) for row in rows)
    failed_checks: List[str] = []
    warnings: List[str] = []
    if preflight_failed:
        failed_checks.append("preflight")
        status = STATUS_FAIL_PREFLIGHT
    elif s8_failed_seeds:
        failed_checks.append("s8_reference_safety")
        status = STATUS_FAIL_SAFETY
    elif scale_1p0_failed:
        failed_checks.append("action_scale_1p0_safety")
        status = STATUS_FAIL_SAFETY
    elif failed_rows:
        failed_checks.append("action_scale_safety")
        status = STATUS_FAIL_SAFETY
    elif not rows or len(passing_rows) == 0 or incomplete_rows:
        failed_checks.append("incomplete_outputs")
        status = STATUS_INCOMPLETE
    elif warning_rows or not monotonic_ok or clamp_limited:
        status = STATUS_PASS_WARN
    else:
        status = STATUS_PASS
    if warning_rows:
        warnings.append("scenario_warnings")
    if not monotonic_ok:
        warnings.append("authority_monotonicity")
    if clamp_limited:
        warnings.append("clamp_or_saturation_limited")
    if not metric_effect:
        warnings.append("metric_effect_not_above_stageA_noise")

    recommended = _recommended_next_action(
        status=status,
        monotonic_ok=monotonic_ok,
        metric_effect=metric_effect,
        safety_limited=safety_limited,
        clamp_limited=clamp_limited)
    diagnosis = {
        "AUTHORITY_DIAGNOSIS": _diagnosis_label(
            status=status,
            monotonic_ok=monotonic_ok,
            metric_effect=metric_effect,
            safety_limited=safety_limited,
            clamp_limited=clamp_limited),
        "authority_monotonicity_ok": int(monotonic_ok),
        "metric_effect_above_noise": int(metric_effect),
        "safety_limited_gain": int(safety_limited),
        "clamp_limited_gain": int(clamp_limited),
        "recommended_next_action": recommended,
    }
    counts = {
        "eval_seed_count": len(_list_value(plan.get("eval_seeds"))),
        "action_scale_count": len(_list_value(plan.get("action_scales"))),
        "completed_scenario_count": len([status for status in row_statuses if status != "incomplete"]),
        "passing_scenario_count": len(passing_rows),
        "warning_scenario_count": len(warning_rows),
        "failed_scenario_count": len(failed_rows),
        "incomplete_scenario_count": len(incomplete_rows),
        "s8_failed_seed_count": len(s8_failed_seeds),
        "scale_1p0_failed_count": len(scale_1p0_failed),
    }
    acceptance = _ordered_row({
        "phase4d_action_authority_status": status,
        "aggregate_ok": int(status in (STATUS_PASS, STATUS_PASS_WARN)),
        "failed_checks": ";".join(_dedupe(failed_checks)),
        "warnings": ";".join(_dedupe(warnings)),
        "workflow_role": "phase4d_action_authority_sweep",
        "source_train_artifact_id": _first_nonempty(
            manifest.get("source_train_artifact_id"),
            plan.get("source_train_artifact_id")),
        **counts,
        **diagnosis,
        "summary_csv": os.path.join(output_dir, "phase4d_action_authority_summary.csv"),
        "summary_json": os.path.join(output_dir, "phase4d_action_authority_summary.json"),
        "report_json": os.path.join(output_dir, "phase4d_action_authority_report.json"),
        "manifest_json": os.path.join(output_dir, "phase4d_action_authority_manifest.json"),
        "output_dir": output_dir,
    }, AGGREGATE_ACCEPTANCE_FIELDS)
    return {
        "phase4d_action_authority_status": status,
        "aggregate_ok": acceptance["aggregate_ok"],
        "failed_checks": _dedupe(failed_checks),
        "warnings": _dedupe(warnings),
        "counts": counts,
        "diagnosis": diagnosis,
        "acceptance": acceptance,
        "rows": list(rows),
        "stageA_report_loaded": int(bool(stageA_report)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_claim_warning": (
            "This authority sweep is an eval-only diagnostic, not a "
            "paper-level performance claim."),
    }


def _reference_failed_checks(
    row: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> List[str]:
    failed: List[str] = []
    if not row:
        failed.append("s8_summary_missing")
    if acceptance and str(acceptance.get("phase4d_eval_s8_status", "")) != "pass":
        failed.append("s8_acceptance_status")
    failed.extend(_route_failed_checks(row, require_policy=False))
    return _dedupe(failed)


def _scenario_failed_checks(
    row: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    action_scale: Optional[float],
) -> List[str]:
    failed: List[str] = []
    if not row:
        failed.append("suite_summary_missing")
        return failed
    if acceptance and str(acceptance.get("phase4d_eval_s4_status", "")) != "pass":
        failed.append("s4_acceptance_status")
    failed.extend(_route_failed_checks(row, require_policy=True))
    final_residual = _float_value(_final_residual_value(row))
    if action_scale is not None and abs(action_scale) <= 1.0e-12:
        if final_residual is not None and final_residual > 1.0e-6:
            failed.append("action_scale_zero_final_residual_nonzero")
    return _dedupe(failed)


def _route_failed_checks(row: Mapping[str, Any], *, require_policy: bool) -> List[str]:
    failed: List[str] = []
    if not row:
        return ["summary_missing"]
    if str(row.get("status", "")) != "ok":
        failed.append("scenario_status")
    if _int_value(row.get("return_code")) != 0:
        failed.append("route_return_code")
    if row.get("global_status") not in ("", "Perfect"):
        failed.append("global_status")
    for key in (
            "collision_count",
            "lane_invasion_count",
            "red_light_count",
            "route_timeout_count",
            "blocked_vehicle_count",
            "sidecar_fatal_errors"):
        if _float_value(row.get(key)) and (_float_value(row.get(key)) or 0.0) > 0.0:
            failed.append(key)
    if require_policy:
        policy_ratio = _float_value(_first_nonempty(
            row.get("policy_available_ratio"),
            row.get("rl_policy_available_ratio")))
        if policy_ratio is None or policy_ratio < 0.99:
            failed.append("policy_available_ratio")
    return _dedupe(failed)


def _scenario_warnings(row: Mapping[str, Any], action_scale: Optional[float]) -> List[str]:
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
    final_residual = _float_value(_final_residual_value(row))
    if action_scale is not None and action_scale > 0.0 and final_residual is not None and final_residual <= 0.0:
        warnings.append("final_residual_zero")
    return _dedupe(warnings)


def _metric_effect_above_noise(
    compare: Mapping[str, Any],
    stageA_report: Mapping[str, Any],
) -> Tuple[Optional[int], str, Optional[float]]:
    metrics = stageA_report.get("metrics") if isinstance(stageA_report, Mapping) else {}
    if not isinstance(metrics, Mapping) or not metrics:
        return None, "", None
    best_metric = ""
    best_std: Optional[float] = None
    for _output_key, compare_key, noise_key in COMPARISON_DELTA_KEYS:
        delta = _float_value(compare.get(compare_key))
        noise = metrics.get(noise_key)
        if delta is None or not isinstance(noise, Mapping):
            continue
        std = _float_value(noise.get("std"))
        if std is None:
            continue
        threshold = max(2.0 * std, 1.0e-12)
        best_metric = noise_key
        best_std = std
        if abs(delta) > threshold:
            return 1, noise_key, std
    if best_metric:
        return 0, best_metric, best_std
    return None, "", None


def _residual_monotonicity_ok(rows: Sequence[Mapping[str, Any]]) -> bool:
    by_scale: Dict[float, List[float]] = {}
    for row in rows:
        if row.get("authority_row_status") not in ("pass", "pass_with_warnings"):
            continue
        scale = _float_value(row.get("action_scale"))
        residual = _float_value(row.get("final_mean_abs_residual_damper"))
        if scale is None or residual is None:
            continue
        by_scale.setdefault(scale, []).append(residual)
    if len(by_scale) < 2:
        return False
    previous: Optional[float] = None
    for scale in sorted(by_scale):
        values = by_scale[scale]
        mean_value = sum(values) / float(len(values))
        if previous is not None and mean_value + 1.0e-9 < previous:
            return False
        previous = mean_value
    return True


def _recommended_next_action(
    *,
    status: str,
    monotonic_ok: bool,
    metric_effect: bool,
    safety_limited: bool,
    clamp_limited: bool,
) -> str:
    if status == STATUS_FAIL_PREFLIGHT:
        return "fix_preflight_inputs_before_running_authority_sweep"
    if status == STATUS_INCOMPLETE:
        return "rerun_missing_authority_sweep_scenarios"
    if safety_limited:
        return "inspect_safety_limited_gains_before_increasing_authority"
    if not monotonic_ok:
        return "inspect_residual_runtime_path_or_summary_fields"
    if clamp_limited:
        return "inspect_residual_clamp_and_final_damper_bounds"
    if not metric_effect:
        return "treat_policy_effect_as_within_stageA_noise_floor"
    return "use_best_safe_gain_for_next_repeat_validation"


def _diagnosis_label(
    *,
    status: str,
    monotonic_ok: bool,
    metric_effect: bool,
    safety_limited: bool,
    clamp_limited: bool,
) -> str:
    if status in (STATUS_FAIL_PREFLIGHT, STATUS_INCOMPLETE):
        return status
    if safety_limited:
        return "SAFETY_LIMITED_AUTHORITY"
    if not monotonic_ok:
        return "RESIDUAL_AUTHORITY_NOT_MONOTONIC"
    if clamp_limited:
        return "CLAMP_LIMITED_AUTHORITY"
    if metric_effect:
        return "AUTHORITY_EFFECT_ABOVE_NOISE"
    return "AUTHORITY_EFFECT_WITHIN_NOISE"


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


def _final_residual_value(row: Mapping[str, Any]) -> Any:
    return _first_nonempty(
        row.get("final_mean_abs_residual_damper"),
        row.get("rl_mean_abs_final_residual_damper"),
        row.get("mean_abs_residual_damper"),
        row.get("rl_mean_abs_residual_damper"))


def _any_positive(rows: Sequence[Mapping[str, Any]], key: str) -> bool:
    for row in rows:
        value = _float_value(row.get(key))
        if value is not None and value > 0.0:
            return True
    return False


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

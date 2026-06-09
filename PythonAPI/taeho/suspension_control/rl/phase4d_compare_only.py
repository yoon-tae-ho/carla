"""CARLA-free Phase 4-D S4-vs-S8 artifact comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PAPER_WARNING = (
    "This is an engineering/canary result only. Paper-level performance claims "
    "require repeated train seeds, eval seeds, and route subsets."
)

S4_SCENARIO = "S4_rl_residual_skyhook"
S8_SCENARIO = "S8_rl_zero_residual_skyhook"


METRICS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("score_route", ("score_route",), "higher"),
    ("score_composed", ("score_composed",), "higher"),
    ("score_penalty", ("score_penalty",), "higher"),
    ("mean_reward_total", ("mean_reward_total",), "higher"),
    ("mean_reward_comfort", ("mean_reward_comfort", "reward_comfort_mean"), "higher"),
    ("mean_reward_stability", ("mean_reward_stability", "reward_stability_mean"), "higher"),
    ("mean_reward_task", ("mean_reward_task", "reward_task_mean"), "higher"),
    ("mean_reward_action", ("mean_reward_action", "reward_action_mean"), "higher"),
    ("mean_reward_safety", ("mean_reward_safety", "reward_safety_mean"), "higher"),
    ("warmup_excluded_comfort_comfort_score", (
        "warmup_excluded_comfort_comfort_score",
    ), "lower"),
    ("warmup_excluded_comfort_rms_vertical_acc", (
        "warmup_excluded_comfort_rms_vertical_acc",
    ), "lower"),
    ("warmup_excluded_comfort_rms_lateral_acc", (
        "warmup_excluded_comfort_rms_lateral_acc",
    ), "lower"),
    ("warmup_excluded_comfort_rms_longitudinal_acc", (
        "warmup_excluded_comfort_rms_longitudinal_acc",
    ), "lower"),
    ("warmup_excluded_comfort_rms_longitudinal_jerk", (
        "warmup_excluded_comfort_rms_longitudinal_jerk",
    ), "lower"),
    ("warmup_excluded_comfort_rms_vertical_jerk", (
        "warmup_excluded_comfort_rms_vertical_jerk",
    ), "lower"),
    ("warmup_excluded_comfort_rms_lateral_jerk", (
        "warmup_excluded_comfort_rms_lateral_jerk",
    ), "lower"),
    ("warmup_excluded_comfort_peak_abs_vertical_acc", (
        "warmup_excluded_comfort_peak_abs_vertical_acc",
    ), "lower"),
    ("warmup_excluded_comfort_peak_abs_vertical_jerk", (
        "warmup_excluded_comfort_peak_abs_vertical_jerk",
    ), "lower"),
    ("warmup_excluded_stability_peak_abs_roll", (
        "warmup_excluded_stability_peak_abs_roll",
        "stability_peak_abs_roll",
    ), "lower"),
    ("warmup_excluded_stability_peak_abs_pitch", (
        "warmup_excluded_stability_peak_abs_pitch",
        "stability_peak_abs_pitch",
    ), "lower"),
    ("warmup_excluded_stability_peak_abs_yaw_rate", (
        "warmup_excluded_stability_peak_abs_yaw_rate",
        "stability_peak_abs_yaw_rate",
    ), "lower"),
    ("warmup_excluded_stability_peak_abs_lateral_acc", (
        "warmup_excluded_stability_peak_abs_lateral_acc",
        "stability_peak_abs_lateral_acc",
    ), "lower"),
    ("warmup_excluded_stability_rms_roll", (
        "warmup_excluded_stability_rms_roll",
        "stability_rms_roll",
    ), "lower"),
    ("warmup_excluded_stability_rms_pitch", (
        "warmup_excluded_stability_rms_pitch",
        "stability_rms_pitch",
    ), "lower"),
    ("warmup_excluded_stability_rms_yaw_rate", (
        "warmup_excluded_stability_rms_yaw_rate",
        "stability_rms_yaw_rate",
    ), "lower"),
    ("warmup_excluded_stability_rms_lateral_acc", (
        "warmup_excluded_stability_rms_lateral_acc",
        "stability_rms_lateral_acc",
    ), "lower"),
    ("route_progress_stall_ratio", ("route_progress_stall_ratio",), "lower"),
    ("fallback_ratio", ("fallback_ratio",), "lower"),
    ("hard_safety_gate_ratio", ("hard_safety_gate_ratio",), "lower"),
    ("observation_clip_ratio", ("observation_clip_ratio",), "lower"),
    ("effective_control_ratio", ("effective_control_ratio",), "higher"),
    ("mean_abs_action", ("mean_abs_action", "rl_mean_abs_action"), "diagnostic"),
    ("mean_abs_residual_damper", (
        "mean_abs_residual_damper",
        "rl_mean_abs_residual_damper",
    ), "diagnostic"),
)


REPEAT_METRICS: Tuple[str, ...] = (
    "warmup_excluded_comfort_rms_vertical_acc",
    "warmup_excluded_comfort_rms_vertical_jerk",
    "warmup_excluded_comfort_rms_lateral_jerk",
    "warmup_excluded_comfort_rms_longitudinal_acc",
    "warmup_excluded_comfort_peak_abs_vertical_jerk",
    "warmup_excluded_stability_peak_abs_lateral_acc",
    "warmup_excluded_stability_peak_abs_roll",
    "warmup_excluded_stability_peak_abs_pitch",
    "warmup_excluded_stability_peak_abs_yaw_rate",
)


SPLIT_COMPARISON_BASE_FIELDS: Tuple[str, ...] = (
    "phase4d_compare_status",
    "policy_selection_status",
    "comparison_ready",
    "s4_summary",
    "s4_acceptance",
    "s4_manifest",
    "s8_summary",
    "s8_acceptance",
    "s8_manifest",
    "train_manifest",
    "s4_lineage_status",
    "s8_reference_lineage_status",
    "source_train_artifact_id",
    "s4_eval_artifact_id",
    "s8_reference_artifact_id",
    "routes_subset_s4",
    "routes_subset_s8",
    "eval_seed_s4",
    "eval_seed_s8",
    "baseline_s4",
    "baseline_s8",
    "planning_provider_s4",
    "planning_provider_s8",
    "controller_s4",
    "controller_s8",
    "s4_phase4d_eval_s4_status",
    "s8_phase4d_eval_s8_status",
    "s4_reference_status",
    "s8_reference_status",
    "safety_failed_checks",
    "control_failed_checks",
    "metadata_warnings",
    "warnings",
    "failed_checks",
    "repeat_improved_metric_count",
)

SPLIT_COMPARISON_FIELDS: Tuple[str, ...] = SPLIT_COMPARISON_BASE_FIELDS + tuple(
    field
    for metric, _keys, _direction in METRICS
    for field in (
        "%s_s4" % metric,
        "%s_s8" % metric,
        "%s_delta_s4_minus_s8" % metric,
        "%s_improvement" % metric,
        "%s_improvement_pct" % metric,
    )
)


COMPARE_ACCEPTANCE_FIELDS: Tuple[str, ...] = (
    "phase4d_compare_status",
    "policy_selection_status",
    "comparison_ready",
    "failed_checks",
    "warnings",
    "safety_failed_checks",
    "control_failed_checks",
    "metadata_warnings",
    "s4_lineage_status",
    "s8_reference_lineage_status",
    "repeat_improved_metric_count",
    "output_dir",
    "phase4d_split_comparison_json",
    "phase4d_policy_selection_report_json",
    "phase4d_compare_manifest_json",
)


COMPARE_MANIFEST_FIELDS: Tuple[str, ...] = (
    "artifact_id",
    "workflow_role",
    "phase",
    "compare_status",
    "policy_selection_status",
    "source_train_artifact_id",
    "s4_eval_artifact_id",
    "s8_reference_artifact_id",
    "train_manifest",
    "s4_summary",
    "s4_acceptance",
    "s4_manifest",
    "s8_summary",
    "s8_acceptance",
    "s8_manifest",
    "output_dir",
    "created_at",
)


def write_phase4d_compare_only(
    *,
    output_dir: str,
    s4_summary_path: str,
    s4_acceptance_path: str,
    s4_manifest_path: str,
    s8_summary_path: str,
    s8_acceptance_path: str = "",
    s8_manifest_path: str = "",
    train_manifest_path: str = "",
) -> Tuple[str, str, str, str, str, Dict[str, Any]]:
    output_dir = _abs_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    report = phase4d_compare_only_report(
        output_dir=output_dir,
        s4_summary_path=s4_summary_path,
        s4_acceptance_path=s4_acceptance_path,
        s4_manifest_path=s4_manifest_path,
        s8_summary_path=s8_summary_path,
        s8_acceptance_path=s8_acceptance_path,
        s8_manifest_path=s8_manifest_path,
        train_manifest_path=train_manifest_path)

    comparison_csv = os.path.join(output_dir, "phase4d_split_comparison.csv")
    comparison_json = os.path.join(output_dir, "phase4d_split_comparison.json")
    selection_json = os.path.join(output_dir, "phase4d_policy_selection_report.json")
    manifest_json = os.path.join(output_dir, "phase4d_compare_manifest.json")
    acceptance_json = os.path.join(output_dir, "phase4d_compare_acceptance.json")
    acceptance_csv = os.path.join(output_dir, "phase4d_compare_acceptance.csv")
    _write_single_row_csv(comparison_csv, report["comparison"], SPLIT_COMPARISON_FIELDS)
    _write_json(comparison_json, report["comparison"])
    _write_json(selection_json, report["policy_selection_report"])
    _write_json(manifest_json, report["compare_manifest"])
    _write_json(acceptance_json, report["compare_acceptance"])
    _write_single_row_csv(acceptance_csv, report["compare_acceptance"], COMPARE_ACCEPTANCE_FIELDS)
    return (
        comparison_csv,
        comparison_json,
        selection_json,
        manifest_json,
        acceptance_json,
        report,
    )


def phase4d_compare_only_report(
    *,
    output_dir: str,
    s4_summary_path: str,
    s4_acceptance_path: str,
    s4_manifest_path: str,
    s8_summary_path: str,
    s8_acceptance_path: str = "",
    s8_manifest_path: str = "",
    train_manifest_path: str = "",
) -> Dict[str, Any]:
    output_dir = _abs_path(output_dir)
    paths = {
        "s4_summary": _resolve_path(s4_summary_path),
        "s4_acceptance": _resolve_path(s4_acceptance_path),
        "s4_manifest": _resolve_path(s4_manifest_path),
        "s8_summary": _resolve_path(s8_summary_path),
        "s8_acceptance": _resolve_path(s8_acceptance_path),
        "s8_manifest": _resolve_path(s8_manifest_path),
        "train_manifest": _resolve_path(train_manifest_path),
    }
    s4_row = _select_row(_load_rows(paths["s4_summary"]), S4_SCENARIO)
    s8_row = _select_row(_load_rows(paths["s8_summary"]), S8_SCENARIO)
    s4_acceptance = _read_json_dict(paths["s4_acceptance"])
    s4_manifest = _read_json_dict(paths["s4_manifest"])
    s8_acceptance = _read_json_dict(paths["s8_acceptance"])
    s8_manifest = _read_json_dict(paths["s8_manifest"])
    train_manifest = _read_json_dict(paths["train_manifest"])

    readiness_failed = _readiness_failed(paths, s4_row, s8_row, s4_acceptance, s4_manifest)
    warnings = _lineage_warnings(paths, s8_acceptance, s8_manifest, train_manifest)
    metadata_warnings = _metadata_warnings(
        s4_row=s4_row,
        s8_row=s8_row,
        s4_acceptance=s4_acceptance,
        s4_manifest=s4_manifest,
        s8_acceptance=s8_acceptance,
        s8_manifest=s8_manifest,
        train_manifest=train_manifest)
    safety_failed = _safety_failed_checks(s4_row or {}, s8_row or {})
    control_failed = _control_failed_checks(s4_row or {})
    s4_lineage_status = _s4_lineage_status(s4_acceptance, s4_manifest)
    s8_lineage_status = _s8_lineage_status(paths, s8_acceptance, s8_manifest)
    metric_details = _metric_details(s4_row, s8_row)
    repeat_count = _repeat_improved_count(metric_details)

    if readiness_failed:
        policy_status = "COMPARISON_NOT_READY"
    elif safety_failed:
        policy_status = "POLICY_REJECTED_SAFETY"
    elif control_failed:
        policy_status = "POLICY_REJECTED_CONTROL_INVALID"
    elif metadata_warnings:
        policy_status = "COMPARISON_METADATA_WARNING"
    elif repeat_count >= 3 and _score_equal(s4_row, s8_row):
        policy_status = "POLICY_CANDIDATE_FOR_REPEAT"
    else:
        policy_status = "POLICY_ACCEPTED_ENGINEERING"

    selection_warnings = _dedupe(warnings + metadata_warnings)
    if policy_status in ("POLICY_ACCEPTED_ENGINEERING", "POLICY_CANDIDATE_FOR_REPEAT"):
        selection_warnings.append(PAPER_WARNING)
    comparison_ready = int(not readiness_failed and policy_status != "COMPARISON_METADATA_WARNING")
    compare_status = "pass" if policy_status in (
        "POLICY_ACCEPTED_ENGINEERING",
        "POLICY_CANDIDATE_FOR_REPEAT",
    ) else "warn" if policy_status == "COMPARISON_METADATA_WARNING" else "fail"
    comparison = _comparison_row(
        paths=paths,
        s4_row=s4_row or {},
        s8_row=s8_row or {},
        s4_acceptance=s4_acceptance,
        s4_manifest=s4_manifest,
        s8_acceptance=s8_acceptance,
        s8_manifest=s8_manifest,
        train_manifest=train_manifest,
        metric_details=metric_details,
        policy_status=policy_status,
        compare_status=compare_status,
        comparison_ready=comparison_ready,
        s4_lineage_status=s4_lineage_status,
        s8_lineage_status=s8_lineage_status,
        safety_failed=safety_failed,
        control_failed=control_failed,
        metadata_warnings=metadata_warnings,
        warnings=selection_warnings,
        readiness_failed=readiness_failed,
        repeat_count=repeat_count)
    acceptance = _compare_acceptance_row(
        output_dir=output_dir,
        comparison=comparison)
    manifest = _compare_manifest_row(
        output_dir=output_dir,
        paths=paths,
        comparison=comparison)
    policy_report = {
        "policy_selection_status": policy_status,
        "phase4d_compare_status": compare_status,
        "comparison_ready": comparison_ready,
        "failed_checks": _split_tokens(comparison.get("failed_checks")),
        "warnings": selection_warnings,
        "safety_failed_checks": safety_failed,
        "control_failed_checks": control_failed,
        "metadata_warnings": metadata_warnings,
        "repeat_improved_metric_count": repeat_count,
        "paper_claim_warning": PAPER_WARNING,
        "metrics": metric_details,
        "lineage": {
            "s4_lineage_status": s4_lineage_status,
            "s8_reference_lineage_status": s8_lineage_status,
            "source_train_artifact_id": comparison.get("source_train_artifact_id", ""),
            "s4_eval_artifact_id": comparison.get("s4_eval_artifact_id", ""),
            "s8_reference_artifact_id": comparison.get("s8_reference_artifact_id", ""),
            "paths": paths,
        },
    }
    return {
        "comparison": comparison,
        "policy_selection_report": policy_report,
        "compare_manifest": manifest,
        "compare_acceptance": acceptance,
        "inputs": paths,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s4-summary", required=True)
    parser.add_argument("--s4-acceptance", required=True)
    parser.add_argument("--s4-manifest", required=True)
    parser.add_argument("--s8-summary", required=True)
    parser.add_argument("--s8-acceptance", default="")
    parser.add_argument("--s8-manifest", default="")
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    (
        comparison_csv,
        comparison_json,
        selection_json,
        manifest_json,
        acceptance_json,
        report,
    ) = write_phase4d_compare_only(
        output_dir=args.output_dir,
        s4_summary_path=args.s4_summary,
        s4_acceptance_path=args.s4_acceptance,
        s4_manifest_path=args.s4_manifest,
        s8_summary_path=args.s8_summary,
        s8_acceptance_path=args.s8_acceptance,
        s8_manifest_path=args.s8_manifest,
        train_manifest_path=args.train_manifest)
    status = report["policy_selection_report"]["policy_selection_status"]
    print("phase4d split comparison: %s" % comparison_csv)
    print("phase4d split comparison json: %s" % comparison_json)
    print("phase4d policy selection report: %s" % selection_json)
    print("phase4d compare manifest: %s" % manifest_json)
    print("phase4d compare acceptance: %s" % acceptance_json)
    print("phase4d compare status: %s" % status)
    return 0


def _comparison_row(
    *,
    paths: Mapping[str, str],
    s4_row: Mapping[str, Any],
    s8_row: Mapping[str, Any],
    s4_acceptance: Mapping[str, Any],
    s4_manifest: Mapping[str, Any],
    s8_acceptance: Mapping[str, Any],
    s8_manifest: Mapping[str, Any],
    train_manifest: Mapping[str, Any],
    metric_details: Mapping[str, Mapping[str, Any]],
    policy_status: str,
    compare_status: str,
    comparison_ready: int,
    s4_lineage_status: str,
    s8_lineage_status: str,
    safety_failed: Sequence[str],
    control_failed: Sequence[str],
    metadata_warnings: Sequence[str],
    warnings: Sequence[str],
    readiness_failed: Sequence[str],
    repeat_count: int,
) -> Dict[str, Any]:
    source_train_artifact_id = _first_nonempty(
        s4_manifest.get("source_train_artifact_id"),
        train_manifest.get("artifact_id"))
    row: Dict[str, Any] = {
        "phase4d_compare_status": compare_status,
        "policy_selection_status": policy_status,
        "comparison_ready": comparison_ready,
        "s4_summary": paths.get("s4_summary", ""),
        "s4_acceptance": paths.get("s4_acceptance", ""),
        "s4_manifest": paths.get("s4_manifest", ""),
        "s8_summary": paths.get("s8_summary", ""),
        "s8_acceptance": paths.get("s8_acceptance", ""),
        "s8_manifest": paths.get("s8_manifest", ""),
        "train_manifest": paths.get("train_manifest", ""),
        "s4_lineage_status": s4_lineage_status,
        "s8_reference_lineage_status": s8_lineage_status,
        "source_train_artifact_id": source_train_artifact_id,
        "s4_eval_artifact_id": s4_manifest.get("artifact_id", ""),
        "s8_reference_artifact_id": s8_manifest.get("artifact_id", ""),
        "routes_subset_s4": _first_nonempty(
            s4_manifest.get("routes_subset"),
            s4_acceptance.get("routes_subset"),
            s4_row.get("routes_subset")),
        "routes_subset_s8": _first_nonempty(
            s8_manifest.get("routes_subset"),
            s8_acceptance.get("routes_subset"),
            s8_row.get("routes_subset")),
        "eval_seed_s4": _first_nonempty(
            s4_manifest.get("eval_seed"),
            s4_acceptance.get("seed"),
            s4_row.get("seed")),
        "eval_seed_s8": _first_nonempty(
            s8_manifest.get("eval_seed"),
            s8_acceptance.get("eval_seed"),
            s8_row.get("seed")),
        "baseline_s4": _first_nonempty(
            s4_manifest.get("baseline"),
            s4_acceptance.get("baseline")),
        "baseline_s8": _first_nonempty(
            s8_manifest.get("baseline"),
            s8_acceptance.get("baseline")),
        "planning_provider_s4": _first_nonempty(
            s4_manifest.get("planning_provider"),
            s4_acceptance.get("planning_provider")),
        "planning_provider_s8": _first_nonempty(
            s8_manifest.get("planning_provider"),
            s8_acceptance.get("planning_provider")),
        "controller_s4": _first_nonempty(s4_row.get("controller"), "rl_residual_skyhook"),
        "controller_s8": _first_nonempty(
            s8_manifest.get("controller"),
            s8_acceptance.get("controller"),
            s8_row.get("controller")),
        "s4_phase4d_eval_s4_status": s4_acceptance.get("phase4d_eval_s4_status", ""),
        "s8_phase4d_eval_s8_status": s8_acceptance.get("phase4d_eval_s8_status", ""),
        "s4_reference_status": s4_manifest.get("artifact_status", ""),
        "s8_reference_status": _first_nonempty(
            s8_manifest.get("reference_status"),
            s8_acceptance.get("reference_status")),
        "safety_failed_checks": ";".join(_dedupe(safety_failed)),
        "control_failed_checks": ";".join(_dedupe(control_failed)),
        "metadata_warnings": ";".join(_dedupe(metadata_warnings)),
        "warnings": ";".join(_dedupe(warnings)),
        "failed_checks": ";".join(_dedupe(readiness_failed)),
        "repeat_improved_metric_count": int(repeat_count),
    }
    for metric, _keys, _direction in METRICS:
        detail = metric_details.get(metric, {})
        row["%s_s4" % metric] = _blank_if_none(detail.get("s4"))
        row["%s_s8" % metric] = _blank_if_none(detail.get("s8"))
        row["%s_delta_s4_minus_s8" % metric] = _blank_if_none(detail.get("delta"))
        row["%s_improvement" % metric] = _blank_if_none(detail.get("improvement"))
        row["%s_improvement_pct" % metric] = _blank_if_none(detail.get("improvement_pct"))
    return _ordered_row(row, SPLIT_COMPARISON_FIELDS)


def _compare_acceptance_row(
    *,
    output_dir: str,
    comparison: Mapping[str, Any],
) -> Dict[str, Any]:
    return _ordered_row({
        "phase4d_compare_status": comparison.get("phase4d_compare_status", ""),
        "policy_selection_status": comparison.get("policy_selection_status", ""),
        "comparison_ready": comparison.get("comparison_ready", ""),
        "failed_checks": comparison.get("failed_checks", ""),
        "warnings": comparison.get("warnings", ""),
        "safety_failed_checks": comparison.get("safety_failed_checks", ""),
        "control_failed_checks": comparison.get("control_failed_checks", ""),
        "metadata_warnings": comparison.get("metadata_warnings", ""),
        "s4_lineage_status": comparison.get("s4_lineage_status", ""),
        "s8_reference_lineage_status": comparison.get("s8_reference_lineage_status", ""),
        "repeat_improved_metric_count": comparison.get("repeat_improved_metric_count", ""),
        "output_dir": output_dir,
        "phase4d_split_comparison_json": os.path.join(output_dir, "phase4d_split_comparison.json"),
        "phase4d_policy_selection_report_json": os.path.join(output_dir, "phase4d_policy_selection_report.json"),
        "phase4d_compare_manifest_json": os.path.join(output_dir, "phase4d_compare_manifest.json"),
    }, COMPARE_ACCEPTANCE_FIELDS)


def _compare_manifest_row(
    *,
    output_dir: str,
    paths: Mapping[str, str],
    comparison: Mapping[str, Any],
) -> Dict[str, Any]:
    identity = "|".join((
        comparison.get("source_train_artifact_id", ""),
        comparison.get("s4_eval_artifact_id", ""),
        comparison.get("s8_reference_artifact_id", ""),
        paths.get("s4_summary", ""),
        paths.get("s8_summary", ""),
        output_dir,
    ))
    artifact_id = "phase4d-compare-%s" % hashlib.sha256(
        identity.encode("utf-8")).hexdigest()[:12]
    return _ordered_row({
        "artifact_id": artifact_id,
        "workflow_role": "compare_only",
        "phase": "phase4d",
        "compare_status": comparison.get("phase4d_compare_status", ""),
        "policy_selection_status": comparison.get("policy_selection_status", ""),
        "source_train_artifact_id": comparison.get("source_train_artifact_id", ""),
        "s4_eval_artifact_id": comparison.get("s4_eval_artifact_id", ""),
        "s8_reference_artifact_id": comparison.get("s8_reference_artifact_id", ""),
        "train_manifest": paths.get("train_manifest", ""),
        "s4_summary": paths.get("s4_summary", ""),
        "s4_acceptance": paths.get("s4_acceptance", ""),
        "s4_manifest": paths.get("s4_manifest", ""),
        "s8_summary": paths.get("s8_summary", ""),
        "s8_acceptance": paths.get("s8_acceptance", ""),
        "s8_manifest": paths.get("s8_manifest", ""),
        "output_dir": output_dir,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }, COMPARE_MANIFEST_FIELDS)


def _readiness_failed(
    paths: Mapping[str, str],
    s4_row: Optional[Mapping[str, Any]],
    s8_row: Optional[Mapping[str, Any]],
    s4_acceptance: Mapping[str, Any],
    s4_manifest: Mapping[str, Any],
) -> List[str]:
    failed: List[str] = []
    for key in ("s4_summary", "s4_acceptance", "s4_manifest", "s8_summary"):
        if not paths.get(key) or not os.path.isfile(paths.get(key, "")):
            failed.append("%s_missing" % key)
    if s4_row is None:
        failed.append("s4_summary_row")
    if s8_row is None:
        failed.append("s8_summary_row")
    if not s4_acceptance:
        failed.append("s4_acceptance_unreadable")
    elif str(s4_acceptance.get("phase4d_eval_s4_status", "")) != "pass":
        failed.append("s4_acceptance_status")
    if not s4_manifest:
        failed.append("s4_manifest_unreadable")
    return _dedupe(failed)


def _lineage_warnings(
    paths: Mapping[str, str],
    s8_acceptance: Mapping[str, Any],
    s8_manifest: Mapping[str, Any],
    train_manifest: Mapping[str, Any],
) -> List[str]:
    warnings: List[str] = []
    if not paths.get("s8_acceptance") or not s8_acceptance:
        warnings.append("s8_reference_lineage_partial")
    if not paths.get("s8_manifest") or not s8_manifest:
        warnings.append("s8_reference_manifest_partial")
    if not paths.get("train_manifest") or not train_manifest:
        warnings.append("train_manifest_missing")
    return warnings


def _metadata_warnings(
    *,
    s4_row: Optional[Mapping[str, Any]],
    s8_row: Optional[Mapping[str, Any]],
    s4_acceptance: Mapping[str, Any],
    s4_manifest: Mapping[str, Any],
    s8_acceptance: Mapping[str, Any],
    s8_manifest: Mapping[str, Any],
    train_manifest: Mapping[str, Any],
) -> List[str]:
    del s4_row, s8_row
    warnings: List[str] = []
    if not _matches(
            _first_nonempty(s4_manifest.get("routes_subset"), s4_acceptance.get("routes_subset")),
            _first_nonempty(s8_manifest.get("routes_subset"), s8_acceptance.get("routes_subset"))):
        warnings.append("routes_subset_mismatch")
    if not _matches(
            _first_nonempty(s4_manifest.get("eval_seed"), s4_acceptance.get("seed")),
            _first_nonempty(s8_manifest.get("eval_seed"), s8_acceptance.get("eval_seed"))):
        warnings.append("eval_seed_mismatch")
    if not _matches(
            _first_nonempty(s4_manifest.get("baseline"), s4_acceptance.get("baseline")),
            _first_nonempty(s8_manifest.get("baseline"), s8_acceptance.get("baseline"))):
        warnings.append("baseline_mismatch")
    if not _matches(
            _first_nonempty(s4_manifest.get("planning_provider"), s4_acceptance.get("planning_provider")),
            _first_nonempty(s8_manifest.get("planning_provider"), s8_acceptance.get("planning_provider"))):
        warnings.append("planning_provider_mismatch")
    if (
            train_manifest and
            s4_manifest.get("source_train_artifact_id") and
            train_manifest.get("artifact_id") and
            s4_manifest.get("source_train_artifact_id") != train_manifest.get("artifact_id")):
        warnings.append("train_artifact_id_mismatch")
    return _dedupe(warnings)


def _s4_lineage_status(
    s4_acceptance: Mapping[str, Any],
    s4_manifest: Mapping[str, Any],
) -> str:
    if s4_acceptance and s4_manifest and str(s4_acceptance.get("phase4d_eval_s4_status", "")) == "pass":
        return "complete"
    if s4_acceptance or s4_manifest:
        return "partial"
    return "missing"


def _s8_lineage_status(
    paths: Mapping[str, str],
    s8_acceptance: Mapping[str, Any],
    s8_manifest: Mapping[str, Any],
) -> str:
    if paths.get("s8_acceptance") and paths.get("s8_manifest") and s8_acceptance and s8_manifest:
        return "complete"
    if paths.get("s8_summary"):
        return "partial"
    return "missing"


def _safety_failed_checks(
    s4: Mapping[str, Any],
    s8: Mapping[str, Any],
) -> List[str]:
    failed: List[str] = []
    for field in (
            "collision_count",
            "lane_invasion_count",
            "red_light_count",
            "route_timeout_count",
            "blocked_vehicle_count"):
        if (_float_first(s4, (field,)) or 0.0) > 0.0:
            failed.append(field)
    s4_score = _float_first(s4, ("score_composed",))
    s8_score = _float_first(s8, ("score_composed",))
    if s4_score is not None and s8_score is not None and s4_score < s8_score:
        failed.append("score_composed_below_reference")
    if (_float_first(s4, ("hard_safety_gate_ratio",)) or 0.0) > 0.0:
        failed.append("hard_safety_gate_ratio")
    return _dedupe(failed)


def _control_failed_checks(s4: Mapping[str, Any]) -> List[str]:
    failed: List[str] = []
    if (_float_first(s4, ("rl_policy_available_ratio", "policy_available_ratio")) or 0.0) < 0.99:
        failed.append("policy_available_ratio")
    if (_float_first(s4, ("rl_mean_abs_action", "mean_abs_action")) or 0.0) <= 0.0:
        failed.append("mean_abs_action")
    if (_float_first(s4, (
            "rl_mean_abs_residual_damper",
            "mean_abs_residual_damper")) or 0.0) <= 0.0:
        failed.append("mean_abs_residual_damper")
    if (_float_first(s4, ("effective_control_ratio",)) or 0.0) <= 0.0:
        failed.append("effective_control_ratio")
    return failed


def _metric_details(
    s4_row: Optional[Mapping[str, Any]],
    s8_row: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    s4 = s4_row or {}
    s8 = s8_row or {}
    result: Dict[str, Dict[str, Any]] = {}
    for metric, keys, direction in METRICS:
        s4_value = _float_first(s4, keys)
        s8_value = _float_first(s8, keys)
        delta = (
            s4_value - s8_value
            if s4_value is not None and s8_value is not None else
            None)
        improvement = _improvement(s4_value, s8_value, direction)
        improvement_pct = _improvement_pct(s4_value, s8_value, direction)
        result[metric] = {
            "s4": s4_value,
            "s8": s8_value,
            "delta": delta,
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "direction": direction,
        }
    return result


def _repeat_improved_count(metrics: Mapping[str, Mapping[str, Any]]) -> int:
    count = 0
    for metric in REPEAT_METRICS:
        improvement = metrics.get(metric, {}).get("improvement")
        if improvement is not None and improvement > 0.0:
            count += 1
    return count


def _score_equal(
    s4_row: Optional[Mapping[str, Any]],
    s8_row: Optional[Mapping[str, Any]],
) -> bool:
    s4 = _float_first(s4_row or {}, ("score_composed",))
    s8 = _float_first(s8_row or {}, ("score_composed",))
    return s4 is not None and s8 is not None and abs(s4 - s8) <= 1.0e-9


def _improvement(
    s4: Optional[float],
    s8: Optional[float],
    direction: str,
) -> Optional[float]:
    if s4 is None or s8 is None:
        return None
    if direction == "higher":
        return s4 - s8
    if direction == "lower":
        return s8 - s4
    return None


def _improvement_pct(
    s4: Optional[float],
    s8: Optional[float],
    direction: str,
) -> Optional[float]:
    improvement = _improvement(s4, s8, direction)
    if improvement is None or s8 is None:
        return None
    denominator = abs(s8)
    if denominator <= 1.0e-12:
        return 0.0 if abs(improvement) <= 1.0e-12 else None
    return improvement / denominator


def _load_rows(path: str) -> List[Dict[str, Any]]:
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


def _select_row(rows: Sequence[Mapping[str, Any]], scenario: str) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if str(row.get("scenario", row.get("name", ""))) == scenario:
            return row
    return rows[0] if len(rows) == 1 else None


def _read_json_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as json_file:
            value = json.load(json_file)
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _write_single_row_csv(path: str, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def _write_json(path: str, value: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as json_file:
        json.dump(value, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _ordered_row(row: Mapping[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    return {key: row.get(key, "") for key in fields}


def _float_first(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = _float_value(row.get(key))
        if value is not None:
            return value
    return None


def _float_value(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _matches(left: Any, right: Any) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return True
    return left_text == right_text


def _split_tokens(value: Any) -> List[str]:
    return [item.strip() for item in str(value or "").split(";") if item.strip()]


def _dedupe(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def _resolve_path(path: Any) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    expanded = os.path.expanduser(raw)
    candidates = [expanded]
    if expanded.startswith("/workspace/"):
        candidates.append(os.path.join(_sim_root(), expanded[len("/workspace/"):]))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(expanded)


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path or ".")))


def _sim_root() -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "..",
    ))


if __name__ == "__main__":
    raise SystemExit(main())

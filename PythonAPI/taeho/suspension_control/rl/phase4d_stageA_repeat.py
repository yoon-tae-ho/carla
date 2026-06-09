"""Phase 4-D Stage A same-artifact eval-repeat planning.

This module is intentionally CARLA-free. It validates one accepted train-only
artifact and writes a per-eval-seed plan that later runner steps can execute.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ACTION_SEMANTICS = "normalized_damper_residual_v1"
WORKFLOW_ROLE = "stageA_same_artifact_eval_repeat"
PAPER_WARNING = (
    "This is a same-artifact eval stochasticity check. Paper-level claims "
    "require train-seed repeat, route repeat, and planning-aware comparison."
)

STAGEA_ACCEPTANCE_FIELDS: Tuple[str, ...] = (
    "stageA_status",
    "preflight_ok",
    "failed_checks",
    "warnings",
    "workflow_role",
    "source_train_artifact_id",
    "train_manifest_path",
    "train_dir",
    "policy_path",
    "normalizer_path",
    "policy_sha256",
    "normalizer_sha256",
    "eval_seeds",
    "routes_subset",
    "baseline",
    "planning_provider",
    "output_dir",
    "dry_run",
    "plan_path",
    "manifest_path",
    "paper_claim_warning",
)

STAGEA_MANIFEST_FIELDS: Tuple[str, ...] = (
    "artifact_id",
    "workflow_role",
    "artifact_status",
    "phase",
    "source_train_artifact_id",
    "train_manifest_path",
    "train_dir",
    "policy_path",
    "normalizer_path",
    "policy_sha256",
    "normalizer_sha256",
    "eval_seeds",
    "routes_subset",
    "baseline",
    "planning_provider",
    "output_dir",
    "plan_path",
    "acceptance_path",
    "created_at",
    "stageA_status",
    "failed_checks",
    "warnings",
)

STAGEA_SUMMARY_FIELDS: Tuple[str, ...] = (
    "eval_seed",
    "seed_stageA_status",
    "s4_eval_status",
    "s8_eval_status",
    "compare_status",
    "s4_score_composed",
    "s8_score_composed",
    "delta_score_composed",
    "s4_policy_available_ratio",
    "s4_mean_abs_action",
    "s4_mean_abs_residual_damper",
    "s4_effective_control_ratio",
    "s4_fallback_ratio",
    "s4_hard_safety_gate_ratio",
    "delta_comfort_score",
    "delta_rms_vertical_acc",
    "delta_rms_vertical_jerk",
    "delta_peak_abs_vertical_acc",
    "delta_peak_abs_vertical_jerk",
    "delta_rms_lateral_acc",
    "delta_rms_lateral_jerk",
    "delta_rms_longitudinal_acc",
    "delta_rms_longitudinal_jerk",
    "delta_rms_roll",
    "delta_rms_pitch",
    "delta_rms_yaw_rate",
    "delta_peak_roll",
    "delta_peak_pitch",
    "delta_peak_yaw_rate",
    "comfort_score_improved",
    "vertical_acc_rms_worse",
    "vertical_jerk_rms_worse",
    "per_seed_policy_selection_status",
    "warnings",
    "failed_checks",
)

STAGEA_AGGREGATE_ACCEPTANCE_FIELDS: Tuple[str, ...] = STAGEA_ACCEPTANCE_FIELDS + (
    "aggregate_ok",
    "eval_seed_count",
    "completed_seed_count",
    "passing_seed_count",
    "incomplete_seed_count",
    "safety_failed_seed_count",
    "control_failed_seed_count",
    "summary_path",
    "summary_json_path",
    "stochasticity_report_path",
)

SAFETY_FAILED_TOKENS: Tuple[str, ...] = (
    "collision_count",
    "lane_invasion_count",
    "red_light_count",
    "route_timeout_count",
    "blocked_vehicle_count",
    "hard_safety_gate_ratio",
    "sidecar_fatal_errors",
    "score_composed_below_reference",
)

CONTROL_FAILED_TOKENS: Tuple[str, ...] = (
    "policy_available_ratio",
    "mean_abs_action",
    "mean_abs_residual_damper",
    "effective_control_ratio",
)

CORRECTED_METRIC_PROVENANCE_FIELDS: Tuple[str, ...] = (
    "metric_source",
    "metric_main_episode_index",
    "metric_main_actor_id",
)

STOCHASTICITY_NUMERIC_FIELDS: Tuple[str, ...] = (
    "s4_score_composed",
    "s8_score_composed",
    "delta_score_composed",
    "s4_policy_available_ratio",
    "s4_mean_abs_action",
    "s4_mean_abs_residual_damper",
    "s4_effective_control_ratio",
    "delta_comfort_score",
    "delta_rms_vertical_acc",
    "delta_rms_vertical_jerk",
    "delta_peak_abs_vertical_acc",
    "delta_peak_abs_vertical_jerk",
    "delta_rms_lateral_acc",
    "delta_rms_lateral_jerk",
    "delta_rms_longitudinal_acc",
    "delta_rms_longitudinal_jerk",
    "delta_rms_roll",
    "delta_rms_pitch",
    "delta_rms_yaw_rate",
    "delta_peak_roll",
    "delta_peak_pitch",
    "delta_peak_yaw_rate",
)


def write_phase4d_stageA_repeat_plan(
    *,
    output_dir: str,
    train_manifest_path: str,
    train_dir: str,
    eval_seeds: Sequence[Any],
    routes_subset: str = "00",
    baseline: str = "skyhook",
    planning_provider: str = "empty",
    dry_run: bool = False,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """Validate a train artifact and write Stage A plan/acceptance/manifest."""
    output_dir = _resolve_output_path(output_dir)
    train_dir = _resolve_path(train_dir, os.getcwd())
    train_manifest_path = _resolve_path(train_manifest_path, os.path.dirname(train_dir))
    os.makedirs(output_dir, exist_ok=True)

    manifest = _read_json_dict(train_manifest_path)
    policy_path = _resolve_path(
        _first_nonempty(manifest.get("policy_path"), os.path.join(train_dir, "policy.ts")),
        train_dir)
    normalizer_path = _resolve_path(
        _first_nonempty(manifest.get("normalizer_path"), os.path.join(train_dir, "normalizer.json")),
        train_dir)
    policy_sha = _sha256_file(policy_path)
    normalizer_sha = _sha256_file(normalizer_path)
    seed_values = _normalize_seeds(eval_seeds)

    context: Dict[str, Any] = {
        "output_dir": output_dir,
        "train_manifest_path": train_manifest_path,
        "train_dir": train_dir,
        "train_manifest": manifest,
        "policy_path": policy_path,
        "normalizer_path": normalizer_path,
        "policy_sha256": policy_sha,
        "normalizer_sha256": normalizer_sha,
        "eval_seeds": seed_values,
        "routes_subset": routes_subset,
        "baseline": baseline,
        "planning_provider": planning_provider,
        "source_train_artifact_id": manifest.get("artifact_id", ""),
        "dry_run": int(bool(dry_run)),
    }
    failed, warnings = _preflight_checks(context)
    preflight_ok = not failed
    status = "ready_for_execution" if preflight_ok else "preflight_failed"

    plan = _plan_document(context, status=status, failed_checks=failed, warnings=warnings)
    plan_path = os.path.join(output_dir, "phase4d_stageA_repeat_plan.json")
    _write_json(plan_path, plan)

    acceptance = _acceptance_row(
        context,
        status=status,
        preflight_ok=preflight_ok,
        failed_checks=failed,
        warnings=warnings,
        plan_path=plan_path)
    acceptance_csv, acceptance_json = _write_acceptance_outputs(output_dir, acceptance)
    del acceptance_csv

    stage_manifest = _manifest_row(
        context,
        status=status,
        failed_checks=failed,
        warnings=warnings,
        plan_path=plan_path,
        acceptance_path=acceptance_json)
    manifest_json, _manifest_csv = _write_manifest_outputs(output_dir, stage_manifest)

    return plan_path, acceptance_json, manifest_json, {
        "plan": plan,
        "acceptance": acceptance,
        "manifest": stage_manifest,
    }


def write_phase4d_stageA_repeat_aggregate(
    *,
    output_dir: str,
) -> Tuple[str, str, str, str, str, Dict[str, Any]]:
    """Aggregate per-seed Stage A outputs without invoking CARLA."""
    output_dir = _resolve_output_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    plan_path = os.path.join(output_dir, "phase4d_stageA_repeat_plan.json")
    manifest_path = os.path.join(output_dir, "phase4d_stageA_repeat_manifest.json")
    seed_status_path = os.path.join(output_dir, "phase4d_stageA_seed_status.csv")
    plan = _read_json_dict(plan_path)
    stage_manifest = _read_json_dict(manifest_path)
    seed_status_rows = {
        str(row.get("eval_seed", "")): row
        for row in _load_rows(seed_status_path)
        if str(row.get("eval_seed", ""))
    }
    train_manifest_path = _first_nonempty(
        plan.get("train_manifest_path"),
        stage_manifest.get("train_manifest_path"))
    train_manifest = _read_json_dict(train_manifest_path)
    seed_plans = _stageA_seed_plans(plan, output_dir)

    summary_rows = [
        _aggregate_seed_row(
            seed_plan,
            train_manifest=train_manifest,
            seed_status=seed_status_rows.get(str(seed_plan.get("eval_seed", "")), {}))
        for seed_plan in seed_plans
    ]
    stochasticity_report = _stochasticity_report(
        output_dir=output_dir,
        rows=summary_rows,
        plan=plan,
        manifest=stage_manifest)
    aggregate_status, failed_checks, warnings = _aggregate_status(summary_rows)

    summary_csv = os.path.join(output_dir, "phase4d_stageA_repeat_summary.csv")
    summary_json = os.path.join(output_dir, "phase4d_stageA_repeat_summary.json")
    stochasticity_json = os.path.join(output_dir, "phase4d_stageA_stochasticity_report.json")
    _write_rows_csv(summary_csv, summary_rows, STAGEA_SUMMARY_FIELDS)
    _write_json(summary_json, {
        "stageA_status": aggregate_status,
        "failed_checks": failed_checks,
        "warnings": warnings,
        "seed_rows": summary_rows,
        "stochasticity_report": stochasticity_report,
    })
    _write_json(stochasticity_json, stochasticity_report)

    acceptance = _aggregate_acceptance_row(
        output_dir=output_dir,
        plan=plan,
        manifest=stage_manifest,
        status=aggregate_status,
        failed_checks=failed_checks,
        warnings=warnings,
        rows=summary_rows,
        summary_csv=summary_csv,
        summary_json=summary_json,
        stochasticity_json=stochasticity_json)
    acceptance_csv = os.path.join(output_dir, "phase4d_stageA_repeat_acceptance.csv")
    acceptance_json = os.path.join(output_dir, "phase4d_stageA_repeat_acceptance.json")
    _write_single_row_csv(acceptance_csv, acceptance, STAGEA_AGGREGATE_ACCEPTANCE_FIELDS)
    _write_json(acceptance_json, acceptance)

    aggregate_manifest = dict(stage_manifest)
    aggregate_manifest.update({
        "artifact_status": aggregate_status,
        "stageA_status": aggregate_status,
        "failed_checks": ";".join(failed_checks),
        "warnings": ";".join(warnings),
        "summary_path": summary_csv,
        "summary_json_path": summary_json,
        "stochasticity_report_path": stochasticity_json,
        "acceptance_path": acceptance_json,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    _write_json(manifest_path, aggregate_manifest)
    _write_single_row_csv(
        os.path.join(output_dir, "phase4d_stageA_repeat_manifest.csv"),
        aggregate_manifest,
        tuple(_dedupe(list(STAGEA_MANIFEST_FIELDS) + [
            "summary_path",
            "summary_json_path",
            "stochasticity_report_path",
            "updated_at",
        ])))

    return (
        summary_csv,
        summary_json,
        acceptance_json,
        manifest_path,
        stochasticity_json,
        {
            "stageA_status": aggregate_status,
            "failed_checks": failed_checks,
            "warnings": warnings,
            "seed_rows": summary_rows,
            "stochasticity_report": stochasticity_report,
            "acceptance": acceptance,
            "manifest": aggregate_manifest,
        },
    )


def phase4d_stageA_seed_reuse_decision(
    *,
    output_dir: str,
    train_manifest_path: str,
    eval_seed: str,
    routes_subset: str = "00",
    baseline: str = "skyhook",
    planning_provider: str = "empty",
) -> Dict[str, Any]:
    """Return whether an existing seed output can be safely reused."""
    output_dir = _resolve_output_path(output_dir)
    train_manifest_path = _resolve_path(train_manifest_path, os.getcwd())
    seed = str(eval_seed)
    seed_dir = os.path.join(output_dir, "eval_seed_%s" % seed)
    s4_manifest_path = os.path.join(seed_dir, "s4", "phase4d_eval_s4_manifest.json")
    s8_manifest_path = os.path.join(seed_dir, "s8", "phase4d_eval_s8_manifest.json")
    compare_acceptance_path = os.path.join(seed_dir, "compare", "phase4d_compare_acceptance.json")
    compare_manifest_path = os.path.join(seed_dir, "compare", "phase4d_compare_manifest.json")

    train_manifest = _read_json_dict(train_manifest_path)
    s4_manifest = _read_json_dict(s4_manifest_path)
    s8_manifest = _read_json_dict(s8_manifest_path)
    compare_acceptance = _read_json_dict(compare_acceptance_path)
    compare_manifest = _read_json_dict(compare_manifest_path)
    failed: List[str] = []
    warnings: List[str] = []
    source_artifact_id = str(train_manifest.get("artifact_id", ""))
    policy_sha = str(train_manifest.get("policy_sha256", ""))
    normalizer_sha = str(train_manifest.get("normalizer_sha256", ""))

    for path_key, path in (
            ("s4_manifest", s4_manifest_path),
            ("s8_manifest", s8_manifest_path),
            ("compare_acceptance", compare_acceptance_path),
            ("compare_manifest", compare_manifest_path)):
        if not os.path.isfile(path):
            failed.append("%s_missing" % path_key)

    if not train_manifest:
        failed.append("train_manifest")
    if not s4_manifest:
        failed.append("s4_manifest_unreadable")
    if not s8_manifest:
        failed.append("s8_manifest_unreadable")
    if not compare_acceptance:
        failed.append("compare_acceptance_unreadable")
    if not compare_manifest:
        failed.append("compare_manifest_unreadable")

    if source_artifact_id and str(s4_manifest.get("source_train_artifact_id", "")) != source_artifact_id:
        failed.append("source_train_artifact_id")
    if policy_sha and str(s4_manifest.get("policy_sha256", "")) != policy_sha:
        failed.append("policy_sha256")
    if normalizer_sha and str(s4_manifest.get("normalizer_sha256", "")) != normalizer_sha:
        failed.append("normalizer_sha256")
    if str(s4_manifest.get("eval_seed", "")) != seed:
        failed.append("s4_eval_seed")
    if str(s8_manifest.get("eval_seed", "")) != seed:
        failed.append("s8_eval_seed")
    if str(s4_manifest.get("routes_subset", "")) != str(routes_subset):
        failed.append("s4_routes_subset")
    if str(s8_manifest.get("routes_subset", "")) != str(routes_subset):
        failed.append("s8_routes_subset")
    if str(s4_manifest.get("baseline", "")) != str(baseline):
        failed.append("s4_baseline")
    if str(s8_manifest.get("baseline", "")) != str(baseline):
        failed.append("s8_baseline")
    if str(s4_manifest.get("planning_provider", "")) != str(planning_provider):
        failed.append("s4_planning_provider")
    if str(s8_manifest.get("planning_provider", "")) != str(planning_provider):
        failed.append("s8_planning_provider")
    if compare_manifest and source_artifact_id and str(compare_manifest.get("source_train_artifact_id", "")) != source_artifact_id:
        failed.append("compare_source_train_artifact_id")
    if compare_acceptance and str(compare_acceptance.get("phase4d_compare_status", "")) not in ("pass", "warn", "fail"):
        warnings.append("compare_status_unrecognized")

    failed = _dedupe(failed)
    warnings = _dedupe(warnings)
    return {
        "reuse_ok": int(not failed),
        "failed_checks": failed,
        "warnings": warnings,
        "eval_seed": seed,
        "seed_dir": seed_dir,
        "source_train_artifact_id": source_artifact_id,
        "policy_sha256": policy_sha,
        "normalizer_sha256": normalizer_sha,
        "s4_manifest_path": s4_manifest_path,
        "s8_manifest_path": s8_manifest_path,
        "compare_acceptance_path": compare_acceptance_path,
        "compare_manifest_path": compare_manifest_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "aggregate", "reuse-check"), default="plan")
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--train-dir", default="")
    parser.add_argument("--eval-seeds", default="100,101,102")
    parser.add_argument("--eval-seed", default="")
    parser.add_argument("--routes-subset", default="00")
    parser.add_argument("--baseline", default="skyhook")
    parser.add_argument("--planning-provider", default="empty")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-preflight", action="store_true")
    parser.add_argument("--print-shell", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.action == "aggregate":
        summary_csv, summary_json, acceptance_json, manifest_json, stochasticity_json, report = (
            write_phase4d_stageA_repeat_aggregate(output_dir=args.output_dir))
        print("phase4d Stage A repeat summary: %s" % summary_csv)
        print("phase4d Stage A repeat summary json: %s" % summary_json)
        print("phase4d Stage A repeat acceptance: %s" % acceptance_json)
        print("phase4d Stage A repeat manifest: %s" % manifest_json)
        print("phase4d Stage A stochasticity report: %s" % stochasticity_json)
        print("phase4d Stage A status: %s failed_checks=%s" % (
            report.get("stageA_status", ""),
            ";".join(report.get("failed_checks", []))))
        return 0

    if args.action == "reuse-check":
        if not args.train_manifest:
            raise SystemExit("--action reuse-check requires --train-manifest")
        if not args.eval_seed:
            raise SystemExit("--action reuse-check requires --eval-seed")
        decision = phase4d_stageA_seed_reuse_decision(
            output_dir=args.output_dir,
            train_manifest_path=args.train_manifest,
            eval_seed=args.eval_seed,
            routes_subset=args.routes_subset,
            baseline=args.baseline,
            planning_provider=args.planning_provider)
        if args.print_shell:
            print("PHASE4D_STAGEA_REUSE_OK=%s" % int(decision.get("reuse_ok") or 0))
            print("PHASE4D_STAGEA_REUSE_FAILED_CHECKS=%s" % _shell_quote(
                ";".join(decision.get("failed_checks", []))))
            print("PHASE4D_STAGEA_REUSE_WARNINGS=%s" % _shell_quote(
                ";".join(decision.get("warnings", []))))
        else:
            print(json.dumps(decision, indent=2, sort_keys=True))
        return 0 if int(decision.get("reuse_ok") or 0) else 1

    if not args.train_manifest:
        raise SystemExit("--action plan requires --train-manifest")
    if not args.train_dir:
        raise SystemExit("--action plan requires --train-dir")
    plan_path, acceptance_json, manifest_json, result = write_phase4d_stageA_repeat_plan(
        output_dir=args.output_dir,
        train_manifest_path=args.train_manifest,
        train_dir=args.train_dir,
        eval_seeds=_parse_csv(args.eval_seeds),
        routes_subset=args.routes_subset,
        baseline=args.baseline,
        planning_provider=args.planning_provider,
        dry_run=args.dry_run)
    acceptance = result["acceptance"]
    print("phase4d Stage A repeat plan: %s" % plan_path)
    print("phase4d Stage A repeat acceptance: %s" % acceptance_json)
    print("phase4d Stage A repeat manifest: %s" % manifest_json)
    print("phase4d Stage A status: %s failed_checks=%s" % (
        acceptance.get("stageA_status", ""),
        acceptance.get("failed_checks", "")))
    if args.fail_on_preflight and not int(acceptance.get("preflight_ok") or 0):
        return 1
    return 0


def _preflight_checks(context: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    failed: List[str] = []
    warnings: List[str] = []
    manifest = _mapping(context.get("train_manifest"))
    if not manifest:
        failed.append("train_manifest_missing")
    if str(manifest.get("workflow_role", "")) != "train_only":
        failed.append("workflow_role")
    if str(manifest.get("artifact_status", "")) != "accepted_for_eval":
        failed.append("artifact_status")
    if _int_value(manifest.get("eval_allowed")) != 1:
        failed.append("eval_allowed")
    if _int_value(manifest.get("invalid_for_eval")) != 0:
        failed.append("invalid_for_eval")
    if str(manifest.get("baseline", "")) != str(context.get("baseline", "")):
        failed.append("baseline")
    if str(manifest.get("planning_provider", "")) != str(context.get("planning_provider", "")):
        failed.append("planning_provider")
    if str(manifest.get("action_semantics", "")) != ACTION_SEMANTICS:
        failed.append("action_semantics")
    if not os.path.isdir(str(context.get("train_dir", ""))):
        failed.append("train_dir")
    if not os.path.isfile(str(context.get("policy_path", ""))):
        failed.append("policy_path")
    if not os.path.isfile(str(context.get("normalizer_path", ""))):
        failed.append("normalizer_path")
    if manifest.get("policy_sha256") and context.get("policy_sha256") != manifest.get("policy_sha256"):
        failed.append("policy_sha256")
    if (
            manifest.get("normalizer_sha256") and
            context.get("normalizer_sha256") != manifest.get("normalizer_sha256")):
        failed.append("normalizer_sha256")
    if str(manifest.get("routes_subset", "")) and str(manifest.get("routes_subset", "")) != str(context.get("routes_subset", "")):
        warnings.append("routes_subset_differs_from_train")
    if not context.get("eval_seeds"):
        failed.append("eval_seeds")
    return _dedupe(failed), _dedupe(warnings)


def _plan_document(
    context: Mapping[str, Any],
    *,
    status: str,
    failed_checks: Sequence[str],
    warnings: Sequence[str],
) -> Dict[str, Any]:
    output_dir = str(context.get("output_dir", ""))
    seed_plans = []
    for seed in context.get("eval_seeds", []):
        seed_dir = os.path.join(output_dir, "eval_seed_%s" % seed)
        s4_dir = os.path.join(seed_dir, "s4")
        s8_dir = os.path.join(seed_dir, "s8")
        compare_dir = os.path.join(seed_dir, "compare")
        seed_plans.append({
            "eval_seed": seed,
            "seed_dir": seed_dir,
            "s4_output_dir": s4_dir,
            "s8_output_dir": s8_dir,
            "compare_output_dir": compare_dir,
            "s4_acceptance_path": os.path.join(s4_dir, "phase4d_eval_s4_acceptance.json"),
            "s4_manifest_path": os.path.join(s4_dir, "phase4d_eval_s4_manifest.json"),
            "s4_suite_summary_path": os.path.join(s4_dir, "suite_summary.csv"),
            "s8_acceptance_path": os.path.join(s8_dir, "phase4d_eval_s8_acceptance.json"),
            "s8_manifest_path": os.path.join(s8_dir, "phase4d_eval_s8_manifest.json"),
            "s8_suite_summary_path": os.path.join(s8_dir, "suite_summary.csv"),
            "compare_acceptance_path": os.path.join(compare_dir, "phase4d_compare_acceptance.json"),
            "compare_manifest_path": os.path.join(compare_dir, "phase4d_compare_manifest.json"),
            "compare_summary_path": os.path.join(compare_dir, "phase4d_split_comparison.json"),
            "run_s4": 1,
            "run_s8": 1,
            "compare": 1,
        })
    return {
        "workflow_role": WORKFLOW_ROLE,
        "stageA_status": status,
        "preflight_ok": int(status != "preflight_failed"),
        "failed_checks": list(_dedupe(failed_checks)),
        "warnings": list(_dedupe(warnings)),
        "source_train_artifact_id": context.get("source_train_artifact_id", ""),
        "train_manifest_path": context.get("train_manifest_path", ""),
        "train_dir": context.get("train_dir", ""),
        "policy_path": context.get("policy_path", ""),
        "normalizer_path": context.get("normalizer_path", ""),
        "policy_sha256": context.get("policy_sha256", ""),
        "normalizer_sha256": context.get("normalizer_sha256", ""),
        "eval_seeds": list(context.get("eval_seeds", [])),
        "routes_subset": context.get("routes_subset", ""),
        "baseline": context.get("baseline", ""),
        "planning_provider": context.get("planning_provider", ""),
        "output_dir": output_dir,
        "dry_run": int(context.get("dry_run") or 0),
        "seed_plans": seed_plans,
        "paper_claim_warning": PAPER_WARNING,
    }


def _acceptance_row(
    context: Mapping[str, Any],
    *,
    status: str,
    preflight_ok: bool,
    failed_checks: Sequence[str],
    warnings: Sequence[str],
    plan_path: str,
) -> Dict[str, Any]:
    output_dir = str(context.get("output_dir", ""))
    return _ordered_row({
        "stageA_status": status,
        "preflight_ok": int(bool(preflight_ok)),
        "failed_checks": ";".join(_dedupe(failed_checks)),
        "warnings": ";".join(_dedupe(warnings)),
        "workflow_role": WORKFLOW_ROLE,
        "source_train_artifact_id": context.get("source_train_artifact_id", ""),
        "train_manifest_path": context.get("train_manifest_path", ""),
        "train_dir": context.get("train_dir", ""),
        "policy_path": context.get("policy_path", ""),
        "normalizer_path": context.get("normalizer_path", ""),
        "policy_sha256": context.get("policy_sha256", ""),
        "normalizer_sha256": context.get("normalizer_sha256", ""),
        "eval_seeds": ",".join(str(seed) for seed in context.get("eval_seeds", [])),
        "routes_subset": context.get("routes_subset", ""),
        "baseline": context.get("baseline", ""),
        "planning_provider": context.get("planning_provider", ""),
        "output_dir": output_dir,
        "dry_run": int(context.get("dry_run") or 0),
        "plan_path": plan_path,
        "manifest_path": os.path.join(output_dir, "phase4d_stageA_repeat_manifest.json"),
        "paper_claim_warning": PAPER_WARNING,
    }, STAGEA_ACCEPTANCE_FIELDS)


def _manifest_row(
    context: Mapping[str, Any],
    *,
    status: str,
    failed_checks: Sequence[str],
    warnings: Sequence[str],
    plan_path: str,
    acceptance_path: str,
) -> Dict[str, Any]:
    identity = "|".join((
        str(context.get("source_train_artifact_id", "")),
        str(context.get("policy_sha256", "")),
        str(context.get("normalizer_sha256", "")),
        ",".join(str(seed) for seed in context.get("eval_seeds", [])),
        str(context.get("routes_subset", "")),
        str(context.get("baseline", "")),
        str(context.get("planning_provider", "")),
    ))
    artifact_id = "phase4d-stageA-repeat-%s" % hashlib.sha256(
        identity.encode("utf-8")).hexdigest()[:12]
    return _ordered_row({
        "artifact_id": artifact_id,
        "workflow_role": WORKFLOW_ROLE,
        "artifact_status": "ready_for_execution" if status != "preflight_failed" else "preflight_failed",
        "phase": "phase4d_stageA",
        "source_train_artifact_id": context.get("source_train_artifact_id", ""),
        "train_manifest_path": context.get("train_manifest_path", ""),
        "train_dir": context.get("train_dir", ""),
        "policy_path": context.get("policy_path", ""),
        "normalizer_path": context.get("normalizer_path", ""),
        "policy_sha256": context.get("policy_sha256", ""),
        "normalizer_sha256": context.get("normalizer_sha256", ""),
        "eval_seeds": ",".join(str(seed) for seed in context.get("eval_seeds", [])),
        "routes_subset": context.get("routes_subset", ""),
        "baseline": context.get("baseline", ""),
        "planning_provider": context.get("planning_provider", ""),
        "output_dir": context.get("output_dir", ""),
        "plan_path": plan_path,
        "acceptance_path": acceptance_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stageA_status": status,
        "failed_checks": ";".join(_dedupe(failed_checks)),
        "warnings": ";".join(_dedupe(warnings)),
    }, STAGEA_MANIFEST_FIELDS)


def _write_acceptance_outputs(output_dir: str, row: Mapping[str, Any]) -> Tuple[str, str]:
    csv_path = os.path.join(output_dir, "phase4d_stageA_repeat_acceptance.csv")
    json_path = os.path.join(output_dir, "phase4d_stageA_repeat_acceptance.json")
    _write_single_row_csv(csv_path, row, STAGEA_ACCEPTANCE_FIELDS)
    _write_json(json_path, row)
    return csv_path, json_path


def _write_manifest_outputs(output_dir: str, row: Mapping[str, Any]) -> Tuple[str, str]:
    json_path = os.path.join(output_dir, "phase4d_stageA_repeat_manifest.json")
    csv_path = os.path.join(output_dir, "phase4d_stageA_repeat_manifest.csv")
    _write_json(json_path, row)
    _write_single_row_csv(csv_path, row, STAGEA_MANIFEST_FIELDS)
    return json_path, csv_path


def _stageA_seed_plans(plan: Mapping[str, Any], output_dir: str) -> List[Dict[str, Any]]:
    seed_plans = [dict(row) for row in plan.get("seed_plans", []) if isinstance(row, Mapping)]
    if seed_plans:
        return seed_plans
    result: List[Dict[str, Any]] = []
    if not os.path.isdir(output_dir):
        return result
    for name in sorted(os.listdir(output_dir)):
        if not name.startswith("eval_seed_"):
            continue
        seed = name[len("eval_seed_"):]
        seed_dir = os.path.join(output_dir, name)
        result.append({
            "eval_seed": seed,
            "seed_dir": seed_dir,
            "s4_output_dir": os.path.join(seed_dir, "s4"),
            "s8_output_dir": os.path.join(seed_dir, "s8"),
            "compare_output_dir": os.path.join(seed_dir, "compare"),
        })
    return result


def _aggregate_seed_row(
    seed_plan: Mapping[str, Any],
    *,
    train_manifest: Mapping[str, Any],
    seed_status: Mapping[str, Any],
) -> Dict[str, Any]:
    del train_manifest
    eval_seed = str(seed_plan.get("eval_seed", ""))
    s4_dir = str(seed_plan.get("s4_output_dir") or os.path.join(
        str(seed_plan.get("seed_dir", "")),
        "s4"))
    s8_dir = str(seed_plan.get("s8_output_dir") or os.path.join(
        str(seed_plan.get("seed_dir", "")),
        "s8"))
    compare_dir = str(seed_plan.get("compare_output_dir") or os.path.join(
        str(seed_plan.get("seed_dir", "")),
        "compare"))
    s4_acceptance = _read_json_dict(os.path.join(s4_dir, "phase4d_eval_s4_acceptance.json"))
    s4_manifest = _read_json_dict(os.path.join(s4_dir, "phase4d_eval_s4_manifest.json"))
    s8_acceptance = _read_json_dict(os.path.join(s8_dir, "phase4d_eval_s8_acceptance.json"))
    s8_manifest = _read_json_dict(os.path.join(s8_dir, "phase4d_eval_s8_manifest.json"))
    compare_acceptance = _read_json_dict(os.path.join(compare_dir, "phase4d_compare_acceptance.json"))
    comparison = _read_json_dict(os.path.join(compare_dir, "phase4d_split_comparison.json"))
    s4_row = _select_summary_row(_load_rows(os.path.join(s4_dir, "suite_summary.csv")), "S4_rl_residual_skyhook")
    s8_row = _select_summary_row(_load_rows(os.path.join(s8_dir, "suite_summary.csv")), "S8_rl_zero_residual_skyhook")
    if not s4_row:
        s4_row = _select_summary_row(_load_rows(os.path.join(s4_dir, "suite_summary.json")), "S4_rl_residual_skyhook")
    if not s8_row:
        s8_row = _select_summary_row(_load_rows(os.path.join(s8_dir, "suite_summary.json")), "S8_rl_zero_residual_skyhook")

    failed = _split_tokens(seed_status.get("failed_stage"))
    warnings = _split_tokens(seed_status.get("warnings"))
    failed.extend(_split_tokens(s4_acceptance.get("failed_checks")))
    failed.extend(_split_tokens(s8_acceptance.get("failed_checks")))
    failed.extend(_split_tokens(compare_acceptance.get("failed_checks")))
    failed.extend(_split_tokens(comparison.get("failed_checks")))
    warnings.extend(_split_tokens(s4_acceptance.get("warnings")))
    warnings.extend(_split_tokens(s8_acceptance.get("warnings")))
    warnings.extend(_split_tokens(compare_acceptance.get("warnings")))
    warnings.extend(_split_tokens(comparison.get("warnings")))
    warnings.extend(_metric_provenance_warnings("s4", s4_row))
    warnings.extend(_metric_provenance_warnings("s8", s8_row))

    s4_eval_status = _first_nonempty(s4_acceptance.get("phase4d_eval_s4_status"), "missing")
    s8_eval_status = _first_nonempty(s8_acceptance.get("phase4d_eval_s8_status"), "missing")
    compare_status = _first_nonempty(
        compare_acceptance.get("phase4d_compare_status"),
        comparison.get("phase4d_compare_status"),
        "missing")
    policy_status = _first_nonempty(
        compare_acceptance.get("policy_selection_status"),
        comparison.get("policy_selection_status"))
    safety_failed = _split_tokens(comparison.get("safety_failed_checks"))
    control_failed = _split_tokens(comparison.get("control_failed_checks"))
    failed.extend(safety_failed)
    failed.extend(control_failed)

    missing_checks = []
    if s4_eval_status == "missing":
        missing_checks.append("s4_missing")
    if s8_eval_status == "missing":
        missing_checks.append("s8_missing")
    if compare_status == "missing":
        missing_checks.append("compare_missing")
    failed.extend(missing_checks)

    row_values = {
        "eval_seed": eval_seed,
        "s4_eval_status": s4_eval_status,
        "s8_eval_status": s8_eval_status,
        "compare_status": compare_status,
        "s4_score_composed": _metric_value("score_composed", "s4", comparison, s4_acceptance, s4_row),
        "s8_score_composed": _metric_value("score_composed", "s8", comparison, s8_acceptance, s8_row),
        "delta_score_composed": _metric_delta("score_composed", comparison, s4_acceptance, s8_acceptance, s4_row, s8_row),
        "s4_policy_available_ratio": _first_float(
            s4_acceptance,
            s4_row,
            comparison,
            keys=("policy_available_ratio", "rl_policy_available_ratio", "policy_available_ratio_s4")),
        "s4_mean_abs_action": _metric_value("mean_abs_action", "s4", comparison, s4_acceptance, s4_row),
        "s4_mean_abs_residual_damper": _metric_value("mean_abs_residual_damper", "s4", comparison, s4_acceptance, s4_row),
        "s4_effective_control_ratio": _metric_value("effective_control_ratio", "s4", comparison, s4_acceptance, s4_row),
        "s4_fallback_ratio": _metric_value("fallback_ratio", "s4", comparison, s4_acceptance, s4_row),
        "s4_hard_safety_gate_ratio": _metric_value("hard_safety_gate_ratio", "s4", comparison, s4_acceptance, s4_row),
        "delta_comfort_score": _metric_delta(
            "warmup_excluded_comfort_comfort_score",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_rms_vertical_acc": _metric_delta(
            "warmup_excluded_comfort_rms_vertical_acc",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_rms_vertical_jerk": _metric_delta(
            "warmup_excluded_comfort_rms_vertical_jerk",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_peak_abs_vertical_acc": _metric_delta(
            "warmup_excluded_comfort_peak_abs_vertical_acc",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_peak_abs_vertical_jerk": _metric_delta(
            "warmup_excluded_comfort_peak_abs_vertical_jerk",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_rms_lateral_acc": _metric_delta(
            "warmup_excluded_comfort_rms_lateral_acc",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_rms_lateral_jerk": _metric_delta(
            "warmup_excluded_comfort_rms_lateral_jerk",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_rms_longitudinal_acc": _metric_delta(
            "warmup_excluded_comfort_rms_longitudinal_acc",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_rms_longitudinal_jerk": _metric_delta(
            "warmup_excluded_comfort_rms_longitudinal_jerk",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_rms_roll": _metric_delta(
            "warmup_excluded_stability_rms_roll",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_rms_pitch": _metric_delta(
            "warmup_excluded_stability_rms_pitch",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_rms_yaw_rate": _metric_delta(
            "warmup_excluded_stability_rms_yaw_rate",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_peak_roll": _metric_delta(
            "warmup_excluded_stability_peak_abs_roll",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_peak_pitch": _metric_delta(
            "warmup_excluded_stability_peak_abs_pitch",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "delta_peak_yaw_rate": _metric_delta(
            "warmup_excluded_stability_peak_abs_yaw_rate",
            comparison,
            s4_acceptance,
            s8_acceptance,
            s4_row,
            s8_row),
        "per_seed_policy_selection_status": policy_status,
    }
    comfort_delta = _float_value(row_values.get("delta_comfort_score"))
    vertical_acc_delta = _float_value(row_values.get("delta_rms_vertical_acc"))
    vertical_jerk_delta = _float_value(row_values.get("delta_rms_vertical_jerk"))
    row_values["comfort_score_improved"] = int(comfort_delta is not None and comfort_delta < 0.0)
    row_values["vertical_acc_rms_worse"] = int(vertical_acc_delta is not None and vertical_acc_delta > 0.0)
    row_values["vertical_jerk_rms_worse"] = int(vertical_jerk_delta is not None and vertical_jerk_delta > 0.0)
    row_values["seed_stageA_status"] = _seed_stageA_status(
        failed=failed,
        missing=missing_checks,
        policy_status=policy_status,
        safety_failed=safety_failed,
        control_failed=control_failed)
    row_values["warnings"] = ";".join(_dedupe(warnings))
    row_values["failed_checks"] = ";".join(_dedupe(failed))
    row = _ordered_row(row_values, STAGEA_SUMMARY_FIELDS)
    row["_detail"] = {
        "s4_metric_provenance": _metric_provenance_detail(s4_row),
        "s8_metric_provenance": _metric_provenance_detail(s8_row),
        "s4_manifest": s4_manifest,
        "s8_manifest": s8_manifest,
    }
    return row


def _aggregate_status(rows: Sequence[Mapping[str, Any]]) -> Tuple[str, List[str], List[str]]:
    if not rows:
        return "STAGE_A_INCOMPLETE", ["no_seed_rows"], []
    statuses = [str(row.get("seed_stageA_status", "")) for row in rows]
    failed: List[str] = []
    warnings: List[str] = []
    for row in rows:
        failed.extend(_split_tokens(row.get("failed_checks")))
        warnings.extend(_split_tokens(row.get("warnings")))
    if any(status == "STAGE_A_FAIL_SAFETY" for status in statuses):
        return "STAGE_A_FAIL_SAFETY", _dedupe(failed), _dedupe(warnings)
    if any(status == "STAGE_A_FAIL_CONTROL_VALIDITY" for status in statuses):
        return "STAGE_A_FAIL_CONTROL_VALIDITY", _dedupe(failed), _dedupe(warnings)
    if any(status == "STAGE_A_INCOMPLETE" for status in statuses):
        return "STAGE_A_INCOMPLETE", _dedupe(failed), _dedupe(warnings)
    if any(status == "STAGE_A_FAIL_EVAL_INFRA" for status in statuses):
        return "STAGE_A_FAIL_EVAL_INFRA", _dedupe(failed), _dedupe(warnings)
    if _has_repeatable_key_improvement(rows):
        return "STAGE_A_PASS_REPEATABLE_SAFETY", [], _dedupe(warnings)
    return "STAGE_A_PASS_MIXED_PERFORMANCE", [], _dedupe(warnings)


def _aggregate_acceptance_row(
    *,
    output_dir: str,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    status: str,
    failed_checks: Sequence[str],
    warnings: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    summary_csv: str,
    summary_json: str,
    stochasticity_json: str,
) -> Dict[str, Any]:
    passing = sum(1 for row in rows if str(row.get("seed_stageA_status")) == "STAGE_A_PASS")
    incomplete = sum(1 for row in rows if str(row.get("seed_stageA_status")) == "STAGE_A_INCOMPLETE")
    safety = sum(1 for row in rows if str(row.get("seed_stageA_status")) == "STAGE_A_FAIL_SAFETY")
    control = sum(1 for row in rows if str(row.get("seed_stageA_status")) == "STAGE_A_FAIL_CONTROL_VALIDITY")
    base = _ordered_row({
        "stageA_status": status,
        "preflight_ok": 1,
        "failed_checks": ";".join(_dedupe(failed_checks)),
        "warnings": ";".join(_dedupe(warnings)),
        "workflow_role": WORKFLOW_ROLE,
        "source_train_artifact_id": _first_nonempty(
            manifest.get("source_train_artifact_id"),
            plan.get("source_train_artifact_id")),
        "train_manifest_path": _first_nonempty(
            manifest.get("train_manifest_path"),
            plan.get("train_manifest_path")),
        "train_dir": _first_nonempty(manifest.get("train_dir"), plan.get("train_dir")),
        "policy_path": _first_nonempty(manifest.get("policy_path"), plan.get("policy_path")),
        "normalizer_path": _first_nonempty(
            manifest.get("normalizer_path"),
            plan.get("normalizer_path")),
        "policy_sha256": _first_nonempty(manifest.get("policy_sha256"), plan.get("policy_sha256")),
        "normalizer_sha256": _first_nonempty(
            manifest.get("normalizer_sha256"),
            plan.get("normalizer_sha256")),
        "eval_seeds": ",".join(str(row.get("eval_seed", "")) for row in rows),
        "routes_subset": _first_nonempty(manifest.get("routes_subset"), plan.get("routes_subset")),
        "baseline": _first_nonempty(manifest.get("baseline"), plan.get("baseline")),
        "planning_provider": _first_nonempty(
            manifest.get("planning_provider"),
            plan.get("planning_provider")),
        "output_dir": output_dir,
        "dry_run": 0,
        "plan_path": os.path.join(output_dir, "phase4d_stageA_repeat_plan.json"),
        "manifest_path": os.path.join(output_dir, "phase4d_stageA_repeat_manifest.json"),
        "paper_claim_warning": PAPER_WARNING,
        "aggregate_ok": int(status in (
            "STAGE_A_PASS_REPEATABLE_SAFETY",
            "STAGE_A_PASS_MIXED_PERFORMANCE",
        )),
        "eval_seed_count": len(rows),
        "completed_seed_count": sum(1 for row in rows if str(row.get("seed_stageA_status")) != "STAGE_A_INCOMPLETE"),
        "passing_seed_count": passing,
        "incomplete_seed_count": incomplete,
        "safety_failed_seed_count": safety,
        "control_failed_seed_count": control,
        "summary_path": summary_csv,
        "summary_json_path": summary_json,
        "stochasticity_report_path": stochasticity_json,
    }, STAGEA_AGGREGATE_ACCEPTANCE_FIELDS)
    return base


def _stochasticity_report(
    *,
    output_dir: str,
    rows: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for field in STOCHASTICITY_NUMERIC_FIELDS:
        values = [_float_value(row.get(field)) for row in rows]
        metrics[field] = _numeric_stats([value for value in values if value is not None])
    num_s4_pass = sum(1 for row in rows if str(row.get("s4_eval_status")) == "pass")
    num_s8_pass = sum(1 for row in rows if str(row.get("s8_eval_status")) == "pass")
    num_compare_pass = sum(1 for row in rows if str(row.get("compare_status")) == "pass")
    return {
        "workflow_role": WORKFLOW_ROLE,
        "stageA_output_dir": output_dir,
        "source_train_artifact_id": _first_nonempty(
            manifest.get("source_train_artifact_id"),
            plan.get("source_train_artifact_id")),
        "policy_path": _first_nonempty(manifest.get("policy_path"), plan.get("policy_path")),
        "normalizer_path": _first_nonempty(
            manifest.get("normalizer_path"),
            plan.get("normalizer_path")),
        "policy_sha256": _first_nonempty(manifest.get("policy_sha256"), plan.get("policy_sha256")),
        "normalizer_sha256": _first_nonempty(
            manifest.get("normalizer_sha256"),
            plan.get("normalizer_sha256")),
        "eval_seeds": [row.get("eval_seed", "") for row in rows],
        "routes_subset": _first_nonempty(manifest.get("routes_subset"), plan.get("routes_subset")),
        "baseline": _first_nonempty(manifest.get("baseline"), plan.get("baseline")),
        "planning_provider": _first_nonempty(
            manifest.get("planning_provider"),
            plan.get("planning_provider")),
        "num_eval_seeds": len(rows),
        "num_s4_pass": num_s4_pass,
        "num_s8_pass": num_s8_pass,
        "num_compare_pass": num_compare_pass,
        "s4_safety_pass_all": int(not any(
            str(row.get("seed_stageA_status")) == "STAGE_A_FAIL_SAFETY"
            for row in rows)),
        "s4_control_valid_all": int(not any(
            str(row.get("seed_stageA_status")) == "STAGE_A_FAIL_CONTROL_VALIDITY"
            for row in rows)),
        "seed_status_counts": _status_counts(rows),
        "metrics": metrics,
        "metric_repeatability": _metric_repeatability(rows),
        "metric_provenance": [
            {
                "eval_seed": row.get("eval_seed", ""),
                "s4": row.get("_detail", {}).get("s4_metric_provenance", {}),
                "s8": row.get("_detail", {}).get("s8_metric_provenance", {}),
            }
            for row in rows
        ],
        "paper_claim_warning": PAPER_WARNING,
    }


def _seed_stageA_status(
    *,
    failed: Sequence[str],
    missing: Sequence[str],
    policy_status: str,
    safety_failed: Sequence[str],
    control_failed: Sequence[str],
) -> str:
    failed_set = set(_dedupe(failed))
    if policy_status == "POLICY_REJECTED_SAFETY" or safety_failed or failed_set.intersection(SAFETY_FAILED_TOKENS):
        return "STAGE_A_FAIL_SAFETY"
    if (
            policy_status == "POLICY_REJECTED_CONTROL_INVALID" or
            control_failed or
            failed_set.intersection(CONTROL_FAILED_TOKENS)):
        return "STAGE_A_FAIL_CONTROL_VALIDITY"
    if missing:
        return "STAGE_A_INCOMPLETE"
    if policy_status == "COMPARISON_NOT_READY":
        return "STAGE_A_INCOMPLETE"
    if failed_set:
        return "STAGE_A_FAIL_EVAL_INFRA"
    return "STAGE_A_PASS"


def _has_repeatable_key_improvement(rows: Sequence[Mapping[str, Any]]) -> bool:
    if not rows:
        return False
    threshold = max(1, int(math.ceil((2.0 * len(rows)) / 3.0)))
    for field in (
            "delta_comfort_score",
            "delta_rms_lateral_jerk",
            "delta_rms_longitudinal_acc",
            "delta_rms_longitudinal_jerk",
            "delta_rms_yaw_rate",
            "delta_peak_yaw_rate"):
        improved = sum(
            1 for row in rows
            if (_float_value(row.get(field)) is not None and _float_value(row.get(field)) < 0.0)
        )
        if improved >= threshold:
            return True
    return False


def _metric_repeatability(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    mapping = {
        "comfort_score": "delta_comfort_score",
        "rms_vertical_acc": "delta_rms_vertical_acc",
        "rms_lateral_jerk": "delta_rms_lateral_jerk",
        "rms_longitudinal_acc": "delta_rms_longitudinal_acc",
        "rms_longitudinal_jerk": "delta_rms_longitudinal_jerk",
        "rms_yaw_rate": "delta_rms_yaw_rate",
        "peak_abs_yaw_rate": "delta_peak_yaw_rate",
    }
    result: Dict[str, Any] = {}
    for name, field in mapping.items():
        values = [
            _float_value(row.get(field))
            for row in rows
            if _float_value(row.get(field)) is not None
        ]
        improved = sum(1 for value in values if value is not None and value < 0.0)
        worsened = sum(1 for value in values if value is not None and value > 0.0)
        denominator = max(len(values), 1)
        stats = _numeric_stats([value for value in values if value is not None])
        result[name] = {
            "mean_delta": stats["mean"],
            "std_delta": stats["std"],
            "improved_count": improved,
            "worsened_count": worsened,
            "consistency_ratio": max(improved, worsened) / float(denominator),
        }
    return result


def _metric_value(
    metric: str,
    suffix: str,
    comparison: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> Any:
    value = _float_value(comparison.get("%s_%s" % (metric, suffix)))
    if value is not None:
        return value
    value = _float_value(acceptance.get(metric))
    if value is not None:
        return value
    return _blank_if_none(_float_value(summary.get(metric)))


def _metric_delta(
    metric: str,
    comparison: Mapping[str, Any],
    s4_acceptance: Mapping[str, Any],
    s8_acceptance: Mapping[str, Any],
    s4_summary: Mapping[str, Any],
    s8_summary: Mapping[str, Any],
) -> Any:
    value = _float_value(comparison.get("%s_delta_s4_minus_s8" % metric))
    if value is not None:
        return value
    s4 = _metric_value(metric, "s4", comparison, s4_acceptance, s4_summary)
    s8 = _metric_value(metric, "s8", comparison, s8_acceptance, s8_summary)
    s4_float = _float_value(s4)
    s8_float = _float_value(s8)
    if s4_float is None or s8_float is None:
        return ""
    return s4_float - s8_float


def _first_float(*sources: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for source in sources:
        for key in keys:
            value = _float_value(source.get(key))
            if value is not None:
                return value
    return ""


def _metric_provenance_warnings(prefix: str, row: Mapping[str, Any]) -> List[str]:
    if not row:
        return ["%s_metric_provenance_missing" % prefix]
    warnings: List[str] = []
    if any(str(row.get(field, "")).strip() == "" for field in CORRECTED_METRIC_PROVENANCE_FIELDS):
        warnings.append("%s_metric_provenance_missing" % prefix)
    if _int_value(row.get("metric_invalid")) == 1:
        warnings.append("%s_metric_invalid" % prefix)
    return warnings


def _metric_provenance_detail(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "metric_source": row.get("metric_source", ""),
        "metric_main_episode_index": row.get("metric_main_episode_index", ""),
        "metric_main_actor_id": row.get("metric_main_actor_id", ""),
        "profile_total_rows": row.get("profile_total_rows", ""),
        "profile_main_rows": row.get("profile_main_rows", ""),
        "profile_excluded_non_main_rows": row.get("profile_excluded_non_main_rows", ""),
        "warmup_excluded_profile_rows": row.get("warmup_excluded_profile_rows", ""),
        "warmup_excluded_start_elapsed_seconds": row.get("warmup_excluded_start_elapsed_seconds", ""),
        "metric_actor_switch_detected": row.get("metric_actor_switch_detected", ""),
        "metric_elapsed_reset_detected": row.get("metric_elapsed_reset_detected", ""),
        "metric_invalid": row.get("metric_invalid", ""),
    }


def _numeric_stats(values: Sequence[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": "", "std": "", "min": "", "max": ""}
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
    }


def _status_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        status = str(row.get("seed_stageA_status", ""))
        counts[status] = counts.get(status, 0) + 1
    return counts


def _select_summary_row(rows: Sequence[Mapping[str, Any]], scenario: str) -> Dict[str, Any]:
    for row in rows:
        if str(row.get("scenario", row.get("name", ""))) == scenario:
            return dict(row)
    return dict(rows[0]) if len(rows) == 1 else {}


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
        if isinstance(value.get("seed_rows"), list):
            return [dict(item) for item in value["seed_rows"] if isinstance(item, Mapping)]
        return [dict(value)]
    return []


def _write_rows_csv(path: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: str, value: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as json_file:
        json.dump(value, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _write_single_row_csv(path: str, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def _read_json_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as json_file:
            value = json.load(json_file)
    except Exception:
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _sha256_file(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_output_path(path: Any) -> str:
    value = os.path.expanduser(str(path or ""))
    if value.startswith("/workspace/"):
        return os.path.abspath(os.path.join(_sim_root(), value[len("/workspace/"):]))
    return os.path.abspath(value)


def _resolve_path(path: Any, base_dir: str) -> str:
    value = os.path.expanduser(str(path or ""))
    if not value:
        return ""
    if os.path.isabs(value):
        if value.startswith("/workspace/"):
            host_value = os.path.join(_sim_root(), value[len("/workspace/"):])
            if os.path.exists(host_value):
                return os.path.abspath(host_value)
        return value
    return os.path.abspath(os.path.join(base_dir, value))


def _sim_root() -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "..",
    ))


def _normalize_seeds(values: Sequence[Any]) -> List[str]:
    result: List[str] = []
    for value in values:
        for part in _parse_csv(value):
            if part and part not in result:
                result.append(part)
    return result


def _parse_csv(value: Any) -> List[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _int_value(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _split_tokens(value: Any) -> List[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def _shell_quote(value: Any) -> str:
    text = str(value or "")
    return "'" + text.replace("'", "'\"'\"'") + "'"


def _dedupe(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _ordered_row(row: Mapping[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    return {field: row.get(field, "") for field in fields}


if __name__ == "__main__":
    raise SystemExit(main())

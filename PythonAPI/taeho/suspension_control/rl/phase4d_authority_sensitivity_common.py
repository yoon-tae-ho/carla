"""Shared CARLA-free planning helpers for Phase 4-D diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ACTION_SEMANTICS = "normalized_damper_residual_v1"
DEFAULT_ACTION_SCALES: Tuple[float, ...] = (0.0, 1.0, 3.0, 5.0, 10.0)
DEFAULT_SCRIPTED_RESIDUALS: Tuple[str, ...] = (
    "zero",
    "const_p0p05",
    "const_m0p05",
    "const_p0p10",
    "const_m0p10",
)

AUTHORITY_WORKFLOW_ROLE = "phase4d_action_authority_sweep"
SCRIPTED_WORKFLOW_ROLE = "phase4d_scripted_residual_sensitivity"

PAPER_WARNING = (
    "This is an eval-only diagnostic. action_scale>1 and scripted residual "
    "perturbations are not paper-level performance claims."
)

SCRIPTED_RESIDUAL_VALUES: Dict[str, float] = {
    "zero": 0.0,
    "const_p0p05": 0.05,
    "const_m0p05": -0.05,
    "const_p0p10": 0.10,
    "const_m0p10": -0.10,
}

SCRIPTED_RESIDUAL_ALIASES: Dict[str, str] = {
    "zero": "zero",
    "zero_residual": "zero",
    "const_p0p05": "const_p0p05",
    "constant_p0p05": "const_p0p05",
    "p0p05": "const_p0p05",
    "+0.05": "const_p0p05",
    "0.05": "const_p0p05",
    "constant_+0.05": "const_p0p05",
    "const_m0p05": "const_m0p05",
    "constant_m0p05": "const_m0p05",
    "m0p05": "const_m0p05",
    "-0.05": "const_m0p05",
    "constant_-0.05": "const_m0p05",
    "const_p0p10": "const_p0p10",
    "constant_p0p10": "const_p0p10",
    "p0p10": "const_p0p10",
    "+0.10": "const_p0p10",
    "0.10": "const_p0p10",
    "0.1": "const_p0p10",
    "+0.1": "const_p0p10",
    "constant_+0.10": "const_p0p10",
    "constant_+0.1": "const_p0p10",
    "const_m0p10": "const_m0p10",
    "constant_m0p10": "const_m0p10",
    "m0p10": "const_m0p10",
    "-0.10": "const_m0p10",
    "-0.1": "const_m0p10",
    "constant_-0.10": "const_m0p10",
    "constant_-0.1": "const_m0p10",
}

AUTHORITY_ACCEPTANCE_FIELDS: Tuple[str, ...] = (
    "phase4d_action_authority_status",
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
    "action_scales",
    "residual_gains",
    "scenario_names",
    "run_s8_reference",
    "compare",
    "stageA_report_path",
    "stageA_report_loaded",
    "output_dir",
    "dry_run",
    "plan_path",
    "manifest_path",
    "paper_claim_warning",
)

AUTHORITY_MANIFEST_FIELDS: Tuple[str, ...] = (
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
    "action_scales",
    "residual_gains",
    "scenario_names",
    "run_s8_reference",
    "compare",
    "stageA_report_path",
    "stageA_report_loaded",
    "output_dir",
    "plan_path",
    "acceptance_path",
    "created_at",
    "phase4d_action_authority_status",
    "failed_checks",
    "warnings",
)

SCRIPTED_ACCEPTANCE_FIELDS: Tuple[str, ...] = (
    "phase4d_scripted_residual_status",
    "preflight_ok",
    "failed_checks",
    "warnings",
    "workflow_role",
    "eval_seeds",
    "routes_subset",
    "baseline",
    "planning_provider",
    "scripted_residuals",
    "scripted_residual_values",
    "scenario_names",
    "compare",
    "stageA_report_path",
    "stageA_report_loaded",
    "output_dir",
    "dry_run",
    "plan_path",
    "manifest_path",
    "paper_claim_warning",
)

SCRIPTED_MANIFEST_FIELDS: Tuple[str, ...] = (
    "artifact_id",
    "workflow_role",
    "artifact_status",
    "phase",
    "eval_seeds",
    "routes_subset",
    "baseline",
    "planning_provider",
    "scripted_residuals",
    "scripted_residual_values",
    "scenario_names",
    "compare",
    "stageA_report_path",
    "stageA_report_loaded",
    "output_dir",
    "plan_path",
    "acceptance_path",
    "created_at",
    "phase4d_scripted_residual_status",
    "failed_checks",
    "warnings",
)


def write_action_authority_plan(
    *,
    output_dir: str,
    train_manifest_path: str,
    train_dir: str,
    eval_seeds: Sequence[Any],
    routes_subset: str = "00",
    baseline: str = "skyhook",
    planning_provider: str = "empty",
    action_scales: Sequence[Any] = DEFAULT_ACTION_SCALES,
    stageA_report_path: str = "",
    run_s8_reference: bool = True,
    compare: bool = True,
    dry_run: bool = False,
) -> Tuple[str, str, str, Dict[str, Any]]:
    output_dir = resolve_output_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    scales = parse_action_scales(action_scales)
    seed_values = normalize_seeds(eval_seeds)
    train_context = train_artifact_context(
        train_manifest_path=train_manifest_path,
        train_dir=train_dir,
        baseline=baseline,
        planning_provider=planning_provider,
        routes_subset=routes_subset)
    stageA_report, stageA_loaded, stageA_warnings = load_optional_stageA_report(stageA_report_path)
    context: Dict[str, Any] = dict(train_context)
    context.update({
        "output_dir": output_dir,
        "eval_seeds": seed_values,
        "routes_subset": routes_subset,
        "baseline": baseline,
        "planning_provider": planning_provider,
        "action_scales": scales,
        "stageA_report_path": resolve_optional_path(stageA_report_path, output_dir),
        "stageA_report_loaded": int(stageA_loaded),
        "stageA_report": stageA_report,
        "run_s8_reference": int(bool(run_s8_reference)),
        "compare": int(bool(compare)),
        "dry_run": int(bool(dry_run)),
    })
    failed, warnings = authority_preflight_checks(context)
    warnings.extend(stageA_warnings)
    failed = dedupe(failed)
    warnings = dedupe(warnings)
    status = "ready_for_execution" if not failed else "preflight_failed"
    scenario_names = authority_scenario_names(scales)

    plan = authority_plan_document(
        context,
        status=status,
        scenario_names=scenario_names,
        failed_checks=failed,
        warnings=warnings)
    plan_path = os.path.join(output_dir, "phase4d_action_authority_plan.json")
    write_json(plan_path, plan)
    acceptance = authority_acceptance_row(
        context,
        status=status,
        scenario_names=scenario_names,
        preflight_ok=not failed,
        failed_checks=failed,
        warnings=warnings,
        plan_path=plan_path)
    acceptance_csv, acceptance_json = write_authority_acceptance_outputs(output_dir, acceptance)
    del acceptance_csv
    manifest = authority_manifest_row(
        context,
        status=status,
        scenario_names=scenario_names,
        failed_checks=failed,
        warnings=warnings,
        plan_path=plan_path,
        acceptance_path=acceptance_json)
    manifest_json, _manifest_csv = write_authority_manifest_outputs(output_dir, manifest)
    return plan_path, acceptance_json, manifest_json, {
        "plan": plan,
        "acceptance": acceptance,
        "manifest": manifest,
    }


def write_scripted_residual_plan(
    *,
    output_dir: str,
    eval_seeds: Sequence[Any],
    routes_subset: str = "00",
    baseline: str = "skyhook",
    planning_provider: str = "empty",
    scripted_residuals: Sequence[Any] = DEFAULT_SCRIPTED_RESIDUALS,
    stageA_report_path: str = "",
    compare: bool = True,
    dry_run: bool = False,
) -> Tuple[str, str, str, Dict[str, Any]]:
    output_dir = resolve_output_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    seed_values = normalize_seeds(eval_seeds)
    residual_names = parse_scripted_residuals(scripted_residuals)
    stageA_report, stageA_loaded, stageA_warnings = load_optional_stageA_report(stageA_report_path)
    context: Dict[str, Any] = {
        "output_dir": output_dir,
        "eval_seeds": seed_values,
        "routes_subset": routes_subset,
        "baseline": baseline,
        "planning_provider": planning_provider,
        "scripted_residuals": residual_names,
        "stageA_report_path": resolve_optional_path(stageA_report_path, output_dir),
        "stageA_report_loaded": int(stageA_loaded),
        "stageA_report": stageA_report,
        "compare": int(bool(compare)),
        "dry_run": int(bool(dry_run)),
    }
    failed, warnings = scripted_preflight_checks(context)
    warnings.extend(stageA_warnings)
    failed = dedupe(failed)
    warnings = dedupe(warnings)
    status = "ready_for_execution" if not failed else "preflight_failed"
    scenario_names = scripted_scenario_names(residual_names)

    plan = scripted_plan_document(
        context,
        status=status,
        scenario_names=scenario_names,
        failed_checks=failed,
        warnings=warnings)
    plan_path = os.path.join(output_dir, "phase4d_scripted_residual_plan.json")
    write_json(plan_path, plan)
    acceptance = scripted_acceptance_row(
        context,
        status=status,
        scenario_names=scenario_names,
        preflight_ok=not failed,
        failed_checks=failed,
        warnings=warnings,
        plan_path=plan_path)
    acceptance_csv, acceptance_json = write_scripted_acceptance_outputs(output_dir, acceptance)
    del acceptance_csv
    manifest = scripted_manifest_row(
        context,
        status=status,
        scenario_names=scenario_names,
        failed_checks=failed,
        warnings=warnings,
        plan_path=plan_path,
        acceptance_path=acceptance_json)
    manifest_json, _manifest_csv = write_scripted_manifest_outputs(output_dir, manifest)
    return plan_path, acceptance_json, manifest_json, {
        "plan": plan,
        "acceptance": acceptance,
        "manifest": manifest,
    }


def train_artifact_context(
    *,
    train_manifest_path: str,
    train_dir: str,
    baseline: str,
    planning_provider: str,
    routes_subset: str,
) -> Dict[str, Any]:
    train_dir = resolve_path(train_dir, os.getcwd())
    train_manifest_path = resolve_path(train_manifest_path, os.path.dirname(train_dir))
    manifest = read_json_dict(train_manifest_path)
    policy_path = resolve_path(
        first_nonempty(manifest.get("policy_path"), os.path.join(train_dir, "policy.ts")),
        train_dir)
    normalizer_path = resolve_path(
        first_nonempty(manifest.get("normalizer_path"), os.path.join(train_dir, "normalizer.json")),
        train_dir)
    normalizer = read_json_dict(normalizer_path)
    action_semantics = first_nonempty(
        manifest.get("action_semantics"),
        normalizer.get("action_semantics"))
    return {
        "train_manifest_path": train_manifest_path,
        "train_dir": train_dir,
        "train_manifest": manifest,
        "policy_path": policy_path,
        "normalizer_path": normalizer_path,
        "policy_sha256": sha256_file(policy_path),
        "normalizer_sha256": sha256_file(normalizer_path),
        "source_train_artifact_id": manifest.get("artifact_id", ""),
        "baseline": baseline,
        "planning_provider": planning_provider,
        "routes_subset": routes_subset,
        "action_semantics": action_semantics,
    }


def authority_preflight_checks(context: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    failed: List[str] = []
    warnings: List[str] = []
    manifest = as_mapping(context.get("train_manifest"))
    if not manifest:
        failed.append("train_manifest_missing")
    if str(manifest.get("workflow_role", "")) != "train_only":
        failed.append("workflow_role")
    if str(manifest.get("artifact_status", "")) != "accepted_for_eval":
        failed.append("artifact_status")
    if int_value(manifest.get("eval_allowed")) != 1:
        failed.append("eval_allowed")
    if int_value(manifest.get("invalid_for_eval")) != 0:
        failed.append("invalid_for_eval")
    if str(manifest.get("baseline", "")) != str(context.get("baseline", "")):
        failed.append("baseline")
    if str(manifest.get("planning_provider", "")) != str(context.get("planning_provider", "")):
        failed.append("planning_provider")
    if str(context.get("action_semantics", "")) != ACTION_SEMANTICS:
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
    if not context.get("action_scales"):
        failed.append("action_scales")
    return dedupe(failed), dedupe(warnings)


def scripted_preflight_checks(context: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    failed: List[str] = []
    warnings: List[str] = []
    if not context.get("eval_seeds"):
        failed.append("eval_seeds")
    residuals = list(context.get("scripted_residuals") or [])
    if not residuals:
        failed.append("scripted_residuals")
    if "zero" not in residuals:
        warnings.append("zero_reference_not_requested")
    if str(context.get("baseline", "")) != "skyhook":
        failed.append("baseline")
    if str(context.get("planning_provider", "")) != "empty":
        failed.append("planning_provider")
    return dedupe(failed), dedupe(warnings)


def authority_plan_document(
    context: Mapping[str, Any],
    *,
    status: str,
    scenario_names: Sequence[str],
    failed_checks: Sequence[str],
    warnings: Sequence[str],
) -> Dict[str, Any]:
    output_dir = str(context.get("output_dir", ""))
    scales = list(context.get("action_scales") or [])
    seed_plans: List[Dict[str, Any]] = []
    for seed in context.get("eval_seeds", []):
        seed_dir = os.path.join(output_dir, "eval_seed_%s" % seed)
        gain_plans: List[Dict[str, Any]] = []
        for scale in scales:
            label = action_scale_label(scale)
            scenario = authority_scenario_name(scale)
            gain_dir = os.path.join(seed_dir, "gain_%s" % label)
            compare_dir = os.path.join(seed_dir, "compare_gain_%s" % label)
            gain_plans.append({
                "eval_seed": seed,
                "action_scale": scale,
                "residual_gain": scale,
                "action_scale_label": label,
                "scenario": scenario,
                "output_dir": gain_dir,
                "suite_summary_path": os.path.join(gain_dir, "suite_summary.csv"),
                "controller_diagnostics_path": os.path.join(
                    gain_dir,
                    "seed%s_%s" % (seed, scenario),
                    "controller_diagnostics.csv"),
                "compare_output_dir": compare_dir,
                "compare_acceptance_path": os.path.join(compare_dir, "phase4d_compare_acceptance.json"),
                "compare_summary_path": os.path.join(compare_dir, "phase4d_split_comparison.json"),
            })
        seed_plans.append({
            "eval_seed": seed,
            "seed_dir": seed_dir,
            "s8_reference_output_dir": os.path.join(seed_dir, "s8_reference"),
            "s8_reference_acceptance_path": os.path.join(
                seed_dir,
                "s8_reference",
                "phase4d_eval_s8_acceptance.json"),
            "action_scale_plans": gain_plans,
            "run_s8_reference": int(context.get("run_s8_reference") or 0),
            "compare": int(context.get("compare") or 0),
        })
    return {
        "workflow_role": AUTHORITY_WORKFLOW_ROLE,
        "phase4d_action_authority_status": status,
        "preflight_ok": int(status != "preflight_failed"),
        "failed_checks": list(dedupe(failed_checks)),
        "warnings": list(dedupe(warnings)),
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
        "action_scales": scales,
        "residual_gains": scales,
        "scenario_names": list(scenario_names),
        "run_s8_reference": int(context.get("run_s8_reference") or 0),
        "compare": int(context.get("compare") or 0),
        "stageA_report_path": context.get("stageA_report_path", ""),
        "stageA_report_loaded": int(context.get("stageA_report_loaded") or 0),
        "output_dir": output_dir,
        "dry_run": int(context.get("dry_run") or 0),
        "seed_plans": seed_plans,
        "paper_claim_warning": PAPER_WARNING,
    }


def scripted_plan_document(
    context: Mapping[str, Any],
    *,
    status: str,
    scenario_names: Sequence[str],
    failed_checks: Sequence[str],
    warnings: Sequence[str],
) -> Dict[str, Any]:
    output_dir = str(context.get("output_dir", ""))
    residuals = list(context.get("scripted_residuals") or [])
    seed_plans: List[Dict[str, Any]] = []
    for seed in context.get("eval_seeds", []):
        seed_dir = os.path.join(output_dir, "eval_seed_%s" % seed)
        residual_plans: List[Dict[str, Any]] = []
        for residual in residuals:
            scenario = scripted_scenario_name(residual)
            residual_dir = os.path.join(seed_dir, residual)
            compare_dir = os.path.join(seed_dir, "compare_%s" % residual)
            residual_plans.append({
                "eval_seed": seed,
                "scripted_residual": residual,
                "scripted_residual_value": SCRIPTED_RESIDUAL_VALUES[residual],
                "scenario": scenario,
                "output_dir": residual_dir,
                "suite_summary_path": os.path.join(residual_dir, "suite_summary.csv"),
                "controller_diagnostics_path": os.path.join(
                    residual_dir,
                    "seed%s_%s" % (seed, scenario),
                    "controller_diagnostics.csv"),
                "compare_output_dir": compare_dir,
                "compare_acceptance_path": os.path.join(compare_dir, "phase4d_compare_acceptance.json"),
                "compare_summary_path": os.path.join(compare_dir, "phase4d_split_comparison.json"),
            })
        seed_plans.append({
            "eval_seed": seed,
            "seed_dir": seed_dir,
            "residual_plans": residual_plans,
            "zero_reference_output_dir": os.path.join(seed_dir, "zero"),
            "compare": int(context.get("compare") or 0),
        })
    return {
        "workflow_role": SCRIPTED_WORKFLOW_ROLE,
        "phase4d_scripted_residual_status": status,
        "preflight_ok": int(status != "preflight_failed"),
        "failed_checks": list(dedupe(failed_checks)),
        "warnings": list(dedupe(warnings)),
        "eval_seeds": list(context.get("eval_seeds", [])),
        "routes_subset": context.get("routes_subset", ""),
        "baseline": context.get("baseline", ""),
        "planning_provider": context.get("planning_provider", ""),
        "scripted_residuals": residuals,
        "scripted_residual_values": scripted_residual_values_csv(residuals),
        "scenario_names": list(scenario_names),
        "compare": int(context.get("compare") or 0),
        "stageA_report_path": context.get("stageA_report_path", ""),
        "stageA_report_loaded": int(context.get("stageA_report_loaded") or 0),
        "output_dir": output_dir,
        "dry_run": int(context.get("dry_run") or 0),
        "seed_plans": seed_plans,
        "paper_claim_warning": PAPER_WARNING,
    }


def authority_acceptance_row(
    context: Mapping[str, Any],
    *,
    status: str,
    scenario_names: Sequence[str],
    preflight_ok: bool,
    failed_checks: Sequence[str],
    warnings: Sequence[str],
    plan_path: str,
) -> Dict[str, Any]:
    output_dir = str(context.get("output_dir", ""))
    return ordered_row({
        "phase4d_action_authority_status": status,
        "preflight_ok": int(bool(preflight_ok)),
        "failed_checks": ";".join(dedupe(failed_checks)),
        "warnings": ";".join(dedupe(warnings)),
        "workflow_role": AUTHORITY_WORKFLOW_ROLE,
        "source_train_artifact_id": context.get("source_train_artifact_id", ""),
        "train_manifest_path": context.get("train_manifest_path", ""),
        "train_dir": context.get("train_dir", ""),
        "policy_path": context.get("policy_path", ""),
        "normalizer_path": context.get("normalizer_path", ""),
        "policy_sha256": context.get("policy_sha256", ""),
        "normalizer_sha256": context.get("normalizer_sha256", ""),
        "eval_seeds": csv_join(context.get("eval_seeds", [])),
        "routes_subset": context.get("routes_subset", ""),
        "baseline": context.get("baseline", ""),
        "planning_provider": context.get("planning_provider", ""),
        "action_scales": csv_join(context.get("action_scales", [])),
        "residual_gains": csv_join(context.get("action_scales", [])),
        "scenario_names": csv_join(scenario_names),
        "run_s8_reference": int(context.get("run_s8_reference") or 0),
        "compare": int(context.get("compare") or 0),
        "stageA_report_path": context.get("stageA_report_path", ""),
        "stageA_report_loaded": int(context.get("stageA_report_loaded") or 0),
        "output_dir": output_dir,
        "dry_run": int(context.get("dry_run") or 0),
        "plan_path": plan_path,
        "manifest_path": os.path.join(output_dir, "phase4d_action_authority_manifest.json"),
        "paper_claim_warning": PAPER_WARNING,
    }, AUTHORITY_ACCEPTANCE_FIELDS)


def scripted_acceptance_row(
    context: Mapping[str, Any],
    *,
    status: str,
    scenario_names: Sequence[str],
    preflight_ok: bool,
    failed_checks: Sequence[str],
    warnings: Sequence[str],
    plan_path: str,
) -> Dict[str, Any]:
    output_dir = str(context.get("output_dir", ""))
    residuals = list(context.get("scripted_residuals") or [])
    return ordered_row({
        "phase4d_scripted_residual_status": status,
        "preflight_ok": int(bool(preflight_ok)),
        "failed_checks": ";".join(dedupe(failed_checks)),
        "warnings": ";".join(dedupe(warnings)),
        "workflow_role": SCRIPTED_WORKFLOW_ROLE,
        "eval_seeds": csv_join(context.get("eval_seeds", [])),
        "routes_subset": context.get("routes_subset", ""),
        "baseline": context.get("baseline", ""),
        "planning_provider": context.get("planning_provider", ""),
        "scripted_residuals": csv_join(residuals),
        "scripted_residual_values": scripted_residual_values_csv(residuals),
        "scenario_names": csv_join(scenario_names),
        "compare": int(context.get("compare") or 0),
        "stageA_report_path": context.get("stageA_report_path", ""),
        "stageA_report_loaded": int(context.get("stageA_report_loaded") or 0),
        "output_dir": output_dir,
        "dry_run": int(context.get("dry_run") or 0),
        "plan_path": plan_path,
        "manifest_path": os.path.join(output_dir, "phase4d_scripted_residual_manifest.json"),
        "paper_claim_warning": PAPER_WARNING,
    }, SCRIPTED_ACCEPTANCE_FIELDS)


def authority_manifest_row(
    context: Mapping[str, Any],
    *,
    status: str,
    scenario_names: Sequence[str],
    failed_checks: Sequence[str],
    warnings: Sequence[str],
    plan_path: str,
    acceptance_path: str,
) -> Dict[str, Any]:
    identity = "|".join((
        str(context.get("source_train_artifact_id", "")),
        str(context.get("policy_sha256", "")),
        str(context.get("normalizer_sha256", "")),
        csv_join(context.get("eval_seeds", [])),
        csv_join(context.get("action_scales", [])),
        str(context.get("routes_subset", "")),
        str(context.get("baseline", "")),
        str(context.get("planning_provider", "")),
    ))
    return ordered_row({
        "artifact_id": "phase4d-authority-sweep-%s" % hashlib.sha256(
            identity.encode("utf-8")).hexdigest()[:12],
        "workflow_role": AUTHORITY_WORKFLOW_ROLE,
        "artifact_status": "ready_for_execution" if status != "preflight_failed" else "preflight_failed",
        "phase": "phase4d_authority_sensitivity",
        "source_train_artifact_id": context.get("source_train_artifact_id", ""),
        "train_manifest_path": context.get("train_manifest_path", ""),
        "train_dir": context.get("train_dir", ""),
        "policy_path": context.get("policy_path", ""),
        "normalizer_path": context.get("normalizer_path", ""),
        "policy_sha256": context.get("policy_sha256", ""),
        "normalizer_sha256": context.get("normalizer_sha256", ""),
        "eval_seeds": csv_join(context.get("eval_seeds", [])),
        "routes_subset": context.get("routes_subset", ""),
        "baseline": context.get("baseline", ""),
        "planning_provider": context.get("planning_provider", ""),
        "action_scales": csv_join(context.get("action_scales", [])),
        "residual_gains": csv_join(context.get("action_scales", [])),
        "scenario_names": csv_join(scenario_names),
        "run_s8_reference": int(context.get("run_s8_reference") or 0),
        "compare": int(context.get("compare") or 0),
        "stageA_report_path": context.get("stageA_report_path", ""),
        "stageA_report_loaded": int(context.get("stageA_report_loaded") or 0),
        "output_dir": context.get("output_dir", ""),
        "plan_path": plan_path,
        "acceptance_path": acceptance_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase4d_action_authority_status": status,
        "failed_checks": ";".join(dedupe(failed_checks)),
        "warnings": ";".join(dedupe(warnings)),
    }, AUTHORITY_MANIFEST_FIELDS)


def scripted_manifest_row(
    context: Mapping[str, Any],
    *,
    status: str,
    scenario_names: Sequence[str],
    failed_checks: Sequence[str],
    warnings: Sequence[str],
    plan_path: str,
    acceptance_path: str,
) -> Dict[str, Any]:
    residuals = list(context.get("scripted_residuals") or [])
    identity = "|".join((
        csv_join(context.get("eval_seeds", [])),
        csv_join(residuals),
        str(context.get("routes_subset", "")),
        str(context.get("baseline", "")),
        str(context.get("planning_provider", "")),
    ))
    return ordered_row({
        "artifact_id": "phase4d-scripted-residual-%s" % hashlib.sha256(
            identity.encode("utf-8")).hexdigest()[:12],
        "workflow_role": SCRIPTED_WORKFLOW_ROLE,
        "artifact_status": "ready_for_execution" if status != "preflight_failed" else "preflight_failed",
        "phase": "phase4d_authority_sensitivity",
        "eval_seeds": csv_join(context.get("eval_seeds", [])),
        "routes_subset": context.get("routes_subset", ""),
        "baseline": context.get("baseline", ""),
        "planning_provider": context.get("planning_provider", ""),
        "scripted_residuals": csv_join(residuals),
        "scripted_residual_values": scripted_residual_values_csv(residuals),
        "scenario_names": csv_join(scenario_names),
        "compare": int(context.get("compare") or 0),
        "stageA_report_path": context.get("stageA_report_path", ""),
        "stageA_report_loaded": int(context.get("stageA_report_loaded") or 0),
        "output_dir": context.get("output_dir", ""),
        "plan_path": plan_path,
        "acceptance_path": acceptance_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase4d_scripted_residual_status": status,
        "failed_checks": ";".join(dedupe(failed_checks)),
        "warnings": ";".join(dedupe(warnings)),
    }, SCRIPTED_MANIFEST_FIELDS)


def parse_action_scales(values: Sequence[Any]) -> List[float]:
    tokens = parse_csv_values(values)
    if not tokens:
        return list(DEFAULT_ACTION_SCALES)
    scales: List[float] = []
    for token in tokens:
        try:
            value = float(token)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid action scale: %s" % token) from exc
        if value < 0.0:
            raise ValueError("action scale must be non-negative: %s" % token)
        scales.append(value)
    return scales


def action_scale_label(value: Any) -> str:
    numeric = float(value)
    text = ("%.6f" % numeric).rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text.replace("-", "m").replace(".", "p")


def authority_scenario_name(action_scale: Any) -> str:
    return "S4_gain_%s_rl_residual_skyhook" % action_scale_label(action_scale)


def authority_scenario_names(action_scales: Sequence[Any]) -> List[str]:
    return [authority_scenario_name(scale) for scale in action_scales]


def parse_scripted_residuals(values: Sequence[Any]) -> List[str]:
    tokens = parse_csv_values(values)
    if not tokens:
        return list(DEFAULT_SCRIPTED_RESIDUALS)
    result: List[str] = []
    for token in tokens:
        key = str(token).strip().lower()
        canonical = SCRIPTED_RESIDUAL_ALIASES.get(key)
        if not canonical:
            raise ValueError("invalid scripted residual: %s" % token)
        result.append(canonical)
    return dedupe(result)


def scripted_scenario_name(name: str) -> str:
    if name not in SCRIPTED_RESIDUAL_VALUES:
        raise ValueError("invalid scripted residual: %s" % name)
    if name == "zero":
        return "SC_zero_residual_skyhook"
    return "SC_%s_residual_skyhook" % name


def scripted_scenario_names(names: Sequence[str]) -> List[str]:
    return [scripted_scenario_name(name) for name in names]


def scripted_residual_values_csv(names: Sequence[str]) -> str:
    return ",".join("%s:%s" % (name, SCRIPTED_RESIDUAL_VALUES[name]) for name in names)


def normalize_seeds(values: Sequence[Any]) -> List[str]:
    return [str(value).strip() for value in parse_csv_values(values) if str(value).strip()]


def parse_csv_values(values: Sequence[Any]) -> List[str]:
    if values is None:
        return []
    if isinstance(values, str):
        raw_values: Sequence[Any] = [values]
    else:
        raw_values = values
    result: List[str] = []
    for value in raw_values:
        if value is None:
            continue
        for token in str(value).split(","):
            token = token.strip()
            if token:
                result.append(token)
    return result


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError("invalid boolean value: %s" % value)


def load_optional_stageA_report(path: str) -> Tuple[Dict[str, Any], bool, List[str]]:
    if not path:
        return {}, False, []
    resolved = resolve_optional_path(path, os.getcwd())
    report = read_json_dict(resolved)
    if not report:
        return {}, False, ["stageA_report_unreadable"]
    return report, True, []


def write_authority_acceptance_outputs(output_dir: str, row: Mapping[str, Any]) -> Tuple[str, str]:
    csv_path = os.path.join(output_dir, "phase4d_action_authority_acceptance.csv")
    json_path = os.path.join(output_dir, "phase4d_action_authority_acceptance.json")
    write_single_row_csv(csv_path, row, AUTHORITY_ACCEPTANCE_FIELDS)
    write_json(json_path, row)
    return csv_path, json_path


def write_authority_manifest_outputs(output_dir: str, row: Mapping[str, Any]) -> Tuple[str, str]:
    json_path = os.path.join(output_dir, "phase4d_action_authority_manifest.json")
    csv_path = os.path.join(output_dir, "phase4d_action_authority_manifest.csv")
    write_json(json_path, row)
    write_single_row_csv(csv_path, row, AUTHORITY_MANIFEST_FIELDS)
    return json_path, csv_path


def write_scripted_acceptance_outputs(output_dir: str, row: Mapping[str, Any]) -> Tuple[str, str]:
    csv_path = os.path.join(output_dir, "phase4d_scripted_residual_acceptance.csv")
    json_path = os.path.join(output_dir, "phase4d_scripted_residual_acceptance.json")
    write_single_row_csv(csv_path, row, SCRIPTED_ACCEPTANCE_FIELDS)
    write_json(json_path, row)
    return csv_path, json_path


def write_scripted_manifest_outputs(output_dir: str, row: Mapping[str, Any]) -> Tuple[str, str]:
    json_path = os.path.join(output_dir, "phase4d_scripted_residual_manifest.json")
    csv_path = os.path.join(output_dir, "phase4d_scripted_residual_manifest.csv")
    write_json(json_path, row)
    write_single_row_csv(csv_path, row, SCRIPTED_MANIFEST_FIELDS)
    return json_path, csv_path


def resolve_output_path(path: str) -> str:
    return os.path.abspath(resolve_workspace_path(path or os.getcwd(), must_exist=False))


def resolve_optional_path(path: str, base_dir: str) -> str:
    if not path:
        return ""
    return resolve_path(path, base_dir)


def resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return ""
    candidate = path if os.path.isabs(path) else os.path.join(base_dir, path)
    return os.path.abspath(resolve_workspace_path(candidate, must_exist=True))


def resolve_workspace_path(path: str, *, must_exist: bool) -> str:
    candidates = [path]
    if path.startswith("/workspace/"):
        candidates.append("/home/yth/sim/" + path[len("/workspace/"):])
    elif path.startswith("/home/yth/sim/"):
        candidates.append("/workspace/" + path[len("/home/yth/sim/"):])
    if must_exist:
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    else:
        for candidate in candidates:
            parent = os.path.dirname(candidate) or "."
            if os.path.exists(parent):
                return candidate
    return path


def read_json_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: str, value: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as target:
        json.dump(value, target, indent=2, sort_keys=True)
        target.write("\n")


def write_single_row_csv(path: str, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def sha256_file(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def int_value(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def ordered_row(values: Mapping[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    row = {field: values.get(field, "") for field in fields}
    for key, value in values.items():
        if key not in row:
            row[key] = value
    return row


def csv_join(values: Sequence[Any]) -> str:
    return ",".join(str(value) for value in values)


def dedupe(values: Sequence[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


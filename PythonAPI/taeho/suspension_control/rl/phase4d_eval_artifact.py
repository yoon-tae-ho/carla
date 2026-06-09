"""Artifact preflight and acceptance for Phase 4-D S4-only eval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


PHASE4D_EVAL_S4_ACCEPTANCE_FIELDS: Tuple[str, ...] = (
    "phase4d_eval_s4_status",
    "eval_status",
    "failed_checks",
    "warnings",
    "workflow_role",
    "scenario",
    "train_dir",
    "source_train_artifact_id",
    "train_acceptance_path",
    "train_manifest_path",
    "training_summary_path",
    "policy_path",
    "normalizer_path",
    "policy_sha256",
    "normalizer_sha256",
    "baseline",
    "planning_provider",
    "action_semantics",
    "routes_subset",
    "seed",
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
    "rl_fallback_rows",
    "rl_fallback_reasons",
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
    "sidecar_runtime_error_ratio",
    "sidecar_consecutive_runtime_errors",
    "sidecar_final_verification_recovered",
    "diagnostic_rows",
    "reward_rows",
    "reward_row_ratio",
)


PHASE4D_EVAL_S4_MANIFEST_FIELDS: Tuple[str, ...] = (
    "artifact_id",
    "workflow_role",
    "artifact_status",
    "phase",
    "source_train_artifact_id",
    "source_train_run_dir",
    "train_dir",
    "eval_dir",
    "train_acceptance_path",
    "train_manifest_path",
    "training_summary_path",
    "policy_path",
    "normalizer_path",
    "policy_sha256",
    "normalizer_sha256",
    "baseline",
    "planning_provider",
    "action_semantics",
    "routes_subset",
    "eval_seed",
    "suite_summary_path",
    "acceptance_path",
    "created_at",
    "phase4d_eval_s4_status",
    "eval_status",
    "failed_checks",
    "warnings",
)


S4_SCENARIO = "S4_rl_residual_skyhook"
ACTION_SEMANTICS = "normalized_damper_residual_v1"
SIDECAR_RUNTIME_ERROR_RATIO_FAIL = 0.05
SIDECAR_RUNTIME_ERROR_COUNT_FAIL = 10
SIDECAR_CONSECUTIVE_RUNTIME_ERROR_FAIL = 3


def prepare_phase4d_eval_s4(
    *,
    train_dir: str,
    output_dir: str,
    train_acceptance_path: str = "",
    train_manifest_path: str = "",
    training_summary_path: str = "",
    policy_path: str = "",
    normalizer_path: str = "",
    routes_subset: str = "",
    seed: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    context = _resolve_context(
        train_dir=train_dir,
        output_dir=output_dir,
        train_acceptance_path=train_acceptance_path,
        train_manifest_path=train_manifest_path,
        training_summary_path=training_summary_path,
        policy_path=policy_path,
        normalizer_path=normalizer_path,
        routes_subset=routes_subset,
        seed=seed)
    failed, warnings = _preflight_checks(context)
    preflight_ok = not failed
    row = _acceptance_base_row(
        context,
        status="ready_for_eval" if preflight_ok else "fail",
        eval_status="ready_for_eval" if preflight_ok else "skipped_due_to_invalid_train_artifact",
        failed_checks=failed,
        warnings=warnings)
    os.makedirs(context["output_dir"], exist_ok=True)
    _write_json(os.path.join(context["output_dir"], "phase4d_eval_s4_preflight.json"), {
        "preflight_ok": int(preflight_ok),
        "context": context,
        "failed_checks": failed,
        "warnings": warnings,
    })
    if not preflight_ok:
        _write_acceptance_outputs(context["output_dir"], row)
    manifest = _manifest_row(
        context,
        acceptance=row,
        artifact_status="ready_for_eval" if preflight_ok else "invalid_train_artifact")
    _write_manifest_outputs(context["output_dir"], manifest)
    return row, manifest


def write_phase4d_eval_s4_acceptance(
    *,
    train_dir: str,
    output_dir: str,
    train_acceptance_path: str = "",
    train_manifest_path: str = "",
    training_summary_path: str = "",
    policy_path: str = "",
    normalizer_path: str = "",
    routes_subset: str = "",
    seed: str = "",
    suite_summary_path: str = "",
    route_return_code: int = 0,
) -> Tuple[str, str, str, Dict[str, Any], Dict[str, Any]]:
    context = _resolve_context(
        train_dir=train_dir,
        output_dir=output_dir,
        train_acceptance_path=train_acceptance_path,
        train_manifest_path=train_manifest_path,
        training_summary_path=training_summary_path,
        policy_path=policy_path,
        normalizer_path=normalizer_path,
        routes_subset=routes_subset,
        seed=seed)
    preflight_failed, warnings = _preflight_checks(context)
    suite_path = _resolve_path(suite_summary_path or os.path.join(context["output_dir"], "suite_summary.csv"), context["output_dir"])
    source = _select_s4_row(_load_suite_rows(suite_path))
    failed = list(preflight_failed)
    if preflight_failed:
        eval_status = "skipped_due_to_invalid_train_artifact"
        status = "fail"
    else:
        eval_status = "completed"
        status = "pass"
        eval_failed, eval_warnings = _eval_checks(source, route_return_code)
        failed.extend(eval_failed)
        warnings.extend(eval_warnings)
        if failed:
            status = "fail"
    row = _acceptance_base_row(
        context,
        status=status,
        eval_status=eval_status,
        failed_checks=failed,
        warnings=warnings)
    row.update(_suite_metrics(source or {}, suite_path, route_return_code))
    row = _ordered_row(row, PHASE4D_EVAL_S4_ACCEPTANCE_FIELDS)
    csv_path, json_path = _write_acceptance_outputs(context["output_dir"], row)
    manifest = _manifest_row(
        context,
        acceptance=row,
        artifact_status="accepted_eval" if status == "pass" else "eval_failed",
        suite_summary_path=suite_path)
    manifest_json, _manifest_csv = _write_manifest_outputs(context["output_dir"], manifest)
    return csv_path, json_path, manifest_json, row, manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "accept"), required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--train-acceptance", default="")
    parser.add_argument("--train-manifest", default="")
    parser.add_argument("--training-summary", default="")
    parser.add_argument("--policy-path", default="")
    parser.add_argument("--normalizer-path", default="")
    parser.add_argument("--routes-subset", default="")
    parser.add_argument("--seed", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suite-summary", default="")
    parser.add_argument("--route-return-code", type=int, default=0)
    parser.add_argument("--print-shell", action="store_true")
    parser.add_argument("--fail-on-reject", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.action == "prepare":
        row, manifest = prepare_phase4d_eval_s4(
            train_dir=args.train_dir,
            output_dir=args.output_dir,
            train_acceptance_path=args.train_acceptance,
            train_manifest_path=args.train_manifest,
            training_summary_path=args.training_summary,
            policy_path=args.policy_path,
            normalizer_path=args.normalizer_path,
            routes_subset=args.routes_subset,
            seed=args.seed)
        if args.print_shell:
            _print_shell_exports(row, manifest)
        else:
            print("phase4d eval s4 preflight status: %s" % row.get("eval_status"))
            print("phase4d eval s4 manifest: %s" % os.path.join(
                os.path.abspath(args.output_dir),
                "phase4d_eval_s4_manifest.json"))
        if args.fail_on_reject and row.get("eval_status") != "ready_for_eval":
            return 1
        return 0

    csv_path, json_path, manifest_json, row, _manifest = write_phase4d_eval_s4_acceptance(
        train_dir=args.train_dir,
        output_dir=args.output_dir,
        train_acceptance_path=args.train_acceptance,
        train_manifest_path=args.train_manifest,
        training_summary_path=args.training_summary,
        policy_path=args.policy_path,
        normalizer_path=args.normalizer_path,
        routes_subset=args.routes_subset,
        seed=args.seed,
        suite_summary_path=args.suite_summary,
        route_return_code=args.route_return_code)
    print("phase4d eval s4 acceptance: %s" % csv_path)
    print("phase4d eval s4 acceptance json: %s" % json_path)
    print("phase4d eval s4 manifest: %s" % manifest_json)
    print("phase4d eval s4 status: %s eval_status=%s failed_checks=%s" % (
        row.get("phase4d_eval_s4_status"),
        row.get("eval_status"),
        row.get("failed_checks")))
    if args.fail_on_reject and row.get("phase4d_eval_s4_status") != "pass":
        return 1
    return 0


def _resolve_context(
    *,
    train_dir: str,
    output_dir: str,
    train_acceptance_path: str = "",
    train_manifest_path: str = "",
    training_summary_path: str = "",
    policy_path: str = "",
    normalizer_path: str = "",
    routes_subset: str = "",
    seed: str = "",
) -> Dict[str, Any]:
    train_dir = _abs_path(train_dir)
    output_dir = _abs_path(output_dir)
    train_root = os.path.dirname(train_dir) if os.path.basename(train_dir) == "train" else train_dir
    acceptance_path = _resolve_path(
        train_acceptance_path or os.path.join(train_root, "phase4d_train_acceptance.json"),
        train_root)
    manifest_path = _resolve_path(
        train_manifest_path or os.path.join(train_root, "phase4d_train_manifest.json"),
        train_root)
    summary_path = _resolve_path(
        training_summary_path or os.path.join(train_dir, "training_summary.json"),
        train_dir)
    acceptance = _read_json_dict(acceptance_path)
    manifest = _read_json_dict(manifest_path)
    summary = _read_json_dict(summary_path)
    normalizer_default = _first_nonempty(
        normalizer_path,
        manifest.get("normalizer_path"),
        summary.get("normalizer_path"),
        os.path.join(train_dir, "normalizer.json"))
    policy_default = _first_nonempty(
        policy_path,
        manifest.get("policy_path"),
        summary.get("policy_path"),
        os.path.join(train_dir, "policy.ts"))
    policy_resolved = _resolve_path(policy_default, train_dir)
    normalizer_resolved = _resolve_path(normalizer_default, train_dir)
    normalizer = _read_json_dict(normalizer_resolved)
    routes_value = _first_nonempty(
        routes_subset,
        manifest.get("routes_subset"),
        summary.get("routes_subset"),
        acceptance.get("routes_subset"),
        "00")
    seed_value = _first_nonempty(
        seed,
        manifest.get("train_seed"),
        summary.get("seed"),
        acceptance.get("train_seed"),
        "100")
    return {
        "train_dir": train_dir,
        "train_root": train_root,
        "output_dir": output_dir,
        "train_acceptance_path": acceptance_path,
        "train_manifest_path": manifest_path,
        "training_summary_path": summary_path,
        "policy_path": policy_resolved,
        "normalizer_path": normalizer_resolved,
        "policy_sha256": _sha256_file(policy_resolved),
        "normalizer_sha256": _sha256_file(normalizer_resolved),
        "routes_subset": routes_value,
        "seed": str(seed_value),
        "train_acceptance": acceptance,
        "train_manifest": manifest,
        "training_summary": summary,
        "normalizer": normalizer,
        "source_train_artifact_id": manifest.get("artifact_id", ""),
        "baseline": _first_nonempty(manifest.get("baseline"), summary.get("baseline")),
        "planning_provider": _first_nonempty(
            manifest.get("planning_provider"),
            summary.get("planning_provider")),
        "action_semantics": _first_nonempty(
            manifest.get("action_semantics"),
            summary.get("action_semantics"),
            normalizer.get("action_semantics")),
    }


def _preflight_checks(context: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    failed: List[str] = []
    warnings: List[str] = []
    acceptance = _mapping(context.get("train_acceptance"))
    manifest = _mapping(context.get("train_manifest"))
    summary = _mapping(context.get("training_summary"))
    if not acceptance:
        failed.append("train_acceptance_missing")
    if not manifest:
        failed.append("train_manifest_missing")
    if not summary:
        failed.append("training_summary_missing")
    if _int_value(acceptance.get("eval_allowed")) != 1:
        failed.append("eval_allowed")
    if _int_value(acceptance.get("invalid_for_eval")) != 0:
        failed.append("invalid_for_eval")
    if str(acceptance.get("phase4d_train_status", "")) != "pass":
        failed.append("phase4d_train_status")
    if str(summary.get("status", "")) != "trained":
        failed.append("training_status")
    if _nonempty(summary.get("learn_error")):
        failed.append("learn_error")
    if not os.path.isfile(str(context.get("policy_path", ""))):
        failed.append("policy_path")
    if not os.path.isfile(str(context.get("normalizer_path", ""))):
        failed.append("normalizer_path")
    if str(context.get("baseline", "")) != "skyhook":
        failed.append("baseline")
    if str(context.get("planning_provider", "")) != "empty":
        failed.append("planning_provider")
    if str(context.get("action_semantics", "")) != ACTION_SEMANTICS:
        failed.append("action_semantics")

    if not _path_matches_artifact(
            str(context.get("policy_path", "")),
            (manifest.get("policy_path"), summary.get("policy_path")),
            str(context.get("train_dir", ""))):
        failed.append("policy_path_lineage")
    if not _path_matches_artifact(
            str(context.get("normalizer_path", "")),
            (manifest.get("normalizer_path"), summary.get("normalizer_path")),
            str(context.get("train_dir", ""))):
        failed.append("normalizer_path_lineage")
    if manifest.get("policy_sha256") and context.get("policy_sha256") != manifest.get("policy_sha256"):
        failed.append("policy_sha256")
    if manifest.get("normalizer_sha256") and context.get("normalizer_sha256") != manifest.get("normalizer_sha256"):
        failed.append("normalizer_sha256")
    return _dedupe(failed), _dedupe(warnings)


def _acceptance_base_row(
    context: Mapping[str, Any],
    *,
    status: str,
    eval_status: str,
    failed_checks: Sequence[str],
    warnings: Sequence[str],
) -> Dict[str, Any]:
    return _ordered_row({
        "phase4d_eval_s4_status": status,
        "eval_status": eval_status,
        "failed_checks": ";".join(_dedupe(failed_checks)),
        "warnings": ";".join(_dedupe(warnings)),
        "workflow_role": "eval_s4_only",
        "scenario": S4_SCENARIO,
        "train_dir": context.get("train_dir", ""),
        "source_train_artifact_id": context.get("source_train_artifact_id", ""),
        "train_acceptance_path": context.get("train_acceptance_path", ""),
        "train_manifest_path": context.get("train_manifest_path", ""),
        "training_summary_path": context.get("training_summary_path", ""),
        "policy_path": context.get("policy_path", ""),
        "normalizer_path": context.get("normalizer_path", ""),
        "policy_sha256": context.get("policy_sha256", ""),
        "normalizer_sha256": context.get("normalizer_sha256", ""),
        "baseline": context.get("baseline", ""),
        "planning_provider": context.get("planning_provider", ""),
        "action_semantics": context.get("action_semantics", ""),
        "routes_subset": context.get("routes_subset", ""),
        "seed": context.get("seed", ""),
        "output_dir": context.get("output_dir", ""),
    }, PHASE4D_EVAL_S4_ACCEPTANCE_FIELDS)


def _eval_failed_checks(source: Optional[Mapping[str, Any]], route_return_code: int) -> List[str]:
    failed, _warnings = _eval_checks(source, route_return_code)
    return failed


def _eval_checks(
    source: Optional[Mapping[str, Any]],
    route_return_code: int,
) -> Tuple[List[str], List[str]]:
    failed: List[str] = []
    warnings: List[str] = []
    if int(route_return_code) != 0:
        failed.append("route_return_code")
    if source is None:
        failed.append("suite_summary")
        return _dedupe(failed), warnings
    if str(source.get("status", "")) != "ok":
        failed.append("scenario_status")
    if _int_value(source.get("return_code")) != 0:
        failed.append("scenario_return_code")
    if str(source.get("global_status", "")).strip() != "Perfect":
        failed.append("global_status")
    if (_float_first(source, ("sidecar_command_verifies",)) or 0.0) <= 0.0:
        failed.append("verification")
    if _float_missing_or_below(_float_first(source, ("rl_policy_available_ratio", "policy_available_ratio")), 0.99):
        failed.append("policy_available_ratio")
    if _float_missing_or_at_most(_float_first(source, ("rl_mean_abs_action", "mean_abs_action")), 0.0):
        failed.append("mean_abs_action")
    if _float_missing_or_at_most(_float_first(source, ("rl_mean_abs_residual_damper", "mean_abs_residual_damper")), 0.0):
        failed.append("mean_abs_residual_damper")
    if _float_missing_or_at_most(_float_first(source, ("effective_control_ratio",)), 0.0):
        failed.append("effective_control_ratio")
    hard_gate = _float_first(source, ("hard_safety_gate_ratio",))
    if hard_gate is None or hard_gate > 0.0:
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

    runtime_errors = _float_first(source, ("sidecar_runtime_errors",)) or 0.0
    verifies = max(_float_first(source, ("sidecar_command_verifies",)) or 0.0, 1.0)
    runtime_ratio = runtime_errors / verifies
    event_stats = _sidecar_event_stats(source.get("sidecar_events_csv", ""))
    consecutive_runtime_errors = event_stats.get("consecutive_runtime_errors")
    final_recovered = event_stats.get("final_verification_recovered")
    if runtime_errors > 0.0:
        warnings.append("sidecar_runtime_errors")
        if runtime_ratio > SIDECAR_RUNTIME_ERROR_RATIO_FAIL:
            failed.append("sidecar_runtime_error_ratio")
        if runtime_errors >= SIDECAR_RUNTIME_ERROR_COUNT_FAIL:
            failed.append("sidecar_runtime_errors")
        if (
                consecutive_runtime_errors is not None and
                consecutive_runtime_errors >= SIDECAR_CONSECUTIVE_RUNTIME_ERROR_FAIL):
            failed.append("consecutive_runtime_errors")
        if final_recovered == 0:
            failed.append("runtime_errors_no_recovery")
        if event_stats.get("command_readback_mismatch_count", 0) > 0:
            warnings.append("minor_command_readback_mismatch")
        if event_stats.get("runtime_timeout_count", 0) > 0:
            warnings.append("sidecar_runtime_timeout")

    for field in (
            "fallback_ratio",
            "low_speed_mask_ratio",
            "soft_safety_gain_ratio"):
        if (_float_first(source, (field,)) or 0.0) > 0.0:
            warnings.append(field)
    return _dedupe(failed), _dedupe(warnings)


def _suite_metrics(
    source: Mapping[str, Any],
    suite_summary_path: str,
    route_return_code: int,
) -> Dict[str, Any]:
    policy_ratio = _float_first(source, ("rl_policy_available_ratio", "policy_available_ratio"))
    action = _float_first(source, ("rl_mean_abs_action", "mean_abs_action"))
    residual = _float_first(
        source,
        ("rl_mean_abs_residual_damper", "mean_abs_residual_damper"))
    runtime_errors = _float_first(source, ("sidecar_runtime_errors",)) or 0.0
    verifies = max(_float_first(source, ("sidecar_command_verifies",)) or 0.0, 1.0)
    event_stats = _sidecar_event_stats(source.get("sidecar_events_csv", ""))
    return {
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
        "rl_fallback_rows": source.get("rl_fallback_rows", ""),
        "rl_fallback_reasons": source.get("rl_fallback_reasons", ""),
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
        "sidecar_runtime_error_ratio": runtime_errors / verifies,
        "sidecar_consecutive_runtime_errors": _blank_if_none(
            event_stats.get("consecutive_runtime_errors")),
        "sidecar_final_verification_recovered": _blank_if_none(
            event_stats.get("final_verification_recovered")),
        "diagnostic_rows": source.get("diagnostic_rows", source.get("rl_diagnostic_rows", "")),
        "reward_rows": source.get("reward_rows", ""),
        "reward_row_ratio": source.get("reward_row_ratio", ""),
    }


def _manifest_row(
    context: Mapping[str, Any],
    *,
    acceptance: Mapping[str, Any],
    artifact_status: str,
    suite_summary_path: str = "",
) -> Dict[str, Any]:
    identity = "|".join((
        str(context.get("source_train_artifact_id", "")),
        str(context.get("policy_sha256", "")),
        str(context.get("normalizer_sha256", "")),
        str(context.get("routes_subset", "")),
        str(context.get("seed", "")),
        str(context.get("output_dir", "")),
    ))
    artifact_id = "phase4d-eval-s4-%s" % hashlib.sha256(
        identity.encode("utf-8")).hexdigest()[:12]
    return _ordered_row({
        "artifact_id": artifact_id,
        "workflow_role": "eval_s4_only",
        "artifact_status": artifact_status,
        "phase": "phase4d",
        "source_train_artifact_id": context.get("source_train_artifact_id", ""),
        "source_train_run_dir": context.get("train_dir", ""),
        "train_dir": context.get("train_dir", ""),
        "eval_dir": context.get("output_dir", ""),
        "train_acceptance_path": context.get("train_acceptance_path", ""),
        "train_manifest_path": context.get("train_manifest_path", ""),
        "training_summary_path": context.get("training_summary_path", ""),
        "policy_path": context.get("policy_path", ""),
        "normalizer_path": context.get("normalizer_path", ""),
        "policy_sha256": context.get("policy_sha256", ""),
        "normalizer_sha256": context.get("normalizer_sha256", ""),
        "baseline": context.get("baseline", ""),
        "planning_provider": context.get("planning_provider", ""),
        "action_semantics": context.get("action_semantics", ""),
        "routes_subset": context.get("routes_subset", ""),
        "eval_seed": context.get("seed", ""),
        "suite_summary_path": suite_summary_path or os.path.join(
            str(context.get("output_dir", "")),
            "suite_summary.csv"),
        "acceptance_path": os.path.join(
            str(context.get("output_dir", "")),
            "phase4d_eval_s4_acceptance.json"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase4d_eval_s4_status": acceptance.get("phase4d_eval_s4_status", ""),
        "eval_status": acceptance.get("eval_status", ""),
        "failed_checks": acceptance.get("failed_checks", ""),
        "warnings": acceptance.get("warnings", ""),
    }, PHASE4D_EVAL_S4_MANIFEST_FIELDS)


def _print_shell_exports(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    values = {
        "PHASE4D_PREFLIGHT_OK": "1" if row.get("eval_status") == "ready_for_eval" else "0",
        "PHASE4D_EVAL_STATUS": row.get("eval_status", ""),
        "PHASE4D_FAILED_CHECKS": row.get("failed_checks", ""),
        "PHASE4D_POLICY_PATH": row.get("policy_path", ""),
        "PHASE4D_NORMALIZER_PATH": row.get("normalizer_path", ""),
        "PHASE4D_ROUTES_SUBSET": row.get("routes_subset", ""),
        "PHASE4D_SEED": row.get("seed", ""),
        "PHASE4D_TRAIN_MANIFEST": row.get("train_manifest_path", ""),
        "PHASE4D_EVAL_MANIFEST": os.path.join(
            str(row.get("output_dir", "")),
            "phase4d_eval_s4_manifest.json"),
        "PHASE4D_EVAL_ARTIFACT_ID": manifest.get("artifact_id", ""),
    }
    for key, value in values.items():
        print("%s=%s" % (key, shlex.quote(str(value))))


def _write_acceptance_outputs(output_dir: str, row: Mapping[str, Any]) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "phase4d_eval_s4_acceptance.csv")
    json_path = os.path.join(output_dir, "phase4d_eval_s4_acceptance.json")
    _write_single_row_csv(csv_path, row, PHASE4D_EVAL_S4_ACCEPTANCE_FIELDS)
    _write_json(json_path, row)
    return csv_path, json_path


def _write_manifest_outputs(output_dir: str, row: Mapping[str, Any]) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "phase4d_eval_s4_manifest.json")
    csv_path = os.path.join(output_dir, "phase4d_eval_s4_manifest.csv")
    _write_json(json_path, row)
    _write_single_row_csv(csv_path, row, PHASE4D_EVAL_S4_MANIFEST_FIELDS)
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


def _select_s4_row(rows: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for row in rows:
        scenario = str(row.get("scenario", ""))
        if scenario == S4_SCENARIO or scenario == "rl_residual_skyhook":
            return row
    return rows[0] if len(rows) == 1 else None


def _sidecar_event_stats(path: Any) -> Dict[str, Any]:
    resolved = _resolve_existing_path(str(path or ""))
    result: Dict[str, Any] = {
        "consecutive_runtime_errors": None,
        "final_verification_recovered": None,
        "command_readback_mismatch_count": 0,
        "runtime_timeout_count": 0,
    }
    if not resolved:
        return result
    max_consecutive = 0
    current_consecutive = 0
    seen_runtime = False
    recovered_after_last_runtime = None
    try:
        with open(resolved, newline="") as csv_file:
            for row in csv.DictReader(csv_file):
                event = str(row.get("event", ""))
                message = str(row.get("message", ""))
                if event == "runtime_error":
                    seen_runtime = True
                    recovered_after_last_runtime = 0
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                    if "scale mismatch" in message or "readback" in message:
                        result["command_readback_mismatch_count"] += 1
                    if "time-out" in message or "timeout" in message:
                        result["runtime_timeout_count"] += 1
                else:
                    current_consecutive = 0
                    if seen_runtime and event == "command_verified":
                        recovered_after_last_runtime = 1
    except OSError:
        return result
    result["consecutive_runtime_errors"] = max_consecutive
    result["final_verification_recovered"] = recovered_after_last_runtime
    return result


def _path_matches_artifact(path: str, expected_values: Sequence[Any], train_dir: str) -> bool:
    resolved = _resolve_path(path, train_dir)
    expected_paths = [
        _resolve_path(value, train_dir)
        for value in expected_values
        if str(value or "").strip()
    ]
    if not expected_paths:
        return True
    if resolved in expected_paths:
        return True
    for expected in expected_paths:
        if (
                os.path.basename(resolved) == os.path.basename(expected) and
                _is_under(resolved, train_dir) and
                _is_under(expected, train_dir)):
            return True
    return False


def _is_under(path: str, root: str) -> bool:
    try:
        common = os.path.commonpath([_abs_path(path), _abs_path(root)])
    except ValueError:
        return False
    return common == _abs_path(root)


def _sha256_file(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if value.startswith("/workspace/"):
            host_value = os.path.join(_sim_root(), value[len("/workspace/"):])
            if os.path.exists(host_value):
                return os.path.abspath(host_value)
        return value
    return os.path.abspath(os.path.join(base_dir, value))


def _resolve_existing_path(path: str) -> str:
    value = os.path.expanduser(str(path or ""))
    if not value:
        return ""
    candidates = [value]
    if value.startswith("/workspace/"):
        candidates.append(os.path.join(_sim_root(), value[len("/workspace/"):]))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return ""


def _sim_root() -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "..",
    ))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _nonempty(value: Any) -> str:
    return str(value or "").strip()


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


def _float_missing_or_below(value: Optional[float], threshold: float) -> bool:
    return value is None or value < threshold


def _float_missing_or_at_most(value: Optional[float], threshold: float) -> bool:
    return value is None or value <= threshold


def _dedupe(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())

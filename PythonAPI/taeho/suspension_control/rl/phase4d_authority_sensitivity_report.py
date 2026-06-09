"""Combined CARLA-free report for Phase 4-D authority diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


DIAGNOSIS_UNDER_ACTUATED_POLICY = "DIAGNOSIS_UNDER_ACTUATED_POLICY"
DIAGNOSIS_LOW_RESIDUAL_AUTHORITY = "DIAGNOSIS_LOW_RESIDUAL_AUTHORITY"
DIAGNOSIS_REWARD_TOO_CONSERVATIVE = "DIAGNOSIS_REWARD_TOO_CONSERVATIVE"
DIAGNOSIS_SAFETY_LIMITED_AUTHORITY = "DIAGNOSIS_SAFETY_LIMITED_AUTHORITY"
DIAGNOSIS_MIXED_TRADEOFF = "DIAGNOSIS_MIXED_TRADEOFF"
DIAGNOSIS_INCOMPLETE = "DIAGNOSIS_INCOMPLETE"

REPORT_FIELDS: Tuple[str, ...] = (
    "phase4d_authority_sensitivity_status",
    "diagnosis",
    "diagnosis_ok",
    "failed_checks",
    "warnings",
    "authority_status",
    "scripted_status",
    "authority_aggregate_ok",
    "scripted_aggregate_ok",
    "authority_diagnosis",
    "scripted_diagnosis",
    "authority_monotonicity_ok",
    "authority_metric_effect_above_noise",
    "scripted_effect_above_noise",
    "signed_residual_direction_consistency",
    "magnitude_response_consistency",
    "safety_limited_authority",
    "clamp_limited_authority",
    "authority_max_final_mean_abs_residual_damper",
    "scripted_max_final_mean_abs_residual_damper",
    "stageA_report_loaded",
    "stageA_report_path",
    "authority_dir",
    "scripted_dir",
    "authority_report_json",
    "scripted_report_json",
    "combined_report_json",
    "combined_report_csv",
    "recommended_next_action",
    "paper_claim_warning",
)


def write_authority_sensitivity_report(
    *,
    authority_dir: str,
    scripted_dir: str,
    stageA_report: str = "",
    output_dir: str,
) -> Tuple[str, str, Dict[str, Any]]:
    """Write a combined report from already aggregated diagnostic outputs."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    authority_dir = _resolve_workspace_path(authority_dir)
    scripted_dir = _resolve_workspace_path(scripted_dir)
    stageA_report = _resolve_workspace_path(stageA_report)
    authority_report_path = os.path.join(authority_dir, "phase4d_action_authority_report.json")
    scripted_report_path = os.path.join(scripted_dir, "phase4d_scripted_residual_report.json")
    authority_acceptance_path = os.path.join(authority_dir, "phase4d_action_authority_acceptance.json")
    scripted_acceptance_path = os.path.join(scripted_dir, "phase4d_scripted_residual_acceptance.json")

    authority_report = _read_json_dict(authority_report_path)
    scripted_report = _read_json_dict(scripted_report_path)
    authority_acceptance = _read_json_dict(authority_acceptance_path)
    scripted_acceptance = _read_json_dict(scripted_acceptance_path)
    stageA_doc = _read_json_dict(stageA_report) if stageA_report else {}

    report = _combined_report(
        authority_dir=authority_dir,
        scripted_dir=scripted_dir,
        stageA_report_path=stageA_report,
        stageA_report=stageA_doc,
        authority_report_path=authority_report_path,
        scripted_report_path=scripted_report_path,
        authority_report=authority_report,
        scripted_report=scripted_report,
        authority_acceptance=authority_acceptance,
        scripted_acceptance=scripted_acceptance,
        output_dir=output_dir,
    )
    report_json = os.path.join(output_dir, "phase4d_authority_sensitivity_diagnostic_report.json")
    report_csv = os.path.join(output_dir, "phase4d_authority_sensitivity_diagnostic_report.csv")
    report["combined_report_json"] = report_json
    report["combined_report_csv"] = report_csv
    report["acceptance"]["combined_report_json"] = report_json
    report["acceptance"]["combined_report_csv"] = report_csv
    _write_json(report_json, report)
    _write_single_row_csv(report_csv, report["acceptance"], REPORT_FIELDS)
    return report_csv, report_json, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-dir", required=True)
    parser.add_argument("--scripted-dir", required=True)
    parser.add_argument("--stageA-report", default="")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report_csv, report_json, report = write_authority_sensitivity_report(
        authority_dir=args.authority_dir,
        scripted_dir=args.scripted_dir,
        stageA_report=args.stageA_report,
        output_dir=args.output_dir,
    )
    print("phase4d authority sensitivity diagnostic report: %s" % report_json)
    print("phase4d authority sensitivity diagnostic csv: %s" % report_csv)
    print("phase4d authority sensitivity diagnosis: %s failed_checks=%s" % (
        report["diagnosis"],
        ";".join(report.get("failed_checks") or [])))
    return 0


def _combined_report(
    *,
    authority_dir: str,
    scripted_dir: str,
    stageA_report_path: str,
    stageA_report: Mapping[str, Any],
    authority_report_path: str,
    scripted_report_path: str,
    authority_report: Mapping[str, Any],
    scripted_report: Mapping[str, Any],
    authority_acceptance: Mapping[str, Any],
    scripted_acceptance: Mapping[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    authority_diag = _dict_value(authority_report.get("diagnosis"))
    scripted_diag = _dict_value(scripted_report.get("diagnosis"))
    authority_status = _first_nonempty(
        authority_report.get("phase4d_action_authority_status"),
        authority_acceptance.get("phase4d_action_authority_status"))
    scripted_status = _first_nonempty(
        scripted_report.get("phase4d_scripted_residual_status"),
        scripted_acceptance.get("phase4d_scripted_residual_status"))
    authority_ok = _boolish(_first_nonempty(
        authority_report.get("aggregate_ok"),
        authority_acceptance.get("aggregate_ok")))
    scripted_ok = _boolish(_first_nonempty(
        scripted_report.get("aggregate_ok"),
        scripted_acceptance.get("aggregate_ok")))
    authority_missing = not bool(authority_report)
    scripted_missing = not bool(scripted_report)
    incomplete = (
        authority_missing or
        scripted_missing or
        _status_is_incomplete(authority_status) or
        _status_is_incomplete(scripted_status))

    authority_safety = (
        "FAIL_SAFETY" in authority_status or
        _boolish(authority_diag.get("safety_limited_gain")))
    scripted_safety = (
        "FAIL_SAFETY" in scripted_status or
        str(scripted_diag.get("SAFETY_AUTHORITY_STATUS", "")) == "safety_limited")
    clamp_limited = _boolish(authority_diag.get("clamp_limited_gain"))
    safety_limited = authority_safety or scripted_safety
    authority_effect = _boolish(authority_diag.get("metric_effect_above_noise"))
    scripted_effect = _boolish(scripted_diag.get("scripted_effect_above_noise"))
    authority_monotonic = _boolish(authority_diag.get("authority_monotonicity_ok"))
    signed_ok = _boolish(scripted_diag.get("signed_residual_direction_consistency"))
    magnitude_ok = _boolish(scripted_diag.get("magnitude_response_consistency"))
    authority_max_residual = _max_row_float(authority_report, "final_mean_abs_residual_damper")
    scripted_max_residual = _max_row_float(scripted_report, "final_mean_abs_residual_damper")

    failed_checks = []
    warnings = []
    if authority_missing:
        failed_checks.append("authority_report_missing")
    if scripted_missing:
        failed_checks.append("scripted_report_missing")
    if incomplete:
        failed_checks.append("incomplete_inputs")
    if not authority_ok:
        warnings.append("authority_aggregate_not_ok")
    if not scripted_ok:
        warnings.append("scripted_aggregate_not_ok")
    if clamp_limited:
        warnings.append("authority_clamp_or_saturation_limited")
    if not authority_effect:
        warnings.append("learned_policy_effect_not_above_stageA_noise")
    if not scripted_effect:
        warnings.append("scripted_effect_not_above_stageA_noise")
    if scripted_effect and not signed_ok:
        warnings.append("scripted_signed_direction_inconsistent")
    if scripted_effect and not magnitude_ok:
        warnings.append("scripted_magnitude_response_inconsistent")

    diagnosis = _diagnosis_label(
        incomplete=incomplete,
        safety_limited=safety_limited,
        authority_effect=authority_effect,
        scripted_effect=scripted_effect,
        authority_monotonic=authority_monotonic,
        signed_ok=signed_ok,
        magnitude_ok=magnitude_ok,
        authority_max_residual=authority_max_residual,
    )
    recommended = _recommended_next_action(
        diagnosis=diagnosis,
        authority_status=authority_status,
        scripted_status=scripted_status)
    status = "complete" if diagnosis != DIAGNOSIS_INCOMPLETE else "incomplete"
    paper_warning = (
        "This report combines eval-only diagnostics. action_scale>1 and "
        "large scripted residuals are authority probes, not paper-level "
        "performance claims.")
    row = {
        "phase4d_authority_sensitivity_status": status,
        "diagnosis": diagnosis,
        "diagnosis_ok": int(diagnosis != DIAGNOSIS_INCOMPLETE),
        "failed_checks": ";".join(_dedupe(failed_checks)),
        "warnings": ";".join(_dedupe(warnings)),
        "authority_status": authority_status,
        "scripted_status": scripted_status,
        "authority_aggregate_ok": int(authority_ok),
        "scripted_aggregate_ok": int(scripted_ok),
        "authority_diagnosis": authority_diag.get("AUTHORITY_DIAGNOSIS", ""),
        "scripted_diagnosis": scripted_diag.get("SENSITIVITY_DIAGNOSIS", ""),
        "authority_monotonicity_ok": int(authority_monotonic),
        "authority_metric_effect_above_noise": int(authority_effect),
        "scripted_effect_above_noise": int(scripted_effect),
        "signed_residual_direction_consistency": int(signed_ok),
        "magnitude_response_consistency": int(magnitude_ok),
        "safety_limited_authority": int(safety_limited),
        "clamp_limited_authority": int(clamp_limited),
        "authority_max_final_mean_abs_residual_damper": _blank_if_none(authority_max_residual),
        "scripted_max_final_mean_abs_residual_damper": _blank_if_none(scripted_max_residual),
        "stageA_report_loaded": int(bool(stageA_report)),
        "stageA_report_path": stageA_report_path,
        "authority_dir": authority_dir,
        "scripted_dir": scripted_dir,
        "authority_report_json": authority_report_path,
        "scripted_report_json": scripted_report_path,
        "combined_report_json": os.path.join(output_dir, "phase4d_authority_sensitivity_diagnostic_report.json"),
        "combined_report_csv": os.path.join(output_dir, "phase4d_authority_sensitivity_diagnostic_report.csv"),
        "recommended_next_action": recommended,
        "paper_claim_warning": paper_warning,
    }
    return {
        **row,
        "phase4d_authority_sensitivity_status": status,
        "diagnosis": diagnosis,
        "diagnosis_ok": row["diagnosis_ok"],
        "failed_checks": _dedupe(failed_checks),
        "warnings": _dedupe(warnings),
        "acceptance": _ordered_row(row, REPORT_FIELDS),
        "authority_report": authority_report,
        "scripted_report": scripted_report,
        "stageA_report_loaded": int(bool(stageA_report)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_claim_warning": paper_warning,
    }


def _diagnosis_label(
    *,
    incomplete: bool,
    safety_limited: bool,
    authority_effect: bool,
    scripted_effect: bool,
    authority_monotonic: bool,
    signed_ok: bool,
    magnitude_ok: bool,
    authority_max_residual: Optional[float],
) -> str:
    if incomplete:
        return DIAGNOSIS_INCOMPLETE
    if safety_limited:
        return DIAGNOSIS_SAFETY_LIMITED_AUTHORITY
    if scripted_effect and not authority_effect:
        if authority_monotonic and (authority_max_residual or 0.0) > 1.0e-6:
            return DIAGNOSIS_REWARD_TOO_CONSERVATIVE
        return DIAGNOSIS_UNDER_ACTUATED_POLICY
    if not scripted_effect:
        return DIAGNOSIS_LOW_RESIDUAL_AUTHORITY
    if not signed_ok or not magnitude_ok:
        return DIAGNOSIS_MIXED_TRADEOFF
    if authority_effect and scripted_effect:
        return DIAGNOSIS_MIXED_TRADEOFF
    return DIAGNOSIS_MIXED_TRADEOFF


def _recommended_next_action(*, diagnosis: str, authority_status: str, scripted_status: str) -> str:
    del authority_status, scripted_status
    if diagnosis == DIAGNOSIS_INCOMPLETE:
        return "complete_or_rerun_missing_authority_and_scripted_aggregate_outputs"
    if diagnosis == DIAGNOSIS_SAFETY_LIMITED_AUTHORITY:
        return "inspect_safety_gates_and_clamp_limits_before_increasing_residual_authority"
    if diagnosis == DIAGNOSIS_UNDER_ACTUATED_POLICY:
        return "increase_or_debug_learned_policy_residual_authority_before_claiming_performance"
    if diagnosis == DIAGNOSIS_LOW_RESIDUAL_AUTHORITY:
        return "increase_scripted_residual_probe_or_check_metric_noise_floor_before_reward_changes"
    if diagnosis == DIAGNOSIS_REWARD_TOO_CONSERVATIVE:
        return "keep_action_path_fixed_and_revisit_reward_or_training_exploration"
    return "review_metric_tradeoffs_before_policy_selection_or_next_training_change"


def _status_is_incomplete(status: str) -> bool:
    return (not status) or status.endswith("INCOMPLETE") or status == "preflight_failed"


def _max_row_float(report: Mapping[str, Any], key: str) -> Optional[float]:
    rows = report.get("rows")
    if not isinstance(rows, list):
        return None
    values = []
    for row in rows:
        if isinstance(row, Mapping):
            value = _float_value(row.get(key))
            if value is not None:
                values.append(abs(value))
    if not values:
        return None
    return max(values)


def _read_json_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_single_row_csv(path: str, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerow({field: _format_value(row.get(field, "")) for field in fields})


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


def _dict_value(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value not in (None, ""):
            return str(value)
    return ""


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "on", "pass", "ok")


def _float_value(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _blank_if_none(value: Optional[float]) -> Any:
    if value is None:
        return ""
    return value


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        text = ("%.12g" % value)
        return "0" if text == "-0" else text
    return value


def _ordered_row(row: Mapping[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    return {field: row.get(field, "") for field in fields}


def _dedupe(values: Sequence[str]) -> Sequence[str]:
    seen = set()
    result = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

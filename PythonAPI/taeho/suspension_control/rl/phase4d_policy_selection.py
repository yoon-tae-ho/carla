"""Phase 4-D no-planning policy selection report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .phase4_acceptance import (
    PHASE4B_LEARNED_SCENARIO,
    PHASE4B_REFERENCE_SCENARIO,
)
from .phase4b_policy_eval_acceptance import (
    load_phase4b_summary_rows,
    phase4b_split_comparison_rows,
    phase4b_split_reference_acceptance_rows,
)


FORBIDDEN_FALLBACK_REASONS: Tuple[str, ...] = (
    "policy_unavailable",
    "missing_path",
    "missing_file",
    "policy_file_missing",
    "torch_unavailable",
    "load_failed",
)
ALLOWED_FALLBACK_REASONS: Tuple[str, ...] = ("safety_gate_zero",)


PHASE4D_METRICS: Tuple[Tuple[str, Tuple[str, ...], str], ...] = (
    ("mean_reward_total", ("mean_reward_total",), "higher"),
    ("mean_reward_comfort", ("mean_reward_comfort", "reward_comfort_mean"), "higher"),
    ("mean_reward_stability", ("mean_reward_stability", "reward_stability_mean"), "higher"),
    ("mean_reward_task", ("mean_reward_task", "reward_task_mean"), "higher"),
    ("mean_reward_action", ("mean_reward_action", "reward_action_mean"), "higher"),
    ("mean_reward_safety", ("mean_reward_safety", "reward_safety_mean"), "higher"),
    ("warmup_excluded_comfort_comfort_score", (
        "warmup_excluded_comfort_comfort_score",
    ), "higher"),
    ("warmup_excluded_comfort_rms_vertical_acc", (
        "warmup_excluded_comfort_rms_vertical_acc",
    ), "lower"),
    ("warmup_excluded_comfort_rms_lateral_acc", (
        "warmup_excluded_comfort_rms_lateral_acc",
    ), "lower"),
    ("warmup_excluded_stability_peak_abs_roll", (
        "warmup_excluded_stability_peak_abs_roll",
        "stability_peak_abs_roll",
    ), "lower"),
    ("warmup_excluded_stability_rms_roll", (
        "warmup_excluded_stability_rms_roll",
    ), "lower"),
    ("warmup_excluded_stability_peak_abs_lateral_acc", (
        "warmup_excluded_stability_peak_abs_lateral_acc",
    ), "lower"),
    ("warmup_excluded_stability_rms_lateral_acc", (
        "warmup_excluded_stability_rms_lateral_acc",
    ), "lower"),
    ("duration_game", ("duration_game",), "lower"),
    ("route_progress_stall_ratio", ("route_progress_stall_ratio",), "lower"),
    ("low_speed_mask_ratio", ("low_speed_mask_ratio",), "lower"),
    ("fallback_ratio", ("fallback_ratio",), "lower"),
    ("effective_control_ratio", ("effective_control_ratio",), "higher"),
    ("mean_abs_action", ("mean_abs_action", "rl_mean_abs_action"), "diagnostic"),
    ("mean_abs_residual_damper", (
        "mean_abs_residual_damper",
        "rl_mean_abs_residual_damper",
    ), "diagnostic"),
)


PHASE4D_GATE_FIELDS: Tuple[str, ...] = (
    "route_score_ok",
    "score_composed_ok",
    "score_penalty_ok",
    "infraction_ok",
    "verification_ok",
    "policy_available_ok",
    "no_policy_fallback_ok",
    "allowed_fallback_only_ok",
    "route_completion_ok",
    "reward_present_ok",
    "action_nonzero_ok",
    "residual_nonzero_ok",
    "effective_control_ok",
    "fallback_ratio_ok",
    "low_speed_mask_ok",
    "hard_safety_ok",
    "observation_clip_ok",
)


PHASE4D_SELECTION_FIELDS: Tuple[str, ...] = (
    "phase4d_status",
    "engineering_pass",
    "reward_acceptable",
    "comfort_promising",
    "stability_risk",
    "action_overuse_warning",
    "policy_tags",
    "failed_checks",
    "warnings",
    "candidate_scenario",
    "reference_scenario",
    "candidate_summary",
    "reference_summary",
    "training_summary",
    "training_status",
    "training_backend",
    "total_timesteps_requested",
    "total_timesteps_collected",
) + PHASE4D_GATE_FIELDS + tuple(
    field
    for metric, _keys, _direction in PHASE4D_METRICS
    for field in (
        "%s_candidate" % metric,
        "%s_reference" % metric,
        "%s_delta" % metric,
        "%s_pct_change" % metric,
        "%s_improvement_pct" % metric,
    )
)


def write_phase4d_policy_selection(
    output_dir: str,
    *,
    candidate_summary_path: str,
    reference_summary_path: str,
    training_summary_path: str = "",
    candidate_scenario: str = PHASE4B_LEARNED_SCENARIO,
    reference_scenario: str = PHASE4B_REFERENCE_SCENARIO,
) -> Tuple[str, str, Dict[str, Any]]:
    """Write Phase 4-D policy-selection CSV/JSON."""

    output_dir = _abs_path(output_dir)
    report = phase4d_policy_selection_report(
        candidate_summary_path=candidate_summary_path,
        reference_summary_path=reference_summary_path,
        training_summary_path=training_summary_path,
        candidate_scenario=candidate_scenario,
        reference_scenario=reference_scenario,
    )
    csv_path = os.path.join(output_dir, "phase4d_policy_selection.csv")
    json_path = os.path.join(output_dir, "phase4d_policy_selection.json")
    _write_single_row_csv(csv_path, report["summary"], PHASE4D_SELECTION_FIELDS)
    _write_json(json_path, report)
    return csv_path, json_path, report


def phase4d_policy_selection_report(
    *,
    candidate_summary_path: str,
    reference_summary_path: str,
    training_summary_path: str = "",
    candidate_scenario: str = PHASE4B_LEARNED_SCENARIO,
    reference_scenario: str = PHASE4B_REFERENCE_SCENARIO,
) -> Dict[str, Any]:
    candidate_path = _resolve_path(candidate_summary_path)
    reference_path = _resolve_path(reference_summary_path)
    training_path = _resolve_path(training_summary_path)
    candidate_rows = load_phase4b_summary_rows(
        candidate_path,
        scenario_hint=candidate_scenario,
    )
    reference_rows = load_phase4b_summary_rows(
        reference_path,
        scenario_hint=reference_scenario,
    )
    candidate = _select_scenario_row(candidate_rows, candidate_scenario)
    reference = _select_scenario_row(reference_rows, reference_scenario)
    acceptance_rows = phase4b_split_reference_acceptance_rows(
        candidate_rows,
        reference_rows,
        candidate_scenario=candidate_scenario,
        reference_scenario=reference_scenario,
        reference_summary_provided=bool(reference_path),
    )
    split_comparison = phase4b_split_comparison_rows(
        candidate,
        reference,
        acceptance_rows,
        candidate_summary_path=candidate_path,
        reference_summary_path=reference_path,
        candidate_scenario=candidate_scenario,
        reference_scenario=reference_scenario,
    )[0]
    training_summary = _read_json_dict(training_path)

    gates = _engineering_gates(candidate, acceptance_rows)
    failed_checks = [name[:-3] for name, ok in gates.items() if not ok]
    warnings: List[str] = []
    if candidate is None:
        warnings.append("candidate_missing")
    if reference is None:
        warnings.append("reference_missing")
    if training_path and not training_summary:
        warnings.append("training_summary_unreadable")
    if not training_path:
        warnings.append("training_summary_missing")

    metric_details = _metric_details(candidate, reference)
    reward_acceptable = _reward_acceptable(metric_details)
    comfort_promising = _comfort_promising(metric_details)
    stability_risk = _stability_risk(metric_details)
    action_overuse = _action_overuse(metric_details)
    if action_overuse:
        warnings.append("action_overuse")

    engineering_pass = int(all(gates.values()))
    phase4d_status = _phase4d_status(
        engineering_pass=engineering_pass,
        reward_acceptable=reward_acceptable,
        comfort_promising=comfort_promising,
        stability_risk=stability_risk,
    )
    tags = _policy_tags(
        engineering_pass=engineering_pass,
        reward_acceptable=reward_acceptable,
        comfort_promising=comfort_promising,
        stability_risk=stability_risk,
        action_overuse=action_overuse,
    )

    row: Dict[str, Any] = {
        "phase4d_status": phase4d_status,
        "engineering_pass": engineering_pass,
        "reward_acceptable": int(reward_acceptable),
        "comfort_promising": int(comfort_promising),
        "stability_risk": int(stability_risk),
        "action_overuse_warning": int(action_overuse),
        "policy_tags": ";".join(tags),
        "failed_checks": ";".join(failed_checks),
        "warnings": ";".join(_dedup(warnings)),
        "candidate_scenario": candidate_scenario,
        "reference_scenario": reference_scenario,
        "candidate_summary": candidate_path,
        "reference_summary": reference_path,
        "training_summary": training_path,
        "training_status": training_summary.get("status", ""),
        "training_backend": training_summary.get("backend", ""),
        "total_timesteps_requested": _first_value(
            training_summary,
            ("total_timesteps", "total_timesteps_requested"),
        ),
        "total_timesteps_collected": _first_value(
            training_summary,
            ("rollout_rows", "total_timesteps_collected"),
        ),
    }
    row.update({field: int(gates.get(field, 0)) for field in PHASE4D_GATE_FIELDS})
    for metric, detail in metric_details.items():
        row["%s_candidate" % metric] = _blank_if_none(detail.get("candidate"))
        row["%s_reference" % metric] = _blank_if_none(detail.get("reference"))
        row["%s_delta" % metric] = _blank_if_none(detail.get("delta"))
        row["%s_pct_change" % metric] = _blank_if_none(detail.get("pct_change"))
        row["%s_improvement_pct" % metric] = _blank_if_none(detail.get("improvement_pct"))
    row = _ordered_row(row, PHASE4D_SELECTION_FIELDS)

    return {
        "summary": row,
        "phase4d_status": phase4d_status,
        "policy_tags": tags,
        "failed_checks": failed_checks,
        "warnings": _dedup(warnings),
        "gates": gates,
        "metrics": metric_details,
        "acceptance": acceptance_rows,
        "split_comparison": split_comparison,
        "training_summary": training_summary,
        "inputs": {
            "candidate_summary": candidate_path,
            "reference_summary": reference_path,
            "training_summary": training_path,
            "candidate_scenario": candidate_scenario,
            "reference_scenario": reference_scenario,
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--reference-summary", required=True)
    parser.add_argument("--training-summary", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-scenario", default=PHASE4B_LEARNED_SCENARIO)
    parser.add_argument("--reference-scenario", default=PHASE4B_REFERENCE_SCENARIO)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    csv_path, json_path, report = write_phase4d_policy_selection(
        args.output_dir,
        candidate_summary_path=args.candidate_summary,
        reference_summary_path=args.reference_summary,
        training_summary_path=args.training_summary,
        candidate_scenario=args.candidate_scenario,
        reference_scenario=args.reference_scenario,
    )
    print("phase4d policy selection: %s" % csv_path)
    print("phase4d policy status: %s" % report.get("phase4d_status", ""))
    print("phase4d policy tags: %s" % ";".join(report.get("policy_tags", [])))
    print("phase4d policy selection json: %s" % json_path)
    return 0


def _engineering_gates(
    candidate: Optional[Mapping[str, Any]],
    acceptance_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    learned_acceptance = _acceptance_role(acceptance_rows, "learned")
    candidate = candidate or {}
    fallback_reasons = _reason_set(candidate.get("rl_fallback_reasons"))
    reward_ratio = _reward_row_ratio(candidate)
    route_completion = _float_value(candidate.get("route_completion_proxy"))
    route_progress = _float_value(candidate.get("route_progress_monotonic_fraction_max"))
    fallback_ratio = _float_value(candidate.get("fallback_ratio"))
    low_speed_mask_ratio = _float_value(candidate.get("low_speed_mask_ratio"))
    hard_safety_gate_ratio = _float_value(candidate.get("hard_safety_gate_ratio"))
    observation_clip_ratio = _float_value(candidate.get("observation_clip_ratio"))
    gates = {
        "route_score_ok": _score_equals(candidate.get("score_route"), 100.0),
        "score_composed_ok": _score_equals(candidate.get("score_composed"), 100.0),
        "score_penalty_ok": _score_equals(candidate.get("score_penalty"), 1.0),
        "infraction_ok": int(_int_value(learned_acceptance.get("infraction_ok")) == 1),
        "verification_ok": int((_float_value(candidate.get("sidecar_command_verifies")) or 0.0) > 0.0),
        "policy_available_ok": int(
            (_float_first(candidate, ("rl_policy_available_ratio", "policy_available_ratio")) or 0.0)
            >= 0.99
        ),
        "no_policy_fallback_ok": int(
            not any(reason in FORBIDDEN_FALLBACK_REASONS for reason in fallback_reasons)
        ),
        "allowed_fallback_only_ok": int(
            all(reason in ALLOWED_FALLBACK_REASONS for reason in fallback_reasons)
        ),
        "route_completion_ok": int(
            route_completion is not None and
            abs(route_completion - 1.0) <= 1.0e-9 and
            route_progress is not None and
            route_progress >= 0.98
        ),
        "reward_present_ok": int(reward_ratio is not None and reward_ratio >= 0.95),
        "action_nonzero_ok": int(
            (_float_first(candidate, ("rl_mean_abs_action", "mean_abs_action")) or 0.0) > 0.01
        ),
        "residual_nonzero_ok": int(
            (_float_first(candidate, (
                "rl_mean_abs_residual_damper",
                "mean_abs_residual_damper",
            )) or 0.0) > 0.001
        ),
        "effective_control_ok": int(
            (_float_value(candidate.get("effective_control_ratio")) or 0.0) >= 0.50
        ),
        "fallback_ratio_ok": int(
            fallback_ratio is not None and fallback_ratio <= 0.45
        ),
        "low_speed_mask_ok": int(
            low_speed_mask_ratio is not None and low_speed_mask_ratio <= 0.45
        ),
        "hard_safety_ok": int(
            hard_safety_gate_ratio is not None and
            abs(hard_safety_gate_ratio) <= 1.0e-12
        ),
        "observation_clip_ok": int(
            observation_clip_ratio is not None and observation_clip_ratio <= 0.05
        ),
    }
    return gates


def _metric_details(
    candidate: Optional[Mapping[str, Any]],
    reference: Optional[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    candidate = candidate or {}
    reference = reference or {}
    for metric, keys, direction in PHASE4D_METRICS:
        candidate_value = _float_first(candidate, keys)
        reference_value = _float_first(reference, keys)
        delta = (
            candidate_value - reference_value
            if candidate_value is not None and reference_value is not None
            else None
        )
        pct_change = _pct_change(candidate_value, reference_value)
        improvement_pct = _improvement_pct(candidate_value, reference_value, direction)
        result[metric] = {
            "candidate": candidate_value,
            "reference": reference_value,
            "delta": delta,
            "pct_change": pct_change,
            "improvement_pct": improvement_pct,
            "direction": direction,
        }
    return result


def _reward_acceptable(metrics: Mapping[str, Mapping[str, Any]]) -> bool:
    detail = metrics.get("mean_reward_total", {})
    candidate = detail.get("candidate")
    reference = detail.get("reference")
    return (
        candidate is not None and
        reference is not None and
        candidate >= reference - 0.10
    )


def _comfort_promising(metrics: Mapping[str, Mapping[str, Any]]) -> bool:
    comfort_score = metrics.get("warmup_excluded_comfort_comfort_score", {})
    rms_vertical = metrics.get("warmup_excluded_comfort_rms_vertical_acc", {})
    return (
        _improves_by(comfort_score, 0.02) and
        _improves_by(rms_vertical, 0.01)
    )


def _stability_risk(metrics: Mapping[str, Mapping[str, Any]]) -> bool:
    names = (
        "warmup_excluded_stability_peak_abs_roll",
        "warmup_excluded_stability_rms_roll",
        "warmup_excluded_stability_peak_abs_lateral_acc",
        "warmup_excluded_stability_rms_lateral_acc",
    )
    return any(_worsens_by(metrics.get(name, {}), 0.05) for name in names)


def _action_overuse(metrics: Mapping[str, Mapping[str, Any]]) -> bool:
    action = metrics.get("mean_abs_action", {}).get("candidate")
    residual = metrics.get("mean_abs_residual_damper", {}).get("candidate")
    action_reward = metrics.get("mean_reward_action", {})
    action_reward_delta = action_reward.get("delta")
    return (
        (action is not None and action > 0.70) or
        (residual is not None and residual > 0.035) or
        (action_reward_delta is not None and action_reward_delta < -0.10)
    )


def _phase4d_status(
    *,
    engineering_pass: int,
    reward_acceptable: bool,
    comfort_promising: bool,
    stability_risk: bool,
) -> str:
    if not engineering_pass:
        return "fail"
    if not reward_acceptable or not comfort_promising:
        return "safe_but_not_better"
    if stability_risk:
        return "comfort_promising_with_tradeoff"
    return "candidate_for_repeat"


def _policy_tags(
    *,
    engineering_pass: int,
    reward_acceptable: bool,
    comfort_promising: bool,
    stability_risk: bool,
    action_overuse: bool,
) -> List[str]:
    tags = ["no_planning", "split_reference"]
    tags.append("engineering_pass" if engineering_pass else "engineering_fail")
    tags.append("reward_acceptable" if reward_acceptable else "reward_not_acceptable")
    tags.append("comfort_promising" if comfort_promising else "no_comfort_gain")
    tags.append("stability_risk" if stability_risk else "stability_ok")
    tags.append("action_overuse" if action_overuse else "action_within_bounds")
    return tags


def _improves_by(detail: Mapping[str, Any], threshold: float) -> bool:
    improvement = detail.get("improvement_pct")
    return improvement is not None and improvement >= threshold


def _worsens_by(detail: Mapping[str, Any], threshold: float) -> bool:
    improvement = detail.get("improvement_pct")
    return improvement is not None and improvement <= -threshold


def _improvement_pct(
    candidate: Optional[float],
    reference: Optional[float],
    direction: str,
) -> Optional[float]:
    if candidate is None or reference is None:
        return None
    denominator = abs(reference)
    if denominator <= 1.0e-12:
        if abs(candidate - reference) <= 1.0e-12:
            return 0.0
        return None
    if direction == "higher":
        return (candidate - reference) / denominator
    if direction == "lower":
        return (reference - candidate) / denominator
    return None


def _pct_change(
    candidate: Optional[float],
    reference: Optional[float],
) -> Optional[float]:
    if candidate is None or reference is None:
        return None
    denominator = abs(reference)
    if denominator <= 1.0e-12:
        if abs(candidate - reference) <= 1.0e-12:
            return 0.0
        return None
    return (candidate - reference) / denominator


def _acceptance_role(
    rows: Sequence[Mapping[str, Any]],
    role: str,
) -> Mapping[str, Any]:
    for row in rows:
        if str(row.get("phase4b_role", "")) == role:
            return row
    return {}


def _select_scenario_row(
    rows: Sequence[Mapping[str, Any]],
    scenario: str,
) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if str(row.get("scenario", row.get("name", "")) or "").strip() == scenario:
            return row
    if len(rows) == 1:
        return rows[0]
    return None


def _reason_set(value: Any) -> List[str]:
    reasons: List[str] = []
    for chunk in str(value or "").replace(",", ";").split(";"):
        reason = chunk.strip()
        if not reason:
            continue
        if "=" in reason:
            reason = reason.split("=", 1)[0].strip()
        if reason:
            reasons.append(reason)
    return reasons


def _reward_row_ratio(row: Mapping[str, Any]) -> Optional[float]:
    ratio = _float_value(row.get("reward_row_ratio"))
    if ratio is not None:
        return ratio
    reward_rows = _float_value(row.get("reward_rows"))
    diagnostic_rows = _float_value(row.get("diagnostic_rows"))
    if reward_rows is not None and diagnostic_rows is not None and diagnostic_rows > 0.0:
        return reward_rows / diagnostic_rows
    return None


def _score_equals(value: Any, expected: float) -> int:
    parsed = _float_value(value)
    return int(parsed is not None and abs(parsed - expected) <= 1.0e-6)


def _float_first(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = _float_value(row.get(key))
        if value is not None:
            return value
    return None


def _float_value(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return parsed
    except (TypeError, ValueError):
        pass
    return None


def _int_value(value: Any) -> int:
    parsed = _float_value(value)
    return int(parsed) if parsed is not None else 0


def _first_value(source: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in source:
            return source.get(key, "")
    return ""


def _read_json_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as json_file:
            data = json.load(json_file)
        return dict(data) if isinstance(data, Mapping) else {}
    except Exception:
        return {}


def _write_single_row_csv(
    path: str,
    row: Mapping[str, Any],
    fields: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _ordered_row(row: Mapping[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    return {field: row.get(field, "") for field in fields}


def _blank_if_none(value: Any) -> Any:
    return "" if value is None else value


def _dedup(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


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
    return os.path.abspath(os.path.expanduser(str(path or "")))


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

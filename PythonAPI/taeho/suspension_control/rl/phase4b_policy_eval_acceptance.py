"""Split-reference Phase 4-B exported policy acceptance reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .phase4_acceptance import (
    PHASE4B_LEARNED_SCENARIO,
    PHASE4B_POLICY_EVAL_FIELDS,
    PHASE4B_REFERENCE_SCENARIO,
    phase4b_policy_eval_acceptance_rows,
)


PHASE4B_SPLIT_COMPARISON_METRICS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("score_route", ("score_route",)),
    ("score_composed", ("score_composed",)),
    ("score_penalty", ("score_penalty",)),
    ("sidecar_command_verifies", ("sidecar_command_verifies",)),
    ("mean_reward_total", ("mean_reward_total",)),
    ("mean_reward_comfort", ("mean_reward_comfort", "reward_comfort_mean")),
    ("mean_reward_stability", ("mean_reward_stability", "reward_stability_mean")),
    ("mean_reward_task", ("mean_reward_task", "reward_task_mean")),
    ("mean_reward_action", ("mean_reward_action", "reward_action_mean")),
    ("mean_reward_safety", ("mean_reward_safety", "reward_safety_mean")),
    ("warmup_excluded_comfort_comfort_score", ("warmup_excluded_comfort_comfort_score",)),
    ("warmup_excluded_comfort_rms_vertical_acc", ("warmup_excluded_comfort_rms_vertical_acc",)),
    ("warmup_excluded_comfort_rms_lateral_acc", ("warmup_excluded_comfort_rms_lateral_acc",)),
    ("warmup_excluded_stability_peak_abs_roll", (
        "warmup_excluded_stability_peak_abs_roll",
        "stability_peak_abs_roll",
    )),
    ("warmup_excluded_stability_rms_roll", ("warmup_excluded_stability_rms_roll",)),
    ("warmup_excluded_stability_peak_abs_lateral_acc", (
        "warmup_excluded_stability_peak_abs_lateral_acc",
    )),
    ("warmup_excluded_stability_rms_lateral_acc", (
        "warmup_excluded_stability_rms_lateral_acc",
    )),
    ("duration_game", ("duration_game",)),
    ("duration_system", ("duration_system",)),
    ("route_progress_stall_ratio", ("route_progress_stall_ratio",)),
    ("route_progress_monotonic_fraction_max", ("route_progress_monotonic_fraction_max",)),
    ("route_completion_proxy", ("route_completion_proxy",)),
    ("low_speed_mask_ratio", ("low_speed_mask_ratio",)),
    ("fallback_ratio", ("fallback_ratio",)),
    ("hard_safety_gate_ratio", ("hard_safety_gate_ratio",)),
    ("observation_clip_ratio", ("observation_clip_ratio",)),
    ("effective_control_ratio", ("effective_control_ratio",)),
    ("mean_abs_action", ("mean_abs_action", "rl_mean_abs_action")),
    ("mean_abs_residual_damper", (
        "mean_abs_residual_damper",
        "rl_mean_abs_residual_damper",
    )),
)


PHASE4B_SPLIT_COMPARISON_FIELDS: Tuple[str, ...] = (
    "candidate_scenario",
    "reference_scenario",
    "phase4b_split_status",
    "failed_checks",
    "warnings",
    "candidate_summary",
    "reference_summary",
) + tuple(
    field
    for metric, _keys in PHASE4B_SPLIT_COMPARISON_METRICS
    for field in (
        "%s_candidate" % metric,
        "%s_reference" % metric,
        "%s_delta" % metric,
    )
)


def write_phase4b_split_reference_acceptance(
    output_dir: str,
    *,
    candidate_summary_path: str = "",
    reference_summary_path: str = "",
    candidate_suite_summary_path: str = "",
    reference_suite_summary_path: str = "",
    candidate_scenario: str = PHASE4B_LEARNED_SCENARIO,
    reference_scenario: str = PHASE4B_REFERENCE_SCENARIO,
) -> Tuple[str, str, str, str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Write split-reference acceptance and candidate-vs-reference comparison."""

    output_dir = _abs_path(output_dir)
    candidate_path = _choose_summary_path(
        candidate_summary_path,
        candidate_suite_summary_path,
        output_dir,
    )
    reference_path = _choose_summary_path(
        reference_summary_path,
        reference_suite_summary_path,
        output_dir,
    )
    candidate_rows = load_phase4b_summary_rows(
        candidate_path,
        scenario_hint=candidate_scenario,
    )
    reference_rows = load_phase4b_summary_rows(
        reference_path,
        scenario_hint=reference_scenario,
    )
    acceptance_rows = phase4b_split_reference_acceptance_rows(
        candidate_rows,
        reference_rows,
        candidate_scenario=candidate_scenario,
        reference_scenario=reference_scenario,
        reference_summary_provided=bool(reference_path),
    )
    candidate = _select_scenario_row(candidate_rows, candidate_scenario)
    reference = _select_scenario_row(reference_rows, reference_scenario)
    comparison_rows = phase4b_split_comparison_rows(
        candidate,
        reference,
        acceptance_rows,
        candidate_summary_path=candidate_path,
        reference_summary_path=reference_path,
        candidate_scenario=candidate_scenario,
        reference_scenario=reference_scenario,
    )

    acceptance_csv = os.path.join(output_dir, "phase4b_policy_eval_acceptance.csv")
    acceptance_json = os.path.join(output_dir, "phase4b_policy_eval_acceptance.json")
    comparison_csv = os.path.join(output_dir, "phase4b_split_comparison.csv")
    comparison_json = os.path.join(output_dir, "phase4b_split_comparison.json")
    _write_csv_rows(acceptance_csv, acceptance_rows, PHASE4B_POLICY_EVAL_FIELDS)
    _write_json(acceptance_json, acceptance_rows)
    _write_csv_rows(comparison_csv, comparison_rows, PHASE4B_SPLIT_COMPARISON_FIELDS)
    _write_json(comparison_json, comparison_rows)
    return (
        acceptance_csv,
        acceptance_json,
        comparison_csv,
        comparison_json,
        acceptance_rows,
        comparison_rows,
    )


def phase4b_split_reference_acceptance_rows(
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_scenario: str = PHASE4B_LEARNED_SCENARIO,
    reference_scenario: str = PHASE4B_REFERENCE_SCENARIO,
    reference_summary_provided: bool = False,
) -> List[Dict[str, Any]]:
    """Build reference and candidate acceptance rows from separate inputs."""

    combined: List[Mapping[str, Any]] = []
    reference = _select_scenario_row(reference_rows, reference_scenario)
    candidate = _select_scenario_row(candidate_rows, candidate_scenario)
    if reference is not None:
        combined.append(reference)
    if candidate is not None:
        combined.append(candidate)

    rows = phase4b_policy_eval_acceptance_rows(
        combined,
        learned_scenario=candidate_scenario,
        reference_scenario=reference_scenario,
    )
    if reference_summary_provided and reference is None:
        for row in rows:
            _append_warning(row, "reference_summary_missing_scenario")
    return rows


def phase4b_split_comparison_rows(
    candidate: Optional[Mapping[str, Any]],
    reference: Optional[Mapping[str, Any]],
    acceptance_rows: Sequence[Mapping[str, Any]],
    *,
    candidate_summary_path: str,
    reference_summary_path: str,
    candidate_scenario: str = PHASE4B_LEARNED_SCENARIO,
    reference_scenario: str = PHASE4B_REFERENCE_SCENARIO,
) -> List[Dict[str, Any]]:
    failed: List[str] = []
    warnings: List[str] = []
    by_role = {
        str(row.get("phase4b_role", "")): row
        for row in acceptance_rows
        if str(row.get("phase4b_role", ""))
    }
    reference_acceptance = by_role.get("reference", {})
    candidate_acceptance = by_role.get("learned", {})

    candidate_status = str(candidate_acceptance.get("phase4b_status", ""))
    reference_status = str(reference_acceptance.get("phase4b_status", ""))
    if candidate is None:
        failed.append("candidate_missing")
    if candidate_status != "pass":
        failed.extend(_split_tokens(candidate_acceptance.get("failed_checks")))
    if reference is None:
        warnings.append("reference_missing")
    elif reference_status != "pass":
        warnings.append("reference_not_pass")
        warnings.extend(
            "reference_%s" % item
            for item in _split_tokens(reference_acceptance.get("failed_checks"))
        )
    warnings.extend(
        "candidate_%s" % item
        for item in _split_tokens(candidate_acceptance.get("warnings"))
    )
    warnings.extend(
        "reference_%s" % item
        for item in _split_tokens(reference_acceptance.get("warnings"))
    )

    status = "fail" if failed else ("warn" if warnings else "pass")
    row: Dict[str, Any] = {
        "candidate_scenario": candidate_scenario,
        "reference_scenario": reference_scenario,
        "phase4b_split_status": status,
        "failed_checks": ";".join(_dedup(failed)),
        "warnings": ";".join(_dedup(warnings)),
        "candidate_summary": candidate_summary_path,
        "reference_summary": reference_summary_path,
    }
    for metric, keys in PHASE4B_SPLIT_COMPARISON_METRICS:
        candidate_value = _float_first(candidate or {}, keys)
        reference_value = _float_first(reference or {}, keys)
        row["%s_candidate" % metric] = (
            candidate_value if candidate_value is not None else "")
        row["%s_reference" % metric] = (
            reference_value if reference_value is not None else "")
        row["%s_delta" % metric] = (
            candidate_value - reference_value
            if candidate_value is not None and reference_value is not None
            else "")
    return [_ordered_row(row, PHASE4B_SPLIT_COMPARISON_FIELDS)]


def load_phase4b_summary_rows(
    path: str,
    *,
    scenario_hint: str = "",
) -> List[Dict[str, Any]]:
    """Load suite CSV/JSON, per-scenario summary JSON, or leaderboard result JSON."""

    resolved = _resolve_path(path)
    if not resolved or not os.path.isfile(resolved):
        return []
    if resolved.endswith(".csv"):
        rows = _read_csv_rows(resolved)
    else:
        rows = _rows_from_json(resolved)
    result: List[Dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        scenario = _infer_scenario(normalized, resolved, scenario_hint)
        if scenario and not _scenario_name(normalized):
            normalized["scenario"] = scenario
        result.append(normalized)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-summary", default="")
    parser.add_argument("--reference-summary", default="")
    parser.add_argument("--candidate-suite-summary", default="")
    parser.add_argument("--reference-suite-summary", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--candidate-scenario", default=PHASE4B_LEARNED_SCENARIO)
    parser.add_argument("--reference-scenario", default=PHASE4B_REFERENCE_SCENARIO)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    (
        acceptance_csv,
        acceptance_json,
        comparison_csv,
        comparison_json,
        acceptance_rows,
        comparison_rows,
    ) = write_phase4b_split_reference_acceptance(
        args.output_dir,
        candidate_summary_path=args.candidate_summary,
        reference_summary_path=args.reference_summary,
        candidate_suite_summary_path=args.candidate_suite_summary,
        reference_suite_summary_path=args.reference_suite_summary,
        candidate_scenario=args.candidate_scenario,
        reference_scenario=args.reference_scenario,
    )
    print("phase4b policy eval acceptance: %s" % acceptance_csv)
    print("phase4b policy eval rows: %d" % len(acceptance_rows))
    print("phase4b policy eval json: %s" % acceptance_json)
    print("phase4b split comparison: %s" % comparison_csv)
    print("phase4b split status: %s" % comparison_rows[0].get("phase4b_split_status", ""))
    print("phase4b split comparison json: %s" % comparison_json)
    return 0


def _rows_from_json(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path) as json_file:
            data = json.load(json_file)
    except Exception:
        return []
    if isinstance(data, list):
        return [dict(item) for item in data if isinstance(item, Mapping)]
    if isinstance(data, Mapping):
        leaderboard = _leaderboard_result_row(data)
        if leaderboard:
            return [leaderboard]
        for key in ("rows", "summary", "suite_summary"):
            value = data.get(key)
            if isinstance(value, list):
                return [dict(item) for item in value if isinstance(item, Mapping)]
        return [dict(data)]
    return []


def _leaderboard_result_row(data: Mapping[str, Any]) -> Dict[str, Any]:
    checkpoint = data.get("_checkpoint")
    if not isinstance(checkpoint, Mapping):
        return {}
    record = checkpoint.get("global_record")
    if not isinstance(record, Mapping):
        records = checkpoint.get("records")
        record = records[0] if isinstance(records, list) and records else {}
    if not isinstance(record, Mapping):
        return {}

    row: Dict[str, Any] = {
        "leaderboard_status": record.get("status", ""),
        "num_infractions": record.get("num_infractions", ""),
    }
    scores = record.get("scores_mean", record.get("scores", {}))
    if isinstance(scores, Mapping):
        row.update({
            "score_route": scores.get("score_route", ""),
            "score_composed": scores.get("score_composed", ""),
            "score_penalty": scores.get("score_penalty", ""),
        })
    infractions = record.get("infractions", {})
    if isinstance(infractions, Mapping):
        row["infractions"] = dict(infractions)
        for key, value in infractions.items():
            row[str(key)] = value
    meta = record.get("meta", {})
    if isinstance(meta, Mapping):
        row.update({
            "duration_game": meta.get("duration_game", ""),
            "duration_system": meta.get("duration_system", ""),
            "route_length": meta.get("total_length", ""),
        })
    return row


def _choose_summary_path(*paths: str) -> str:
    for path in paths:
        resolved = _resolve_path(path)
        if resolved and os.path.isfile(resolved):
            return resolved
    output_dir = _resolve_path(paths[-1] if paths else "")
    for name in ("suite_summary.csv", "suite_summary.json", "summary.json"):
        candidate = os.path.join(output_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return ""


def _select_scenario_row(
    rows: Sequence[Mapping[str, Any]],
    scenario: str,
) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if _scenario_name(row) == scenario:
            return row
    if len(rows) == 1:
        return rows[0]
    return None


def _scenario_name(row: Mapping[str, Any]) -> str:
    return str(row.get("scenario", row.get("name", "")) or "").strip()


def _infer_scenario(row: Mapping[str, Any], path: str, scenario_hint: str) -> str:
    scenario = _scenario_name(row)
    if scenario:
        return scenario
    controller = str(row.get("controller", "") or "")
    if controller == "rl_residual_skyhook":
        return PHASE4B_LEARNED_SCENARIO
    if controller == "rl_zero_residual_skyhook":
        return PHASE4B_REFERENCE_SCENARIO
    for part in reversed(_resolve_path(path).split(os.sep)):
        match = re.search(r"(S\d+_[A-Za-z0-9_]+)$", part)
        if match:
            return match.group(1)
    return scenario_hint


def _append_warning(row: Dict[str, Any], warning: str) -> None:
    warnings = _split_tokens(row.get("warnings"))
    if warning not in warnings:
        warnings.append(warning)
    row["warnings"] = ";".join(warnings)


def _split_tokens(value: Any) -> List[str]:
    return [
        item.strip()
        for item in str(value or "").replace(",", ";").split(";")
        if item.strip()
    ]


def _dedup(values: Sequence[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _float_first(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = _float_value(row.get(key))
        if value is not None:
            return value
    return None


def _float_value(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
        if parsed == parsed and parsed not in (float("inf"), float("-inf")):
            return parsed
    except (TypeError, ValueError):
        pass
    return None


def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, newline="") as csv_file:
            return [dict(row) for row in csv.DictReader(csv_file)]
    except Exception:
        return []


def _write_csv_rows(
    path: str,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _ordered_row(row: Mapping[str, Any], fields: Sequence[str]) -> Dict[str, Any]:
    return {field: row.get(field, "") for field in fields}


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

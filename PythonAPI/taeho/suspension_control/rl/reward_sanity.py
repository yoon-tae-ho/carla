"""Phase 3 reward/task/progress sanity gate for suspension route suites."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REFERENCE_SCENARIO = "S8_rl_zero_residual_skyhook"
CANARY_SCENARIOS = (
    REFERENCE_SCENARIO,
    "S12_rl_const_action_plus_0p25_skyhook",
    "S13_rl_const_action_minus_0p25_skyhook",
    "S14_rl_random_action_0p10_skyhook",
)
CONSTANT_ACTION_SCENARIOS = (
    "S12_rl_const_action_plus_0p25_skyhook",
    "S13_rl_const_action_minus_0p25_skyhook",
)
RANDOM_ACTION_SCENARIO = "S14_rl_random_action_0p10_skyhook"

PHASE3_INFRACTION_FIELDS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "collision_count",
    "red_light",
    "red_light_count",
    "stop_infraction",
    "outside_route_lanes",
    "route_dev",
    "lane_invasion_count",
    "vehicle_blocked",
    "blocked_vehicle_count",
    "route_timeout",
    "route_timeout_count",
    "scenario_timeouts",
    "min_speed_infractions",
)

PHASE3_REWARD_SANITY_FIELDS = (
    "seed",
    "phase3_status",
    "failed_checks",
    "warnings",
    "scenario",
    "reference_scenario",
    "route_score_ok",
    "score_composed_ok",
    "infraction_ok",
    "verification_ok",
    "policy_available_ok",
    "reward_present_ok",
    "route_progress_present_ok",
    "route_completion_ok",
    "zero_action_penalty_ok",
    "constant_action_penalty_ok",
    "constant_baseline_dev_penalty_ok",
    "random_action_rate_penalty_ok",
    "slow_shortcut_detection_present_ok",
    "mean_reward_total",
    "mean_reward_action",
    "mean_reward_task",
    "mean_reward_cost_action",
    "mean_reward_cost_task",
    "mean_reward_term_action_mag",
    "mean_reward_term_action_rate",
    "mean_reward_term_baseline_dev",
    "mean_reward_term_low_speed_not_planned",
    "mean_reward_term_progress_stall",
    "route_progress_monotonic_fraction_max",
    "low_speed_not_planned_ratio",
    "progress_stall_ratio",
    "duration_game",
)


def phase3_reward_sanity_rows(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    reference_scenario: str = REFERENCE_SCENARIO,
    canary_scenarios: Sequence[str] = CANARY_SCENARIOS,
    reward_ratio_threshold: float = 0.95,
    route_progress_ratio_threshold: float = 0.95,
    policy_available_threshold: float = 0.99,
    zero_action_cost_threshold: float = 0.05,
    penalty_margin: float = 1.0e-6,
) -> List[Dict[str, Any]]:
    """Return per-scenario Phase 3 reward sanity rows."""

    rows = list(summary_rows or [])
    seeds = _ordered_unique(row.get("seed", "") for row in rows)
    if not seeds:
        seeds = [""]
    by_seed_scenario = {
        (row.get("seed", ""), str(row.get("scenario", ""))): row
        for row in rows
    }

    report_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        reference = by_seed_scenario.get((seed, reference_scenario))
        constants = [
            by_seed_scenario.get((seed, scenario))
            for scenario in CONSTANT_ACTION_SCENARIOS
        ]
        random_row = by_seed_scenario.get((seed, RANDOM_ACTION_SCENARIO))
        for scenario in canary_scenarios:
            row = by_seed_scenario.get((seed, scenario))
            report_rows.append(_scenario_report_row(
                seed=seed,
                scenario=scenario,
                row=row,
                reference=reference,
                constants=constants,
                random_row=random_row,
                reference_scenario=reference_scenario,
                reward_ratio_threshold=reward_ratio_threshold,
                route_progress_ratio_threshold=route_progress_ratio_threshold,
                policy_available_threshold=policy_available_threshold,
                zero_action_cost_threshold=zero_action_cost_threshold,
                penalty_margin=penalty_margin))
    return report_rows


def write_phase3_reward_sanity_report(
    output_dir: str,
    summary_rows: Sequence[Mapping[str, Any]],
) -> Tuple[str, str, List[Dict[str, Any]]]:
    rows = phase3_reward_sanity_rows(summary_rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "phase3_reward_sanity.csv")
    json_path = os.path.join(output_dir, "phase3_reward_sanity.json")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=PHASE3_REWARD_SANITY_FIELDS,
            extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: _format_value(row.get(field, ""))
                for field in PHASE3_REWARD_SANITY_FIELDS
            })
    with open(json_path, "w") as json_file:
        json.dump(rows, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    return csv_path, json_path, rows


def phase3_overall_status(rows: Sequence[Mapping[str, Any]]) -> str:
    return "pass" if all(row.get("phase3_status") == "pass" for row in rows) else "fail"


def _scenario_report_row(
    *,
    seed: Any,
    scenario: str,
    row: Optional[Mapping[str, Any]],
    reference: Optional[Mapping[str, Any]],
    constants: Sequence[Optional[Mapping[str, Any]]],
    random_row: Optional[Mapping[str, Any]],
    reference_scenario: str,
    reward_ratio_threshold: float,
    route_progress_ratio_threshold: float,
    policy_available_threshold: float,
    zero_action_cost_threshold: float,
    penalty_margin: float,
) -> Dict[str, Any]:
    failed: List[str] = []
    warnings: List[str] = []
    if row is None:
        return _base_report_row(
            seed,
            scenario,
            reference_scenario,
            failed_checks=("missing_scenario",))

    route_score_ok = _score_is_100(row.get("score_route"))
    score_composed_ok = _score_is_100(row.get("score_composed"))
    infraction_ok = _infractions_are_zero(row)
    verification_ok = _positive(row.get("sidecar_command_verifies"))
    policy_available_ok = _at_least(row.get("rl_policy_available_ratio"), policy_available_threshold)
    reward_present_ok = _at_least(row.get("reward_row_ratio"), reward_ratio_threshold)
    route_progress_present_ok = _at_least(
        row.get("route_progress_available_ratio"),
        route_progress_ratio_threshold)
    route_completion_ok = (
        _at_least(row.get("route_progress_monotonic_fraction_max"), 0.95) or
        _score_is_100(row.get("score_route")))
    slow_shortcut_detection_present_ok = (
        _is_number(row.get("low_speed_not_planned_ratio")) or
        _is_number(row.get("progress_stall_ratio")))

    for name, ok in (
        ("route_score", route_score_ok),
        ("score_composed", score_composed_ok),
        ("infractions", infraction_ok),
        ("verification", verification_ok),
        ("policy_available_ratio", policy_available_ok),
        ("reward_row_ratio", reward_present_ok),
        ("route_progress_available_ratio", route_progress_present_ok),
        ("route_completion", route_completion_ok),
        ("slow_shortcut_detection", slow_shortcut_detection_present_ok),
    ):
        if not ok:
            failed.append(name)

    reference_missing = reference is None
    if reference_missing:
        failed.append("missing_reference")

    zero_action_penalty_ok: Any = ""
    constant_action_penalty_ok: Any = ""
    constant_baseline_dev_penalty_ok: Any = ""
    random_action_rate_penalty_ok: Any = ""

    if scenario == reference_scenario:
        zero_cost = _number(row.get("mean_reward_cost_action"))
        zero_action_penalty_ok = int(
            zero_cost is not None and zero_cost <= zero_action_cost_threshold)
        if not zero_action_penalty_ok:
            failed.append("zero_action_penalty")
    elif not reference_missing:
        ref_action_cost = _number(reference.get("mean_reward_cost_action"))
        ref_baseline_dev = _number(reference.get("mean_reward_term_baseline_dev"))
        ref_action_rate = _number(reference.get("mean_reward_term_action_rate"))
        if scenario in CONSTANT_ACTION_SCENARIOS:
            action_cost = _number(row.get("mean_reward_cost_action"))
            baseline_dev = _number(row.get("mean_reward_term_baseline_dev"))
            constant_action_penalty_ok = int(
                action_cost is not None and
                ref_action_cost is not None and
                action_cost > ref_action_cost + penalty_margin)
            constant_baseline_dev_penalty_ok = int(
                baseline_dev is not None and
                ref_baseline_dev is not None and
                baseline_dev > ref_baseline_dev + penalty_margin)
            if not constant_action_penalty_ok:
                failed.append("constant_action_penalty")
            if not constant_baseline_dev_penalty_ok:
                failed.append("constant_baseline_dev_penalty")
        elif scenario == RANDOM_ACTION_SCENARIO:
            random_rate = _number(row.get("mean_reward_term_action_rate"))
            constant_rates = [
                _number(item.get("mean_reward_term_action_rate"))
                for item in constants
                if item is not None
            ]
            random_action_rate_penalty_ok = int(
                random_rate is not None and
                ref_action_rate is not None and
                len(constant_rates) == len(CONSTANT_ACTION_SCENARIOS) and
                all(value is not None for value in constant_rates) and
                random_rate > ref_action_rate + penalty_margin and
                random_rate > max(constant_rates) + penalty_margin)
            if not random_action_rate_penalty_ok:
                failed.append("random_action_rate_penalty")

    if reference is not None and scenario != reference_scenario:
        if (
                scenario == RANDOM_ACTION_SCENARIO and
                _greater(row.get("mean_reward_total"), reference.get("mean_reward_total"))):
            warnings.append("random_total_reward_above_reference")
        ref_duration = _number(reference.get("duration_game"))
        duration = _number(row.get("duration_game"))
        if ref_duration is not None and duration is not None and duration > ref_duration * 1.10:
            ref_task_cost = _number(reference.get("mean_reward_cost_task"))
            task_cost = _number(row.get("mean_reward_cost_task"))
            if task_cost is None or ref_task_cost is None or task_cost <= ref_task_cost:
                warnings.append("slow_run_not_task_penalized")

    result = _base_report_row(seed, scenario, reference_scenario)
    result.update({
        "phase3_status": "pass" if not failed else "fail",
        "failed_checks": ";".join(failed),
        "warnings": ";".join(warnings),
        "route_score_ok": int(route_score_ok),
        "score_composed_ok": int(score_composed_ok),
        "infraction_ok": int(infraction_ok),
        "verification_ok": int(verification_ok),
        "policy_available_ok": int(policy_available_ok),
        "reward_present_ok": int(reward_present_ok),
        "route_progress_present_ok": int(route_progress_present_ok),
        "route_completion_ok": int(route_completion_ok),
        "zero_action_penalty_ok": zero_action_penalty_ok,
        "constant_action_penalty_ok": constant_action_penalty_ok,
        "constant_baseline_dev_penalty_ok": constant_baseline_dev_penalty_ok,
        "random_action_rate_penalty_ok": random_action_rate_penalty_ok,
        "slow_shortcut_detection_present_ok": int(slow_shortcut_detection_present_ok),
    })
    for field in PHASE3_REWARD_SANITY_FIELDS:
        if field in result:
            continue
        result[field] = row.get(field, "")
    return result


def _base_report_row(
    seed: Any,
    scenario: str,
    reference_scenario: str,
    failed_checks: Sequence[str] = (),
) -> Dict[str, Any]:
    return {
        "seed": seed,
        "phase3_status": "fail" if failed_checks else "",
        "failed_checks": ";".join(failed_checks),
        "warnings": "",
        "scenario": scenario,
        "reference_scenario": reference_scenario,
    }


def _ordered_unique(values: Iterable[Any]) -> List[Any]:
    seen = set()
    result = []
    for value in values:
        key = str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def _infractions_are_zero(row: Mapping[str, Any]) -> bool:
    for field in PHASE3_INFRACTION_FIELDS:
        value = row.get(field, "")
        if value in ("", None, "[]", "{}", "0", "0.0"):
            continue
        number = _number(value)
        if number is not None and abs(number) <= 1.0e-12:
            continue
        return False
    return True


def _score_is_100(value: Any) -> bool:
    number = _number(value)
    return number is not None and abs(number - 100.0) <= 1.0e-9


def _at_least(value: Any, threshold: float) -> bool:
    number = _number(value)
    return number is not None and number >= threshold


def _positive(value: Any) -> bool:
    number = _number(value)
    return number is not None and number > 0.0


def _greater(left: Any, right: Any) -> bool:
    left_number = _number(left)
    right_number = _number(right)
    return left_number is not None and right_number is not None and left_number > right_number


def _is_number(value: Any) -> bool:
    return _number(value) is not None


def _number(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        return "%0.9g" % value
    return value

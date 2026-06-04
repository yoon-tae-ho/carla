"""Phase 3-B reward/progress calibration gate for suspension route suites."""

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

INFRACTION_FIELDS = (
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
)

PHASE3B_REWARD_CALIBRATION_FIELDS = (
    "seed",
    "phase3b_status",
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
    "monotonic_progress_ok",
    "task_reward_scale_ok",
    "route_deviation_not_used_when_disabled_ok",
    "zero_action_penalty_ok",
    "constant_action_penalty_ok",
    "constant_baseline_dev_penalty_ok",
    "random_action_rate_penalty_ok",
    "random_total_reward_not_above_reference_ok",
    "reward_random_total_delta_vs_reference",
    "reward_random_total_tolerance",
    "mean_reward_total",
    "mean_reward_task",
    "mean_reward_action",
    "mean_reward_cost_task",
    "mean_reward_term_action_mag",
    "mean_reward_term_action_rate",
    "mean_reward_term_baseline_dev",
    "mean_reward_term_route_deviation",
    "mean_reward_route_deviation_used_ratio",
    "route_progress_available_ratio",
    "route_progress_monotonic_fraction_max",
    "route_progress_monotonic_negative_rows",
    "route_progress_raw_negative_ratio",
    "route_progress_stall_ratio",
    "low_speed_not_planned_ratio",
    "reward_row_ratio",
    "rl_policy_available_ratio",
    "sidecar_command_verifies",
    "score_route",
    "score_composed",
    "collision_count",
    "lane_invasion_count",
    "red_light_count",
    "blocked_vehicle_count",
    "route_timeout_count",
)


def phase3b_reward_calibration_rows(
    summary_rows: Sequence[Mapping[str, Any]],
    *,
    reference_scenario: str = REFERENCE_SCENARIO,
    canary_scenarios: Sequence[str] = CANARY_SCENARIOS,
    reward_ratio_threshold: float = 0.95,
    route_progress_ratio_threshold: float = 0.95,
    policy_available_threshold: float = 0.99,
    route_completion_threshold: float = 0.98,
    task_abs_max_without_infraction: float = 2.0,
    zero_action_threshold: float = 1.0e-9,
    constant_action_margin: float = 1.0e-4,
    constant_baseline_dev_margin: float = 1.0e-5,
    random_action_rate_margin: float = 1.0e-4,
    reward_random_total_tolerance: float = 0.02,
) -> List[Dict[str, Any]]:
    """Return per-scenario Phase 3-B calibration report rows."""

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
        for scenario in canary_scenarios:
            row = by_seed_scenario.get((seed, scenario))
            report_rows.append(_scenario_report_row(
                seed=seed,
                scenario=scenario,
                row=row,
                reference=reference,
                constants=constants,
                reference_scenario=reference_scenario,
                reward_ratio_threshold=reward_ratio_threshold,
                route_progress_ratio_threshold=route_progress_ratio_threshold,
                policy_available_threshold=policy_available_threshold,
                route_completion_threshold=route_completion_threshold,
                task_abs_max_without_infraction=task_abs_max_without_infraction,
                zero_action_threshold=zero_action_threshold,
                constant_action_margin=constant_action_margin,
                constant_baseline_dev_margin=constant_baseline_dev_margin,
                random_action_rate_margin=random_action_rate_margin,
                reward_random_total_tolerance=reward_random_total_tolerance))
    return report_rows


def write_phase3b_reward_calibration_report(
    output_dir: str,
    summary_rows: Sequence[Mapping[str, Any]],
) -> Tuple[str, str, List[Dict[str, Any]]]:
    rows = phase3b_reward_calibration_rows(summary_rows)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "phase3b_reward_calibration.csv")
    json_path = os.path.join(output_dir, "phase3b_reward_calibration.json")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=PHASE3B_REWARD_CALIBRATION_FIELDS,
            extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: _format_value(row.get(field, ""))
                for field in PHASE3B_REWARD_CALIBRATION_FIELDS
            })
    with open(json_path, "w") as json_file:
        json.dump(rows, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    return csv_path, json_path, rows


def phase3b_overall_status(rows: Sequence[Mapping[str, Any]]) -> str:
    return "pass" if all(row.get("phase3b_status") == "pass" for row in rows) else "fail"


def _scenario_report_row(
    *,
    seed: Any,
    scenario: str,
    row: Optional[Mapping[str, Any]],
    reference: Optional[Mapping[str, Any]],
    constants: Sequence[Optional[Mapping[str, Any]]],
    reference_scenario: str,
    reward_ratio_threshold: float,
    route_progress_ratio_threshold: float,
    policy_available_threshold: float,
    route_completion_threshold: float,
    task_abs_max_without_infraction: float,
    zero_action_threshold: float,
    constant_action_margin: float,
    constant_baseline_dev_margin: float,
    random_action_rate_margin: float,
    reward_random_total_tolerance: float,
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
    policy_available_ok = _at_least(
        row.get("rl_policy_available_ratio", row.get("policy_available_ratio")),
        policy_available_threshold)
    reward_present_ok = _at_least(
        row.get("reward_row_ratio"),
        reward_ratio_threshold)
    route_progress_present_ok = _at_least(
        row.get("route_progress_available_ratio"),
        route_progress_ratio_threshold)
    route_completion_ok = _at_least(
        row.get("route_progress_monotonic_fraction_max"),
        route_completion_threshold)
    monotonic_progress_ok = _zero(row.get("route_progress_monotonic_negative_rows"))
    route_deviation_not_used_when_disabled_ok = _zero(
        row.get("mean_reward_route_deviation_used_ratio"))
    task_reward_scale_ok = _task_reward_scale_ok(
        row,
        infraction_ok=infraction_ok,
        threshold=task_abs_max_without_infraction)

    for name, ok in (
        ("route_score", route_score_ok),
        ("score_composed", score_composed_ok),
        ("infractions", infraction_ok),
        ("verification", verification_ok),
        ("policy_available_ratio", policy_available_ok),
        ("reward_row_ratio", reward_present_ok),
        ("route_progress_available_ratio", route_progress_present_ok),
        ("route_completion", route_completion_ok),
        ("monotonic_progress", monotonic_progress_ok),
        ("task_reward_scale", task_reward_scale_ok),
        (
            "route_deviation_not_used_when_disabled",
            route_deviation_not_used_when_disabled_ok),
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
    random_total_reward_not_above_reference_ok: Any = ""
    random_total_delta: Any = ""

    if scenario == reference_scenario:
        action_mag = _number(row.get("mean_reward_term_action_mag"))
        action_rate = _number(row.get("mean_reward_term_action_rate"))
        zero_action_penalty_ok = int(
            action_mag is not None and
            action_rate is not None and
            action_mag <= zero_action_threshold and
            action_rate <= zero_action_threshold)
        if not zero_action_penalty_ok:
            failed.append("zero_action_penalty")
    elif not reference_missing:
        ref_action_mag = _number(reference.get("mean_reward_term_action_mag"))
        ref_baseline_dev = _number(reference.get("mean_reward_term_baseline_dev"))
        ref_action_rate = _number(reference.get("mean_reward_term_action_rate"))
        if scenario in CONSTANT_ACTION_SCENARIOS:
            action_mag = _number(row.get("mean_reward_term_action_mag"))
            baseline_dev = _number(row.get("mean_reward_term_baseline_dev"))
            constant_action_penalty_ok = int(
                action_mag is not None and
                ref_action_mag is not None and
                action_mag > ref_action_mag + constant_action_margin)
            constant_baseline_dev_penalty_ok = int(
                baseline_dev is not None and
                ref_baseline_dev is not None and
                baseline_dev > ref_baseline_dev + constant_baseline_dev_margin)
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
                random_rate > max([ref_action_rate] + constant_rates) +
                random_action_rate_margin)
            if not random_action_rate_penalty_ok:
                failed.append("random_action_rate_penalty")

            ref_total = _number(reference.get("mean_reward_total"))
            random_total = _number(row.get("mean_reward_total"))
            if ref_total is not None and random_total is not None:
                random_total_delta = random_total - ref_total
                random_total_reward_not_above_reference_ok = int(
                    random_total_delta <= reward_random_total_tolerance)
                if not random_total_reward_not_above_reference_ok:
                    warnings.append("random_total_reward_above_reference")
            else:
                random_total_reward_not_above_reference_ok = 0
                warnings.append("random_total_reward_missing")

    result = _base_report_row(seed, scenario, reference_scenario)
    result.update({
        "phase3b_status": "pass" if not failed else "fail",
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
        "monotonic_progress_ok": int(monotonic_progress_ok),
        "task_reward_scale_ok": int(task_reward_scale_ok),
        "route_deviation_not_used_when_disabled_ok": int(
            route_deviation_not_used_when_disabled_ok),
        "zero_action_penalty_ok": zero_action_penalty_ok,
        "constant_action_penalty_ok": constant_action_penalty_ok,
        "constant_baseline_dev_penalty_ok": constant_baseline_dev_penalty_ok,
        "random_action_rate_penalty_ok": random_action_rate_penalty_ok,
        "random_total_reward_not_above_reference_ok": (
            random_total_reward_not_above_reference_ok),
        "reward_random_total_delta_vs_reference": random_total_delta,
        "reward_random_total_tolerance": reward_random_total_tolerance,
    })
    for field in PHASE3B_REWARD_CALIBRATION_FIELDS:
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
        "phase3b_status": "fail" if failed_checks else "",
        "failed_checks": ";".join(failed_checks),
        "warnings": "",
        "scenario": scenario,
        "reference_scenario": reference_scenario,
    }


def _task_reward_scale_ok(
    row: Mapping[str, Any],
    *,
    infraction_ok: bool,
    threshold: float,
) -> bool:
    if not infraction_ok:
        return True
    task_reward = _number(row.get("mean_reward_task"))
    return task_reward is not None and abs(task_reward) <= threshold


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
    for field in INFRACTION_FIELDS:
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


def _zero(value: Any) -> bool:
    number = _number(value)
    return number is not None and abs(number) <= 1.0e-12


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

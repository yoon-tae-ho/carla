"""Phase 4-C route-health analyzer for live-training outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SUMMARY_FIELDS: Tuple[str, ...] = (
    "phase4_route_health_status",
    "failed_checks",
    "warnings",
    "backend",
    "real_backend_used",
    "fake_backend_used",
    "carla_connected",
    "route_process_started",
    "hero_attached",
    "total_timesteps_requested",
    "total_timesteps_collected",
    "reward_rows",
    "reward_row_ratio",
    "route_progress_available_ratio",
    "route_progress_fraction_max",
    "route_progress_monotonic_fraction_max",
    "route_progress_monotonic_negative_rows",
    "route_progress_delta_mean",
    "route_progress_rate_mps_mean",
    "route_progress_rate_mps_p10",
    "route_progress_rate_mps_p50",
    "route_progress_rate_mps_p90",
    "route_progress_stall_ratio",
    "progress_per_1000_steps",
    "expected_progress_fraction",
    "progress_vs_reference_ratio",
    "low_speed_mask_ratio",
    "fallback_ratio",
    "hard_safety_gate_ratio",
    "soft_safety_gain_ratio",
    "effective_control_ratio",
    "mean_abs_action",
    "mean_abs_residual_damper",
    "mean_reward_total",
    "mean_reward_comfort",
    "mean_reward_stability",
    "mean_reward_task",
    "mean_reward_action",
    "mean_reward_safety",
    "observation_clip_ratio",
    "safety_gate_reason_counts",
    "fallback_reason_counts",
    "collision_count",
    "lane_invasion_count",
    "red_light_count",
    "blocked_vehicle_count",
    "route_timeout_count",
    "real_backend_ok",
    "hero_ok",
    "reward_ok",
    "finite_reward_ok",
    "progress_available_ok",
    "monotonic_progress_ok",
    "hard_safety_ok",
    "low_speed_mask_ok",
    "fallback_ok",
    "effective_control_ok",
    "residual_ok",
    "action_ok",
    "observation_clip_ok",
    "infraction_proxy_ok",
)


BUCKET_FIELDS: Tuple[str, ...] = (
    "bucket",
    "rows",
    "mean_speed",
    "mean_abs_action",
    "mean_abs_residual_damper",
    "fallback_ratio",
    "low_speed_mask_ratio",
    "hard_safety_gate_ratio",
    "mean_reward_total",
    "mean_reward_task",
    "mean_reward_action",
    "progress_stall_ratio",
    "most_common_safety_reason",
)


PROGRESS_BUCKETS: Tuple[Tuple[float, float, str], ...] = (
    (0.00, 0.05, "0.00-0.05"),
    (0.05, 0.10, "0.05-0.10"),
    (0.10, 0.20, "0.10-0.20"),
    (0.20, 0.30, "0.20-0.30"),
    (0.30, 0.40, "0.30-0.40"),
    (0.40, 0.60, "0.40-0.60"),
    (0.60, 0.80, "0.60-0.80"),
    (0.80, 1.00, "0.80-1.00"),
)

REWARD_COLUMNS: Tuple[str, ...] = (
    "reward_total",
    "reward_comfort",
    "reward_stability",
    "reward_task",
    "reward_action",
    "reward_safety",
)


def write_phase4_route_health(
    output_dir: str,
    *,
    rollout_csv_path: str = "",
    rollout_jsonl_path: str = "",
    training_summary_path: str = "",
    training_config_path: str = "",
    phase4_canary_path: str = "",
    reference_diagnostic_rows: int = 0,
) -> Tuple[str, str, str, Dict[str, Any], List[Dict[str, Any]]]:
    """Write Phase 4 route-health summary and progress-bucket diagnostics."""

    output_dir = _abs_path(output_dir)
    rollout_csv_path = _resolve_path(
        rollout_csv_path or os.path.join(output_dir, "training_rollout.csv"),
        output_dir)
    rollout_jsonl_path = _resolve_path(
        rollout_jsonl_path or os.path.join(output_dir, "training_rollout.jsonl"),
        output_dir)
    summary_path = _resolve_path(
        training_summary_path or os.path.join(output_dir, "training_summary.json"),
        output_dir)
    config_path = _resolve_path(
        training_config_path or os.path.join(output_dir, "training_config.json"),
        output_dir)
    canary_path = _resolve_path(
        phase4_canary_path or os.path.join(output_dir, "phase4_training_canary.json"),
        output_dir)

    rows = _read_csv_rows(rollout_csv_path)
    if not rows:
        rows = _read_jsonl_rows(rollout_jsonl_path)
    summary = _read_json_dict(summary_path)
    config = _read_json_dict(config_path)
    canary = _read_json_dict(canary_path)

    health = phase4_route_health_row(
        rows,
        training_summary=summary,
        training_config=config,
        phase4_canary=canary,
        reference_diagnostic_rows=reference_diagnostic_rows)
    bucket_rows = phase4_route_health_bucket_rows(rows)

    csv_path = os.path.join(output_dir, "phase4_route_health.csv")
    json_path = os.path.join(output_dir, "phase4_route_health.json")
    buckets_path = os.path.join(output_dir, "phase4_route_health_by_progress_bucket.csv")
    _write_single_row_csv(csv_path, health, SUMMARY_FIELDS)
    _write_json(json_path, health)
    _write_csv_rows(buckets_path, bucket_rows, BUCKET_FIELDS)
    return csv_path, json_path, buckets_path, health, bucket_rows


def phase4_route_health_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    training_summary: Mapping[str, Any] = (),
    training_config: Mapping[str, Any] = (),
    phase4_canary: Mapping[str, Any] = (),
    reference_diagnostic_rows: int = 0,
) -> Dict[str, Any]:
    """Compute one route-health summary row."""

    rows = list(rows or [])
    training_summary = dict(training_summary or {})
    training_config = dict(training_config or {})
    phase4_canary = dict(phase4_canary or {})
    collected = len(rows) or _int_first(
        training_summary,
        phase4_canary,
        keys=("rollout_rows", "total_timesteps_collected", "diagnostic_rows"))
    requested = _int_first(
        training_summary,
        training_config,
        phase4_canary,
        keys=("total_timesteps", "total_timesteps_requested"))

    reward_rows = _finite_count(rows, "reward_total")
    reward_row_ratio = (
        reward_rows / float(collected)
        if collected > 0 else _float_first(
            training_summary,
            phase4_canary,
            keys=("reward_row_ratio",)))
    route_available_ratio, route_available_missing = _availability_ratio(
        rows,
        "route_progress_available")
    monotonic_values = _finite_values(rows, "route_progress_monotonic_fraction")
    progress_values = _finite_values(rows, "route_progress_fraction")
    progress_delta = _finite_values(rows, "route_progress_delta_m")
    progress_rate = _finite_values(rows, "route_progress_rate_mps")
    monotonic_max = _max_or_empty(monotonic_values)
    progress_max = _max_or_empty(progress_values)
    monotonic_negative_rows = _monotonic_negative_rows(monotonic_values)

    fallback_ratio = _ratio(rows, lambda row: bool(_nonempty(row.get("rl_fallback_reason"))))
    low_speed_ratio = _ratio(rows, _low_speed_mask_active)
    hard_safety_ratio = _ratio(rows, _hard_safety_active)
    soft_safety_ratio = _ratio(rows, _soft_safety_active)
    effective_ratio = _ratio(rows, lambda row: (_row_abs_residual(row) or 0.0) > 1.0e-12)
    observation_clip_ratio = _ratio(
        rows,
        lambda row: (_float_first_row(row, ("rl_observation_clip_count",
                                            "observation_clip_count")) or 0.0) > 0.0)
    stall_ratio = _ratio(
        rows,
        lambda row: _positive(row.get("route_progress_stall")) or
        _positive(row.get("progress_stall")))

    safety_counts = _reason_counts(row.get("rl_safety_gate_reason") for row in rows)
    fallback_counts = _reason_counts(row.get("rl_fallback_reason") for row in rows)
    collision_count = _max_first(rows, ("collision_count",))
    lane_count = _max_first(rows, ("lane_invasion_count", "lane_invasion"))
    red_count = _max_first(rows, ("red_light_count", "red_light"))
    blocked_count = _max_first(rows, (
        "blocked_vehicle_count",
        "blocked_vehicle",
        "vehicle_blocked"))
    timeout_count = _max_first(rows, ("route_timeout_count", "route_timeout"))

    expected_progress = ""
    progress_vs_reference = ""
    warnings: List[str] = []
    if int(reference_diagnostic_rows or 0) > 0 and collected > 0:
        expected_progress = min(1.0, collected / float(reference_diagnostic_rows))
        progress_value = _float_value(monotonic_max)
        if progress_value is not None and expected_progress > 0.0:
            progress_vs_reference = progress_value / expected_progress
            if progress_value < 0.60 * expected_progress:
                warnings.append("progress_below_reference_expectation")
    if route_available_missing:
        warnings.append("route_progress_available_missing")
    if not rows:
        warnings.append("rollout_rows_missing")
    if _missing_any_column(rows, ("rl_safety_gain",)):
        warnings.append("soft_safety_gain_may_be_underestimated")

    row: Dict[str, Any] = {
        "backend": _first_nonempty(
            training_summary,
            training_config,
            phase4_canary,
            rows,
            key="backend"),
        "real_backend_used": _max_first(
            rows,
            ("real_backend_used",),
            fallback=_first_value(training_summary, phase4_canary, key="real_backend_used")),
        "fake_backend_used": _max_first(
            rows,
            ("fake_backend_used",),
            fallback=_first_value(training_summary, phase4_canary, key="fake_backend_used")),
        "carla_connected": _max_first(
            rows,
            ("carla_connected",),
            fallback=_first_value(phase4_canary, training_summary, key="carla_connected")),
        "route_process_started": _max_first(
            rows,
            ("route_process_started",),
            fallback=_first_value(phase4_canary, training_summary, key="route_process_started")),
        "hero_attached": _max_first(
            rows,
            ("hero_attached",),
            fallback=_first_value(phase4_canary, training_summary, key="hero_attached")),
        "total_timesteps_requested": requested,
        "total_timesteps_collected": collected,
        "reward_rows": reward_rows,
        "reward_row_ratio": reward_row_ratio,
        "route_progress_available_ratio": route_available_ratio,
        "route_progress_fraction_max": progress_max,
        "route_progress_monotonic_fraction_max": monotonic_max,
        "route_progress_monotonic_negative_rows": monotonic_negative_rows,
        "route_progress_delta_mean": _mean_or_empty(progress_delta),
        "route_progress_rate_mps_mean": _mean_or_empty(progress_rate),
        "route_progress_rate_mps_p10": _percentile_or_empty(progress_rate, 0.10),
        "route_progress_rate_mps_p50": _percentile_or_empty(progress_rate, 0.50),
        "route_progress_rate_mps_p90": _percentile_or_empty(progress_rate, 0.90),
        "route_progress_stall_ratio": stall_ratio,
        "progress_per_1000_steps": (
            monotonic_max / (collected / 1000.0)
            if isinstance(monotonic_max, float) and collected > 0 else ""),
        "expected_progress_fraction": expected_progress,
        "progress_vs_reference_ratio": progress_vs_reference,
        "low_speed_mask_ratio": low_speed_ratio,
        "fallback_ratio": fallback_ratio,
        "hard_safety_gate_ratio": hard_safety_ratio,
        "soft_safety_gain_ratio": soft_safety_ratio,
        "effective_control_ratio": effective_ratio,
        "mean_abs_action": _mean_or_empty(_row_abs_action(row) for row in rows),
        "mean_abs_residual_damper": _mean_or_empty(_row_abs_residual(row) for row in rows),
        "mean_reward_total": _mean_or_empty(_finite_values(rows, "reward_total")),
        "mean_reward_comfort": _mean_or_empty(_finite_values(rows, "reward_comfort")),
        "mean_reward_stability": _mean_or_empty(_finite_values(rows, "reward_stability")),
        "mean_reward_task": _mean_or_empty(_finite_values(rows, "reward_task")),
        "mean_reward_action": _mean_or_empty(_finite_values(rows, "reward_action")),
        "mean_reward_safety": _mean_or_empty(_finite_values(rows, "reward_safety")),
        "observation_clip_ratio": observation_clip_ratio,
        "safety_gate_reason_counts": _format_counts(safety_counts),
        "fallback_reason_counts": _format_counts(fallback_counts),
        "collision_count": collision_count,
        "lane_invasion_count": lane_count,
        "red_light_count": red_count,
        "blocked_vehicle_count": blocked_count,
        "route_timeout_count": timeout_count,
    }
    row.update(_phase4_route_health_checks(row, rows))
    if row["warnings"]:
        warnings.extend(item for item in str(row["warnings"]).split(";") if item)
    row["warnings"] = ";".join(_unique(warnings))
    return _ordered_row(row, SUMMARY_FIELDS)


def phase4_route_health_bucket_rows(
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    buckets: List[Dict[str, Any]] = []
    for lower, upper, label in PROGRESS_BUCKETS:
        bucket_rows = [
            row for row in rows
            if _in_progress_bucket(_row_progress_fraction(row), lower, upper)
        ]
        reason_counter = _reason_counts(
            row.get("rl_safety_gate_reason")
            for row in bucket_rows)
        common_reason = ""
        if reason_counter:
            common_reason = sorted(
                reason_counter.items(),
                key=lambda item: (-item[1], item[0]))[0][0]
        buckets.append(_ordered_row({
            "bucket": label,
            "rows": len(bucket_rows),
            "mean_speed": _mean_or_empty(_finite_values(bucket_rows, "speed")),
            "mean_abs_action": _mean_or_empty(_row_abs_action(row) for row in bucket_rows),
            "mean_abs_residual_damper": _mean_or_empty(
                _row_abs_residual(row) for row in bucket_rows),
            "fallback_ratio": _ratio(
                bucket_rows,
                lambda row: bool(_nonempty(row.get("rl_fallback_reason")))),
            "low_speed_mask_ratio": _ratio(bucket_rows, _low_speed_mask_active),
            "hard_safety_gate_ratio": _ratio(bucket_rows, _hard_safety_active),
            "mean_reward_total": _mean_or_empty(_finite_values(bucket_rows, "reward_total")),
            "mean_reward_task": _mean_or_empty(_finite_values(bucket_rows, "reward_task")),
            "mean_reward_action": _mean_or_empty(_finite_values(bucket_rows, "reward_action")),
            "progress_stall_ratio": _ratio(
                bucket_rows,
                lambda row: _positive(row.get("route_progress_stall")) or
                _positive(row.get("progress_stall"))),
            "most_common_safety_reason": common_reason,
        }, BUCKET_FIELDS))
    return buckets


def _phase4_route_health_checks(
    row: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    failed: List[str] = []
    warnings: List[str] = []
    real_backend_ok = int(
        _int_value(row.get("real_backend_used")) == 1 and
        _int_value(row.get("fake_backend_used")) == 0)
    hero_ok = int(
        _int_value(row.get("carla_connected")) == 1 and
        _int_value(row.get("route_process_started")) == 1 and
        _int_value(row.get("hero_attached")) == 1)
    reward_ok = int((_float_value(row.get("reward_row_ratio")) or 0.0) >= 0.95)
    finite_reward_ok = int(_reward_columns_finite(rows))
    progress_available = _float_value(row.get("route_progress_available_ratio"))
    if progress_available is None:
        progress_available_ok = 1 if row.get("route_progress_monotonic_fraction_max") != "" else 0
        warnings.append("route_progress_available_ratio_missing")
    else:
        progress_available_ok = int(progress_available >= 0.95)
    monotonic_progress_ok = int(_int_value(row.get("route_progress_monotonic_negative_rows")) == 0)
    hard_safety_ok = int((_float_value(row.get("hard_safety_gate_ratio")) or 0.0) <= 1.0e-12)
    low_speed_mask_ok = int((_float_value(row.get("low_speed_mask_ratio")) or 0.0) <= 0.45)
    fallback_ok = int((_float_value(row.get("fallback_ratio")) or 0.0) <= 0.45)
    effective_control_ok = int((_float_value(row.get("effective_control_ratio")) or 0.0) >= 0.50)
    residual = _float_value(row.get("mean_abs_residual_damper"))
    residual_ok = int(residual is not None and 0.001 <= residual <= 0.06)
    action = _float_value(row.get("mean_abs_action"))
    action_ok = int(action is not None and 0.01 <= action <= 0.95)
    observation_clip_ok = int((_float_value(row.get("observation_clip_ratio")) or 0.0) <= 0.05)
    infraction_proxy_ok = int(all(
        (_float_value(row.get(key)) or 0.0) <= 1.0e-12
        for key in (
            "collision_count",
            "lane_invasion_count",
            "red_light_count",
            "blocked_vehicle_count",
            "route_timeout_count")))

    checks = {
        "real_backend_ok": real_backend_ok,
        "hero_ok": hero_ok,
        "reward_ok": reward_ok,
        "finite_reward_ok": finite_reward_ok,
        "progress_available_ok": progress_available_ok,
        "monotonic_progress_ok": monotonic_progress_ok,
        "hard_safety_ok": hard_safety_ok,
        "low_speed_mask_ok": low_speed_mask_ok,
        "fallback_ok": fallback_ok,
        "effective_control_ok": effective_control_ok,
        "residual_ok": residual_ok,
        "action_ok": action_ok,
        "observation_clip_ok": observation_clip_ok,
        "infraction_proxy_ok": infraction_proxy_ok,
    }
    for name, ok in checks.items():
        if not ok:
            failed.append(name[:-3] if name.endswith("_ok") else name)
    result = dict(checks)
    result["phase4_route_health_status"] = "pass" if not failed else "fail"
    result["failed_checks"] = ";".join(failed)
    result["warnings"] = ";".join(warnings)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rollout-csv", default="")
    parser.add_argument("--rollout-jsonl", default="")
    parser.add_argument("--training-summary", default="")
    parser.add_argument("--training-config", default="")
    parser.add_argument("--phase4-canary", default="")
    parser.add_argument("--reference-diagnostic-rows", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    csv_path, json_path, buckets_path, row, bucket_rows = write_phase4_route_health(
        args.output_dir,
        rollout_csv_path=args.rollout_csv,
        rollout_jsonl_path=args.rollout_jsonl,
        training_summary_path=args.training_summary,
        training_config_path=args.training_config,
        phase4_canary_path=args.phase4_canary,
        reference_diagnostic_rows=args.reference_diagnostic_rows)
    print("phase4 route health: %s" % csv_path)
    print("phase4 route health status: %s" % row.get("phase4_route_health_status"))
    print("phase4 route health json: %s" % json_path)
    print("phase4 route health buckets: %s" % buckets_path)
    print("phase4 route health bucket rows: %d" % len(bucket_rows))
    return 0 if row.get("phase4_route_health_status") == "pass" else 1


def _in_progress_bucket(value: Optional[float], lower: float, upper: float) -> bool:
    if value is None:
        return lower == 0.0
    if upper >= 1.0:
        return lower <= value <= upper
    return lower <= value < upper


def _row_progress_fraction(row: Mapping[str, Any]) -> Optional[float]:
    value = _float_value(row.get("route_progress_monotonic_fraction"))
    if value is not None:
        return max(0.0, min(1.0, value))
    value = _float_value(row.get("route_progress_fraction"))
    if value is not None:
        return max(0.0, min(1.0, value))
    return None


def _row_abs_action(row: Mapping[str, Any]) -> Optional[float]:
    value = _float_value(row.get("rl_mean_abs_action"))
    if value is not None:
        return abs(value)
    values = [
        abs(value)
        for value in (
            _float_value(row.get("action_fl")),
            _float_value(row.get("action_fr")),
            _float_value(row.get("action_rl")),
            _float_value(row.get("action_rr")),
        )
        if value is not None
    ]
    return sum(values) / float(len(values)) if values else None


def _row_abs_residual(row: Mapping[str, Any]) -> Optional[float]:
    value = _float_value(row.get("rl_mean_abs_residual_damper"))
    if value is not None:
        return abs(value)
    values = []
    for key in (
            "rl_residual_damper_fl",
            "rl_residual_damper_fr",
            "rl_residual_damper_rl",
            "rl_residual_damper_rr",
            "final_residual_damper_fl",
            "final_residual_damper_fr",
            "final_residual_damper_rl",
            "final_residual_damper_rr"):
        parsed = _float_value(row.get(key))
        if parsed is not None:
            values.append(abs(parsed))
    return sum(values) / float(len(values)) if values else None


def _low_speed_mask_active(row: Mapping[str, Any]) -> bool:
    reasons = set(_split_reasons(row.get("rl_safety_gate_reason")))
    return (
        _positive(row.get("rl_safety_gate_speed_limit")) or
        "speed_below_min" in reasons)


def _hard_safety_active(row: Mapping[str, Any]) -> bool:
    if _low_speed_mask_active(row):
        return False
    reasons = set(_split_reasons(row.get("rl_safety_gate_reason")))
    hard_condition = (
        _positive(row.get("rl_safety_gate_nonfinite_obs")) or
        _positive(row.get("rl_safety_gate_action_invalid")) or
        "nonfinite_obs" in reasons or
        "action_invalid" in reasons)
    safety_gain = _float_value(row.get("rl_safety_gain"))
    gate_active = _positive(row.get("rl_safety_gate_active")) or bool(reasons)
    return bool(hard_condition or (
        gate_active and safety_gain is not None and safety_gain <= 1.0e-12))


def _soft_safety_active(row: Mapping[str, Any]) -> bool:
    if _low_speed_mask_active(row):
        return False
    safety_gain = _float_value(row.get("rl_safety_gain"))
    return bool(safety_gain is not None and 1.0e-12 < safety_gain < 1.0 - 1.0e-12)


def _reward_columns_finite(rows: Sequence[Mapping[str, Any]]) -> bool:
    for row in rows:
        for key in REWARD_COLUMNS:
            if key not in row or str(row.get(key, "")).strip() == "":
                continue
            if _float_value(row.get(key)) is None:
                return False
    return True


def _availability_ratio(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> Tuple[Any, bool]:
    if not rows:
        return "", True
    present = [row for row in rows if key in row and _nonempty(row.get(key))]
    if not present:
        return "", True
    positive = sum(1 for row in present if _positive(row.get(key)))
    return positive / float(len(present)), False


def _ratio(rows: Sequence[Mapping[str, Any]], predicate: Any) -> Any:
    if not rows:
        return ""
    return sum(1 for row in rows if predicate(row)) / float(len(rows))


def _reason_counts(values: Iterable[Any]) -> Counter:
    counter: Counter = Counter()
    for value in values:
        for reason in _split_reasons(value):
            counter[reason] += 1
    return counter


def _split_reasons(value: Any) -> List[str]:
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


def _format_counts(counter: Counter) -> str:
    return ";".join("%s=%d" % (key, counter[key]) for key in sorted(counter))


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = _float_value(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _finite_count(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return len(_finite_values(rows, key))


def _mean_or_empty(values: Iterable[Optional[float]]) -> Any:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / float(len(finite)) if finite else ""


def _percentile_or_empty(values: Sequence[float], fraction: float) -> Any:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return ""
    if len(finite) == 1:
        return finite[0]
    index = (len(finite) - 1) * max(0.0, min(1.0, fraction))
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return finite[lower]
    weight = index - lower
    return finite[lower] * (1.0 - weight) + finite[upper] * weight


def _max_or_empty(values: Sequence[float]) -> Any:
    return max(values) if values else ""


def _monotonic_negative_rows(values: Sequence[float]) -> int:
    previous: Optional[float] = None
    count = 0
    for value in values:
        if previous is not None and value < previous - 1.0e-9:
            count += 1
        previous = value
    return count


def _max_first(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    fallback: Any = 0,
) -> Any:
    values: List[float] = []
    for row in rows:
        for key in keys:
            value = _float_value(row.get(key))
            if value is not None:
                values.append(value)
                break
    if values:
        return max(values)
    fallback_value = _float_value(fallback)
    return fallback_value if fallback_value is not None else 0.0


def _missing_any_column(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> bool:
    if not rows:
        return True
    for key in keys:
        if all(key not in row or not _nonempty(row.get(key)) for row in rows):
            return True
    return False


def _first_value(*sources: Mapping[str, Any], key: str = "") -> Any:
    if key:
        for source in sources:
            if key in source and _nonempty(source.get(key)):
                return source.get(key)
    return ""


def _first_nonempty(
    *sources: Any,
    key: str,
) -> str:
    for source in sources:
        if isinstance(source, Mapping):
            value = source.get(key)
            if _nonempty(value):
                return _nonempty(value)
        elif isinstance(source, Sequence):
            for row in source:
                if isinstance(row, Mapping) and _nonempty(row.get(key)):
                    return _nonempty(row.get(key))
    return ""


def _int_first(*sources: Mapping[str, Any], keys: Sequence[str]) -> int:
    for source in sources:
        for key in keys:
            if key in source:
                return _int_value(source.get(key))
    return 0


def _float_first(*sources: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for source in sources:
        for key in keys:
            value = _float_value(source.get(key))
            if value is not None:
                return value
    return ""


def _float_first_row(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        value = _float_value(row.get(key))
        if value is not None:
            return value
    return None


def _positive(value: Any) -> bool:
    parsed = _float_value(value)
    return bool(parsed is not None and parsed > 0.0)


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


def _nonempty(value: Any) -> str:
    return str(value or "").strip()


def _unique(items: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, newline="") as csv_file:
        return [dict(row) for row in csv.DictReader(csv_file)]


def _read_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path) as jsonl_file:
        for line in jsonl_file:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, Mapping):
                rows.append(dict(data))
    return rows


def _read_json_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as json_file:
            data = json.load(json_file)
        return dict(data) if isinstance(data, Mapping) else {}
    except Exception:
        return {}


def _write_single_row_csv(path: str, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def _write_csv_rows(path: str, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
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


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path or "")))


def _resolve_path(path: Any, output_dir: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    expanded = os.path.expanduser(raw)
    candidates = []
    if os.path.isabs(expanded):
        candidates.append(expanded)
    else:
        candidates.append(os.path.join(output_dir, expanded))
    if expanded.startswith("/workspace/"):
        candidates.append(os.path.join(_sim_root(), expanded[len("/workspace/"):]))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return os.path.abspath(candidates[0])


def _sim_root() -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        ".."))


if __name__ == "__main__":
    raise SystemExit(main())

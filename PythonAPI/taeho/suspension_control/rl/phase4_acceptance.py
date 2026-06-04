"""Phase 4 training and evaluation acceptance reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .normalizer import FixedScaleNormalizer
from .policy import PolicyAdapter


PHASE4_TRAINING_FIELDS: Tuple[str, ...] = (
    "phase4_status",
    "failed_checks",
    "warnings",
    "backend",
    "real_backend_used",
    "fake_backend_used",
    "carla_connected",
    "route_process_started",
    "hero_attached",
    "warmup_seconds",
    "warmup_steps",
    "total_timesteps_requested",
    "total_timesteps_collected",
    "episode_count",
    "terminated_count",
    "truncated_count",
    "reward_rows",
    "reward_row_ratio",
    "mean_reward_total",
    "mean_reward_comfort",
    "mean_reward_stability",
    "mean_reward_task",
    "mean_reward_action",
    "mean_reward_safety",
    "mean_abs_action",
    "mean_abs_residual_damper",
    "effective_control_ratio",
    "fallback_ratio",
    "hard_safety_gate_ratio",
    "low_speed_mask_ratio",
    "observation_clip_ratio",
    "route_progress_available_ratio",
    "route_progress_monotonic_fraction_max",
    "route_progress_monotonic_negative_rows",
    "policy_saved",
    "policy_exported",
    "policy_adapter_load_ok",
    "normalizer_saved",
    "normalizer_load_ok",
    "rollout_csv",
    "rollout_jsonl",
    "sb3_model_path",
    "policy_path",
    "normalizer_path",
)


PHASE4_EVAL_FIELDS: Tuple[str, ...] = (
    "scenario",
    "phase4_eval_status",
    "failed_checks",
    "warnings",
    "route_score_ok",
    "score_composed_ok",
    "infraction_ok",
    "verification_ok",
    "policy_available_ok",
    "policy_fallback_ok",
    "reward_rows_ok",
    "score_route",
    "score_composed",
    "score_penalty",
    "sidecar_command_verifies",
    "rl_policy_available_ratio",
    "rl_fallback_rows",
    "rl_fallback_reasons",
    "reward_rows",
    "phase3b_status",
)


def write_phase4_training_canary(
    output_dir: str,
    *,
    training_summary_path: str = "",
    training_config_path: str = "",
) -> Tuple[str, str, Dict[str, Any]]:
    """Write Phase 4 training canary CSV/JSON for a training output dir."""

    output_dir = _abs_path(output_dir)
    row = phase4_training_canary_row(
        output_dir,
        training_summary_path=training_summary_path,
        training_config_path=training_config_path)
    csv_path = os.path.join(output_dir, "phase4_training_canary.csv")
    json_path = os.path.join(output_dir, "phase4_training_canary.json")
    _write_single_row_csv(csv_path, row, PHASE4_TRAINING_FIELDS)
    _write_json(json_path, row)
    return csv_path, json_path, row


def phase4_training_canary_row(
    output_dir: str,
    *,
    training_summary_path: str = "",
    training_config_path: str = "",
) -> Dict[str, Any]:
    output_dir = _abs_path(output_dir)
    summary_path = _resolve_path(
        training_summary_path or os.path.join(output_dir, "training_summary.json"),
        output_dir)
    config_path = _resolve_path(
        training_config_path or os.path.join(output_dir, "training_config.json"),
        output_dir)
    summary = _read_json_dict(summary_path)
    config = _read_json_dict(config_path)

    rollout_csv = _resolve_path(
        summary.get("training_rollout_csv") or os.path.join(
            output_dir,
            "training_rollout.csv"),
        output_dir)
    rollout_jsonl = _resolve_path(
        summary.get("training_rollout_jsonl") or os.path.join(
            output_dir,
            "training_rollout.jsonl"),
        output_dir)
    rows = _read_csv_rows(rollout_csv)

    sb3_model_path = _resolve_path(
        summary.get("sb3_model_path") or os.path.join(output_dir, "sb3_model.zip"),
        output_dir)
    policy_path = _resolve_path(
        summary.get("policy_path") or _first_existing(
            output_dir,
            ("policy.ts", "policy.pt")),
        output_dir)
    normalizer_path = _resolve_path(
        summary.get("normalizer_path") or os.path.join(output_dir, "normalizer.json"),
        output_dir)

    total_requested = _int_first(
        summary,
        config,
        keys=("total_timesteps", "total_timesteps_requested"))
    total_collected = len(rows) if rows else _int_value(summary.get("rollout_rows"))
    reward_rows = _finite_count(rows, "reward_total")
    reward_row_ratio = (
        float(reward_rows) / float(total_collected)
        if total_collected > 0 else 0.0)
    warmup_seconds = _float_first(summary, config, keys=("warmup_seconds",))
    fixed_delta_value = _float_first(summary, config, keys=("fixed_delta_seconds",))
    fixed_delta = fixed_delta_value if fixed_delta_value > 0.0 else 0.05
    warmup_steps = int(round(warmup_seconds / fixed_delta)) if warmup_seconds else 0

    policy_saved = int(os.path.isfile(sb3_model_path))
    policy_exported = int(
        bool(summary.get("policy_exported")) and
        os.path.isfile(policy_path))
    policy_adapter_ok = _policy_adapter_load_ok(
        policy_path,
        _int_value(summary.get("observation_dim")) or _normalizer_dim(normalizer_path))
    normalizer_saved = int(os.path.isfile(normalizer_path))
    normalizer_load_ok = _normalizer_load_ok(normalizer_path)

    carla_connected = _max_int(rows, "carla_connected", summary.get("carla_connected"))
    route_process_started = _max_int(
        rows,
        "route_process_started",
        summary.get("route_process_started"))
    hero_attached = _max_int(rows, "hero_attached", summary.get("hero_attached"))
    real_backend_used = _max_int(rows, "real_backend_used", summary.get("real_backend_used"))
    fake_backend_used = _max_int(rows, "fake_backend_used", summary.get("fake_backend_used"))
    route_available_ratio = _ratio_positive(rows, "route_progress_available")
    monotonic_values = _finite_values(rows, "route_progress_monotonic_fraction")
    monotonic_negative_rows = _negative_delta_rows(monotonic_values)
    hard_safety_gate_ratio = _metric_ratio(
        summary,
        rows,
        "hard_safety_gate_ratio",
        "rl_safety_gate_active")

    row: Dict[str, Any] = {
        "backend": summary.get("backend", config.get("backend", "")),
        "real_backend_used": real_backend_used,
        "fake_backend_used": fake_backend_used,
        "carla_connected": carla_connected,
        "route_process_started": route_process_started,
        "hero_attached": hero_attached,
        "warmup_seconds": warmup_seconds,
        "warmup_steps": warmup_steps,
        "total_timesteps_requested": total_requested,
        "total_timesteps_collected": total_collected,
        "episode_count": len(set(_nonempty(row.get("episode_id")) for row in rows)) if rows else 0,
        "terminated_count": _sum_int(rows, "terminated"),
        "truncated_count": _sum_int(rows, "truncated"),
        "reward_rows": reward_rows,
        "reward_row_ratio": reward_row_ratio,
        "mean_reward_total": _metric_mean(summary, rows, "mean_reward_total", "reward_total"),
        "mean_reward_comfort": _metric_mean(summary, rows, "mean_reward_comfort", "reward_comfort"),
        "mean_reward_stability": _metric_mean(summary, rows, "mean_reward_stability", "reward_stability"),
        "mean_reward_task": _metric_mean(summary, rows, "mean_reward_task", "reward_task"),
        "mean_reward_action": _metric_mean(summary, rows, "mean_reward_action", "reward_action"),
        "mean_reward_safety": _metric_mean(summary, rows, "mean_reward_safety", "reward_safety"),
        "mean_abs_action": _float_or(summary.get("mean_abs_action"), _mean_abs_actions(rows)),
        "mean_abs_residual_damper": _float_or(
            summary.get("mean_abs_residual_damper"),
            _mean_abs_columns(rows, (
                "rl_residual_damper_fl",
                "rl_residual_damper_fr",
                "rl_residual_damper_rl",
                "rl_residual_damper_rr"))),
        "effective_control_ratio": _float_or(
            summary.get("effective_control_ratio"),
            _ratio_any_abs(rows, (
                "rl_residual_damper_fl",
                "rl_residual_damper_fr",
                "rl_residual_damper_rl",
                "rl_residual_damper_rr"))),
        "fallback_ratio": _float_or(
            summary.get("fallback_ratio"),
            _ratio_nonempty(rows, "rl_fallback_reason")),
        "hard_safety_gate_ratio": hard_safety_gate_ratio,
        "low_speed_mask_ratio": _metric_ratio(
            summary,
            rows,
            "low_speed_mask_ratio",
            "low_speed_not_planned"),
        "observation_clip_ratio": _float_or(
            summary.get("observation_clip_ratio"),
            _ratio_positive(rows, "observation_clip_count")),
        "route_progress_available_ratio": route_available_ratio,
        "route_progress_monotonic_fraction_max": (
            max(monotonic_values) if monotonic_values else ""),
        "route_progress_monotonic_negative_rows": monotonic_negative_rows,
        "policy_saved": policy_saved,
        "policy_exported": policy_exported,
        "policy_adapter_load_ok": policy_adapter_ok,
        "normalizer_saved": normalizer_saved,
        "normalizer_load_ok": normalizer_load_ok,
        "rollout_csv": rollout_csv,
        "rollout_jsonl": rollout_jsonl,
        "sb3_model_path": sb3_model_path,
        "policy_path": policy_path,
        "normalizer_path": normalizer_path,
    }
    failed_checks, warnings = _phase4_training_checks(
        row,
        summary,
        actions_finite=_actions_finite(rows),
        rewards_finite=_rewards_finite(rows))
    row["phase4_status"] = "pass" if not failed_checks else "fail"
    row["failed_checks"] = ";".join(failed_checks)
    row["warnings"] = ";".join(warnings)
    return _ordered_row(row, PHASE4_TRAINING_FIELDS)


def write_phase4_eval_acceptance_report(
    output_dir: str,
    *,
    suite_summary_path: str = "",
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """Write optional route-suite evaluation acceptance CSV/JSON."""

    output_dir = _abs_path(output_dir)
    summary_path = _resolve_path(
        suite_summary_path or os.path.join(output_dir, "suite_summary.csv"),
        output_dir)
    summary_rows = _read_csv_rows(summary_path)
    rows = phase4_eval_acceptance_rows(summary_rows)
    csv_path = os.path.join(output_dir, "phase4_eval_acceptance.csv")
    json_path = os.path.join(output_dir, "phase4_eval_acceptance.json")
    _write_csv_rows(csv_path, rows, PHASE4_EVAL_FIELDS)
    _write_json(json_path, rows)
    return csv_path, json_path, rows


def phase4_eval_acceptance_rows(
    summary_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    if not summary_rows:
        return [_ordered_row({
            "scenario": "",
            "phase4_eval_status": "fail",
            "failed_checks": "suite_summary",
            "warnings": "",
        }, PHASE4_EVAL_FIELDS)]
    rows: List[Dict[str, Any]] = []
    for source in summary_rows:
        failed: List[str] = []
        warnings: List[str] = []
        score_route = _float_value(source.get("score_route"))
        route_score_ok = int(score_route is not None and abs(score_route - 100.0) <= 1.0e-6)
        if not route_score_ok:
            failed.append("route_score")
        score_composed = _float_value(source.get("score_composed"))
        composed_ok = int(score_composed is not None and abs(score_composed - 100.0) <= 1.0e-6)
        if not composed_ok:
            failed.append("score_composed")
        infraction_ok = int(_infractions_zero(source))
        if not infraction_ok:
            failed.append("infractions")
        verification_ok = int((_float_value(source.get("sidecar_command_verifies")) or 0.0) > 0.0)
        if not verification_ok:
            failed.append("verification")
        policy_ratio = _float_value(
            source.get("rl_policy_available_ratio", source.get("policy_available_ratio")))
        policy_available_ok = int(policy_ratio is not None and policy_ratio >= 0.99)
        if not policy_available_ok:
            failed.append("policy_available_ratio")
        fallback_text = "%s;%s;%s" % (
            source.get("rl_fallback_reasons", ""),
            source.get("policy_error", ""),
            source.get("rl_policy_error", ""))
        policy_fallback_ok = int(
            "policy_unavailable" not in fallback_text and
            "missing_path" not in fallback_text and
            "policy_file_missing" not in fallback_text)
        if not policy_fallback_ok:
            failed.append("policy_fallback")
        reward_rows = _float_value(
            source.get("reward_rows",
                       source.get("reward_diagnostic_rows",
                                  source.get("rl_diagnostic_rows",
                                             source.get("diagnostic_rows")))))
        reward_rows_ok = int(reward_rows is not None and reward_rows > 0.0)
        if not reward_rows_ok:
            failed.append("reward_rows")
        phase3b_status = str(source.get("phase3b_status", ""))
        if phase3b_status and phase3b_status not in ("pass", "warn"):
            warnings.append("phase3b_status_%s" % phase3b_status)
        row = {
            "scenario": source.get("scenario", ""),
            "phase4_eval_status": "pass" if not failed else "fail",
            "failed_checks": ";".join(failed),
            "warnings": ";".join(warnings),
            "route_score_ok": route_score_ok,
            "score_composed_ok": composed_ok,
            "infraction_ok": infraction_ok,
            "verification_ok": verification_ok,
            "policy_available_ok": policy_available_ok,
            "policy_fallback_ok": policy_fallback_ok,
            "reward_rows_ok": reward_rows_ok,
            "score_route": source.get("score_route", ""),
            "score_composed": source.get("score_composed", ""),
            "score_penalty": source.get("score_penalty", ""),
            "sidecar_command_verifies": source.get("sidecar_command_verifies", ""),
            "rl_policy_available_ratio": (
                policy_ratio if policy_ratio is not None else ""),
            "rl_fallback_rows": source.get("rl_fallback_rows", ""),
            "rl_fallback_reasons": source.get("rl_fallback_reasons", ""),
            "reward_rows": reward_rows if reward_rows is not None else "",
            "phase3b_status": phase3b_status,
        }
        rows.append(_ordered_row(row, PHASE4_EVAL_FIELDS))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output-dir", default="")
    parser.add_argument("--training-summary", default="")
    parser.add_argument("--training-config", default="")
    parser.add_argument("--eval-output-dir", default="")
    parser.add_argument("--suite-summary", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    wrote_any = False
    if args.training_output_dir:
        csv_path, json_path, row = write_phase4_training_canary(
            args.training_output_dir,
            training_summary_path=args.training_summary,
            training_config_path=args.training_config)
        print("phase4 training canary: %s" % csv_path)
        print("phase4 training status: %s" % row.get("phase4_status"))
        print("phase4 training json: %s" % json_path)
        wrote_any = True
    if args.eval_output_dir:
        csv_path, json_path, rows = write_phase4_eval_acceptance_report(
            args.eval_output_dir,
            suite_summary_path=args.suite_summary)
        print("phase4 eval acceptance: %s" % csv_path)
        print("phase4 eval rows: %d" % len(rows))
        print("phase4 eval json: %s" % json_path)
        wrote_any = True
    if not wrote_any:
        build_arg_parser().print_help()
        return 2
    return 0


def _phase4_training_checks(
    row: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    actions_finite: bool,
    rewards_finite: bool,
) -> Tuple[List[str], List[str]]:
    failed: List[str] = []
    warnings: List[str] = []
    if str(row.get("backend", "")) != "live":
        failed.append("backend")
        if str(row.get("backend", "")) == "fake":
            warnings.append("fake_backend_smoke_cannot_pass_real_canary")
    if _int_value(row.get("real_backend_used")) != 1:
        failed.append("real_backend_used")
    if _int_value(row.get("carla_connected")) != 1:
        failed.append("carla_connected")
    if _int_value(row.get("route_process_started")) != 1:
        failed.append("route_process_started")
    if _int_value(row.get("hero_attached")) != 1:
        failed.append("hero_attached")
    requested = _int_value(row.get("total_timesteps_requested"))
    collected = _int_value(row.get("total_timesteps_collected"))
    if requested > 0 and collected < min(requested, 128):
        failed.append("total_timesteps_collected")
    if _int_value(row.get("reward_rows")) <= 0:
        failed.append("reward_rows")
    if _float_value(row.get("reward_row_ratio")) is None or _float_value(row.get("reward_row_ratio")) < 0.95:
        failed.append("reward_row_ratio")
    for field in ("policy_saved", "policy_exported", "policy_adapter_load_ok",
                  "normalizer_saved", "normalizer_load_ok"):
        if _int_value(row.get(field)) != 1:
            failed.append(field)
    hard_gate_ratio = _float_value(row.get("hard_safety_gate_ratio"))
    if hard_gate_ratio is not None and hard_gate_ratio > 0.01:
        failed.append("hard_safety_gate_ratio")
    if not actions_finite:
        failed.append("actions_finite")
    if not rewards_finite:
        failed.append("rewards_finite")
    if str(summary.get("status", "")) and str(summary.get("status")) != "trained":
        failed.append("training_status")
    if row.get("route_progress_available_ratio", "") == "":
        warnings.append("route_progress_available_ratio_missing")
    return failed, warnings


def _policy_adapter_load_ok(path: str, observation_dim: int) -> int:
    if not path or not os.path.isfile(path):
        return 0
    try:
        policy = PolicyAdapter(path, action_dim=4)
        if not policy.is_available:
            return 0
        observation_dim = int(observation_dim or 0)
        if observation_dim <= 0:
            return 1
        action = policy.predict([0.0 for _ in range(observation_dim)])
        return int(len(action) == 4 and all(math.isfinite(float(value)) for value in action))
    except Exception:
        return 0


def _normalizer_load_ok(path: str) -> int:
    if not path or not os.path.isfile(path):
        return 0
    normalizer = FixedScaleNormalizer(path)
    return int(bool(normalizer.loaded))


def _normalizer_dim(path: str) -> int:
    data = _read_json_dict(path)
    names = data.get("feature_names", [])
    return len(names) if isinstance(names, list) else 0


def _metric_mean(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary_key: str,
    row_key: str,
) -> Any:
    value = _float_value(summary.get(summary_key))
    return value if value is not None else _mean(_finite_values(rows, row_key))


def _metric_ratio(
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    summary_key: str,
    row_key: str,
) -> Any:
    value = _float_value(summary.get(summary_key))
    return value if value is not None else _ratio_positive(rows, row_key)


def _mean_abs_actions(rows: Sequence[Mapping[str, Any]]) -> Any:
    return _mean_abs_columns(rows, ("action_fl", "action_fr", "action_rl", "action_rr"))


def _mean_abs_columns(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Any:
    values: List[float] = []
    for row in rows:
        for column in columns:
            value = _float_value(row.get(column))
            if value is not None:
                values.append(abs(value))
    return _mean(values)


def _ratio_any_abs(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> Any:
    if not rows:
        return ""
    active = 0
    for row in rows:
        if any(abs(_float_value(row.get(column)) or 0.0) > 1.0e-12 for column in columns):
            active += 1
    return float(active) / float(len(rows))


def _ratio_nonempty(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    if not rows:
        return ""
    return float(sum(1 for row in rows if str(row.get(key, "")).strip())) / float(len(rows))


def _ratio_positive(rows: Sequence[Mapping[str, Any]], key: str) -> Any:
    if not rows:
        return ""
    return float(sum(1 for row in rows if (_float_value(row.get(key)) or 0.0) > 0.0)) / float(len(rows))


def _finite_count(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for row in rows if _float_value(row.get(key)) is not None)


def _actions_finite(rows: Sequence[Mapping[str, Any]]) -> bool:
    columns = ("action_fl", "action_fr", "action_rl", "action_rr")
    return _columns_finite(rows, columns)


def _rewards_finite(rows: Sequence[Mapping[str, Any]]) -> bool:
    return _columns_finite(rows, ("reward_total",))


def _columns_finite(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bool:
    for row in rows:
        for column in columns:
            if column not in row or str(row.get(column, "")).strip() == "":
                continue
            if _float_value(row.get(column)) is None:
                return False
    return True


def _negative_delta_rows(values: Sequence[float]) -> int:
    count = 0
    previous: Optional[float] = None
    for value in values:
        if previous is not None and value < previous - 1.0e-9:
            count += 1
        previous = value
    return count


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    result: List[float] = []
    for row in rows:
        value = _float_value(row.get(key))
        if value is not None:
            result.append(value)
    return result


def _mean(values: Sequence[float]) -> Any:
    return sum(values) / float(len(values)) if values else ""


def _sum_int(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(_int_value(row.get(key)) for row in rows)


def _max_int(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    fallback: Any = 0,
) -> int:
    values = [_int_value(row.get(key)) for row in rows if key in row]
    if values:
        return max(values)
    return _int_value(fallback)


def _int_first(*sources: Mapping[str, Any], keys: Sequence[str]) -> int:
    for source in sources:
        for key in keys:
            if key in source:
                return _int_value(source.get(key))
    return 0


def _float_first(*sources: Mapping[str, Any], keys: Sequence[str]) -> float:
    for source in sources:
        for key in keys:
            if key in source:
                value = _float_value(source.get(key))
                if value is not None:
                    return value
    return 0.0


def _float_or(value: Any, fallback: Any) -> Any:
    parsed = _float_value(value)
    return parsed if parsed is not None else fallback


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


def _infractions_zero(row: Mapping[str, Any]) -> bool:
    for key in (
            "collision_count",
            "lane_invasion_count",
            "red_light_count",
            "blocked_vehicle_count",
            "route_timeout_count",
            "collisions_layout",
            "collisions_pedestrian",
            "collisions_vehicle",
            "red_light",
            "outside_route_lanes"):
        value = _float_value(row.get(key))
        if value is not None and abs(value) > 1.0e-9:
            return False
    penalty = _float_value(row.get("score_penalty"))
    if penalty is not None and abs(penalty) > 1.0e-9:
        return False
    return True


def _first_existing(output_dir: str, names: Sequence[str]) -> str:
    for name in names:
        path = os.path.join(output_dir, name)
        if os.path.isfile(path):
            return path
    return os.path.join(output_dir, names[0]) if names else ""


def _read_json_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as json_file:
            data = json.load(json_file)
        return dict(data) if isinstance(data, Mapping) else {}
    except Exception:
        return {}


def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, newline="") as csv_file:
            return [dict(row) for row in csv.DictReader(csv_file)]
    except Exception:
        return []


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
    if expanded.startswith("/workspace"):
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

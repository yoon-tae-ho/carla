"""Phase 4-D training gate for exported-policy evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


PHASE4D_TRAIN_ACCEPTANCE_FIELDS: Tuple[str, ...] = (
    "phase4d_train_status",
    "eval_allowed",
    "invalid_for_eval",
    "eval_status",
    "failed_checks",
    "warnings",
    "training_return_code",
    "training_status",
    "phase4_status",
    "backend",
    "real_backend_used",
    "fake_backend_used",
    "synthetic_training",
    "route_process_started",
    "hero_attached",
    "total_timesteps_requested",
    "total_timesteps_collected",
    "rollout_rows",
    "reward_row_ratio",
    "route_progress_available_ratio",
    "route_progress_fraction_max",
    "route_progress_fraction_min_required",
    "route_progress_monotonic_negative_rows",
    "observation_clip_ratio",
    "hard_safety_gate_ratio",
    "collision_count",
    "lane_invasion_count",
    "red_light_count",
    "route_timeout_count",
    "blocked_vehicle_count",
    "policy_exported",
    "normalizer_saved",
    "policy_path_exists",
    "normalizer_path_exists",
    "learn_error",
    "export_error",
    "training_summary_path",
    "phase4_training_canary_path",
    "policy_path",
    "normalizer_path",
)


def write_phase4d_train_acceptance(
    output_dir: str,
    *,
    training_output_dir: str = "",
    training_summary_path: str = "",
    phase4_training_canary_path: str = "",
    training_return_code: int = 0,
) -> Tuple[str, str, Dict[str, Any]]:
    output_dir = _abs_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    row = phase4d_train_acceptance_row(
        training_output_dir=training_output_dir,
        training_summary_path=training_summary_path,
        phase4_training_canary_path=phase4_training_canary_path,
        training_return_code=training_return_code)
    csv_path = os.path.join(output_dir, "phase4d_train_acceptance.csv")
    json_path = os.path.join(output_dir, "phase4d_train_acceptance.json")
    _write_single_row_csv(csv_path, row)
    _write_json(json_path, row)
    return csv_path, json_path, row


def phase4d_train_acceptance_row(
    *,
    training_output_dir: str = "",
    training_summary_path: str = "",
    phase4_training_canary_path: str = "",
    training_return_code: int = 0,
) -> Dict[str, Any]:
    training_output_dir = _abs_path(training_output_dir or ".")
    summary_path = _resolve_path(
        training_summary_path or os.path.join(training_output_dir, "training_summary.json"),
        training_output_dir)
    canary_path = _resolve_path(
        phase4_training_canary_path or os.path.join(
            training_output_dir,
            "phase4_training_canary.json"),
        training_output_dir)
    summary = _read_json_dict(summary_path)
    canary = _read_json_dict(canary_path)

    requested = _int_first(
        canary,
        summary,
        keys=("total_timesteps_requested", "total_timesteps"))
    collected = _int_first(
        canary,
        summary,
        keys=("total_timesteps_collected", "rollout_rows"))
    rollout_rows = _int_first(summary, canary, keys=("rollout_rows",))
    if rollout_rows <= 0:
        rollout_rows = collected
    policy_path = _resolve_path(
        _first_nonempty(summary.get("policy_path"), canary.get("policy_path"),
                        os.path.join(training_output_dir, "policy.ts")),
        training_output_dir)
    normalizer_path = _resolve_path(
        _first_nonempty(summary.get("normalizer_path"), canary.get("normalizer_path"),
                        os.path.join(training_output_dir, "normalizer.json")),
        training_output_dir)

    row: Dict[str, Any] = {
        "training_return_code": int(training_return_code),
        "training_status": summary.get("status", ""),
        "phase4_status": canary.get("phase4_status", ""),
        "backend": _first_nonempty(summary.get("backend"), canary.get("backend")),
        "real_backend_used": _int_first(summary, canary, keys=("real_backend_used",)),
        "fake_backend_used": _int_first(summary, canary, keys=("fake_backend_used",)),
        "synthetic_training": _int_first(summary, canary, keys=("synthetic_training",)),
        "route_process_started": _int_first(
            canary,
            summary,
            keys=("route_process_started",)),
        "hero_attached": _int_first(canary, summary, keys=("hero_attached",)),
        "total_timesteps_requested": requested,
        "total_timesteps_collected": collected,
        "rollout_rows": rollout_rows,
        "reward_row_ratio": _float_first(
            canary,
            summary,
            keys=("reward_row_ratio",)),
        "route_progress_available_ratio": _float_first(
            canary,
            summary,
            keys=("route_progress_available_ratio",)),
        "route_progress_fraction_max": _float_first(
            canary,
            summary,
            keys=(
                "route_progress_monotonic_fraction_max",
                "route_progress_fraction_max")),
        "route_progress_fraction_min_required": _route_progress_fraction_threshold(requested),
        "route_progress_monotonic_negative_rows": _int_first(
            canary,
            summary,
            keys=("route_progress_monotonic_negative_rows",)),
        "observation_clip_ratio": _float_first(
            canary,
            summary,
            keys=("observation_clip_ratio",)),
        "hard_safety_gate_ratio": _float_first(
            canary,
            summary,
            keys=("hard_safety_gate_ratio",)),
        "collision_count": _float_first(summary, canary, keys=("collision_count",)),
        "lane_invasion_count": _float_first(
            summary,
            canary,
            keys=("lane_invasion_count",)),
        "red_light_count": _float_first(summary, canary, keys=("red_light_count",)),
        "route_timeout_count": _float_first(
            summary,
            canary,
            keys=("route_timeout_count",)),
        "blocked_vehicle_count": _float_first(
            summary,
            canary,
            keys=("blocked_vehicle_count",)),
        "policy_exported": _int_first(summary, canary, keys=("policy_exported",)),
        "normalizer_saved": _int_first(canary, summary, keys=("normalizer_saved",)),
        "policy_path_exists": int(os.path.isfile(policy_path)),
        "normalizer_path_exists": int(os.path.isfile(normalizer_path)),
        "learn_error": summary.get("learn_error", ""),
        "export_error": summary.get("export_error", ""),
        "training_summary_path": summary_path,
        "phase4_training_canary_path": canary_path,
        "policy_path": policy_path,
        "normalizer_path": normalizer_path,
    }
    failed, warnings = _phase4d_train_failed_checks(row, summary, canary)
    row["phase4d_train_status"] = "pass" if not failed else "fail"
    row["eval_allowed"] = int(not failed)
    row["invalid_for_eval"] = int(bool(failed))
    row["eval_status"] = (
        "ready_for_eval"
        if not failed else
        "skipped_due_to_failed_training")
    row["failed_checks"] = ";".join(failed)
    row["warnings"] = ";".join(warnings)
    return _ordered_row(row)


def _phase4d_train_failed_checks(
    row: Mapping[str, Any],
    summary: Mapping[str, Any],
    canary: Mapping[str, Any],
) -> Tuple[list, list]:
    failed = []
    warnings = []

    if _int_value(row.get("training_return_code")) != 0:
        failed.append("training_return_code")
    if str(row.get("training_status", "")) != "trained":
        failed.append("training_status")
    if str(row.get("phase4_status", "")) != "pass":
        failed.append("phase4_status")
    if _nonempty(canary.get("failed_checks")):
        failed.append("phase4_failed_checks")
    if _nonempty(row.get("learn_error")):
        failed.append("learn_error")
    if _nonempty(row.get("export_error")):
        failed.append("export_error")
    if str(row.get("backend", "")) != "live":
        failed.append("backend")
    for field in (
            "real_backend_used",
            "route_process_started",
            "hero_attached",
            "policy_exported",
            "normalizer_saved",
            "policy_path_exists",
            "normalizer_path_exists"):
        if _int_value(row.get(field)) != 1:
            failed.append(field)
    for field in ("fake_backend_used", "synthetic_training"):
        if _int_value(row.get(field)) != 0:
            failed.append(field)

    requested = _int_value(row.get("total_timesteps_requested"))
    collected = _int_value(row.get("total_timesteps_collected"))
    rollout_rows = _int_value(row.get("rollout_rows"))
    if requested <= 0:
        failed.append("total_timesteps_requested")
    if collected != requested:
        failed.append("total_timesteps_collected")
    if rollout_rows < requested:
        failed.append("rollout_rows")

    if _float_missing_or_below(row.get("reward_row_ratio"), 0.99):
        failed.append("reward_row_ratio")
    if _float_missing_or_below(row.get("route_progress_available_ratio"), 0.99):
        failed.append("route_progress_available_ratio")
    if _int_value(row.get("route_progress_monotonic_negative_rows")) != 0:
        failed.append("route_progress_monotonic_negative_rows")
    if _float_missing_or_above(row.get("observation_clip_ratio"), 0.05):
        failed.append("observation_clip_ratio")
    if _float_missing_or_above(row.get("hard_safety_gate_ratio"), 0.0):
        failed.append("hard_safety_gate_ratio")
    route_progress_threshold = _float_value(
        row.get("route_progress_fraction_min_required"))
    if route_progress_threshold is None:
        route_progress_threshold = _route_progress_fraction_threshold(requested)
    if _float_missing_or_below(
            row.get("route_progress_fraction_max"),
            route_progress_threshold):
        failed.append("route_progress_fraction_max")
    for field in (
            "collision_count",
            "lane_invasion_count",
            "red_light_count",
            "route_timeout_count",
            "blocked_vehicle_count"):
        if _float_missing_or_above(row.get(field), 0.0):
            failed.append(field)

    if not summary:
        failed.append("training_summary_missing")
    if not canary:
        failed.append("phase4_training_canary_missing")
    return _dedupe(failed), _dedupe(warnings)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output-dir", default="")
    parser.add_argument("--training-summary", default="")
    parser.add_argument("--phase4-training-canary", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-return-code", type=int, default=0)
    parser.add_argument("--fail-on-reject", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    csv_path, json_path, row = write_phase4d_train_acceptance(
        args.output_dir,
        training_output_dir=args.training_output_dir,
        training_summary_path=args.training_summary,
        phase4_training_canary_path=args.phase4_training_canary,
        training_return_code=args.training_return_code)
    print("phase4d train acceptance: %s" % csv_path)
    print("phase4d train acceptance json: %s" % json_path)
    print("phase4d train status: %s eval_status=%s failed_checks=%s" % (
        row.get("phase4d_train_status"),
        row.get("eval_status"),
        row.get("failed_checks")))
    if args.fail_on_reject and _int_value(row.get("eval_allowed")) != 1:
        return 1
    return 0


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path or ".")))


def _resolve_path(path: Any, base_dir: str) -> str:
    value = os.path.expanduser(str(path or ""))
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(base_dir, value))


def _read_json_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as json_file:
            value = json.load(json_file)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _write_single_row_csv(path: str, row: Mapping[str, Any]) -> None:
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PHASE4D_TRAIN_ACCEPTANCE_FIELDS)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in PHASE4D_TRAIN_ACCEPTANCE_FIELDS})


def _write_json(path: str, value: Mapping[str, Any]) -> None:
    with open(path, "w") as json_file:
        json.dump(value, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _ordered_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: row.get(key, "") for key in PHASE4D_TRAIN_ACCEPTANCE_FIELDS}


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _int_first(*sources: Mapping[str, Any], keys: Sequence[str]) -> int:
    for key in keys:
        for source in sources:
            if key in source:
                return _int_value(source.get(key))
    return 0


def _float_first(*sources: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        for source in sources:
            if key in source:
                value = _float_value(source.get(key))
                return value if value is not None else ""
    return ""


def _int_value(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _float_value(value: Any) -> Optional[float]:
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nonempty(value: Any) -> str:
    return str(value or "").strip()


def _float_missing_or_below(value: Any, threshold: float) -> bool:
    parsed = _float_value(value)
    return parsed is None or parsed < threshold


def _float_missing_or_above(value: Any, threshold: float) -> bool:
    parsed = _float_value(value)
    return parsed is None or parsed > threshold


def _route_progress_fraction_threshold(total_timesteps_requested: int) -> float:
    """Minimum route progress for accepting a train artifact.

    Phase 4-D canary training may intentionally use short horizons. A fixed
    4096-step progress threshold incorrectly rejects successful 1024-step
    route-backed rollouts, so the gate scales conservatively by horizon bucket.
    """
    if total_timesteps_requested <= 0:
        return 0.45
    if total_timesteps_requested <= 1024:
        return 0.25
    if total_timesteps_requested <= 2048:
        return 0.35
    return 0.45


def _dedupe(values: Sequence[str]) -> list:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

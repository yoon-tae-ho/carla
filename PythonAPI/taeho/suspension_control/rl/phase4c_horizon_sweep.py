"""Phase 4-C horizon-sweep summary writer."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from suspension_control.rl.phase4_route_health import write_phase4_route_health


PROGRESS_MIN_BY_STEPS: Dict[int, float] = {
    512: 0.03,
    1024: 0.08,
    2048: 0.18,
    4096: 0.45,
}

SUMMARY_FIELDS: Tuple[str, ...] = (
    "phase4c_horizon_status",
    "horizon_steps",
    "run_dir",
    "failed_checks",
    "warnings",
    "phase4_route_health_status",
    "total_timesteps_requested",
    "total_timesteps_collected",
    "collected_steps_ok",
    "route_progress_monotonic_fraction_max",
    "progress_min_threshold",
    "progress_min_ok",
    "real_backend_used",
    "fake_backend_used",
    "carla_connected",
    "route_process_started",
    "hero_attached",
    "reward_row_ratio",
    "route_progress_available_ratio",
    "route_progress_monotonic_negative_rows",
    "hard_safety_gate_ratio",
    "low_speed_mask_ratio",
    "fallback_ratio",
    "effective_control_ratio",
    "mean_abs_action",
    "mean_abs_residual_damper",
    "observation_clip_ratio",
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

CHECK_FIELDS: Tuple[str, ...] = (
    "collected_steps_ok",
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
    "progress_min_ok",
)


def write_phase4c_horizon_sweep_summary(
    output_root: str,
    *,
    run_dirs: Optional[Sequence[str]] = None,
    force_route_health: bool = False,
    reference_diagnostic_rows: int = 0,
) -> Tuple[str, str, List[Dict[str, Any]], Dict[str, Any]]:
    """Aggregate per-run Phase 4 route-health outputs into a horizon summary."""

    output_root = _abs_path(output_root)
    os.makedirs(output_root, exist_ok=True)
    resolved_run_dirs = [
        _resolve_path(path, output_root)
        for path in (run_dirs or discover_horizon_run_dirs(output_root))
    ]

    rows = [
        phase4c_horizon_sweep_row(
            run_dir,
            force_route_health=force_route_health,
            reference_diagnostic_rows=reference_diagnostic_rows)
        for run_dir in resolved_run_dirs
    ]
    rows.sort(key=lambda row: (_int_value(row.get("horizon_steps")), str(row.get("run_dir"))))

    csv_path = os.path.join(output_root, "phase4c_horizon_sweep_summary.csv")
    json_path = os.path.join(output_root, "phase4c_horizon_sweep_summary.json")
    summary = {
        "phase4c_horizon_sweep_status": "pass"
        if rows and all(row.get("phase4c_horizon_status") == "pass" for row in rows)
        else "fail",
        "output_root": output_root,
        "row_count": len(rows),
        "summary_csv": csv_path,
        "summary_json": json_path,
        "rows": rows,
    }
    _write_csv_rows(csv_path, rows, SUMMARY_FIELDS)
    _write_json(json_path, summary)
    return csv_path, json_path, rows, summary


def phase4c_horizon_sweep_row(
    run_dir: str,
    *,
    force_route_health: bool = False,
    reference_diagnostic_rows: int = 0,
) -> Dict[str, Any]:
    """Compute one Phase 4-C horizon row from a single training output directory."""

    run_dir = _abs_path(run_dir)
    health_path = os.path.join(run_dir, "phase4_route_health.json")
    if force_route_health or not os.path.isfile(health_path):
        _, health_path, _, health, _ = write_phase4_route_health(
            run_dir,
            reference_diagnostic_rows=reference_diagnostic_rows)
    else:
        health = _read_json_dict(health_path)

    requested = _int_first(
        health,
        _read_json_dict(os.path.join(run_dir, "training_summary.json")),
        _read_json_dict(os.path.join(run_dir, "training_config.json")),
        keys=("total_timesteps_requested", "total_timesteps", "requested_steps"))
    horizon_steps = requested or _parse_horizon_steps(run_dir)
    collected = _int_value(health.get("total_timesteps_collected"))
    threshold = PROGRESS_MIN_BY_STEPS.get(horizon_steps, _default_progress_threshold(horizon_steps))
    progress = _float_value(health.get("route_progress_monotonic_fraction_max"))

    row: Dict[str, Any] = {
        "horizon_steps": horizon_steps,
        "run_dir": run_dir,
        "phase4_route_health_status": health.get("phase4_route_health_status", ""),
        "total_timesteps_requested": requested or horizon_steps,
        "total_timesteps_collected": collected,
        "route_progress_monotonic_fraction_max": "" if progress is None else progress,
        "progress_min_threshold": threshold,
        "real_backend_used": health.get("real_backend_used", ""),
        "fake_backend_used": health.get("fake_backend_used", ""),
        "carla_connected": health.get("carla_connected", ""),
        "route_process_started": health.get("route_process_started", ""),
        "hero_attached": health.get("hero_attached", ""),
        "reward_row_ratio": health.get("reward_row_ratio", ""),
        "route_progress_available_ratio": health.get("route_progress_available_ratio", ""),
        "route_progress_monotonic_negative_rows": health.get(
            "route_progress_monotonic_negative_rows", ""),
        "hard_safety_gate_ratio": health.get("hard_safety_gate_ratio", ""),
        "low_speed_mask_ratio": health.get("low_speed_mask_ratio", ""),
        "fallback_ratio": health.get("fallback_ratio", ""),
        "effective_control_ratio": health.get("effective_control_ratio", ""),
        "mean_abs_action": health.get("mean_abs_action", ""),
        "mean_abs_residual_damper": health.get("mean_abs_residual_damper", ""),
        "observation_clip_ratio": health.get("observation_clip_ratio", ""),
        "collision_count": health.get("collision_count", ""),
        "lane_invasion_count": health.get("lane_invasion_count", ""),
        "red_light_count": health.get("red_light_count", ""),
        "blocked_vehicle_count": health.get("blocked_vehicle_count", ""),
        "route_timeout_count": health.get("route_timeout_count", ""),
        "warnings": health.get("warnings", ""),
    }

    row.update(_acceptance_checks(row, health, progress, threshold))
    failed = [
        name[:-3] if name.endswith("_ok") else name
        for name in CHECK_FIELDS
        if _int_value(row.get(name)) != 1
    ]
    if not row["phase4_route_health_status"]:
        row["warnings"] = _append_reason(row.get("warnings"), "route_health_status_missing")
    elif row["phase4_route_health_status"] != "pass":
        row["warnings"] = _append_reason(row.get("warnings"), "route_health_status_failed")
    row["failed_checks"] = ";".join(failed)
    row["phase4c_horizon_status"] = "pass" if not failed else "fail"
    return _ordered_row(row, SUMMARY_FIELDS)


def discover_horizon_run_dirs(output_root: str) -> List[str]:
    """Return child directories that look like Phase 4-C horizon training runs."""

    output_root = _abs_path(output_root)
    if not os.path.isdir(output_root):
        return []
    result = []
    for name in os.listdir(output_root):
        path = os.path.join(output_root, name)
        if not os.path.isdir(path):
            continue
        if re.search(r"_train_\d+steps_", name) or os.path.isfile(
                os.path.join(path, "phase4_route_health.json")):
            result.append(path)
    return sorted(result)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--run-dirs",
        default="",
        help="Comma-separated horizon run directories. If omitted, scan output root.")
    parser.add_argument("--force-route-health", action="store_true")
    parser.add_argument("--reference-diagnostic-rows", type=int, default=0)
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Write reports but return success even if a horizon fails.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dirs = _split_csv(args.run_dirs) if args.run_dirs else None
    csv_path, json_path, rows, summary = write_phase4c_horizon_sweep_summary(
        args.output_root,
        run_dirs=run_dirs,
        force_route_health=args.force_route_health,
        reference_diagnostic_rows=args.reference_diagnostic_rows)
    print("phase4c horizon sweep summary: %s" % csv_path)
    print("phase4c horizon sweep json: %s" % json_path)
    print("phase4c horizon sweep status: %s" % summary["phase4c_horizon_sweep_status"])
    for row in rows:
        print("  %s steps: %s progress=%s collected=%s failed=%s" % (
            row.get("horizon_steps"),
            row.get("phase4c_horizon_status"),
            row.get("route_progress_monotonic_fraction_max"),
            row.get("total_timesteps_collected"),
            row.get("failed_checks")))
    if args.no_strict:
        return 0
    return 0 if summary["phase4c_horizon_sweep_status"] == "pass" else 1


def _acceptance_checks(
    row: Mapping[str, Any],
    health: Mapping[str, Any],
    progress: Optional[float],
    threshold: float,
) -> Dict[str, int]:
    requested = _int_value(row.get("total_timesteps_requested"))
    collected = _int_value(row.get("total_timesteps_collected"))
    return {
        "collected_steps_ok": int(collected >= min(requested or collected, 128)),
        "real_backend_ok": _check_from_health(
            health,
            "real_backend_ok",
            _int_value(row.get("real_backend_used")) == 1 and
            _int_value(row.get("fake_backend_used")) == 0),
        "hero_ok": _check_from_health(
            health,
            "hero_ok",
            _int_value(row.get("carla_connected")) == 1 and
            _int_value(row.get("route_process_started")) == 1 and
            _int_value(row.get("hero_attached")) == 1),
        "reward_ok": _check_from_health(
            health,
            "reward_ok",
            (_float_value(row.get("reward_row_ratio")) or 0.0) >= 0.95),
        "finite_reward_ok": _check_from_health(health, "finite_reward_ok", True),
        "progress_available_ok": _check_from_health(
            health,
            "progress_available_ok",
            _progress_available_ok(row)),
        "monotonic_progress_ok": _check_from_health(
            health,
            "monotonic_progress_ok",
            _int_value(row.get("route_progress_monotonic_negative_rows")) == 0),
        "hard_safety_ok": _check_from_health(
            health,
            "hard_safety_ok",
            (_float_value(row.get("hard_safety_gate_ratio")) or 0.0) <= 1.0e-12),
        "low_speed_mask_ok": _check_from_health(
            health,
            "low_speed_mask_ok",
            (_float_value(row.get("low_speed_mask_ratio")) or 0.0) <= 0.45),
        "fallback_ok": _check_from_health(
            health,
            "fallback_ok",
            (_float_value(row.get("fallback_ratio")) or 0.0) <= 0.45),
        "effective_control_ok": _check_from_health(
            health,
            "effective_control_ok",
            (_float_value(row.get("effective_control_ratio")) or 0.0) >= 0.50),
        "residual_ok": _check_from_health(
            health,
            "residual_ok",
            _between(row.get("mean_abs_residual_damper"), 0.001, 0.06)),
        "action_ok": _check_from_health(
            health,
            "action_ok",
            _between(row.get("mean_abs_action"), 0.01, 0.95)),
        "observation_clip_ok": _check_from_health(
            health,
            "observation_clip_ok",
            (_float_value(row.get("observation_clip_ratio")) or 0.0) <= 0.05),
        "infraction_proxy_ok": _check_from_health(
            health,
            "infraction_proxy_ok",
            all((_float_value(row.get(key)) or 0.0) <= 1.0e-12 for key in (
                "collision_count",
                "lane_invasion_count",
                "red_light_count",
                "blocked_vehicle_count",
                "route_timeout_count"))),
        "progress_min_ok": int(progress is not None and progress >= threshold),
    }


def _check_from_health(
    health: Mapping[str, Any],
    key: str,
    fallback: bool,
) -> int:
    if key in health and str(health.get(key, "")).strip() != "":
        return int(_int_value(health.get(key)) == 1)
    return int(bool(fallback))


def _progress_available_ok(row: Mapping[str, Any]) -> bool:
    value = _float_value(row.get("route_progress_available_ratio"))
    if value is None:
        return str(row.get("route_progress_monotonic_fraction_max", "")).strip() != ""
    return value >= 0.95


def _between(value: Any, lower: float, upper: float) -> bool:
    parsed = _float_value(value)
    return parsed is not None and lower <= parsed <= upper


def _default_progress_threshold(horizon_steps: int) -> float:
    if horizon_steps <= 0:
        return 0.0
    if horizon_steps < 4096:
        return min(0.45, 0.18 * horizon_steps / 2048.0)
    return min(0.98, 0.45 * horizon_steps / 4096.0)


def _parse_horizon_steps(path: str) -> int:
    match = re.search(r"_train_(\d+)steps_", os.path.basename(path))
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)steps", os.path.basename(path))
    if match:
        return int(match.group(1))
    return 0


def _int_first(*sources: Mapping[str, Any], keys: Sequence[str]) -> int:
    for source in sources:
        for key in keys:
            if key in source:
                return _int_value(source.get(key))
    return 0


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


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _append_reason(existing: Any, reason: str) -> str:
    reasons = [item for item in str(existing or "").split(";") if item]
    if reason not in reasons:
        reasons.append(reason)
    return ";".join(reasons)


def _read_json_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as json_file:
            data = json.load(json_file)
        return dict(data) if isinstance(data, Mapping) else {}
    except Exception:
        return {}


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


def _abs_path(path: Any) -> str:
    return os.path.abspath(os.path.expanduser(str(path or "")))


def _resolve_path(path: Any, output_root: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return ""
    expanded = os.path.expanduser(raw)
    candidates = [expanded if os.path.isabs(expanded) else os.path.join(output_root, expanded)]
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

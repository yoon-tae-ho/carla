#!/usr/bin/env python3

"""Summarize and gate Step 07 LEAD target-speed suspension outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SMOKE_SCENARIOS = (
    "LEAD_stock",
    "LEAD_export_only",
    "LEAD_identity",
    "LEAD_identity_jsonl",
    "LEAD_constant_damper_1.02",
    "LEAD_constant_damper_1.03",
    "LEAD_pard_v2_shadow",
    "LEAD_pard_v2_active_ultra_safe",
)
FULL_SCENARIOS = (
    "LEAD_stock",
    "LEAD_identity",
    "LEAD_export_only",
    "LEAD_identity_jsonl",
    "LEAD_skyhook",
    "LEAD_constant_damper_1.02",
    "LEAD_constant_damper_1.03",
    "LEAD_target_speed_schedule_v0",
    "LEAD_pard_v2_shadow",
    "LEAD_pard_v2_active_ultra_safe",
)
SIDECAR_SCENARIOS = {
    "LEAD_stock_sidecar",
    "LEAD_identity",
    "LEAD_identity_jsonl",
    "LEAD_skyhook",
    "LEAD_skyhook_roll",
    "LEAD_planning_aware",
    "LEAD_constant_damper_1.02",
    "LEAD_constant_damper_1.03",
    "LEAD_pard_v2_shadow",
    "LEAD_pard_v2_active_ultra_safe",
    "LEAD_pard_v2_active_safe_1p06",
    "LEAD_pard_v2_active_aggressive_0p75_1p25",
    "LEAD_target_speed_schedule_v0",
    "LEAD_target_speed_schedule_shadow",
    "LEAD_target_speed_schedule_v0_safe",
}
PROFILE_ONLY_SIDECAR_SCENARIOS = {
    "LEAD_stock_sidecar",
}
JSONL_SCENARIOS = {
    "LEAD_export_only",
    "LEAD_identity_jsonl",
    "LEAD_planning_aware",
    "LEAD_pard_v2_shadow",
    "LEAD_pard_v2_active_ultra_safe",
    "LEAD_pard_v2_active_safe_1p06",
    "LEAD_pard_v2_active_aggressive_0p75_1p25",
    "LEAD_target_speed_schedule_v0",
    "LEAD_target_speed_schedule_shadow",
    "LEAD_target_speed_schedule_v0_safe",
}
JSONL_SIDECAR_SCENARIOS = {
    "LEAD_identity_jsonl",
    "LEAD_planning_aware",
    "LEAD_pard_v2_shadow",
    "LEAD_pard_v2_active_ultra_safe",
    "LEAD_pard_v2_active_safe_1p06",
    "LEAD_pard_v2_active_aggressive_0p75_1p25",
    "LEAD_target_speed_schedule_v0",
    "LEAD_target_speed_schedule_shadow",
    "LEAD_target_speed_schedule_v0_safe",
}
PARD_V2_SCENARIOS = {
    "LEAD_pard_v2_shadow",
    "LEAD_pard_v2_active_ultra_safe",
    "LEAD_pard_v2_active_safe_1p06",
    "LEAD_pard_v2_active_aggressive_0p75_1p25",
}
PARD_V2_SHADOW_SCENARIOS = {
    "LEAD_pard_v2_shadow",
}
PARD_V2_ACTIVE_SCENARIOS = {
    "LEAD_pard_v2_active_ultra_safe",
    "LEAD_pard_v2_active_safe_1p06",
    "LEAD_pard_v2_active_aggressive_0p75_1p25",
}
PARD_DAMPER_MIN_BY_SCENARIO = {
    "LEAD_pard_v2_active_ultra_safe": 1.0,
    "LEAD_pard_v2_active_safe_1p06": 1.0,
    "LEAD_pard_v2_active_aggressive_0p75_1p25": 0.75,
}
PARD_DAMPER_MAX_BY_SCENARIO = {
    "LEAD_pard_v2_active_ultra_safe": 1.035,
    "LEAD_pard_v2_active_safe_1p06": 1.06,
    "LEAD_pard_v2_active_aggressive_0p75_1p25": 1.25,
}
CONSTANT_DAMPER_BY_SCENARIO = {
    "LEAD_constant_damper_1.02": 1.02,
    "LEAD_constant_damper_1.03": 1.03,
}
CONSTANT_DAMPING_SCENARIOS = set(CONSTANT_DAMPER_BY_SCENARIO)
TARGET_SPEED_SCENARIOS = {
    "LEAD_target_speed_schedule_v0",
    "LEAD_target_speed_schedule_shadow",
    "LEAD_target_speed_schedule_v0_safe",
}
TSS_DAMPER_MAX_BY_SCENARIO = {
    "LEAD_target_speed_schedule_v0": 1.05,
    "LEAD_target_speed_schedule_shadow": 1.0,
    "LEAD_target_speed_schedule_v0_safe": 1.01,
}
MAJOR_INFRACTION_FIELDS = (
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
    "scenario_timeouts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        help="Step 07 output root")
    parser.add_argument(
        "--mode",
        choices=("auto", "smoke", "full"),
        default="auto",
        help="expected scenario set (default: auto)")
    parser.add_argument(
        "--seeds",
        default="",
        help="comma-separated expected seeds; defaults from mode or runner plan")
    parser.add_argument(
        "--scenarios",
        default="",
        help="comma-separated expected scenarios; defaults from mode or runner plan")
    parser.add_argument(
        "--sim-root",
        default=os.environ.get("SIM_ROOT", os.path.expanduser("~/sim")),
        help="host sim root used to translate /workspace paths")
    parser.add_argument(
        "--score-tolerance",
        default=2.0,
        type=float,
        help="allowed score_composed drop vs baseline (default: 2.0)")
    parser.add_argument(
        "--route-score-tolerance",
        default=1.0,
        type=float,
        help="allowed score_route drop vs baseline (default: 1.0)")
    parser.add_argument(
        "--planning-age-max",
        default=5.0,
        type=float,
        help="max accepted planning age in frames for JSONL sidecar runs")
    return parser.parse_args()


def sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", value)


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def host_path(path: Any, sim_root: str) -> str:
    if not path:
        return ""
    text = str(path)
    if text == "/workspace":
        return sim_root
    if text.startswith("/workspace/"):
        return os.path.join(sim_root, text[len("/workspace/"):])
    return os.path.expanduser(text)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.isfile(path):
        return []
    with open(path, newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv_rows(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ("status",)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def read_json(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as json_file:
        return json.load(json_file)


def float_or_none(value: Any) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def nonfinite_numeric_literal_count(row: Mapping[str, Any]) -> int:
    count = 0
    for value in row.values():
        if value in ("", None):
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            count += 1
    return count


def finite_values(row: Mapping[str, Any], fields: Sequence[str]) -> List[float]:
    values: List[float] = []
    for field in fields:
        number = float_or_none(row.get(field))
        if number is not None:
            values.append(number)
    return values


def first_available_finite_values(
    row: Mapping[str, Any],
    field_groups: Sequence[Sequence[str]],
) -> List[float]:
    for fields in field_groups:
        values = finite_values(row, fields)
        if values:
            return values
    return []


def int_or_zero(value: Any) -> int:
    number = float_or_none(value)
    if number is None:
        return 0
    return int(round(number))


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        return "%0.9g" % value
    return value


def mean_or_empty(values: Sequence[float]) -> Any:
    return statistics.fmean(values) if values else ""


def min_or_empty(values: Sequence[float]) -> Any:
    return min(values) if values else ""


def max_or_empty(values: Sequence[float]) -> Any:
    return max(values) if values else ""


def infraction_count(value: Any) -> int:
    if value in ("", None, "[]", "{}", "0", "0.0"):
        return 0
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    number = float_or_none(value)
    if number is not None:
        return 0 if abs(number) <= 1.0e-12 else int(math.ceil(abs(number)))
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text.replace("'", '"'))
            return infraction_count(parsed)
        except Exception:
            return 1
    return 1


def extract_checkpoint_metrics(path: str) -> Dict[str, Any]:
    data = read_json(path)
    if not data:
        return {"checkpoint_exists": 0}
    checkpoint = data.get("_checkpoint", {})
    global_record = checkpoint.get("global_record", {})
    scores = global_record.get("scores_mean") or global_record.get("scores") or {}
    infractions = global_record.get("infractions", {})
    meta = global_record.get("meta", {})
    records = checkpoint.get("records", [])
    sensors = data.get("sensors") or []
    exceptions = meta.get("exceptions") or []

    row: Dict[str, Any] = {
        "checkpoint_exists": 1,
        "entry_status": data.get("entry_status", ""),
        "eligible": data.get("eligible", ""),
        "sensors_count": len(sensors) if isinstance(sensors, list) else "",
        "checkpoint_exceptions": json.dumps(exceptions, sort_keys=True),
        "progress": "%s/%s" % tuple(checkpoint.get("progress", ["", ""])),
        "global_status": global_record.get("status", ""),
        "score_composed": scores.get("score_composed", ""),
        "score_route": scores.get("score_route", ""),
        "score_penalty": scores.get("score_penalty", ""),
        "duration_game": meta.get("duration_game", ""),
        "duration_system": meta.get("duration_system", ""),
        "route_statuses": ";".join(
            "%s:%s" % (record.get("route_id", ""), record.get("status", ""))
            for record in records),
        "route_scores": ";".join(
            "%s:%s" % (
                record.get("route_id", ""),
                record.get("scores", {}).get("score_composed", ""))
            for record in records),
    }
    for field in MAJOR_INFRACTION_FIELDS + ("min_speed_infractions",):
        row[field] = infractions.get(field, "")
    row["major_infraction_count"] = sum(
        infraction_count(row.get(field)) for field in MAJOR_INFRACTION_FIELDS)
    return row


def count_jsonl_rows(path: str) -> int:
    if not path or not os.path.isfile(path):
        return 0
    rows = 0
    with open(path) as jsonl_file:
        for line in jsonl_file:
            if line.strip():
                rows += 1
    return rows


def read_text_tail(path: str, max_bytes: int = 200000) -> str:
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as text_file:
        text_file.seek(0, os.SEEK_END)
        size = text_file.tell()
        text_file.seek(max(0, size - max_bytes), os.SEEK_SET)
        return text_file.read().decode("utf-8", errors="replace")


def parse_scale_mismatch(message: str) -> Optional[float]:
    match = re.search(
        r"scale mismatch: expected\s+([-+0-9.eE]+)\s+got\s+([-+0-9.eE]+)",
        message)
    if not match:
        return None
    expected = float_or_none(match.group(1))
    actual = float_or_none(match.group(2))
    if expected is None or actual is None:
        return math.inf
    return abs(actual - expected)


def summarize_verification_events(
    path: str,
    actor_removed_is_warning: bool = False,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "verify_warning_count": 0,
        "verify_hard_error_count": 0,
        "verify_unclassified_runtime_errors": 0,
        "sidecar_actor_removed_warning_count": 0,
        "sidecar_runtime_hard_error_count": 0,
        "verify_max_scale_error": "",
        "verify_warning_messages": "",
        "verify_hard_error_messages": "",
        "sidecar_actor_removed_warning_messages": "",
        "sidecar_runtime_hard_error_messages": "",
    }
    if not path or not os.path.isfile(path):
        return summary

    scale_errors: List[float] = []
    warning_messages = set()
    hard_messages = set()
    actor_removed_messages = set()
    runtime_hard_messages = set()
    with open(path, newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            if row.get("event", "") != "runtime_error":
                continue
            message = row.get("message", "")
            lower = message.lower()
            if "scale mismatch" in lower:
                error = parse_scale_mismatch(message)
                if error is None or not math.isfinite(error):
                    summary["verify_hard_error_count"] += 1
                    hard_messages.add(message)
                else:
                    scale_errors.append(error)
                    if error <= 0.002:
                        summary["verify_warning_count"] += 1
                        warning_messages.add(message)
                    elif error > 0.005:
                        summary["verify_hard_error_count"] += 1
                        hard_messages.add(message)
                    else:
                        summary["verify_warning_count"] += 1
                        warning_messages.add(message)
                continue
            if "non-finite" in lower or "nonfinite" in lower or "bound" in lower:
                summary["verify_hard_error_count"] += 1
                hard_messages.add(message)
                summary["sidecar_runtime_hard_error_count"] += 1
                runtime_hard_messages.add(message)
            elif (
                    "actor could not be found in the registry" in lower and
                    actor_removed_is_warning):
                summary["sidecar_actor_removed_warning_count"] += 1
                actor_removed_messages.add(message)
            else:
                summary["verify_unclassified_runtime_errors"] += 1
                summary["sidecar_runtime_hard_error_count"] += 1
                runtime_hard_messages.add(message)

    summary["verify_max_scale_error"] = max_or_empty(scale_errors)
    summary["verify_warning_messages"] = " | ".join(sorted(warning_messages)[:5])
    summary["verify_hard_error_messages"] = " | ".join(sorted(hard_messages)[:5])
    summary["sidecar_actor_removed_warning_messages"] = " | ".join(
        sorted(actor_removed_messages)[:5])
    summary["sidecar_runtime_hard_error_messages"] = " | ".join(
        sorted(runtime_hard_messages)[:5])
    return summary


def detect_agent_setup_failure(row: Mapping[str, Any]) -> bool:
    status_text = " ".join(str(row.get(field, "")) for field in (
        "entry_status",
        "global_status",
        "route_statuses",
        "checkpoint_exceptions",
    ))
    contains_setup_failure = (
        "Agent couldn't be set up" in status_text or
        int_or_zero(row.get("route_stdout_agent_setup_failure")) != 0)
    duration_game = float_or_none(row.get("duration_game"))
    sensors_count = int_or_zero(row.get("sensors_count"))
    checkpoint_exists = int_or_zero(row.get("checkpoint_exists")) != 0
    duration_zero = (
        checkpoint_exists and
        duration_game is not None and
        abs(duration_game) <= 1.0e-9)
    sensors_empty = checkpoint_exists and sensors_count == 0
    return bool(contains_setup_failure or duration_zero or sensors_empty)


def route_completed_with_game_time(row: Mapping[str, Any]) -> bool:
    status = str(row.get("global_status", ""))
    duration_game = float_or_none(row.get("duration_game"))
    return bool(
        int_or_zero(row.get("checkpoint_exists")) != 0 and
        status in ("Perfect", "Completed") and
        duration_game is not None and
        duration_game > 0.0)


def first_existing(paths: Iterable[str]) -> str:
    for path in paths:
        if path and os.path.isfile(path):
            return path
    return ""


def path_from_row(row: Mapping[str, Any], key: str, sim_root: str) -> str:
    return host_path(row.get(key, ""), sim_root)


def summarize_diagnostics(
    path: str,
    scenario_key: str,
    planning_age_limit: float,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "controller_diagnostics_exists": int(bool(path and os.path.isfile(path))),
        "controller_diagnostic_rows": 0,
        "planning_available_rows": 0,
        "planning_sources": "",
        "planning_age_min": "",
        "planning_age_mean": "",
        "planning_age_max": "",
        "planning_jsonl_malformed_lines_max": "",
        "planning_jsonl_rejected_messages_max": "",
        "planning_jsonl_read_errors_max": "",
        "diagnostic_nonfinite_values": 0,
        "damper_nonfinite_violations": 0,
        "damper_bound_violations": 0,
        "damper_min": "",
        "damper_mean": "",
        "damper_max": "",
        "spring_identity_violations": 0,
        "tss_rows": 0,
        "tss_valid_rows": 0,
        "tss_fallback_reasons": "",
        "tss_required_field_missing": 0,
        "tss_forbidden_usage_rows": 0,
        "tss_shadow_rows": 0,
        "tss_damper_min": "",
        "tss_damper_mean": "",
        "tss_damper_max": "",
        "pard_rows": 0,
        "pard_required_field_missing": 0,
        "pard_fallback_rows": 0,
        "pard_fallback_ratio": "",
        "pard_shadow_rows": 0,
        "pard_active_rows": 0,
        "pard_shadow_actual_identity_violations": 0,
        "pard_would_command_rows": 0,
        "pard_damper_bound_violations": 0,
        "pard_damper_min": "",
        "pard_damper_mean": "",
        "pard_damper_max": "",
        "pard_would_damper_min": "",
        "pard_would_damper_mean": "",
        "pard_would_damper_max": "",
        "planning_age_gate_ok": "",
    }
    if not path or not os.path.isfile(path):
        return summary

    rows = read_csv_rows(path)
    summary["controller_diagnostic_rows"] = len(rows)
    planning_sources = set()
    planning_ages: List[float] = []
    malformed: List[float] = []
    rejected: List[float] = []
    read_errors: List[float] = []
    dampers: List[float] = []
    springs: List[float] = []
    tss_dampers: List[float] = []
    pard_dampers: List[float] = []
    pard_would_dampers: List[float] = []
    tss_fallback_reasons = set()
    tss_required_fields = (
        "tss_target_speed_raw",
        "tss_target_speed_valid",
        "tss_damper_scale_computed",
        "tss_damper_scale_applied",
        "tss_shadow_mode",
        "tss_used_curvature",
        "tss_used_trajectory",
        "tss_used_control",
    )
    pard_required_fields = (
        "fallback_active",
        "planning_age_frames",
        "shadow_mode",
        "uniform_damper_cmd",
        "uniform_damper_would",
        "spring_FL",
        "spring_FR",
        "spring_RL",
        "spring_RR",
        "damper_FL",
        "damper_FR",
        "damper_RL",
        "damper_RR",
    )
    pard_actual_damper_groups = (
        ("damper_FL", "damper_FR", "damper_RL", "damper_RR"),
        ("uniform_damper_cmd",),
        ("damper_scale",),
    )
    pard_would_damper_groups = (
        ("uniform_damper_would",),
        ("front_damper_would", "rear_damper_would"),
    )
    generic_damper_groups = (
        ("damper_scale",),
        ("damper_FL", "damper_FR", "damper_RL", "damper_RR"),
        ("uniform_damper_cmd",),
    )
    spring_fields = (
        "spring_scale",
        "spring_FL",
        "spring_FR",
        "spring_RL",
        "spring_RR",
    )

    for row in rows:
        summary["diagnostic_nonfinite_values"] += nonfinite_numeric_literal_count(row)

        planning_available = float_or_none(row.get("planning_available"))
        if planning_available is not None and planning_available > 0.0:
            summary["planning_available_rows"] += 1
        source = row.get("planning_source", "")
        if source:
            planning_sources.add(source)
        for field, target in (
                ("planning_age_frames", planning_ages),
                ("planning_jsonl_malformed_lines", malformed),
                ("planning_jsonl_rejected_messages", rejected),
                ("planning_jsonl_read_errors", read_errors)):
            number = float_or_none(row.get(field))
            if number is not None:
                target.append(number)

        tss_values = finite_values(row, ("tss_damper_scale_applied",))
        if tss_values:
            summary["tss_rows"] += 1
            tss_dampers.extend(tss_values)

        if scenario_key in PARD_V2_SCENARIOS:
            actual_dampers = first_available_finite_values(
                row,
                pard_actual_damper_groups)
        elif tss_values:
            actual_dampers = tss_values
        else:
            actual_dampers = first_available_finite_values(
                row,
                generic_damper_groups)
        damper = actual_dampers[0] if actual_dampers else None
        if not actual_dampers:
            summary["damper_nonfinite_violations"] += 1
        else:
            dampers.extend(actual_dampers)

        row_springs = finite_values(row, spring_fields)
        springs.extend(row_springs)
        for spring in row_springs:
            if not math.isclose(
                    spring,
                    1.0,
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-6):
                summary["spring_identity_violations"] += 1

        if scenario_key in TARGET_SPEED_SCENARIOS:
            if any(field not in row for field in tss_required_fields):
                summary["tss_required_field_missing"] += 1
            damper_max = TSS_DAMPER_MAX_BY_SCENARIO.get(scenario_key, 1.05)
            if damper is not None and not (
                    1.0 - 1.0e-9 <= damper <= damper_max + 1.0e-9):
                summary["damper_bound_violations"] += 1
            if int_or_zero(row.get("tss_target_speed_valid")):
                summary["tss_valid_rows"] += 1
            if int_or_zero(row.get("tss_shadow_mode")):
                summary["tss_shadow_rows"] += 1
            reason = row.get("tss_fallback_reason", "")
            if reason:
                tss_fallback_reasons.add(reason)
            forbidden = (
                int_or_zero(row.get("tss_used_curvature")) or
                int_or_zero(row.get("tss_used_trajectory")) or
                int_or_zero(row.get("tss_used_control")))
            if forbidden:
                summary["tss_forbidden_usage_rows"] += 1
        elif scenario_key in PARD_V2_SCENARIOS:
            summary["pard_rows"] += 1
            if any(field not in row for field in pard_required_fields):
                summary["pard_required_field_missing"] += 1
            if int_or_zero(row.get("fallback_active")):
                summary["pard_fallback_rows"] += 1
            if int_or_zero(row.get("shadow_mode")):
                summary["pard_shadow_rows"] += 1
            if scenario_key in PARD_V2_ACTIVE_SCENARIOS:
                summary["pard_active_rows"] += 1
            pard_dampers.extend(actual_dampers)
            would_dampers = first_available_finite_values(
                row,
                pard_would_damper_groups)
            pard_would_dampers.extend(would_dampers)
            if any(abs(value - 1.0) > 1.0e-6 for value in would_dampers):
                summary["pard_would_command_rows"] += 1
            if scenario_key in PARD_V2_SHADOW_SCENARIOS:
                if actual_dampers and any(
                        not math.isclose(
                            value,
                            1.0,
                            rel_tol=1.0e-6,
                            abs_tol=1.0e-6)
                        for value in actual_dampers):
                    summary["pard_shadow_actual_identity_violations"] += 1
            if scenario_key in PARD_V2_ACTIVE_SCENARIOS:
                damper_min = PARD_DAMPER_MIN_BY_SCENARIO.get(scenario_key, 1.0)
                damper_max = PARD_DAMPER_MAX_BY_SCENARIO.get(scenario_key)
                for value in actual_dampers:
                    if value < damper_min - 1.0e-6:
                        summary["pard_damper_bound_violations"] += 1
                        summary["damper_bound_violations"] += 1
                    if damper_max is not None and value > damper_max + 1.0e-6:
                        summary["pard_damper_bound_violations"] += 1
                        summary["damper_bound_violations"] += 1
        elif scenario_key in CONSTANT_DAMPING_SCENARIOS:
            expected = CONSTANT_DAMPER_BY_SCENARIO[scenario_key]
            for value in actual_dampers:
                if not math.isclose(
                        value,
                        expected,
                        rel_tol=1.0e-6,
                        abs_tol=1.0e-6):
                    summary["damper_bound_violations"] += 1

    summary["planning_sources"] = ";".join(sorted(planning_sources))
    summary["planning_age_min"] = min_or_empty(planning_ages)
    summary["planning_age_mean"] = mean_or_empty(planning_ages)
    summary["planning_age_max"] = max_or_empty(planning_ages)
    summary["planning_jsonl_malformed_lines_max"] = max_or_empty(malformed)
    summary["planning_jsonl_rejected_messages_max"] = max_or_empty(rejected)
    summary["planning_jsonl_read_errors_max"] = max_or_empty(read_errors)
    summary["damper_min"] = min_or_empty(dampers)
    summary["damper_mean"] = mean_or_empty(dampers)
    summary["damper_max"] = max_or_empty(dampers)
    summary["tss_fallback_reasons"] = ";".join(sorted(tss_fallback_reasons))
    summary["tss_damper_min"] = min_or_empty(tss_dampers)
    summary["tss_damper_mean"] = mean_or_empty(tss_dampers)
    summary["tss_damper_max"] = max_or_empty(tss_dampers)
    if summary["pard_rows"]:
        summary["pard_fallback_ratio"] = (
            summary["pard_fallback_rows"] / summary["pard_rows"])
    summary["pard_damper_min"] = min_or_empty(pard_dampers)
    summary["pard_damper_mean"] = mean_or_empty(pard_dampers)
    summary["pard_damper_max"] = max_or_empty(pard_dampers)
    summary["pard_would_damper_min"] = min_or_empty(pard_would_dampers)
    summary["pard_would_damper_mean"] = mean_or_empty(pard_would_dampers)
    summary["pard_would_damper_max"] = max_or_empty(pard_would_dampers)
    if planning_ages:
        summary["planning_age_gate_ok"] = int(max(planning_ages) <= planning_age_limit)
    return summary


def read_runner_plan(root: str) -> List[Dict[str, str]]:
    return read_csv_rows(os.path.join(root, "step07_runner_plan.csv"))


def read_runner_results(root: str) -> Dict[Tuple[str, str], str]:
    rows = read_csv_rows(os.path.join(root, "step07_runner_results.csv"))
    results: Dict[Tuple[str, str], str] = {}
    for row in rows:
        results[(row.get("seed", ""), row.get("scenario_key", ""))] = row.get(
            "return_code", "")
    return results


def expected_plan(args: argparse.Namespace, root: str) -> List[Dict[str, str]]:
    plan = read_runner_plan(root)
    if plan:
        requested_seeds = set(parse_csv_list(args.seeds))
        requested_scenarios = set(parse_csv_list(args.scenarios))
        if requested_seeds or requested_scenarios:
            return [
                row for row in plan
                if (
                    not requested_seeds or
                    row.get("seed", "") in requested_seeds
                ) and (
                    not requested_scenarios or
                    row.get("scenario_key", "") in requested_scenarios
                )
            ]
        return plan

    if args.mode == "full":
        scenarios = FULL_SCENARIOS
        default_seeds = "111,112,113"
    else:
        scenarios = SMOKE_SCENARIOS
        default_seeds = "111"
    if args.scenarios:
        scenarios = tuple(parse_csv_list(args.scenarios))
    seeds = parse_csv_list(args.seeds or default_seeds)
    rows = []
    for seed in seeds:
        for scenario in scenarios:
            run_dir = os.path.join(root, "seed%s_%s" % (seed, sanitize_label(scenario)))
            planning = os.path.join(run_dir, "planning_preview.jsonl") if scenario in JSONL_SCENARIOS else ""
            rows.append({
                "mode": args.mode if args.mode != "auto" else "smoke",
                "seed": seed,
                "scenario_key": scenario,
                "run_kind": "sidecar" if scenario in SIDECAR_SCENARIOS else "direct",
                "run_dir_host": run_dir,
                "run_dir_container": "",
                "planning_jsonl_host": planning,
                "planning_jsonl_container": "",
            })
    return rows


def summarize_direct_run(
    plan_row: Mapping[str, str],
    sim_root: str,
) -> Dict[str, Any]:
    run_dir = host_path(plan_row.get("run_dir_host", ""), sim_root)
    result_json = first_existing((
        os.path.join(run_dir, "lead", "checkpoint_endpoint.json"),
        os.path.join(run_dir, "checkpoint_endpoint.json"),
    ))
    row: Dict[str, Any] = {
        "run_dir": run_dir,
        "result_json": result_json,
        "controller_diagnostics_csv": "",
        "metrics_by_episode_csv": "",
        "sidecar_events_csv": "",
        "route_stdout_log": first_existing((
            os.path.join(run_dir, "docker_stdout.log"),
            os.path.join(run_dir, "lead", "route_stdout.log"),
        )),
    }
    row.update(extract_checkpoint_metrics(result_json))
    row["route_stdout_agent_setup_failure"] = int(
        "Agent couldn't be set up" in read_text_tail(row["route_stdout_log"]))
    return row


def summarize_sidecar_run(
    plan_row: Mapping[str, str],
    sim_root: str,
    planning_age_limit: float,
) -> Dict[str, Any]:
    run_dir = host_path(plan_row.get("run_dir_host", ""), sim_root)
    suite_dir = os.path.join(run_dir, "suite")
    suite_rows = read_csv_rows(os.path.join(suite_dir, "suite_summary.csv"))
    suite_row: Dict[str, Any] = dict(suite_rows[0]) if suite_rows else {}

    if not suite_row:
        summary_candidates = []
        for root, _, files in os.walk(suite_dir):
            if "summary.json" in files:
                summary_candidates.append(os.path.join(root, "summary.json"))
        if summary_candidates:
            suite_row = read_json(sorted(summary_candidates)[0])

    result_json = first_existing((
        path_from_row(suite_row, "result_json", sim_root),
        *(
            os.path.join(root, "checkpoint_endpoint.json")
            for root, _, files in os.walk(suite_dir)
            if "checkpoint_endpoint.json" in files
        ),
    ))
    diagnostics_csv = first_existing((
        path_from_row(suite_row, "controller_diagnostics_csv", sim_root),
        *(
            os.path.join(root, "controller_diagnostics.csv")
            for root, _, files in os.walk(suite_dir)
            if "controller_diagnostics.csv" in files
        ),
    ))
    metrics_csv = first_existing((
        path_from_row(suite_row, "metrics_by_episode_csv", sim_root),
        *(
            os.path.join(root, "metrics_by_episode.csv")
            for root, _, files in os.walk(suite_dir)
            if "metrics_by_episode.csv" in files
        ),
    ))
    events_csv = first_existing((
        path_from_row(suite_row, "sidecar_events_csv", sim_root),
        *(
            os.path.join(root, "sidecar_events.csv")
            for root, _, files in os.walk(suite_dir)
            if "sidecar_events.csv" in files
        ),
    ))
    route_stdout_log = first_existing((
        path_from_row(suite_row, "route_stdout_log", sim_root),
        *(
            os.path.join(root, "route_stdout.log")
            for root, _, files in os.walk(suite_dir)
            if "route_stdout.log" in files
        ),
    ))

    row: Dict[str, Any] = dict(suite_row)
    row.update({
        "run_dir": run_dir,
        "suite_dir": suite_dir,
        "result_json": result_json,
        "controller_diagnostics_csv": diagnostics_csv,
        "metrics_by_episode_csv": metrics_csv,
        "sidecar_events_csv": events_csv,
        "route_stdout_log": route_stdout_log,
        "route_stdout_agent_setup_failure": int(
            "Agent couldn't be set up" in read_text_tail(route_stdout_log)),
    })
    checkpoint_metrics = extract_checkpoint_metrics(result_json)
    for key, value in checkpoint_metrics.items():
        row.setdefault(key, value)
    row.update(summarize_diagnostics(
        diagnostics_csv,
        plan_row.get("scenario_key", ""),
        planning_age_limit))
    row.update(summarize_verification_events(
        events_csv,
        actor_removed_is_warning=route_completed_with_game_time(row)))
    return row


def build_summary_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = os.path.abspath(os.path.expanduser(args.root))
    os.makedirs(root, exist_ok=True)
    plan = expected_plan(args, root)
    runner_results = read_runner_results(root)
    rows: List[Dict[str, Any]] = []
    for plan_row in plan:
        scenario = plan_row.get("scenario_key", "")
        seed = plan_row.get("seed", "")
        run_kind = plan_row.get("run_kind", "")
        base: Dict[str, Any] = {
            "mode": plan_row.get("mode", args.mode),
            "seed": seed,
            "scenario_key": scenario,
            "run_kind": run_kind,
            "runner_return_code": runner_results.get((seed, scenario), ""),
            "planning_jsonl": host_path(plan_row.get("planning_jsonl_host", ""), args.sim_root),
        }
        base["planning_jsonl_exists"] = int(
            bool(base["planning_jsonl"] and os.path.isfile(base["planning_jsonl"])))
        base["planning_jsonl_rows"] = count_jsonl_rows(base["planning_jsonl"])
        if run_kind == "sidecar" or scenario in SIDECAR_SCENARIOS:
            detail = summarize_sidecar_run(plan_row, args.sim_root, args.planning_age_max)
        else:
            detail = summarize_direct_run(plan_row, args.sim_root)
        base.update(detail)
        if not base.get("result_json"):
            base.setdefault("checkpoint_exists", 0)
        base["major_infraction_count"] = sum(
            infraction_count(base.get(field)) for field in MAJOR_INFRACTION_FIELDS)
        base["invalid_run"] = int(detect_agent_setup_failure(base))
        base["run_valid"] = int(not base["invalid_run"])
        rows.append(base)
    return rows


def evaluate_gates(
    rows: Sequence[Dict[str, Any]],
    score_tolerance: float,
    route_score_tolerance: float,
    planning_age_limit: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    def accepted_route_status(row: Mapping[str, Any]) -> Tuple[bool, List[str]]:
        status = str(row.get("global_status", ""))
        if status in ("", "Perfect"):
            return True, []
        if status == "Completed":
            route_score = float_or_none(row.get("score_route"))
            composed = float_or_none(row.get("score_composed"))
            major_count = int_or_zero(row.get("major_infraction_count"))
            if (
                    route_score is not None and route_score >= 99.0 and
                    composed is not None and composed >= 98.0 and
                    major_count == 0):
                warnings = ["global_completed_minor_penalty"]
                if infraction_count(row.get("min_speed_infractions")):
                    warnings.append("minor_min_speed_only_penalty")
                return True, warnings
        return False, []

    def row_valid_for_pairwise(row: Mapping[str, Any]) -> bool:
        if int_or_zero(row.get("invalid_run")):
            return False
        if not int_or_zero(row.get("checkpoint_exists")):
            return False
        if int_or_zero(row.get("major_infraction_count")) != 0:
            return False
        ok, _ = accepted_route_status(row)
        if not ok:
            return False
        scenario = str(row.get("scenario_key", ""))
        if scenario in JSONL_SCENARIOS and int_or_zero(row.get("planning_jsonl_rows")) == 0:
            return False
        if scenario in JSONL_SIDECAR_SCENARIOS:
            age_max = float_or_none(row.get("planning_age_max"))
            if age_max is None or age_max > planning_age_limit:
                return False
            if int_or_zero(row.get("planning_available_rows")) == 0:
                return False
        if int_or_zero(row.get("sidecar_fatal_errors")):
            return False
        if int_or_zero(row.get("verify_hard_error_count")):
            return False
        if int_or_zero(row.get("sidecar_runtime_hard_error_count")):
            return False
        if scenario in PARD_V2_SCENARIOS:
            if int_or_zero(row.get("diagnostic_nonfinite_values")):
                return False
            if int_or_zero(row.get("pard_damper_bound_violations")):
                return False
            if int_or_zero(row.get("pard_required_field_missing")):
                return False
            if (
                    scenario in PARD_V2_SHADOW_SCENARIOS and
                    int_or_zero(row.get(
                        "pard_shadow_actual_identity_violations"))):
                return False
        return True

    by_seed_scenario = {
        (str(row.get("seed", "")), str(row.get("scenario_key", ""))): row
        for row in rows
    }
    baseline_by_seed: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        seed = str(row.get("seed", ""))
        for scenario in (
                "LEAD_identity_jsonl",
                "LEAD_identity",
                "LEAD_stock_sidecar",
                "LEAD_stock"):
            baseline = by_seed_scenario.get((seed, scenario))
            if baseline and baseline.get("checkpoint_exists"):
                baseline_by_seed[seed] = baseline
                break

    gate_rows: List[Dict[str, Any]] = []
    pairwise_rows: List[Dict[str, Any]] = []

    for row in rows:
        seed = str(row.get("seed", ""))
        scenario = str(row.get("scenario_key", ""))
        failures: List[str] = []
        warnings: List[str] = []
        baseline = baseline_by_seed.get(seed)
        invalid_run = int_or_zero(row.get("invalid_run")) != 0

        if not row.get("checkpoint_exists"):
            failures.append("missing_checkpoint")
        if invalid_run:
            failures.append("invalid_run")
        if not invalid_run and row.get("entry_status") not in ("", "Finished"):
            failures.append("entry_status")
        route_ok, route_warnings = accepted_route_status(row)
        warnings.extend(route_warnings)
        if not invalid_run and not route_ok:
            failures.append("global_status")
        if not invalid_run and int_or_zero(row.get("major_infraction_count")) != 0:
            failures.append("major_infractions")

        collision_count = sum(
            infraction_count(row.get(field))
            for field in ("collisions_layout", "collisions_pedestrian", "collisions_vehicle"))
        if not invalid_run and collision_count != 0:
            failures.append("collisions")

        if not invalid_run and scenario in SIDECAR_SCENARIOS:
            if int_or_zero(row.get("sidecar_fatal_errors")) != 0:
                failures.append("sidecar_fatal_errors")
            if int_or_zero(row.get("verify_warning_count")) != 0:
                warnings.append("verify_warning")
            if int_or_zero(row.get("verify_hard_error_count")) != 0:
                failures.append("verify_hard_error")
            if int_or_zero(row.get("sidecar_actor_removed_warning_count")) != 0:
                warnings.append("actor_removed_warning")
            if int_or_zero(row.get("sidecar_runtime_hard_error_count")) != 0:
                failures.append("sidecar_runtime_hard_error")
            if (
                    scenario not in PROFILE_ONLY_SIDECAR_SCENARIOS and
                    int_or_zero(row.get("controller_diagnostic_rows")) == 0):
                failures.append("missing_controller_diagnostics")
            if (
                    scenario not in PROFILE_ONLY_SIDECAR_SCENARIOS and
                    int_or_zero(row.get("damper_nonfinite_violations")) != 0):
                failures.append("damper_nonfinite")
            if (
                    scenario not in PROFILE_ONLY_SIDECAR_SCENARIOS and
                    int_or_zero(row.get("spring_identity_violations")) != 0):
                failures.append("spring_not_identity")

        if not invalid_run and scenario in JSONL_SCENARIOS:
            if row.get("checkpoint_exists") and int_or_zero(row.get("planning_jsonl_rows")) == 0:
                failures.append("missing_planning_jsonl")
            parse_error_count = (
                int_or_zero(row.get("planning_jsonl_malformed_lines_max")) +
                int_or_zero(row.get("planning_jsonl_rejected_messages_max")) +
                int_or_zero(row.get("planning_jsonl_read_errors_max")))
            if parse_error_count != 0:
                failures.append("planning_jsonl_parse_errors")

        if not invalid_run and scenario in JSONL_SIDECAR_SCENARIOS:
            age_max = float_or_none(row.get("planning_age_max"))
            if age_max is None:
                failures.append("missing_planning_age")
            elif age_max > planning_age_limit:
                failures.append("planning_age")
            if int_or_zero(row.get("planning_available_rows")) == 0:
                failures.append("planning_unavailable")

        if not invalid_run and scenario in TARGET_SPEED_SCENARIOS:
            if int_or_zero(row.get("tss_rows")) == 0:
                failures.append("missing_tss_diagnostics")
            if int_or_zero(row.get("tss_required_field_missing")) != 0:
                failures.append("missing_tss_fields")
            if int_or_zero(row.get("damper_bound_violations")) != 0:
                failures.append("damper_bounds")
            if int_or_zero(row.get("tss_forbidden_usage_rows")) != 0:
                failures.append("forbidden_planning_signal_usage")
            if scenario == "LEAD_target_speed_schedule_shadow":
                rows_count = int_or_zero(row.get("tss_rows"))
                shadow_rows = int_or_zero(row.get("tss_shadow_rows"))
                if rows_count == 0 or shadow_rows != rows_count:
                    failures.append("shadow_mode_not_applied")

        if not invalid_run and scenario in PARD_V2_SCENARIOS:
            if int_or_zero(row.get("pard_rows")) == 0:
                failures.append("missing_pard_diagnostics")
            if int_or_zero(row.get("pard_required_field_missing")) != 0:
                failures.append("missing_pard_fields")
            if int_or_zero(row.get("diagnostic_nonfinite_values")) != 0:
                failures.append("diagnostic_nonfinite")
            if int_or_zero(row.get("pard_damper_bound_violations")) != 0:
                failures.append("pard_damper_bounds")
            if scenario in PARD_V2_SHADOW_SCENARIOS:
                rows_count = int_or_zero(row.get("pard_rows"))
                shadow_rows = int_or_zero(row.get("pard_shadow_rows"))
                if rows_count == 0 or shadow_rows != rows_count:
                    failures.append("shadow_mode_not_applied")
                if int_or_zero(row.get(
                        "pard_shadow_actual_identity_violations")) != 0:
                    failures.append("pard_shadow_actual_not_identity")

        if not invalid_run and scenario in CONSTANT_DAMPING_SCENARIOS:
            if int_or_zero(row.get("damper_bound_violations")) != 0:
                failures.append("constant_damper_not_expected")

        score_delta = ""
        route_score_delta = ""
        major_infraction_delta = ""
        baseline_scenario = ""
        pairwise_status = ""
        if baseline:
            baseline_scenario = str(baseline.get("scenario_key", ""))
            if not row_valid_for_pairwise(baseline):
                pairwise_status = "baseline_invalid"
                warnings.append("baseline_invalid")
            elif invalid_run:
                pairwise_status = "row_invalid"
            else:
                pairwise_status = "ok"
                score = float_or_none(row.get("score_composed"))
                base_score = float_or_none(baseline.get("score_composed"))
                route_score = float_or_none(row.get("score_route"))
                base_route_score = float_or_none(baseline.get("score_route"))
                if score is not None and base_score is not None:
                    score_delta = score - base_score
                    if score < base_score - score_tolerance:
                        failures.append("score_composed_vs_baseline")
                elif row.get("checkpoint_exists"):
                    warnings.append("score_composed_missing")
                if route_score is not None and base_route_score is not None:
                    route_score_delta = route_score - base_route_score
                    if route_score < base_route_score - route_score_tolerance:
                        failures.append("route_score_vs_baseline")
                major_infraction_delta = (
                    int_or_zero(row.get("major_infraction_count")) -
                    int_or_zero(baseline.get("major_infraction_count")))
                if major_infraction_delta > 0:
                    failures.append("new_major_infractions_vs_baseline")
        elif row.get("checkpoint_exists"):
            warnings.append("missing_identity_baseline")
            pairwise_status = "missing_baseline"

        gate_row = {
            "seed": seed,
            "scenario_key": scenario,
            "baseline_scenario": baseline_scenario,
            "gate_status": "invalid_run" if invalid_run else (
                "fail" if failures else "pass"),
            "failed_checks": ";".join(sorted(set(failures))),
            "warnings": ";".join(sorted(set(warnings))),
            "pairwise_status": pairwise_status,
            "score_composed_delta_vs_baseline": score_delta,
            "score_route_delta_vs_baseline": route_score_delta,
            "major_infraction_delta_vs_baseline": major_infraction_delta,
        }
        gate_rows.append(gate_row)
        pairwise_rows.append(dict(gate_row))
    return gate_rows, pairwise_rows


def write_readme(
    path: str,
    root: str,
    rows: Sequence[Mapping[str, Any]],
    gate_rows: Sequence[Mapping[str, Any]],
) -> None:
    failures = [row for row in gate_rows if row.get("gate_status") != "pass"]
    warnings = [row for row in gate_rows if row.get("warnings")]
    with open(path, "w") as readme:
        readme.write("# Step 07 LEAD Matrix Results\n\n")
        readme.write("Output root:\n\n")
        readme.write("```text\n%s\n```\n\n" % root)
        readme.write("Rows: %d\n\n" % len(rows))
        readme.write("Gate status: %s\n\n" % ("fail" if failures else "pass"))
        readme.write("Generated files:\n\n")
        for name in (
                "matrix_summary.csv",
                "matrix_summary.json",
                "matrix_pairwise.csv",
                "gate_report.json",
                "README_results.md"):
            readme.write("- `%s`\n" % name)
        readme.write("\n")
        if failures:
            readme.write("Failed checks:\n\n")
            for row in failures:
                readme.write("- seed `%s` `%s`: `%s`\n" % (
                    row.get("seed", ""),
                    row.get("scenario_key", ""),
                    row.get("failed_checks", "")))
            readme.write("\n")
        if warnings:
            readme.write("Warnings:\n\n")
            for row in warnings:
                readme.write("- seed `%s` `%s`: `%s`\n" % (
                    row.get("seed", ""),
                    row.get("scenario_key", ""),
                    row.get("warnings", "")))
            readme.write("\n")
        readme.write("Use this report after smoke first. Run the full matrix only after the smoke gate passes.\n")


def main(args: argparse.Namespace) -> None:
    root = os.path.abspath(os.path.expanduser(args.root))
    rows = build_summary_rows(args)
    gate_rows, pairwise_rows = evaluate_gates(
        rows,
        args.score_tolerance,
        args.route_score_tolerance,
        args.planning_age_max)
    failed = [row for row in gate_rows if row.get("gate_status") != "pass"]
    warnings = [row for row in gate_rows if row.get("warnings")]

    summary_csv = os.path.join(root, "matrix_summary.csv")
    summary_json = os.path.join(root, "matrix_summary.json")
    pairwise_csv = os.path.join(root, "matrix_pairwise.csv")
    gate_json = os.path.join(root, "gate_report.json")
    readme_md = os.path.join(root, "README_results.md")

    write_csv_rows(summary_csv, rows)
    write_json(summary_json, rows)
    write_csv_rows(pairwise_csv, pairwise_rows)
    write_json(gate_json, {
        "status": "fail" if failed else "pass",
        "row_count": len(rows),
        "failed_count": len(failed),
        "warning_count": len(warnings),
        "failed_rows": failed,
        "warning_rows": warnings,
        "score_tolerance": args.score_tolerance,
        "route_score_tolerance": args.route_score_tolerance,
        "planning_age_max": args.planning_age_max,
        "files": {
            "matrix_summary_csv": summary_csv,
            "matrix_summary_json": summary_json,
            "matrix_pairwise_csv": pairwise_csv,
            "gate_report_json": gate_json,
            "readme_results_md": readme_md,
        },
    })
    write_readme(readme_md, root, rows, gate_rows)

    print("Step 07 summary/gate complete")
    print("  root=%s" % root)
    print("  rows=%d" % len(rows))
    print("  gate_status=%s" % ("fail" if failed else "pass"))
    print("  summary_csv=%s" % summary_csv)
    print("  gate_report=%s" % gate_json)


if __name__ == "__main__":
    main(parse_args())

#!/usr/bin/env python3

"""Summarize and gate official LEAD sanity route outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


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
    parser.add_argument("--root", required=True, help="official sanity output root")
    parser.add_argument(
        "--routes",
        default="",
        help="comma-separated route keys to include; defaults to plan rows")
    parser.add_argument(
        "--sim-root",
        default=os.environ.get("SIM_ROOT", os.path.expanduser("~/sim")),
        help="host sim root used to translate /workspace paths")
    return parser.parse_args()


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
        fields = ["status"]
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


def int_or_zero(value: Any) -> int:
    number = float_or_none(value)
    if number is None:
        return 0
    return int(round(number))


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        return "%0.9g" % value
    return value


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


def read_text_tail(path: str, max_bytes: int = 300000) -> str:
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rb") as text_file:
        text_file.seek(0, os.SEEK_END)
        size = text_file.tell()
        text_file.seek(max(0, size - max_bytes), os.SEEK_SET)
        return text_file.read().decode("utf-8", errors="replace")


def extract_closed_loop_config(text: str) -> str:
    matches = [
        line.split("=", 1)[1].strip()
        for line in text.splitlines()
        if line.startswith("LEAD_CLOSED_LOOP_CONFIG=")
    ]
    return matches[-1] if matches else ""


def extract_loaded_checkpoints(text: str) -> List[str]:
    matches = re.findall(r"Loading model weight from\s+(\S+)", text)
    return [os.path.basename(match) for match in matches]


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


def detect_invalid_run(row: Mapping[str, Any]) -> bool:
    status_text = " ".join(str(row.get(field, "")) for field in (
        "entry_status",
        "global_status",
        "route_statuses",
        "checkpoint_exceptions",
        "stdout_tail",
    ))
    if "Agent couldn't be set up" in status_text:
        return True
    duration_game = float_or_none(row.get("duration_game"))
    sensors_count = int_or_zero(row.get("sensors_count"))
    checkpoint_exists = int_or_zero(row.get("checkpoint_exists")) != 0
    if checkpoint_exists and duration_game is not None and abs(duration_game) <= 1.0e-9:
        return True
    if checkpoint_exists and sensors_count == 0:
        return True
    return False


def selected_plan_rows(args: argparse.Namespace, root: str) -> List[Dict[str, str]]:
    plan = read_csv_rows(os.path.join(root, "official_sanity_plan.csv"))
    requested = set(parse_csv_list(args.routes))
    if requested:
        plan = [row for row in plan if row.get("route_key", "") in requested]
    return plan


def result_by_route(root: str) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(os.path.join(root, "official_sanity_results.csv"))
    return {row.get("route_key", ""): row for row in rows}


def build_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    root = os.path.abspath(os.path.expanduser(args.root))
    plan_rows = selected_plan_rows(args, root)
    results = result_by_route(root)
    rows: List[Dict[str, Any]] = []
    for plan in plan_rows:
        key = plan.get("route_key", "")
        result = results.get(key, {})
        run_dir = host_path(
            result.get("output_dir_host") or plan.get("output_dir_host", ""),
            args.sim_root)
        stdout_log = host_path(
            result.get("stdout_log") or plan.get("stdout_log", ""),
            args.sim_root)
        result_json = host_path(
            result.get("result_json") or os.path.join(run_dir, "lead", "checkpoint_endpoint.json"),
            args.sim_root)
        stdout_tail = read_text_tail(stdout_log)
        loaded_checkpoints = extract_loaded_checkpoints(stdout_tail)

        row: Dict[str, Any] = {
            "route_key": key,
            "shard_index": plan.get("shard_index", ""),
            "town": plan.get("town", ""),
            "route_id": plan.get("route_id", ""),
            "route_host": plan.get("route_host", ""),
            "route_container": plan.get("route_container", ""),
            "bench2drive": plan.get("bench2drive", ""),
            "fail2drive": plan.get("fail2drive", ""),
            "traffic_manager_seed": plan.get("traffic_manager_seed", ""),
            "carla_port": plan.get("carla_port", ""),
            "traffic_manager_port": plan.get("traffic_manager_port", ""),
            "compose_project_name": plan.get("compose_project_name", ""),
            "runner_status": result.get("status", ""),
            "runner_return_code": result.get("return_code", ""),
            "run_dir": run_dir,
            "stdout_log": stdout_log,
            "result_json": result_json,
            "closed_loop_config": extract_closed_loop_config(stdout_tail),
            "closed_loop_reproduction_config": int(
                "sensor_agent_creeping=True" in stdout_tail and
                "use_kalman_filter=True" in stdout_tail and
                "slower_for_stop_sign=True" in stdout_tail),
            "loaded_checkpoint_count": len(loaded_checkpoints),
            "loaded_checkpoints": ";".join(loaded_checkpoints),
            "stdout_agent_setup_failure": int("Agent couldn't be set up" in stdout_tail),
            "stdout_tail": stdout_tail[-1000:],
        }
        row.update(extract_checkpoint_metrics(result_json))
        if row.get("checkpoint_exists") != 1:
            row.setdefault("major_infraction_count", 0)
        row["invalid_run"] = int(detect_invalid_run(row))
        collision_count = sum(
            infraction_count(row.get(field))
            for field in ("collisions_layout", "collisions_pedestrian", "collisions_vehicle"))
        row["collision_count"] = collision_count
        rows.append(row)
    return rows


def evaluate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    gate_rows: List[Dict[str, Any]] = []
    if rows and all(str(row.get("runner_return_code", "")) == "dry_run" for row in rows):
        gate_status = "dry_run"
    else:
        gate_status = "pass"

    for row in rows:
        failures: List[str] = []
        warnings: List[str] = []
        runner_rc = str(row.get("runner_return_code", ""))
        if runner_rc == "dry_run":
            warnings.append("dry_run_no_route_result")
        elif runner_rc not in ("", "0"):
            failures.append("runner_return_code")
        if runner_rc != "dry_run":
            if int_or_zero(row.get("checkpoint_exists")) == 0:
                failures.append("missing_checkpoint")
            if int_or_zero(row.get("invalid_run")) != 0:
                failures.append("invalid_run")
            if str(row.get("entry_status", "")) not in ("", "Finished"):
                failures.append("entry_status")
            if str(row.get("global_status", "")) not in ("Perfect", "Completed"):
                failures.append("global_status")
            score_route = float_or_none(row.get("score_route"))
            if score_route is None or score_route < 99.0:
                failures.append("route_score")
            if int_or_zero(row.get("major_infraction_count")) != 0:
                failures.append("major_infractions")
            if int_or_zero(row.get("collision_count")) != 0:
                failures.append("collisions")
            if int_or_zero(row.get("closed_loop_reproduction_config")) == 0:
                failures.append("closed_loop_config")
            if int_or_zero(row.get("loaded_checkpoint_count")) < 3:
                warnings.append("checkpoint_ensemble_not_confirmed")
            if str(row.get("global_status", "")) == "Completed" and not failures:
                warnings.append("global_completed")

        if failures and gate_status != "dry_run":
            gate_status = "fail"
        elif warnings and gate_status == "pass":
            gate_status = "warning"
        gate_rows.append({
            "route_key": row.get("route_key", ""),
            "status": "fail" if failures else ("warning" if warnings else "pass"),
            "failures": ";".join(sorted(set(failures))),
            "warnings": ";".join(sorted(set(warnings))),
            "score_composed": row.get("score_composed", ""),
            "score_route": row.get("score_route", ""),
            "global_status": row.get("global_status", ""),
            "major_infraction_count": row.get("major_infraction_count", ""),
            "collision_count": row.get("collision_count", ""),
        })

    return {
        "gate_status": gate_status,
        "route_count": len(rows),
        "failed_count": sum(1 for row in gate_rows if row["status"] == "fail"),
        "warning_count": sum(1 for row in gate_rows if row["status"] == "warning"),
        "rows": gate_rows,
    }


def write_readme(path: str, gate: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    with open(path, "w") as md:
        md.write("# LEAD Official Sanity Results\n\n")
        md.write(f"- gate_status: `{gate.get('gate_status', '')}`\n")
        md.write(f"- route_count: `{gate.get('route_count', '')}`\n")
        md.write(f"- failed_count: `{gate.get('failed_count', '')}`\n")
        md.write(f"- warning_count: `{gate.get('warning_count', '')}`\n\n")
        md.write("| route | status | score | route | infractions | collisions |\n")
        md.write("| --- | --- | --- | --- | --- | --- |\n")
        gate_by_route = {row["route_key"]: row for row in gate.get("rows", [])}
        for row in rows:
            gate_row = gate_by_route.get(str(row.get("route_key", "")), {})
            md.write(
                "| {route} | {status} | {score} | {route_score} | {infractions} | {collisions} |\n".format(
                    route=row.get("route_key", ""),
                    status=gate_row.get("status", ""),
                    score=row.get("score_composed", ""),
                    route_score=row.get("score_route", ""),
                    infractions=row.get("major_infraction_count", ""),
                    collisions=row.get("collision_count", ""),
                ))


def main() -> None:
    args = parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))
    rows = build_rows(args)
    gate = evaluate_rows(rows)
    write_csv_rows(os.path.join(root, "official_sanity_summary.csv"), rows)
    write_json(os.path.join(root, "official_sanity_summary.json"), rows)
    write_json(os.path.join(root, "official_sanity_gate.json"), gate)
    write_readme(os.path.join(root, "README_official_sanity.md"), gate, rows)
    print("Official LEAD sanity summary complete")
    print(f"  root={root}")
    print(f"  rows={len(rows)}")
    print(f"  gate_status={gate['gate_status']}")
    print(f"  summary_csv={os.path.join(root, 'official_sanity_summary.csv')}")
    print(f"  gate_report={os.path.join(root, 'official_sanity_gate.json')}")


if __name__ == "__main__":
    main()

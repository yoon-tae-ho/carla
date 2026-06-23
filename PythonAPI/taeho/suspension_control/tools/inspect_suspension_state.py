#!/usr/bin/env python3
"""Inspect CARLA ``vehicle.get_suspension_state()`` without applying commands.

The route runner or another CARLA process must own simulation stepping. This
tool only waits for external ticks, reads the hero vehicle suspension state,
writes per-wheel CSV rows, and can emit a compact validation report.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


EXPECTED_WHEEL_NAMES = ("FL", "FR", "RL", "RR")

CSV_FIELDS = (
    "frame",
    "timestamp",
    "elapsed_seconds",
    "actor_id",
    "vehicle_type_id",
    "role_name",
    "state_source",
    "state_valid",
    "state_velocity_valid",
    "compression_convention_validated",
    "failure_reason",
    "wheel_count",
    "wheel_index_raw",
    "wheel_name_canonical",
    "raw_suspension_offset_m",
    "compression_m",
    "suspension_travel_m",
    "suspension_velocity_mps",
    "normalized_travel",
    "velocity_valid",
    "contact_valid",
    "wheel_in_air",
    "field_valid",
    "normalized_tire_load_valid",
    "suspension_force_n",
    "spring_force_n",
    "damper_force_n",
    "tire_load_n",
    "normalized_tire_load",
)

NUMERIC_FIELDS = (
    "raw_suspension_offset_m",
    "compression_m",
    "suspension_travel_m",
    "suspension_velocity_mps",
    "normalized_travel",
    "suspension_force_n",
    "spring_force_n",
    "damper_force_n",
    "tire_load_n",
    "normalized_tire_load",
)

BOOL_FIELDS = (
    "state_valid",
    "state_velocity_valid",
    "compression_convention_validated",
    "velocity_valid",
    "contact_valid",
    "wheel_in_air",
    "field_valid",
    "normalized_tire_load_valid",
)


def _existing_paths(paths: Iterable[str]) -> List[str]:
    result = []
    for path in paths:
        if path and os.path.exists(path) and path not in result:
            result.append(path)
    return result


def _pythonapi_paths(pythonapi_root: str) -> List[str]:
    root = os.path.abspath(os.path.expanduser(pythonapi_root))
    py_major, py_minor = sys.version_info[:2]
    py_dot = "%d.%d" % (py_major, py_minor)
    platform = "win-amd64" if os.name == "nt" else "linux-x86_64"
    build_glob = os.path.join(
        root,
        "carla",
        "build",
        "lib.*-cpython-%d%d" % (py_major, py_minor),
    )
    legacy_build_glob = os.path.join(root, "carla", "build", "lib.*")
    egg_glob = os.path.join(
        root,
        "carla",
        "dist",
        "carla-0.9.15-py%s-%s.egg" % (py_dot, platform),
    )
    return _existing_paths(
        sorted(glob.glob(build_glob))
        + sorted(glob.glob(legacy_build_glob))
        + [os.path.join(root, "carla")]
        + sorted(glob.glob(egg_glob))
        + [root]
    )


def prepend_pythonapi(pythonapi_root: str) -> None:
    """Put a local patched CARLA PythonAPI ahead of site-packages."""

    for path in reversed(_pythonapi_paths(pythonapi_root)):
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def load_carla(pythonapi_root: str) -> Any:
    if pythonapi_root:
        prepend_pythonapi(pythonapi_root)
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "ERROR: failed to import carla. Rebuild/install the patched "
            "PythonAPI or pass --pythonapi-root. Original error: %s" % exc
        )
    return carla


def text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def attr_text(obj: Any, name: str, default: Any = "") -> str:
    return text_value(getattr(obj, name, default))


def attr_float_text(obj: Any, name: str) -> str:
    value = getattr(obj, name, None)
    if value is None:
        return ""
    try:
        return "%.12g" % float(value)
    except (TypeError, ValueError):
        return str(value)


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def finite_values(values: Iterable[Any]) -> List[float]:
    result = []
    for value in values:
        parsed = parse_float(value)
        if parsed is not None and math.isfinite(parsed):
            result.append(parsed)
    return result


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def value_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "min": None,
            "mean": None,
            "max": None,
            "std": None,
            "p95": None,
            "p99": None,
        }
    mean = sum(values) / float(len(values))
    variance = sum((value - mean) ** 2 for value in values) / float(len(values))
    return {
        "min": min(values),
        "mean": mean,
        "max": max(values),
        "std": math.sqrt(variance),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def fmt_optional(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "%.6g" % value


def ratio(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def fmt_ratio(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "%.3f" % value


def sign_label(value: Optional[float], eps: float = 1.0e-9) -> str:
    if value is None:
        return "n/a"
    if value > eps:
        return "positive"
    if value < -eps:
        return "negative"
    return "near_zero"


def find_vehicle(world: Any, role_name: str, actor_id: Optional[int]) -> Any:
    if actor_id is not None:
        actor = world.get_actor(actor_id)
        if actor is None:
            raise RuntimeError("no actor found with id %d" % actor_id)
        return actor

    vehicles = list(world.get_actors().filter("vehicle.*"))
    for actor in vehicles:
        attributes = getattr(actor, "attributes", {}) or {}
        if attributes.get("role_name") == role_name:
            return actor
    raise RuntimeError("no vehicle found with role_name=%r" % role_name)


def verify_vehicle_api(vehicle: Any) -> None:
    if not hasattr(vehicle, "get_suspension_state"):
        raise RuntimeError(
            "vehicle actor does not expose get_suspension_state(); "
            "check that the rebuilt Step02 Python artifact is active"
        )


def wait_for_external_sample(world: Any, args: argparse.Namespace) -> None:
    if args.sample_mode == "wait-for-tick":
        try:
            world.wait_for_tick(args.wait_timeout)
            return
        except RuntimeError as exc:
            print(
                "WARNING: wait_for_tick timed out after %.3fs: %s" % (
                    args.wait_timeout,
                    exc,
                ),
                file=sys.stderr,
            )
    time.sleep(max(0.0, args.sample_period_sec))


def state_to_rows(
    state: Any,
    vehicle: Any,
    elapsed_seconds: float,
) -> List[Dict[str, str]]:
    wheels = list(getattr(state, "wheels", []) or [])
    actor_id = getattr(state, "actor_id", getattr(vehicle, "id", ""))
    attributes = getattr(vehicle, "attributes", {}) or {}
    common = {
        "frame": attr_text(state, "frame"),
        "timestamp": attr_float_text(state, "timestamp"),
        "elapsed_seconds": "%.9f" % elapsed_seconds,
        "actor_id": text_value(actor_id),
        "vehicle_type_id": text_value(getattr(vehicle, "type_id", "")),
        "role_name": text_value(attributes.get("role_name", "")),
        "state_source": attr_text(state, "state_source"),
        "state_valid": attr_text(state, "state_valid"),
        "state_velocity_valid": attr_text(state, "velocity_valid"),
        "compression_convention_validated": attr_text(
            state,
            "compression_convention_validated",
        ),
        "failure_reason": attr_text(state, "failure_reason"),
        "wheel_count": attr_text(state, "wheel_count", len(wheels)),
    }

    rows = []
    for index, wheel in enumerate(wheels):
        row = dict(common)
        row.update(
            {
                "wheel_index_raw": attr_text(wheel, "wheel_index_raw", index),
                "wheel_name_canonical": attr_text(
                    wheel,
                    "wheel_name_canonical",
                    EXPECTED_WHEEL_NAMES[index] if index < len(EXPECTED_WHEEL_NAMES) else "",
                ),
                "raw_suspension_offset_m": attr_float_text(
                    wheel,
                    "raw_suspension_offset_m",
                ),
                "compression_m": attr_float_text(
                    wheel,
                    "suspension_compression_m",
                ),
                "suspension_travel_m": attr_float_text(
                    wheel,
                    "suspension_travel_m",
                ),
                "suspension_velocity_mps": attr_float_text(
                    wheel,
                    "suspension_velocity_mps",
                ),
                "normalized_travel": attr_float_text(wheel, "normalized_travel"),
                "velocity_valid": attr_text(wheel, "velocity_valid"),
                "contact_valid": attr_text(wheel, "contact_valid"),
                "wheel_in_air": attr_text(wheel, "wheel_in_air"),
                "field_valid": attr_text(wheel, "field_valid"),
                "normalized_tire_load_valid": attr_text(
                    wheel,
                    "normalized_tire_load_valid",
                ),
                "suspension_force_n": attr_float_text(wheel, "suspension_force_n"),
                "spring_force_n": attr_float_text(wheel, "spring_force_n"),
                "damper_force_n": attr_float_text(wheel, "damper_force_n"),
                "tire_load_n": attr_float_text(wheel, "tire_load_n"),
                "normalized_tire_load": attr_float_text(wheel, "normalized_tire_load"),
            }
        )
        rows.append(row)
    return rows


def summarize_rows(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    wheel_groups: MutableMapping[str, List[Mapping[str, str]]] = defaultdict(list)
    sample_orders = set()
    sample_order_map: MutableMapping[str, List[Tuple[str, str]]] = defaultdict(list)

    for row in rows:
        wheel_key = "%s:%s" % (
            row.get("wheel_index_raw", ""),
            row.get("wheel_name_canonical", ""),
        )
        wheel_groups[wheel_key].append(row)
        frame_key = "%s/%s" % (row.get("frame", ""), row.get("elapsed_seconds", ""))
        sample_order_map[frame_key].append(
            (row.get("wheel_index_raw", ""), row.get("wheel_name_canonical", ""))
        )

    for order in sample_order_map.values():
        sample_orders.add(tuple(order))

    total_rows = len(rows)
    overall_field_valid = sum(1 for row in rows if parse_bool(row.get("field_valid")) is True)
    overall_state_valid = sum(1 for row in rows if parse_bool(row.get("state_valid")) is True)
    overall_velocity_valid = sum(
        1 for row in rows if parse_bool(row.get("velocity_valid")) is True
    )

    nan_counts = {field: 0 for field in NUMERIC_FIELDS}
    inf_counts = {field: 0 for field in NUMERIC_FIELDS}
    missing_counts = {field: 0 for field in NUMERIC_FIELDS}
    for row in rows:
        for field in NUMERIC_FIELDS:
            value = parse_float(row.get(field))
            if value is None:
                missing_counts[field] += 1
            elif math.isnan(value):
                nan_counts[field] += 1
            elif math.isinf(value):
                inf_counts[field] += 1

    wheel_summaries = {}
    for wheel_key in sorted(wheel_groups):
        wheel_rows = wheel_groups[wheel_key]
        total = len(wheel_rows)
        numeric = {
            field: finite_values(row.get(field) for row in wheel_rows)
            for field in NUMERIC_FIELDS
        }
        raw_values = numeric["raw_suspension_offset_m"]
        compression_values = numeric["compression_m"]
        raw_comp_pairs = []
        for row in wheel_rows:
            raw = parse_float(row.get("raw_suspension_offset_m"))
            compression = parse_float(row.get("compression_m"))
            if (
                raw is not None
                and compression is not None
                and math.isfinite(raw)
                and math.isfinite(compression)
            ):
                raw_comp_pairs.append((raw, compression))
        sign_pairs = [
            (raw, compression)
            for raw, compression in raw_comp_pairs
            if abs(raw) > 1.0e-9 and abs(compression) > 1.0e-9
        ]
        same_sign = sum(
            1
            for raw, compression in sign_pairs
            if (raw > 0.0 and compression > 0.0)
            or (raw < 0.0 and compression < 0.0)
        )
        abs_diffs = [abs(compression - raw) for raw, compression in raw_comp_pairs]
        wheel_summaries[wheel_key] = {
            "rows": total,
            "field_valid_ratio": ratio(
                sum(1 for row in wheel_rows if parse_bool(row.get("field_valid")) is True),
                total,
            ),
            "state_valid_ratio": ratio(
                sum(1 for row in wheel_rows if parse_bool(row.get("state_valid")) is True),
                total,
            ),
            "velocity_valid_ratio": ratio(
                sum(1 for row in wheel_rows if parse_bool(row.get("velocity_valid")) is True),
                total,
            ),
            "contact_valid_ratio": ratio(
                sum(1 for row in wheel_rows if parse_bool(row.get("contact_valid")) is True),
                total,
            ),
            "wheel_in_air_ratio": ratio(
                sum(1 for row in wheel_rows if parse_bool(row.get("wheel_in_air")) is True),
                total,
            ),
            "numeric": {field: value_stats(values) for field, values in numeric.items()},
            "finite_counts": {field: len(values) for field, values in numeric.items()},
            "raw_mean_sign": sign_label(
                value_stats(raw_values)["mean"] if raw_values else None
            ),
            "compression_mean_sign": sign_label(
                value_stats(compression_values)["mean"] if compression_values else None
            ),
            "raw_compression_same_sign_ratio": ratio(same_sign, len(sign_pairs)),
            "compression_minus_raw_abs_max": max(abs_diffs) if abs_diffs else None,
        }

    return {
        "total_rows": total_rows,
        "sample_count": len(sample_order_map),
        "overall_field_valid_ratio": ratio(overall_field_valid, total_rows),
        "overall_state_valid_ratio": ratio(overall_state_valid, total_rows),
        "overall_velocity_valid_ratio": ratio(overall_velocity_valid, total_rows),
        "nan_counts": nan_counts,
        "inf_counts": inf_counts,
        "missing_counts": missing_counts,
        "sample_orders": sorted(sample_orders),
        "wheel_summaries": wheel_summaries,
    }


def summary_lines(summary: Mapping[str, Any], route_note: str) -> List[str]:
    lines = []
    lines.append("samples=%s rows=%s" % (summary["sample_count"], summary["total_rows"]))
    lines.append(
        "valid_ratio state=%s field=%s velocity=%s"
        % (
            fmt_ratio(summary["overall_state_valid_ratio"]),
            fmt_ratio(summary["overall_field_valid_ratio"]),
            fmt_ratio(summary["overall_velocity_valid_ratio"]),
        )
    )
    lines.append("route_note=%s" % route_note)
    lines.append("wheel_order_signatures=%d" % len(summary["sample_orders"]))
    for order in summary["sample_orders"][:5]:
        lines.append(
            "  order: %s"
            % ", ".join("%s=%s" % (index, name) for index, name in order)
        )
    for wheel_key, wheel_summary in summary["wheel_summaries"].items():
        raw = wheel_summary["numeric"]["raw_suspension_offset_m"]
        comp = wheel_summary["numeric"]["compression_m"]
        vel = wheel_summary["numeric"]["suspension_velocity_mps"]
        lines.append(
            "%s: valid field=%s state=%s contact=%s air=%s velocity=%s"
            % (
                wheel_key,
                fmt_ratio(wheel_summary["field_valid_ratio"]),
                fmt_ratio(wheel_summary["state_valid_ratio"]),
                fmt_ratio(wheel_summary["contact_valid_ratio"]),
                fmt_ratio(wheel_summary["wheel_in_air_ratio"]),
                fmt_ratio(wheel_summary["velocity_valid_ratio"]),
            )
        )
        lines.append(
            "  raw_offset m min/mean/max/std=%s/%s/%s/%s sign=%s"
            % (
                fmt_optional(raw["min"]),
                fmt_optional(raw["mean"]),
                fmt_optional(raw["max"]),
                fmt_optional(raw["std"]),
                wheel_summary["raw_mean_sign"],
            )
        )
        lines.append(
            "  compression m min/mean/max/std=%s/%s/%s/%s sign=%s same_sign_with_raw=%s max_abs_diff_raw=%s"
            % (
                fmt_optional(comp["min"]),
                fmt_optional(comp["mean"]),
                fmt_optional(comp["max"]),
                fmt_optional(comp["std"]),
                wheel_summary["compression_mean_sign"],
                fmt_ratio(wheel_summary["raw_compression_same_sign_ratio"]),
                fmt_optional(wheel_summary["compression_minus_raw_abs_max"]),
            )
        )
        lines.append(
            "  velocity mps min/mean/max/std/p95/p99=%s/%s/%s/%s/%s/%s"
            % (
                fmt_optional(vel["min"]),
                fmt_optional(vel["mean"]),
                fmt_optional(vel["max"]),
                fmt_optional(vel["std"]),
                fmt_optional(vel["p95"]),
                fmt_optional(vel["p99"]),
            )
        )
    return lines


def print_summary(summary: Mapping[str, Any], route_note: str) -> None:
    for line in summary_lines(summary, route_note):
        print(line)


def write_markdown_report(
    path: str,
    args: argparse.Namespace,
    output_csv: str,
    summary: Mapping[str, Any],
) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    lines = [
        "# PR1 Runtime Validation Report",
        "",
        "Generated: `%s`" % datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "",
        "Command:",
        "",
        "```bash",
        "python %s --role-name %s --duration-sec %.3g --output %s --print-summary"
        % (
            os.path.relpath(__file__, os.getcwd()),
            args.role_name,
            args.duration_sec,
            output_csv,
        ),
        "```",
        "",
        "Output CSV:",
        "",
        "```text",
        output_csv,
        "```",
        "",
        "Route Surface Note:",
        "",
        "- %s" % args.route_note,
        "",
        "Summary:",
        "",
    ]
    lines.extend("- %s" % line for line in summary_lines(summary, args.route_note))
    lines.extend(
        [
            "",
            "NaN/Inf Counts:",
            "",
        ]
    )
    for field in NUMERIC_FIELDS:
        lines.append(
            "- `%s`: nan=%d, inf=%d, missing=%d"
            % (
                field,
                summary["nan_counts"][field],
                summary["inf_counts"][field],
                summary["missing_counts"][field],
            )
        )
    lines.extend(
        [
            "",
            "CSV Schema:",
            "",
            "```text",
            ",".join(CSV_FIELDS),
            "```",
        ]
    )
    with open(path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines) + "\n")


def open_csv_writer(path: str) -> Tuple[Any, csv.DictWriter]:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    csv_file = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    return csv_file, writer


def inspect(args: argparse.Namespace) -> int:
    carla = load_carla(args.pythonapi_root)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    vehicle = find_vehicle(world, role_name=args.role_name, actor_id=args.actor_id)
    verify_vehicle_api(vehicle)

    print(
        "Inspecting actor id=%s type_id=%s role_name=%s"
        % (
            getattr(vehicle, "id", ""),
            getattr(vehicle, "type_id", ""),
            (getattr(vehicle, "attributes", {}) or {}).get("role_name", ""),
        )
    )
    print("Writing CSV to %s" % args.output)

    csv_file, writer = open_csv_writer(args.output)
    rows: List[Dict[str, str]] = []
    start = time.monotonic()
    last_frame = None
    sample_count = 0
    try:
        while True:
            elapsed = time.monotonic() - start
            if sample_count > 0 and elapsed >= args.duration_sec:
                break
            if args.max_samples is not None and sample_count >= args.max_samples:
                break

            wait_for_external_sample(world, args)
            elapsed = time.monotonic() - start
            state = vehicle.get_suspension_state()
            frame = getattr(state, "frame", None)
            if (
                args.skip_duplicate_frames
                and last_frame is not None
                and frame == last_frame
            ):
                continue
            last_frame = frame

            state_rows = state_to_rows(state, vehicle, elapsed)
            for row in state_rows:
                writer.writerow(row)
            csv_file.flush()
            rows.extend(state_rows)
            sample_count += 1
    finally:
        csv_file.close()

    summary = summarize_rows(rows)
    if args.print_summary:
        print_summary(summary, args.route_note)
    if args.report_output:
        write_markdown_report(args.report_output, args, args.output, summary)
        print("Wrote report to %s" % args.report_output)

    if not rows:
        print("WARNING: no suspension rows were collected", file=sys.stderr)
        return 2
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default=os.environ.get("CARLA_HOST", "127.0.0.1"),
        help="CARLA host (default: env CARLA_HOST or 127.0.0.1)",
    )
    parser.add_argument(
        "-p",
        "--port",
        default=int(os.environ.get("CARLA_PORT", os.environ.get("PORT", "2000"))),
        type=int,
        help="CARLA RPC port (default: env CARLA_PORT/PORT or 2000)",
    )
    parser.add_argument(
        "--timeout",
        default=5.0,
        type=float,
        help="CARLA client timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--pythonapi-root",
        default=os.environ.get("CARLA_PYTHONAPI_ROOT", ""),
        help="optional patched CARLA PythonAPI root to prepend before import",
    )
    parser.add_argument(
        "--role-name",
        default="hero",
        help="vehicle role_name to inspect when --actor-id is omitted (default: hero)",
    )
    parser.add_argument(
        "--actor-id",
        type=int,
        default=None,
        help="explicit vehicle actor id; overrides --role-name",
    )
    parser.add_argument(
        "--duration-sec",
        default=10.0,
        type=float,
        help="wall-clock inspection duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--max-samples",
        default=None,
        type=int,
        help="optional cap on sampled suspension states",
    )
    parser.add_argument(
        "--sample-mode",
        choices=("wait-for-tick", "poll"),
        default="wait-for-tick",
        help="wait for external CARLA ticks or poll by wall-clock sleep (default: wait-for-tick)",
    )
    parser.add_argument(
        "--wait-timeout",
        default=2.0,
        type=float,
        help="wait_for_tick timeout before falling back to sleep (default: 2)",
    )
    parser.add_argument(
        "--sample-period-sec",
        default=0.05,
        type=float,
        help="poll/fallback sleep period in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--include-duplicate-frames",
        dest="skip_duplicate_frames",
        action="store_false",
        help="keep repeated frame ids when polling a paused world",
    )
    parser.set_defaults(skip_duplicate_frames=True)
    parser.add_argument(
        "--output",
        default="/tmp/suspension_state.csv",
        help="per-wheel CSV output path (default: /tmp/suspension_state.csv)",
    )
    parser.add_argument(
        "--report-output",
        default="",
        help="optional markdown report path for the collected CSV",
    )
    parser.add_argument(
        "--route-note",
        default="not provided; record whether this run included curbs, jumps, or off-road sections",
        help="free-text surface note included in generated reports",
    )
    parser.add_argument(
        "--print-summary",
        action="store_true",
        help="print valid ratios and per-wheel min/mean/max/std statistics",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return inspect(args)


if __name__ == "__main__":
    raise SystemExit(main())

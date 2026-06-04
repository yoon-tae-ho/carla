"""Route XML parsing and 2D progress projection helpers."""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROUTE_PROGRESS_FIELDS = (
    "route_progress_available",
    "route_progress_raw_m",
    "route_progress_m",
    "route_progress_fraction",
    "route_progress_total_length_m",
    "route_total_length_m",
    "route_progress_delta_m",
    "route_delta_progress_m",
    "route_progress_rate_mps",
    "route_progress_raw_negative_delta_m",
    "route_progress_negative_raw",
    "route_progress_raw_deviation_m",
    "route_progress_deviation_valid",
    "route_deviation_m",
    "route_distance_to_end_m",
    "route_progress_monotonic_m",
    "route_progress_monotonic_fraction",
    "route_progress_nearest_segment_index",
    "route_progress_projection_segment",
    "route_progress_global_research_used",
    "route_progress_search_window_start",
    "route_progress_search_window_end",
    "route_progress_tracker_status",
    "route_progress_error",
)

DEFAULT_DENSIFY_SPACING_M = 1.0
DEFAULT_DEVIATION_VALID_THRESHOLD_M = 8.0
DEFAULT_SEARCH_WINDOW_SEGMENTS = 30
DEFAULT_GLOBAL_RESEARCH_MIN_DEVIATION_M = 8.0
DEFAULT_GLOBAL_RESEARCH_IMPROVEMENT_M = 2.0


@dataclass(frozen=True)
class RoutePolyline:
    route_id: str
    points: Tuple[Tuple[float, float, float], ...]
    cumulative_lengths: Tuple[float, ...]
    total_length_m: float


class RouteProgressTracker:
    """Project vehicle positions onto a selected leaderboard route polyline."""

    def __init__(
        self,
        route_xml_path: str = "",
        route_id: Optional[Any] = None,
        polyline: Optional[RoutePolyline] = None,
        densify_spacing_m: float = DEFAULT_DENSIFY_SPACING_M,
        search_window_segments: int = DEFAULT_SEARCH_WINDOW_SEGMENTS,
        deviation_valid_threshold_m: float = DEFAULT_DEVIATION_VALID_THRESHOLD_M,
        global_research_min_deviation_m: float = (
            DEFAULT_GLOBAL_RESEARCH_MIN_DEVIATION_M),
        global_research_improvement_m: float = (
            DEFAULT_GLOBAL_RESEARCH_IMPROVEMENT_M),
    ):
        self.route_xml_path = os.path.abspath(os.path.expanduser(route_xml_path or ""))
        self.requested_route_id = "" if route_id is None else str(route_id)
        self.polyline = polyline
        self.error = ""
        self.search_window_segments = max(1, int(search_window_segments))
        self.deviation_valid_threshold_m = max(
            0.0,
            float(deviation_valid_threshold_m))
        self.global_research_min_deviation_m = max(
            0.0,
            float(global_research_min_deviation_m))
        self.global_research_improvement_m = max(
            0.0,
            float(global_research_improvement_m))
        self._actor_state: Dict[str, Dict[str, Any]] = {}
        if polyline is None:
            self.polyline, self.error = load_route_polyline(
                self.route_xml_path,
                self.requested_route_id,
                densify_spacing_m=densify_spacing_m)

    @property
    def available(self) -> bool:
        return self.polyline is not None and self.polyline.total_length_m > 0.0

    def update(
        self,
        state: Any = None,
        *,
        actor_id: Any = "hero",
        frame: Optional[Any] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        speed: Optional[float] = None,
        elapsed_seconds: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return route-progress diagnostics for one actor at one frame."""

        del speed
        if not self.available:
            return unavailable_route_progress(self.error or "route_unavailable")
        x_value = _first_position_value(state, x, "x")
        y_value = _first_position_value(state, y, "y")
        _first_position_value(state, z, "z")
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            return unavailable_route_progress("nonfinite_position")

        actor_key = str(actor_id)
        previous = self._actor_state.get(actor_key, {})
        projection = self._project_with_stabilization(
            previous=previous,
            x=x_value,
            y=y_value)
        previous_progress = previous.get("raw_progress_m")
        previous_time = previous.get("elapsed_seconds")
        elapsed = _first_time_value(state, elapsed_seconds, "elapsed_seconds")
        dt_value = _finite_optional(dt)
        if dt_value is None:
            dt_value = _first_time_value(state, None, "dt")
        if (
                (dt_value is None or dt_value <= 0.0) and
                elapsed is not None and
                previous_time is not None):
            dt_value = elapsed - previous_time
        if dt_value is None or dt_value <= 0.0 or not math.isfinite(dt_value):
            dt_value = 0.0

        progress_m = projection["progress_m"]
        delta_progress_m = (
            0.0 if previous_progress is None else progress_m - previous_progress)
        previous_monotonic = previous.get("monotonic_progress_m")
        if previous_monotonic is None:
            monotonic_progress_m = progress_m
            monotonic_delta_progress_m = 0.0
        else:
            monotonic_progress_m = max(previous_monotonic, progress_m)
            monotonic_delta_progress_m = max(
                0.0,
                monotonic_progress_m - previous_monotonic)
        progress_rate_mps = (
            monotonic_delta_progress_m / dt_value if dt_value > 0.0 else 0.0)
        raw_negative_delta_m = delta_progress_m if delta_progress_m < 0.0 else 0.0
        raw_negative = 1 if delta_progress_m < -1.0e-9 else 0
        deviation_m = projection["deviation_m"]
        deviation_valid = (
            1 if deviation_m <= self.deviation_valid_threshold_m else 0)

        total = max(self.polyline.total_length_m, 1.0e-9)
        self._actor_state[actor_key] = {
            "raw_progress_m": progress_m,
            "monotonic_progress_m": monotonic_progress_m,
            "elapsed_seconds": elapsed if elapsed is not None else previous_time,
            "nearest_segment_index": projection["segment_index"],
            "frame": _finite_optional(frame),
        }
        return {
            "route_progress_available": 1,
            "route_progress_raw_m": progress_m,
            "route_progress_m": progress_m,
            "route_progress_fraction": _clamp(progress_m / total, 0.0, 1.0),
            "route_progress_total_length_m": self.polyline.total_length_m,
            "route_total_length_m": self.polyline.total_length_m,
            "route_progress_delta_m": monotonic_delta_progress_m,
            "route_delta_progress_m": delta_progress_m,
            "route_progress_rate_mps": progress_rate_mps,
            "route_progress_raw_negative_delta_m": raw_negative_delta_m,
            "route_progress_negative_raw": raw_negative,
            "route_progress_raw_deviation_m": deviation_m,
            "route_progress_deviation_valid": deviation_valid,
            "route_deviation_m": deviation_m,
            "route_distance_to_end_m": max(
                0.0,
                self.polyline.total_length_m - monotonic_progress_m),
            "route_progress_monotonic_m": monotonic_progress_m,
            "route_progress_monotonic_fraction": _clamp(
                monotonic_progress_m / total,
                0.0,
                1.0),
            "route_progress_nearest_segment_index": projection["segment_index"],
            "route_progress_projection_segment": projection["segment_index"],
            "route_progress_global_research_used": projection[
                "global_research_used"],
            "route_progress_search_window_start": projection["search_window_start"],
            "route_progress_search_window_end": projection["search_window_end"],
            "route_progress_tracker_status": projection["tracker_status"],
            "route_progress_error": "",
        }

    def _project_with_stabilization(
        self,
        *,
        previous: Mapping[str, Any],
        x: float,
        y: float,
    ) -> Dict[str, Any]:
        segment_count = max(0, len(self.polyline.points) - 1)
        if segment_count <= 0:
            return _projection_with_status(
                {"progress_m": 0.0, "deviation_m": 0.0, "segment_index": -1},
                global_research_used=0,
                search_window_start=-1,
                search_window_end=-1,
                tracker_status="empty_polyline")

        previous_segment = _int_optional(previous.get("nearest_segment_index"))
        if previous_segment is None or previous_segment < 0:
            projection = _project_point_to_polyline_segments(
                self.polyline,
                x,
                y,
                0,
                segment_count - 1)
            return _projection_with_status(
                projection,
                global_research_used=0,
                search_window_start=0,
                search_window_end=segment_count - 1,
                tracker_status="global_initial")

        window_radius = self.search_window_segments
        start = max(0, previous_segment - window_radius)
        end = min(segment_count - 1, previous_segment + window_radius)
        local_projection = _project_point_to_polyline_segments(
            self.polyline,
            x,
            y,
            start,
            end)
        local_bad = (
            local_projection["segment_index"] < 0 or
            local_projection["deviation_m"] > self.global_research_min_deviation_m)
        if local_bad:
            global_projection = _project_point_to_polyline_segments(
                self.polyline,
                x,
                y,
                0,
                segment_count - 1)
            global_better = (
                global_projection["segment_index"] >= 0 and
                (
                    local_projection["segment_index"] < 0 or
                    global_projection["deviation_m"] +
                    self.global_research_improvement_m <
                    local_projection["deviation_m"]))
            if global_better:
                return _projection_with_status(
                    global_projection,
                    global_research_used=1,
                    search_window_start=start,
                    search_window_end=end,
                    tracker_status="global_research")

        return _projection_with_status(
            local_projection,
            global_research_used=0,
            search_window_start=start,
            search_window_end=end,
            tracker_status="local_window")


def load_route_polyline(
    route_xml_path: str,
    route_id: Optional[Any] = None,
    *,
    densify_spacing_m: float = DEFAULT_DENSIFY_SPACING_M,
) -> Tuple[Optional[RoutePolyline], str]:
    path = os.path.abspath(os.path.expanduser(route_xml_path or ""))
    if not path:
        return None, "route_xml_missing"
    if not os.path.isfile(path):
        return None, "route_xml_not_found:%s" % path
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        return None, "route_xml_parse_failed:%s" % error

    routes = list(_iter_route_nodes(root))
    if not routes and _tag_name(root.tag) == "route":
        routes = [root]
    if not routes:
        return None, "route_missing"

    selected, selected_id = _select_route(routes, route_id)
    if selected is None:
        return None, "route_id_missing:%s" % route_id
    points = _route_points(selected)
    if len(points) < 2:
        return None, "route_waypoints_too_few:%s" % selected_id
    points = _densify_points(points, densify_spacing_m)
    cumulative = _cumulative_lengths(points)
    total = cumulative[-1]
    if total <= 0.0:
        return None, "route_length_zero:%s" % selected_id
    return RoutePolyline(
        route_id=selected_id,
        points=tuple(points),
        cumulative_lengths=tuple(cumulative),
        total_length_m=total), ""


def unavailable_route_progress(error: str) -> Dict[str, Any]:
    return {
        "route_progress_available": 0,
        "route_progress_raw_m": 0.0,
        "route_progress_m": 0.0,
        "route_progress_fraction": 0.0,
        "route_progress_total_length_m": 0.0,
        "route_total_length_m": 0.0,
        "route_progress_delta_m": 0.0,
        "route_delta_progress_m": 0.0,
        "route_progress_rate_mps": 0.0,
        "route_progress_raw_negative_delta_m": 0.0,
        "route_progress_negative_raw": 0,
        "route_progress_raw_deviation_m": 0.0,
        "route_progress_deviation_valid": 0,
        "route_deviation_m": 0.0,
        "route_distance_to_end_m": 0.0,
        "route_progress_monotonic_m": 0.0,
        "route_progress_monotonic_fraction": 0.0,
        "route_progress_nearest_segment_index": -1,
        "route_progress_projection_segment": -1,
        "route_progress_global_research_used": 0,
        "route_progress_search_window_start": -1,
        "route_progress_search_window_end": -1,
        "route_progress_tracker_status": "unavailable",
        "route_progress_error": str(error or "route_unavailable"),
    }


def project_point_to_polyline(
    polyline: RoutePolyline,
    x: float,
    y: float,
) -> Dict[str, float]:
    return _project_point_to_polyline_segments(
        polyline,
        x,
        y,
        0,
        len(polyline.points) - 2)


def _project_point_to_polyline_segments(
    polyline: RoutePolyline,
    x: float,
    y: float,
    start_index: int,
    end_index: int,
) -> Dict[str, float]:
    best_distance_sq = float("inf")
    best_progress = 0.0
    best_segment = -1
    best_t = 0.0
    points = polyline.points
    segment_count = max(0, len(points) - 1)
    if segment_count <= 0:
        return {"progress_m": 0.0, "deviation_m": 0.0, "segment_index": -1}
    start = max(0, min(segment_count - 1, int(start_index)))
    end = max(0, min(segment_count - 1, int(end_index)))
    if end < start:
        start, end = end, start
    for index in range(start, end + 1):
        ax, ay, _ = points[index]
        bx, by, _ = points[index + 1]
        abx = bx - ax
        aby = by - ay
        length_sq = abx * abx + aby * aby
        if length_sq <= 1.0e-18:
            continue
        apx = x - ax
        apy = y - ay
        t = _clamp((apx * abx + apy * aby) / length_sq, 0.0, 1.0)
        qx = ax + t * abx
        qy = ay + t * aby
        dx = x - qx
        dy = y - qy
        distance_sq = dx * dx + dy * dy
        if distance_sq < best_distance_sq:
            best_distance_sq = distance_sq
            best_segment = index
            best_t = t
            segment_length = math.sqrt(length_sq)
            best_progress = polyline.cumulative_lengths[index] + t * segment_length
    if best_segment < 0:
        return {"progress_m": 0.0, "deviation_m": 0.0, "segment_index": -1}
    return {
        "progress_m": _clamp(best_progress, 0.0, polyline.total_length_m),
        "deviation_m": math.sqrt(best_distance_sq),
        "segment_index": best_segment,
        "segment_t": best_t,
    }


def _projection_with_status(
    projection: Mapping[str, Any],
    *,
    global_research_used: int,
    search_window_start: int,
    search_window_end: int,
    tracker_status: str,
) -> Dict[str, Any]:
    result = dict(projection)
    result.update({
        "global_research_used": int(global_research_used),
        "search_window_start": int(search_window_start),
        "search_window_end": int(search_window_end),
        "tracker_status": tracker_status,
    })
    return result


def _iter_route_nodes(root: ET.Element) -> Iterable[ET.Element]:
    for node in root.iter():
        if _tag_name(node.tag) == "route":
            yield node


def _select_route(
    routes: Sequence[ET.Element],
    requested_route_id: Optional[Any],
) -> Tuple[Optional[ET.Element], str]:
    route_ids = [_route_id(route, index) for index, route in enumerate(routes)]
    if requested_route_id is None or str(requested_route_id).strip() == "":
        return routes[0], route_ids[0]
    requested_aliases = _route_id_aliases(requested_route_id)
    for route, route_id in zip(routes, route_ids):
        if requested_aliases & _route_id_aliases(route_id):
            return route, route_id
    return None, ""


def _route_id(route: ET.Element, index: int) -> str:
    for name in ("id", "route_id", "name"):
        value = route.attrib.get(name)
        if value is not None and str(value).strip() != "":
            return str(value)
    return str(index)


def _route_id_aliases(value: Any) -> set:
    text = str(value).strip()
    lower = text.lower()
    aliases = {text, lower}
    digit_groups = re.findall(r"\d+", text)
    if digit_groups:
        digits = digit_groups[-1]
        number = int(digits)
        aliases.update({
            str(number),
            "%02d" % number,
            "%03d" % number,
            "route_%02d" % number,
            "routescenario_%02d" % number,
            "routescenario_%d" % number,
        })
    return {alias.lower() for alias in aliases if alias != ""}


def _route_points(route: ET.Element) -> List[Tuple[float, float, float]]:
    points: List[Tuple[float, float, float]] = []
    for node in route.iter():
        if node is route:
            continue
        if _tag_name(node.tag) not in ("waypoint", "position", "point"):
            continue
        if "x" not in node.attrib or "y" not in node.attrib:
            continue
        x = _finite_optional(node.attrib.get("x"))
        y = _finite_optional(node.attrib.get("y"))
        z = _finite_optional(node.attrib.get("z"))
        if x is None or y is None:
            continue
        points.append((x, y, 0.0 if z is None else z))
    return points


def _cumulative_lengths(points: Sequence[Tuple[float, float, float]]) -> List[float]:
    cumulative = [0.0]
    total = 0.0
    for left, right in zip(points, points[1:]):
        total += math.hypot(right[0] - left[0], right[1] - left[1])
        cumulative.append(total)
    return cumulative


def _densify_points(
    points: Sequence[Tuple[float, float, float]],
    target_spacing_m: float,
) -> List[Tuple[float, float, float]]:
    if len(points) < 2:
        return list(points)
    spacing = _finite_optional(target_spacing_m)
    if spacing is None or spacing <= 0.0:
        return list(points)

    densified: List[Tuple[float, float, float]] = [points[0]]
    for start, end in zip(points, points[1:]):
        segment_length = math.hypot(end[0] - start[0], end[1] - start[1])
        if segment_length <= 1.0e-9:
            continue
        steps = max(1, int(math.ceil(segment_length / spacing)))
        for step in range(1, steps + 1):
            t = step / float(steps)
            densified.append((
                start[0] + (end[0] - start[0]) * t,
                start[1] + (end[1] - start[1]) * t,
                start[2] + (end[2] - start[2]) * t,
            ))
    return densified


def _tag_name(tag: str) -> str:
    return str(tag).split("}", 1)[-1].lower()


def _first_position_value(state: Any, explicit: Optional[float], name: str) -> float:
    value = _finite_optional(explicit)
    if value is not None:
        return value
    if isinstance(state, Mapping):
        return _finite_optional(state.get(name)) or 0.0
    return _finite_optional(getattr(state, name, None)) or 0.0


def _first_time_value(
    state: Any,
    explicit: Optional[float],
    name: str,
) -> Optional[float]:
    value = _finite_optional(explicit)
    if value is not None:
        return value
    if isinstance(state, Mapping):
        return _finite_optional(state.get(name))
    return _finite_optional(getattr(state, name, None))


def _finite_optional(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_optional(value: Any) -> Optional[int]:
    number = _finite_optional(value)
    if number is None:
        return None
    return int(number)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

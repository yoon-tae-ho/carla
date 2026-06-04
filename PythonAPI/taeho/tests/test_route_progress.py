import math

from suspension_control.controllers.base import VehicleState
from suspension_control.runtime.route_progress import RouteProgressTracker


def write_route_xml(tmp_path):
    path = tmp_path / "routes.xml"
    path.write_text(
        """<routes>
  <route id="RouteScenario_00" town="Town13">
    <waypoint x="0" y="0" z="0" />
    <waypoint x="10" y="0" z="0" />
    <waypoint x="10" y="10" z="0" />
  </route>
  <route id="01" town="Town13">
    <waypoint x="100" y="0" z="0" />
    <waypoint x="110" y="0" z="0" />
  </route>
</routes>
""",
        encoding="utf-8")
    return path


def write_sparse_route_xml(tmp_path):
    path = tmp_path / "sparse_routes.xml"
    path.write_text(
        """<routes>
  <route id="00" town="Town13">
    <waypoints>
      <position x="0" y="0" z="0" />
      <position x="100" y="0" z="0" />
    </waypoints>
  </route>
</routes>
""",
        encoding="utf-8")
    return path


def test_route_xml_parsing_and_flexible_route_id_matching(tmp_path):
    path = write_route_xml(tmp_path)

    tracker = RouteProgressTracker(str(path), route_id="0")

    assert tracker.available
    assert tracker.polyline.route_id == "RouteScenario_00"
    assert len(tracker.polyline.points) > 3
    assert math.isclose(tracker.polyline.total_length_m, 20.0)


def test_progress_monotonic_even_when_raw_projection_goes_backwards(tmp_path):
    path = write_route_xml(tmp_path)
    tracker = RouteProgressTracker(str(path), route_id="00")

    first = tracker.update(
        VehicleState(x=0.0, y=1.0, elapsed_seconds=0.0, dt=0.1),
        actor_id="hero")
    second = tracker.update(
        VehicleState(x=5.0, y=2.0, elapsed_seconds=1.0, dt=1.0),
        actor_id="hero")
    backward = tracker.update(
        VehicleState(x=3.0, y=0.0, elapsed_seconds=2.0, dt=1.0),
        actor_id="hero")

    assert first["route_progress_available"] == 1
    assert first["route_progress_projection_segment"] == 0
    assert math.isclose(first["route_deviation_m"], 1.0)
    assert math.isclose(first["route_progress_raw_m"], first["route_progress_m"])
    assert math.isclose(second["route_progress_m"], 5.0)
    assert math.isclose(second["route_progress_fraction"], 0.25)
    assert math.isclose(second["route_delta_progress_m"], 5.0)
    assert math.isclose(second["route_progress_delta_m"], 5.0)
    assert math.isclose(second["route_progress_rate_mps"], 5.0)
    assert math.isclose(backward["route_delta_progress_m"], -2.0)
    assert math.isclose(backward["route_progress_delta_m"], 0.0)
    assert math.isclose(backward["route_progress_rate_mps"], 0.0)
    assert math.isclose(backward["route_progress_raw_negative_delta_m"], -2.0)
    assert backward["route_progress_negative_raw"] == 1
    assert math.isclose(backward["route_progress_monotonic_m"], 5.0)
    assert math.isclose(backward["route_progress_monotonic_fraction"], 0.25)


def test_progress_delta_nonnegative(tmp_path):
    path = write_route_xml(tmp_path)
    tracker = RouteProgressTracker(str(path), route_id="00")

    positions = [(0.0, 0.0), (6.0, 0.0), (4.0, 0.0), (8.0, 0.0)]
    deltas = []
    for frame, (x, y) in enumerate(positions):
        result = tracker.update(
            actor_id="hero",
            frame=frame,
            x=x,
            y=y,
            speed=2.0,
            dt=1.0)
        deltas.append(result["route_progress_delta_m"])

    assert all(delta >= 0.0 for delta in deltas)
    assert math.isclose(deltas[2], 0.0)


def test_completion_fraction_reaches_near_one(tmp_path):
    path = write_route_xml(tmp_path)
    tracker = RouteProgressTracker(str(path), route_id="00")

    tracker.update(actor_id="hero", x=0.0, y=0.0, dt=0.1)
    result = tracker.update(actor_id="hero", x=10.0, y=10.0, dt=1.0)

    assert result["route_progress_monotonic_fraction"] >= 0.999
    assert math.isclose(result["route_distance_to_end_m"], 0.0)


def test_sparse_route_is_densified(tmp_path):
    path = write_sparse_route_xml(tmp_path)
    tracker = RouteProgressTracker(str(path), route_id="00")

    assert tracker.available
    assert len(tracker.polyline.points) > 50
    max_segment_length = max(
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(tracker.polyline.points, tracker.polyline.points[1:]))
    assert max_segment_length <= 1.01
    assert math.isclose(tracker.polyline.total_length_m, 100.0)


def test_search_window_recovers_with_global_research(tmp_path):
    path = write_sparse_route_xml(tmp_path)
    tracker = RouteProgressTracker(
        str(path),
        route_id="00",
        search_window_segments=2,
        global_research_min_deviation_m=5.0)

    tracker.update(actor_id="hero", x=0.0, y=0.0, dt=0.1)
    result = tracker.update(actor_id="hero", x=95.0, y=0.0, dt=1.0)

    assert result["route_progress_global_research_used"] == 1
    assert result["route_progress_tracker_status"] == "global_research"
    assert result["route_progress_nearest_segment_index"] >= 90
    assert math.isclose(result["route_progress_monotonic_m"], 95.0)


def test_route_deviation_valid_threshold(tmp_path):
    path = write_route_xml(tmp_path)
    tracker = RouteProgressTracker(
        str(path),
        route_id="00",
        deviation_valid_threshold_m=2.0)

    invalid = tracker.update(actor_id="far", x=5.0, y=3.0, dt=0.1)
    valid = tracker.update(actor_id="near", x=5.0, y=1.0, dt=0.1)

    assert math.isclose(invalid["route_progress_raw_deviation_m"], 3.0)
    assert invalid["route_progress_deviation_valid"] == 0
    assert math.isclose(valid["route_progress_raw_deviation_m"], 1.0)
    assert valid["route_progress_deviation_valid"] == 1


def test_missing_route_xml_returns_unavailable(tmp_path):
    tracker = RouteProgressTracker(str(tmp_path / "missing.xml"), route_id="00")
    result = tracker.update(VehicleState(x=0.0, y=0.0), actor_id="hero")

    assert result["route_progress_available"] == 0
    assert result["route_progress_error"].startswith("route_xml_not_found")
    assert result["route_progress_tracker_status"] == "unavailable"


def test_missing_route_id_returns_unavailable(tmp_path):
    path = write_route_xml(tmp_path)
    tracker = RouteProgressTracker(str(path), route_id="99")
    result = tracker.update(VehicleState(x=0.0, y=0.0), actor_id="hero")

    assert result["route_progress_available"] == 0
    assert "route_id_missing" in result["route_progress_error"]

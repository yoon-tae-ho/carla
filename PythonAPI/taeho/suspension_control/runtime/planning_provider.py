"""Planning-preview providers for suspension controller sidecars."""

from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..controllers.base import PlanningInfo, PlanningPoint, VehicleState


class PlanningInfoProvider:
    """Base interface for polling planning preview at a CARLA frame."""

    def get(
        self,
        frame: int,
        state: VehicleState,
        previous_state: Optional[VehicleState],
    ) -> PlanningInfo:
        raise NotImplementedError


class EmptyPlanningInfoProvider(PlanningInfoProvider):
    """Provider that always returns an empty planning preview."""

    def get(
        self,
        frame: int,
        state: VehicleState,
        previous_state: Optional[VehicleState],
    ) -> PlanningInfo:
        del frame, state, previous_state
        return PlanningInfo.empty()


class JsonlPlanningInfoProvider(PlanningInfoProvider):
    """Incrementally reads one JSON planning-preview message per line."""

    def __init__(
        self,
        path: str,
        max_frame_lag: int = 5,
        allowed_future_frame_lag: int = 0,
        horizon_dt: float = 0.1,
        allow_stale_preview: bool = False,
    ):
        self.path = os.path.abspath(os.path.expanduser(path or ""))
        self.max_frame_lag = max(0, int(max_frame_lag))
        self.allowed_future_frame_lag = max(0, int(allowed_future_frame_lag))
        self.horizon_dt = float(horizon_dt)
        self.allow_stale_preview = bool(allow_stale_preview)
        self.lock = threading.Lock()
        self.offset = 0
        self.messages: Dict[int, PlanningInfo] = {}
        self.malformed_lines = 0
        self.rejected_messages = 0
        self.read_errors = 0
        self.last_error = ""

    def get(
        self,
        frame: int,
        state: VehicleState,
        previous_state: Optional[VehicleState],
    ) -> PlanningInfo:
        del state, previous_state
        with self.lock:
            self._read_new_lines()
            frame = int(frame)
            upper = frame + self.allowed_future_frame_lag
            candidates = [
                info for message_frame, info in self.messages.items()
                if message_frame <= upper
            ]
            if not candidates:
                self._prune_old(frame)
                return self._empty("jsonl_empty")
            selected = max(candidates, key=lambda info: info.frame)
            if (
                selected.frame < frame - self.max_frame_lag and
                not self.allow_stale_preview
            ):
                self._prune_old(frame)
                return self._empty("jsonl_stale", stale_frame=selected.frame)
            self._prune_old(frame)
            return self._with_stats(selected)

    def _read_new_lines(self) -> None:
        if not self.path or not os.path.isfile(self.path):
            return
        try:
            size = os.path.getsize(self.path)
            if size < self.offset:
                self.offset = 0
            with open(self.path) as jsonl_file:
                jsonl_file.seek(self.offset)
                while True:
                    line_offset = jsonl_file.tell()
                    line = jsonl_file.readline()
                    if not line:
                        break
                    if not line.endswith("\n"):
                        jsonl_file.seek(line_offset)
                        break
                    self._read_line(line)
                self.offset = jsonl_file.tell()
        except OSError as error:
            self.read_errors += 1
            self.last_error = str(error)

    def _read_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        try:
            data = json.loads(text)
        except ValueError:
            self.malformed_lines += 1
            return
        try:
            info = planning_info_from_mapping(
                data,
                source="jsonl",
                default_horizon_dt=self.horizon_dt)
        except (TypeError, ValueError):
            self.rejected_messages += 1
            return
        if info.frame < 0:
            self.rejected_messages += 1
            return
        self.messages[int(info.frame)] = info

    def _prune_old(self, frame: int) -> None:
        if self.allow_stale_preview:
            return
        oldest = frame - self.max_frame_lag
        for message_frame in list(self.messages.keys()):
            if message_frame < oldest:
                self.messages.pop(message_frame, None)

    def _empty(self, source: str, **metadata: Any) -> PlanningInfo:
        data = self._stats()
        data.update(metadata)
        return PlanningInfo(source=source, metadata=data)

    def _with_stats(self, info: PlanningInfo) -> PlanningInfo:
        metadata = dict(info.metadata or {})
        metadata.update(self._stats())
        return replace(info, metadata=metadata)

    def _stats(self) -> Dict[str, Any]:
        return {
            "planning_jsonl_path": self.path,
            "planning_jsonl_malformed_lines": self.malformed_lines,
            "planning_jsonl_rejected_messages": self.rejected_messages,
            "planning_jsonl_read_errors": self.read_errors,
            "planning_jsonl_last_error": self.last_error,
        }


class ControlHistoryPlanningInfoProvider(PlanningInfoProvider):
    """Weak fallback preview derived from current controls and state history."""

    def __init__(
        self,
        horizon_steps: int = 8,
        horizon_dt: float = 0.1,
    ):
        self.horizon_steps = max(1, int(horizon_steps))
        self.horizon_dt = float(horizon_dt)

    def get(
        self,
        frame: int,
        state: VehicleState,
        previous_state: Optional[VehicleState],
    ) -> PlanningInfo:
        speed = _float(getattr(state, "speed", 0.0))
        steer_value = _float(getattr(state, "steer", 0.0))
        throttle_value = _float(getattr(state, "throttle", 0.0))
        brake_value = _float(getattr(state, "brake", 0.0))
        curvature = _steer_to_curvature(steer_value)
        ax = _float(getattr(state, "local_ax", 0.0))
        if previous_state is not None:
            dt = max(_float(getattr(state, "dt", self.horizon_dt)), 1.0e-6)
            ax = (
                speed -
                _float(getattr(previous_state, "speed", speed))) / dt
        predicted_ay = speed * speed * curvature
        points = tuple(
            PlanningPoint(
                time_seconds=index * self.horizon_dt,
                x=_float(getattr(state, "x", 0.0)) + speed * index * self.horizon_dt,
                y=_float(getattr(state, "y", 0.0)),
                z=_float(getattr(state, "z", 0.0)),
                speed=speed,
                yaw=_float(getattr(state, "yaw", 0.0)),
                curvature=curvature)
            for index in range(self.horizon_steps))
        repeated_speed = tuple(speed for _ in range(self.horizon_steps))
        return PlanningInfo(
            points=points,
            available=True,
            source="control_history",
            frame=int(frame),
            horizon_dt=self.horizon_dt,
            target_speed=repeated_speed,
            curvature=tuple(curvature for _ in range(self.horizon_steps)),
            steer=tuple(steer_value for _ in range(self.horizon_steps)),
            throttle=tuple(throttle_value for _ in range(self.horizon_steps)),
            brake=tuple(brake_value for _ in range(self.horizon_steps)),
            predicted_ax=tuple(ax for _ in range(self.horizon_steps)),
            predicted_ay=tuple(predicted_ay for _ in range(self.horizon_steps)),
            extra={"fallback": "control_history"})


def make_planning_provider(args: Any) -> PlanningInfoProvider:
    provider = str(getattr(args, "planning_provider", "empty")).strip().lower()
    if provider == "empty":
        return EmptyPlanningInfoProvider()
    if provider == "jsonl":
        return JsonlPlanningInfoProvider(
            path=getattr(args, "planning_preview_jsonl", ""),
            max_frame_lag=getattr(args, "planning_max_frame_lag", 5),
            horizon_dt=getattr(args, "planning_horizon_dt", 0.1))
    if provider == "control_history":
        return ControlHistoryPlanningInfoProvider(
            horizon_dt=getattr(args, "planning_horizon_dt", 0.1))
    raise ValueError("unknown planning provider %s" % provider)


def planning_info_from_mapping(
    data: Mapping[str, Any],
    source: str = "jsonl",
    default_horizon_dt: float = 0.1,
) -> PlanningInfo:
    frame = _int(data.get("frame", -1), -1)
    horizon_dt = _float(data.get("horizon_dt", default_horizon_dt), default_horizon_dt)
    trajectory_xy = _xy_tuple(data.get("trajectory_xy", ()))
    target_speed = _float_tuple(data.get("target_speed", ()))
    trajectory_yaw = _float_tuple(data.get("trajectory_yaw", ()))
    curvature = _float_tuple(data.get("curvature", ()))
    points = _points_tuple(
        data.get("points", ()),
        trajectory_xy,
        trajectory_yaw,
        target_speed,
        curvature,
        horizon_dt)
    metadata = dict(data.get("metadata", {}) or {})
    for key in (
            "exporter_schema_version",
            "exporter_code_version",
            "producer_step",
            "producer_frame",
            "producer_timestamp",
            "write_timestamp",
            "trajectory_source",
            "trajectory_frame",
            "trajectory_point_semantics",
            "horizon_dt_source",
            "speed_semantics",
            "control_semantics",
            "validity_reason"):
        if key in data:
            metadata.setdefault(key, data.get(key))
    return PlanningInfo(
        points=points,
        available=bool(data.get("available", True)),
        source=str(data.get("source", source) or source),
        frame=frame,
        horizon_dt=horizon_dt,
        trajectory_xy=trajectory_xy,
        trajectory_yaw=trajectory_yaw,
        target_speed=target_speed,
        curvature=curvature,
        steer=_float_tuple(data.get("steer", ())),
        throttle=_float_tuple(data.get("throttle", ())),
        brake=_float_tuple(data.get("brake", ())),
        predicted_ax=_float_tuple(data.get("predicted_ax", ())),
        predicted_ay=_float_tuple(data.get("predicted_ay", ())),
        route_deviation=_optional_float(data.get("route_deviation")),
        lane_invasion_count=_int(data.get("lane_invasion_count", 0), 0),
        collision_count=_int(data.get("collision_count", 0), 0),
        metadata=metadata,
        extra=dict(data.get("extra", {}) or {}))


def planning_diagnostics(
    planning: Optional[PlanningInfo],
    current_frame: Optional[int] = None,
) -> Dict[str, Any]:
    if planning is None:
        planning = PlanningInfo.empty()
    try:
        summary = dict(planning.preview_summary())
    except Exception:
        summary = {"planning_available": 0.0}
    summary["planning_source"] = getattr(planning, "source", "")
    summary["planning_frame"] = getattr(planning, "frame", -1)
    if current_frame is not None and getattr(planning, "frame", -1) >= 0:
        summary["planning_age_frames"] = int(current_frame) - int(planning.frame)
    else:
        summary["planning_age_frames"] = ""
    metadata = dict(getattr(planning, "metadata", {}) or {})
    for key in (
        "planning_jsonl_malformed_lines",
        "planning_jsonl_rejected_messages",
        "planning_jsonl_read_errors",
    ):
        summary[key] = metadata.get(key, 0)
    return summary


def _points_tuple(
    raw_points: Any,
    trajectory_xy: Sequence[Tuple[float, float]],
    trajectory_yaw: Sequence[float],
    target_speed: Sequence[float],
    curvature: Sequence[float],
    horizon_dt: float,
) -> Tuple[PlanningPoint, ...]:
    points = []
    if raw_points:
        for index, value in enumerate(raw_points):
            if isinstance(value, Mapping):
                points.append(PlanningPoint(
                    time_seconds=_float(
                        value.get("time_seconds", index * horizon_dt)),
                    x=_float(value.get("x", 0.0)),
                    y=_float(value.get("y", 0.0)),
                    z=_float(value.get("z", 0.0)),
                    speed=_float(value.get("speed", _at(target_speed, index))),
                    yaw=_float(value.get("yaw", _at(trajectory_yaw, index))),
                    curvature=_float(value.get(
                        "curvature",
                        _at(curvature, index)))))
        return tuple(points)

    horizon = max(
        len(trajectory_xy),
        len(trajectory_yaw),
        len(target_speed),
        len(curvature))
    for index in range(horizon):
        xy = trajectory_xy[index] if index < len(trajectory_xy) else (0.0, 0.0)
        points.append(PlanningPoint(
            time_seconds=index * horizon_dt,
            x=xy[0],
            y=xy[1],
            z=0.0,
            speed=_at(target_speed, index),
            yaw=_at(trajectory_yaw, index),
            curvature=_at(curvature, index)))
    return tuple(points)


def _xy_tuple(values: Any) -> Tuple[Tuple[float, float], ...]:
    result = []
    for value in values or ():
        try:
            x_value, y_value = value[0], value[1]
        except (TypeError, IndexError):
            continue
        result.append((_float(x_value), _float(y_value)))
    return tuple(result)


def _float_tuple(values: Any) -> Tuple[float, ...]:
    return tuple(_float(value) for value in (values or ()))


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return _float(value)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _at(values: Sequence[float], index: int, default: float = 0.0) -> float:
    return values[index] if index < len(values) else default


def _steer_to_curvature(steer_value: float) -> float:
    return max(-0.10, min(0.10, steer_value * 0.05))

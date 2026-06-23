"""CARLA adapter functions for suspension controller outputs."""

from __future__ import annotations

import glob
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

from ..controllers.base import SuspensionCommand, VehicleState


SUSPENSION_FIELDS = (
    "spring_strength",
    "spring_damper_rate",
    "max_compression",
    "max_droop",
    "sprung_mass",
)


def pythonapi_root_from_here() -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."))


def add_carla_to_path(pythonapi_root: Optional[str] = None) -> tuple:
    """Add local CARLA build/dist artifacts to `sys.path`.

    This mirrors the existing `PythonAPI/taeho` experiment scripts while keeping
    path handling centralized for the controller package.
    """

    root = pythonapi_root or pythonapi_root_from_here()
    platform = "win-amd64" if os.name == "nt" else "linux-x86_64"
    build_candidates = glob.glob(os.path.join(root, "carla", "build", "lib.*"))
    py_build_tag = "cpython-%d%d" % sys.version_info[:2]
    preferred_build_candidates = [
        path for path in build_candidates
        if py_build_tag in os.path.basename(path)
    ]
    other_build_candidates = [
        path for path in build_candidates
        if path not in preferred_build_candidates
    ]
    candidates = sorted(preferred_build_candidates) + sorted(other_build_candidates)

    if os.environ.get("CARLA_USE_EGG", "0") == "1":
        candidates += sorted(glob.glob(os.path.join(
            root,
            "carla",
            "dist",
            "carla-*%d.%d-%s.egg" % (
                sys.version_info.major,
                sys.version_info.minor,
                platform))))

    added = []
    for path in reversed(candidates):
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
        added.append(path)
    return tuple(added)


def import_carla(pythonapi_root: Optional[str] = None):
    add_carla_to_path(pythonapi_root)
    import carla
    return carla


def vector_norm(vector: Any) -> float:
    return math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)


def to_local_xy(world_x: float, world_y: float, yaw_degrees: float):
    yaw = math.radians(yaw_degrees)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    local_x = cos_yaw * world_x + sin_yaw * world_y
    local_y = -sin_yaw * world_x + cos_yaw * world_y
    return local_x, local_y


def _control_value(control: Any, name: str) -> float:
    if control is None:
        return 0.0
    return float(getattr(control, name, 0.0))


def read_vehicle_state(
    world: Any,
    vehicle: Any,
    step: int = 0,
    previous_state: Optional[VehicleState] = None,
    control: Any = None,
) -> VehicleState:
    snapshot = world.get_snapshot()
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    angular_velocity = vehicle.get_angular_velocity()

    local_vx, local_vy = to_local_xy(
        velocity.x,
        velocity.y,
        transform.rotation.yaw)
    local_ax, local_ay = to_local_xy(
        acceleration.x,
        acceleration.y,
        transform.rotation.yaw)

    elapsed = float(snapshot.timestamp.elapsed_seconds)
    if previous_state is None:
        dt = 0.0
    else:
        dt = max(0.0, elapsed - previous_state.elapsed_seconds)

    state = VehicleState(
        step=int(step),
        frame=int(snapshot.frame),
        elapsed_seconds=elapsed,
        dt=dt,
        x=float(transform.location.x),
        y=float(transform.location.y),
        z=float(transform.location.z),
        vx=float(velocity.x),
        vy=float(velocity.y),
        vz=float(velocity.z),
        speed=vector_norm(velocity),
        local_vx=float(local_vx),
        local_vy=float(local_vy),
        ax=float(acceleration.x),
        ay=float(acceleration.y),
        az=float(acceleration.z),
        local_ax=float(local_ax),
        local_ay=float(local_ay),
        roll=float(transform.rotation.roll),
        pitch=float(transform.rotation.pitch),
        yaw=float(transform.rotation.yaw),
        roll_rate=float(angular_velocity.x),
        pitch_rate=float(angular_velocity.y),
        yaw_rate=float(angular_velocity.z),
        throttle=_control_value(control, "throttle"),
        brake=_control_value(control, "brake"),
        steer=_control_value(control, "steer"),
    )
    finite_or_raise(state.as_dict().items(), "vehicle state")
    return state


def finite_or_raise(values, label: str) -> None:
    for name, value in values:
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise RuntimeError("%s has non-finite %s=%r" % (label, name, value))


def native_spring_strength_by_wheel(native_control: Any) -> Tuple[float, ...]:
    return tuple(
        float(getattr(wheel, "spring_strength"))
        for wheel in tuple(getattr(native_control, "wheels", ()) or ()))


def native_damper_rate_by_wheel(native_control: Any) -> Tuple[float, ...]:
    return tuple(
        float(getattr(wheel, "spring_damper_rate"))
        for wheel in tuple(getattr(native_control, "wheels", ()) or ()))


def suspension_state_invalid_reason(
    state: Any,
    expected_wheels: Optional[int] = None,
    require_velocity: bool = True,
    require_contact: bool = True,
) -> str:
    if state is None:
        return "suspension_state_missing"
    if not bool(getattr(state, "state_valid", False)):
        return str(getattr(state, "failure_reason", "") or "state_valid_false")
    wheels = tuple(getattr(state, "wheels", ()) or ())
    if expected_wheels is not None and len(wheels) != int(expected_wheels):
        return "wheel_count_%d_expected_%d" % (len(wheels), int(expected_wheels))
    if not wheels:
        return "wheel_state_empty"
    if require_velocity and not bool(getattr(state, "velocity_valid", False)):
        return "state_velocity_valid_false"
    for index, wheel in enumerate(wheels):
        if not bool(getattr(wheel, "field_valid", False)):
            return "wheel_%d_field_valid_false" % index
        if require_velocity and not bool(getattr(wheel, "velocity_valid", False)):
            return "wheel_%d_velocity_valid_false" % index
        if require_contact and not bool(getattr(wheel, "contact_valid", False)):
            return "wheel_%d_contact_valid_false" % index
        if require_contact and bool(getattr(wheel, "wheel_in_air", False)):
            return "wheel_%d_in_air" % index
        for field in (
                "raw_suspension_offset_m",
                "suspension_compression_m",
                "suspension_travel_m"):
            value = getattr(wheel, field, 0.0)
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                return "wheel_%d_%s_nonfinite" % (index, field)
        if require_velocity:
            value = getattr(wheel, "suspension_velocity_mps", 0.0)
            try:
                finite = math.isfinite(float(value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                return "wheel_%d_suspension_velocity_mps_nonfinite" % index
    return ""


def read_suspension_state(
    vehicle: Any,
    expected_wheels: Optional[int] = None,
    require_velocity: bool = True,
    require_contact: bool = True,
) -> Tuple[Any, bool, str]:
    if not hasattr(vehicle, "get_suspension_state"):
        return None, False, "get_suspension_state_missing"
    try:
        state = vehicle.get_suspension_state()
    except Exception as error:
        return None, False, "get_suspension_state_%s" % error.__class__.__name__
    reason = suspension_state_invalid_reason(
        state,
        expected_wheels=expected_wheels,
        require_velocity=require_velocity,
        require_contact=require_contact)
    return state, reason == "", reason


def validate_suspension_control(control: Any, expected_wheels: int = 4) -> None:
    if len(control.wheels) != expected_wheels:
        raise RuntimeError(
            "expected %d suspension wheels, got %d" %
            (expected_wheels, len(control.wheels)))

    for index, wheel in enumerate(control.wheels):
        for field in SUSPENSION_FIELDS:
            value = getattr(wheel, field)
            if not math.isfinite(value):
                raise RuntimeError(
                    "wheel %d %s is not finite: %r" %
                    (index, field, value))
            if value <= 0.0:
                raise RuntimeError(
                    "wheel %d %s must be positive, got %r" %
                    (index, field, value))


def make_scaled_suspension_control(
    native_control: Any,
    command: SuspensionCommand,
    carla_module: Any = None,
):
    command.validate(expected_wheels=len(native_control.wheels))
    carla = carla_module or import_carla()
    control = carla.SuspensionPhysicsControl()
    wheels = []
    for native_wheel, scale in zip(native_control.wheels, command.wheels):
        wheels.append(carla.WheelSuspensionPhysicsControl(
            spring_strength=native_wheel.spring_strength * scale.spring_scale,
            spring_damper_rate=(
                native_wheel.spring_damper_rate * scale.damper_scale),
            max_compression=native_wheel.max_compression,
            max_droop=native_wheel.max_droop,
            sprung_mass=native_wheel.sprung_mass,
        ))
    control.wheels = wheels
    validate_suspension_control(control, expected_wheels=len(native_control.wheels))
    return control


def read_suspension_scale_summary(native_control: Any, actual_control: Any) -> Dict[str, Any]:
    validate_suspension_control(native_control, expected_wheels=len(native_control.wheels))
    validate_suspension_control(actual_control, expected_wheels=len(native_control.wheels))

    spring_scales = []
    damper_scales = []
    for native_wheel, actual_wheel in zip(native_control.wheels, actual_control.wheels):
        spring_scales.append(
            actual_wheel.spring_strength / native_wheel.spring_strength)
        damper_scales.append(
            actual_wheel.spring_damper_rate / native_wheel.spring_damper_rate)

    return {
        "spring_scales": tuple(spring_scales),
        "damper_scales": tuple(damper_scales),
        "mean_spring_scale": sum(spring_scales) / float(len(spring_scales)),
        "mean_damper_scale": sum(damper_scales) / float(len(damper_scales)),
        "min_spring_scale": min(spring_scales),
        "max_spring_scale": max(spring_scales),
        "min_damper_scale": min(damper_scales),
        "max_damper_scale": max(damper_scales),
    }


def assert_scale_match(
    expected_command: SuspensionCommand,
    native_control: Any,
    actual_control: Any,
    tolerance: float = 1.0e-4,
) -> None:
    summary = read_suspension_scale_summary(native_control, actual_control)
    for index, wheel in enumerate(expected_command.wheels):
        spring_error = abs(summary["spring_scales"][index] - wheel.spring_scale)
        damper_error = abs(summary["damper_scales"][index] - wheel.damper_scale)
        if spring_error > tolerance:
            raise RuntimeError(
                "wheel %d spring scale mismatch: expected %0.9g got %0.9g" %
                (index, wheel.spring_scale, summary["spring_scales"][index]))
        if damper_error > tolerance:
            raise RuntimeError(
                "wheel %d damper scale mismatch: expected %0.9g got %0.9g" %
                (index, wheel.damper_scale, summary["damper_scales"][index]))


def apply_suspension_command(
    vehicle: Any,
    native_control: Any,
    command: SuspensionCommand,
    verify_readback: bool = False,
    readback_tolerance: float = 1.0e-4,
    carla_module: Any = None,
) -> Dict[str, Any]:
    control = make_scaled_suspension_control(
        native_control,
        command,
        carla_module=carla_module)
    vehicle.apply_suspension_physics_control(control)
    result = {"applied_control": control}
    if verify_readback:
        readback = vehicle.get_suspension_physics_control()
        assert_scale_match(
            command,
            native_control,
            readback,
            tolerance=readback_tolerance)
        result["readback"] = readback
        result["readback_summary"] = read_suspension_scale_summary(
            native_control,
            readback)
    return result

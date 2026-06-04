from pathlib import Path

from suspension_control.controllers.base import SuspensionCommand
from suspension_control.rl.carla_route_backend import LiveRouteProcessBackend


class FakeProcess:
    pid = 4242

    def __init__(self):
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if not self.terminated else 0

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        del timeout
        self.terminated = True
        return 0

    def kill(self):
        self.killed = True
        self.terminated = True


class FakeWheel:
    def __init__(self):
        self.spring_strength = 100.0
        self.spring_damper_rate = 10.0
        self.max_compression = 0.1
        self.max_droop = 0.1
        self.sprung_mass = 250.0


class FakeSuspensionControl:
    def __init__(self):
        self.wheels = [FakeWheel(), FakeWheel(), FakeWheel(), FakeWheel()]


class FakeCarlaModule:
    class WheelSuspensionPhysicsControl:
        def __init__(self, **kwargs):
            self.spring_strength = kwargs["spring_strength"]
            self.spring_damper_rate = kwargs["spring_damper_rate"]
            self.max_compression = kwargs["max_compression"]
            self.max_droop = kwargs["max_droop"]
            self.sprung_mass = kwargs["sprung_mass"]

    class SuspensionPhysicsControl:
        def __init__(self):
            self.wheels = []

    def __init__(self, world):
        self.world = world

    def Client(self, host, port):
        return FakeClient(self.world, host, port)


class FakeClient:
    def __init__(self, world, host, port):
        self.world = world
        self.host = host
        self.port = port
        self.timeout = None

    def set_timeout(self, timeout):
        self.timeout = timeout

    def get_world(self):
        return self.world


class FakeActors(list):
    def filter(self, pattern):
        assert pattern == "vehicle.*"
        return list(self)


class FakeVector:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class FakeRotation:
    roll = 0.0
    pitch = 0.0
    yaw = 0.0


class FakeTransform:
    def __init__(self, x):
        self.location = FakeVector(x=x, y=0.0, z=0.0)
        self.rotation = FakeRotation()


class FakeTimestamp:
    def __init__(self, elapsed_seconds):
        self.elapsed_seconds = elapsed_seconds


class FakeSnapshot:
    def __init__(self, frame):
        self.frame = frame
        self.timestamp = FakeTimestamp(frame * 0.05)


class FakeControl:
    throttle = 0.2
    brake = 0.0
    steer = 0.0


class FakeVehicle:
    id = 7
    type_id = "vehicle.fake"
    attributes = {"role_name": "hero"}

    def __init__(self, world):
        self.world = world
        self.suspension = FakeSuspensionControl()
        self.applied_native_restore = False

    def get_suspension_physics_control(self):
        return self.suspension

    def apply_suspension_physics_control(self, control):
        self.suspension = control
        if isinstance(control, FakeSuspensionControl):
            self.applied_native_restore = True

    def get_transform(self):
        return FakeTransform(x=float(self.world.frame))

    def get_velocity(self):
        return FakeVector(x=8.0, y=0.0, z=0.0)

    def get_acceleration(self):
        return FakeVector(x=0.0, y=0.1, z=0.2)

    def get_angular_velocity(self):
        return FakeVector(x=0.0, y=0.0, z=0.0)

    def get_control(self):
        return FakeControl()


class FakeWorld:
    def __init__(self):
        self.frame = 1
        self.wait_for_tick_calls = 0
        self.vehicle = FakeVehicle(self)

    def get_actors(self):
        return FakeActors([self.vehicle])

    def wait_for_tick(self, timeout):
        assert timeout >= 0.0
        self.wait_for_tick_calls += 1
        self.frame += 1
        return FakeSnapshot(self.frame)

    def get_snapshot(self):
        return FakeSnapshot(self.frame)

    def tick(self):
        raise AssertionError("external-clock backend must not call world.tick()")


class HeroAppearsAfterTimeoutWorld(FakeWorld):
    def __init__(self):
        super().__init__()
        self._actors = FakeActors([])

    def get_actors(self):
        if self.wait_for_tick_calls >= 2:
            return FakeActors([self.vehicle])
        return self._actors

    def wait_for_tick(self, timeout):
        self.wait_for_tick_calls += 1
        if self.wait_for_tick_calls <= 2:
            raise RuntimeError("time-out of 2000ms while waiting for the simulator")
        self.frame += 1
        return FakeSnapshot(self.frame)


def make_backend(tmp_path, **kwargs):
    route_script = tmp_path / "run_tfpp_debug_route.sh"
    route_script.write_text("#!/usr/bin/env bash\n")
    world = kwargs.pop("world", FakeWorld())
    hero_timeout_seconds = kwargs.pop("hero_timeout_seconds", 1.0)
    route_wait_timeout_seconds = kwargs.pop("route_wait_timeout_seconds", 0.01)
    processes = []

    def popen_factory(command, **popen_kwargs):
        process = FakeProcess()
        processes.append((process, command, popen_kwargs))
        return process

    backend = LiveRouteProcessBackend(
        route_script=str(route_script),
        routes=str(tmp_path / "routes.xml"),
        output_dir=str(tmp_path / "out"),
        carla_module=FakeCarlaModule(world),
        required_route_modules=(),
        popen_factory=popen_factory,
        hero_timeout_seconds=hero_timeout_seconds,
        route_wait_timeout_seconds=route_wait_timeout_seconds,
        **kwargs)
    return backend, world, processes


def test_live_backend_route_command_and_env_are_assembled(tmp_path):
    backend, _, processes = make_backend(tmp_path)

    assert backend.unavailable_message == ""
    env = backend.build_route_env(seed=100, route_id="00")
    assert backend.build_route_command() == ("bash", backend.route_script)
    assert env["PORT"] == "2000"
    assert env["ROUTES_SUBSET"] == "00"
    assert env["TRAFFIC_MANAGER_SEED"] == "100"
    assert env["OUT_DIR"] == backend.route_output_dir

    process = backend.start_route_process(seed=100, route_id="00")
    assert process is processes[0][0]
    assert processes[0][1] == ["bash", backend.route_script]
    assert processes[0][2]["env"]["OUT_DIR"] == backend.route_output_dir
    backend.close()


def test_live_backend_missing_dependency_fails_without_fake_fallback(tmp_path):
    route_script = tmp_path / "run_tfpp_debug_route.sh"
    route_script.write_text("#!/usr/bin/env bash\n")
    backend = LiveRouteProcessBackend(
        route_script=str(route_script),
        output_dir=str(tmp_path / "out"),
        carla_importer=lambda: (_ for _ in ()).throw(
            ImportError("carla missing")),
        required_route_modules=())

    assert "LiveRouteProcessBackend dependencies are unavailable" in (
        backend.unavailable_message)
    assert "carla import failed" in backend.unavailable_message
    try:
        backend.reset(seed=100, route_id="00")
    except RuntimeError as error:
        assert "must not fall back to synthetic training" in str(error)
    else:
        raise AssertionError("missing live dependency should fail")


def test_live_backend_reset_apply_tick_and_close_are_external_clock(tmp_path):
    backend, world, processes = make_backend(tmp_path)

    reset = backend.reset(seed=100, route_id="00")
    assert reset.info["backend"] == "live"
    assert reset.info["real_backend_used"] == 1
    assert reset.info["fake_backend_used"] == 0
    assert reset.info["route_process_started"] == 1
    assert reset.info["hero_attached"] == 1
    assert reset.task_info

    apply_info = backend.apply_suspension(
        SuspensionCommand.uniform(damper_scale=1.02),
        verify=True)
    assert apply_info["backend_apply_suspension"] == 1
    assert apply_info["readback_mean_damper_scale"] != ""

    step = backend.tick()
    assert world.wait_for_tick_calls == 1
    assert step.info["route_process_alive"] == 1
    assert step.terminated is False
    assert step.truncated is False
    assert step.task_info

    backend.close()
    assert processes[0][0].terminated is True
    assert world.vehicle.applied_native_restore is True


def test_live_backend_hero_wait_tolerates_initial_tick_timeouts(tmp_path):
    world = HeroAppearsAfterTimeoutWorld()
    backend, _, _ = make_backend(
        tmp_path,
        world=world,
        hero_timeout_seconds=2.0,
        connect_retry_seconds=0.0)

    reset = backend.reset(seed=100, route_id="00")

    assert reset.info["hero_attached"] == 1
    assert world.wait_for_tick_calls >= 2
    backend.close()

import json
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


class SequencedWorldClient(FakeClient):
    def __init__(self, worlds, host, port):
        self.worlds = list(worlds)
        self.host = host
        self.port = port
        self.timeout = None
        self.get_world_calls = 0

    def get_world(self):
        index = min(self.get_world_calls, len(self.worlds) - 1)
        self.get_world_calls += 1
        return self.worlds[index]


class SequencedCarlaModule(FakeCarlaModule):
    def __init__(self, worlds):
        self.worlds = tuple(worlds)
        self.clients = []

    def Client(self, host, port):
        client = SequencedWorldClient(self.worlds, host, port)
        self.clients.append(client)
        return client


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


class FakeSettings:
    def __init__(self, synchronous_mode=False, fixed_delta_seconds=None):
        self.synchronous_mode = synchronous_mode
        self.fixed_delta_seconds = fixed_delta_seconds


class FakeVehicle:
    id = 7
    type_id = "vehicle.fake"
    attributes = {"role_name": "hero"}

    def __init__(self, world):
        self.world = world
        self.suspension = FakeSuspensionControl()
        self.applied_native_restore = False
        self.is_alive = True
        self.suspension_read_count = 0
        self.suspension_apply_count = 0
        self.raise_actor_not_found_on_suspension_read = False

    def get_suspension_physics_control(self):
        self.suspension_read_count += 1
        if (
                not self.is_alive or
                self.raise_actor_not_found_on_suspension_read):
            raise RuntimeError(
                "Responding error from function "
                "get_suspension_physics_control: Actor could not be "
                "found in the registry.  Actor Id: %s" % self.id)
        return self.suspension

    def apply_suspension_physics_control(self, control):
        self.suspension_apply_count += 1
        if not self.is_alive:
            raise RuntimeError(
                "Responding error from function "
                "apply_suspension_physics_control: Actor could not be "
                "found in the registry.  Actor Id: %s" % self.id)
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
        self.settings = FakeSettings()
        self.applied_settings = []

    def get_actors(self):
        return FakeActors([self.vehicle])

    def get_settings(self):
        return self.settings

    def apply_settings(self, settings):
        self.settings = settings
        self.applied_settings.append(settings)

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


class NoVehicleWorld(FakeWorld):
    def get_actors(self):
        return FakeActors([])


class DeadHeroThenUsableWorld(FakeWorld):
    def __init__(self):
        super().__init__()
        self.dead_vehicle = FakeVehicle(self)
        self.dead_vehicle.id = 70
        self.dead_vehicle.is_alive = False
        self.vehicle.id = 71

    def get_actors(self):
        if self.wait_for_tick_calls >= 2:
            return FakeActors([self.vehicle])
        return FakeActors([self.dead_vehicle])


class SuspensionReadFailsThenUsableWorld(FakeWorld):
    def __init__(self):
        super().__init__()
        self.stale_vehicle = FakeVehicle(self)
        self.stale_vehicle.id = 80
        self.stale_vehicle.raise_actor_not_found_on_suspension_read = True
        self.vehicle.id = 81

    def get_actors(self):
        if self.wait_for_tick_calls >= 2:
            return FakeActors([self.vehicle])
        return FakeActors([self.stale_vehicle])


def make_backend(tmp_path, **kwargs):
    route_script = tmp_path / "run_tfpp_debug_route.sh"
    route_script.write_text("#!/usr/bin/env bash\n")
    world = kwargs.pop("world", FakeWorld())
    carla_module = kwargs.pop("carla_module", FakeCarlaModule(world))
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
        carla_module=carla_module,
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
    assert reset.info["hero_actor_id"] == "7"
    assert reset.info["hero_actor_alive_last"] == 1
    assert reset.info["hero_destroy_detected"] == 0
    assert reset.task_info

    apply_info = backend.apply_suspension(
        SuspensionCommand.uniform(damper_scale=1.02),
        verify=True)
    assert apply_info["backend_apply_suspension"] == 1
    assert apply_info["readback_mean_damper_scale"] != ""

    step = backend.tick()
    assert world.wait_for_tick_calls == 1
    assert step.info["route_process_alive"] == 1
    assert step.info["hero_actor_id"] == "7"
    assert step.info["stale_actor_api_call_count"] == 0
    assert step.terminated is False
    assert step.truncated is False
    assert step.task_info

    backend.close()
    assert processes[0][0].terminated is True
    assert world.vehicle.applied_native_restore is True


def test_live_backend_restores_stale_sync_world_before_route_start(tmp_path):
    world = FakeWorld()
    world.settings.synchronous_mode = True
    world.settings.fixed_delta_seconds = 0.05
    backend, _, processes = make_backend(tmp_path, world=world)

    reset = backend.reset(seed=100, route_id="00")

    assert reset.info["route_process_started"] == 1
    assert processes
    assert world.applied_settings
    assert world.settings.synchronous_mode is False
    assert world.settings.fixed_delta_seconds is None
    assert world.wait_for_tick_calls >= 1
    backend.close()


def test_live_backend_close_restores_async_after_early_route_stop(tmp_path):
    backend, world, processes = make_backend(tmp_path)
    backend.reset(seed=100, route_id="00")
    world.settings.synchronous_mode = True
    world.settings.fixed_delta_seconds = 0.05

    backend.close()

    assert processes[0][0].terminated is True
    assert world.settings.synchronous_mode is False
    assert world.settings.fixed_delta_seconds is None
    assert world.applied_settings
    assert world.wait_for_tick_calls >= 1


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


def test_live_backend_hero_wait_refreshes_world_after_route_load(tmp_path):
    stale_world = NoVehicleWorld()
    route_world = FakeWorld()
    carla_module = SequencedCarlaModule([stale_world, route_world])
    backend, _, _ = make_backend(
        tmp_path,
        world=stale_world,
        carla_module=carla_module,
        hero_timeout_seconds=2.0)

    reset = backend.reset(seed=100, route_id="00")

    assert reset.info["hero_attached"] == 1
    assert backend.world is route_world
    assert len(carla_module.clients) >= 2
    assert carla_module.clients[-1].get_world_calls >= 2
    backend.close()


def test_live_backend_hero_wait_ignores_dead_actor_candidates(tmp_path):
    world = DeadHeroThenUsableWorld()
    backend, _, _ = make_backend(
        tmp_path,
        world=world,
        hero_timeout_seconds=2.0)

    reset = backend.reset(seed=100, route_id="00")

    assert reset.info["hero_attached"] == 1
    assert reset.info["hero_actor_id"] == "71"
    assert world.wait_for_tick_calls >= 2
    assert world.dead_vehicle.suspension_read_count == 0
    backend.close()


def test_live_backend_hero_wait_requires_suspension_read_success(tmp_path):
    world = SuspensionReadFailsThenUsableWorld()
    backend, _, _ = make_backend(
        tmp_path,
        world=world,
        hero_timeout_seconds=2.0)

    reset = backend.reset(seed=100, route_id="00")

    assert reset.info["hero_attached"] == 1
    assert reset.info["hero_actor_id"] == "81"
    assert world.wait_for_tick_calls >= 2
    assert world.stale_vehicle.suspension_read_count >= 1
    assert world.vehicle.suspension_read_count == 1
    backend.close()


def test_live_backend_actor_disappears_before_suspension_read_is_guarded(tmp_path):
    backend, world, _ = make_backend(tmp_path)
    backend.reset(seed=100, route_id="00")
    before_reads = world.vehicle.suspension_read_count
    world.vehicle.is_alive = False

    try:
        backend.get_current_suspension()
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("missing hero actor should fail clearly")

    assert "live route backend suspension API unavailable" in message
    assert "api=get_suspension_physics_control" in message
    assert "terminal_reason=hero_actor_unavailable" in message
    assert world.vehicle.suspension_read_count == before_reads
    info = backend.backend_info()
    assert info["hero_destroy_detected"] == 1
    assert info["hero_actor_alive_last"] == 0
    assert info["stale_actor_api_call_count"] == 1
    assert info["actor_not_found_error_count"] == 0
    assert info["last_suspension_api_call_actor_id"] == "7"
    backend.close()


def test_live_backend_actor_not_found_exception_records_context(tmp_path):
    backend, world, _ = make_backend(tmp_path)
    backend.reset(seed=100, route_id="00")
    world.vehicle.raise_actor_not_found_on_suspension_read = True

    try:
        backend.get_current_suspension()
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("actor-not-found suspension read should fail")

    assert "Actor could not be found in the registry" in message
    info = backend.backend_info()
    assert info["actor_not_found_error_count"] == 1
    assert info["stale_actor_api_call_count"] == 1
    assert info["hero_destroy_detected"] == 1
    assert info["last_suspension_api_error"]
    backend.close()


def test_live_backend_result_file_without_terminal_records_is_not_route_done(tmp_path):
    backend, world, _ = make_backend(tmp_path)
    backend.reset(seed=100, route_id="00")
    world.vehicle.is_alive = False
    Path(backend.route_result_path).parent.mkdir(parents=True, exist_ok=True)
    Path(backend.route_result_path).write_text(json.dumps({
        "entry_status": "Started",
        "_checkpoint": {
            "progress": [0, 1],
            "records": [],
        },
    }))

    try:
        backend.get_current_suspension()
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("missing hero before route ready should fail clearly")

    assert "terminal_reason=hero_actor_unavailable_before_route_ready" in message
    assert "episode_end_reason=hero_actor_unavailable_before_route_ready" in message
    assert "route_checkpoint_progress=0/1" in message
    info = backend.backend_info()
    assert info["route_finished_detected"] == 0
    assert info["terminal_reason"] == "hero_actor_unavailable_before_route_ready"
    backend.close()


def test_live_backend_route_result_and_missing_actor_records_terminal_reason(tmp_path):
    backend, world, processes = make_backend(tmp_path)
    backend.reset(seed=100, route_id="00")
    processes[0][0].terminated = True
    world.vehicle.is_alive = False
    Path(backend.route_result_path).parent.mkdir(parents=True, exist_ok=True)
    Path(backend.route_result_path).write_text(json.dumps({
        "entry_status": "Started",
        "_checkpoint": {
            "progress": [1, 1],
            "records": [{
                "status": "Perfect",
                "scores": {
                    "score_route": 100,
                    "score_composed": 100.0,
                },
                "meta": {
                    "duration_game": 312.2,
                },
            }],
        },
    }))

    try:
        backend.get_current_suspension()
    except RuntimeError as error:
        message = str(error)
    else:
        raise AssertionError("missing actor after route terminal should fail")

    assert "terminal_reason=route_finished_or_hero_destroyed" in message
    assert "episode_end_reason=hero_destroyed_after_route_terminal" in message
    assert "route_record_status=Perfect" in message
    info = backend.backend_info()
    assert info["route_finished_detected"] == 1
    assert info["route_process_returncode"] == 0
    assert info["route_record_status"] == "Perfect"
    assert info["route_record_score_route"] == 100
    assert info["route_checkpoint_progress"] == "1/1"
    assert info["terminal_reason"] == "route_finished_or_hero_destroyed"
    assert info["episode_end_reason"] == "hero_destroyed_after_route_terminal"
    backend.close()

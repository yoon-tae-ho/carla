#!/usr/bin/env python

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
# de Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Synchronous Chrono suspension-control experiment skeleton.

This script is intentionally small and self-contained: it enables Chrono
physics for one ego vehicle, asks a planner-output provider for per-tick
vehicle control/trajectory data, maps that output to FL/FR/RL/RR suspension
commands, applies the Phase 2 Python API, and writes one JSON object per tick.
"""

from __future__ import print_function

import argparse
import datetime
import glob
import json
import logging
import math
import os
import random
import sys
import weakref


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    sys.path.append(glob.glob(os.path.join(
        SCRIPT_DIR,
        '..',
        'carla',
        'dist',
        'carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64')))[0])
except IndexError:
    pass

try:
    sys.path.append(os.path.join(SCRIPT_DIR, '..', 'carla'))
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla


# ==============================================================================
# -- helpers -------------------------------------------------------------------
# ==============================================================================


WHEEL_ORDER = ('FL', 'FR', 'RL', 'RR')


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def vector_length(vector):
    return math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)


def vector_to_dict(vector):
    return {
        'x': float(vector.x),
        'y': float(vector.y),
        'z': float(vector.z),
    }


def rotation_to_dict(rotation):
    return {
        'pitch': float(rotation.pitch),
        'yaw': float(rotation.yaw),
        'roll': float(rotation.roll),
    }


def transform_to_dict(transform):
    return {
        'location': vector_to_dict(transform.location),
        'rotation': rotation_to_dict(transform.rotation),
    }


def control_to_dict(control):
    return {
        'throttle': float(control.throttle),
        'steer': float(control.steer),
        'brake': float(control.brake),
        'hand_brake': bool(control.hand_brake),
        'reverse': bool(control.reverse),
        'manual_gear_shift': bool(control.manual_gear_shift),
        'gear': int(control.gear),
    }


def waypoint_to_dict(waypoint):
    transform = waypoint.transform
    return {
        'location': vector_to_dict(transform.location),
        'yaw': float(transform.rotation.yaw),
        'road_id': int(waypoint.road_id),
        'lane_id': int(waypoint.lane_id),
        's': float(waypoint.s),
    }


def timestamp_to_dict(timestamp):
    return {
        'elapsed_seconds': float(timestamp.elapsed_seconds),
        'delta_seconds': float(timestamp.delta_seconds),
        'platform_timestamp': float(timestamp.platform_timestamp),
    }


def default_chrono_base_path():
    return os.path.abspath(os.path.join(
        SCRIPT_DIR,
        '..',
        '..',
        'Build',
        'chrono-install',
        'share',
        'chrono',
        'data',
        'vehicle')) + os.sep


def parse_four_floats(raw, label):
    values = [float(item.strip()) for item in raw.split(',') if item.strip()]
    if len(values) != 4:
        raise argparse.ArgumentTypeError(
            '%s must contain exactly four comma-separated values in FL,FR,RL,RR order' % label)
    if any((not math.isfinite(value)) or value < 0.0 for value in values):
        raise argparse.ArgumentTypeError('%s values must be finite and non-negative' % label)
    return values


# ==============================================================================
# -- planner provider hook -----------------------------------------------------
# ==============================================================================


class PlannerOutput(object):
    """Container for the per-tick output expected from a planner/model."""

    def __init__(self, control, trajectory=None, target_speed=0.0,
                 curvature=0.0, summary=None, source='mock'):
        self.control = control
        self.trajectory = trajectory or []
        self.target_speed = float(target_speed)
        self.curvature = float(curvature)
        self.summary = summary or {}
        self.source = source

    def to_log_dict(self):
        return {
            'source': self.source,
            'target_speed_mps': self.target_speed,
            'curvature': self.curvature,
            'trajectory': self.trajectory,
            'summary': self.summary,
        }


class PlannerOutputProvider(object):
    """Base interface for TransFuser++ or any other planner adapter."""

    def run_step(self, frame, timestamp, vehicle, world):
        raise NotImplementedError


class MockPlannerOutputProvider(PlannerOutputProvider):
    """Waypoint-following mock provider used until the real model is wired in."""

    def __init__(self, target_speed, trajectory_points, trajectory_spacing):
        self.target_speed = float(target_speed)
        self.trajectory_points = int(trajectory_points)
        self.trajectory_spacing = float(trajectory_spacing)

    def run_step(self, frame, timestamp, vehicle, world):
        del frame
        del timestamp

        carla_map = world.get_map()
        transform = vehicle.get_transform()
        location = transform.location
        waypoint = carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving)

        trajectory = self._build_trajectory(waypoint)
        target_location = waypoint.transform.location
        if trajectory:
            target_location = carla.Location(
                x=trajectory[min(2, len(trajectory) - 1)]['location']['x'],
                y=trajectory[min(2, len(trajectory) - 1)]['location']['y'],
                z=trajectory[min(2, len(trajectory) - 1)]['location']['z'])

        heading_error = self._heading_error(transform, target_location)
        speed_mps = vector_length(vehicle.get_velocity())
        steer = clamp(heading_error / math.radians(45.0), -0.65, 0.65)

        speed_error = self.target_speed - speed_mps
        throttle = clamp(0.30 + 0.10 * speed_error, 0.0, 0.75)
        brake = 0.0
        if speed_error < -1.0:
            throttle = 0.0
            brake = clamp(-0.20 * speed_error, 0.0, 0.60)

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False)

        lookahead = max(self.trajectory_spacing, self.trajectory_spacing * 3.0)
        curvature = heading_error / lookahead
        summary = {
            'heading_error_rad': heading_error,
            'speed_mps': speed_mps,
            'lookahead_m': lookahead,
        }
        return PlannerOutput(
            control=control,
            trajectory=trajectory,
            target_speed=self.target_speed,
            curvature=curvature,
            summary=summary,
            source='mock_waypoint')

    def _build_trajectory(self, waypoint):
        trajectory = []
        current = waypoint
        for _ in range(self.trajectory_points):
            next_waypoints = current.next(self.trajectory_spacing)
            if not next_waypoints:
                break
            current = next_waypoints[0]
            trajectory.append(waypoint_to_dict(current))
        return trajectory

    @staticmethod
    def _heading_error(transform, target_location):
        forward = transform.get_forward_vector()
        to_target = target_location - transform.location
        target_norm = math.sqrt(to_target.x * to_target.x + to_target.y * to_target.y)
        if target_norm < 1e-3:
            return 0.0
        target_x = to_target.x / target_norm
        target_y = to_target.y / target_norm
        forward_norm = math.sqrt(forward.x * forward.x + forward.y * forward.y)
        if forward_norm < 1e-3:
            return 0.0
        forward_x = forward.x / forward_norm
        forward_y = forward.y / forward_norm
        cross_z = forward_x * target_y - forward_y * target_x
        dot = clamp(forward_x * target_x + forward_y * target_y, -1.0, 1.0)
        return math.atan2(cross_z, dot)


class JsonlPlannerOutputProvider(PlannerOutputProvider):
    """Replay planner output from JSONL records.

    Each line may contain `control`, `trajectory`, `target_speed_mps`, and
    `curvature`. This is a simple bridge for precomputed TransFuser++ outputs.
    """

    def __init__(self, path):
        self.path = path
        self.records = []
        self.index = 0
        with open(path, 'r') as input_file:
            for line in input_file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                self.records.append(json.loads(line))
        if not self.records:
            raise ValueError('planner JSONL is empty: %s' % path)

    def run_step(self, frame, timestamp, vehicle, world):
        del frame
        del timestamp
        del vehicle
        del world

        if self.index >= len(self.records):
            record = self.records[-1]
        else:
            record = self.records[self.index]
            self.index += 1

        raw_control = record.get('control', record)
        control = carla.VehicleControl(
            throttle=float(raw_control.get('throttle', 0.0)),
            steer=float(raw_control.get('steer', 0.0)),
            brake=float(raw_control.get('brake', 0.0)),
            hand_brake=bool(raw_control.get('hand_brake', False)),
            reverse=bool(raw_control.get('reverse', False)),
            manual_gear_shift=bool(raw_control.get('manual_gear_shift', False)),
            gear=int(raw_control.get('gear', 0)))

        return PlannerOutput(
            control=control,
            trajectory=record.get('trajectory', []),
            target_speed=float(record.get('target_speed_mps', record.get('target_speed', 0.0))),
            curvature=float(record.get('curvature', 0.0)),
            summary=record.get('summary', {}),
            source='jsonl')


# ==============================================================================
# -- suspension controller -----------------------------------------------------
# ==============================================================================


class SuspensionCommand(object):
    def __init__(self, damping, stiffness, summary):
        self.damping = [float(value) for value in damping]
        self.stiffness = [float(value) for value in stiffness]
        self.summary = summary

    def to_log_dict(self):
        return {
            'wheel_order': WHEEL_ORDER,
            'damping': self.damping,
            'stiffness': self.stiffness,
            'summary': self.summary,
        }


class RuleBasedSuspensionController(object):
    """Baseline planner-output to suspension-command mapping."""

    def __init__(self, base_damping, base_stiffness, min_scale, max_scale):
        self.base_damping = [float(value) for value in base_damping]
        self.base_stiffness = [float(value) for value in base_stiffness]
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)

    def compute(self, planner_output, speed_mps):
        control = planner_output.control
        steer = float(control.steer)
        throttle = float(control.throttle)
        brake = float(control.brake)
        curvature = abs(float(planner_output.curvature))

        lateral_demand = clamp(abs(steer) + curvature * max(speed_mps, 1.0) * 4.0, 0.0, 1.0)
        brake_demand = clamp(brake, 0.0, 1.0)
        throttle_demand = clamp(throttle, 0.0, 1.0)

        front_pitch = 1.0 + 0.28 * brake_demand + 0.10 * lateral_demand
        rear_pitch = 1.0 + 0.14 * throttle_demand + 0.10 * lateral_demand

        # Positive CARLA steer is treated as a right turn, so the left side is
        # the nominal outside side for this first-pass load-transfer heuristic.
        side_bias = clamp(0.18 * steer, -0.18, 0.18)

        damping_scales = [
            front_pitch * (1.0 + side_bias),
            front_pitch * (1.0 - side_bias),
            rear_pitch * (1.0 + side_bias),
            rear_pitch * (1.0 - side_bias),
        ]
        stiffness_scales = [
            (1.0 + 0.18 * brake_demand + 0.08 * lateral_demand) * (1.0 + 0.5 * side_bias),
            (1.0 + 0.18 * brake_demand + 0.08 * lateral_demand) * (1.0 - 0.5 * side_bias),
            (1.0 + 0.10 * throttle_demand + 0.08 * lateral_demand) * (1.0 + 0.5 * side_bias),
            (1.0 + 0.10 * throttle_demand + 0.08 * lateral_demand) * (1.0 - 0.5 * side_bias),
        ]

        damping = [
            self.base_damping[index] * clamp(damping_scales[index], self.min_scale, self.max_scale)
            for index in range(4)
        ]
        stiffness = [
            self.base_stiffness[index] * clamp(stiffness_scales[index], self.min_scale, self.max_scale)
            for index in range(4)
        ]
        summary = {
            'mapping': 'rule_based_v0',
            'lateral_demand': lateral_demand,
            'brake_demand': brake_demand,
            'throttle_demand': throttle_demand,
            'side_bias': side_bias,
            'min_scale': self.min_scale,
            'max_scale': self.max_scale,
        }
        return SuspensionCommand(damping, stiffness, summary)


# ==============================================================================
# -- collision and logging -----------------------------------------------------
# ==============================================================================


class CollisionMonitor(object):
    def __init__(self, vehicle):
        self.sensor = None
        self.events = []
        world = vehicle.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionMonitor._on_collision(weak_self, event))

    def pop_events_up_to(self, frame):
        ready = [event for event in self.events if event['frame'] <= frame]
        self.events = [event for event in self.events if event['frame'] > frame]
        return ready

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if self is None:
            return
        impulse = event.normal_impulse
        other_actor = event.other_actor
        self.events.append({
            'frame': int(event.frame),
            'other_actor_id': int(other_actor.id) if other_actor is not None else None,
            'other_actor_type': other_actor.type_id if other_actor is not None else None,
            'normal_impulse': vector_to_dict(impulse),
            'intensity': vector_length(impulse),
        })


class JsonlExperimentLogger(object):
    def __init__(self, log_dir, log_name):
        if not os.path.isabs(log_dir):
            log_dir = os.path.join(SCRIPT_DIR, log_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        if not log_name:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_name = 'chrono_suspension_%s.jsonl' % stamp
        self.path = os.path.join(log_dir, log_name)
        self._file = open(self.path, 'w')

    def write(self, record):
        self._file.write(json.dumps(record, sort_keys=True) + '\n')
        self._file.flush()

    def close(self):
        self._file.close()


# ==============================================================================
# -- world setup ---------------------------------------------------------------
# ==============================================================================


def get_actor_blueprints(world, filter_pattern, generation):
    blueprints = world.get_blueprint_library().filter(filter_pattern)
    if generation.lower() == 'all':
        return list(blueprints)
    if len(blueprints) == 1:
        return list(blueprints)
    try:
        generation_value = int(generation)
    except ValueError:
        return []
    return [
        blueprint for blueprint in blueprints
        if blueprint.has_attribute('generation') and
        int(blueprint.get_attribute('generation')) == generation_value
    ]


def choose_blueprint(world, args, rng):
    blueprints = sorted(
        get_actor_blueprints(world, args.filter, args.generation),
        key=lambda blueprint: blueprint.id)
    if not blueprints:
        raise RuntimeError('no vehicle blueprints matched filter=%s generation=%s' % (
            args.filter, args.generation))
    blueprint = rng.choice(blueprints)
    blueprint.set_attribute('role_name', args.rolename)
    if blueprint.has_attribute('color'):
        color = rng.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)
    if blueprint.has_attribute('driver_id'):
        driver_id = rng.choice(blueprint.get_attribute('driver_id').recommended_values)
        blueprint.set_attribute('driver_id', driver_id)
    if blueprint.has_attribute('is_invincible'):
        blueprint.set_attribute('is_invincible', 'true')
    return blueprint


def spawn_ego_vehicle(world, args, rng):
    blueprint = choose_blueprint(world, args, rng)
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError('map has no vehicle spawn points')

    if args.spawn_index is not None:
        spawn_points = [spawn_points[args.spawn_index % len(spawn_points)]]
    else:
        rng.shuffle(spawn_points)

    for spawn_point in spawn_points:
        spawn_point.location.z += args.spawn_z_offset
        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle is not None:
            logging.info('spawned ego %s at %s', blueprint.id, spawn_point)
            return vehicle
    raise RuntimeError('failed to spawn ego vehicle after trying %d spawn points' % len(spawn_points))


def set_spectator(world, vehicle):
    transform = vehicle.get_transform()
    spectator_transform = carla.Transform(transform.location, transform.rotation)
    spectator_transform.location += transform.get_forward_vector() * -8.0
    spectator_transform.location.z += 4.0
    spectator_transform.rotation.pitch = -18.0
    world.get_spectator().set_transform(spectator_transform)


def enable_chrono_physics(vehicle, args):
    base_path = args.chrono_base_path
    if not base_path.endswith(os.sep):
        base_path += os.sep
    logging.info(
        'enabling Chrono physics: vehicle=%s powertrain=%s tire=%s base=%s',
        args.chrono_vehicle_json,
        args.chrono_powertrain_json,
        args.chrono_tire_json,
        base_path)
    vehicle.enable_chrono_physics(
        args.chrono_max_substeps,
        args.chrono_max_substep_delta_time,
        args.chrono_vehicle_json,
        args.chrono_powertrain_json,
        args.chrono_tire_json,
        base_path)


def make_planner_provider(args):
    if args.planner_jsonl:
        return JsonlPlannerOutputProvider(args.planner_jsonl)
    return MockPlannerOutputProvider(
        target_speed=args.target_speed,
        trajectory_points=args.trajectory_points,
        trajectory_spacing=args.trajectory_spacing)


def make_ego_state(vehicle):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    angular_velocity = vehicle.get_angular_velocity()
    return {
        'transform': transform_to_dict(transform),
        'velocity': vector_to_dict(velocity),
        'acceleration': vector_to_dict(acceleration),
        'angular_velocity': vector_to_dict(angular_velocity),
        'speed_mps': vector_length(velocity),
        'control': control_to_dict(vehicle.get_control()),
    }


# ==============================================================================
# -- experiment loop -----------------------------------------------------------
# ==============================================================================


def run_experiment(args):
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG if args.debug else logging.INFO)

    rng = random.Random(args.seed)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    original_settings = world.get_settings()
    vehicle = None
    collision_monitor = None
    logger = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta_seconds
        settings.no_rendering_mode = args.no_rendering
        world.apply_settings(settings)

        vehicle = spawn_ego_vehicle(world, args, rng)
        collision_monitor = CollisionMonitor(vehicle)
        enable_chrono_physics(vehicle, args)
        world.tick()

        if args.spectator:
            set_spectator(world, vehicle)

        planner_provider = make_planner_provider(args)
        suspension_controller = RuleBasedSuspensionController(
            base_damping=args.base_damping,
            base_stiffness=args.base_stiffness,
            min_scale=args.suspension_min_scale,
            max_scale=args.suspension_max_scale)
        logger = JsonlExperimentLogger(args.log_dir, args.log_name)
        logging.info('writing experiment log to %s', logger.path)

        logger.write({
            'type': 'metadata',
            'created_at': datetime.datetime.now().isoformat(),
            'script': os.path.basename(__file__),
            'wheel_order': WHEEL_ORDER,
            'args': vars(args),
            'chrono': {
                'vehicle_json': args.chrono_vehicle_json,
                'powertrain_json': args.chrono_powertrain_json,
                'tire_json': args.chrono_tire_json,
                'base_json_path': args.chrono_base_path,
                'max_substeps': args.chrono_max_substeps,
                'max_substep_delta_time': args.chrono_max_substep_delta_time,
            },
        })

        chrono_active = True
        chrono_error = None
        chrono_inactive_reason = None
        completed_ticks = 0

        for step in range(args.ticks):
            snapshot = world.get_snapshot()
            timestamp = snapshot.timestamp
            pre_state = make_ego_state(vehicle)
            planner_output = planner_provider.run_step(snapshot.frame, timestamp, vehicle, world)
            suspension_command = suspension_controller.compute(
                planner_output,
                pre_state['speed_mps'])

            suspension_applied = False
            if chrono_active:
                try:
                    vehicle.apply_chrono_suspension_control(
                        damping=suspension_command.damping,
                        stiffness=suspension_command.stiffness)
                    suspension_applied = True
                    chrono_error = None
                except RuntimeError as error:
                    chrono_active = False
                    chrono_error = str(error)
                    chrono_inactive_reason = 'suspension command failed: %s' % chrono_error
                    logging.error('Chrono suspension command failed at step %d: %s', step, chrono_error)
                    if args.stop_on_chrono_error:
                        break
            else:
                chrono_error = chrono_inactive_reason or 'Chrono disabled before suspension command'

            vehicle.apply_control(planner_output.control)
            frame = world.tick()
            snapshot = world.get_snapshot()
            post_state = make_ego_state(vehicle)
            collision_events = collision_monitor.pop_events_up_to(snapshot.frame)

            if collision_events:
                chrono_active = False
                chrono_inactive_reason = 'Chrono disabled after collision/fallback'
                chrono_error = chrono_inactive_reason
                logging.warning(
                    'collision detected at frame %d; Chrono may have fallen back to default physics',
                    snapshot.frame)

            logger.write({
                'type': 'tick',
                'step': step,
                'frame': int(frame),
                'snapshot_frame': int(snapshot.frame),
                'timestamp': timestamp_to_dict(snapshot.timestamp),
                'planner': planner_output.to_log_dict(),
                'vehicle_control': control_to_dict(planner_output.control),
                'suspension': suspension_command.to_log_dict(),
                'suspension_applied': suspension_applied,
                'chrono': {
                    'active': chrono_active,
                    'apply_error': chrono_error,
                },
                'ego_pre_tick': pre_state,
                'ego_post_tick': post_state,
                'collisions': collision_events,
            })
            completed_ticks += 1

            if args.stop_on_collision and collision_events:
                break

        logging.info('completed %d ticks', completed_ticks)
        if logger is not None:
            logging.info('log written to %s', logger.path)

    finally:
        if logger is not None:
            logger.close()
        if collision_monitor is not None:
            collision_monitor.destroy()
        if vehicle is not None:
            vehicle.destroy()
        world.apply_settings(original_settings)


# ==============================================================================
# -- arguments -----------------------------------------------------------------
# ==============================================================================


def parse_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', default='127.0.0.1', help='CARLA host (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='CARLA port (default: 2000)')
    argparser.add_argument('--timeout', default=20.0, type=float, help='client timeout in seconds')
    argparser.add_argument('--debug', action='store_true', help='print debug logging')

    argparser.add_argument('--ticks', default=600, type=int, help='number of synchronous ticks to run')
    argparser.add_argument('--fixed-delta-seconds', default=0.05, type=float, help='fixed simulation dt')
    argparser.add_argument('--no-rendering', action='store_true', help='enable CARLA no-rendering mode')
    argparser.add_argument('--seed', default=7, type=int, help='deterministic spawn/blueprint seed')
    argparser.add_argument('--spectator', action='store_true', help='move spectator behind the ego vehicle')

    argparser.add_argument('--filter', default='vehicle.*', help='ego blueprint filter')
    argparser.add_argument('--generation', default='All', help='ego blueprint generation: 1, 2, or All')
    argparser.add_argument('--rolename', default='hero', help='ego actor role_name')
    argparser.add_argument('--spawn-index', type=int, help='use a deterministic map spawn index')
    argparser.add_argument('--spawn-z-offset', default=0.5, type=float, help='extra ego spawn height in meters')

    argparser.add_argument('--chrono-base-path', default=default_chrono_base_path(), help='Chrono vehicle data path')
    argparser.add_argument('--chrono-vehicle-json', default='hmmwv/vehicle/HMMWV_Vehicle.json')
    argparser.add_argument('--chrono-powertrain-json', default='hmmwv/powertrain/HMMWV_ShaftsPowertrain.json')
    argparser.add_argument('--chrono-tire-json', default='hmmwv/tire/HMMWV_Pac02Tire.json')
    argparser.add_argument('--chrono-max-substeps', default=5000, type=int)
    argparser.add_argument('--chrono-max-substep-delta-time', default=0.002, type=float)

    argparser.add_argument('--planner-jsonl', help='optional JSONL planner output replay file')
    argparser.add_argument('--target-speed', default=8.0, type=float, help='mock planner target speed in m/s')
    argparser.add_argument('--trajectory-points', default=8, type=int, help='mock trajectory point count')
    argparser.add_argument('--trajectory-spacing', default=2.0, type=float, help='mock trajectory spacing in meters')

    argparser.add_argument(
        '--base-damping',
        default=parse_four_floats('3500,3500,4200,4200', 'base damping'),
        type=lambda raw: parse_four_floats(raw, 'base damping'),
        help='base damping FL,FR,RL,RR')
    argparser.add_argument(
        '--base-stiffness',
        default=parse_four_floats('70000,70000,80000,80000', 'base stiffness'),
        type=lambda raw: parse_four_floats(raw, 'base stiffness'),
        help='base stiffness FL,FR,RL,RR')
    argparser.add_argument('--suspension-min-scale', default=0.70, type=float)
    argparser.add_argument('--suspension-max-scale', default=1.45, type=float)
    argparser.add_argument('--stop-on-chrono-error', action='store_true')
    argparser.add_argument('--stop-on-collision', action='store_true')

    argparser.add_argument('--log-dir', default='chrono_suspension_logs', help='log dir relative to this script')
    argparser.add_argument('--log-name', help='optional log filename')

    args = argparser.parse_args()
    if args.fixed_delta_seconds <= 0.0:
        raise ValueError('--fixed-delta-seconds must be positive')
    if args.ticks <= 0:
        raise ValueError('--ticks must be positive')
    if args.suspension_min_scale < 0.0 or args.suspension_max_scale < args.suspension_min_scale:
        raise ValueError('invalid suspension scale range')
    return args


def main():
    run_experiment(parse_arguments())


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

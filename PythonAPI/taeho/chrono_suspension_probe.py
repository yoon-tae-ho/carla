#!/usr/bin/env python

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
# de Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Minimal Chrono suspension-control probe.

This script is intentionally separate from chrono_suspension_transfuser_experiment.py.
It only verifies the Chrono + real-time suspension command path:

1. spawn one ego vehicle;
2. enable Chrono physics;
3. apply simple FL/FR/RL/RR damping and stiffness commands every tick;
4. log the command, whether CARLA accepted it, and the observable vehicle state.

CARLA currently exposes suspension command write access through Python, but it
does not expose a Python getter for the internal Chrono spring/damper
coefficients. Use analyze_chrono_suspension_probe.py to read this log and check
the applied command sequence plus the vehicle response.
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


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if 'PYTHON_EGG_CACHE' not in os.environ:
    os.environ['PYTHON_EGG_CACHE'] = os.path.join('/tmp', 'python-eggs')

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

sys.path.append(os.path.join(SCRIPT_DIR, '..', 'carla'))


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla


# ==============================================================================
# -- small helpers -------------------------------------------------------------
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


def scale_values(values, scale):
    return [float(value) * float(scale) for value in values]


def make_vehicle_state(vehicle):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    angular_velocity = vehicle.get_angular_velocity()
    rotation = transform.rotation
    return {
        'transform': transform_to_dict(transform),
        'velocity': vector_to_dict(velocity),
        'acceleration': vector_to_dict(acceleration),
        'angular_velocity': vector_to_dict(angular_velocity),
        'speed_mps': vector_length(velocity),
        'abs_roll_deg': abs(float(rotation.roll)),
        'abs_pitch_deg': abs(float(rotation.pitch)),
        'control': control_to_dict(vehicle.get_control()),
    }


# ==============================================================================
# -- command profile -----------------------------------------------------------
# ==============================================================================


class SuspensionPhase(object):
    def __init__(self, name, duration_ticks, damping_scale, stiffness_scale):
        self.name = name
        self.duration_ticks = int(duration_ticks)
        self.damping_scale = float(damping_scale)
        self.stiffness_scale = float(stiffness_scale)

    def command(self, base_damping, base_stiffness):
        return {
            'wheel_order': WHEEL_ORDER,
            'phase': self.name,
            'damping_scale': self.damping_scale,
            'stiffness_scale': self.stiffness_scale,
            'damping': scale_values(base_damping, self.damping_scale),
            'stiffness': scale_values(base_stiffness, self.stiffness_scale),
        }

    def to_dict(self):
        return {
            'name': self.name,
            'duration_ticks': self.duration_ticks,
            'damping_scale': self.damping_scale,
            'stiffness_scale': self.stiffness_scale,
        }


def build_base_phase_plan(args):
    return [
        SuspensionPhase('soft', args.phase_ticks, args.soft_damping_scale, args.soft_stiffness_scale),
        SuspensionPhase('baseline', args.phase_ticks, 1.0, 1.0),
        SuspensionPhase('stiff', args.phase_ticks, args.stiff_damping_scale, args.stiff_stiffness_scale),
    ]


def build_phase_plan(args):
    single_cycle = build_base_phase_plan(args)
    plan = []
    for cycle in range(args.cycles):
        for phase in single_cycle:
            if args.cycles == 1:
                name = phase.name
            else:
                name = '%s_cycle_%d' % (phase.name, cycle + 1)
            plan.append(SuspensionPhase(
                name,
                phase.duration_ticks,
                phase.damping_scale,
                phase.stiffness_scale))
    return plan


def build_independent_trial_plan(args):
    plan = []
    for repeat in range(args.trials_per_phase):
        for phase in build_base_phase_plan(args):
            plan.append({
                'trial_id': len(plan),
                'repeat': repeat,
                'phase': phase,
            })
    return plan


def iter_phase_ticks(phases):
    step = 0
    for phase in phases:
        for phase_tick in range(phase.duration_ticks):
            yield step, phase, phase_tick
            step += 1


def make_drive_control(args, global_step, phase_tick, speed_mps):
    del phase_tick

    if args.drive_mode == 'stationary':
        return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

    brake = 0.0
    throttle = args.cruise_throttle
    if speed_mps > args.target_speed:
        throttle = 0.0
        brake = clamp((speed_mps - args.target_speed) / max(args.target_speed, 0.1), 0.0, args.speed_limiter_brake)

    if args.drive_mode == 'straight':
        steer = 0.0
    else:
        radians = 2.0 * math.pi * float(global_step) / max(float(args.steer_period_ticks), 1.0)
        steer = args.steer_amplitude * math.sin(radians)

    return carla.VehicleControl(
        throttle=clamp(throttle, 0.0, 1.0),
        steer=clamp(steer, -1.0, 1.0),
        brake=clamp(brake, 0.0, 1.0))


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
    blueprints = get_actor_blueprints(world, args.filter, args.generation)
    blueprints = [
        blueprint for blueprint in blueprints
        if not blueprint.has_attribute('number_of_wheels') or
        int(blueprint.get_attribute('number_of_wheels')) == 4
    ]
    blueprints = sorted(blueprints, key=lambda blueprint: blueprint.id)
    if not blueprints:
        raise RuntimeError('no four-wheel vehicle blueprints matched filter=%s generation=%s' % (
            args.filter, args.generation))

    if args.blueprint_id:
        matches = [blueprint for blueprint in blueprints if blueprint.id == args.blueprint_id]
        if not matches:
            raise RuntimeError('blueprint id %s did not match the filtered vehicle list' % args.blueprint_id)
        blueprint = matches[0]
    else:
        blueprint = blueprints[args.blueprint_index % len(blueprints)]

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


def copy_transform(transform):
    return carla.Transform(
        carla.Location(
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z)),
        carla.Rotation(
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
            roll=float(transform.rotation.roll)))


def choose_spawn_transform(world, args, rng):
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError('map has no vehicle spawn points')

    if args.spawn_index is not None:
        spawn_point = copy_transform(spawn_points[args.spawn_index % len(spawn_points)])
    else:
        rng.shuffle(spawn_points)
        spawn_point = copy_transform(spawn_points[0])

    spawn_point.location.z += args.spawn_z_offset
    return spawn_point


def spawn_vehicle_at(world, blueprint, spawn_transform):
    vehicle = world.try_spawn_actor(blueprint, copy_transform(spawn_transform))
    if vehicle is None:
        raise RuntimeError('failed to spawn ego vehicle at %s' % spawn_transform)
    logging.info('spawned ego %s at %s', blueprint.id, spawn_transform)
    return vehicle


def spawn_ego_vehicle(world, args, rng):
    blueprint = choose_blueprint(world, args, rng)
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError('map has no vehicle spawn points')

    if args.spawn_index is not None:
        spawn_points = [copy_transform(spawn_points[args.spawn_index % len(spawn_points)])]
    else:
        rng.shuffle(spawn_points)

    for raw_spawn_point in spawn_points:
        spawn_point = copy_transform(raw_spawn_point)
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


class JsonlLogger(object):
    def __init__(self, log_dir, log_name):
        if not os.path.isabs(log_dir):
            log_dir = os.path.join(SCRIPT_DIR, log_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        if not log_name:
            stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            log_name = 'chrono_suspension_probe_%s.jsonl' % stamp
        self.path = os.path.join(log_dir, log_name)
        self._file = open(self.path, 'w')

    def write(self, record):
        self._file.write(json.dumps(record, sort_keys=True) + '\n')
        self._file.flush()

    def close(self):
        self._file.close()


# ==============================================================================
# -- probe loop ----------------------------------------------------------------
# ==============================================================================


def apply_suspension_command(vehicle, command):
    vehicle.apply_chrono_suspension_control(
        damping=command['damping'],
        stiffness=command['stiffness'])


def add_trial_fields(record, trial_id=None, phase_name=None, repeat=None, sample_role=None):
    if trial_id is not None:
        record['trial_id'] = int(trial_id)
    if phase_name is not None:
        record['phase'] = phase_name
    if repeat is not None:
        record['repeat'] = int(repeat)
    if sample_role is not None:
        record['sample_role'] = sample_role
    return record


def write_metadata(logger, args, phases, independent_trials=None):
    metadata = {
        'type': 'metadata',
        'created_at': datetime.datetime.now().isoformat(),
        'script': os.path.basename(__file__),
        'wheel_order': WHEEL_ORDER,
        'args': vars(args),
        'test_design': args.test_design,
        'chrono': {
            'vehicle_json': args.chrono_vehicle_json,
            'powertrain_json': args.chrono_powertrain_json,
            'tire_json': args.chrono_tire_json,
            'base_json_path': args.chrono_base_path,
            'max_substeps': args.chrono_max_substeps,
            'max_substep_delta_time': args.chrono_max_substep_delta_time,
        },
        'phase_plan': [phase.to_dict() for phase in phases],
        'readback_note': (
            'Python API has no direct getter for Chrono spring/damper coefficients; '
            'this log records the accepted command and observable vehicle response.'),
    }
    if independent_trials is not None:
        metadata['independent_trials'] = [
            {
                'trial_id': trial['trial_id'],
                'repeat': trial['repeat'],
                'phase': trial['phase'].to_dict(),
            }
            for trial in independent_trials
        ]
        metadata['fairness_note'] = (
            'independent_phases respawns the same blueprint at the same transform for each phase; '
            'settle records are warm-up only and tick records are measurement samples.')
    logger.write(metadata)


def write_precheck(logger, vehicle, command, trial_id=None, phase_name=None, repeat=None):
    try:
        apply_suspension_command(vehicle, command)
        logger.write(add_trial_fields({
            'type': 'precheck',
            'command': command,
            'command_applied': True,
            'apply_error': None,
            'state': make_vehicle_state(vehicle),
        }, trial_id=trial_id, phase_name=phase_name, repeat=repeat, sample_role='precheck'))
    except RuntimeError as error:
        logger.write(add_trial_fields({
            'type': 'precheck',
            'command': command,
            'command_applied': False,
            'apply_error': str(error),
            'state': make_vehicle_state(vehicle),
        }, trial_id=trial_id, phase_name=phase_name, repeat=repeat, sample_role='precheck'))
        raise


def run_braked_settle(world, vehicle, args, logger, command,
                      trial_id=None, phase_name=None, repeat=None):
    for settle_tick in range(args.settle_ticks):
        command_applied = False
        apply_error = None
        try:
            apply_suspension_command(vehicle, command)
            command_applied = True
        except RuntimeError as error:
            apply_error = str(error)
            logging.error('Chrono command failed during settle tick %d: %s', settle_tick, apply_error)

        vehicle_control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        vehicle.apply_control(vehicle_control)
        frame = world.tick()
        snapshot = world.get_snapshot()
        logger.write(add_trial_fields({
            'type': 'settle',
            'settle_tick': settle_tick,
            'step': -args.settle_ticks + settle_tick if trial_id is None else settle_tick,
            'frame': int(frame),
            'snapshot_frame': int(snapshot.frame),
            'timestamp': timestamp_to_dict(snapshot.timestamp),
            'command': command,
            'command_applied': command_applied,
            'apply_error': apply_error,
            'vehicle_control': control_to_dict(vehicle_control),
            'state': make_vehicle_state(vehicle),
        }, trial_id=trial_id, phase_name=phase_name, repeat=repeat, sample_role='settle'))
        if not command_applied:
            raise RuntimeError(apply_error)


def run_measurement_phase(world, vehicle, args, logger, phase, global_step_start,
                          trial_id=None, repeat=None, restart_control_step=False):
    completed_ticks = 0
    stopped = False
    for phase_tick in range(phase.duration_ticks):
        global_step = global_step_start + phase_tick
        control_step = phase_tick if restart_control_step else global_step
        pre_state = make_vehicle_state(vehicle)
        command = phase.command(args.base_damping, args.base_stiffness)
        vehicle_control = make_drive_control(args, control_step, phase_tick, pre_state['speed_mps'])

        command_applied = False
        apply_error = None
        try:
            apply_suspension_command(vehicle, command)
            command_applied = True
        except RuntimeError as error:
            apply_error = str(error)
            logging.error('Chrono suspension command failed at step %d: %s', global_step, apply_error)

        vehicle.apply_control(vehicle_control)
        frame = world.tick()
        snapshot = world.get_snapshot()
        post_state = make_vehicle_state(vehicle)
        unsafe_pose = (
            post_state['abs_roll_deg'] > args.max_abs_roll or
            post_state['abs_pitch_deg'] > args.max_abs_pitch)

        logger.write(add_trial_fields({
            'type': 'tick',
            'step': global_step,
            'phase_tick': phase_tick,
            'frame': int(frame),
            'snapshot_frame': int(snapshot.frame),
            'timestamp': timestamp_to_dict(snapshot.timestamp),
            'command': command,
            'command_applied': command_applied,
            'apply_error': apply_error,
            'vehicle_control': control_to_dict(vehicle_control),
            'pre_state': pre_state,
            'state': post_state,
            'unsafe_pose': unsafe_pose,
        }, trial_id=trial_id, phase_name=phase.name, repeat=repeat, sample_role='measurement'))

        completed_ticks += 1
        if not command_applied:
            stopped = True
            break
        if unsafe_pose and not args.keep_going_on_unsafe_pose:
            logging.warning(
                'stopping on unsafe pose at step %d: roll=%.2f pitch=%.2f',
                global_step,
                post_state['abs_roll_deg'],
                post_state['abs_pitch_deg'])
            stopped = True
            break
    return completed_ticks, stopped


def run_probe(args):
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG if args.debug else logging.INFO)
    if args.test_design == 'independent_phases' and args.settle_ticks < 100:
        logging.warning(
            'independent_phases is fairer with a long settle window; consider --settle-ticks 200 or higher')
    if args.test_design == 'independent_phases' and args.spawn_z_offset > 0.2:
        logging.warning(
            'large spawn z offset creates drop/settling transients; consider --spawn-z-offset 0.1')
    if args.test_design == 'independent_phases' and args.cycles != 1:
        logging.warning(
            '--cycles only affects sequential_phases; use --trials-per-phase for independent_phases')

    rng = random.Random(args.seed)
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    original_settings = world.get_settings()
    vehicle = None
    logger = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta_seconds
        settings.no_rendering_mode = args.no_rendering
        world.apply_settings(settings)

        independent_trials = None
        if args.test_design == 'independent_phases':
            phases = build_base_phase_plan(args)
            independent_trials = build_independent_trial_plan(args)
        else:
            phases = build_phase_plan(args)
        logger = JsonlLogger(args.log_dir, args.log_name)
        logging.info('writing probe log to %s', logger.path)
        write_metadata(logger, args, phases, independent_trials=independent_trials)

        completed_ticks = 0
        if args.test_design == 'sequential_phases':
            vehicle = spawn_ego_vehicle(world, args, rng)
            enable_chrono_physics(vehicle, args)
            world.tick()

            if args.spectator:
                set_spectator(world, vehicle)

            baseline = SuspensionPhase('precheck_baseline', 1, 1.0, 1.0).command(
                args.base_damping,
                args.base_stiffness)
            write_precheck(logger, vehicle, baseline)
            run_braked_settle(world, vehicle, args, logger, baseline)

            for phase in phases:
                phase_ticks, stopped = run_measurement_phase(
                    world,
                    vehicle,
                    args,
                    logger,
                    phase,
                    completed_ticks,
                    restart_control_step=False)
                completed_ticks += phase_ticks
                if stopped:
                    break
        else:
            blueprint = choose_blueprint(world, args, rng)
            spawn_transform = choose_spawn_transform(world, args, rng)
            logger.write({
                'type': 'fair_test_setup',
                'blueprint_id': blueprint.id,
                'spawn_transform': transform_to_dict(spawn_transform),
                'settle_ticks_per_trial': int(args.settle_ticks),
                'measurement_ticks_per_trial': int(args.phase_ticks),
                'trials_per_phase': int(args.trials_per_phase),
            })

            for trial in independent_trials:
                phase = trial['phase']
                trial_id = trial['trial_id']
                repeat = trial['repeat']
                logging.info(
                    'starting fair trial %d phase=%s repeat=%d',
                    trial_id,
                    phase.name,
                    repeat)
                vehicle = spawn_vehicle_at(world, blueprint, spawn_transform)
                enable_chrono_physics(vehicle, args)
                world.tick()

                if args.spectator:
                    set_spectator(world, vehicle)

                command = phase.command(args.base_damping, args.base_stiffness)
                write_precheck(
                    logger,
                    vehicle,
                    command,
                    trial_id=trial_id,
                    phase_name=phase.name,
                    repeat=repeat)
                run_braked_settle(
                    world,
                    vehicle,
                    args,
                    logger,
                    command,
                    trial_id=trial_id,
                    phase_name=phase.name,
                    repeat=repeat)
                phase_ticks, stopped = run_measurement_phase(
                    world,
                    vehicle,
                    args,
                    logger,
                    phase,
                    completed_ticks,
                    trial_id=trial_id,
                    repeat=repeat,
                    restart_control_step=True)
                completed_ticks += phase_ticks

                vehicle.destroy()
                vehicle = None
                world.tick()

                if stopped:
                    break

        logging.info('completed %d probe ticks', completed_ticks)
        logging.info('log written to %s', logger.path)

    finally:
        if logger is not None:
            logger.close()
        if vehicle is not None:
            vehicle.destroy()
        world.apply_settings(original_settings)


# ==============================================================================
# -- arguments -----------------------------------------------------------------
# ==============================================================================


def parse_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', default='127.0.0.1', help='CARLA host')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='CARLA port')
    argparser.add_argument('--timeout', default=20.0, type=float, help='client timeout in seconds')
    argparser.add_argument('--debug', action='store_true', help='print debug logging')

    argparser.add_argument('--fixed-delta-seconds', default=0.05, type=float, help='fixed simulation dt')
    argparser.add_argument('--no-rendering', action='store_true', help='enable CARLA no-rendering mode')
    argparser.add_argument('--seed', default=7, type=int, help='deterministic spawn/blueprint seed')
    argparser.add_argument('--spectator', action='store_true', help='move spectator behind the ego vehicle')

    argparser.add_argument('--filter', default='vehicle.*', help='ego blueprint filter')
    argparser.add_argument('--generation', default='All', help='ego blueprint generation: 1, 2, or All')
    argparser.add_argument('--blueprint-id', help='exact ego blueprint id, after filter/generation are applied')
    argparser.add_argument('--blueprint-index', default=0, type=int, help='fallback deterministic blueprint index')
    argparser.add_argument('--rolename', default='hero', help='ego actor role_name')
    argparser.add_argument('--spawn-index', type=int, help='deterministic map spawn index')
    argparser.add_argument('--spawn-z-offset', default=0.5, type=float, help='extra ego spawn height in meters')

    argparser.add_argument('--chrono-base-path', default=default_chrono_base_path(), help='Chrono vehicle data path')
    argparser.add_argument('--chrono-vehicle-json', default='hmmwv/vehicle/HMMWV_Vehicle.json')
    argparser.add_argument('--chrono-powertrain-json', default='hmmwv/powertrain/HMMWV_ShaftsPowertrain.json')
    argparser.add_argument('--chrono-tire-json', default='hmmwv/tire/HMMWV_Pac02Tire.json')
    argparser.add_argument('--chrono-max-substeps', default=5000, type=int)
    argparser.add_argument('--chrono-max-substep-delta-time', default=0.002, type=float)

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
    argparser.add_argument('--phase-ticks', default=60, type=int, help='ticks for each soft/baseline/stiff phase')
    argparser.add_argument(
        '--cycles',
        default=1,
        type=int,
        help='repeat the soft/baseline/stiff phase set in sequential_phases')
    argparser.add_argument(
        '--test-design',
        default='sequential_phases',
        choices=['sequential_phases', 'independent_phases'],
        help='sequential smoke test or fair per-phase respawn test')
    argparser.add_argument(
        '--trials-per-phase',
        default=1,
        type=int,
        help='independent_phases repetitions for each soft/baseline/stiff setting')
    argparser.add_argument('--soft-damping-scale', default=0.65, type=float)
    argparser.add_argument('--soft-stiffness-scale', default=0.75, type=float)
    argparser.add_argument('--stiff-damping-scale', default=1.45, type=float)
    argparser.add_argument('--stiff-stiffness-scale', default=1.30, type=float)

    argparser.add_argument(
        '--drive-mode',
        default='low_speed_sine',
        choices=['stationary', 'straight', 'low_speed_sine'],
        help='vehicle motion during the probe')
    argparser.add_argument('--settle-ticks', default=20, type=int, help='braked ticks before the probe starts')
    argparser.add_argument('--target-speed', default=4.0, type=float, help='simple speed limiter target in m/s')
    argparser.add_argument('--cruise-throttle', default=0.22, type=float, help='throttle below target speed')
    argparser.add_argument('--speed-limiter-brake', default=0.25, type=float, help='maximum automatic brake')
    argparser.add_argument('--steer-amplitude', default=0.10, type=float, help='low_speed_sine steering amplitude')
    argparser.add_argument('--steer-period-ticks', default=120, type=int, help='low_speed_sine period')
    argparser.add_argument('--max-abs-roll', default=35.0, type=float, help='stop threshold in degrees')
    argparser.add_argument('--max-abs-pitch', default=35.0, type=float, help='stop threshold in degrees')
    argparser.add_argument('--keep-going-on-unsafe-pose', action='store_true')

    argparser.add_argument(
        '--log-dir',
        default='chrono_suspension_probe_logs',
        help='log dir relative to this script')
    argparser.add_argument('--log-name', help='optional log filename')

    args = argparser.parse_args()
    if args.fixed_delta_seconds <= 0.0:
        raise ValueError('--fixed-delta-seconds must be positive')
    if args.phase_ticks <= 0:
        raise ValueError('--phase-ticks must be positive')
    if args.cycles <= 0:
        raise ValueError('--cycles must be positive')
    if args.trials_per_phase <= 0:
        raise ValueError('--trials-per-phase must be positive')
    if args.settle_ticks < 0:
        raise ValueError('--settle-ticks must be non-negative')
    if args.target_speed <= 0.0:
        raise ValueError('--target-speed must be positive')
    for label, value in [
            ('--soft-damping-scale', args.soft_damping_scale),
            ('--soft-stiffness-scale', args.soft_stiffness_scale),
            ('--stiff-damping-scale', args.stiff_damping_scale),
            ('--stiff-stiffness-scale', args.stiff_stiffness_scale)]:
        if value < 0.0 or not math.isfinite(value):
            raise ValueError('%s must be finite and non-negative' % label)
    return args


def main():
    run_probe(parse_arguments())


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

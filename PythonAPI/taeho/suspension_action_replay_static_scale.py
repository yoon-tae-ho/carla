#!/usr/bin/env python

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Replay identical controls with static suspension scale variants.

Scenarios:
  S0 stock: no suspension API call
  S1 identity: spring_scale=1.00, damper_scale=1.00
  S2 soft_sqrt: spring_scale=0.95, damper_scale=sqrt(0.95)
  S3 stiff_sqrt: spring_scale=1.05, damper_scale=sqrt(1.05)
  S4 damper_only: spring_scale=1.00, damper_scale=1.10
  S5 stiff_damper: spring_scale=1.05, damper_scale=1.10

Each scenario uses the same spawn transform and the same synchronous
throttle/brake sequence. Profiles are compared against S0, and CSV files are
written under PythonAPI/taeho/results by default.
"""

from __future__ import print_function

import argparse
import csv
import glob
import math
import os
import sys


def add_carla_to_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    platform = 'win-amd64' if os.name == 'nt' else 'linux-x86_64'
    candidates = glob.glob(os.path.join(
        script_dir, '..', 'carla', 'build', 'lib.*'))
    candidates += glob.glob(os.path.join(
        script_dir,
        '..',
        'carla',
        'dist',
        'carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            platform)))

    for path in reversed(candidates):
        sys.path.insert(0, path)


add_carla_to_path()

import carla


TARGET_VEHICLE = 'vehicle.lincoln.mkz_2020'
SUSPENSION_FIELDS = (
    'spring_strength',
    'spring_damper_rate',
    'max_compression',
    'max_droop',
    'sprung_mass',
)


def scenario_definitions():
    return (
        {
            'name': 'S0_stock',
            'label': 'S0 stock',
            'spring_scale': None,
            'damper_scale': None,
            'uses_suspension_api': False,
        },
        {
            'name': 'S1_identity',
            'label': 'S1 identity',
            'spring_scale': 1.0,
            'damper_scale': 1.0,
            'uses_suspension_api': True,
        },
        {
            'name': 'S2_k0.95_csqrt',
            'label': 'S2 k=0.95 c=sqrt(0.95)',
            'spring_scale': 0.95,
            'damper_scale': math.sqrt(0.95),
            'uses_suspension_api': True,
        },
        {
            'name': 'S3_k1.05_csqrt',
            'label': 'S3 k=1.05 c=sqrt(1.05)',
            'spring_scale': 1.05,
            'damper_scale': math.sqrt(1.05),
            'uses_suspension_api': True,
        },
        {
            'name': 'S4_k1.00_c1.10',
            'label': 'S4 k=1.00 c=1.10',
            'spring_scale': 1.0,
            'damper_scale': 1.10,
            'uses_suspension_api': True,
        },
        {
            'name': 'S5_k1.05_c1.10',
            'label': 'S5 k=1.05 c=1.10',
            'spring_scale': 1.05,
            'damper_scale': 1.10,
            'uses_suspension_api': True,
        },
    )


def set_hero_role(blueprint):
    if blueprint.has_attribute('role_name'):
        blueprint.set_attribute('role_name', 'hero')


def find_target_blueprint(world):
    blueprints = list(world.get_blueprint_library().filter(TARGET_VEHICLE))
    if not blueprints:
        raise RuntimeError('target blueprint %s is unavailable' %
                           TARGET_VEHICLE)

    blueprint = blueprints[0]
    set_hero_role(blueprint)
    return blueprint


def validate_suspension(control):
    if len(control.wheels) != 4:
        raise RuntimeError('expected 4 suspension wheels, got %d' %
                           len(control.wheels))

    for index, wheel in enumerate(control.wheels):
        for field in SUSPENSION_FIELDS:
            value = getattr(wheel, field)
            if not math.isfinite(value):
                raise RuntimeError('wheel %d %s is not finite: %r' %
                                   (index, field, value))
            if value <= 0.0:
                raise RuntimeError('wheel %d %s must be positive, got %r' %
                                   (index, field, value))


def scale_suspension(native, spring_scale, damper_scale):
    control = carla.SuspensionPhysicsControl()
    wheels = []
    for wheel in native.wheels:
        wheels.append(carla.WheelSuspensionPhysicsControl(
            spring_strength=wheel.spring_strength * spring_scale,
            spring_damper_rate=wheel.spring_damper_rate * damper_scale,
            max_compression=wheel.max_compression,
            max_droop=wheel.max_droop,
            sprung_mass=wheel.sprung_mass,
        ))
    control.wheels = wheels
    return control


def assert_spring_damper_match(expected, actual, label):
    validate_suspension(actual)
    for index, wheels in enumerate(zip(expected.wheels, actual.wheels)):
        expected_wheel, actual_wheel = wheels
        for field in ('spring_strength', 'spring_damper_rate'):
            expected_value = getattr(expected_wheel, field)
            actual_value = getattr(actual_wheel, field)
            if not math.isclose(
                    expected_value,
                    actual_value,
                    rel_tol=1.0e-5,
                    abs_tol=1.0e-3):
                raise RuntimeError(
                    '%s wheel %d %s mismatch: expected %0.9g, got %0.9g' %
                    (label, index, field, expected_value, actual_value))


def spawn_vehicle(world, blueprint, transform=None, preferred_index=0):
    if transform is not None:
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is None:
            raise RuntimeError('unable to spawn %s at selected transform' %
                               blueprint.id)
        return vehicle, transform

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError('map has no spawn points')

    ordered = spawn_points[:]
    if 0 <= preferred_index < len(spawn_points):
        ordered = spawn_points[preferred_index:] + spawn_points[:preferred_index]

    for candidate in ordered:
        vehicle = world.try_spawn_actor(blueprint, candidate)
        if vehicle is not None:
            return vehicle, candidate

    raise RuntimeError('unable to spawn %s at any map spawn point' %
                       blueprint.id)


def make_control(step, args):
    if step < args.throttle_frames:
        return carla.VehicleControl(throttle=args.throttle)
    return carla.VehicleControl(brake=args.brake)


def vector_norm(vector):
    return math.sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z)


def distance_xyz(a_x, a_y, a_z, b_x, b_y, b_z):
    dx = a_x - b_x
    dy = a_y - b_y
    dz = a_z - b_z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def finite_or_raise(values, label):
    for name, value in values:
        if not isinstance(value, (int, float)):
            continue
        if not math.isfinite(value):
            raise RuntimeError('%s has non-finite %s=%r' %
                               (label, name, value))


def record_state(world, vehicle, scenario_name, step):
    snapshot = world.get_snapshot()
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    angular_velocity = vehicle.get_angular_velocity()
    speed = vector_norm(velocity)

    row = {
        'scenario': scenario_name,
        'step': step,
        'frame': snapshot.frame,
        'elapsed_seconds': snapshot.timestamp.elapsed_seconds,
        'x': transform.location.x,
        'y': transform.location.y,
        'z': transform.location.z,
        'vx': velocity.x,
        'vy': velocity.y,
        'vz': velocity.z,
        'speed': speed,
        'ax': acceleration.x,
        'ay': acceleration.y,
        'az': acceleration.z,
        'roll': transform.rotation.roll,
        'pitch': transform.rotation.pitch,
        'yaw': transform.rotation.yaw,
        'angular_velocity_x': angular_velocity.x,
        'angular_velocity_y': angular_velocity.y,
        'angular_velocity_z': angular_velocity.z,
    }

    finite_or_raise(row.items(), '%s step %d' % (scenario_name, step))
    return row


def destroy_actors(world, actors):
    for actor in actors:
        if actor is not None and hasattr(actor, 'stop'):
            actor.stop()

    for actor in actors:
        if actor is not None:
            actor.destroy()

    world.tick()


def attach_collision_sensor(world, vehicle, scenario_name):
    blueprint = world.get_blueprint_library().find('sensor.other.collision')
    sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
    events = []

    def on_collision(event):
        impulse = event.normal_impulse
        events.append({
            'scenario': scenario_name,
            'frame': event.frame,
            'other_actor': event.other_actor.type_id,
            'impulse': vector_norm(impulse),
        })

    sensor.listen(on_collision)
    print('%s: collision sensor attached id=%d' %
          (scenario_name, sensor.id))
    return sensor, events


def run_scenario(world, blueprint, scenario, args, transform=None):
    vehicle = None
    actors = []

    try:
        vehicle, used_transform = spawn_vehicle(
            world,
            blueprint,
            transform=transform,
            preferred_index=args.spawn_index)
        actors.append(vehicle)

        collision_sensor, collision_events = attach_collision_sensor(
            world,
            vehicle,
            scenario['name'])
        actors.append(collision_sensor)

        world.tick()

        native = None
        applied_control = None
        if scenario['uses_suspension_api']:
            native = vehicle.get_suspension_physics_control()
            validate_suspension(native)
            applied_control = scale_suspension(
                native,
                scenario['spring_scale'],
                scenario['damper_scale'])
            validate_suspension(applied_control)
            vehicle.apply_suspension_physics_control(applied_control)
            world.tick()
            readback = vehicle.get_suspension_physics_control()
            assert_spring_damper_match(
                applied_control,
                readback,
                scenario['name'])
        else:
            world.tick()

        print('%s: vehicle=%s id=%d role_name=%s spring_scale=%s damper_scale=%s' % (
            scenario['label'],
            vehicle.type_id,
            vehicle.id,
            vehicle.attributes.get('role_name', ''),
            'stock' if scenario['spring_scale'] is None else
            '%0.6f' % scenario['spring_scale'],
            'stock' if scenario['damper_scale'] is None else
            '%0.6f' % scenario['damper_scale']))

        profile = []
        for step in range(args.frames):
            vehicle.apply_control(make_control(step, args))
            world.tick()
            row = record_state(world, vehicle, scenario['name'], step)
            profile.append(row)

            if args.log_every > 0 and (
                    step % args.log_every == 0 or step == args.frames - 1):
                print(
                    '%s step=%03d loc=(%0.3f,%0.3f,%0.3f) '
                    'speed=%0.4f roll=%0.4f pitch=%0.4f' % (
                        scenario['label'],
                        step,
                        row['x'],
                        row['y'],
                        row['z'],
                        row['speed'],
                        row['roll'],
                        row['pitch']))

        if native is not None:
            vehicle.apply_suspension_physics_control(native)
            world.tick()

        return {
            'scenario': scenario,
            'transform': used_transform,
            'profile': profile,
            'collisions': collision_events[:],
        }
    finally:
        destroy_actors(world, actors)


def profile_distance(a, b, prefix):
    return distance_xyz(
        a[prefix + 'x'],
        a[prefix + 'y'],
        a[prefix + 'z'],
        b[prefix + 'x'],
        b[prefix + 'y'],
        b[prefix + 'z'])


def compare_to_baseline(baseline, result):
    base_profile = baseline['profile']
    profile = result['profile']
    if len(base_profile) != len(profile):
        raise RuntimeError('%s profile length mismatch: %d vs %d' % (
            result['scenario']['name'],
            len(base_profile),
            len(profile)))

    sum_position_sq = 0.0
    sum_speed_sq = 0.0
    sum_roll_sq = 0.0
    sum_pitch_sq = 0.0
    max_position_diff = 0.0
    max_speed_diff = 0.0
    max_velocity_diff = 0.0
    max_roll_diff = 0.0
    max_pitch_diff = 0.0
    max_position_step = 0
    max_speed_step = 0
    max_roll_step = 0
    max_pitch_step = 0

    for index, rows in enumerate(zip(base_profile, profile)):
        base, row = rows
        position_diff = distance_xyz(
            base['x'], base['y'], base['z'],
            row['x'], row['y'], row['z'])
        speed_diff = abs(base['speed'] - row['speed'])
        velocity_diff = distance_xyz(
            base['vx'], base['vy'], base['vz'],
            row['vx'], row['vy'], row['vz'])
        roll_diff = abs(base['roll'] - row['roll'])
        pitch_diff = abs(base['pitch'] - row['pitch'])

        sum_position_sq += position_diff * position_diff
        sum_speed_sq += speed_diff * speed_diff
        sum_roll_sq += roll_diff * roll_diff
        sum_pitch_sq += pitch_diff * pitch_diff

        if position_diff > max_position_diff:
            max_position_diff = position_diff
            max_position_step = index
        if speed_diff > max_speed_diff:
            max_speed_diff = speed_diff
            max_speed_step = index
        if velocity_diff > max_velocity_diff:
            max_velocity_diff = velocity_diff
        if roll_diff > max_roll_diff:
            max_roll_diff = roll_diff
            max_roll_step = index
        if pitch_diff > max_pitch_diff:
            max_pitch_diff = pitch_diff
            max_pitch_step = index

    count = float(len(profile))
    final_base = base_profile[-1]
    final_row = profile[-1]
    final_position_diff = distance_xyz(
        final_base['x'], final_base['y'], final_base['z'],
        final_row['x'], final_row['y'], final_row['z'])
    final_speed_diff = abs(final_base['speed'] - final_row['speed'])
    final_roll_diff = abs(final_base['roll'] - final_row['roll'])
    final_pitch_diff = abs(final_base['pitch'] - final_row['pitch'])

    return {
        'scenario': result['scenario']['name'],
        'label': result['scenario']['label'],
        'spring_scale': result['scenario']['spring_scale'],
        'damper_scale': result['scenario']['damper_scale'],
        'final_position_diff': final_position_diff,
        'final_speed_diff': final_speed_diff,
        'final_roll_diff': final_roll_diff,
        'final_pitch_diff': final_pitch_diff,
        'max_position_diff': max_position_diff,
        'max_position_step': max_position_step,
        'rms_position_diff': math.sqrt(sum_position_sq / count),
        'max_speed_diff': max_speed_diff,
        'max_speed_step': max_speed_step,
        'rms_speed_diff': math.sqrt(sum_speed_sq / count),
        'max_velocity_diff': max_velocity_diff,
        'max_roll_diff': max_roll_diff,
        'max_roll_step': max_roll_step,
        'rms_roll_diff': math.sqrt(sum_roll_sq / count),
        'max_pitch_diff': max_pitch_diff,
        'max_pitch_step': max_pitch_step,
        'rms_pitch_diff': math.sqrt(sum_pitch_sq / count),
        'collisions': len(result['collisions']),
    }


def print_summary(summary_rows):
    print('')
    print('Scenario comparison against S0 stock:')
    print(
        '%-23s %9s %9s %9s %10s %10s %10s %10s %6s' % (
            'scenario',
            'final_m',
            'max_m',
            'rms_m',
            'max_spd',
            'max_roll',
            'max_pitch',
            'rms_pitch',
            'coll'))
    for row in summary_rows:
        print(
            '%-23s %9.5f %9.5f %9.5f %10.5f %10.5f %10.5f %10.5f %6d' % (
                row['scenario'],
                row['final_position_diff'],
                row['max_position_diff'],
                row['rms_position_diff'],
                row['max_speed_diff'],
                row['max_roll_diff'],
                row['max_pitch_diff'],
                row['rms_pitch_diff'],
                row['collisions']))


def check_identity(summary_rows, args):
    identity = None
    for row in summary_rows:
        if row['scenario'] == 'S1_identity':
            identity = row
            break

    if identity is None:
        raise RuntimeError('missing S1_identity summary row')

    failures = []
    if identity['final_position_diff'] > args.identity_final_position_threshold:
        failures.append(
            'S1 final position diff %0.6f > %0.6f m' % (
                identity['final_position_diff'],
                args.identity_final_position_threshold))
    if identity['max_speed_diff'] > args.identity_speed_threshold:
        failures.append(
            'S1 max speed diff %0.6f > %0.6f m/s' % (
                identity['max_speed_diff'],
                args.identity_speed_threshold))
    if identity['max_roll_diff'] > args.identity_angle_threshold:
        failures.append(
            'S1 max roll diff %0.6f > %0.6f deg' % (
                identity['max_roll_diff'],
                args.identity_angle_threshold))
    if identity['max_pitch_diff'] > args.identity_angle_threshold:
        failures.append(
            'S1 max pitch diff %0.6f > %0.6f deg' % (
                identity['max_pitch_diff'],
                args.identity_angle_threshold))
    if identity['collisions'] > 0:
        failures.append('S1 collision count is %d' % identity['collisions'])

    if failures:
        print('')
        print('FAILED identity guardrail:')
        for failure in failures:
            print('  - %s' % failure)
        raise RuntimeError('identity guardrail failed')

    print('')
    print('PASS: S1 identity stayed equivalent to S0 stock within thresholds.')
    print('S2-S5 are measurement scenarios; nonzero differences are expected.')


def write_csv_outputs(results, summary_rows, args):
    if not args.write_csv:
        return

    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, 'suspension_action_replay_summary.csv')
    profile_path = os.path.join(output_dir, 'suspension_action_replay_profiles.csv')
    collision_path = os.path.join(output_dir, 'suspension_action_replay_collisions.csv')

    summary_fields = (
        'scenario',
        'label',
        'spring_scale',
        'damper_scale',
        'final_position_diff',
        'final_speed_diff',
        'final_roll_diff',
        'final_pitch_diff',
        'max_position_diff',
        'max_position_step',
        'rms_position_diff',
        'max_speed_diff',
        'max_speed_step',
        'rms_speed_diff',
        'max_velocity_diff',
        'max_roll_diff',
        'max_roll_step',
        'rms_roll_diff',
        'max_pitch_diff',
        'max_pitch_step',
        'rms_pitch_diff',
        'collisions',
    )
    with open(summary_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    profile_fields = (
        'scenario',
        'step',
        'frame',
        'elapsed_seconds',
        'x',
        'y',
        'z',
        'vx',
        'vy',
        'vz',
        'speed',
        'ax',
        'ay',
        'az',
        'roll',
        'pitch',
        'yaw',
        'angular_velocity_x',
        'angular_velocity_y',
        'angular_velocity_z',
    )
    with open(profile_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=profile_fields)
        writer.writeheader()
        for result in results:
            for row in result['profile']:
                writer.writerow(row)

    collision_fields = ('scenario', 'frame', 'other_actor', 'impulse')
    with open(collision_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=collision_fields)
        writer.writeheader()
        for result in results:
            for event in result['collisions']:
                writer.writerow(event)

    print('')
    print('Wrote CSV outputs:')
    print('  %s' % summary_path)
    print('  %s' % profile_path)
    print('  %s' % collision_path)


def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    original_settings = world.get_settings()

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta_seconds
        world.apply_settings(settings)

        blueprint = find_target_blueprint(world)
        scenarios = scenario_definitions()
        results = []
        reference_transform = None

        for scenario in scenarios:
            result = run_scenario(
                world,
                blueprint,
                scenario,
                args,
                transform=reference_transform)
            if reference_transform is None:
                reference_transform = result['transform']
            results.append(result)

        baseline = results[0]
        summary_rows = [compare_to_baseline(baseline, result)
                        for result in results]
        print_summary(summary_rows)
        check_identity(summary_rows, args)
        write_csv_outputs(results, summary_rows, args)
    finally:
        world.apply_settings(original_settings)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p',
        '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--timeout',
        default=30.0,
        type=float,
        help='client timeout in seconds (default: 30.0)')
    argparser.add_argument(
        '--frames',
        default=200,
        type=int,
        help='number of synchronous ticks per scenario (default: 200)')
    argparser.add_argument(
        '--throttle-frames',
        default=120,
        type=int,
        help='ticks using throttle before braking (default: 120)')
    argparser.add_argument(
        '--throttle',
        default=0.3,
        type=float,
        help='throttle value for the first phase (default: 0.3)')
    argparser.add_argument(
        '--brake',
        default=0.3,
        type=float,
        help='brake value for the second phase (default: 0.3)')
    argparser.add_argument(
        '--fixed-delta-seconds',
        default=0.05,
        type=float,
        help='synchronous fixed delta seconds (default: 0.05)')
    argparser.add_argument(
        '--spawn-index',
        default=0,
        type=int,
        help='preferred map spawn point index (default: 0)')
    argparser.add_argument(
        '--log-every',
        default=20,
        type=int,
        help='state log period in ticks; 0 disables logs (default: 20)')
    argparser.add_argument(
        '--identity-final-position-threshold',
        default=0.10,
        type=float,
        help='allowed S1 final location difference in meters (default: 0.10)')
    argparser.add_argument(
        '--identity-speed-threshold',
        default=0.05,
        type=float,
        help='allowed S1 max speed profile difference in m/s (default: 0.05)')
    argparser.add_argument(
        '--identity-angle-threshold',
        default=0.05,
        type=float,
        help='allowed S1 max roll/pitch difference in degrees (default: 0.05)')
    argparser.add_argument(
        '--output-dir',
        default='results',
        help='CSV output directory, relative to this script unless absolute '
             '(default: results)')
    argparser.add_argument(
        '--no-write-csv',
        dest='write_csv',
        action='store_false',
        help='disable CSV output')
    argparser.set_defaults(write_csv=True)

    try:
        main(argparser.parse_args())
    except KeyboardInterrupt:
        print(' - Exited by user.')

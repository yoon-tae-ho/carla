#!/usr/bin/env python

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Run lateral maneuvers with static PhysX suspension scale variants.

This script extends the straight action replay tests with lateral excitation:

  * constant steer circle sweeps
  * sine steer response
  * brake-in-turn coupling

Each maneuver is replayed with S0-S5 suspension variants. S0 is stock CARLA
with no suspension API call; S1 is identity; S2-S5 are static scale variants.
Profiles and summary metrics are written under PythonAPI/taeho/results by
default.
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


def parse_float_list(value):
    if value.strip() == '':
        return []
    return [float(item.strip()) for item in value.split(',')]


def maneuver_definitions(args):
    maneuvers = []

    for steer in parse_float_list(args.circle_steers):
        maneuvers.append({
            'name': 'circle_steer_%0.2f' % steer,
            'type': 'circle',
            'frames': args.circle_frames,
            'throttle': args.circle_throttle,
            'steer': steer,
            'brake': 0.0,
            'sine_amplitude': 0.0,
            'sine_period': 0.0,
            'brake_start': -1,
        })

    for amplitude in parse_float_list(args.sine_amplitudes):
        maneuvers.append({
            'name': 'sine_steer_A%0.2f_T%0.2f' % (
                amplitude,
                args.sine_period),
            'type': 'sine',
            'frames': args.sine_frames,
            'throttle': args.sine_throttle,
            'steer': 0.0,
            'brake': 0.0,
            'sine_amplitude': amplitude,
            'sine_period': args.sine_period,
            'brake_start': -1,
        })

    maneuvers.append({
        'name': 'brake_in_turn_steer_%0.2f' % args.brake_turn_steer,
        'type': 'brake_in_turn',
        'frames': args.brake_turn_frames,
        'throttle': args.brake_turn_throttle,
        'steer': args.brake_turn_steer,
        'brake': args.brake_turn_brake,
        'sine_amplitude': 0.0,
        'sine_period': 0.0,
        'brake_start': args.brake_turn_brake_start,
    })

    return maneuvers


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


def make_control(step, maneuver, fixed_delta_seconds):
    if maneuver['type'] == 'sine':
        t = step * fixed_delta_seconds
        steer = maneuver['sine_amplitude'] * math.sin(
            2.0 * math.pi * t / maneuver['sine_period'])
        return carla.VehicleControl(
            throttle=maneuver['throttle'],
            steer=steer)

    if maneuver['type'] == 'brake_in_turn':
        if step < maneuver['brake_start']:
            return carla.VehicleControl(
                throttle=maneuver['throttle'],
                steer=maneuver['steer'])
        return carla.VehicleControl(
            brake=maneuver['brake'],
            steer=maneuver['steer'])

    return carla.VehicleControl(
        throttle=maneuver['throttle'],
        steer=maneuver['steer'])


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


def to_local_xy(world_x, world_y, yaw_degrees):
    yaw = math.radians(yaw_degrees)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    local_x = cos_yaw * world_x + sin_yaw * world_y
    local_y = -sin_yaw * world_x + cos_yaw * world_y
    return local_x, local_y


def record_state(world, vehicle, scenario_name, maneuver, control, step):
    snapshot = world.get_snapshot()
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    angular_velocity = vehicle.get_angular_velocity()
    speed = vector_norm(velocity)
    local_vx, local_vy = to_local_xy(
        velocity.x,
        velocity.y,
        transform.rotation.yaw)
    local_ax, local_ay = to_local_xy(
        acceleration.x,
        acceleration.y,
        transform.rotation.yaw)

    row = {
        'scenario': scenario_name,
        'maneuver': maneuver['name'],
        'maneuver_type': maneuver['type'],
        'step': step,
        'frame': snapshot.frame,
        'elapsed_seconds': snapshot.timestamp.elapsed_seconds,
        'throttle': control.throttle,
        'brake': control.brake,
        'steer': control.steer,
        'x': transform.location.x,
        'y': transform.location.y,
        'z': transform.location.z,
        'vx': velocity.x,
        'vy': velocity.y,
        'vz': velocity.z,
        'speed': speed,
        'local_vx': local_vx,
        'local_vy': local_vy,
        'ax': acceleration.x,
        'ay': acceleration.y,
        'az': acceleration.z,
        'local_ax': local_ax,
        'local_ay': local_ay,
        'roll': transform.rotation.roll,
        'pitch': transform.rotation.pitch,
        'yaw': transform.rotation.yaw,
        'roll_rate': angular_velocity.x,
        'pitch_rate': angular_velocity.y,
        'yaw_rate': angular_velocity.z,
    }

    finite_or_raise(row.items(), '%s %s step %d' % (
        scenario_name,
        maneuver['name'],
        step))
    return row


def destroy_actors(world, actors):
    for actor in actors:
        if actor is not None and hasattr(actor, 'stop'):
            actor.stop()

    for actor in actors:
        if actor is not None:
            actor.destroy()

    world.tick()


def attach_collision_sensor(world, vehicle, scenario_name, maneuver_name):
    blueprint = world.get_blueprint_library().find('sensor.other.collision')
    sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
    events = []

    def on_collision(event):
        impulse = event.normal_impulse
        events.append({
            'scenario': scenario_name,
            'maneuver': maneuver_name,
            'frame': event.frame,
            'other_actor': event.other_actor.type_id,
            'impulse': vector_norm(impulse),
        })

    sensor.listen(on_collision)
    print('%s %s: collision sensor attached id=%d' %
          (scenario_name, maneuver_name, sensor.id))
    return sensor, events


def apply_static_suspension(world, vehicle, scenario):
    if not scenario['uses_suspension_api']:
        world.tick()
        return None

    native = vehicle.get_suspension_physics_control()
    validate_suspension(native)
    command = scale_suspension(
        native,
        scenario['spring_scale'],
        scenario['damper_scale'])
    validate_suspension(command)
    vehicle.apply_suspension_physics_control(command)
    world.tick()
    readback = vehicle.get_suspension_physics_control()
    assert_spring_damper_match(command, readback, scenario['name'])
    return native


def run_trial(world, blueprint, scenario, maneuver, args, transform=None):
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
            scenario['name'],
            maneuver['name'])
        actors.append(collision_sensor)

        world.tick()
        native = apply_static_suspension(world, vehicle, scenario)

        print('%s %s: vehicle=%s id=%d spring_scale=%s damper_scale=%s' % (
            scenario['label'],
            maneuver['name'],
            vehicle.type_id,
            vehicle.id,
            'stock' if scenario['spring_scale'] is None else
            '%0.6f' % scenario['spring_scale'],
            'stock' if scenario['damper_scale'] is None else
            '%0.6f' % scenario['damper_scale']))

        profile = []
        for step in range(maneuver['frames']):
            control = make_control(step, maneuver, args.fixed_delta_seconds)
            vehicle.apply_control(control)
            world.tick()
            row = record_state(
                world,
                vehicle,
                scenario['name'],
                maneuver,
                control,
                step)
            profile.append(row)

            if args.log_every > 0 and (
                    step % args.log_every == 0 or
                    step == maneuver['frames'] - 1):
                print(
                    '%s %s step=%03d speed=%0.4f roll=%0.4f '
                    'pitch=%0.4f yaw_rate=%0.4f lat_acc=%0.4f steer=%0.4f' % (
                        scenario['label'],
                        maneuver['name'],
                        step,
                        row['speed'],
                        row['roll'],
                        row['pitch'],
                        row['yaw_rate'],
                        row['local_ay'],
                        row['steer']))

        if native is not None:
            vehicle.apply_suspension_physics_control(native)
            world.tick()

        return {
            'scenario': scenario,
            'maneuver': maneuver,
            'transform': used_transform,
            'profile': profile,
            'collisions': collision_events[:],
        }
    finally:
        destroy_actors(world, actors)


def rms(values):
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / float(len(values)))


def mean(values):
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def steady_slice(profile, steady_fraction):
    if not profile:
        return []
    start = int(len(profile) * (1.0 - steady_fraction))
    return profile[max(0, start):]


def base_metrics(result, args):
    profile = result['profile']
    steady = steady_slice(profile, args.steady_fraction)
    return {
        'scenario': result['scenario']['name'],
        'maneuver': result['maneuver']['name'],
        'maneuver_type': result['maneuver']['type'],
        'spring_scale': result['scenario']['spring_scale'],
        'damper_scale': result['scenario']['damper_scale'],
        'collisions': len(result['collisions']),
        'mean_speed_tail': mean([row['speed'] for row in steady]),
        'mean_abs_roll_tail': mean([abs(row['roll']) for row in steady]),
        'mean_abs_pitch_tail': mean([abs(row['pitch']) for row in steady]),
        'mean_abs_yaw_rate_tail': mean([abs(row['yaw_rate']) for row in steady]),
        'mean_abs_lateral_acc_tail': mean([abs(row['local_ay']) for row in steady]),
        'peak_abs_roll': max(abs(row['roll']) for row in profile),
        'peak_abs_pitch': max(abs(row['pitch']) for row in profile),
        'peak_abs_yaw_rate': max(abs(row['yaw_rate']) for row in profile),
        'peak_abs_lateral_acc': max(abs(row['local_ay']) for row in profile),
        'rms_roll': rms([row['roll'] for row in profile]),
        'rms_pitch': rms([row['pitch'] for row in profile]),
        'rms_yaw_rate': rms([row['yaw_rate'] for row in profile]),
        'rms_lateral_acc': rms([row['local_ay'] for row in profile]),
    }


def compare_to_baseline(baseline, result, args):
    if len(baseline['profile']) != len(result['profile']):
        raise RuntimeError('%s %s profile length mismatch' % (
            result['scenario']['name'],
            result['maneuver']['name']))

    summary = base_metrics(result, args)
    position_diffs = []
    speed_diffs = []
    roll_diffs = []
    pitch_diffs = []
    yaw_rate_diffs = []
    lateral_acc_diffs = []

    for base, row in zip(baseline['profile'], result['profile']):
        position_diffs.append(distance_xyz(
            base['x'], base['y'], base['z'],
            row['x'], row['y'], row['z']))
        speed_diffs.append(abs(base['speed'] - row['speed']))
        roll_diffs.append(abs(base['roll'] - row['roll']))
        pitch_diffs.append(abs(base['pitch'] - row['pitch']))
        yaw_rate_diffs.append(abs(base['yaw_rate'] - row['yaw_rate']))
        lateral_acc_diffs.append(abs(base['local_ay'] - row['local_ay']))

    base_final = baseline['profile'][-1]
    final = result['profile'][-1]
    summary.update({
        'final_position_diff': distance_xyz(
            base_final['x'], base_final['y'], base_final['z'],
            final['x'], final['y'], final['z']),
        'max_position_diff': max(position_diffs),
        'rms_position_diff': rms(position_diffs),
        'max_speed_diff': max(speed_diffs),
        'rms_speed_diff': rms(speed_diffs),
        'max_roll_diff': max(roll_diffs),
        'rms_roll_diff': rms(roll_diffs),
        'max_pitch_diff': max(pitch_diffs),
        'rms_pitch_diff': rms(pitch_diffs),
        'max_yaw_rate_diff': max(yaw_rate_diffs),
        'rms_yaw_rate_diff': rms(yaw_rate_diffs),
        'max_lateral_acc_diff': max(lateral_acc_diffs),
        'rms_lateral_acc_diff': rms(lateral_acc_diffs),
    })
    return summary


def build_summary(results, args):
    by_maneuver = {}
    for result in results:
        by_maneuver.setdefault(result['maneuver']['name'], []).append(result)

    summary = []
    for maneuver_name, maneuver_results in by_maneuver.items():
        baseline = None
        for result in maneuver_results:
            if result['scenario']['name'] == 'S0_stock':
                baseline = result
                break
        if baseline is None:
            raise RuntimeError('missing S0_stock for %s' % maneuver_name)

        for result in maneuver_results:
            summary.append(compare_to_baseline(baseline, result, args))

    return summary


def print_summary(summary_rows):
    print('')
    print('Lateral maneuver summary against matching S0 stock baseline:')
    print(
        '%-24s %-20s %9s %9s %9s %9s %9s %9s %5s' % (
            'maneuver',
            'scenario',
            'peak_roll',
            'peak_yaw',
            'peak_lat',
            'max_roll',
            'max_yaw',
            'max_lat',
            'coll'))
    for row in summary_rows:
        print(
            '%-24s %-20s %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %5d' % (
                row['maneuver'],
                row['scenario'],
                row['peak_abs_roll'],
                row['peak_abs_yaw_rate'],
                row['peak_abs_lateral_acc'],
                row['max_roll_diff'],
                row['max_yaw_rate_diff'],
                row['max_lateral_acc_diff'],
                row['collisions']))


def check_identity(summary_rows, args):
    failures = []
    for row in summary_rows:
        if row['scenario'] != 'S1_identity':
            continue
        if row['final_position_diff'] > args.identity_final_position_threshold:
            failures.append('%s final position diff %0.6f > %0.6f m' % (
                row['maneuver'],
                row['final_position_diff'],
                args.identity_final_position_threshold))
        if row['max_speed_diff'] > args.identity_speed_threshold:
            failures.append('%s max speed diff %0.6f > %0.6f m/s' % (
                row['maneuver'],
                row['max_speed_diff'],
                args.identity_speed_threshold))
        if row['max_roll_diff'] > args.identity_angle_threshold:
            failures.append('%s max roll diff %0.6f > %0.6f deg' % (
                row['maneuver'],
                row['max_roll_diff'],
                args.identity_angle_threshold))
        if row['max_pitch_diff'] > args.identity_angle_threshold:
            failures.append('%s max pitch diff %0.6f > %0.6f deg' % (
                row['maneuver'],
                row['max_pitch_diff'],
                args.identity_angle_threshold))
        if row['max_yaw_rate_diff'] > args.identity_yaw_rate_threshold:
            failures.append('%s max yaw-rate diff %0.6f > %0.6f deg/s' % (
                row['maneuver'],
                row['max_yaw_rate_diff'],
                args.identity_yaw_rate_threshold))
        if row['collisions'] > 0:
            failures.append('%s S1 collision count is %d' % (
                row['maneuver'],
                row['collisions']))

    if failures:
        print('')
        print('FAILED identity guardrail:')
        for failure in failures:
            print('  - %s' % failure)
        raise RuntimeError('identity guardrail failed')

    print('')
    print('PASS: S1 identity matched S0 stock for all lateral maneuvers.')
    print('S2-S5 are measurement scenarios; lateral response differences are expected.')


def output_dir_path(args):
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, output_dir)
    return output_dir


def write_csv_outputs(results, summary_rows, args):
    if not args.write_csv:
        return

    output_dir = output_dir_path(args)
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, 'suspension_lateral_summary.csv')
    profile_path = os.path.join(output_dir, 'suspension_lateral_profiles.csv')
    collision_path = os.path.join(output_dir, 'suspension_lateral_collisions.csv')

    summary_fields = (
        'maneuver',
        'maneuver_type',
        'scenario',
        'spring_scale',
        'damper_scale',
        'collisions',
        'mean_speed_tail',
        'mean_abs_roll_tail',
        'mean_abs_pitch_tail',
        'mean_abs_yaw_rate_tail',
        'mean_abs_lateral_acc_tail',
        'peak_abs_roll',
        'peak_abs_pitch',
        'peak_abs_yaw_rate',
        'peak_abs_lateral_acc',
        'rms_roll',
        'rms_pitch',
        'rms_yaw_rate',
        'rms_lateral_acc',
        'final_position_diff',
        'max_position_diff',
        'rms_position_diff',
        'max_speed_diff',
        'rms_speed_diff',
        'max_roll_diff',
        'rms_roll_diff',
        'max_pitch_diff',
        'rms_pitch_diff',
        'max_yaw_rate_diff',
        'rms_yaw_rate_diff',
        'max_lateral_acc_diff',
        'rms_lateral_acc_diff',
    )
    with open(summary_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    profile_fields = (
        'scenario',
        'maneuver',
        'maneuver_type',
        'step',
        'frame',
        'elapsed_seconds',
        'throttle',
        'brake',
        'steer',
        'x',
        'y',
        'z',
        'vx',
        'vy',
        'vz',
        'speed',
        'local_vx',
        'local_vy',
        'ax',
        'ay',
        'az',
        'local_ax',
        'local_ay',
        'roll',
        'pitch',
        'yaw',
        'roll_rate',
        'pitch_rate',
        'yaw_rate',
    )
    with open(profile_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=profile_fields)
        writer.writeheader()
        for result in results:
            for row in result['profile']:
                writer.writerow(row)

    collision_fields = ('scenario', 'maneuver', 'frame', 'other_actor', 'impulse')
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
        maneuvers = maneuver_definitions(args)
        results = []
        reference_transform = None

        for maneuver in maneuvers:
            print('')
            print('=== Maneuver: %s (%s), frames=%d ===' % (
                maneuver['name'],
                maneuver['type'],
                maneuver['frames']))
            for scenario in scenarios:
                result = run_trial(
                    world,
                    blueprint,
                    scenario,
                    maneuver,
                    args,
                    transform=reference_transform)
                if reference_transform is None:
                    reference_transform = result['transform']
                results.append(result)

        summary_rows = build_summary(results, args)
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
        default=50,
        type=int,
        help='state log period in ticks; 0 disables logs (default: 50)')
    argparser.add_argument(
        '--circle-steers',
        default='0.10,0.20,0.30',
        help='comma-separated constant steer values (default: 0.10,0.20,0.30)')
    argparser.add_argument(
        '--circle-frames',
        default=260,
        type=int,
        help='ticks for each constant-steer maneuver (default: 260)')
    argparser.add_argument(
        '--circle-throttle',
        default=0.25,
        type=float,
        help='throttle for constant-steer maneuvers (default: 0.25)')
    argparser.add_argument(
        '--sine-amplitudes',
        default='0.20',
        help='comma-separated sine steer amplitudes (default: 0.20)')
    argparser.add_argument(
        '--sine-period',
        default=4.0,
        type=float,
        help='sine steer period in seconds (default: 4.0)')
    argparser.add_argument(
        '--sine-frames',
        default=300,
        type=int,
        help='ticks for each sine-steer maneuver (default: 300)')
    argparser.add_argument(
        '--sine-throttle',
        default=0.30,
        type=float,
        help='throttle for sine-steer maneuvers (default: 0.30)')
    argparser.add_argument(
        '--brake-turn-frames',
        default=240,
        type=int,
        help='ticks for brake-in-turn maneuver (default: 240)')
    argparser.add_argument(
        '--brake-turn-brake-start',
        default=120,
        type=int,
        help='tick when brake replaces throttle in brake-in-turn (default: 120)')
    argparser.add_argument(
        '--brake-turn-steer',
        default=0.20,
        type=float,
        help='constant steer for brake-in-turn (default: 0.20)')
    argparser.add_argument(
        '--brake-turn-throttle',
        default=0.30,
        type=float,
        help='initial throttle for brake-in-turn (default: 0.30)')
    argparser.add_argument(
        '--brake-turn-brake',
        default=0.35,
        type=float,
        help='brake value after brake-start tick (default: 0.35)')
    argparser.add_argument(
        '--steady-fraction',
        default=0.40,
        type=float,
        help='tail fraction used for steady-state means (default: 0.40)')
    argparser.add_argument(
        '--identity-final-position-threshold',
        default=0.15,
        type=float,
        help='allowed S1 final location difference in meters (default: 0.15)')
    argparser.add_argument(
        '--identity-speed-threshold',
        default=0.05,
        type=float,
        help='allowed S1 max speed difference in m/s (default: 0.05)')
    argparser.add_argument(
        '--identity-angle-threshold',
        default=0.05,
        type=float,
        help='allowed S1 max roll/pitch difference in degrees (default: 0.05)')
    argparser.add_argument(
        '--identity-yaw-rate-threshold',
        default=0.05,
        type=float,
        help='allowed S1 max yaw-rate difference in deg/s (default: 0.05)')
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

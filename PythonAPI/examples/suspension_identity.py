#!/usr/bin/env python

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Repeatedly apply the native PhysX suspension values to verify identity mode.

Use get_suspension_physics_control() as the baseline before writing runtime
suspension values. The identity path writes the raw spring_strength and
spring_damper_rate values returned by PhysX, separate from
VehiclePhysicsControl and without recreating the vehicle physics state.
"""

from __future__ import print_function

import argparse
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
WHEEL_NAMES = ('front_left', 'front_right', 'rear_left', 'rear_right')
SUSPENSION_FIELDS = (
    'spring_strength',
    'spring_damper_rate',
    'max_compression',
    'max_droop',
    'sprung_mass',
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


def spawn_vehicle(world, blueprint):
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError('map has no spawn points')

    for transform in spawn_points:
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            return vehicle

    raise RuntimeError('unable to spawn %s at any map spawn point' %
                       blueprint.id)


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


def assert_finite_state(vehicle):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    values = (
        transform.location.x,
        transform.location.y,
        transform.location.z,
        transform.rotation.roll,
        transform.rotation.pitch,
        transform.rotation.yaw,
        velocity.x,
        velocity.y,
        velocity.z,
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError('vehicle state contains a non-finite value')


def suspension_summary(control):
    return ' '.join('%s=(%0.6g,%0.6g)' % (
        WHEEL_NAMES[index] if index < len(WHEEL_NAMES) else str(index),
        wheel.spring_strength,
        wheel.spring_damper_rate) for index, wheel in enumerate(control.wheels))


def log_vehicle_state(world, vehicle, control, step):
    snapshot = world.get_snapshot()
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    print(
        'step=%03d frame=%d t=%0.3f loc=(%0.3f,%0.3f,%0.3f) '
        'vel=(%0.3f,%0.3f,%0.3f) roll=%0.3f pitch=%0.3f suspension=%s' % (
            step,
            snapshot.frame,
            snapshot.timestamp.elapsed_seconds,
            transform.location.x,
            transform.location.y,
            transform.location.z,
            velocity.x,
            velocity.y,
            velocity.z,
            transform.rotation.roll,
            transform.rotation.pitch,
            suspension_summary(control)))


def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    vehicle = None
    original_settings = world.get_settings()

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint = find_target_blueprint(world)
        vehicle = spawn_vehicle(world, blueprint)
        world.tick()

        native = vehicle.get_suspension_physics_control()
        validate_suspension(native)
        print('Vehicle: %s id=%d role_name=%s' % (
            vehicle.type_id,
            vehicle.id,
            vehicle.attributes.get('role_name', '')))

        vehicle.apply_suspension_physics_control(native)
        world.tick()
        after = vehicle.get_suspension_physics_control()
        assert_spring_damper_match(native, after, 'initial identity')
        print('Initial identity read/write matched native spring/damper values.')
        log_vehicle_state(world, vehicle, after, 0)

        for step in range(1, args.frames + 1):
            vehicle.apply_suspension_physics_control(native)
            if step < args.frames * 0.6:
                vehicle.apply_control(carla.VehicleControl(throttle=0.3))
            else:
                vehicle.apply_control(carla.VehicleControl(brake=0.3))

            world.tick()
            assert_finite_state(vehicle)

            if step % args.log_every == 0 or step == args.frames:
                current = vehicle.get_suspension_physics_control()
                assert_spring_damper_match(native, current, 'identity loop')
                log_vehicle_state(world, vehicle, current, step)

        final_control = vehicle.get_suspension_physics_control()
        assert_spring_damper_match(native, final_control, 'final identity')
        print('Repeated identity apply completed for %d ticks.' % args.frames)
    finally:
        world.apply_settings(original_settings)
        if vehicle is not None:
            vehicle.destroy()


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
        default=10.0,
        type=float,
        help='client timeout in seconds (default: 10.0)')
    argparser.add_argument(
        '--frames',
        default=200,
        type=int,
        help='number of synchronous identity ticks (default: 200)')
    argparser.add_argument(
        '--log-every',
        default=20,
        type=int,
        help='state log period in ticks (default: 20)')

    try:
        main(argparser.parse_args())
    except KeyboardInterrupt:
        print(' - Exited by user.')

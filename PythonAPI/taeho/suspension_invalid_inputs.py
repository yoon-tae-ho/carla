#!/usr/bin/env python

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Verify that invalid runtime PhysX suspension commands are rejected.

This script starts from the native suspension values returned by
get_suspension_physics_control(), mutates one invalid field at a time, and
checks that apply_suspension_physics_control() rejects or ignores the command
without changing the native spring/damper values.
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


def validate_native_suspension(control):
    if len(control.wheels) != 4:
        raise RuntimeError('expected 4 suspension wheels, got %d' %
                           len(control.wheels))

    for index, wheel in enumerate(control.wheels):
        for field in SUSPENSION_FIELDS:
            value = getattr(wheel, field)
            if not math.isfinite(value):
                raise RuntimeError('native wheel %d %s is not finite: %r' %
                                   (index, field, value))
            if value <= 0.0:
                raise RuntimeError('native wheel %d %s must be positive, got %r' %
                                   (index, field, value))


def clone_wheel(wheel):
    return carla.WheelSuspensionPhysicsControl(
        spring_strength=wheel.spring_strength,
        spring_damper_rate=wheel.spring_damper_rate,
        max_compression=wheel.max_compression,
        max_droop=wheel.max_droop,
        sprung_mass=wheel.sprung_mass,
    )


def clone_control(control):
    cloned = carla.SuspensionPhysicsControl()
    cloned.wheels = [clone_wheel(wheel) for wheel in control.wheels]
    return cloned


def make_bad_field_control(native, wheel_index, field, value):
    control = clone_control(native)
    setattr(control.wheels[wheel_index], field, value)
    return control


def make_bad_count_control(native, wheel_count):
    control = carla.SuspensionPhysicsControl()
    wheels = [clone_wheel(wheel) for wheel in native.wheels]

    if wheel_count < len(wheels):
        control.wheels = wheels[:wheel_count]
    else:
        while len(wheels) < wheel_count:
            wheels.append(clone_wheel(native.wheels[-1]))
        control.wheels = wheels

    return control


def assert_spring_damper_match(expected, actual, label):
    if len(actual.wheels) != len(expected.wheels):
        raise RuntimeError('%s wheel count changed: expected %d, got %d' %
                           (label, len(expected.wheels), len(actual.wheels)))

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
                    '%s wheel %d %s changed: expected %0.9g, got %0.9g' %
                    (label, index, field, expected_value, actual_value))


def apply_native_and_tick(world, vehicle, native):
    vehicle.apply_suspension_physics_control(native)
    world.tick()


def run_rejection_case(world, vehicle, native, label, control):
    before = vehicle.get_suspension_physics_control()
    assert_spring_damper_match(native, before, '%s before' % label)

    try:
        vehicle.apply_suspension_physics_control(control)
    except RuntimeError as error:
        print('[PASS] %-32s rejected: %s' % (label, error))
    else:
        after_no_error = vehicle.get_suspension_physics_control()
        assert_spring_damper_match(
            native,
            after_no_error,
            '%s after non-throwing apply' % label)
        print('[PASS] %-32s ignored without changing suspension' % label)
    finally:
        apply_native_and_tick(world, vehicle, native)

    after = vehicle.get_suspension_physics_control()
    assert_spring_damper_match(native, after, '%s final' % label)


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
        validate_native_suspension(native)
        print('Vehicle: %s id=%d role_name=%s' % (
            vehicle.type_id,
            vehicle.id,
            vehicle.attributes.get('role_name', '')))
        print('Native suspension captured; running invalid input cases.')

        cases = (
            ('nan spring_strength',
             make_bad_field_control(native, 0, 'spring_strength', float('nan'))),
            ('inf spring_damper_rate',
             make_bad_field_control(native, 1, 'spring_damper_rate', float('inf'))),
            ('zero spring_strength',
             make_bad_field_control(native, 2, 'spring_strength', 0.0)),
            ('negative spring_strength',
             make_bad_field_control(native, 3, 'spring_strength', -1.0)),
            ('zero spring_damper_rate',
             make_bad_field_control(native, 0, 'spring_damper_rate', 0.0)),
            ('negative spring_damper_rate',
             make_bad_field_control(native, 1, 'spring_damper_rate', -1.0)),
            ('too few wheels',
             make_bad_count_control(native, 3)),
            ('too many wheels',
             make_bad_count_control(native, 5)),
        )

        for label, control in cases:
            run_rejection_case(world, vehicle, native, label, control)

        final_control = vehicle.get_suspension_physics_control()
        assert_spring_damper_match(native, final_control, 'final native check')
        print('Invalid suspension input rejection completed: %d cases passed.' %
              len(cases))
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

    try:
        main(argparser.parse_args())
    except KeyboardInterrupt:
        print(' - Exited by user.')

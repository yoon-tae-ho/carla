#!/usr/bin/env python

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Read raw PhysX suspension values from a CARLA vehicle.

The returned carla.SuspensionPhysicsControl contains
carla.WheelSuspensionPhysicsControl entries in wheel order FL, FR, RL/BL,
RR/BR. This API is separate from VehiclePhysicsControl and directly reads
PhysX suspension data without recreating the vehicle physics state.
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


def blueprint_wheel_count(blueprint):
    if not blueprint.has_attribute('number_of_wheels'):
        return None
    return int(blueprint.get_attribute('number_of_wheels'))


def set_hero_role(blueprint):
    if blueprint.has_attribute('role_name'):
        blueprint.set_attribute('role_name', 'hero')


def find_vehicle_blueprint(world):
    library = world.get_blueprint_library()
    target = list(library.filter(TARGET_VEHICLE))
    if target:
        blueprint = target[0]
        set_hero_role(blueprint)
        return blueprint

    candidates = []
    for blueprint in library.filter('vehicle.*'):
        if blueprint_wheel_count(blueprint) == 4:
            candidates.append(blueprint)

    if not candidates:
        raise RuntimeError('no 4-wheel vehicle blueprint is available')

    blueprint = sorted(candidates, key=lambda bp: bp.id)[0]
    set_hero_role(blueprint)
    print('Target blueprint %s unavailable; using %s' %
          (TARGET_VEHICLE, blueprint.id))
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


def tick_or_wait(world):
    if world.get_settings().synchronous_mode:
        world.tick()
    else:
        world.wait_for_tick()


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


def print_suspension(control):
    print('Suspension wheel count: %d' % len(control.wheels))
    print('Wheel order: FL, FR, RL/BL, RR/BR')
    for index, wheel in enumerate(control.wheels):
        print(
            '  %d %-11s spring_strength=%0.9g spring_damper_rate=%0.9g '
            'max_compression=%0.9g max_droop=%0.9g sprung_mass=%0.9g' % (
                index,
                WHEEL_NAMES[index] if index < len(WHEEL_NAMES) else 'unknown',
                wheel.spring_strength,
                wheel.spring_damper_rate,
                wheel.max_compression,
                wheel.max_droop,
                wheel.sprung_mass))


def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    vehicle = None

    try:
        blueprint = find_vehicle_blueprint(world)
        vehicle = spawn_vehicle(world, blueprint)
        tick_or_wait(world)

        control = vehicle.get_suspension_physics_control()
        validate_suspension(control)
        print('Vehicle: %s id=%d role_name=%s' % (
            vehicle.type_id,
            vehicle.id,
            vehicle.attributes.get('role_name', '')))
        print_suspension(control)
    finally:
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

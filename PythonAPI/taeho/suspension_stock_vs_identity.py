#!/usr/bin/env python

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Compare stock PhysX behavior against per-tick suspension identity writes.

S0 runs the target vehicle without calling the suspension API.
S1 spawns the same vehicle at the same transform, reads native suspension, and
calls apply_suspension_physics_control(native) every tick while replaying the
same throttle/brake control sequence.
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


def distance(location_a, location_b):
    dx = location_a.x - location_b.x
    dy = location_a.y - location_b.y
    dz = location_a.z - location_b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def finite_or_raise(values, label):
    for name, value in values:
        if not math.isfinite(value):
            raise RuntimeError('%s has non-finite %s=%r' %
                               (label, name, value))


def record_state(vehicle, step):
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    speed = vector_norm(velocity)
    state = {
        'step': step,
        'location': transform.location,
        'velocity': velocity,
        'speed': speed,
        'roll': transform.rotation.roll,
        'pitch': transform.rotation.pitch,
        'yaw': transform.rotation.yaw,
    }
    finite_or_raise((
        ('location.x', transform.location.x),
        ('location.y', transform.location.y),
        ('location.z', transform.location.z),
        ('velocity.x', velocity.x),
        ('velocity.y', velocity.y),
        ('velocity.z', velocity.z),
        ('speed', speed),
        ('roll', transform.rotation.roll),
        ('pitch', transform.rotation.pitch),
        ('yaw', transform.rotation.yaw),
    ), 'step %d' % step)
    return state


def destroy_actors(world, actors):
    for actor in actors:
        if actor is not None and hasattr(actor, 'stop'):
            actor.stop()

    for actor in actors:
        if actor is not None:
            actor.destroy()

    world.tick()


def attach_collision_sensor(world, vehicle, case_name):
    blueprint = world.get_blueprint_library().find('sensor.other.collision')
    sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=vehicle)
    events = []

    def on_collision(event):
        impulse = event.normal_impulse
        events.append({
            'frame': event.frame,
            'other_actor': event.other_actor.type_id,
            'impulse': vector_norm(impulse),
        })

    sensor.listen(on_collision)
    print('%s: collision sensor attached id=%d' % (case_name, sensor.id))
    return sensor, events


def run_case(world, blueprint, args, case_name, use_identity, transform=None):
    vehicle = None
    collision_sensor = None
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
            case_name)
        actors.append(collision_sensor)

        world.tick()

        native = None
        if use_identity:
            native = vehicle.get_suspension_physics_control()
            validate_suspension(native)

        print('%s: vehicle=%s id=%d role_name=%s identity=%s' % (
            case_name,
            vehicle.type_id,
            vehicle.id,
            vehicle.attributes.get('role_name', ''),
            use_identity))

        profile = []
        for step in range(args.frames):
            if use_identity:
                vehicle.apply_suspension_physics_control(native)

            vehicle.apply_control(make_control(step, args))
            world.tick()
            profile.append(record_state(vehicle, step))

            if args.log_every > 0 and (
                    step % args.log_every == 0 or step == args.frames - 1):
                current = profile[-1]
                print(
                    '%s step=%03d loc=(%0.3f,%0.3f,%0.3f) '
                    'speed=%0.4f roll=%0.4f pitch=%0.4f' % (
                        case_name,
                        step,
                        current['location'].x,
                        current['location'].y,
                        current['location'].z,
                        current['speed'],
                        current['roll'],
                        current['pitch']))

        return {
            'name': case_name,
            'transform': used_transform,
            'profile': profile,
            'collisions': collision_events[:],
        }
    finally:
        destroy_actors(world, actors)


def compare_profiles(stock, identity, args):
    if len(stock['profile']) != len(identity['profile']):
        raise RuntimeError('profile length mismatch: %d vs %d' %
                           (len(stock['profile']), len(identity['profile'])))

    max_position_diff = 0.0
    max_speed_diff = 0.0
    max_roll_diff = 0.0
    max_pitch_diff = 0.0
    max_velocity_vector_diff = 0.0
    max_position_step = 0
    max_speed_step = 0
    max_roll_step = 0
    max_pitch_step = 0

    for index, states in enumerate(zip(stock['profile'], identity['profile'])):
        stock_state, identity_state = states
        position_diff = distance(
            stock_state['location'],
            identity_state['location'])
        speed_diff = abs(stock_state['speed'] - identity_state['speed'])
        roll_diff = abs(stock_state['roll'] - identity_state['roll'])
        pitch_diff = abs(stock_state['pitch'] - identity_state['pitch'])
        velocity_vector_diff = distance(
            stock_state['velocity'],
            identity_state['velocity'])

        if position_diff > max_position_diff:
            max_position_diff = position_diff
            max_position_step = index
        if speed_diff > max_speed_diff:
            max_speed_diff = speed_diff
            max_speed_step = index
        if roll_diff > max_roll_diff:
            max_roll_diff = roll_diff
            max_roll_step = index
        if pitch_diff > max_pitch_diff:
            max_pitch_diff = pitch_diff
            max_pitch_step = index
        if velocity_vector_diff > max_velocity_vector_diff:
            max_velocity_vector_diff = velocity_vector_diff

    final_stock = stock['profile'][-1]
    final_identity = identity['profile'][-1]
    final_position_diff = distance(
        final_stock['location'],
        final_identity['location'])
    final_speed_diff = abs(final_stock['speed'] - final_identity['speed'])
    final_roll_diff = abs(final_stock['roll'] - final_identity['roll'])
    final_pitch_diff = abs(final_stock['pitch'] - final_identity['pitch'])

    print('')
    print('Comparison summary:')
    print('  final_position_diff=%0.6f m' % final_position_diff)
    print('  final_speed_diff=%0.6f m/s' % final_speed_diff)
    print('  final_roll_diff=%0.6f deg' % final_roll_diff)
    print('  final_pitch_diff=%0.6f deg' % final_pitch_diff)
    print('  max_position_diff=%0.6f m at step %d' %
          (max_position_diff, max_position_step))
    print('  max_speed_diff=%0.6f m/s at step %d' %
          (max_speed_diff, max_speed_step))
    print('  max_velocity_vector_diff=%0.6f m/s' %
          max_velocity_vector_diff)
    print('  max_roll_diff=%0.6f deg at step %d' %
          (max_roll_diff, max_roll_step))
    print('  max_pitch_diff=%0.6f deg at step %d' %
          (max_pitch_diff, max_pitch_step))
    print('  stock_collisions=%d identity_collisions=%d' %
          (len(stock['collisions']), len(identity['collisions'])))

    failures = []
    if final_position_diff > args.final_position_threshold:
        failures.append(
            'final position diff %0.6f > %0.6f m' %
            (final_position_diff, args.final_position_threshold))
    if max_speed_diff > args.speed_threshold:
        failures.append(
            'max speed diff %0.6f > %0.6f m/s' %
            (max_speed_diff, args.speed_threshold))
    if max_roll_diff > args.angle_threshold:
        failures.append(
            'max roll diff %0.6f > %0.6f deg' %
            (max_roll_diff, args.angle_threshold))
    if max_pitch_diff > args.angle_threshold:
        failures.append(
            'max pitch diff %0.6f > %0.6f deg' %
            (max_pitch_diff, args.angle_threshold))
    if stock['collisions'] or identity['collisions']:
        failures.append(
            'collision detected: stock=%d identity=%d' %
            (len(stock['collisions']), len(identity['collisions'])))

    if failures:
        print('')
        print('FAILED equivalence checks:')
        for failure in failures:
            print('  - %s' % failure)
        raise RuntimeError('stock vs identity equivalence failed')

    print('')
    print('PASS: stock vs per-tick identity suspension behavior is equivalent '
          'within thresholds.')


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

        stock = run_case(
            world,
            blueprint,
            args,
            case_name='S0 stock',
            use_identity=False)
        identity = run_case(
            world,
            blueprint,
            args,
            case_name='S1 identity',
            use_identity=True,
            transform=stock['transform'])

        compare_profiles(stock, identity, args)
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
        help='number of synchronous ticks per run (default: 200)')
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
        '--final-position-threshold',
        default=0.10,
        type=float,
        help='allowed final location difference in meters (default: 0.10)')
    argparser.add_argument(
        '--speed-threshold',
        default=0.05,
        type=float,
        help='allowed max speed profile difference in m/s (default: 0.05)')
    argparser.add_argument(
        '--angle-threshold',
        default=0.05,
        type=float,
        help='allowed max roll/pitch profile difference in degrees '
             '(default: 0.05)')

    try:
        main(argparser.parse_args())
    except KeyboardInterrupt:
        print(' - Exited by user.')

#!/usr/bin/env python

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Run or support TransFuser++ route tests with identity suspension.

This script deliberately does not modify TransFuser++, carla_garage, or
e2e_models. For A1 it runs a CARLA sidecar client that watches for the hero
vehicle, captures native PhysX suspension, and applies that native suspension
every simulator tick. A0 should use the same TransFuser++ route command without
the sidecar.

Examples:

  A0, no suspension API:
    python3 PythonAPI/taeho/transfuser_identity_route_runner.py \\
      --label A0 -- bash -lc '<same debug route command>'

  A1, identity suspension sidecar plus the same route command:
    python3 PythonAPI/taeho/transfuser_identity_route_runner.py \\
      --label A1 --identity -- bash -lc '<same debug route command>'

  A1 sidecar only, with TransFuser++ launched manually in another terminal:
    python3 PythonAPI/taeho/transfuser_identity_route_runner.py \\
      --label A1 --identity --sidecar-only
"""

from __future__ import print_function

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
import threading
import time


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


SUSPENSION_FIELDS = (
    'spring_strength',
    'spring_damper_rate',
    'max_compression',
    'max_droop',
    'sprung_mass',
)


def output_dir_path(path):
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


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


class IdentitySuspensionSidecar(threading.Thread):

    def __init__(self, args, csv_path):
        threading.Thread.__init__(self)
        self.daemon = True
        self.args = args
        self.csv_path = csv_path
        self.stop_event = threading.Event()
        self.native_by_actor_id = {}
        self.apply_count_by_actor_id = {}
        self.error = None
        self.rows_written = 0

    def stop(self):
        self.stop_event.set()

    def log_row(self, writer, event, frame=None, actor=None, message=''):
        actor_id = actor.id if actor is not None else ''
        role_name = ''
        type_id = ''
        if actor is not None:
            type_id = actor.type_id
            role_name = actor.attributes.get('role_name', '')
        row = {
            'wall_time': '%0.6f' % time.time(),
            'label': self.args.label,
            'event': event,
            'frame': '' if frame is None else frame,
            'actor_id': actor_id,
            'type_id': type_id,
            'role_name': role_name,
            'apply_count': self.apply_count_by_actor_id.get(actor_id, ''),
            'message': message,
        }
        writer.writerow(row)
        self.rows_written += 1

    def connect_world(self, writer):
        while not self.stop_event.is_set():
            try:
                client = carla.Client(self.args.host, self.args.port)
                client.set_timeout(self.args.timeout)
                world = client.get_world()
                self.log_row(writer, 'connected', message='connected to CARLA')
                return world
            except RuntimeError as error:
                self.log_row(writer, 'connect_wait', message=str(error))
                time.sleep(self.args.connect_retry_seconds)
        return None

    def target_vehicles(self, world):
        actors = world.get_actors().filter('vehicle.*')
        vehicles = []
        for actor in actors:
            if self.args.actor_id is not None and actor.id != self.args.actor_id:
                continue
            role_name = actor.attributes.get('role_name', '')
            if self.args.actor_id is None and role_name != self.args.role_name:
                continue
            if not hasattr(actor, 'get_suspension_physics_control'):
                continue
            vehicles.append(actor)
        return vehicles

    def remove_missing_actors(self, writer, active_actor_ids, frame):
        for actor_id in list(self.native_by_actor_id.keys()):
            if actor_id not in active_actor_ids:
                del self.native_by_actor_id[actor_id]
                self.apply_count_by_actor_id.pop(actor_id, None)
                self.log_row(
                    writer,
                    'actor_removed',
                    frame=frame,
                    message='actor %s disappeared' % actor_id)

    def capture_native_if_needed(self, writer, actor, frame):
        if actor.id in self.native_by_actor_id:
            return
        native = actor.get_suspension_physics_control()
        validate_suspension(native)
        self.native_by_actor_id[actor.id] = native
        self.apply_count_by_actor_id[actor.id] = 0
        self.log_row(
            writer,
            'native_captured',
            frame=frame,
            actor=actor,
            message='captured native suspension')

    def apply_identity(self, writer, actor, frame):
        native = self.native_by_actor_id[actor.id]
        actor.apply_suspension_physics_control(native)
        self.apply_count_by_actor_id[actor.id] += 1

        count = self.apply_count_by_actor_id[actor.id]
        if count == 1 or count % self.args.log_every == 0:
            self.log_row(
                writer,
                'identity_applied',
                frame=frame,
                actor=actor,
                message='applied native suspension')

        if self.args.verify_every > 0 and count % self.args.verify_every == 0:
            after = actor.get_suspension_physics_control()
            assert_spring_damper_match(
                native,
                after,
                'actor %d verify' % actor.id)
            self.log_row(
                writer,
                'identity_verified',
                frame=frame,
                actor=actor,
                message='readback matched native')

    def run(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        fields = (
            'wall_time',
            'label',
            'event',
            'frame',
            'actor_id',
            'type_id',
            'role_name',
            'apply_count',
            'message',
        )

        try:
            with open(self.csv_path, 'w', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()
                world = self.connect_world(writer)
                if world is None:
                    return

                last_wait_log = 0.0
                while not self.stop_event.is_set():
                    try:
                        snapshot = world.get_snapshot()
                        frame = snapshot.frame
                        vehicles = self.target_vehicles(world)
                        active_ids = set(actor.id for actor in vehicles)
                        self.remove_missing_actors(writer, active_ids, frame)

                        if not vehicles:
                            now = time.time()
                            if now - last_wait_log >= self.args.wait_log_seconds:
                                self.log_row(
                                    writer,
                                    'waiting_for_vehicle',
                                    frame=frame,
                                    message='no matching vehicle found')
                                last_wait_log = now
                        else:
                            for actor in vehicles:
                                self.capture_native_if_needed(writer, actor, frame)
                                self.apply_identity(writer, actor, frame)

                        csv_file.flush()
                        world.wait_for_tick(self.args.tick_wait_timeout)
                    except RuntimeError as error:
                        self.log_row(writer, 'runtime_error', message=str(error))
                        csv_file.flush()
                        time.sleep(self.args.connect_retry_seconds)
                    except Exception as error:
                        self.error = error
                        self.log_row(writer, 'fatal_error', message=str(error))
                        csv_file.flush()
                        return

                self.log_row(writer, 'stopped', message='sidecar stopped')
                csv_file.flush()
        except Exception as error:
            self.error = error


def write_runner_summary(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)
        json_file.write('\n')


def normalized_command(args):
    command = list(args.command)
    if command and command[0] == '--':
        command = command[1:]
    return command


def run_command(args):
    command = normalized_command(args)
    if not command:
        raise RuntimeError('no route command provided; use --sidecar-only or '
                           'append command after --')

    print('Running route command:')
    print('  %s' % ' '.join(command))
    process = subprocess.Popen(command)
    try:
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            return process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            return process.wait()


def main(args):
    if args.sidecar_only and not args.identity:
        raise RuntimeError('--sidecar-only requires --identity')

    output_dir = output_dir_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    sidecar = None
    sidecar_csv = os.path.join(
        output_dir,
        'transfuser_identity_%s_sidecar.csv' % args.label)
    summary_json = os.path.join(
        output_dir,
        'transfuser_identity_%s_runner_summary.json' % args.label)

    start_time = time.time()
    return_code = None
    status = 'ok'

    try:
        if args.identity:
            sidecar = IdentitySuspensionSidecar(args, sidecar_csv)
            sidecar.start()
            print('Started identity suspension sidecar:')
            print('  %s' % sidecar_csv)

        if args.sidecar_only:
            print('Sidecar-only mode. Press Ctrl-C after the route finishes.')
            while True:
                time.sleep(1.0)
                if sidecar is not None and sidecar.error is not None:
                    raise sidecar.error
        else:
            return_code = run_command(args)
            if return_code != 0:
                status = 'route_command_failed'
    except KeyboardInterrupt:
        status = 'interrupted'
    finally:
        if sidecar is not None:
            sidecar.stop()
            sidecar.join(timeout=5.0)
            if sidecar.error is not None:
                status = 'sidecar_failed'

        end_time = time.time()
        summary = {
            'label': args.label,
            'identity': args.identity,
            'sidecar_only': args.sidecar_only,
            'host': args.host,
            'port': args.port,
            'role_name': args.role_name,
            'actor_id': args.actor_id,
            'command': normalized_command(args),
            'return_code': return_code,
            'status': status,
            'start_time': start_time,
            'end_time': end_time,
            'duration_seconds': end_time - start_time,
            'sidecar_csv': sidecar_csv if args.identity else '',
        }
        write_runner_summary(summary_json, summary)
        print('Wrote runner summary:')
        print('  %s' % summary_json)

    if status not in ('ok', 'interrupted'):
        raise RuntimeError('experiment finished with status %s' % status)
    if return_code not in (None, 0):
        raise RuntimeError('route command failed with return code %s' %
                           return_code)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--label',
        default='A1',
        help='experiment label, for example A0 or A1 (default: A1)')
    parser.add_argument(
        '--identity',
        action='store_true',
        help='run identity suspension sidecar')
    parser.add_argument(
        '--sidecar-only',
        action='store_true',
        help='run only the identity sidecar; launch route manually elsewhere')
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='CARLA host (default: 127.0.0.1)')
    parser.add_argument(
        '-p',
        '--port',
        default=2000,
        type=int,
        help='CARLA port (default: 2000)')
    parser.add_argument(
        '--timeout',
        default=30.0,
        type=float,
        help='CARLA client timeout in seconds (default: 30.0)')
    parser.add_argument(
        '--role-name',
        default='hero',
        help='vehicle role_name to control when actor-id is not set '
             '(default: hero)')
    parser.add_argument(
        '--actor-id',
        default=None,
        type=int,
        help='specific vehicle actor id to control instead of role-name')
    parser.add_argument(
        '--verify-every',
        default=50,
        type=int,
        help='read back suspension every N applies; 0 disables (default: 50)')
    parser.add_argument(
        '--log-every',
        default=50,
        type=int,
        help='log identity_applied every N applies (default: 50)')
    parser.add_argument(
        '--tick-wait-timeout',
        default=5.0,
        type=float,
        help='seconds to wait for each world tick (default: 5.0)')
    parser.add_argument(
        '--connect-retry-seconds',
        default=1.0,
        type=float,
        help='seconds between CARLA reconnect attempts (default: 1.0)')
    parser.add_argument(
        '--wait-log-seconds',
        default=5.0,
        type=float,
        help='seconds between no-vehicle wait logs (default: 5.0)')
    parser.add_argument(
        '--output-dir',
        default='results/transfuser_identity',
        help='output directory, relative to this script unless absolute '
             '(default: results/transfuser_identity)')
    parser.add_argument(
        'command',
        nargs=argparse.REMAINDER,
        help='route command after --, for example: -- bash -lc "<command>"')

    main(parser.parse_args())

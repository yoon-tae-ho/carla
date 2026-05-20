#!/usr/bin/env python

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Run a TransFuser++ static suspension scale suite in one command.

The route command is executed once per scenario:

  T0 stock: no suspension API call
  T1 identity: k=1.00, c=1.00
  T2 stiff_sqrt: k=1.05, c=sqrt(1.05)
  T3 soft_sqrt: k=0.95, c=sqrt(0.95)
  T4 stiff_damper: k=1.05, c=1.10

The script does not modify TransFuser++, carla_garage, or e2e_models. For
T1-T4 it runs a sidecar CARLA client that waits for the hero vehicle, captures
native suspension, applies the scenario's static scale, and verifies readback.
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


def scenario_definitions(include_damper_only=False):
    scenarios = [
        {
            'name': 'T0_stock',
            'label': 'T0 stock',
            'spring_scale': None,
            'damper_scale': None,
            'uses_suspension_api': False,
        },
        {
            'name': 'T1_identity',
            'label': 'T1 identity',
            'spring_scale': 1.0,
            'damper_scale': 1.0,
            'uses_suspension_api': True,
        },
        {
            'name': 'T2_k1.05_csqrt',
            'label': 'T2 k=1.05 c=sqrt(1.05)',
            'spring_scale': 1.05,
            'damper_scale': math.sqrt(1.05),
            'uses_suspension_api': True,
        },
        {
            'name': 'T3_k0.95_csqrt',
            'label': 'T3 k=0.95 c=sqrt(0.95)',
            'spring_scale': 0.95,
            'damper_scale': math.sqrt(0.95),
            'uses_suspension_api': True,
        },
        {
            'name': 'T4_k1.05_c1.10',
            'label': 'T4 k=1.05 c=1.10',
            'spring_scale': 1.05,
            'damper_scale': 1.10,
            'uses_suspension_api': True,
        },
    ]

    if include_damper_only:
        scenarios.append({
            'name': 'T5_k1.00_c1.10',
            'label': 'T5 k=1.00 c=1.10',
            'spring_scale': 1.0,
            'damper_scale': 1.10,
            'uses_suspension_api': True,
        })

    return scenarios


def output_dir_path(path):
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def default_route_command(args):
    command = (
        'source /home/yoon-tae-ho/miniconda3/etc/profile.d/conda.sh && '
        'conda activate garage_2 && '
        'source /home/yoon-tae-ho/sim/e2e_models/scripts/env_garage_2.sh && '
        'PORT=%d DEBUG=%d '
        '/home/yoon-tae-ho/sim/e2e_models/scripts/run_tfpp_debug_route.sh'
    ) % (args.port, args.debug)
    return ['bash', '-lc', command]


def normalized_command(args):
    command = list(args.command)
    if command and command[0] == '--':
        command = command[1:]
    if command:
        return command
    return default_route_command(args)


def debug_result_paths(args):
    pattern = os.path.join(args.tfpp_output_root, 'debug_*', 'result.json')
    return set(glob.glob(pattern))


def newest_result_path(args, before_paths, command_output):
    for line in reversed(command_output):
        candidate = line.strip()
        if candidate.startswith('CHECKPOINT='):
            candidate = candidate.split('=', 1)[1].strip()
        if candidate.endswith('result.json') and os.path.isfile(candidate):
            return candidate

    after_paths = debug_result_paths(args)
    created = sorted(after_paths - before_paths, key=os.path.getmtime)
    if created:
        return created[-1]

    return ''


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


class StaticSuspensionSidecar(threading.Thread):

    def __init__(self, args, scenario, csv_path):
        threading.Thread.__init__(self)
        self.daemon = True
        self.args = args
        self.scenario = scenario
        self.csv_path = csv_path
        self.stop_event = threading.Event()
        self.native_by_actor_id = {}
        self.command_by_actor_id = {}
        self.tick_count_by_actor_id = {}
        self.apply_count_by_actor_id = {}
        self.verify_count_by_actor_id = {}
        self.error = None

    def stop(self):
        self.stop_event.set()

    def log_row(self, writer, event, frame=None, actor=None, message=''):
        actor_id = actor.id if actor is not None else ''
        row = {
            'wall_time': '%0.6f' % time.time(),
            'scenario': self.scenario['name'],
            'label': self.scenario['label'],
            'event': event,
            'frame': '' if frame is None else frame,
            'actor_id': actor_id,
            'type_id': actor.type_id if actor is not None else '',
            'role_name': actor.attributes.get('role_name', '') if actor is not None else '',
            'spring_scale': '' if self.scenario['spring_scale'] is None else
            '%0.9g' % self.scenario['spring_scale'],
            'damper_scale': '' if self.scenario['damper_scale'] is None else
            '%0.9g' % self.scenario['damper_scale'],
            'apply_count': self.apply_count_by_actor_id.get(actor_id, ''),
            'verify_count': self.verify_count_by_actor_id.get(actor_id, ''),
            'message': message,
        }
        writer.writerow(row)

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
        vehicles = []
        for actor in world.get_actors().filter('vehicle.*'):
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
                self.native_by_actor_id.pop(actor_id, None)
                self.command_by_actor_id.pop(actor_id, None)
                self.tick_count_by_actor_id.pop(actor_id, None)
                self.apply_count_by_actor_id.pop(actor_id, None)
                self.verify_count_by_actor_id.pop(actor_id, None)
                self.log_row(
                    writer,
                    'actor_removed',
                    frame=frame,
                    message='actor %s disappeared' % actor_id)

    def capture_and_apply_if_needed(self, writer, actor, frame):
        if actor.id in self.native_by_actor_id:
            return

        native = actor.get_suspension_physics_control()
        validate_suspension(native)
        command = scale_suspension(
            native,
            self.scenario['spring_scale'],
            self.scenario['damper_scale'])
        validate_suspension(command)

        self.native_by_actor_id[actor.id] = native
        self.command_by_actor_id[actor.id] = command
        self.tick_count_by_actor_id[actor.id] = 0
        self.apply_count_by_actor_id[actor.id] = 0
        self.verify_count_by_actor_id[actor.id] = 0

        self.log_row(
            writer,
            'native_captured',
            frame=frame,
            actor=actor,
            message='captured native suspension')
        self.apply_static(writer, actor, frame, event='static_applied')
        self.verify_static(writer, actor, frame)

    def apply_static(self, writer, actor, frame, event='static_reapplied'):
        command = self.command_by_actor_id[actor.id]
        actor.apply_suspension_physics_control(command)
        self.apply_count_by_actor_id[actor.id] += 1
        count = self.apply_count_by_actor_id[actor.id]
        if event == 'static_applied' or count % self.args.log_every == 0:
            self.log_row(
                writer,
                event,
                frame=frame,
                actor=actor,
                message='applied static suspension command')

    def verify_static(self, writer, actor, frame):
        command = self.command_by_actor_id[actor.id]
        after = actor.get_suspension_physics_control()
        assert_spring_damper_match(
            command,
            after,
            '%s actor %d verify' % (self.scenario['name'], actor.id))
        self.verify_count_by_actor_id[actor.id] += 1
        self.log_row(
            writer,
            'static_verified',
            frame=frame,
            actor=actor,
            message='readback matched static command')

    def maybe_reapply_and_verify(self, writer, actor, frame):
        self.tick_count_by_actor_id[actor.id] += 1
        tick_count = self.tick_count_by_actor_id[actor.id]

        if self.args.apply_every_tick:
            self.apply_static(writer, actor, frame)
        elif self.args.reapply_every > 0 and tick_count % self.args.reapply_every == 0:
            self.apply_static(writer, actor, frame)

        if self.args.verify_every > 0 and tick_count % self.args.verify_every == 0:
            self.verify_static(writer, actor, frame)

    def run(self):
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        fields = (
            'wall_time',
            'scenario',
            'label',
            'event',
            'frame',
            'actor_id',
            'type_id',
            'role_name',
            'spring_scale',
            'damper_scale',
            'apply_count',
            'verify_count',
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
                                self.capture_and_apply_if_needed(writer, actor, frame)
                                self.maybe_reapply_and_verify(writer, actor, frame)

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


def read_result_json(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path) as json_file:
        return json.load(json_file)


def extract_result_metrics(path):
    data = read_result_json(path)
    if not data:
        return {}

    checkpoint = data.get('_checkpoint', {})
    global_record = checkpoint.get('global_record', {})
    scores = global_record.get('scores_mean', {})
    infractions = global_record.get('infractions', {})
    meta = global_record.get('meta', {})
    records = checkpoint.get('records', [])

    route_statuses = ';'.join(
        '%s:%s' % (record.get('route_id', ''), record.get('status', ''))
        for record in records)
    route_scores = ';'.join(
        '%s:%s' % (
            record.get('route_id', ''),
            record.get('scores', {}).get('score_composed', ''))
        for record in records)

    return {
        'entry_status': data.get('entry_status', ''),
        'eligible': data.get('eligible', ''),
        'progress': '%s/%s' % tuple(checkpoint.get('progress', ['', ''])),
        'global_status': global_record.get('status', ''),
        'score_composed': scores.get('score_composed', ''),
        'score_route': scores.get('score_route', ''),
        'score_penalty': scores.get('score_penalty', ''),
        'collisions_layout': infractions.get('collisions_layout', ''),
        'collisions_pedestrian': infractions.get('collisions_pedestrian', ''),
        'collisions_vehicle': infractions.get('collisions_vehicle', ''),
        'red_light': infractions.get('red_light', ''),
        'stop_infraction': infractions.get('stop_infraction', ''),
        'outside_route_lanes': infractions.get('outside_route_lanes', ''),
        'route_dev': infractions.get('route_dev', ''),
        'vehicle_blocked': infractions.get('vehicle_blocked', ''),
        'route_timeout': infractions.get('route_timeout', ''),
        'scenario_timeouts': infractions.get('scenario_timeouts', ''),
        'min_speed_infractions': infractions.get('min_speed_infractions', ''),
        'duration_game': meta.get('duration_game', ''),
        'duration_system': meta.get('duration_system', ''),
        'route_statuses': route_statuses,
        'route_scores': route_scores,
    }


def result_finished(path):
    metrics = extract_result_metrics(path)
    return metrics.get('entry_status') == 'Finished'


def run_route_command(command, log_path):
    print('Running route command:')
    print('  %s' % ' '.join(command))
    output_lines = []

    with open(log_path, 'w') as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
                output_lines.append(line.rstrip('\n'))
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                return_code = process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                process.kill()
                return_code = process.wait()

    return return_code, output_lines


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=2, sort_keys=True)
        json_file.write('\n')


def summarize_sidecar_csv(path):
    if not path or not os.path.isfile(path):
        return {
            'sidecar_native_captures': 0,
            'sidecar_static_applies': 0,
            'sidecar_static_verifies': 0,
            'sidecar_actor_ids': '',
            'sidecar_fatal_errors': 0,
            'sidecar_runtime_errors': 0,
        }

    native_captures = 0
    static_applies = 0
    static_verifies = 0
    fatal_errors = 0
    runtime_errors = 0
    actor_ids = set()

    with open(path) as csv_file:
        for row in csv.DictReader(csv_file):
            event = row.get('event', '')
            actor_id = row.get('actor_id', '')
            if actor_id:
                actor_ids.add(actor_id)
            if event == 'native_captured':
                native_captures += 1
            elif event in ('static_applied', 'static_reapplied'):
                static_applies += 1
            elif event == 'static_verified':
                static_verifies += 1
            elif event == 'fatal_error':
                fatal_errors += 1
            elif event == 'runtime_error':
                runtime_errors += 1

    return {
        'sidecar_native_captures': native_captures,
        'sidecar_static_applies': static_applies,
        'sidecar_static_verifies': static_verifies,
        'sidecar_actor_ids': ';'.join(sorted(actor_ids)),
        'sidecar_fatal_errors': fatal_errors,
        'sidecar_runtime_errors': runtime_errors,
    }


def run_scenario(args, scenario, command, output_dir):
    print('')
    print('=== %s ===' % scenario['label'])
    start_time = time.time()
    before_paths = debug_result_paths(args)
    sidecar = None
    sidecar_csv = ''
    log_path = os.path.join(output_dir, '%s_route_stdout.log' % scenario['name'])

    if scenario['uses_suspension_api']:
        sidecar_csv = os.path.join(output_dir, '%s_sidecar.csv' % scenario['name'])
        sidecar = StaticSuspensionSidecar(args, scenario, sidecar_csv)
        sidecar.start()
        print('Started static suspension sidecar:')
        print('  %s' % sidecar_csv)

    return_code = None
    result_path = ''
    status = 'ok'
    output_lines = []

    try:
        return_code, output_lines = run_route_command(command, log_path)
        result_path = newest_result_path(args, before_paths, output_lines)
        if not result_path:
            status = 'missing_result_json'
        elif return_code != 0:
            if result_finished(result_path):
                status = 'finished_return_nonzero'
            else:
                status = 'route_command_failed'
    finally:
        if sidecar is not None:
            sidecar.stop()
            sidecar.join(timeout=5.0)
            if sidecar.error is not None:
                status = 'sidecar_failed'

    end_time = time.time()
    metrics = extract_result_metrics(result_path)
    sidecar_metrics = summarize_sidecar_csv(sidecar_csv)
    if scenario['uses_suspension_api']:
        if sidecar_metrics['sidecar_native_captures'] == 0:
            status = 'sidecar_no_target_vehicle'
        elif sidecar_metrics['sidecar_static_verifies'] == 0:
            status = 'sidecar_not_verified'

    row = {
        'scenario': scenario['name'],
        'label': scenario['label'],
        'spring_scale': scenario['spring_scale'],
        'damper_scale': scenario['damper_scale'],
        'uses_suspension_api': scenario['uses_suspension_api'],
        'status': status,
        'return_code': return_code,
        'result_json': result_path,
        'route_stdout_log': log_path,
        'sidecar_csv': sidecar_csv,
        'start_time': start_time,
        'end_time': end_time,
        'duration_seconds': end_time - start_time,
    }
    row.update(metrics)
    row.update(sidecar_metrics)

    summary_json = os.path.join(output_dir, '%s_summary.json' % scenario['name'])
    write_json(summary_json, row)

    print('Scenario result:')
    print('  status=%s return_code=%s' % (status, return_code))
    print('  result_json=%s' % result_path)
    if sidecar_csv:
        print('  sidecar_csv=%s' % sidecar_csv)

    if status in ('route_command_failed', 'sidecar_failed') and args.stop_on_failure:
        raise RuntimeError('%s failed with status %s' %
                           (scenario['name'], status))

    return row


def write_suite_summary(output_dir, rows):
    fields = (
        'scenario',
        'label',
        'spring_scale',
        'damper_scale',
        'uses_suspension_api',
        'status',
        'return_code',
        'entry_status',
        'global_status',
        'progress',
        'score_composed',
        'score_route',
        'score_penalty',
        'collisions_layout',
        'collisions_pedestrian',
        'collisions_vehicle',
        'red_light',
        'stop_infraction',
        'outside_route_lanes',
        'route_dev',
        'vehicle_blocked',
        'route_timeout',
        'scenario_timeouts',
        'min_speed_infractions',
        'duration_game',
        'duration_system',
        'route_statuses',
        'route_scores',
        'sidecar_native_captures',
        'sidecar_static_applies',
        'sidecar_static_verifies',
        'sidecar_actor_ids',
        'sidecar_fatal_errors',
        'sidecar_runtime_errors',
        'duration_seconds',
        'result_json',
        'route_stdout_log',
        'sidecar_csv',
    )
    path = os.path.join(output_dir, 'transfuser_static_scale_suite_summary.csv')
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})

    json_path = os.path.join(output_dir, 'transfuser_static_scale_suite_summary.json')
    write_json(json_path, rows)
    return path, json_path


def print_suite_table(rows):
    print('')
    print('TransFuser++ static suspension suite summary:')
    print('%-20s %-24s %9s %9s %9s %9s %8s %8s %5s' % (
        'scenario',
        'status',
        'score',
        'route',
        'penalty',
        'minspeed',
        'captures',
        'verifies',
        'rc'))
    for row in rows:
        print('%-20s %-24s %9s %9s %9s %9s %8s %8s %5s' % (
            row.get('scenario', ''),
            row.get('status', ''),
            row.get('score_composed', ''),
            row.get('score_route', ''),
            row.get('score_penalty', ''),
            row.get('min_speed_infractions', ''),
            row.get('sidecar_native_captures', ''),
            row.get('sidecar_static_verifies', ''),
            row.get('return_code', '')))


def main(args):
    command = normalized_command(args)
    output_dir = output_dir_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    scenarios = scenario_definitions(include_damper_only=args.include_damper_only)
    rows = []
    for index, scenario in enumerate(scenarios):
        if index > 0 and args.pause_between_runs > 0.0:
            time.sleep(args.pause_between_runs)
        rows.append(run_scenario(args, scenario, command, output_dir))

    print_suite_table(rows)
    csv_path, json_path = write_suite_summary(output_dir, rows)
    print('')
    print('Wrote suite outputs:')
    print('  %s' % csv_path)
    print('  %s' % json_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='CARLA host (default: 127.0.0.1)')
    parser.add_argument(
        '-p',
        '--port',
        default=2000,
        type=int,
        help='CARLA port and route PORT env for default command (default: 2000)')
    parser.add_argument(
        '--timeout',
        default=30.0,
        type=float,
        help='CARLA client timeout in seconds (default: 30.0)')
    parser.add_argument(
        '--debug',
        default=1,
        type=int,
        help='DEBUG env for default run_tfpp_debug_route.sh command (default: 1)')
    parser.add_argument(
        '--tfpp-output-root',
        default='/home/yoon-tae-ho/sim/e2e_models/outputs/transfuserpp',
        help='TransFuser++ output root used to locate debug result.json files')
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
        help='read back suspension every N sidecar ticks; 0 disables '
             '(default: 50)')
    parser.add_argument(
        '--reapply-every',
        default=0,
        type=int,
        help='reapply static suspension every N sidecar ticks; 0 means once per '
             'actor unless --apply-every-tick is set (default: 0)')
    parser.add_argument(
        '--apply-every-tick',
        action='store_true',
        help='apply the static command every sidecar tick instead of once')
    parser.add_argument(
        '--log-every',
        default=50,
        type=int,
        help='log static_reapplied every N applies (default: 50)')
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
        '--pause-between-runs',
        default=5.0,
        type=float,
        help='seconds to pause between route command runs (default: 5.0)')
    parser.add_argument(
        '--include-damper-only',
        action='store_true',
        help='also run the cautious k=1.00, c=1.10 scenario')
    parser.add_argument(
        '--stop-on-failure',
        action='store_true',
        help='stop the suite on command/sidecar failure')
    parser.add_argument(
        '--output-dir',
        default='results/transfuser_static_scale',
        help='output directory, relative to this script unless absolute '
             '(default: results/transfuser_static_scale)')
    parser.add_argument(
        'command',
        nargs=argparse.REMAINDER,
        help='optional route command after --. If omitted, uses '
             'run_tfpp_debug_route.sh with conda garage_2')

    main(parser.parse_args())

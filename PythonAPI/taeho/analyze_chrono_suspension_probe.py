#!/usr/bin/env python

"""Summarize a chrono_suspension_probe.py JSONL log."""

from __future__ import print_function

import argparse
import glob
import json
import math
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_DIR = os.path.join(SCRIPT_DIR, 'chrono_suspension_probe_logs')


def mean(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def vector_norm(raw):
    if not raw:
        return 0.0
    return math.sqrt(
        float(raw.get('x', 0.0)) * float(raw.get('x', 0.0)) +
        float(raw.get('y', 0.0)) * float(raw.get('y', 0.0)) +
        float(raw.get('z', 0.0)) * float(raw.get('z', 0.0)))


def load_jsonl(path):
    records = []
    with open(path) as infile:
        for line_number, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError as error:
                raise RuntimeError('%s:%d is not valid JSON: %s' % (path, line_number, error))
    return records


def latest_log_path(log_dir):
    candidates = glob.glob(os.path.join(log_dir, 'chrono_suspension_probe_*.jsonl'))
    if not candidates:
        raise RuntimeError('no probe logs found in %s' % log_dir)
    return max(candidates, key=os.path.getmtime)


def command_signature(record):
    command = record.get('command') or {}
    damping = command.get('damping') or []
    stiffness = command.get('stiffness') or []
    return tuple(round(float(value), 4) for value in list(damping) + list(stiffness))


def tail_by_phase_trial(records, tail_ticks):
    if tail_ticks is None or tail_ticks <= 0:
        return records

    groups = []
    keys = []
    for record in records:
        key = (record.get('phase', 'unknown'), record.get('trial_id'))
        if key not in keys:
            keys.append(key)
            groups.append((key, []))
        groups[keys.index(key)][1].append(record)

    selected = []
    for key, group in groups:
        del key
        selected.extend(group[-tail_ticks:])
    return selected


def summarize_phase(records):
    speeds = [float(record['state'].get('speed_mps', 0.0)) for record in records]
    rolls = [float(record['state'].get('abs_roll_deg', 0.0)) for record in records]
    pitches = [float(record['state'].get('abs_pitch_deg', 0.0)) for record in records]
    vertical_accels = [
        float(record['state'].get('acceleration', {}).get('z', 0.0))
        for record in records
    ]
    angular_velocity_norms = [
        vector_norm(record['state'].get('angular_velocity'))
        for record in records
    ]
    damping_fl = [
        float(record.get('command', {}).get('damping', [0.0])[0])
        for record in records
        if record.get('command', {}).get('damping')
    ]
    stiffness_fl = [
        float(record.get('command', {}).get('stiffness', [0.0])[0])
        for record in records
        if record.get('command', {}).get('stiffness')
    ]
    trial_ids = set(
        record.get('trial_id')
        for record in records
        if record.get('trial_id') is not None)

    return {
        'count': len(records),
        'trial_count': len(trial_ids) if trial_ids else (1 if records else 0),
        'applied_count': sum(1 for record in records if record.get('command_applied')),
        'failure_count': sum(1 for record in records if not record.get('command_applied')),
        'avg_speed_mps': mean(speeds),
        'max_speed_mps': max(speeds) if speeds else 0.0,
        'avg_abs_roll_deg': mean(rolls),
        'max_abs_roll_deg': max(rolls) if rolls else 0.0,
        'avg_abs_pitch_deg': mean(pitches),
        'max_abs_pitch_deg': max(pitches) if pitches else 0.0,
        'avg_vertical_accel': mean(vertical_accels),
        'avg_angular_velocity_norm': mean(angular_velocity_norms),
        'avg_damping_fl': mean(damping_fl),
        'avg_stiffness_fl': mean(stiffness_fl),
    }


def summarize(records, tail_ticks=None):
    metadata = None
    prechecks = []
    ticks = []
    for record in records:
        record_type = record.get('type')
        if record_type == 'metadata':
            metadata = record
        elif record_type == 'precheck':
            prechecks.append(record)
        elif record_type == 'tick':
            ticks.append(record)

    ticks = [
        record for record in ticks
        if record.get('sample_role', 'measurement') == 'measurement'
    ]
    ticks = tail_by_phase_trial(ticks, tail_ticks)

    phases = []
    phase_names = []
    for record in ticks:
        phase = record.get('phase', 'unknown')
        if phase not in phase_names:
            phase_names.append(phase)

    for phase in phase_names:
        phase_records = [record for record in ticks if record.get('phase') == phase]
        phases.append((phase, summarize_phase(phase_records)))

    failures = [record for record in ticks if not record.get('command_applied')]
    distinct_commands = len(set(command_signature(record) for record in ticks))
    unsafe_count = sum(1 for record in ticks if record.get('unsafe_pose'))
    precheck_ok = bool(prechecks) and all(record.get('command_applied') for record in prechecks)
    all_ticks_applied = bool(ticks) and not failures
    commands_changed = distinct_commands >= 3

    return {
        'metadata': metadata,
        'tail_ticks': tail_ticks,
        'precheck_ok': precheck_ok,
        'tick_count': len(ticks),
        'applied_tick_count': sum(1 for record in ticks if record.get('command_applied')),
        'failure_count': len(failures),
        'first_failure': failures[0] if failures else None,
        'distinct_command_count': distinct_commands,
        'commands_changed': commands_changed,
        'unsafe_count': unsafe_count,
        'phases': phases,
        'pass': precheck_ok and all_ticks_applied and commands_changed and unsafe_count == 0,
    }


def print_summary(path, summary):
    metadata = summary.get('metadata') or {}
    print('log: %s' % path)
    if metadata.get('test_design'):
        print('test_design: %s' % metadata.get('test_design'))
    if summary.get('tail_ticks'):
        print('tick_filter: last %d measurement ticks per phase/trial' % summary['tail_ticks'])
    print('result: %s' % ('PASS' if summary['pass'] else 'CHECK'))
    print('precheck_ok: %s' % summary['precheck_ok'])
    print('measurement_ticks: %d applied: %d failures: %d' % (
        summary['tick_count'],
        summary['applied_tick_count'],
        summary['failure_count']))
    print('distinct_command_vectors: %d' % summary['distinct_command_count'])
    print('unsafe_pose_count: %d' % summary['unsafe_count'])
    print('')
    print('phase summary:')
    for phase, data in summary['phases']:
        print(
            '- %s: trials=%d ticks=%d applied=%d avg_speed=%.3f max_speed=%.3f '
            'avg_roll=%.3f max_roll=%.3f avg_pitch=%.3f max_pitch=%.3f '
            'avg_z_accel=%.3f avg_ang_vel_norm=%.3f avg_FL_damping=%.1f avg_FL_stiffness=%.1f' % (
                phase,
                data['trial_count'],
                data['count'],
                data['applied_count'],
                data['avg_speed_mps'],
                data['max_speed_mps'],
                data['avg_abs_roll_deg'],
                data['max_abs_roll_deg'],
                data['avg_abs_pitch_deg'],
                data['max_abs_pitch_deg'],
                data['avg_vertical_accel'],
                data['avg_angular_velocity_norm'],
                data['avg_damping_fl'],
                data['avg_stiffness_fl']))

    if summary['first_failure']:
        failure = summary['first_failure']
        print('')
        print('first failure:')
        print('- step: %s phase: %s error: %s' % (
            failure.get('step'),
            failure.get('phase'),
            failure.get('apply_error')))

    fairness_note = metadata.get('fairness_note')
    if fairness_note:
        print('')
        print('fairness: %s' % fairness_note)

    readback_note = metadata.get('readback_note')
    if readback_note:
        print('')
        print('note: %s' % readback_note)


def parse_arguments():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('log_path', nargs='?', help='probe JSONL log path; defaults to latest probe log')
    argparser.add_argument('--log-dir', default=DEFAULT_LOG_DIR, help='directory used when log_path is omitted')
    argparser.add_argument(
        '--tail-ticks',
        type=int,
        help='summarize only the last N measurement ticks of each phase/trial')
    return argparser.parse_args()


def main():
    args = parse_arguments()
    path = args.log_path or latest_log_path(args.log_dir)
    if not os.path.exists(path):
        raise RuntimeError('log path does not exist: %s' % path)
    summary = summarize(load_jsonl(path), tail_ticks=args.tail_ticks)
    print_summary(path, summary)
    return 0 if summary['pass'] else 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

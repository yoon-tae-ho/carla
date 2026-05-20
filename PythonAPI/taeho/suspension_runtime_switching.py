#!/usr/bin/env python

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Run open-loop maneuvers with runtime suspension damping commands.

This is the first runtime-control experiment after the static scale sweeps.
It separates three questions:

  * Can suspension be written while the vehicle is already running?
  * Does get_suspension_physics_control() reflect the command immediately or
    within the next tick?
  * Does the post-switch vehicle response diverge from a per-tick identity
    suspension API baseline?

The experiment keeps spring_scale fixed at 1.00 and changes only damping.

Default scenarios:

  R0_stock:
    No suspension API calls.

  R1_identity_every_tick:
    k=1.00, c=1.00 applied every tick.

  R2_step_safe:
    c=1.00 -> 1.10 -> 0.95 -> 1.00 at steps 100, 180, 260.

  R3_step_diagnostic:
    c=1.00 -> 1.20 -> 0.80 -> 1.00 at steps 100, 180, 260.

  R4_ramp_safe:
    c=1.00, then linearly ramps 1.00 -> 1.10 -> 0.95 -> 1.00.

Profiles, command/readback logs, summaries, and collision logs are written
under PythonAPI/taeho/results by default.
"""

from __future__ import print_function

import argparse
import csv
import math
import os

import suspension_lateral_maneuvers as lateral


carla = lateral.carla


def scenario_definitions(include_diagnostic=True):
    scenarios = [
        {
            'name': 'R0_stock',
            'label': 'R0 stock',
            'uses_suspension_api': False,
            'apply_policy': 'never',
            'schedule_type': 'stock',
        },
        {
            'name': 'R1_identity_every_tick',
            'label': 'R1 identity every tick',
            'uses_suspension_api': True,
            'apply_policy': 'every_tick',
            'schedule_type': 'constant',
            'spring_scale': 1.0,
            'damper_scale': 1.0,
        },
        {
            'name': 'R2_step_safe',
            'label': 'R2 step safe c=1.00/1.10/0.95/1.00',
            'uses_suspension_api': True,
            'apply_policy': 'on_change',
            'schedule_type': 'step',
            'schedule': (
                (0, 1.0, 1.00),
                (100, 1.0, 1.10),
                (180, 1.0, 0.95),
                (260, 1.0, 1.00),
            ),
        },
    ]

    if include_diagnostic:
        scenarios.append({
            'name': 'R3_step_diagnostic',
            'label': 'R3 step diagnostic c=1.00/1.20/0.80/1.00',
            'uses_suspension_api': True,
            'apply_policy': 'on_change',
            'schedule_type': 'step',
            'schedule': (
                (0, 1.0, 1.00),
                (100, 1.0, 1.20),
                (180, 1.0, 0.80),
                (260, 1.0, 1.00),
            ),
        })

    scenarios.append({
        'name': 'R4_ramp_safe',
        'label': 'R4 ramp safe c=1.00->1.10->0.95->1.00',
        'uses_suspension_api': True,
        'apply_policy': 'every_tick',
        'schedule_type': 'ramp_safe',
    })

    return tuple(scenarios)


def maneuver_definitions(args):
    return (
        {
            'name': 'straight_brake',
            'type': 'straight_brake',
            'frames': args.frames,
            'throttle': args.throttle,
            'steer': 0.0,
            'brake': args.brake,
            'brake_start': args.brake_start,
        },
        {
            'name': 'constant_steer_%0.2f' % args.constant_steer,
            'type': 'constant_steer',
            'frames': args.frames,
            'throttle': args.steer_throttle,
            'steer': args.constant_steer,
            'brake': 0.0,
            'brake_start': -1,
        },
        {
            'name': 'brake_in_turn_%0.2f' % args.brake_turn_steer,
            'type': 'brake_in_turn',
            'frames': args.frames,
            'throttle': args.throttle,
            'steer': args.brake_turn_steer,
            'brake': args.brake,
            'brake_start': args.brake_start,
        },
    )


def make_control(step, maneuver):
    if maneuver['type'] == 'constant_steer':
        return carla.VehicleControl(
            throttle=maneuver['throttle'],
            steer=maneuver['steer'])

    if step < maneuver['brake_start']:
        return carla.VehicleControl(
            throttle=maneuver['throttle'],
            steer=maneuver['steer'])

    return carla.VehicleControl(
        brake=maneuver['brake'],
        steer=maneuver['steer'])


def pair_close(a_pair, b_pair, tolerance=1.0e-9):
    if a_pair is None or b_pair is None:
        return False
    if a_pair[0] is None or a_pair[1] is None or b_pair[0] is None or b_pair[1] is None:
        return a_pair == b_pair
    return (
        math.isclose(a_pair[0], b_pair[0], rel_tol=tolerance, abs_tol=tolerance) and
        math.isclose(a_pair[1], b_pair[1], rel_tol=tolerance, abs_tol=tolerance))


def step_schedule(step, transitions):
    spring_scale = transitions[0][1]
    damper_scale = transitions[0][2]
    for start_step, candidate_spring, candidate_damper in transitions:
        if step >= start_step:
            spring_scale = candidate_spring
            damper_scale = candidate_damper
        else:
            break
    return spring_scale, damper_scale


def interpolate(step, start_step, end_step, start_value, end_value):
    if end_step <= start_step:
        return end_value
    alpha = float(step - start_step) / float(end_step - start_step)
    alpha = max(0.0, min(1.0, alpha))
    return start_value + alpha * (end_value - start_value)


def ramp_safe_schedule(step):
    if step < 100:
        return 1.0, 1.00
    if step < 180:
        return 1.0, interpolate(step, 100, 180, 1.00, 1.10)
    if step < 260:
        return 1.0, interpolate(step, 180, 260, 1.10, 0.95)
    if step < 339:
        return 1.0, interpolate(step, 260, 339, 0.95, 1.00)
    return 1.0, 1.00


def target_scales_for_step(scenario, step):
    schedule_type = scenario['schedule_type']
    if schedule_type == 'stock':
        return None, None
    if schedule_type == 'constant':
        return scenario['spring_scale'], scenario['damper_scale']
    if schedule_type == 'step':
        return step_schedule(step, scenario['schedule'])
    if schedule_type == 'ramp_safe':
        return ramp_safe_schedule(step)
    raise RuntimeError('unknown schedule_type %s' % schedule_type)


def scale_mode_name(spring_scale, damper_scale):
    if spring_scale is None or damper_scale is None:
        return 'stock'
    return 'k%s_c%s' % (
        lateral.scale_token(spring_scale),
        lateral.scale_token(damper_scale))


def should_apply(scenario, target_pair, last_applied_pair):
    policy = scenario['apply_policy']
    if policy == 'never':
        return False
    if policy == 'every_tick':
        return True
    if policy == 'on_change':
        return not pair_close(target_pair, last_applied_pair)
    raise RuntimeError('unknown apply_policy %s' % policy)


def readback_summary(native, actual, expected_spring_scale,
                     expected_damper_scale, scale_tolerance):
    lateral.validate_suspension(actual)
    spring_scales = []
    damper_scales = []
    spring_errors = []
    damper_errors = []

    for native_wheel, actual_wheel in zip(native.wheels, actual.wheels):
        spring_scale = actual_wheel.spring_strength / native_wheel.spring_strength
        damper_scale = actual_wheel.spring_damper_rate / native_wheel.spring_damper_rate
        spring_scales.append(spring_scale)
        damper_scales.append(damper_scale)
        spring_errors.append(abs(spring_scale - expected_spring_scale))
        damper_errors.append(abs(damper_scale - expected_damper_scale))

    spring_error = max(spring_errors)
    damper_error = max(damper_errors)
    return {
        'readback_spring_scale': lateral.mean(spring_scales),
        'readback_damper_scale': lateral.mean(damper_scales),
        'readback_spring_scale_min': min(spring_scales),
        'readback_spring_scale_max': max(spring_scales),
        'readback_damper_scale_min': min(damper_scales),
        'readback_damper_scale_max': max(damper_scales),
        'readback_spring_scale_error': spring_error,
        'readback_damper_scale_error': damper_error,
        'readback_match': (
            spring_error <= scale_tolerance and
            damper_error <= scale_tolerance),
    }


def blank_readback_summary():
    return {
        'readback_spring_scale': '',
        'readback_damper_scale': '',
        'readback_spring_scale_min': '',
        'readback_spring_scale_max': '',
        'readback_damper_scale_min': '',
        'readback_damper_scale_max': '',
        'readback_spring_scale_error': '',
        'readback_damper_scale_error': '',
        'readback_match': '',
    }


def apply_event_name(scenario, apply_needed, target_changed, last_target_pair):
    if not scenario['uses_suspension_api']:
        return 'stock'
    if not apply_needed:
        return 'verify_only'
    if last_target_pair is None:
        return 'initial_apply'
    if target_changed:
        return 'scheduled_switch'
    if scenario['schedule_type'] == 'ramp_safe':
        return 'ramp_apply'
    return 'periodic_apply'


def record_command_row(scenario, maneuver, vehicle, control, step,
                       frame_before_apply, frame_after_tick, event,
                       target_pair, immediate, post_tick, latency_ticks):
    commanded_spring_scale = ''
    commanded_damper_scale = ''
    mode = 'stock'
    if target_pair[0] is not None and target_pair[1] is not None:
        commanded_spring_scale = target_pair[0]
        commanded_damper_scale = target_pair[1]
        mode = scale_mode_name(target_pair[0], target_pair[1])

    row = {
        'scenario': scenario['name'],
        'maneuver': maneuver['name'],
        'maneuver_type': maneuver['type'],
        'step': step,
        'frame_before_apply': frame_before_apply,
        'frame_after_tick': frame_after_tick,
        'actor_id': vehicle.id,
        'event': event,
        'mode': mode,
        'apply_policy': scenario['apply_policy'],
        'commanded_spring_scale': commanded_spring_scale,
        'commanded_damper_scale': commanded_damper_scale,
        'immediate_readback_spring_scale': immediate['readback_spring_scale'],
        'immediate_readback_damper_scale': immediate['readback_damper_scale'],
        'immediate_readback_spring_scale_error':
            immediate['readback_spring_scale_error'],
        'immediate_readback_damper_scale_error':
            immediate['readback_damper_scale_error'],
        'immediate_readback_match': immediate['readback_match'],
        'post_tick_readback_spring_scale': post_tick['readback_spring_scale'],
        'post_tick_readback_damper_scale': post_tick['readback_damper_scale'],
        'post_tick_readback_spring_scale_error':
            post_tick['readback_spring_scale_error'],
        'post_tick_readback_damper_scale_error':
            post_tick['readback_damper_scale_error'],
        'post_tick_readback_match': post_tick['readback_match'],
        'apply_latency_ticks': latency_ticks,
        'throttle': control.throttle,
        'brake': control.brake,
        'steer': control.steer,
    }
    return row


def run_trial(world, blueprint, scenario, maneuver, args, transform=None):
    vehicle = None
    actors = []

    try:
        vehicle, used_transform = lateral.spawn_vehicle(
            world,
            blueprint,
            transform=transform,
            preferred_index=args.spawn_index)
        actors.append(vehicle)

        collision_sensor, collision_events = lateral.attach_collision_sensor(
            world,
            vehicle,
            scenario['name'],
            maneuver['name'])
        actors.append(collision_sensor)

        world.tick()

        native = None
        if scenario['uses_suspension_api']:
            native = vehicle.get_suspension_physics_control()
            lateral.validate_suspension(native)

        print('%s %s: vehicle=%s id=%d policy=%s schedule=%s' % (
            scenario['label'],
            maneuver['name'],
            vehicle.type_id,
            vehicle.id,
            scenario['apply_policy'],
            scenario['schedule_type']))

        profile = []
        commands = []
        last_target_pair = None
        last_applied_pair = None
        last_post_readback_pair = None

        apply_count = 0
        verify_count = 0
        command_change_count = 0
        readback_change_count = 0
        readback_mismatch_count = 0
        immediate_mismatch_count = 0
        apply_latency_failure_count = 0
        latencies = []
        max_spring_error = 0.0
        max_damper_error = 0.0

        for step in range(maneuver['frames']):
            control = make_control(step, maneuver)
            target_pair = target_scales_for_step(scenario, step)
            target_changed = not pair_close(target_pair, last_target_pair)
            if last_target_pair is not None and target_changed:
                command_change_count += 1

            frame_before_apply = world.get_snapshot().frame
            apply_needed = should_apply(scenario, target_pair, last_applied_pair)
            event = apply_event_name(
                scenario,
                apply_needed,
                target_changed,
                last_target_pair)

            immediate = blank_readback_summary()
            post_tick = blank_readback_summary()
            latency_ticks = ''

            if scenario['uses_suspension_api']:
                spring_scale, damper_scale = target_pair

                if apply_needed:
                    command = lateral.scale_suspension(
                        native,
                        spring_scale,
                        damper_scale)
                    lateral.validate_suspension(command)
                    vehicle.apply_suspension_physics_control(command)
                    apply_count += 1
                    last_applied_pair = target_pair

                immediate = readback_summary(
                    native,
                    vehicle.get_suspension_physics_control(),
                    spring_scale,
                    damper_scale,
                    args.readback_scale_tolerance)

                if apply_needed and not immediate['readback_match']:
                    immediate_mismatch_count += 1

            vehicle.apply_control(control)
            world.tick()
            frame_after_tick = world.get_snapshot().frame

            if scenario['uses_suspension_api']:
                spring_scale, damper_scale = target_pair
                post_tick = readback_summary(
                    native,
                    vehicle.get_suspension_physics_control(),
                    spring_scale,
                    damper_scale,
                    args.readback_scale_tolerance)
                verify_count += 1

                max_spring_error = max(
                    max_spring_error,
                    post_tick['readback_spring_scale_error'])
                max_damper_error = max(
                    max_damper_error,
                    post_tick['readback_damper_scale_error'])

                if not post_tick['readback_match']:
                    readback_mismatch_count += 1

                post_pair = (
                    post_tick['readback_spring_scale'],
                    post_tick['readback_damper_scale'])
                if (last_post_readback_pair is not None and
                        not pair_close(
                            post_pair,
                            last_post_readback_pair,
                            args.readback_change_tolerance)):
                    readback_change_count += 1
                last_post_readback_pair = post_pair

                if apply_needed:
                    if immediate['readback_match']:
                        latency_ticks = 0
                        latencies.append(0)
                    elif post_tick['readback_match']:
                        latency_ticks = 1
                        latencies.append(1)
                    else:
                        latency_ticks = -1
                        apply_latency_failure_count += 1

            state = lateral.record_state(
                world,
                vehicle,
                scenario['name'],
                maneuver,
                control,
                step)
            state.update({
                'actor_id': vehicle.id,
                'event': event,
                'mode': scale_mode_name(target_pair[0], target_pair[1]),
                'apply_policy': scenario['apply_policy'],
                'commanded_spring_scale': (
                    '' if target_pair[0] is None else target_pair[0]),
                'commanded_damper_scale': (
                    '' if target_pair[1] is None else target_pair[1]),
                'readback_spring_scale': post_tick['readback_spring_scale'],
                'readback_damper_scale': post_tick['readback_damper_scale'],
                'readback_spring_scale_error':
                    post_tick['readback_spring_scale_error'],
                'readback_damper_scale_error':
                    post_tick['readback_damper_scale_error'],
                'readback_match': post_tick['readback_match'],
                'apply_latency_ticks': latency_ticks,
            })
            lateral.finite_or_raise(state.items(), '%s %s step %d' % (
                scenario['name'],
                maneuver['name'],
                step))
            profile.append(state)

            commands.append(record_command_row(
                scenario,
                maneuver,
                vehicle,
                control,
                step,
                frame_before_apply,
                frame_after_tick,
                event,
                target_pair,
                immediate,
                post_tick,
                latency_ticks))

            if args.log_every > 0 and (
                    step % args.log_every == 0 or
                    step == maneuver['frames'] - 1):
                print(
                    '%s %s step=%03d event=%s c=%s rb=%s speed=%0.4f '
                    'roll=%0.4f pitch=%0.4f yaw_rate=%0.4f lat_acc=%0.4f' % (
                        scenario['label'],
                        maneuver['name'],
                        step,
                        event,
                        'stock' if target_pair[1] is None else
                        '%0.4f' % target_pair[1],
                        'stock' if post_tick['readback_damper_scale'] == '' else
                        '%0.4f' % post_tick['readback_damper_scale'],
                        state['speed'],
                        state['roll'],
                        state['pitch'],
                        state['yaw_rate'],
                        state['local_ay']))

            last_target_pair = target_pair

        if native is not None:
            vehicle.apply_suspension_physics_control(native)
            world.tick()

        if latencies:
            mean_latency = lateral.mean(latencies)
            max_latency = max(latencies)
        else:
            mean_latency = ''
            max_latency = ''

        api_metrics = {
            'apply_count': apply_count,
            'verify_count': verify_count,
            'command_change_count': command_change_count,
            'readback_change_count': readback_change_count,
            'readback_mismatch_count': readback_mismatch_count,
            'immediate_mismatch_count': immediate_mismatch_count,
            'apply_latency_failure_count': apply_latency_failure_count,
            'mean_apply_latency_ticks': mean_latency,
            'max_apply_latency_ticks': max_latency,
            'max_readback_spring_scale_error': max_spring_error,
            'max_readback_damper_scale_error': max_damper_error,
        }

        return {
            'scenario': scenario,
            'maneuver': maneuver,
            'transform': used_transform,
            'profile': profile,
            'commands': commands,
            'collisions': collision_events[:],
            'api_metrics': api_metrics,
        }
    finally:
        lateral.destroy_actors(world, actors)


def empty_diff_metrics(suffix, include_final=True):
    metrics = {
        'max_position_diff_%s' % suffix: '',
        'rms_position_diff_%s' % suffix: '',
        'max_speed_diff_%s' % suffix: '',
        'rms_speed_diff_%s' % suffix: '',
        'max_roll_diff_%s' % suffix: '',
        'rms_roll_diff_%s' % suffix: '',
        'max_pitch_diff_%s' % suffix: '',
        'rms_pitch_diff_%s' % suffix: '',
        'max_yaw_rate_diff_%s' % suffix: '',
        'rms_yaw_rate_diff_%s' % suffix: '',
        'max_lateral_acc_diff_%s' % suffix: '',
        'rms_lateral_acc_diff_%s' % suffix: '',
    }
    if include_final:
        metrics.update({
            'final_position_diff_%s' % suffix: '',
            'final_speed_diff_%s' % suffix: '',
            'final_roll_diff_%s' % suffix: '',
            'final_pitch_diff_%s' % suffix: '',
            'final_yaw_rate_diff_%s' % suffix: '',
            'final_lateral_acc_diff_%s' % suffix: '',
        })
    return metrics


def diff_metrics(baseline, result, suffix, start_step=0, include_final=True):
    if baseline is None:
        return empty_diff_metrics(suffix, include_final=include_final)

    baseline_profile = baseline['profile']
    profile = result['profile']
    if len(baseline_profile) != len(profile):
        raise RuntimeError('%s %s profile length mismatch' % (
            result['scenario']['name'],
            result['maneuver']['name']))

    pairs = [
        (base, row)
        for base, row in zip(baseline_profile, profile)
        if row['step'] >= start_step
    ]
    if not pairs:
        return empty_diff_metrics(suffix, include_final=include_final)

    position_diffs = []
    speed_diffs = []
    roll_diffs = []
    pitch_diffs = []
    yaw_rate_diffs = []
    lateral_acc_diffs = []

    for base, row in pairs:
        position_diffs.append(lateral.distance_xyz(
            base['x'], base['y'], base['z'],
            row['x'], row['y'], row['z']))
        speed_diffs.append(abs(base['speed'] - row['speed']))
        roll_diffs.append(abs(base['roll'] - row['roll']))
        pitch_diffs.append(abs(base['pitch'] - row['pitch']))
        yaw_rate_diffs.append(abs(base['yaw_rate'] - row['yaw_rate']))
        lateral_acc_diffs.append(abs(base['local_ay'] - row['local_ay']))

    metrics = {
        'max_position_diff_%s' % suffix: max(position_diffs),
        'rms_position_diff_%s' % suffix: lateral.rms(position_diffs),
        'max_speed_diff_%s' % suffix: max(speed_diffs),
        'rms_speed_diff_%s' % suffix: lateral.rms(speed_diffs),
        'max_roll_diff_%s' % suffix: max(roll_diffs),
        'rms_roll_diff_%s' % suffix: lateral.rms(roll_diffs),
        'max_pitch_diff_%s' % suffix: max(pitch_diffs),
        'rms_pitch_diff_%s' % suffix: lateral.rms(pitch_diffs),
        'max_yaw_rate_diff_%s' % suffix: max(yaw_rate_diffs),
        'rms_yaw_rate_diff_%s' % suffix: lateral.rms(yaw_rate_diffs),
        'max_lateral_acc_diff_%s' % suffix: max(lateral_acc_diffs),
        'rms_lateral_acc_diff_%s' % suffix: lateral.rms(lateral_acc_diffs),
    }

    if include_final:
        base_final = baseline_profile[-1]
        final = profile[-1]
        metrics.update({
            'final_position_diff_%s' % suffix: lateral.distance_xyz(
                base_final['x'], base_final['y'], base_final['z'],
                final['x'], final['y'], final['z']),
            'final_speed_diff_%s' % suffix:
                abs(base_final['speed'] - final['speed']),
            'final_roll_diff_%s' % suffix:
                abs(base_final['roll'] - final['roll']),
            'final_pitch_diff_%s' % suffix:
                abs(base_final['pitch'] - final['pitch']),
            'final_yaw_rate_diff_%s' % suffix:
                abs(base_final['yaw_rate'] - final['yaw_rate']),
            'final_lateral_acc_diff_%s' % suffix:
                abs(base_final['local_ay'] - final['local_ay']),
        })

    return metrics


def base_response_metrics(result):
    profile = result['profile']
    actor_ids = set(row['actor_id'] for row in profile)
    return {
        'scenario': result['scenario']['name'],
        'label': result['scenario']['label'],
        'maneuver': result['maneuver']['name'],
        'maneuver_type': result['maneuver']['type'],
        'uses_suspension_api': result['scenario']['uses_suspension_api'],
        'apply_policy': result['scenario']['apply_policy'],
        'schedule_type': result['scenario']['schedule_type'],
        'collisions': len(result['collisions']),
        'nan_count': 0,
        'actor_ids': ';'.join(str(actor_id) for actor_id in sorted(actor_ids)),
        'actor_id_changed': int(len(actor_ids) != 1),
        'peak_abs_roll': max(abs(row['roll']) for row in profile),
        'peak_abs_pitch': max(abs(row['pitch']) for row in profile),
        'peak_abs_yaw_rate': max(abs(row['yaw_rate']) for row in profile),
        'peak_abs_lateral_acc': max(abs(row['local_ay']) for row in profile),
        'rms_roll': lateral.rms([row['roll'] for row in profile]),
        'rms_pitch': lateral.rms([row['pitch'] for row in profile]),
        'rms_yaw_rate': lateral.rms([row['yaw_rate'] for row in profile]),
        'rms_lateral_acc': lateral.rms([row['local_ay'] for row in profile]),
    }


def build_summary(results, args):
    by_maneuver = {}
    for result in results:
        by_maneuver.setdefault(result['maneuver']['name'], []).append(result)

    summary = []
    for maneuver_name, maneuver_results in by_maneuver.items():
        stock = None
        identity = None
        for result in maneuver_results:
            if result['scenario']['name'] == 'R0_stock':
                stock = result
            elif result['scenario']['name'] == 'R1_identity_every_tick':
                identity = result

        if stock is None:
            raise RuntimeError('missing R0_stock baseline for %s' % maneuver_name)
        if identity is None:
            raise RuntimeError(
                'missing R1_identity_every_tick baseline for %s' %
                maneuver_name)

        for result in maneuver_results:
            row = base_response_metrics(result)
            row.update(result['api_metrics'])
            row.update(diff_metrics(
                stock,
                result,
                'vs_stock',
                start_step=0,
                include_final=True))
            row.update(diff_metrics(
                identity,
                result,
                'vs_identity',
                start_step=0,
                include_final=True))
            row.update(diff_metrics(
                identity,
                result,
                'post_switch_vs_identity',
                start_step=args.effect_start_step,
                include_final=False))
            summary.append(row)

    return summary


def print_summary(summary_rows):
    print('')
    print('Runtime suspension summary against R1 identity_every_tick:')
    print('%-22s %-23s %7s %7s %7s %7s %9s %9s %7s %5s' % (
        'maneuver',
        'scenario',
        'apply',
        'cmdchg',
        'rbchg',
        'mismatch',
        'max_pos',
        'rms_yaw',
        'latmax',
        'coll'))
    for row in summary_rows:
        max_latency = row['max_apply_latency_ticks']
        print('%-22s %-23s %7s %7s %7s %7s %9.4f %9.4f %7s %5d' % (
            row['maneuver'],
            row['scenario'],
            row['apply_count'],
            row['command_change_count'],
            row['readback_change_count'],
            row['readback_mismatch_count'],
            row['max_position_diff_vs_identity'],
            row['rms_yaw_rate_diff_post_switch_vs_identity'],
            '' if max_latency == '' else '%0.0f' % max_latency,
            row['collisions']))


def check_identity(summary_rows, args):
    failures = []
    identity_rows = 0

    for row in summary_rows:
        if row['scenario'] != 'R1_identity_every_tick':
            continue
        identity_rows += 1
        if row['final_position_diff_vs_stock'] > args.identity_final_position_threshold:
            failures.append('%s final position diff %0.6f > %0.6f m' % (
                row['maneuver'],
                row['final_position_diff_vs_stock'],
                args.identity_final_position_threshold))
        if row['max_speed_diff_vs_stock'] > args.identity_speed_threshold:
            failures.append('%s max speed diff %0.6f > %0.6f m/s' % (
                row['maneuver'],
                row['max_speed_diff_vs_stock'],
                args.identity_speed_threshold))
        if row['max_roll_diff_vs_stock'] > args.identity_angle_threshold:
            failures.append('%s max roll diff %0.6f > %0.6f deg' % (
                row['maneuver'],
                row['max_roll_diff_vs_stock'],
                args.identity_angle_threshold))
        if row['max_pitch_diff_vs_stock'] > args.identity_angle_threshold:
            failures.append('%s max pitch diff %0.6f > %0.6f deg' % (
                row['maneuver'],
                row['max_pitch_diff_vs_stock'],
                args.identity_angle_threshold))
        if row['max_yaw_rate_diff_vs_stock'] > args.identity_yaw_rate_threshold:
            failures.append('%s max yaw-rate diff %0.6f > %0.6f deg/s' % (
                row['maneuver'],
                row['max_yaw_rate_diff_vs_stock'],
                args.identity_yaw_rate_threshold))
        if row['readback_mismatch_count'] != 0:
            failures.append('%s identity readback mismatch count is %d' % (
                row['maneuver'],
                row['readback_mismatch_count']))
        if row['collisions'] > 0:
            failures.append('%s identity collision count is %d' % (
                row['maneuver'],
                row['collisions']))

    if identity_rows == 0:
        raise RuntimeError('missing R1_identity_every_tick summary row')

    if failures:
        print('')
        print('FAILED runtime identity guardrail:')
        for failure in failures:
            print('  - %s' % failure)
        raise RuntimeError('runtime identity guardrail failed')

    print('')
    print('PASS: R1 identity_every_tick matched R0 stock within thresholds.')


def check_runtime_api(summary_rows, args):
    failures = []

    for row in summary_rows:
        if not row['uses_suspension_api']:
            continue
        if row['readback_mismatch_count'] != 0:
            failures.append('%s %s readback mismatches=%d' % (
                row['maneuver'],
                row['scenario'],
                row['readback_mismatch_count']))
        if row['apply_latency_failure_count'] != 0:
            failures.append('%s %s latency failures=%d' % (
                row['maneuver'],
                row['scenario'],
                row['apply_latency_failure_count']))
        if (row['max_apply_latency_ticks'] != '' and
                row['max_apply_latency_ticks'] > args.max_apply_latency_ticks):
            failures.append('%s %s max latency %s > %d ticks' % (
                row['maneuver'],
                row['scenario'],
                row['max_apply_latency_ticks'],
                args.max_apply_latency_ticks))
        if row['actor_id_changed']:
            failures.append('%s %s actor id changed: %s' % (
                row['maneuver'],
                row['scenario'],
                row['actor_ids']))
        if row['collisions'] > 0:
            failures.append('%s %s collision count is %d' % (
                row['maneuver'],
                row['scenario'],
                row['collisions']))

    if failures:
        print('')
        print('FAILED runtime API checks:')
        for failure in failures:
            print('  - %s' % failure)
        raise RuntimeError('runtime API checks failed')

    print('PASS: runtime API readback, latency, actor-id, and collision checks passed.')


def output_dir_path(args):
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, output_dir)
    return output_dir


def summary_fields():
    return (
        'maneuver',
        'maneuver_type',
        'scenario',
        'label',
        'uses_suspension_api',
        'apply_policy',
        'schedule_type',
        'collisions',
        'nan_count',
        'actor_ids',
        'actor_id_changed',
        'apply_count',
        'verify_count',
        'command_change_count',
        'readback_change_count',
        'readback_mismatch_count',
        'immediate_mismatch_count',
        'apply_latency_failure_count',
        'mean_apply_latency_ticks',
        'max_apply_latency_ticks',
        'max_readback_spring_scale_error',
        'max_readback_damper_scale_error',
        'peak_abs_roll',
        'peak_abs_pitch',
        'peak_abs_yaw_rate',
        'peak_abs_lateral_acc',
        'rms_roll',
        'rms_pitch',
        'rms_yaw_rate',
        'rms_lateral_acc',
        'final_position_diff_vs_stock',
        'final_speed_diff_vs_stock',
        'final_roll_diff_vs_stock',
        'final_pitch_diff_vs_stock',
        'final_yaw_rate_diff_vs_stock',
        'final_lateral_acc_diff_vs_stock',
        'max_position_diff_vs_stock',
        'rms_position_diff_vs_stock',
        'max_speed_diff_vs_stock',
        'rms_speed_diff_vs_stock',
        'max_roll_diff_vs_stock',
        'rms_roll_diff_vs_stock',
        'max_pitch_diff_vs_stock',
        'rms_pitch_diff_vs_stock',
        'max_yaw_rate_diff_vs_stock',
        'rms_yaw_rate_diff_vs_stock',
        'max_lateral_acc_diff_vs_stock',
        'rms_lateral_acc_diff_vs_stock',
        'final_position_diff_vs_identity',
        'final_speed_diff_vs_identity',
        'final_roll_diff_vs_identity',
        'final_pitch_diff_vs_identity',
        'final_yaw_rate_diff_vs_identity',
        'final_lateral_acc_diff_vs_identity',
        'max_position_diff_vs_identity',
        'rms_position_diff_vs_identity',
        'max_speed_diff_vs_identity',
        'rms_speed_diff_vs_identity',
        'max_roll_diff_vs_identity',
        'rms_roll_diff_vs_identity',
        'max_pitch_diff_vs_identity',
        'rms_pitch_diff_vs_identity',
        'max_yaw_rate_diff_vs_identity',
        'rms_yaw_rate_diff_vs_identity',
        'max_lateral_acc_diff_vs_identity',
        'rms_lateral_acc_diff_vs_identity',
        'max_position_diff_post_switch_vs_identity',
        'rms_position_diff_post_switch_vs_identity',
        'max_speed_diff_post_switch_vs_identity',
        'rms_speed_diff_post_switch_vs_identity',
        'max_roll_diff_post_switch_vs_identity',
        'rms_roll_diff_post_switch_vs_identity',
        'max_pitch_diff_post_switch_vs_identity',
        'rms_pitch_diff_post_switch_vs_identity',
        'max_yaw_rate_diff_post_switch_vs_identity',
        'rms_yaw_rate_diff_post_switch_vs_identity',
        'max_lateral_acc_diff_post_switch_vs_identity',
        'rms_lateral_acc_diff_post_switch_vs_identity',
    )


def profile_fields():
    return (
        'scenario',
        'maneuver',
        'maneuver_type',
        'step',
        'frame',
        'elapsed_seconds',
        'actor_id',
        'event',
        'mode',
        'apply_policy',
        'commanded_spring_scale',
        'commanded_damper_scale',
        'readback_spring_scale',
        'readback_damper_scale',
        'readback_spring_scale_error',
        'readback_damper_scale_error',
        'readback_match',
        'apply_latency_ticks',
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


def command_fields():
    return (
        'scenario',
        'maneuver',
        'maneuver_type',
        'step',
        'frame_before_apply',
        'frame_after_tick',
        'actor_id',
        'event',
        'mode',
        'apply_policy',
        'commanded_spring_scale',
        'commanded_damper_scale',
        'immediate_readback_spring_scale',
        'immediate_readback_damper_scale',
        'immediate_readback_spring_scale_error',
        'immediate_readback_damper_scale_error',
        'immediate_readback_match',
        'post_tick_readback_spring_scale',
        'post_tick_readback_damper_scale',
        'post_tick_readback_spring_scale_error',
        'post_tick_readback_damper_scale_error',
        'post_tick_readback_match',
        'apply_latency_ticks',
        'throttle',
        'brake',
        'steer',
    )


def write_csv_outputs(results, summary_rows, args):
    if not args.write_csv:
        return

    output_dir = output_dir_path(args)
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, '%s_summary.csv' % args.output_prefix)
    profile_path = os.path.join(output_dir, '%s_profiles.csv' % args.output_prefix)
    command_path = os.path.join(output_dir, '%s_commands.csv' % args.output_prefix)
    collision_path = os.path.join(
        output_dir,
        '%s_collisions.csv' % args.output_prefix)

    with open(summary_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=summary_fields())
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field, '') for field in summary_fields()})

    with open(profile_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=profile_fields())
        writer.writeheader()
        for result in results:
            for row in result['profile']:
                writer.writerow({field: row.get(field, '') for field in profile_fields()})

    with open(command_path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=command_fields())
        writer.writeheader()
        for result in results:
            for row in result['commands']:
                writer.writerow({field: row.get(field, '') for field in command_fields()})

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
    print('  %s' % command_path)
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

        blueprint = lateral.find_target_blueprint(world)
        scenarios = scenario_definitions(
            include_diagnostic=args.include_diagnostic)
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
        check_runtime_api(summary_rows, args)
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
        '--frames',
        default=340,
        type=int,
        help='ticks per maneuver (default: 340)')
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
        '--throttle',
        default=0.30,
        type=float,
        help='straight/brake-in-turn throttle before brake-start '
             '(default: 0.30)')
    argparser.add_argument(
        '--steer-throttle',
        default=0.25,
        type=float,
        help='throttle for constant-steer maneuver (default: 0.25)')
    argparser.add_argument(
        '--brake',
        default=0.35,
        type=float,
        help='brake after brake-start (default: 0.35)')
    argparser.add_argument(
        '--brake-start',
        default=120,
        type=int,
        help='tick when braking starts for brake maneuvers (default: 120)')
    argparser.add_argument(
        '--constant-steer',
        default=0.20,
        type=float,
        help='constant steer value for lateral maneuver (default: 0.20)')
    argparser.add_argument(
        '--brake-turn-steer',
        default=0.20,
        type=float,
        help='steer value for brake-in-turn maneuver (default: 0.20)')
    argparser.add_argument(
        '--effect-start-step',
        default=100,
        type=int,
        help='first step used for post-switch response metrics '
             '(default: 100)')
    argparser.add_argument(
        '--readback-scale-tolerance',
        default=1.0e-4,
        type=float,
        help='allowed per-wheel spring/damper scale readback error '
             '(default: 1e-4)')
    argparser.add_argument(
        '--readback-change-tolerance',
        default=1.0e-4,
        type=float,
        help='scale difference threshold for counting readback mode changes '
             '(default: 1e-4)')
    argparser.add_argument(
        '--max-apply-latency-ticks',
        default=1,
        type=int,
        help='maximum acceptable command readback latency in ticks '
             '(default: 1)')
    argparser.add_argument(
        '--identity-final-position-threshold',
        default=0.15,
        type=float,
        help='allowed R1 final location difference in meters (default: 0.15)')
    argparser.add_argument(
        '--identity-speed-threshold',
        default=0.05,
        type=float,
        help='allowed R1 max speed difference in m/s (default: 0.05)')
    argparser.add_argument(
        '--identity-angle-threshold',
        default=0.05,
        type=float,
        help='allowed R1 max roll/pitch difference in degrees (default: 0.05)')
    argparser.add_argument(
        '--identity-yaw-rate-threshold',
        default=0.05,
        type=float,
        help='allowed R1 max yaw-rate difference in deg/s (default: 0.05)')
    argparser.add_argument(
        '--no-diagnostic',
        dest='include_diagnostic',
        action='store_false',
        help='skip R3 c=1.20/0.80 diagnostic step scenario')
    argparser.add_argument(
        '--output-dir',
        default='results',
        help='CSV output directory, relative to this script unless absolute '
             '(default: results)')
    argparser.add_argument(
        '--output-prefix',
        default='suspension_runtime_switching',
        help='CSV output filename prefix (default: '
             'suspension_runtime_switching)')
    argparser.add_argument(
        '--no-write-csv',
        dest='write_csv',
        action='store_false',
        help='disable CSV output')
    argparser.set_defaults(include_diagnostic=True, write_csv=True)

    try:
        main(argparser.parse_args())
    except KeyboardInterrupt:
        print(' - Exited by user.')

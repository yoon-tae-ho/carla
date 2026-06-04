# Residual-RL Suspension Control

This package adds a conservative residual-RL suspension controller for CARLA
experiments. The controller does not replace PID or skyhook. It adds a small,
bounded damping residual on top of a stabilizing baseline and returns the
baseline command exactly whenever policy, observation, action, or safety gates
are invalid.

## Architecture

```text
VehicleState + PlanningInfo + baseline diagnostics
        |
        v
ObservationBuilder + FixedScaleNormalizer
        |
        v
PolicyAdapter -> residual action [-1, 1]^4
        |
        v
safety gate + residual scale + rate limit
        |
        v
baseline command + bounded damper residual
```

The default baseline is `SkyhookController`. `FeedbackPIDController` is
available as an ablation baseline. Springs are frozen at `1.0` by default.

## Observation

The Stage 04 observation vector has 56 features:

- vehicle dynamics: speed, local velocity, acceleration, roll, pitch, angular
  rate, jerk, and body activity;
- baseline diagnostics: per-wheel baseline dampers and skyhook corner velocity;
- memory: previous residual action and previous damper command;
- planning preview: curvature, target speed, predicted acceleration, steering,
  throttle, brake, and time-to-event aggregates.

If planning preview is unavailable, planning features are zero and
`planning_available` is `0.0`.

## Action

The first supported action space is four damping residuals:

```text
[delta_damper_fl, delta_damper_fr, delta_damper_rl, delta_damper_rr]
```

Each value is clipped to `[-1, 1]`, scaled by
`max_damper_residual_scale`, gated by safety, rate-limited, and clamped to
`min_damper_scale` / `max_damper_scale`. Spring residuals remain disabled.

## Reward

`SuspensionReward` returns a dense reward and decomposed diagnostics:

- `reward_comfort`
- `reward_stability`
- `reward_task`
- `reward_action`
- `reward_safety`
- `reward_total`

Each group also records term-level costs, so failed training runs can be
debugged from rollout logs.

## Planning JSONL

The sidecar can read file-based planning preview:

```text
--planning-provider jsonl --planning-preview-jsonl /path/to/planning_preview.jsonl
```

The expected line format is documented in
`suspension_control/docs/planning_preview_jsonl.md`. The provider ignores
malformed lines, counts parse failures, rejects stale frames, and degrades to
empty planning if no valid preview exists.

## Smoke Tests

From `PythonAPI/taeho`:

```bash
bash scripts/check_rl_suspension.sh
```

Equivalent commands:

```bash
python -m py_compile \
  suspension_control/controllers/base.py \
  suspension_control/controllers/rl_residual.py \
  suspension_control/rl/observations.py \
  suspension_control/rl/reward.py \
  suspension_control/rl/policy.py \
  suspension_control/runtime/planning_provider.py \
  suspension_control/rl/evaluate_policy.py \
  transfuser_suspension_control_suite.py

python -m pytest -q tests/test_rl_residual_controller.py \
  tests/test_rl_observations.py tests/test_rl_reward.py \
  tests/test_planning_provider.py tests/test_rl_policy_fallback.py

python -m suspension_control.rl.evaluate_policy \
  --dry-run --steps 100 --output-dir /tmp/rl_suspension_dry_run
```

## Phase 0 Metrics

Route reports should treat the first 3 seconds as controller/sensor warmup.
Use `warmup_excluded_*` values as the main metrics; raw `comfort_*` and
`stability_*` values are diagnostic context. The route suite default is:

```text
--metric-warmup-seconds 3.0
```

The dry-run training scaffold also masks rewards before
`initial_transition_skip_seconds: 3.0`, so early transient sensor values do not
drive the reward signal.

## Zero-Residual Equivalence

Before training or evaluating a policy, replay an existing `profile.csv` offline:

```bash
python suspension_zero_residual_equivalence.py \
  --profile /path/to/profile.csv \
  --baselines skyhook,pid
```

The acceptance criteria are:

- `max_abs_damper_diff <= 1e-6`
- `max_abs_spring_diff == 0`
- `max_rl_mean_abs_action == 0`
- `max_rl_mean_abs_residual_damper == 0`

This proves `rl_residual_skyhook` with a zero policy is command-equivalent to
skyhook, and `rl_residual_pid` with a zero policy is command-equivalent to PID
on the same `VehicleState` sequence.

## Phase 2 Online Dummy Policy

After Phase 1 passes, run the online dummy-policy route suite:

```bash
cd /home/yth/sim/docker
bash run_phase2_online_dummy.sh \
  --routes-subset 00 \
  --seeds 100 \
  --debug 0 \
  --sidecar-tick-mode poll \
  --poll-seconds 0.05 \
  --pause-between-runs 30 \
  --stop-on-failure \
  --output-dir /home/yth/sim/e2e_models/outputs/rl_suspension_phase2_dummy_seed100_route00
```

The `rl_zero_residual_*` scenarios load the explicit built-in `dummy_zero`
policy. They should not rely on missing-policy fallback. The suite writes:

```text
phase2_online_dummy_acceptance.csv
```

A passing zero-residual row has:

- route score 100 and no infractions
- `sidecar_command_verifies > 0`
- `policy_available_ratio >= 0.99`
- `rl_fallback_rows == 0`
- `rl_mean_abs_action == 0`
- `rl_mean_abs_residual_damper == 0`
- matching route score/penalty versus its paired PID or skyhook baseline

Phase 2 also records safety-gate diagnostics in `controller_diagnostics.csv`:

- `rl_safety_gate_active`
- `rl_safety_gate_reason`
- `rl_safety_gate_roll_limit`
- `rl_safety_gate_lateral_acc_limit`
- `rl_safety_gate_yaw_rate_limit`
- `rl_safety_gate_nonfinite_obs`
- `rl_safety_gate_action_invalid`

The suite summary and `phase2_online_dummy_acceptance.csv` include
`safety_gate_active_ratio`, reason counts, and speed/roll/lateral-acc/yaw-rate
statistics for frames where the safety gate is active.

## Training

Training is scaffolded and requires optional dependencies:

```bash
python -m suspension_control.rl.train_sac \
  --algorithm sac \
  --total-timesteps 10000 \
  --output-dir /tmp/rl_suspension_train
```

If `gymnasium` or `stable-baselines3` is missing, the command exits with a
clear dependency message.

## Evaluation

No route-level improvement is claimed until CARLA routes are actually run.
Recommended no-planning smoke:

```bash
python transfuser_suspension_control_suite.py \
  --scenarios stock,identity,pid,skyhook,rl_residual_skyhook \
  --planning-provider empty \
  --output-dir /tmp/rl_suspension_no_planning
```

Recommended planning-preview experiment shape:

```bash
python transfuser_suspension_control_suite.py \
  --scenarios stock,identity,pid,skyhook,rl_residual_skyhook,rl_residual_pid,rl_residual_skyhook_no_planning \
  --seeds 100,101,102 \
  --routes /path/to/routes.xml \
  --rl-residual-config suspension_control/configs/rl_residual.yaml \
  --rl-policy /path/to/policy.pt \
  --rl-normalizer /path/to/normalizer.json \
  --planning-provider jsonl \
  --planning-preview-jsonl /path/to/planning_preview.jsonl \
  --output-dir /path/to/output
```

## Acceptance Criteria

A route result is acceptable only if:

- sidecar has no fatal errors;
- controller commands validate and remain finite;
- observations, actions, and rewards remain finite;
- route infractions do not increase versus skyhook on the same seed/route;
- leaderboard scores are not materially worse than skyhook and stock;
- warmup-excluded comfort improves versus skyhook for most route/seed pairs;
- stability metrics do not regress beyond the chosen tolerance;
- missing policy or missing planning is explained by diagnostics and remains
  baseline-equivalent.

## Ablations

Use these suite scenario keys:

```text
stock
identity
pid
skyhook
rl_residual_skyhook
rl_residual_pid
rl_residual_skyhook_no_planning
rl_zero_residual_pid
rl_zero_residual_skyhook
```

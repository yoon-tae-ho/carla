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
```

# Suspension Control Package

This package keeps suspension-control policy code separate from CARLA runtime
plumbing. Existing experiment scripts in `PythonAPI/taeho` can gradually import
from here instead of duplicating suspension scaling, readback checks, and metric
logic.

## Layout

```text
suspension_control/
  controllers/   Pure controller logic. No CARLA import required.
  runtime/       CARLA adapters, loop helpers, and observers.
  metrics/       Stability and comfort summaries for profile rows.
  configs/       Human-readable defaults for controller experiments.
  tools/         Read-only inspection and validation utilities.
```

## Controller Contract

Controllers implement:

```python
output = controller.compute(context)
```

where `context` contains current vehicle feedback, optional planning preview,
and native/current CARLA suspension data. The output contains a
`SuspensionCommand`, expressed as spring/damper scale factors relative to the
native suspension. The runtime adapter converts these scale factors to CARLA's
absolute `SuspensionPhysicsControl`.

This makes the current feedback PID and future planning-preview controllers
comparable through the same interface.

## Initial Baselines

- `IdentityController`: applies `spring_scale=1.0`, `damper_scale=1.0`.
- `StaticScaleController`: fixed uniform spring/damper scale.
- `FeedbackPIDController`: feedback-only PID baseline that adjusts damping from
  roll, pitch, body-rate, lateral-acceleration, and vertical-acceleration
  activity.
- `PlanningPreviewSuspensionController`: placeholder for the future controller
  that will use autonomous-driving planning outputs.

## Minimal Runtime Use

```python
from suspension_control.controllers.pid import FeedbackPIDController
from suspension_control.runtime.loop import SuspensionControlLoop

controller = FeedbackPIDController()
loop = SuspensionControlLoop(world, vehicle, controller)
loop.reset()

for step in range(300):
    vehicle_control = make_vehicle_control(step)
    result = loop.step(vehicle_control=vehicle_control, step_index=step)

loop.restore_native()
```

The loop assumes synchronous CARLA ticking, matching the existing suspension
experiments.

## Suspension State Inspection

`tools/inspect_suspension_state.py` validates the Python-facing
`vehicle.get_suspension_state()` API before any controller consumes the new
state. It attaches to a running CARLA world, finds the hero vehicle, waits for
external ticks or polls by wall-clock sleep, and writes per-wheel CSV rows. It
does not apply suspension commands and does not advance the world.

Example:

```bash
python carla-0.9.15/PythonAPI/taeho/suspension_control/tools/inspect_suspension_state.py \
  --role-name hero \
  --duration-sec 10 \
  --output /tmp/suspension_state.csv \
  --report-output markdown/for_carla/semiactive_suspension_state_skyhook_v5_steps/PR1_RUNTIME_VALIDATION_REPORT.md \
  --print-summary
```

The CSV includes raw suspension offset, provisional compression, finite-
difference velocity, contact validity, `wheel_in_air`, normalized tire load
fields, and empty force columns reserved for future backend exposure.

# Planning Preview JSONL Interface

The suspension sidecar can read planning preview from a file when launched with:

```text
--planning-provider jsonl --planning-preview-jsonl /path/to/planning_preview.jsonl
```

The autonomous-driving agent should append one JSON object per line:

```json
{
  "frame": 12345,
  "timestamp": 12.34,
  "horizon_dt": 0.1,
  "trajectory_xy": [[0.0, 0.0], [1.0, 0.1]],
  "trajectory_yaw": [0.0, 0.02],
  "target_speed": [8.0, 8.2],
  "curvature": [0.0, 0.01],
  "steer": [0.03, 0.04],
  "throttle": [0.4, 0.4],
  "brake": [0.0, 0.0],
  "predicted_ax": [0.2, 0.1],
  "predicted_ay": [0.1, 0.2],
  "route_deviation": 0.0,
  "extra": {"agent": "transfuserpp"}
}
```

If `SUSPENSION_PLANNING_PREVIEW_JSONL` is set in the TransFuser++ process, a
future writer hook should append this format without blocking or changing
driving behavior when writing fails. The hook is not implemented in this
repository because the current task is limited to `/home/yth/sim/carla-0.9.15`.

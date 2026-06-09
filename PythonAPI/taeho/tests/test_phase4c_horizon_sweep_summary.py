import csv
import json

from suspension_control.rl.phase4c_horizon_sweep import (
    write_phase4c_horizon_sweep_summary,
)


def test_phase4c_horizon_sweep_passes_default_healthy_horizons(tmp_path):
    run_dirs = [
        _write_run(tmp_path, 512, progress=0.04),
        _write_run(tmp_path, 1024, progress=0.10),
        _write_run(tmp_path, 2048, progress=0.30),
    ]

    csv_path, json_path, rows, summary = write_phase4c_horizon_sweep_summary(
        str(tmp_path),
        run_dirs=[str(path) for path in run_dirs])

    assert summary["phase4c_horizon_sweep_status"] == "pass"
    assert [row["horizon_steps"] for row in rows] == [512, 1024, 2048]
    assert all(row["phase4c_horizon_status"] == "pass" for row in rows)
    with open(csv_path) as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    assert len(csv_rows) == 3
    with open(json_path) as json_file:
        json_summary = json.load(json_file)
    assert json_summary["row_count"] == 3


def test_phase4c_horizon_sweep_accepts_2048_progress_point30(tmp_path):
    run_dir = _write_run(tmp_path, 2048, progress=0.30)

    _, _, rows, summary = write_phase4c_horizon_sweep_summary(
        str(tmp_path),
        run_dirs=[str(run_dir)])

    assert summary["phase4c_horizon_sweep_status"] == "pass"
    assert rows[0]["progress_min_threshold"] == 0.18
    assert rows[0]["progress_min_ok"] == 1
    assert rows[0]["phase4c_horizon_status"] == "pass"


def test_phase4c_horizon_sweep_fails_hard_safety(tmp_path):
    run_dir = _write_run(tmp_path, 2048, progress=0.30, hard_safety=0.01)

    _, _, rows, summary = write_phase4c_horizon_sweep_summary(
        str(tmp_path),
        run_dirs=[str(run_dir)])

    assert summary["phase4c_horizon_sweep_status"] == "fail"
    assert rows[0]["phase4c_horizon_status"] == "fail"
    assert "hard_safety" in rows[0]["failed_checks"]


def test_phase4c_horizon_sweep_fails_high_fallback(tmp_path):
    run_dir = _write_run(tmp_path, 2048, progress=0.30, fallback=0.46)

    _, _, rows, summary = write_phase4c_horizon_sweep_summary(
        str(tmp_path),
        run_dirs=[str(run_dir)])

    assert summary["phase4c_horizon_sweep_status"] == "fail"
    assert rows[0]["phase4c_horizon_status"] == "fail"
    assert "fallback" in rows[0]["failed_checks"]


def _write_run(
    root,
    steps,
    *,
    progress,
    fallback=0.26,
    low_speed=0.26,
    hard_safety=0.0,
):
    run_dir = root / ("rl_suspension_phase4c_train_%dsteps_seed100_route00_test" % steps)
    run_dir.mkdir()
    _write_json(run_dir / "phase4_route_health.json", {
        "phase4_route_health_status": "pass",
        "total_timesteps_requested": steps,
        "total_timesteps_collected": steps,
        "real_backend_used": 1,
        "fake_backend_used": 0,
        "carla_connected": 1,
        "route_process_started": 1,
        "hero_attached": 1,
        "reward_row_ratio": 1.0,
        "route_progress_available_ratio": 1.0,
        "route_progress_monotonic_fraction_max": progress,
        "route_progress_monotonic_negative_rows": 0,
        "hard_safety_gate_ratio": hard_safety,
        "low_speed_mask_ratio": low_speed,
        "fallback_ratio": fallback,
        "effective_control_ratio": 0.74,
        "mean_abs_action": 0.52,
        "mean_abs_residual_damper": 0.02,
        "observation_clip_ratio": 0.0,
        "collision_count": 0,
        "lane_invasion_count": 0,
        "red_light_count": 0,
        "blocked_vehicle_count": 0,
        "route_timeout_count": 0,
    })
    _write_json(run_dir / "training_summary.json", {
        "total_timesteps": steps,
        "rollout_rows": steps,
    })
    return run_dir


def _write_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)

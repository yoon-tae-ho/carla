import csv
import json

from suspension_control.rl.phase4_route_health import (
    phase4_route_health_bucket_rows,
    write_phase4_route_health,
)


def test_phase4_route_health_passes_2048_progress_point30(tmp_path):
    rows = [
        _healthy_row(
            index,
            progress=0.30 * index / 2047.0,
            fallback=index < 532,
            low_speed=index < 532)
        for index in range(2048)
    ]
    _write_output(tmp_path, rows, total_timesteps=2048)

    _, _, buckets_path, health, _ = write_phase4_route_health(str(tmp_path))

    assert health["phase4_route_health_status"] == "pass"
    assert health["total_timesteps_collected"] == 2048
    assert abs(health["route_progress_monotonic_fraction_max"] - 0.30) < 1.0e-9
    assert 0.25 < health["fallback_ratio"] < 0.27
    assert 0.25 < health["low_speed_mask_ratio"] < 0.27
    assert health["hard_safety_gate_ratio"] == 0.0
    with open(buckets_path) as csv_file:
        bucket_rows = list(csv.DictReader(csv_file))
    assert len(bucket_rows) == 8


def test_phase4_route_health_fails_hard_safety_gate(tmp_path):
    rows = [_healthy_row(index, progress=0.2) for index in range(10)]
    rows[0].update({
        "rl_safety_gate_reason": "action_invalid",
        "rl_safety_gate_active": "1",
        "rl_safety_gain": "0",
    })
    _write_output(tmp_path, rows)

    _, _, _, health, _ = write_phase4_route_health(str(tmp_path))

    assert health["phase4_route_health_status"] == "fail"
    assert "hard_safety" in health["failed_checks"]


def test_phase4_route_health_fails_high_fallback_ratio(tmp_path):
    rows = [
        _healthy_row(index, progress=0.2, fallback=index < 6)
        for index in range(10)
    ]
    _write_output(tmp_path, rows)

    _, _, _, health, _ = write_phase4_route_health(str(tmp_path))

    assert health["phase4_route_health_status"] == "fail"
    assert "fallback" in health["failed_checks"]


def test_phase4_route_health_fails_high_low_speed_mask_ratio(tmp_path):
    rows = [
        _healthy_row(index, progress=0.2, low_speed=index < 6)
        for index in range(10)
    ]
    _write_output(tmp_path, rows)

    _, _, _, health, _ = write_phase4_route_health(str(tmp_path))

    assert health["phase4_route_health_status"] == "fail"
    assert "low_speed_mask" in health["failed_checks"]


def test_phase4_route_health_fails_nonfinite_reward(tmp_path):
    rows = [_healthy_row(index, progress=0.2) for index in range(10)]
    rows[3]["reward_total"] = "nan"
    _write_output(tmp_path, rows)

    _, _, _, health, _ = write_phase4_route_health(str(tmp_path))

    assert health["phase4_route_health_status"] == "fail"
    assert "finite_reward" in health["failed_checks"]


def test_phase4_route_health_bucket_assignment():
    progresses = [0.02, 0.07, 0.15, 0.25, 0.35, 0.50, 0.70, 0.90]
    rows = [_healthy_row(index, progress=value) for index, value in enumerate(progresses)]

    bucket_rows = phase4_route_health_bucket_rows(rows)

    assert [row["rows"] for row in bucket_rows] == [1, 1, 1, 1, 1, 1, 1, 1]
    assert bucket_rows[0]["bucket"] == "0.00-0.05"
    assert bucket_rows[-1]["bucket"] == "0.80-1.00"


def test_phase4_route_health_missing_optional_columns_warns(tmp_path):
    rows = [_healthy_row(index, progress=0.2) for index in range(10)]
    for row in rows:
        row.pop("route_progress_available", None)
        row.pop("rl_safety_gain", None)
    _write_output(tmp_path, rows)

    _, _, _, health, _ = write_phase4_route_health(str(tmp_path))

    assert health["phase4_route_health_status"] == "pass"
    assert "route_progress_available_missing" in health["warnings"]
    assert "soft_safety_gain_may_be_underestimated" in health["warnings"]


def _healthy_row(index, progress, fallback=False, low_speed=False):
    return {
        "episode_id": "synthetic",
        "step": str(index),
        "backend": "live",
        "real_backend_used": "1",
        "fake_backend_used": "0",
        "carla_connected": "1",
        "route_process_started": "1",
        "hero_attached": "1",
        "speed": "8.0",
        "reward_total": "0.5",
        "reward_comfort": "-0.1",
        "reward_stability": "-0.1",
        "reward_task": "0.2",
        "reward_action": "-0.01",
        "reward_safety": "0.0",
        "route_progress_available": "1",
        "route_progress_fraction": str(progress),
        "route_progress_monotonic_fraction": str(progress),
        "route_progress_delta_m": "0.1",
        "route_progress_rate_mps": "2.0",
        "route_progress_stall": "0",
        "progress_stall": "0",
        "rl_mean_abs_action": "0.2",
        "rl_mean_abs_residual_damper": "0.02",
        "rl_fallback_reason": "safety_gate_zero" if fallback else "",
        "rl_safety_gate_reason": "speed_below_min" if low_speed else "",
        "rl_safety_gate_active": "1" if low_speed else "0",
        "rl_safety_gain": "0" if low_speed else "1",
        "rl_observation_clip_count": "0",
        "collision_count": "0",
        "lane_invasion_count": "0",
        "red_light_count": "0",
        "blocked_vehicle": "0",
        "route_timeout": "0",
    }


def _write_output(output_dir, rows, total_timesteps=None):
    _write_csv(output_dir / "training_rollout.csv", rows)
    with open(output_dir / "training_rollout.jsonl", "w") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row) + "\n")
    _write_json(output_dir / "training_summary.json", {
        "backend": "live",
        "total_timesteps": total_timesteps or len(rows),
        "rollout_rows": len(rows),
        "real_backend_used": 1,
        "fake_backend_used": 0,
    })
    _write_json(output_dir / "training_config.json", {
        "backend": "live",
        "total_timesteps": total_timesteps or len(rows),
    })
    _write_json(output_dir / "phase4_training_canary.json", {
        "real_backend_used": 1,
        "fake_backend_used": 0,
        "carla_connected": 1,
        "route_process_started": 1,
        "hero_attached": 1,
    })


def _write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)

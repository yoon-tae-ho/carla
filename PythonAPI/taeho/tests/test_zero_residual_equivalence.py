import csv
import os

from suspension_zero_residual_equivalence import run_equivalence


def write_profile(path):
    fields = (
        "scenario",
        "episode_index",
        "step",
        "frame",
        "elapsed_seconds",
        "dt",
        "speed",
        "local_vx",
        "local_vy",
        "vz",
        "local_ax",
        "local_ay",
        "az",
        "roll",
        "pitch",
        "yaw",
        "roll_rate",
        "pitch_rate",
        "yaw_rate",
        "throttle",
        "brake",
        "steer",
    )
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for step in range(12):
            writer.writerow({
                "scenario": "synthetic",
                "episode_index": 0,
                "step": step,
                "frame": 1000 + step,
                "elapsed_seconds": 0.05 * step,
                "dt": 0.05,
                "speed": 8.0 + 0.01 * step,
                "local_vx": 8.0 + 0.01 * step,
                "local_vy": 0.05 * step,
                "vz": 0.01 * step,
                "local_ax": 0.1,
                "local_ay": 0.4 + 0.02 * step,
                "az": 0.2 - 0.01 * step,
                "roll": 0.5 + 0.03 * step,
                "pitch": 0.2 - 0.01 * step,
                "yaw": 1.0,
                "roll_rate": 0.4 + 0.01 * step,
                "pitch_rate": 0.2,
                "yaw_rate": 0.1,
                "throttle": 0.3,
                "brake": 0.0,
                "steer": 0.02,
            })


def test_zero_residual_replay_passes_for_skyhook_and_pid(tmp_path):
    profile = tmp_path / "profile.csv"
    output_dir = tmp_path / "equivalence"
    write_profile(str(profile))

    summary_rows, summary_csv = run_equivalence(
        profile=str(profile),
        output_dir=str(output_dir),
        baselines=("skyhook", "pid"))

    assert os.path.isfile(summary_csv)
    assert [row["baseline"] for row in summary_rows] == ["skyhook", "pid"]
    for row in summary_rows:
        assert row["status"] == "pass"
        assert row["max_abs_damper_diff"] <= 1.0e-6
        assert row["max_abs_spring_diff"] == 0.0
        assert row["max_rl_mean_abs_action"] == 0.0
        assert row["max_rl_mean_abs_residual_damper"] == 0.0
        assert os.path.isfile(row["detail_csv"])

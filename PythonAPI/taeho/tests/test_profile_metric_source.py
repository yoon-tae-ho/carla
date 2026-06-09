import csv

from suspension_control.metrics.comfort import jerk_series
from transfuser_suspension_control_suite import compute_profile_metrics


def _write_profile(path, rows):
    fieldnames = [
        "episode_index",
        "actor_id",
        "step",
        "frame",
        "elapsed_seconds",
        "dt",
        "az",
        "local_ax",
        "local_ay",
        "roll",
        "pitch",
        "yaw_rate",
        "roll_rate",
        "pitch_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(episode, actor, step, elapsed, az):
    return {
        "episode_index": episode,
        "actor_id": actor,
        "step": step,
        "frame": 1000 + step,
        "elapsed_seconds": elapsed,
        "dt": 0.05 if step else 0.0,
        "az": az,
        "local_ax": 0.0,
        "local_ay": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw_rate": 0.0,
        "roll_rate": 0.0,
        "pitch_rate": 0.0,
    }


def test_compute_profile_metrics_uses_main_actor_for_main_warmup(tmp_path):
    profile_path = tmp_path / "profile.csv"
    metrics_path = tmp_path / "metrics_by_episode.csv"
    rows = [
        _row("0", "stale", 0, 774.7, 100.0),
        _row("0", "stale", 1, 774.8, 100.0),
        _row("0", "stale", 2, 774.9, 100.0),
        _row("1", "main", 0, 0.0, 90.0),
        _row("1", "main", 1, 1.0, 80.0),
        _row("1", "main", 2, 3.0, 1.0),
        _row("1", "main", 3, 4.0, 2.0),
    ]
    _write_profile(profile_path, rows)

    summary, episode_rows = compute_profile_metrics(
        {
            "name": "S4_rl_residual_skyhook",
            "label": "S4 residual RL over skyhook",
            "controller": "rl_residual",
        },
        100,
        str(profile_path),
        str(metrics_path),
        default_dt=0.05,
        steady_fraction=0.30,
        metric_warmup_seconds=3.0,
    )

    assert summary["profile_total_rows"] == 7
    assert summary["profile_main_rows"] == 4
    assert summary["profile_rows"] == 4
    assert summary["profile_excluded_stale_rows"] == 3
    assert summary["profile_excluded_non_main_rows"] == 3
    assert summary["profile_main_episode_index"] == "1"
    assert summary["profile_main_actor_id"] == "main"
    assert summary["metric_main_episode_index"] == "1"
    assert summary["metric_main_actor_id"] == "main"
    assert summary["metric_source"] == "metrics_by_episode_main_actor"
    assert summary["metric_actor_switch_detected"] == 1
    assert summary["metric_elapsed_reset_detected"] == 1
    assert summary["warmup_excluded_rows"] == 2
    assert summary["warmup_excluded_valid"] == 1
    assert summary["warmup_excluded_elapsed_min"] == 3.0
    assert summary["warmup_excluded_elapsed_max"] == 4.0
    assert summary["warmup_excluded_comfort_peak_abs_vertical_acc"] == 2.0
    assert "actor_switch_detected" in summary["metric_warning"]
    assert "elapsed_reset_detected" in summary["metric_warning"]

    assert len(episode_rows) == 2
    assert [row["actor_id"] for row in episode_rows] == ["stale", "main"]

    with open(metrics_path) as csv_file:
        written_rows = list(csv.DictReader(csv_file))
    assert written_rows[1]["episode_index"] == "1"
    assert written_rows[1]["actor_id"] == "main"
    assert written_rows[1]["warmup_excluded_elapsed_min"] == "3"


def test_compute_profile_metrics_marks_empty_warmup_invalid(tmp_path):
    profile_path = tmp_path / "profile.csv"
    metrics_path = tmp_path / "metrics_by_episode.csv"
    rows = [
        _row("0", "main", 0, 0.0, 90.0),
        _row("0", "main", 1, 1.0, 80.0),
        _row("0", "main", 2, 2.0, 70.0),
    ]
    _write_profile(profile_path, rows)

    summary, _episode_rows = compute_profile_metrics(
        {
            "name": "S4_rl_residual_skyhook",
            "label": "S4 residual RL over skyhook",
            "controller": "rl_residual",
        },
        100,
        str(profile_path),
        str(metrics_path),
        default_dt=0.05,
        steady_fraction=0.30,
        metric_warmup_seconds=3.0,
    )

    assert summary["warmup_excluded_rows"] == 0
    assert summary["warmup_excluded_valid"] == 0
    assert summary["metric_invalid"] == 1
    assert summary["warmup_excluded_comfort_peak_abs_vertical_acc"] == ""
    assert summary["warmup_excluded_stability_peak_abs_roll"] == ""
    assert summary["metric_warning"] == "warmup_filter_empty"


def test_jerk_series_skips_nonpositive_dt_samples():
    rows = [
        {"elapsed_seconds": 0.0, "dt": 0.0, "az": 0.0},
        {"elapsed_seconds": 0.0, "dt": 0.0, "az": 100.0},
        {"elapsed_seconds": 0.1, "dt": 0.1, "az": 101.0},
    ]

    assert jerk_series(rows, axis="az", default_dt=0.05) == [10.0]

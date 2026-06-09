import json

from suspension_control.rl.phase4d_train_acceptance import (
    phase4d_train_acceptance_row,
    write_phase4d_train_acceptance,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_training_output(tmp_path, *, summary_updates=None, canary_updates=None):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    policy_path = train_dir / "policy.ts"
    normalizer_path = train_dir / "normalizer.json"
    policy_path.write_text("policy")
    normalizer_path.write_text("{}\n")
    summary = {
        "status": "trained",
        "backend": "live",
        "real_backend_used": 1,
        "fake_backend_used": 0,
        "synthetic_training": 0,
        "total_timesteps": 4096,
        "rollout_rows": 4096,
        "policy_exported": 1,
        "normalizer_saved": 1,
        "policy_path": str(policy_path),
        "normalizer_path": str(normalizer_path),
        "learn_error": "",
        "export_error": "",
        "collision_count": 0,
        "lane_invasion_count": 0,
        "red_light_count": 0,
        "route_timeout_count": 0,
        "blocked_vehicle_count": 0,
    }
    canary = {
        "phase4_status": "pass",
        "failed_checks": "",
        "backend": "live",
        "real_backend_used": 1,
        "fake_backend_used": 0,
        "route_process_started": 1,
        "hero_attached": 1,
        "total_timesteps_requested": 4096,
        "total_timesteps_collected": 4096,
        "reward_row_ratio": 1.0,
        "route_progress_available_ratio": 1.0,
        "route_progress_monotonic_fraction_max": 0.67,
        "route_progress_monotonic_negative_rows": 0,
        "observation_clip_ratio": 0.0,
        "hard_safety_gate_ratio": 0.0,
        "policy_exported": 1,
        "normalizer_saved": 1,
        "policy_path": str(policy_path),
        "normalizer_path": str(normalizer_path),
    }
    summary.update(summary_updates or {})
    canary.update(canary_updates or {})
    write_json(train_dir / "training_summary.json", summary)
    write_json(train_dir / "phase4_training_canary.json", canary)
    return train_dir


def test_phase4d_train_acceptance_allows_good_live_training(tmp_path):
    train_dir = make_training_output(tmp_path)

    row = phase4d_train_acceptance_row(training_output_dir=str(train_dir))

    assert row["phase4d_train_status"] == "pass"
    assert row["eval_allowed"] == 1
    assert row["invalid_for_eval"] == 0
    assert row["eval_status"] == "ready_for_eval"
    assert row["failed_checks"] == ""
    assert row["route_progress_fraction_min_required"] == 0.45


def test_phase4d_train_acceptance_allows_1024_canary_progress(tmp_path):
    train_dir = make_training_output(
        tmp_path,
        summary_updates={
            "total_timesteps": 1024,
            "rollout_rows": 1024,
        },
        canary_updates={
            "total_timesteps_requested": 1024,
            "total_timesteps_collected": 1024,
            "route_progress_monotonic_fraction_max": 0.30448830588691284,
        })

    row = phase4d_train_acceptance_row(training_output_dir=str(train_dir))

    assert row["phase4d_train_status"] == "pass"
    assert row["eval_allowed"] == 1
    assert row["invalid_for_eval"] == 0
    assert row["failed_checks"] == ""
    assert row["route_progress_fraction_min_required"] == 0.25


def test_phase4d_train_acceptance_blocks_low_1024_canary_progress(tmp_path):
    train_dir = make_training_output(
        tmp_path,
        summary_updates={
            "total_timesteps": 1024,
            "rollout_rows": 1024,
        },
        canary_updates={
            "total_timesteps_requested": 1024,
            "total_timesteps_collected": 1024,
            "route_progress_monotonic_fraction_max": 0.20,
        })

    row = phase4d_train_acceptance_row(training_output_dir=str(train_dir))

    assert row["phase4d_train_status"] == "fail"
    assert row["eval_allowed"] == 0
    assert "route_progress_fraction_max" in row["failed_checks"]
    assert row["route_progress_fraction_min_required"] == 0.25


def test_phase4d_train_acceptance_keeps_4096_progress_gate(tmp_path):
    train_dir = make_training_output(
        tmp_path,
        canary_updates={
            "route_progress_monotonic_fraction_max": 0.30448830588691284,
        })

    row = phase4d_train_acceptance_row(training_output_dir=str(train_dir))

    assert row["phase4d_train_status"] == "fail"
    assert row["eval_allowed"] == 0
    assert "route_progress_fraction_max" in row["failed_checks"]
    assert row["route_progress_fraction_min_required"] == 0.45


def test_phase4d_train_acceptance_blocks_failed_training_even_with_policy(tmp_path):
    train_dir = make_training_output(
        tmp_path,
        summary_updates={
            "status": "failed",
            "learn_error": "Actor could not be found in the registry",
        },
        canary_updates={
            "phase4_status": "fail",
            "failed_checks": "training_status",
        })

    row = phase4d_train_acceptance_row(training_output_dir=str(train_dir))

    assert row["phase4d_train_status"] == "fail"
    assert row["eval_allowed"] == 0
    assert row["invalid_for_eval"] == 1
    assert row["eval_status"] == "skipped_due_to_failed_training"
    assert "training_status" in row["failed_checks"]
    assert "learn_error" in row["failed_checks"]
    assert "phase4_failed_checks" in row["failed_checks"]
    assert row["policy_path_exists"] == 1


def test_phase4d_train_acceptance_blocks_partial_collection(tmp_path):
    train_dir = make_training_output(
        tmp_path,
        summary_updates={
            "total_timesteps": 8192,
            "rollout_rows": 6105,
        },
        canary_updates={
            "total_timesteps_requested": 8192,
            "total_timesteps_collected": 6105,
        })

    row = phase4d_train_acceptance_row(training_output_dir=str(train_dir))

    assert row["eval_allowed"] == 0
    assert "total_timesteps_collected" in row["failed_checks"]
    assert "rollout_rows" in row["failed_checks"]


def test_phase4d_train_acceptance_blocks_fake_backend(tmp_path):
    train_dir = make_training_output(
        tmp_path,
        summary_updates={
            "backend": "fake",
            "real_backend_used": 0,
            "fake_backend_used": 1,
        },
        canary_updates={
            "backend": "fake",
            "real_backend_used": 0,
            "fake_backend_used": 1,
        })

    row = phase4d_train_acceptance_row(training_output_dir=str(train_dir))

    assert row["eval_allowed"] == 0
    assert "backend" in row["failed_checks"]
    assert "real_backend_used" in row["failed_checks"]
    assert "fake_backend_used" in row["failed_checks"]


def test_phase4d_train_acceptance_blocks_missing_normalizer(tmp_path):
    train_dir = make_training_output(tmp_path)
    (train_dir / "normalizer.json").unlink()

    row = phase4d_train_acceptance_row(training_output_dir=str(train_dir))

    assert row["eval_allowed"] == 0
    assert "normalizer_path_exists" in row["failed_checks"]


def test_phase4d_train_acceptance_blocks_missing_route_progress_fields(tmp_path):
    train_dir = make_training_output(
        tmp_path,
        canary_updates={
            "route_progress_available_ratio": "",
            "route_progress_monotonic_fraction_max": "",
        })

    row = phase4d_train_acceptance_row(training_output_dir=str(train_dir))

    assert row["eval_allowed"] == 0
    assert "route_progress_available_ratio" in row["failed_checks"]
    assert "route_progress_fraction_max" in row["failed_checks"]


def test_phase4d_train_acceptance_writes_csv_and_json(tmp_path):
    train_dir = make_training_output(tmp_path)
    output_dir = tmp_path / "comparison"

    csv_path, json_path, row = write_phase4d_train_acceptance(
        str(output_dir),
        training_output_dir=str(train_dir))

    assert row["eval_allowed"] == 1
    assert csv_path.endswith("phase4d_train_acceptance.csv")
    assert json_path.endswith("phase4d_train_acceptance.json")
    assert (output_dir / "phase4d_train_acceptance.csv").is_file()
    assert (output_dir / "phase4d_train_acceptance.json").is_file()

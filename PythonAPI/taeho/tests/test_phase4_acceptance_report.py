import csv
import json

from suspension_control.rl import phase4_acceptance
from suspension_control.rl.policy_export import write_normalizer_metadata


def test_phase4_training_acceptance_rejects_fake_smoke(tmp_path):
    output_dir = tmp_path / "fake"
    output_dir.mkdir()
    _write_json(output_dir / "training_summary.json", {
        "status": "trained",
        "backend": "fake",
        "total_timesteps": 32,
        "rollout_rows": 2,
        "fake_backend_used": 1,
        "real_backend_used": 0,
        "policy_exported": 0,
        "training_rollout_csv": str(output_dir / "training_rollout.csv"),
        "training_rollout_jsonl": str(output_dir / "training_rollout.jsonl"),
    })
    _write_json(output_dir / "training_config.json", {
        "backend": "fake",
        "warmup_seconds": 3.0,
        "fixed_delta_seconds": 0.05,
    })
    _write_rollout(output_dir / "training_rollout.csv", [
        {"episode_id": "fake", "reward_total": "1.0", "fake_backend_used": "1"},
        {"episode_id": "fake", "reward_total": "1.1", "fake_backend_used": "1"},
    ])
    (output_dir / "training_rollout.jsonl").write_text("{}\n")

    _, _, row = phase4_acceptance.write_phase4_training_canary(str(output_dir))

    assert row["phase4_status"] == "fail"
    assert "backend" in row["failed_checks"]
    assert "fake_backend_smoke_cannot_pass_real_canary" in row["warnings"]


def test_phase4_training_acceptance_passes_minimal_live_contract(tmp_path):
    output_dir = tmp_path / "live"
    output_dir.mkdir()
    normalizer = write_normalizer_metadata(
        str(output_dir / "normalizer.json"),
        feature_names=("speed", "roll"),
        clip=5.0,
        action_semantics="normalized_damper_residual_v1",
        max_damper_residual_scale=0.08)
    policy_path = output_dir / "policy.ts"
    policy_path.write_text("placeholder")
    model_path = output_dir / "sb3_model.zip"
    model_path.write_text("placeholder")
    rollout_path = output_dir / "training_rollout.csv"
    _write_rollout(rollout_path, [
        _live_rollout_row(0, reward="1.0"),
        _live_rollout_row(1, reward="1.1", truncated="1"),
    ])
    (output_dir / "training_rollout.jsonl").write_text("{}\n{}\n")
    _write_json(output_dir / "training_config.json", {
        "backend": "live",
        "warmup_seconds": 3.0,
        "fixed_delta_seconds": 0.05,
    })
    _write_json(output_dir / "training_summary.json", {
        "status": "trained",
        "backend": "live",
        "total_timesteps": 2,
        "rollout_rows": 2,
        "observation_dim": 2,
        "training_rollout_csv": str(rollout_path),
        "training_rollout_jsonl": str(output_dir / "training_rollout.jsonl"),
        "sb3_model_path": str(model_path),
        "policy_path": str(policy_path),
        "normalizer_path": normalizer,
        "policy_exported": 1,
        "real_backend_used": 1,
        "fake_backend_used": 0,
    })

    original_policy_check = phase4_acceptance._policy_adapter_load_ok
    try:
        phase4_acceptance._policy_adapter_load_ok = lambda path, observation_dim: 1
        csv_path, json_path, row = phase4_acceptance.write_phase4_training_canary(
            str(output_dir))
    finally:
        phase4_acceptance._policy_adapter_load_ok = original_policy_check

    assert row["phase4_status"] == "pass"
    assert row["failed_checks"] == ""
    with open(csv_path) as csv_file:
        rows = list(csv.DictReader(csv_file))
    with open(json_path) as json_file:
        data = json.load(json_file)
    assert rows[0]["phase4_status"] == "pass"
    assert data["phase4_status"] == "pass"
    assert data["policy_saved"] == 1
    assert data["normalizer_load_ok"] == 1


def _live_rollout_row(index, reward="1.0", truncated="0"):
    return {
        "episode_id": "live_seed100",
        "step": str(index),
        "backend": "live",
        "real_backend_used": "1",
        "fake_backend_used": "0",
        "carla_connected": "1",
        "route_process_started": "1",
        "hero_attached": "1",
        "reward_total": reward,
        "reward_comfort": "0.1",
        "reward_stability": "0.2",
        "reward_task": "0.3",
        "reward_action": "0.0",
        "reward_safety": "0.0",
        "action_fl": "0.1",
        "action_fr": "0.1",
        "action_rl": "0.1",
        "action_rr": "0.1",
        "rl_residual_damper_fl": "0.01",
        "rl_residual_damper_fr": "0.01",
        "rl_residual_damper_rl": "0.01",
        "rl_residual_damper_rr": "0.01",
        "rl_safety_gate_active": "0",
        "route_progress_available": "1",
        "route_progress_monotonic_fraction": str(0.1 + 0.1 * index),
        "terminated": "0",
        "truncated": truncated,
    }


def _write_rollout(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, data):
    with open(path, "w") as json_file:
        json.dump(data, json_file)


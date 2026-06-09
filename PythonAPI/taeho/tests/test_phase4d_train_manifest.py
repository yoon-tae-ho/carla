import csv
import hashlib
import json
import subprocess
import sys

from suspension_control.rl.phase4d_train_acceptance import (
    write_phase4d_train_acceptance,
)
from suspension_control.rl.phase4d_train_manifest import (
    write_phase4d_train_manifest,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_train_dir(tmp_path, *, acceptance_updates=None):
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    policy_path = train_dir / "policy.ts"
    normalizer_path = train_dir / "normalizer.json"
    sb3_path = train_dir / "sb3_model.zip"
    policy_path.write_bytes(b"policy-bytes")
    normalizer_path.write_text(json.dumps({
        "action_semantics": "normalized_damper_residual_v1",
        "feature_names": ["speed", "roll"],
    }) + "\n")
    sb3_path.write_bytes(b"sb3-bytes")
    write_json(train_dir / "training_summary.json", {
        "status": "trained",
        "backend": "live",
        "algorithm": "sac",
        "baseline": "skyhook",
        "planning_provider": "empty",
        "routes_subset": "00",
        "seed": 100,
        "total_timesteps": 4096,
        "rollout_rows": 4096,
        "observation_dim": 2,
        "action_dim": 4,
        "action_semantics": "normalized_damper_residual_v1",
        "policy_path": str(policy_path),
        "normalizer_path": str(normalizer_path),
        "sb3_model_path": str(sb3_path),
        "real_backend_used": 1,
        "fake_backend_used": 0,
        "synthetic_training": 0,
        "policy_exported": 1,
        "normalizer_saved": 1,
        "learn_error": "",
        "export_error": "",
        "collision_count": 0,
        "lane_invasion_count": 0,
        "red_light_count": 0,
        "route_timeout_count": 0,
        "blocked_vehicle_count": 0,
    })
    write_json(train_dir / "phase4_training_canary.json", {
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
    })
    write_json(train_dir / "training_config.json", {
        "backend": "live",
        "algorithm": "sac",
        "baseline": "skyhook",
        "planning_provider": "empty",
        "routes_subset": "00",
        "seed": 100,
        "total_timesteps": 4096,
        "observation_feature_names": ["speed", "roll"],
        "action_semantics": "normalized_damper_residual_v1",
    })
    output_dir = tmp_path / "out"
    _, acceptance_json, acceptance = write_phase4d_train_acceptance(
        str(output_dir),
        training_output_dir=str(train_dir))
    if acceptance_updates:
        acceptance.update(acceptance_updates)
        write_json(output_dir / "phase4d_train_acceptance.json", acceptance)
    return train_dir, output_dir, acceptance_json


def test_phase4d_train_manifest_writes_lineage_and_checksums(tmp_path):
    train_dir, output_dir, acceptance_json = make_train_dir(tmp_path)

    json_path, csv_path, manifest = write_phase4d_train_manifest(
        str(output_dir),
        training_output_dir=str(train_dir),
        train_acceptance_path=acceptance_json,
        docker_image_id="image-id",
        docker_image_tag="runner:test",
        carla_build_commit="cb77f0c")

    assert manifest["workflow_role"] == "train_only"
    assert manifest["artifact_status"] == "accepted_for_eval"
    assert manifest["phase"] == "phase4d"
    assert manifest["backend"] == "live"
    assert manifest["algorithm"] == "sac"
    assert manifest["baseline"] == "skyhook"
    assert manifest["planning_provider"] == "empty"
    assert manifest["routes_subset"] == "00"
    assert manifest["train_seed"] == 100
    assert manifest["total_timesteps"] == 4096
    assert manifest["action_semantics"] == "normalized_damper_residual_v1"
    assert manifest["observation_dim"] == 2
    assert manifest["action_dim"] == 4
    assert manifest["policy_sha256"] == _sha256(train_dir / "policy.ts")
    assert manifest["normalizer_sha256"] == _sha256(train_dir / "normalizer.json")
    assert manifest["sb3_model_sha256"] == _sha256(train_dir / "sb3_model.zip")
    assert manifest["docker_image_id"] == "image-id"
    assert manifest["docker_image_tag"] == "runner:test"
    assert manifest["carla_build_commit"] == "cb77f0c"
    assert manifest["eval_allowed"] == 1
    assert manifest["invalid_for_eval"] == 0
    assert json_path.endswith("phase4d_train_manifest.json")
    assert csv_path.endswith("phase4d_train_manifest.csv")

    with open(json_path) as json_file:
        loaded = json.load(json_file)
    assert loaded["artifact_id"] == manifest["artifact_id"]
    with open(csv_path, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert rows[0]["artifact_status"] == "accepted_for_eval"


def test_phase4d_train_manifest_marks_rejected_acceptance_invalid(tmp_path):
    train_dir, output_dir, acceptance_json = make_train_dir(
        tmp_path,
        acceptance_updates={
            "phase4d_train_status": "fail",
            "eval_allowed": 0,
            "invalid_for_eval": 1,
            "failed_checks": "rollout_rows",
        })

    _, _, manifest = write_phase4d_train_manifest(
        str(output_dir),
        training_output_dir=str(train_dir),
        train_acceptance_path=acceptance_json)

    assert manifest["artifact_status"] == "invalid_for_eval"
    assert manifest["eval_allowed"] == 0
    assert manifest["invalid_for_eval"] == 1
    assert manifest["failed_checks"] == "rollout_rows"


def test_phase4d_train_manifest_cli_writes_files(tmp_path):
    train_dir, output_dir, acceptance_json = make_train_dir(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "suspension_control.rl.phase4d_train_manifest",
            "--training-output-dir",
            str(train_dir),
            "--train-acceptance",
            acceptance_json,
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert "phase4d train manifest:" in result.stdout
    assert (output_dir / "phase4d_train_manifest.json").is_file()
    assert (output_dir / "phase4d_train_manifest.csv").is_file()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

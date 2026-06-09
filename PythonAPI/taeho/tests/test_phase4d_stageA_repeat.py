import hashlib
import json
import subprocess
import sys

from suspension_control.rl.phase4d_stageA_repeat import (
    write_phase4d_stageA_repeat_plan,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def make_train_manifest(tmp_path, *, updates=None, policy_bytes=b"policy", normalizer_bytes=None):
    root = tmp_path / "phase4d_train"
    train_dir = root / "train"
    train_dir.mkdir(parents=True)
    policy_path = train_dir / "policy.ts"
    normalizer_path = train_dir / "normalizer.json"
    policy_path.write_bytes(policy_bytes)
    normalizer_payload = {
        "action_semantics": "normalized_damper_residual_v1",
        "feature_names": ["speed", "roll"],
    }
    if normalizer_bytes is None:
        normalizer_path.write_text(json.dumps(normalizer_payload) + "\n")
    else:
        normalizer_path.write_bytes(normalizer_bytes)
    manifest = {
        "artifact_id": "phase4d-train-test-artifact",
        "workflow_role": "train_only",
        "artifact_status": "accepted_for_eval",
        "eval_allowed": 1,
        "invalid_for_eval": 0,
        "baseline": "skyhook",
        "planning_provider": "empty",
        "routes_subset": "00",
        "train_seed": 100,
        "total_timesteps": 4096,
        "action_semantics": "normalized_damper_residual_v1",
        "train_dir": str(train_dir),
        "policy_path": str(policy_path),
        "normalizer_path": str(normalizer_path),
        "policy_sha256": _sha256(policy_path),
        "normalizer_sha256": _sha256(normalizer_path),
    }
    manifest.update(updates or {})
    manifest_path = root / "phase4d_train_manifest.json"
    write_json(manifest_path, manifest)
    return root, train_dir, manifest_path


def test_stageA_plan_accepts_fixed_artifact_and_writes_seed_plan(tmp_path):
    _root, train_dir, manifest_path = make_train_manifest(tmp_path)
    output_dir = tmp_path / "stageA"

    plan_path, acceptance_path, manifest_out_path, result = write_phase4d_stageA_repeat_plan(
        output_dir=str(output_dir),
        train_manifest_path=str(manifest_path),
        train_dir=str(train_dir),
        eval_seeds=["100", "101", "102"],
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty",
        dry_run=True,
    )

    acceptance = result["acceptance"]
    plan = result["plan"]
    manifest = result["manifest"]
    assert acceptance["stageA_status"] == "ready_for_execution"
    assert acceptance["preflight_ok"] == 1
    assert acceptance["failed_checks"] == ""
    assert acceptance["eval_seeds"] == "100,101,102"
    assert plan["dry_run"] == 1
    assert [row["eval_seed"] for row in plan["seed_plans"]] == ["100", "101", "102"]
    assert plan["seed_plans"][1]["s4_output_dir"].endswith("eval_seed_101/s4")
    assert manifest["workflow_role"] == "stageA_same_artifact_eval_repeat"
    assert manifest["source_train_artifact_id"] == "phase4d-train-test-artifact"
    assert manifest["artifact_status"] == "ready_for_execution"
    assert plan_path.endswith("phase4d_stageA_repeat_plan.json")
    assert acceptance_path.endswith("phase4d_stageA_repeat_acceptance.json")
    assert manifest_out_path.endswith("phase4d_stageA_repeat_manifest.json")

    loaded_text = (output_dir / "phase4d_stageA_repeat_plan.json").read_text()
    assert "transfuser_suspension_control_suite.py" not in loaded_text


def test_stageA_plan_rejects_invalid_train_artifact_before_route_plan(tmp_path):
    _root, train_dir, manifest_path = make_train_manifest(
        tmp_path,
        updates={
            "artifact_status": "invalid_for_eval",
            "eval_allowed": 0,
            "invalid_for_eval": 1,
        })

    _plan_path, acceptance_path, _manifest_out_path, result = write_phase4d_stageA_repeat_plan(
        output_dir=str(tmp_path / "stageA"),
        train_manifest_path=str(manifest_path),
        train_dir=str(train_dir),
        eval_seeds=["100"],
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty",
        dry_run=True,
    )

    acceptance = result["acceptance"]
    assert acceptance["stageA_status"] == "preflight_failed"
    assert acceptance["preflight_ok"] == 0
    assert "artifact_status" in acceptance["failed_checks"]
    assert "eval_allowed" in acceptance["failed_checks"]
    assert "invalid_for_eval" in acceptance["failed_checks"]

    loaded = json.loads(open(acceptance_path).read())
    assert loaded["stageA_status"] == "preflight_failed"


def test_stageA_plan_rejects_checksum_mismatch(tmp_path):
    _root, train_dir, manifest_path = make_train_manifest(
        tmp_path,
        updates={"policy_sha256": "definitely-wrong"})

    _plan_path, _acceptance_path, _manifest_out_path, result = write_phase4d_stageA_repeat_plan(
        output_dir=str(tmp_path / "stageA"),
        train_manifest_path=str(manifest_path),
        train_dir=str(train_dir),
        eval_seeds=["100"],
        routes_subset="00",
        baseline="skyhook",
        planning_provider="empty",
        dry_run=True,
    )

    assert result["acceptance"]["stageA_status"] == "preflight_failed"
    assert "policy_sha256" in result["acceptance"]["failed_checks"]


def test_stageA_plan_cli_dry_run_writes_plan_without_route_command(tmp_path):
    _root, train_dir, manifest_path = make_train_manifest(tmp_path)
    output_dir = tmp_path / "stageA_cli"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "suspension_control.rl.phase4d_stageA_repeat",
            "--action",
            "plan",
            "--train-manifest",
            str(manifest_path),
            "--train-dir",
            str(train_dir),
            "--eval-seeds",
            "100,101",
            "--routes-subset",
            "00",
            "--baseline",
            "skyhook",
            "--planning-provider",
            "empty",
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert "phase4d Stage A repeat plan:" in result.stdout
    assert "transfuser_suspension_control_suite.py" not in result.stdout
    assert (output_dir / "phase4d_stageA_repeat_plan.json").is_file()
    assert (output_dir / "phase4d_stageA_repeat_acceptance.json").is_file()
    assert (output_dir / "phase4d_stageA_repeat_manifest.json").is_file()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

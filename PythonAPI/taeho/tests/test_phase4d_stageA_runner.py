import json
import subprocess
from pathlib import Path

from tests.test_phase4d_stageA_repeat import make_train_manifest


TAEHO_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path(__file__).resolve().parents[4]


def test_stageA_host_wrapper_help_does_not_start_docker():
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_stageA_same_artifact_eval_repeat.sh"),
        "--help",
    ], cwd=SIM_ROOT / "docker")

    assert "Run Phase 4-D Stage A same-artifact eval repeat" in result.stdout
    assert "docker compose" not in result.stdout
    assert result.stderr == ""


def test_stageA_host_wrapper_translates_workspace_path():
    host_path = SIM_ROOT / "e2e_models" / "outputs" / "phase4d_stageA"
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_stageA_same_artifact_eval_repeat.sh"),
        "--translate-path",
        str(host_path),
    ], cwd=SIM_ROOT / "docker")

    assert result.stdout.strip() == "/workspace/e2e_models/outputs/phase4d_stageA"


def test_stageA_host_wrapper_print_command_uses_translated_paths():
    train_root = SIM_ROOT / "e2e_models" / "outputs" / "phase4d_train"
    output_root = SIM_ROOT / "e2e_models" / "outputs" / "phase4d_stageA"

    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_stageA_same_artifact_eval_repeat.sh"),
        "--print-command",
        "--train-manifest",
        str(train_root / "phase4d_train_manifest.json"),
        "--train-dir",
        str(train_root / "train"),
        "--eval-seeds",
        "100,101",
        "--routes-subset",
        "00",
        "--output-dir",
        str(output_root),
    ], cwd=SIM_ROOT / "docker")
    stdout = result.stdout

    assert "docker compose --env-file .env -f compose.dev.yaml run --rm runner" in stdout
    assert "bash scripts/run_phase4d_stageA_same_artifact_eval_repeat.sh" in stdout
    assert "--print-command" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_train/phase4d_train_manifest.json" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_train/train" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_stageA" in stdout
    assert str(SIM_ROOT) not in stdout
    assert result.stderr == ""


def test_stageA_in_container_print_command_has_seed_eval_and_compare_commands(tmp_path):
    train_root = tmp_path / "phase4d_train"
    train_dir = train_root / "train"
    output_dir = tmp_path / "stageA"

    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4d_stageA_same_artifact_eval_repeat.sh"),
        "--print-command",
        "--train-manifest",
        str(train_root / "phase4d_train_manifest.json"),
        "--train-dir",
        str(train_dir),
        "--eval-seeds",
        "100,101",
        "--routes-subset",
        "00",
        "--output-dir",
        str(output_dir),
    ], cwd=TAEHO_ROOT)
    stdout = result.stdout

    assert "STAGEA_PLAN:" in stdout
    assert "phase4d_stageA_repeat" in stdout
    assert "STAGEA_S4_SEED_100:" in stdout
    assert "run_phase4d_eval_s4_only.sh" in stdout
    assert "STAGEA_S8_SEED_101:" in stdout
    assert "run_phase4d_eval_s8_reference.sh" in stdout
    assert "STAGEA_COMPARE_SEED_100:" in stdout
    assert "phase4d_compare_only" in stdout
    assert "eval_seed_101/s4" in stdout
    assert "eval_seed_101/s8" in stdout
    assert "eval_seed_101/compare" in stdout

    assert "train_real_carla_sac" not in stdout
    assert "train_sac" not in stdout


def test_stageA_in_container_dry_run_writes_plan_without_carla(tmp_path):
    _root, train_dir, manifest_path = make_train_manifest(tmp_path)
    output_dir = tmp_path / "stageA"

    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4d_stageA_same_artifact_eval_repeat.sh"),
        "--dry-run",
        "--train-manifest",
        str(manifest_path),
        "--train-dir",
        str(train_dir),
        "--eval-seeds",
        "100,101",
        "--routes-subset",
        "00",
        "--output-dir",
        str(output_dir),
    ], cwd=TAEHO_ROOT)

    assert "Phase 4-D Stage A dry-run complete." in result.stdout
    assert "transfuser_suspension_control_suite.py" not in result.stdout
    assert (output_dir / "phase4d_stageA_repeat_plan.json").is_file()
    assert (output_dir / "phase4d_stageA_repeat_acceptance.json").is_file()
    assert (output_dir / "phase4d_stageA_repeat_manifest.json").is_file()

    plan = json.loads((output_dir / "phase4d_stageA_repeat_plan.json").read_text())
    assert plan["dry_run"] == 1
    assert plan["seed_plans"][0]["s4_output_dir"].endswith("eval_seed_100/s4")
    assert plan["seed_plans"][1]["s8_output_dir"].endswith("eval_seed_101/s8")
    assert plan["seed_plans"][1]["compare_output_dir"].endswith("eval_seed_101/compare")


def _run(args, cwd):
    return subprocess.run(
        args,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

import subprocess
from pathlib import Path


TAEHO_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path(__file__).resolve().parents[4]


def test_phase4d_host_compare_only_help_does_not_start_docker():
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_compare_only.sh"),
        "--help",
    ], cwd=SIM_ROOT / "docker")

    assert "Run Phase 4-D compare-only artifact selection" in result.stdout
    assert "docker compose" not in result.stdout
    assert result.stderr == ""


def test_phase4d_host_compare_only_wrapper_translates_workspace_path():
    host_path = SIM_ROOT / "e2e_models" / "outputs" / "phase4d_compare"
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_compare_only.sh"),
        "--translate-path",
        str(host_path),
    ], cwd=SIM_ROOT / "docker")

    assert result.stdout.strip() == "/workspace/e2e_models/outputs/phase4d_compare"


def test_phase4d_host_compare_only_print_command_uses_module_and_translated_paths():
    train_root = SIM_ROOT / "e2e_models" / "outputs" / "phase4d_train"
    s4_root = SIM_ROOT / "e2e_models" / "outputs" / "phase4d_eval_s4"
    s8_root = SIM_ROOT / "e2e_models" / "outputs" / "phase4d_eval_s8"
    compare_root = SIM_ROOT / "e2e_models" / "outputs" / "phase4d_compare"

    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_compare_only.sh"),
        "--print-command",
        "--train-manifest",
        str(train_root / "phase4d_train_manifest.json"),
        "--s4-summary",
        str(s4_root / "suite_summary.csv"),
        "--s4-acceptance",
        str(s4_root / "phase4d_eval_s4_acceptance.json"),
        "--s4-manifest",
        str(s4_root / "phase4d_eval_s4_manifest.json"),
        "--s8-summary",
        str(s8_root / "suite_summary.csv"),
        "--s8-acceptance",
        str(s8_root / "phase4d_eval_s8_acceptance.json"),
        "--s8-manifest",
        str(s8_root / "phase4d_eval_s8_manifest.json"),
        "--output-dir",
        str(compare_root),
    ], cwd=SIM_ROOT / "docker")
    stdout = result.stdout

    assert "docker compose --env-file .env -f compose.dev.yaml run --rm runner" in stdout
    assert "source /workspace/e2e_models/scripts/env_garage_2.sh" in stdout
    assert "python3 -m suspension_control.rl.phase4d_compare_only" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_train/phase4d_train_manifest.json" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_eval_s4/suite_summary.csv" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_eval_s8/suite_summary.csv" in stdout
    assert "/workspace/e2e_models/outputs/phase4d_compare" in stdout
    assert str(SIM_ROOT) not in stdout
    assert result.stderr == ""


def _run(args, cwd):
    return subprocess.run(
        args,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

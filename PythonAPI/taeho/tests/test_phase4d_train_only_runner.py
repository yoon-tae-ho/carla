import subprocess
from pathlib import Path


TAEHO_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path(__file__).resolve().parents[4]


def test_phase4d_host_train_only_help_does_not_start_docker():
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_train_only.sh"),
        "--help",
    ], cwd=SIM_ROOT / "docker")

    assert "Run Phase 4-D train-only live SAC" in result.stdout
    assert "docker compose" not in result.stdout
    assert result.stderr == ""


def test_phase4d_host_train_only_wrapper_translates_workspace_path(tmp_path):
    host_path = SIM_ROOT / "e2e_models" / "outputs" / "phase4d_train_example"
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_train_only.sh"),
        "--translate-path",
        str(host_path),
    ], cwd=SIM_ROOT / "docker")

    assert result.stdout.strip() == "/workspace/e2e_models/outputs/phase4d_train_example"


def test_phase4d_train_only_print_command_forces_train_only_flags(tmp_path):
    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4d_train_only.sh"),
        "--print-command",
        "--seed",
        "100",
        "--routes-subset",
        "00",
        "--output-root",
        str(tmp_path / "phase4d_train"),
    ], cwd=TAEHO_ROOT)
    stdout = result.stdout

    assert "TRAIN:" in stdout
    assert "--backend live" in stdout
    assert "--algorithm sac" in stdout
    assert "--baseline skyhook" in stdout
    assert "--total-timesteps 1024" in stdout
    assert "--max-episode-steps 0" in stdout
    assert "--verify-every 0" in stdout
    assert "--eval-after-training false" in stdout
    assert "--planning-provider empty" in stdout
    assert "--timeout 120" in stdout
    assert "--hero-timeout-seconds 900" in stdout
    assert "--route-wait-timeout-seconds 120" in stdout
    assert "--connect-retry-seconds 1.0" in stdout
    assert "--output-dir" in stdout
    assert "/train" in stdout

    assert "TRAIN_ACCEPTANCE:" in stdout
    assert "phase4d_train_acceptance" in stdout
    assert "--training-return-code" in stdout

    assert "TRAIN_MANIFEST:" in stdout
    assert "phase4d_train_manifest" in stdout
    assert "--train-acceptance" in stdout

    assert "EVAL_S4:" not in stdout
    assert "transfuser_suspension_control_suite.py" not in stdout
    assert "--reference-summary" not in stdout
    assert "--scenarios rl_residual_skyhook" not in stdout


def test_phase4d_train_only_warns_for_explicit_horizon_above_route00_main_path(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(TAEHO_ROOT / "scripts" / "run_phase4d_train_only.sh"),
            "--print-command",
            "--total-timesteps",
            "8192",
            "--output-root",
            str(tmp_path / "phase4d_train"),
        ],
        cwd=str(TAEHO_ROOT),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert "--total-timesteps 8192" in result.stdout
    assert "WARNING: Phase 4-D route00 training above 4096 steps" in result.stderr


def _run(args, cwd):
    return subprocess.run(
        args,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

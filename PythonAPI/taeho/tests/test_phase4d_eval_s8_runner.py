import subprocess
from pathlib import Path


TAEHO_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path(__file__).resolve().parents[4]


def test_phase4d_host_eval_s8_help_does_not_start_docker():
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_eval_s8_reference.sh"),
        "--help",
    ], cwd=SIM_ROOT / "docker")

    assert "Run Phase 4-D S8 zero-residual reference eval" in result.stdout
    assert "docker compose" not in result.stdout
    assert result.stderr == ""


def test_phase4d_host_eval_s8_wrapper_translates_workspace_path():
    host_path = SIM_ROOT / "e2e_models" / "outputs" / "phase4d_s8_reference"
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_eval_s8_reference.sh"),
        "--translate-path",
        str(host_path),
    ], cwd=SIM_ROOT / "docker")

    assert result.stdout.strip() == "/workspace/e2e_models/outputs/phase4d_s8_reference"


def test_phase4d_eval_s8_print_command_only_uses_zero_residual(tmp_path):
    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4d_eval_s8_reference.sh"),
        "--print-command",
        "--seed",
        "100",
        "--routes-subset",
        "00",
        "--output-dir",
        str(tmp_path / "s8_reference"),
    ], cwd=TAEHO_ROOT)
    stdout = result.stdout

    assert "REFERENCE_S8:" in stdout
    assert "--scenarios rl_zero_residual_skyhook" in stdout
    assert "--metric-warmup-seconds 3.0" in stdout
    assert "--timeout 120" in stdout
    assert "--tick-wait-timeout 120" in stdout
    assert "--sidecar-tick-mode poll" in stdout
    assert "--poll-seconds 0.05" in stdout
    assert "--verify-every 50" in stdout
    assert "--require-verification" in stdout
    assert "--planning-provider empty" in stdout
    assert "--routes-subset 00" in stdout
    assert "--seeds 100" in stdout
    assert "ACCEPTANCE:" in stdout
    assert "phase4d_eval_reference" in stdout
    assert "--fail-on-reject" in stdout

    assert "train_real_carla_sac" not in stdout
    assert "--rl-policy" not in stdout
    assert "--rl-normalizer" not in stdout
    assert "rl_residual_skyhook" not in stdout


def test_phase4d_legacy_s8_reference_wrapper_uses_official_runner(tmp_path):
    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4d_s8_reference_only.sh"),
        "--print-command",
        "--output-dir",
        str(tmp_path / "s8_reference"),
    ], cwd=TAEHO_ROOT)

    assert "REFERENCE_S8:" in result.stdout
    assert "ACCEPTANCE:" in result.stdout
    assert "phase4d_eval_reference" in result.stdout


def _run(args, cwd):
    return subprocess.run(
        args,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

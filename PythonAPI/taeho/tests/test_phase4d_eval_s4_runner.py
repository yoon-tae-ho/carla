import subprocess
from pathlib import Path


TAEHO_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path(__file__).resolve().parents[4]


def test_phase4d_host_eval_s4_help_does_not_start_docker():
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_eval_s4_only.sh"),
        "--help",
    ], cwd=SIM_ROOT / "docker")

    assert "Run Phase 4-D S4-only artifact eval" in result.stdout
    assert "docker compose" not in result.stdout
    assert result.stderr == ""


def test_phase4d_host_eval_s4_wrapper_translates_workspace_path():
    host_path = SIM_ROOT / "e2e_models" / "outputs" / "phase4d_train" / "train"
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_eval_s4_only.sh"),
        "--translate-path",
        str(host_path),
    ], cwd=SIM_ROOT / "docker")

    assert result.stdout.strip() == "/workspace/e2e_models/outputs/phase4d_train/train"


def test_phase4d_eval_s4_print_command_is_s4_only_and_has_preflight(tmp_path):
    train_dir = tmp_path / "phase4d_train" / "train"
    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4d_eval_s4_only.sh"),
        "--print-command",
        "--train-dir",
        str(train_dir),
        "--routes-subset",
        "00",
        "--seed",
        "100",
        "--output-dir",
        str(tmp_path / "eval_s4"),
    ], cwd=TAEHO_ROOT)
    stdout = result.stdout

    assert "PREFLIGHT:" in stdout
    assert "phase4d_eval_artifact" in stdout
    assert "--action prepare" in stdout
    assert "EVAL_S4:" in stdout
    assert "--scenarios rl_residual_skyhook" in stdout
    assert "--rl-policy" in stdout
    assert "--rl-normalizer" in stdout
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
    assert "--action accept" in stdout
    assert "--fail-on-reject" in stdout

    assert "train_real_carla_sac" not in stdout
    assert "rl_zero_residual_skyhook" not in stdout
    assert "--scenarios rl_zero_residual_skyhook" not in stdout


def test_phase4d_eval_s4_requires_train_dir(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(TAEHO_ROOT / "scripts" / "run_phase4d_eval_s4_only.sh"),
            "--print-command",
            "--output-dir",
            str(tmp_path / "eval_s4"),
        ],
        cwd=str(TAEHO_ROOT),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert result.returncode == 2
    assert "requires --train-dir" in result.stderr


def _run(args, cwd):
    return subprocess.run(
        args,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

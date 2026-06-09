import subprocess
from pathlib import Path


TAEHO_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path(__file__).resolve().parents[4]


def test_phase4c_host_wrapper_help_does_not_start_docker():
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4c_training_horizon_sweep.sh"),
        "--help",
    ], cwd=SIM_ROOT / "docker")

    assert "Run Phase 4-C live-training horizon sweep" in result.stdout
    assert "4096-step training and exported-policy eval are opt-in only" in result.stdout


def test_phase4c_host_wrapper_translates_workspace_path(tmp_path):
    host_path = SIM_ROOT / "e2e_models" / "outputs" / "phase4c_example"
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4c_training_horizon_sweep.sh"),
        "--translate-path",
        str(host_path),
    ], cwd=SIM_ROOT / "docker")

    assert result.stdout.strip() == "/workspace/e2e_models/outputs/phase4c_example"


def test_phase4c_runner_default_excludes_4096_and_forces_stable_flags(tmp_path):
    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4c_training_horizon_sweep.sh"),
        "--print-command",
        "--output-root",
        str(tmp_path),
    ], cwd=TAEHO_ROOT)
    stdout = result.stdout

    assert "--backend live" in stdout
    assert "--algorithm sac" in stdout
    assert "--max-episode-steps 0" in stdout
    assert "--verify-every 0" in stdout
    assert "--eval-after-training false" in stdout
    assert "--planning-provider empty" in stdout
    assert "--timeout 120" in stdout
    assert "--hero-timeout-seconds 900" in stdout
    assert "--route-wait-timeout-seconds 120" in stdout
    assert "--connect-retry-seconds 1.0" in stdout
    assert "--total-timesteps 512" in stdout
    assert "--total-timesteps 1024" in stdout
    assert "--total-timesteps 2048" in stdout
    assert "--total-timesteps 4096" not in stdout
    assert "4096steps" not in stdout


def test_phase4c_runner_include_4096_adds_horizon(tmp_path):
    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4c_training_horizon_sweep.sh"),
        "--print-command",
        "--include-4096",
        "--output-root",
        str(tmp_path),
    ], cwd=TAEHO_ROOT)

    assert "--total-timesteps 4096" in result.stdout
    assert "4096steps" in result.stdout


def test_phase4_real_carla_canary_help_uses_stable_training_defaults():
    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4_real_carla_sac_canary.sh"),
        "--help",
    ], cwd=TAEHO_ROOT)
    stdout = result.stdout

    assert "max_episode_steps=0" in stdout
    assert "verify_every=0" in stdout
    assert "eval_after_training=false" in stdout
    assert "max_episode_steps=256" not in stdout
    assert "verify_every=50" not in stdout
    assert "eval_after_training=true" not in stdout


def _run(args, cwd):
    return subprocess.run(
        args,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

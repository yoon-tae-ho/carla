import subprocess
from pathlib import Path


def test_phase4_host_runner_help_does_not_start_docker():
    sim_root = Path(__file__).resolve().parents[4]
    script = sim_root / "docker" / "run_phase4_real_carla_sac_canary.sh"

    result = subprocess.run(
        ["bash", str(script), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False)

    assert result.returncode == 0
    assert "Phase 4 real-CARLA SAC canary" in result.stdout
    assert "run_phase4_real_carla_sac_canary.sh" in result.stdout


def test_phase4_in_container_runner_help():
    sim_root = Path(__file__).resolve().parents[4]
    script = (
        sim_root /
        "carla-0.9.15" /
        "PythonAPI" /
        "taeho" /
        "scripts" /
        "run_phase4_real_carla_sac_canary.sh")

    result = subprocess.run(
        ["bash", str(script), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False)

    assert result.returncode == 0
    assert "Phase 4 real-CARLA SAC canary" in result.stdout

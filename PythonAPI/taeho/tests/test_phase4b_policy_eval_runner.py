import subprocess
from pathlib import Path


def test_phase4b_host_runner_help_does_not_start_docker():
    sim_root = Path(__file__).resolve().parents[4]
    script = sim_root / "docker" / "run_phase4b_exported_policy_eval.sh"

    result = subprocess.run(
        ["bash", str(script), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False)

    assert result.returncode == 0
    assert "Phase 4-B exported policy" in result.stdout
    assert "run_phase4b_exported_policy_eval.sh" in result.stdout
    assert result.stderr == ""


def test_phase4b_host_runner_translates_host_paths():
    sim_root = Path(__file__).resolve().parents[4]
    script = sim_root / "docker" / "run_phase4b_exported_policy_eval.sh"

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--translate-path",
            str(sim_root / "e2e_models" / "outputs" / "example"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False)

    assert result.returncode == 0
    assert result.stdout.strip() == "/workspace/e2e_models/outputs/example"
    assert result.stderr == ""


def test_phase4b_in_container_runner_print_command_contains_required_flags(tmp_path):
    sim_root = Path(__file__).resolve().parents[4]
    script = (
        sim_root /
        "carla-0.9.15" /
        "PythonAPI" /
        "taeho" /
        "scripts" /
        "run_phase4b_exported_policy_eval.sh")
    policy_dir = tmp_path / "policy"
    policy_dir.mkdir()
    (policy_dir / "policy.ts").write_text("placeholder")
    (policy_dir / "normalizer.json").write_text("{}")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-command",
            "--policy-dir",
            str(policy_dir),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False)

    assert result.returncode == 0
    assert "--scenarios rl_zero_residual_skyhook\\,rl_residual_skyhook" in result.stdout
    assert "--verify-every 50" in result.stdout
    assert "--require-verification" in result.stdout
    assert "--planning-provider empty" in result.stdout
    assert "--rl-policy" in result.stdout
    assert "--rl-normalizer" in result.stdout
    assert "--phase4b-output-dir" in result.stdout
    assert "phase4b_policy_eval_acceptance" not in result.stderr


def test_phase4b_in_container_runner_fails_when_policy_missing(tmp_path):
    sim_root = Path(__file__).resolve().parents[4]
    script = (
        sim_root /
        "carla-0.9.15" /
        "PythonAPI" /
        "taeho" /
        "scripts" /
        "run_phase4b_exported_policy_eval.sh")

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-command",
            "--policy-dir",
            str(tmp_path / "missing_policy_dir"),
            "--output-dir",
            str(tmp_path / "out"),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False)

    assert result.returncode == 2
    assert "missing policy file" in result.stderr
    assert "policy.ts" in result.stderr

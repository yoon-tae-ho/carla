import json
import subprocess
from pathlib import Path


TAEHO_ROOT = Path(__file__).resolve().parents[1]
SIM_ROOT = Path(__file__).resolve().parents[4]


def test_phase4d_host_no_planning_help_does_not_start_docker():
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_no_planning_train_eval.sh"),
        "--help",
    ], cwd=SIM_ROOT / "docker")

    assert "Run Phase 4-D no-planning live SAC train/eval" in result.stdout
    assert "docker compose" not in result.stdout
    assert result.stderr == ""


def test_phase4d_host_s8_reference_help_does_not_start_docker():
    result = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_s8_reference_only.sh"),
        "--help",
    ], cwd=SIM_ROOT / "docker")

    assert "Run Phase 4-D S8 zero-residual reference" in result.stdout
    assert "docker compose" not in result.stdout
    assert result.stderr == ""


def test_phase4d_host_wrappers_translate_host_paths():
    host_path = SIM_ROOT / "e2e_models" / "outputs" / "phase4d_example"

    no_planning = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_no_planning_train_eval.sh"),
        "--translate-path",
        str(host_path),
    ], cwd=SIM_ROOT / "docker")
    s8_reference = _run([
        "bash",
        str(SIM_ROOT / "docker" / "run_phase4d_s8_reference_only.sh"),
        "--translate-path",
        str(host_path),
    ], cwd=SIM_ROOT / "docker")

    assert no_planning.stdout.strip() == "/workspace/e2e_models/outputs/phase4d_example"
    assert s8_reference.stdout.strip() == "/workspace/e2e_models/outputs/phase4d_example"


def test_phase4d_no_planning_print_command_forces_train_and_eval_flags(tmp_path):
    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4d_no_planning_train_eval.sh"),
        "--print-command",
        "--seed",
        "100",
        "--routes-subset",
        "00",
        "--reference-summary",
        str(tmp_path / "seed100_S8_rl_zero_residual_skyhook" / "summary.json"),
        "--output-root",
        str(tmp_path / "phase4d"),
    ], cwd=TAEHO_ROOT)
    stdout = result.stdout

    assert "TRAIN:" in stdout
    assert "--backend live" in stdout
    assert "--algorithm sac" in stdout
    assert "--total-timesteps 4096" in stdout
    assert "--max-episode-steps 0" in stdout
    assert "--verify-every 0" in stdout
    assert "--eval-after-training false" in stdout
    assert "--planning-provider empty" in stdout
    assert "--timeout 120" in stdout
    assert "--hero-timeout-seconds 900" in stdout
    assert "--route-wait-timeout-seconds 120" in stdout
    assert "--connect-retry-seconds 1.0" in stdout

    assert "TRAIN_ACCEPTANCE:" in stdout
    assert "phase4d_train_acceptance" in stdout
    assert "--training-return-code" in stdout

    assert "EVAL_S4:" in stdout
    assert "--scenarios rl_residual_skyhook" in stdout
    assert "--metric-warmup-seconds 3.0" in stdout
    assert "--timeout 120" in stdout
    assert "--tick-wait-timeout 120" in stdout
    assert "--sidecar-tick-mode poll" in stdout
    assert "--poll-seconds 0.05" in stdout
    assert "--verify-every 50" in stdout
    assert "--require-verification" in stdout
    assert "--rl-policy" in stdout
    assert "--rl-normalizer" in stdout

    assert "ACCEPTANCE:" in stdout
    assert "phase4b_policy_eval_acceptance" in stdout
    assert "POLICY_SELECTION:" in stdout
    assert "phase4d_policy_selection" in stdout
    assert "SUMMARY:" in stdout
    assert "phase4d_lifecycle_summary.json" in stdout


def test_phase4d_no_planning_print_command_allows_explicit_advanced_8192_override(tmp_path):
    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4d_no_planning_train_eval.sh"),
        "--print-command",
        "--total-timesteps",
        "8192",
        "--reference-summary",
        str(tmp_path / "s8" / "summary.json"),
        "--output-root",
        str(tmp_path / "phase4d"),
    ], cwd=TAEHO_ROOT)

    assert "--total-timesteps 8192" in result.stdout


def test_phase4d_no_planning_print_command_keeps_s8_and_s4_split(tmp_path):
    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4d_no_planning_train_eval.sh"),
        "--print-command",
        "--reference-summary",
        str(tmp_path / "s8" / "summary.json"),
        "--output-root",
        str(tmp_path / "phase4d"),
    ], cwd=TAEHO_ROOT)
    stdout = result.stdout

    assert "--scenarios rl_residual_skyhook" in stdout
    assert "--scenarios rl_zero_residual_skyhook\\,rl_residual_skyhook" not in stdout
    assert "--scenarios rl_residual_skyhook\\,rl_zero_residual_skyhook" not in stdout


def test_phase4d_s8_reference_print_command_only_uses_zero_residual(tmp_path):
    result = _run([
        "bash",
        str(TAEHO_ROOT / "scripts" / "run_phase4d_s8_reference_only.sh"),
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
    assert "--scenarios rl_zero_residual_skyhook\\,rl_residual_skyhook" not in stdout
    assert "--scenarios rl_residual_skyhook" not in stdout
    assert "--verify-every 50" in stdout
    assert "--require-verification" in stdout
    assert "--planning-provider empty" in stdout


def test_phase4d_no_planning_requires_reference_summary(tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(TAEHO_ROOT / "scripts" / "run_phase4d_no_planning_train_eval.sh"),
            "--print-command",
            "--output-root",
            str(tmp_path / "phase4d"),
        ],
        cwd=str(TAEHO_ROOT),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert result.returncode == 2
    assert "requires --reference-summary" in result.stderr


def test_phase4d_no_planning_failed_train_gate_writes_summaries(tmp_path):
    output_root = tmp_path / "phase4d_failed_gate"
    result = subprocess.run(
        [
            "bash",
            str(TAEHO_ROOT / "scripts" / "run_phase4d_no_planning_train_eval.sh"),
            "--reference-summary",
            str(tmp_path / "s8" / "summary.json"),
            "--output-root",
            str(output_root),
            "--route-script",
            str(tmp_path / "missing_route.sh"),
        ],
        cwd=str(TAEHO_ROOT),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    assert result.returncode != 0
    assert "Phase 4-D training gate failed; S4 eval skipped." in result.stderr
    assert not (output_root / "eval_s4" / "suite_summary.csv").exists()

    summary = _read_json(output_root / "phase4d_no_planning_summary.json")
    comparison_summary = _read_json(
        output_root / "comparison" / "phase4d_no_planning_summary.json")
    train_acceptance = _read_json(
        output_root / "comparison" / "phase4d_train_acceptance.json")
    lifecycle = _read_json(
        output_root / "comparison" / "phase4d_lifecycle_summary.json")

    assert summary["phase4d_status"] == "failed_training"
    assert summary["eval_status"] == "skipped_due_to_failed_training"
    assert summary["invalid_for_eval"] == 1
    assert summary["engineering_pass"] == 0
    assert summary["phase4d_lifecycle_summary_json"].endswith(
        "comparison/phase4d_lifecycle_summary.json")
    assert comparison_summary["phase4d_status"] == "failed_training"
    assert train_acceptance["eval_allowed"] == 0
    assert train_acceptance["invalid_for_eval"] == 1
    assert lifecycle["phase4d_status"] == "failed_training"
    assert lifecycle["eval_status"] == "skipped_due_to_failed_training"
    assert lifecycle["invalid_for_eval"] == 1
    assert lifecycle["training_summary"].endswith("train/training_summary.json")
    assert (output_root / "comparison" / "phase4d_lifecycle_summary.csv").is_file()


def _run(args, cwd):
    return subprocess.run(
        args,
        cwd=str(cwd),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)


def _read_json(path):
    with open(path) as json_file:
        return json.load(json_file)

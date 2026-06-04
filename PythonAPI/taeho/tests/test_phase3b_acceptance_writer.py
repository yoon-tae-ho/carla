import csv
import json
import subprocess
from pathlib import Path

from suspension_control.rl.reward_calibration import (
    write_phase3b_reward_calibration_report,
)
from tests.test_phase3b_reward_calibration_report import valid_rows


def test_phase3b_reward_calibration_writer_creates_csv_and_json(tmp_path):
    csv_path, json_path, rows = write_phase3b_reward_calibration_report(
        str(tmp_path),
        valid_rows())

    assert rows
    assert all(row["phase3b_status"] == "pass" for row in rows)

    with open(csv_path) as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    with open(json_path) as json_file:
        json_rows = json.load(json_file)

    assert len(csv_rows) == 4
    assert len(json_rows) == 4
    assert csv_rows[0]["phase3b_status"] == "pass"
    assert json_rows[0]["phase3b_status"] == "pass"
    assert "random_total_reward_not_above_reference_ok" in csv_rows[0]
    assert "reward_random_total_delta_vs_reference" in csv_rows[0]


def test_phase3b_runner_help_smoke():
    sim_root = Path(__file__).resolve().parents[4]
    script = sim_root / "docker" / "run_phase3b_reward_calibration.sh"

    result = subprocess.run(
        ["bash", str(script), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False)

    assert result.returncode == 0
    assert "Phase 3-B reward/progress calibration" in result.stdout
    assert "run_phase3b_reward_calibration.sh" in result.stdout

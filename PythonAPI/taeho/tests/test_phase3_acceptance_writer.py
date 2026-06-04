import csv
import json

from suspension_control.rl.reward_sanity import write_phase3_reward_sanity_report
from tests.test_reward_sanity_report import valid_rows


def test_phase3_reward_sanity_writer_creates_csv_and_json(tmp_path):
    csv_path, json_path, rows = write_phase3_reward_sanity_report(
        str(tmp_path),
        valid_rows())

    assert rows
    assert all(row["phase3_status"] == "pass" for row in rows)

    with open(csv_path) as csv_file:
        csv_rows = list(csv.DictReader(csv_file))
    with open(json_path) as json_file:
        json_rows = json.load(json_file)

    assert len(csv_rows) == 4
    assert len(json_rows) == 4
    assert csv_rows[0]["phase3_status"] == "pass"
    assert json_rows[0]["phase3_status"] == "pass"
    assert "mean_reward_term_action_rate" in csv_rows[0]

"""CSV/JSONL rollout logger for residual-RL suspension experiments."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from .diagnostics import command_scales, flatten_state, indexed_values, planning_summary


DEFAULT_CSV_FIELDS = (
    "episode_id",
    "step",
    "frame",
    "elapsed_seconds",
    "dt",
    "reward_total",
    "reward_comfort",
    "reward_stability",
    "reward_task",
    "reward_action",
    "reward_safety",
    "rl_fallback_reason",
    "rl_safety_gain",
    "planning_available",
)


class RolloutLogger:
    """Write rollout rows to JSONL and optionally CSV."""

    def __init__(
        self,
        output_dir: str,
        jsonl_name: str = "rollout.jsonl",
        csv_name: str = "rollout.csv",
        csv_fields: Sequence[str] = DEFAULT_CSV_FIELDS,
    ):
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        os.makedirs(self.output_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.output_dir, jsonl_name)
        self.csv_path = os.path.join(self.output_dir, csv_name)
        self.csv_fields = tuple(csv_fields)
        self._jsonl_file = None
        self._csv_file = None
        self._csv_writer = None

    def __enter__(self) -> "RolloutLogger":
        self.open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def open(self) -> None:
        if self._jsonl_file is None:
            self._jsonl_file = open(self.jsonl_path, "w")
        if self._csv_file is None:
            self._csv_file = open(self.csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self.csv_fields,
                extrasaction="ignore")
            self._csv_writer.writeheader()

    def close(self) -> None:
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    def log(self, row: Mapping[str, Any]) -> None:
        if self._jsonl_file is None or self._csv_writer is None:
            self.open()
        assert self._jsonl_file is not None
        assert self._csv_writer is not None
        plain = _jsonable_dict(dict(row))
        self._jsonl_file.write(json.dumps(plain, sort_keys=True) + "\n")
        self._csv_writer.writerow({
            field: plain.get(field, "")
            for field in self.csv_fields
        })

    def log_transition(
        self,
        episode_id: str,
        step: int,
        state: Any,
        planning: Any,
        baseline_command: Any,
        action: Sequence[float],
        final_command: Any,
        reward_diagnostics: Mapping[str, Any],
        controller_diagnostics: Optional[Mapping[str, Any]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        row: Dict[str, Any] = {
            "episode_id": episode_id,
            "step": step,
            "frame": getattr(state, "frame", ""),
            "elapsed_seconds": getattr(state, "elapsed_seconds", ""),
            "dt": getattr(state, "dt", ""),
        }
        row.update(flatten_state(state))
        row.update(planning_summary(planning))
        row.update(command_scales(baseline_command, "baseline_"))
        row.update(indexed_values("action_", action))
        row.update(command_scales(final_command, "final_"))
        row.update(dict(reward_diagnostics or {}))
        row.update(dict(controller_diagnostics or {}))
        row.update(dict(extra or {}))
        self.log(row)
        return row


def _jsonable_dict(row: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        else:
            result[key] = str(value)
    return result

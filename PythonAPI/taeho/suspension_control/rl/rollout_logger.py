"""CSV/JSONL rollout logger for residual-RL suspension experiments."""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Mapping, Optional, Sequence

from .diagnostics import command_scales, flatten_state, indexed_values, planning_summary


DEFAULT_CSV_FIELDS = (
    "episode_id",
    "step",
    "frame",
    "backend",
    "real_backend_used",
    "fake_backend_used",
    "carla_connected",
    "route_process_started",
    "hero_attached",
    "route_process_alive",
    "route_id",
    "seed",
    "elapsed_seconds",
    "dt",
    "speed",
    "local_ax",
    "local_ay",
    "az",
    "roll",
    "pitch",
    "yaw_rate",
    "roll_rate",
    "pitch_rate",
    "planning_available",
    "planning_source",
    "planning_stale",
    "planning_age_frames",
    "baseline_damper_fl",
    "baseline_damper_fr",
    "baseline_damper_rl",
    "baseline_damper_rr",
    "action_fl",
    "action_fr",
    "action_rl",
    "action_rr",
    "rl_mean_abs_action",
    "rl_action_rate",
    "raw_residual_damper_fl",
    "raw_residual_damper_fr",
    "raw_residual_damper_rl",
    "raw_residual_damper_rr",
    "safety_scaled_residual_damper_fl",
    "safety_scaled_residual_damper_fr",
    "safety_scaled_residual_damper_rl",
    "safety_scaled_residual_damper_rr",
    "rate_limited_residual_damper_fl",
    "rate_limited_residual_damper_fr",
    "rate_limited_residual_damper_rl",
    "rate_limited_residual_damper_rr",
    "final_residual_damper_fl",
    "final_residual_damper_fr",
    "final_residual_damper_rl",
    "final_residual_damper_rr",
    "rl_safety_gain",
    "rl_safety_gate_reason",
    "final_damper_fl",
    "final_damper_fr",
    "final_damper_rl",
    "final_damper_rr",
    "readback_mean_damper_scale",
    "reward_total",
    "reward_comfort",
    "reward_stability",
    "reward_task",
    "reward_action",
    "reward_safety",
    "reward_cost_comfort",
    "reward_cost_stability",
    "reward_cost_task",
    "reward_cost_action",
    "reward_cost_safety",
    "reward_term_action_mag",
    "reward_term_action_rate",
    "reward_term_damper_rate",
    "reward_term_baseline_dev",
    "reward_term_low_speed_not_planned",
    "reward_term_progress_stall",
    "reward_term_negative_progress",
    "reward_term_abs_speed_error",
    "reward_term_route_deviation",
    "reward_route_deviation_used",
    "reward_route_deviation_enabled",
    "reward_route_deviation_valid",
    "reward_term_collision",
    "reward_term_lane_invasion",
    "reward_term_red_light",
    "reward_term_blocked_vehicle",
    "reward_term_route_timeout",
    "reward_task_capped_without_infraction",
    "reward_task_abs_max_without_infraction",
    "reward_total_unmasked",
    "reward_initial_masked",
    "reward_initial_skip_seconds",
    "route_progress_available",
    "route_progress_raw_m",
    "route_progress_m",
    "route_progress_fraction",
    "route_progress_monotonic_m",
    "route_progress_monotonic_fraction",
    "route_progress_total_length_m",
    "route_total_length_m",
    "route_progress_delta_m",
    "route_delta_progress_m",
    "route_progress_rate_mps",
    "route_progress_raw_negative_delta_m",
    "route_progress_negative_raw",
    "route_progress_raw_deviation_m",
    "route_progress_deviation_valid",
    "route_deviation_m",
    "route_deviation_valid",
    "route_distance_to_end_m",
    "route_progress_nearest_segment_index",
    "route_progress_projection_segment",
    "route_progress_tracker_status",
    "route_progress_error",
    "delta_progress",
    "target_speed",
    "target_speed_source",
    "speed_error",
    "abs_speed_error",
    "target_speed_error",
    "planned_stop",
    "route_deviation",
    "low_speed_not_planned",
    "low_speed_not_planned_severity",
    "route_progress_stall",
    "progress_stall",
    "progress_stall_count",
    "negative_progress",
    "route_completion_proxy",
    "collision_count",
    "lane_invasion_count",
    "red_light_count",
    "terminated",
    "truncated",
    "done_reason",
    "truncated_reason",
    "rl_fallback_reason",
    "rl_mean_abs_residual_damper",
    "rl_raw_residual_damper_fl",
    "rl_raw_residual_damper_fr",
    "rl_raw_residual_damper_rl",
    "rl_raw_residual_damper_rr",
    "rl_residual_damper_fl",
    "rl_residual_damper_fr",
    "rl_residual_damper_rl",
    "rl_residual_damper_rr",
    "rl_safety_scaled_residual_damper_fl",
    "rl_safety_scaled_residual_damper_fr",
    "rl_safety_scaled_residual_damper_rl",
    "rl_safety_scaled_residual_damper_rr",
    "rl_rate_limited_residual_damper_fl",
    "rl_rate_limited_residual_damper_fr",
    "rl_rate_limited_residual_damper_rl",
    "rl_rate_limited_residual_damper_rr",
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
        row = build_rollout_row(
            episode_id=episode_id,
            step=step,
            state=state,
            planning=planning,
            baseline_command=baseline_command,
            action=action,
            final_command=final_command,
            reward_diagnostics=reward_diagnostics,
            controller_diagnostics=controller_diagnostics,
            extra=extra)
        self.log(row)
        return row


def build_rollout_row(
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
    row.update(flatten_state(state, prefix=""))
    row.update(flatten_state(state))
    row.update(planning_summary(
        planning,
        current_frame=getattr(state, "frame", None)))
    row.update(command_scales(baseline_command, "baseline_"))
    row.update(indexed_values("action_", action))
    row.update(command_scales(final_command, "final_"))
    row.update(dict(reward_diagnostics or {}))
    row.update(dict(controller_diagnostics or {}))
    row.update(dict(extra or {}))
    _add_projection_aliases(row)
    return row


def _add_projection_aliases(row: Dict[str, Any]) -> None:
    labels = ("fl", "fr", "rl", "rr")
    mappings = (
        ("rl_raw_residual_damper_", "raw_residual_damper_"),
        ("rl_safety_scaled_residual_damper_", "safety_scaled_residual_damper_"),
        ("rl_rate_limited_residual_damper_", "rate_limited_residual_damper_"),
        ("rl_final_residual_damper_", "final_residual_damper_"),
        ("rl_residual_damper_", "final_residual_damper_"),
    )
    for source_prefix, alias_prefix in mappings:
        for label in labels:
            source = source_prefix + label
            alias = alias_prefix + label
            if alias not in row and source in row:
                row[alias] = row[source]


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

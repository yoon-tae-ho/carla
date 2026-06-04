"""Evaluate residual-RL policy scaffolds or dry-run rollouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

from .carla_backend import FakeCarlaSuspensionBackend
from .carla_env import SuspensionCarlaEnv
from .carla_online_env import CarlaSuspensionEnv
from .policy import PolicyAdapter
from .rollout_logger import RolloutLogger
from .task_info import task_summary_fields


def evaluate_dry_run(args: argparse.Namespace) -> Dict[str, Any]:
    env = SuspensionCarlaEnv(dry_run=True, max_steps=args.steps)
    observation, info = env.reset()
    del info
    policy = PolicyAdapter(
        policy_path=args.policy,
        action_dim=env.wheel_count,
        allow_dummy_zero=True)
    rewards: List[float] = []
    rows: List[Dict[str, Any]] = []
    with RolloutLogger(args.output_dir) as logger:
        for step in range(args.steps):
            if policy.is_available:
                try:
                    action = policy.predict(observation)
                except Exception:
                    action = [0.0 for _ in range(env.wheel_count)]
            else:
                action = [0.0 for _ in range(env.wheel_count)]
            observation, reward, terminated, truncated, step_info = env.step(action)
            rewards.append(float(reward))
            reward_diag = dict(step_info.get("reward_diagnostics", {}))
            row = logger.log_transition(
                episode_id="dry_run",
                step=step,
                state=env.state,
                planning=None,
                baseline_command=step_info.get("baseline_command"),
                action=action,
                final_command=step_info.get("final_command"),
                reward_diagnostics=reward_diag,
                controller_diagnostics=step_info.get("controller_diagnostics", {}),
                extra={"terminated": int(terminated), "truncated": int(truncated)})
            rows.append(row)
            if terminated or truncated:
                break
    summary = {
        "mode": "dry_run",
        "steps": len(rewards),
        "total_reward": sum(rewards),
        "mean_reward": sum(rewards) / float(len(rewards)) if rewards else 0.0,
        "rollout_jsonl": os.path.join(os.path.abspath(args.output_dir), "rollout.jsonl"),
        "rollout_csv": os.path.join(os.path.abspath(args.output_dir), "rollout.csv"),
    }
    summary.update(task_summary_fields(rows))
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as json_file:
        json.dump(summary, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    summary["summary_json"] = summary_path
    return summary


def evaluate_fake_online(args: argparse.Namespace) -> Dict[str, Any]:
    backend = FakeCarlaSuspensionBackend(max_steps=args.steps + 80)
    env = CarlaSuspensionEnv(
        backend=backend,
        baseline="skyhook",
        planning_provider="empty",
        warmup_seconds=args.warmup_seconds,
        route_id=args.route_id,
        seed=args.seed)
    observation, info = env.reset(seed=args.seed)
    policy = PolicyAdapter(
        policy_path=args.policy,
        action_dim=env.wheel_count,
        allow_dummy_zero=True)
    rewards: List[float] = []
    rows: List[Dict[str, Any]] = []
    with RolloutLogger(args.output_dir) as logger:
        for step in range(args.steps):
            if policy.is_available:
                try:
                    action = policy.predict(observation)
                except Exception:
                    action = [0.0 for _ in range(env.wheel_count)]
            else:
                action = [0.0 for _ in range(env.wheel_count)]
            observation, reward, terminated, truncated, step_info = env.step(action)
            rewards.append(float(reward))
            row = logger.log_transition(
                episode_id="fake_online",
                step=step,
                state=env.state,
                planning=env.planning,
                baseline_command=step_info.get("baseline_command"),
                action=action,
                final_command=step_info.get("final_command"),
                reward_diagnostics=step_info.get("reward_diagnostics", {}),
                controller_diagnostics=step_info.get("controller_diagnostics", {}),
                extra=dict(
                    _merged_dicts(
                        step_info.get("task_info", {}),
                        step_info.get("backend_info", {}),
                        {
                            "route_id": step_info.get("route_id", ""),
                            "seed": step_info.get("seed", ""),
                            "done_reason": step_info.get("done_reason", ""),
                            "truncated_reason": step_info.get("truncated_reason", ""),
                            "terminated": int(terminated),
                            "truncated": int(truncated),
                        })))
            rows.append(row)
            if terminated or truncated:
                break
    summary = {
        "mode": "fake_online",
        "steps": len(rewards),
        "warmup_skipped_steps": info.get("warmup_skipped_steps", ""),
        "total_reward": sum(rewards),
        "mean_reward": sum(rewards) / float(len(rewards)) if rewards else 0.0,
        "rollout_jsonl": os.path.join(os.path.abspath(args.output_dir), "rollout.jsonl"),
        "rollout_csv": os.path.join(os.path.abspath(args.output_dir), "rollout.csv"),
    }
    summary.update(task_summary_fields(rows))
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as json_file:
        json.dump(summary, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    summary["summary_json"] = summary_path
    env.close()
    return summary


def run_route_evaluation(args: argparse.Namespace) -> int:
    command = [
        sys.executable,
        args.route_suite,
        "--scenarios",
        args.scenarios,
        "--rl-residual-config",
        args.config,
        "--output-dir",
        args.output_dir,
    ]
    if args.policy:
        command.extend(["--rl-policy", args.policy])
    if args.normalizer:
        command.extend(["--rl-normalizer", args.normalizer])
    return subprocess.call(command)


def _merged_dicts(*items: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for item in items:
        result.update(dict(item or {}))
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fake-online", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--route-id", default="00")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="/tmp/rl_suspension_dry_run")
    parser.add_argument("--config", default="suspension_control/configs/rl_residual.yaml")
    parser.add_argument("--policy", default="")
    parser.add_argument("--normalizer", default="")
    parser.add_argument(
        "--route-suite",
        default="transfuser_suspension_control_suite.py")
    parser.add_argument(
        "--scenarios",
        default="rl_residual_skyhook")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.dry_run:
        summary = evaluate_dry_run(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.fake_online:
        summary = evaluate_fake_online(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    return run_route_evaluation(args)


if __name__ == "__main__":
    raise SystemExit(main())

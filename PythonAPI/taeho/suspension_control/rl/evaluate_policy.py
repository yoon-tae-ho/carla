"""Evaluate residual-RL policy scaffolds or dry-run rollouts."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List

from .carla_env import SuspensionCarlaEnv
from .policy import PolicyAdapter
from .rollout_logger import RolloutLogger


def evaluate_dry_run(args: argparse.Namespace) -> Dict[str, Any]:
    env = SuspensionCarlaEnv(dry_run=True, max_steps=args.steps)
    observation, info = env.reset()
    del info
    policy = PolicyAdapter(
        policy_path=args.policy,
        action_dim=env.wheel_count,
        allow_dummy_zero=True)
    rewards: List[float] = []
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
            logger.log_transition(
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
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as json_file:
        json.dump(summary, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    summary["summary_json"] = summary_path
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int, default=100)
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
    return run_route_evaluation(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Training entrypoint scaffold for residual-RL suspension policies."""

from __future__ import annotations

import argparse
import os
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="suspension_control/configs/rl_training.yaml")
    parser.add_argument("--output-dir", default="/tmp/rl_suspension_train")
    parser.add_argument("--total-timesteps", type=int, default=10000)
    parser.add_argument("--algorithm", choices=("sac", "td3"), default="sac")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        import gymnasium  # noqa: F401
        from stable_baselines3 import SAC, TD3  # type: ignore
    except Exception as error:
        print(
            "Optional training dependencies are missing: %s. "
            "Install gymnasium and stable-baselines3 to train policies." % error,
            file=sys.stderr)
        return 1

    from .carla_env import SuspensionCarlaEnv

    env = SuspensionCarlaEnv(dry_run=True)
    if args.algorithm == "sac":
        model = SAC("MlpPolicy", env, seed=args.seed, verbose=1)
    else:
        model = TD3("MlpPolicy", env, seed=args.seed, verbose=1)
    model.learn(total_timesteps=args.total_timesteps)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save("%s/policy" % args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Training entrypoint for residual-RL suspension policies.

This CLI is deliberately conservative: synthetic and fake backends require an
explicit opt-in, and the real CARLA backend must fail clearly until it is wired
to the route runner.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple

from .action_projection import ResidualActionProjectorConfig
from .carla_backend import (
    FakeCarlaSuspensionBackend,
    LiveCarlaSuspensionBackend,
)
from .carla_env import SuspensionSyntheticEnv
from .carla_online_env import CarlaSuspensionEnv
from .policy_export import (
    export_sb3_actor_to_torchscript,
    write_normalizer_metadata,
)
from .rollout_logger import DEFAULT_CSV_FIELDS, RolloutLogger
from .task_info import task_summary_fields


ACTION_SEMANTICS = "normalized_damper_residual_v1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="suspension_control/configs/rl_training_carla.yaml")
    parser.add_argument(
        "--env",
        choices=("carla", "synthetic", "fake"),
        default="carla")
    parser.add_argument(
        "--allow-synthetic-training",
        action="store_true",
        help="Required for --env synthetic or --env fake smoke training.")
    parser.add_argument("--algorithm", choices=("sac", "td3"), default="sac")
    parser.add_argument("--baseline", choices=("skyhook", "pid"), default="skyhook")
    parser.add_argument(
        "--planning-provider",
        choices=("empty", "jsonl", "control_history"),
        default="empty")
    parser.add_argument("--routes", default="")
    parser.add_argument("--routes-subset", default="00")
    parser.add_argument("--team-agent", default="")
    parser.add_argument("--team-config", default="")
    parser.add_argument("--total-timesteps", type=int, default=10000)
    parser.add_argument("--output-dir", default="/tmp/rl_suspension_train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--export-policy", action="store_true")
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--observation-clip", type=float, default=5.0)
    parser.add_argument("--max-damper-residual-scale", type=float, default=0.08)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--role-name", default="hero")
    parser.add_argument("--actor-id", default="")
    parser.add_argument("--verify-suspension", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    guard_error = _guard_training_mode(args)
    if guard_error:
        return _fail(guard_error, code=2)

    try:
        import gymnasium  # noqa: F401
        from stable_baselines3 import SAC, TD3  # type: ignore
        from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
    except Exception as error:
        return _fail(
            "Optional training dependencies are missing: %s. "
            "Install gymnasium and stable-baselines3 to train policies." % error)

    os.makedirs(args.output_dir, exist_ok=True)
    env = _make_env(args)
    feature_names = _observation_feature_names(env)
    _write_training_config(args, feature_names)
    write_normalizer_metadata(
        os.path.join(args.output_dir, "normalizer.json"),
        feature_names=feature_names,
        clip=args.observation_clip,
        action_semantics=ACTION_SEMANTICS,
        max_damper_residual_scale=args.max_damper_residual_scale)

    if args.algorithm == "sac":
        model = SAC("MlpPolicy", env, seed=args.seed, verbose=1)
    else:
        model = TD3("MlpPolicy", env, seed=args.seed, verbose=1)
    rollout_rows = []
    with RolloutLogger(
            args.output_dir,
            jsonl_name="rollout_train.jsonl",
            csv_name="rollout_train.csv") as rollout_logger:
        callback = _make_rollout_callback(BaseCallback, rollout_logger, rollout_rows)
        model.learn(
            total_timesteps=max(0, int(args.total_timesteps)),
            callback=callback)

    policy_base = os.path.join(args.output_dir, "policy")
    model.save(policy_base)
    policy_zip = policy_base + ".zip"
    if args.export_policy:
        export_sb3_actor_to_torchscript(
            model_path=policy_zip,
            output_path=os.path.join(args.output_dir, "policy_actor.ts"),
            observation_dim=len(feature_names),
            action_dim=4)
    _write_training_artifact_summaries(args, rollout_rows)
    if hasattr(env, "close"):
        env.close()
    return 0


def _guard_training_mode(args: argparse.Namespace) -> str:
    if args.env in ("synthetic", "fake") and not args.allow_synthetic_training:
        return "synthetic training requires --allow-synthetic-training"
    if args.env == "carla":
        backend = _make_live_backend(args)
        return backend.unavailable_message
    return ""


def _make_env(args: argparse.Namespace) -> Any:
    if args.env == "synthetic":
        return SuspensionSyntheticEnv(
            dry_run=True,
            max_steps=max(100, int(args.total_timesteps)),
            initial_transition_skip_seconds=args.warmup_seconds,
            residual_scale=args.max_damper_residual_scale)
    if args.env == "fake":
        backend = FakeCarlaSuspensionBackend(
            max_steps=max(200, int(args.total_timesteps) + 80),
            planning_provider=args.planning_provider)
        projector_config = ResidualActionProjectorConfig(
            max_damper_residual_scale=args.max_damper_residual_scale)
        return CarlaSuspensionEnv(
            backend=backend,
            baseline=args.baseline,
            planning_provider=args.planning_provider,
            projector_config=projector_config,
            warmup_seconds=args.warmup_seconds,
            route_id=args.routes_subset,
            seed=args.seed)
    return CarlaSuspensionEnv(
        backend=_make_live_backend(args),
        baseline=args.baseline,
        planning_provider=args.planning_provider,
        warmup_seconds=args.warmup_seconds,
        verify_suspension=args.verify_suspension,
        route_id=args.routes_subset,
        seed=args.seed)


def _make_live_backend(args: argparse.Namespace) -> LiveCarlaSuspensionBackend:
    return LiveCarlaSuspensionBackend(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        routes=args.routes,
        routes_subset=args.routes_subset,
        team_agent=args.team_agent,
        team_config=args.team_config,
        baseline=args.baseline,
        planning_provider=args.planning_provider,
        traffic_manager_port=args.traffic_manager_port,
        fixed_delta_seconds=args.fixed_delta_seconds,
        seed=args.seed,
        role_name=args.role_name,
        actor_id=args.actor_id,
        verify_suspension=args.verify_suspension)


def _observation_feature_names(env: Any) -> Tuple[str, ...]:
    builder = getattr(env, "observation_builder", None)
    if builder is not None and hasattr(builder, "spec"):
        return tuple(builder.spec().feature_names)
    space = getattr(env, "observation_space", None)
    shape = tuple(getattr(space, "shape", ()) or ())
    count = int(shape[0]) if shape else 0
    return tuple("obs_%03d" % index for index in range(count))


def _write_training_config(
    args: argparse.Namespace,
    feature_names: Sequence[str],
) -> str:
    config = {
        "training_env": args.env,
        "algorithm": args.algorithm,
        "baseline": args.baseline,
        "planning_provider": args.planning_provider,
        "routes": args.routes,
        "routes_subset": args.routes_subset,
        "team_agent": args.team_agent,
        "team_config": args.team_config,
        "seed": int(args.seed),
        "total_timesteps": int(args.total_timesteps),
        "observation_feature_names": list(feature_names),
        "action_semantics": ACTION_SEMANTICS,
        "max_damper_residual_scale": float(args.max_damper_residual_scale),
        "spring_frozen": True,
        "warmup_seconds": float(args.warmup_seconds),
        "observation_clip": float(args.observation_clip),
        "host": args.host,
        "port": int(args.port),
        "timeout": float(args.timeout),
        "traffic_manager_port": int(args.traffic_manager_port),
        "fixed_delta_seconds": float(args.fixed_delta_seconds),
        "role_name": args.role_name,
        "actor_id": args.actor_id,
        "verify_suspension": bool(args.verify_suspension),
        "synthetic_training": args.env in ("synthetic", "fake"),
        "live_backend_available": False,
        "export_policy": bool(args.export_policy),
        "config": args.config,
    }
    path = os.path.join(args.output_dir, "training_config.json")
    with open(path, "w") as json_file:
        json.dump(config, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    return path


def _make_rollout_callback(base_callback_cls: Any, rollout_logger: Any, rows: list):
    class TrainingRolloutCallback(base_callback_cls):  # type: ignore
        def _on_step(self) -> bool:
            infos = self.locals.get("infos", []) or []
            for info in infos:
                row = dict((info or {}).get("rollout_row", {}) or {})
                if row:
                    rollout_logger.log(row)
                    rows.append(row)
            return True

    return TrainingRolloutCallback()


def _write_training_artifact_summaries(
    args: argparse.Namespace,
    rollout_rows: Sequence[Mapping[str, Any]],
) -> None:
    rollout_csv = os.path.join(args.output_dir, "rollout_train.csv")
    if not os.path.isfile(rollout_csv):
        with open(rollout_csv, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=DEFAULT_CSV_FIELDS)
            writer.writeheader()
    rollout_jsonl = os.path.join(args.output_dir, "rollout_train.jsonl")
    if not os.path.isfile(rollout_jsonl):
        open(rollout_jsonl, "w").close()
    summary = task_summary_fields(rollout_rows)
    _write_single_row_csv(
        os.path.join(args.output_dir, "reward_summary.csv"),
        dict(summary, **{
            "training_env": args.env,
            "algorithm": args.algorithm,
            "total_timesteps": int(args.total_timesteps),
            "reward_rows": len(rollout_rows),
        }))
    _write_single_row_csv(
        os.path.join(args.output_dir, "acceptance_summary.csv"),
        dict(summary, **{
            "training_env": args.env,
            "algorithm": args.algorithm,
            "synthetic_training": int(args.env in ("synthetic", "fake")),
            "live_backend_available": 0,
            "status": "trained",
        }))


def _write_single_row_csv(path: str, row: Dict[str, Any]) -> None:
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _fail(message: str, code: int = 1) -> int:
    print(message, file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

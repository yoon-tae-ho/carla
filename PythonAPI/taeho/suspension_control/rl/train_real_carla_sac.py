"""Phase 4 real-CARLA route training entrypoint.

The default backend is the live route-process backend. Fake training is kept as
an explicit smoke path for CLI and artifact validation; there is no synthetic
fallback from live CARLA.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .action_projection import ResidualActionProjectorConfig
from .carla_route_backend import FakeCarlaRouteBackend, LiveRouteProcessBackend
from .carla_route_env import CarlaRouteSuspensionEnv
from .policy_export import (
    export_sb3_actor_to_torchscript,
    export_torch_module,
    write_normalizer_metadata,
)
from .rollout_logger import DEFAULT_CSV_FIELDS, RolloutLogger
from .task_info import task_summary_fields


ACTION_SEMANTICS = "normalized_damper_residual_v1"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("live", "fake"), default="live")
    parser.add_argument("--algorithm", choices=("sac", "td3"), default="sac")
    parser.add_argument("--baseline", choices=("skyhook", "pid"), default="skyhook")
    parser.add_argument("--total-timesteps", type=int, default=10000)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--routes", default="")
    parser.add_argument("--routes-subset", default="00")
    parser.add_argument("--route-script", default="")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--hero-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--route-wait-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--connect-retry-seconds", type=float, default=1.0)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--role-name", default="hero")
    parser.add_argument("--actor-id", default="")
    parser.add_argument("--team-agent", default="")
    parser.add_argument("--team-config", default="")
    parser.add_argument("--verify-every", type=int, default=0)
    parser.add_argument("--readback-tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--planning-provider",
        choices=("empty", "jsonl", "control_history"),
        default="empty")
    parser.add_argument("--planning-preview-jsonl", default="")
    parser.add_argument("--output-dir", default="/tmp/phase4_real_carla_train")
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--save-freq", type=int, default=0)
    parser.add_argument(
        "--eval-after-training",
        type=_parse_bool,
        choices=(True, False),
        default=False)
    parser.add_argument("--eval-route-suite", default="")
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--observation-clip", type=float, default=5.0)
    parser.add_argument("--max-damper-residual-scale", type=float, default=0.08)
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    parser.add_argument("--debug", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.backend == "fake":
        print(
            "WARNING: Phase 4 fake backend selected; fake_backend_used=1 and "
            "no real CARLA training is occurring.",
            file=sys.stderr)
    else:
        unavailable = _make_live_backend(args).unavailable_message
        if unavailable:
            return _fail(unavailable, code=2)

    try:
        import gymnasium  # noqa: F401
        from stable_baselines3 import SAC, TD3  # type: ignore
        from stable_baselines3.common.callbacks import BaseCallback  # type: ignore
    except Exception as error:
        return _fail(
            "Optional training dependencies are missing: %s. Install "
            "gymnasium and stable-baselines3 inside the Docker runner to "
            "train Phase 4 policies." % error)

    os.makedirs(args.output_dir, exist_ok=True)
    env = _make_env(args)
    feature_names = _observation_feature_names(env)
    _write_training_config(args, feature_names, env)
    normalizer_path = write_normalizer_metadata(
        os.path.join(args.output_dir, "normalizer.json"),
        feature_names=feature_names,
        clip=args.observation_clip,
        action_semantics=ACTION_SEMANTICS,
        max_damper_residual_scale=args.max_damper_residual_scale)

    model = _make_or_load_model(args, env, SAC, TD3)
    rollout_rows = []
    callbacks = [
        _make_rollout_callback(BaseCallback, args.output_dir, rollout_rows)
    ]
    if int(args.save_freq) > 0:
        callbacks.append(_make_checkpoint_callback(
            BaseCallback,
            args.output_dir,
            args.save_freq))
    callback = callbacks[0] if len(callbacks) == 1 else callbacks

    learn_error = ""
    export_error = ""
    backend_lifecycle = {}
    try:
        model.learn(
            total_timesteps=max(0, int(args.total_timesteps)),
            callback=callback)
        sb3_model_path = _save_sb3_model(model, args.output_dir)
        policy_path = _export_policy(
            model=model,
            model_path=sb3_model_path,
            output_dir=args.output_dir,
            observation_dim=len(feature_names))
    except Exception as error:
        learn_error = str(error)
        policy_path = ""
        sb3_model_path = os.path.join(args.output_dir, "sb3_model.zip")
        if not os.path.isfile(sb3_model_path):
            try:
                model.save(os.path.join(args.output_dir, "sb3_model"))
            except Exception:
                pass
        if os.path.isfile(sb3_model_path):
            try:
                policy_path = _export_policy(
                    model=model,
                    model_path=sb3_model_path,
                    output_dir=args.output_dir,
                    observation_dim=len(feature_names))
            except Exception as export_exception:
                export_error = str(export_exception)
    finally:
        backend_lifecycle = _backend_lifecycle_summary(env)
        if hasattr(env, "close"):
            env.close()
        backend_lifecycle.update(_backend_lifecycle_summary(env))

    _ensure_rollout_files(args.output_dir)
    summary = _write_training_summary(
        args=args,
        rollout_rows=rollout_rows,
        feature_names=feature_names,
        normalizer_path=normalizer_path,
        sb3_model_path=sb3_model_path,
        policy_path=policy_path,
        learn_error=learn_error,
        export_error=export_error,
        backend_lifecycle=backend_lifecycle)
    _write_canary_artifacts(args.output_dir, summary)
    if learn_error:
        return _fail("Phase 4 training failed: %s" % learn_error)
    if not policy_path:
        return _fail("Phase 4 policy export failed: %s" % (
            export_error or "policy artifact missing"))
    return 0


def _make_or_load_model(args: argparse.Namespace, env: Any, sac_cls: Any, td3_cls: Any) -> Any:
    algorithm_cls = sac_cls if args.algorithm == "sac" else td3_cls
    resume_arg = str(args.resume_from or "").strip()
    if resume_arg:
        resume_from = os.path.abspath(os.path.expanduser(resume_arg))
        return algorithm_cls.load(resume_from, env=env, seed=args.seed, device="cpu")
    return algorithm_cls("MlpPolicy", env, seed=args.seed, verbose=1, device="cpu")


def _make_env(args: argparse.Namespace) -> Any:
    backend = (
        _make_fake_backend(args)
        if args.backend == "fake"
        else _make_live_backend(args))
    projector_config = ResidualActionProjectorConfig(
        max_damper_residual_scale=args.max_damper_residual_scale)
    return CarlaRouteSuspensionEnv(
        backend=backend,
        baseline=args.baseline,
        planning_provider=args.planning_provider,
        projector_config=projector_config,
        warmup_seconds=args.warmup_seconds,
        max_episode_steps=args.max_episode_steps,
        verify_suspension=int(args.verify_every) > 0,
        verify_every=args.verify_every,
        route_id=args.routes_subset,
        seed=args.seed)


def _make_fake_backend(args: argparse.Namespace) -> FakeCarlaRouteBackend:
    warmup_steps = int(max(0.0, args.warmup_seconds) / 0.05) + 1
    max_steps = max(
        200,
        int(args.total_timesteps) + warmup_steps + max(0, int(args.max_episode_steps)) + 20)
    return FakeCarlaRouteBackend(
        max_steps=max_steps,
        planning_provider=args.planning_provider)


def _make_live_backend(args: argparse.Namespace) -> LiveRouteProcessBackend:
    return LiveRouteProcessBackend(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        routes=args.routes,
        routes_subset=args.routes_subset,
        route_script=args.route_script,
        output_dir=args.output_dir,
        team_agent=args.team_agent,
        team_config=args.team_config,
        baseline=args.baseline,
        planning_provider=args.planning_provider,
        planning_preview_jsonl=args.planning_preview_jsonl,
        traffic_manager_port=args.traffic_manager_port,
        fixed_delta_seconds=args.fixed_delta_seconds,
        seed=args.seed,
        role_name=args.role_name,
        actor_id=args.actor_id,
        verify_suspension=int(args.verify_every) > 0,
        readback_tolerance=args.readback_tolerance,
        connect_retry_seconds=args.connect_retry_seconds,
        hero_timeout_seconds=args.hero_timeout_seconds,
        route_wait_timeout_seconds=args.route_wait_timeout_seconds,
        debug=args.debug)


def _observation_feature_names(env: Any) -> Tuple[str, ...]:
    builder = getattr(env, "observation_builder", None)
    if builder is not None and hasattr(builder, "spec"):
        return tuple(builder.spec().feature_names)
    space = getattr(env, "observation_space", None)
    shape = tuple(getattr(space, "shape", ()) or ())
    count = int(shape[0]) if shape else 0
    return tuple("obs_%03d" % index for index in range(count))


def _make_rollout_callback(base_callback_cls: Any, output_dir: str, rows: list):
    class TrainingRolloutCallback(base_callback_cls):  # type: ignore
        def __init__(self) -> None:
            super().__init__()
            self._logger = RolloutLogger(
                output_dir,
                jsonl_name="training_rollout.jsonl",
                csv_name="training_rollout.csv")

        def _on_training_start(self) -> None:
            self._logger.open()

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", []) or []
            for info in infos:
                row = dict((info or {}).get("rollout_row", {}) or {})
                if row:
                    self._logger.log(row)
                    rows.append(row)
            return True

        def _on_training_end(self) -> None:
            self._logger.close()

    return TrainingRolloutCallback()


def _make_checkpoint_callback(
    base_callback_cls: Any,
    output_dir: str,
    save_freq: int,
):
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    class Phase4CheckpointCallback(base_callback_cls):  # type: ignore
        def _on_training_start(self) -> None:
            os.makedirs(checkpoint_dir, exist_ok=True)

        def _on_step(self) -> bool:
            if int(self.num_timesteps) > 0 and int(self.num_timesteps) % int(save_freq) == 0:
                self.model.save(os.path.join(
                    checkpoint_dir,
                    "phase4_step_%08d" % int(self.num_timesteps)))
            return True

    return Phase4CheckpointCallback()


def _save_sb3_model(model: Any, output_dir: str) -> str:
    model_base = os.path.join(output_dir, "sb3_model")
    model.save(model_base)
    model_path = model_base if model_base.endswith(".zip") else model_base + ".zip"
    if not os.path.isfile(model_path):
        raise RuntimeError("sb3 model save did not create %s" % model_path)
    return model_path


def _export_policy(
    model: Any,
    model_path: str,
    output_dir: str,
    observation_dim: int,
) -> str:
    output_path = os.path.join(output_dir, "policy.ts")
    actor = getattr(getattr(model, "policy", None), "actor", None)
    if actor is not None:
        try:
            return export_torch_module(
                actor,
                output_path=output_path,
                observation_dim=observation_dim,
                action_dim=4)
        except Exception:
            pass
    return export_sb3_actor_to_torchscript(
        model_path=model_path,
        output_path=output_path,
        observation_dim=observation_dim,
        action_dim=4)


def _write_training_config(
    args: argparse.Namespace,
    feature_names: Sequence[str],
    env: Any,
) -> str:
    backend = getattr(env, "backend", None)
    backend_config = (
        backend.config_dict()
        if backend is not None and hasattr(backend, "config_dict")
        else {})
    config = {
        "backend": args.backend,
        "algorithm": args.algorithm,
        "baseline": args.baseline,
        "planning_provider": args.planning_provider,
        "planning_preview_jsonl": args.planning_preview_jsonl,
        "routes": args.routes,
        "routes_subset": args.routes_subset,
        "route_script": args.route_script,
        "seed": int(args.seed),
        "total_timesteps": int(args.total_timesteps),
        "max_episode_steps": int(args.max_episode_steps),
        "observation_feature_names": list(feature_names),
        "action_semantics": ACTION_SEMANTICS,
        "max_damper_residual_scale": float(args.max_damper_residual_scale),
        "spring_frozen": True,
        "warmup_seconds": float(args.warmup_seconds),
        "observation_clip": float(args.observation_clip),
        "host": args.host,
        "port": int(args.port),
        "timeout": float(args.timeout),
        "hero_timeout_seconds": float(args.hero_timeout_seconds),
        "route_wait_timeout_seconds": float(args.route_wait_timeout_seconds),
        "connect_retry_seconds": float(args.connect_retry_seconds),
        "traffic_manager_port": int(args.traffic_manager_port),
        "fixed_delta_seconds": float(args.fixed_delta_seconds),
        "role_name": args.role_name,
        "actor_id": args.actor_id,
        "verify_every": int(args.verify_every),
        "readback_tolerance": float(args.readback_tolerance),
        "resume_from": args.resume_from,
        "save_freq": int(args.save_freq),
        "eval_after_training": bool(args.eval_after_training),
        "eval_route_suite": args.eval_route_suite,
        "fake_backend_used": int(args.backend == "fake"),
        "real_backend_requested": int(args.backend == "live"),
        "synthetic_training": 0,
        "backend_config": backend_config,
    }
    path = os.path.join(args.output_dir, "training_config.json")
    with open(path, "w") as json_file:
        json.dump(config, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    return path


def _write_training_summary(
    args: argparse.Namespace,
    rollout_rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    normalizer_path: str,
    sb3_model_path: str,
    policy_path: str,
    learn_error: str,
    export_error: str,
    backend_lifecycle: Mapping[str, Any] = (),
) -> Dict[str, Any]:
    summary = dict(task_summary_fields(rollout_rows))
    backend_counts = _backend_counts(rollout_rows)
    summary.update({
        "status": "failed" if learn_error or not policy_path else "trained",
        "backend": args.backend,
        "algorithm": args.algorithm,
        "baseline": args.baseline,
        "planning_provider": args.planning_provider,
        "routes": args.routes,
        "routes_subset": args.routes_subset,
        "seed": int(args.seed),
        "total_timesteps": int(args.total_timesteps),
        "max_episode_steps": int(args.max_episode_steps),
        "rollout_rows": len(rollout_rows),
        "observation_dim": len(feature_names),
        "action_dim": 4,
        "action_semantics": ACTION_SEMANTICS,
        "normalizer_path": normalizer_path,
        "sb3_model_path": sb3_model_path,
        "policy_path": policy_path,
        "policy_exported": int(bool(policy_path)),
        "training_rollout_csv": os.path.join(args.output_dir, "training_rollout.csv"),
        "training_rollout_jsonl": os.path.join(args.output_dir, "training_rollout.jsonl"),
        "fake_backend_used": backend_counts.get("fake_backend_used", int(args.backend == "fake")),
        "real_backend_used": backend_counts.get("real_backend_used", int(args.backend == "live")),
        "live_backend_requested": int(args.backend == "live"),
        "synthetic_training": 0,
        "learn_error": learn_error,
        "export_error": export_error,
        "eval_after_training": int(bool(args.eval_after_training)),
        "eval_route_suite": args.eval_route_suite,
        "eval_status": (
            "not_requested"
            if not args.eval_after_training else
            "route_suite_external"
            if args.eval_route_suite else
            "skipped_no_eval_route_suite"),
    })
    summary.update(dict(backend_lifecycle or {}))
    path = os.path.join(args.output_dir, "training_summary.json")
    with open(path, "w") as json_file:
        json.dump(summary, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    return summary


def _write_canary_artifacts(output_dir: str, summary: Mapping[str, Any]) -> None:
    del summary
    from .phase4_acceptance import write_phase4_training_canary

    write_phase4_training_canary(output_dir)


def _ensure_rollout_files(output_dir: str) -> None:
    csv_path = os.path.join(output_dir, "training_rollout.csv")
    jsonl_path = os.path.join(output_dir, "training_rollout.jsonl")
    if not os.path.isfile(csv_path):
        with open(csv_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=DEFAULT_CSV_FIELDS)
            writer.writeheader()
    if not os.path.isfile(jsonl_path):
        open(jsonl_path, "w").close()


def _backend_counts(rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    if not rows:
        return {}
    fake = 1 if any(_as_int(row.get("fake_backend_used", 0)) for row in rows) else 0
    real = 1 if any(_as_int(row.get("real_backend_used", 0)) for row in rows) else 0
    return {
        "fake_backend_used": fake,
        "real_backend_used": real,
    }


def _backend_lifecycle_summary(env: Any) -> Dict[str, Any]:
    backend = getattr(env, "backend", None)
    if backend is None or not hasattr(backend, "lifecycle_summary"):
        return {}
    try:
        summary = backend.lifecycle_summary()
    except Exception as error:
        return {"backend_lifecycle_summary_error": str(error)}
    return dict(summary or {})


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "y", "on"):
        return True
    if normalized in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _fail(message: str, code: int = 1) -> int:
    print(message, file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

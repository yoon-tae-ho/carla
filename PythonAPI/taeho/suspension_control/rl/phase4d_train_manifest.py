"""Manifest writer for Phase 4-D train-only artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


PHASE4D_TRAIN_MANIFEST_FIELDS: Tuple[str, ...] = (
    "artifact_id",
    "workflow_role",
    "artifact_status",
    "phase",
    "source_train_run_dir",
    "train_dir",
    "training_summary_path",
    "phase4_training_canary_path",
    "train_acceptance_path",
    "policy_path",
    "normalizer_path",
    "sb3_model_path",
    "policy_sha256",
    "normalizer_sha256",
    "sb3_model_sha256",
    "backend",
    "algorithm",
    "baseline",
    "planning_provider",
    "routes_subset",
    "train_seed",
    "total_timesteps",
    "action_semantics",
    "observation_dim",
    "action_dim",
    "git_commit",
    "docker_image_id",
    "docker_image_tag",
    "carla_build_commit",
    "created_at",
    "phase4d_train_status",
    "eval_allowed",
    "invalid_for_eval",
    "failed_checks",
    "warnings",
)


def write_phase4d_train_manifest(
    output_dir: str,
    *,
    training_output_dir: str,
    train_acceptance_path: str = "",
    docker_image_id: str = "",
    docker_image_tag: str = "",
    carla_build_commit: str = "",
) -> Tuple[str, str, Dict[str, Any]]:
    output_dir = _abs_path(output_dir)
    train_dir = _abs_path(training_output_dir)
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(train_dir, "training_summary.json")
    canary_path = os.path.join(train_dir, "phase4_training_canary.json")
    config_path = os.path.join(train_dir, "training_config.json")
    acceptance_path = _resolve_path(
        train_acceptance_path or os.path.join(output_dir, "phase4d_train_acceptance.json"),
        output_dir)

    summary = _read_json_dict(summary_path)
    canary = _read_json_dict(canary_path)
    config = _read_json_dict(config_path)
    normalizer_path = _resolve_path(
        _first_nonempty(
            summary.get("normalizer_path"),
            canary.get("normalizer_path"),
            os.path.join(train_dir, "normalizer.json")),
        train_dir)
    normalizer = _read_json_dict(normalizer_path)
    acceptance = _read_json_dict(acceptance_path)

    policy_path = _resolve_path(
        _first_nonempty(
            summary.get("policy_path"),
            canary.get("policy_path"),
            os.path.join(train_dir, "policy.ts")),
        train_dir)
    sb3_model_path = _resolve_path(
        _first_nonempty(
            summary.get("sb3_model_path"),
            os.path.join(train_dir, "sb3_model.zip")),
        train_dir)

    policy_sha = _sha256_file(policy_path)
    normalizer_sha = _sha256_file(normalizer_path)
    sb3_sha = _sha256_file(sb3_model_path)
    eval_allowed = _int_value(acceptance.get("eval_allowed"))
    invalid_for_eval = _int_value(acceptance.get("invalid_for_eval"))
    accepted = eval_allowed == 1 and invalid_for_eval == 0
    total_timesteps = _int_first(
        acceptance,
        canary,
        summary,
        config,
        keys=("total_timesteps_requested", "total_timesteps"))
    seed = _int_first(summary, config, keys=("seed", "train_seed"))
    routes_subset = _first_nonempty(
        summary.get("routes_subset"),
        config.get("routes_subset"),
        acceptance.get("routes_subset"))
    manifest: Dict[str, Any] = {
        "artifact_id": _artifact_id(
            seed=seed,
            routes_subset=routes_subset,
            total_timesteps=total_timesteps,
            policy_sha256=policy_sha,
            normalizer_sha256=normalizer_sha,
            sb3_model_sha256=sb3_sha,
            train_dir=train_dir),
        "workflow_role": "train_only",
        "artifact_status": "accepted_for_eval" if accepted else "invalid_for_eval",
        "phase": "phase4d",
        "source_train_run_dir": train_dir,
        "train_dir": train_dir,
        "training_summary_path": summary_path,
        "phase4_training_canary_path": canary_path,
        "train_acceptance_path": acceptance_path,
        "policy_path": policy_path,
        "normalizer_path": normalizer_path,
        "sb3_model_path": sb3_model_path,
        "policy_sha256": policy_sha,
        "normalizer_sha256": normalizer_sha,
        "sb3_model_sha256": sb3_sha,
        "backend": _first_nonempty(summary.get("backend"), config.get("backend")),
        "algorithm": _first_nonempty(summary.get("algorithm"), config.get("algorithm")),
        "baseline": _first_nonempty(summary.get("baseline"), config.get("baseline")),
        "planning_provider": _first_nonempty(
            summary.get("planning_provider"),
            config.get("planning_provider")),
        "routes_subset": routes_subset,
        "train_seed": seed,
        "total_timesteps": total_timesteps,
        "action_semantics": _first_nonempty(
            summary.get("action_semantics"),
            config.get("action_semantics"),
            normalizer.get("action_semantics")),
        "observation_dim": _int_first(
            summary,
            normalizer,
            keys=("observation_dim", "observation_size")),
        "action_dim": _int_first(summary, keys=("action_dim",)),
        "git_commit": _git_commit(),
        "docker_image_id": _first_nonempty(
            docker_image_id,
            os.environ.get("PHASE4D_DOCKER_IMAGE_ID"),
            os.environ.get("DOCKER_IMAGE_ID"),
            "unknown"),
        "docker_image_tag": _first_nonempty(
            docker_image_tag,
            os.environ.get("PHASE4D_DOCKER_IMAGE_TAG"),
            os.environ.get("DOCKER_IMAGE_TAG"),
            "unknown"),
        "carla_build_commit": _first_nonempty(
            carla_build_commit,
            os.environ.get("CARLA_BUILD_COMMIT"),
            "unknown"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase4d_train_status": acceptance.get("phase4d_train_status", ""),
        "eval_allowed": eval_allowed,
        "invalid_for_eval": invalid_for_eval,
        "failed_checks": acceptance.get("failed_checks", ""),
        "warnings": acceptance.get("warnings", ""),
    }

    if manifest["observation_dim"] <= 0:
        feature_names = normalizer.get("feature_names") or config.get("observation_feature_names") or []
        if isinstance(feature_names, list):
            manifest["observation_dim"] = len(feature_names)

    ordered = _ordered_manifest(manifest)
    json_path = os.path.join(output_dir, "phase4d_train_manifest.json")
    csv_path = os.path.join(output_dir, "phase4d_train_manifest.csv")
    _write_json(json_path, ordered)
    _write_single_row_csv(csv_path, ordered)
    return json_path, csv_path, ordered


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-output-dir", required=True)
    parser.add_argument("--train-acceptance", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--docker-image-id", default="")
    parser.add_argument("--docker-image-tag", default="")
    parser.add_argument("--carla-build-commit", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    json_path, csv_path, manifest = write_phase4d_train_manifest(
        args.output_dir,
        training_output_dir=args.training_output_dir,
        train_acceptance_path=args.train_acceptance,
        docker_image_id=args.docker_image_id,
        docker_image_tag=args.docker_image_tag,
        carla_build_commit=args.carla_build_commit)
    print("phase4d train manifest: %s" % json_path)
    print("phase4d train manifest csv: %s" % csv_path)
    print("phase4d train artifact: %s status=%s eval_allowed=%s" % (
        manifest.get("artifact_id"),
        manifest.get("artifact_status"),
        manifest.get("eval_allowed")))
    return 0


def _artifact_id(
    *,
    seed: int,
    routes_subset: str,
    total_timesteps: int,
    policy_sha256: str,
    normalizer_sha256: str,
    sb3_model_sha256: str,
    train_dir: str,
) -> str:
    identity = "|".join((
        str(seed),
        str(routes_subset),
        str(total_timesteps),
        policy_sha256,
        normalizer_sha256,
        sb3_model_sha256,
        train_dir,
    ))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    route_label = _sanitize_label(routes_subset or "unknown")
    return "phase4d-train-seed%s-route%s-steps%s-%s" % (
        seed,
        route_label,
        total_timesteps,
        digest)


def _sha256_file(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL)
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path or ".")))


def _resolve_path(path: Any, base_dir: str) -> str:
    value = os.path.expanduser(str(path or ""))
    if not value:
        return ""
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(base_dir, value))


def _read_json_dict(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path) as json_file:
            value = json.load(json_file)
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: str, value: Mapping[str, Any]) -> None:
    with open(path, "w") as json_file:
        json.dump(value, json_file, indent=2, sort_keys=True)
        json_file.write("\n")


def _write_single_row_csv(path: str, row: Mapping[str, Any]) -> None:
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PHASE4D_TRAIN_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in PHASE4D_TRAIN_MANIFEST_FIELDS})


def _ordered_manifest(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: row.get(key, "") for key in PHASE4D_TRAIN_MANIFEST_FIELDS}


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _int_first(*sources: Mapping[str, Any], keys: Sequence[str]) -> int:
    for key in keys:
        for source in sources:
            if key in source:
                return _int_value(source.get(key))
    return 0


def _int_value(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _sanitize_label(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))


if __name__ == "__main__":
    raise SystemExit(main())

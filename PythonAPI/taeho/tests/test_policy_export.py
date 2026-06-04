import contextlib
import importlib
import io
import json
import math
import os
import sys

from suspension_control.rl import train_sac
from suspension_control.rl.policy import PolicyAdapter
from suspension_control.rl.policy_export import (
    export_torch_module,
    write_normalizer_metadata,
)


def test_policy_export_import_does_not_require_stable_baselines3():
    sys.modules.pop("suspension_control.rl.policy_export", None)
    module = importlib.import_module("suspension_control.rl.policy_export")
    assert hasattr(module, "export_sb3_actor_to_torchscript")
    assert hasattr(module, "export_torch_module")


def test_train_sac_rejects_synthetic_without_explicit_opt_in(tmp_path):
    code, stderr = _run_train([
        "--env",
        "synthetic",
        "--total-timesteps",
        "1",
        "--output-dir",
        str(tmp_path / "out"),
    ])
    assert code == 2
    assert "synthetic training requires --allow-synthetic-training" in stderr


def test_train_sac_carla_fails_with_live_backend_unavailable(tmp_path):
    output_dir = tmp_path / "carla_out"
    previous = os.environ.get("PHASE4_ROUTE_SCRIPT")
    os.environ["PHASE4_ROUTE_SCRIPT"] = str(tmp_path / "missing_route.sh")
    try:
        code, stderr = _run_train([
            "--env",
            "carla",
            "--total-timesteps",
            "1",
            "--output-dir",
            str(output_dir),
        ])
    finally:
        if previous is None:
            os.environ.pop("PHASE4_ROUTE_SCRIPT", None)
        else:
            os.environ["PHASE4_ROUTE_SCRIPT"] = previous
    assert code == 2
    assert "LiveRouteProcessBackend dependencies are unavailable" in stderr
    assert (
        "route runner script missing" in stderr or
        "srunner import failed" in stderr or
        "leaderboard import failed" in stderr)
    assert not output_dir.exists()


def test_simple_torchscript_actor_export_loads_through_policy_adapter(tmp_path):
    try:
        import torch
    except Exception:
        try:
            export_torch_module(object(), str(tmp_path / "actor.ts"), 3, 4)
        except RuntimeError as error:
            assert "torch is required" in str(error)
            return
        raise AssertionError("export_torch_module should require torch")

    actor = torch.nn.Linear(3, 4)
    actor_path = export_torch_module(
        actor,
        output_path=str(tmp_path / "actor.ts"),
        observation_dim=3,
        action_dim=4)

    loaded = torch.jit.load(actor_path)
    assert tuple(loaded(torch.zeros(3)).shape) == (4,)
    assert tuple(loaded(torch.zeros((2, 3))).shape) == (2, 4)

    policy = PolicyAdapter(policy_path=actor_path, action_dim=4)
    action = policy.predict([0.1, -0.2, 0.3])
    assert len(action) == 4
    assert all(math.isfinite(value) for value in action)
    assert all(-1.0 <= value <= 1.0 for value in action)


def test_normalizer_metadata_writer_contains_deploy_contract(tmp_path):
    path = write_normalizer_metadata(
        str(tmp_path / "normalizer.json"),
        feature_names=("speed", "roll", "previous_action_fl"),
        clip=5.0,
        action_semantics="normalized_damper_residual_v1",
        max_damper_residual_scale=0.08)

    with open(path) as json_file:
        data = json.load(json_file)

    assert data["feature_names"] == ["speed", "roll", "previous_action_fl"]
    assert data["clip"] == 5.0
    assert data["action_semantics"] == "normalized_damper_residual_v1"
    assert data["max_damper_residual_scale"] == 0.08
    assert data["spring_frozen"] is True


def _run_train(argv):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        code = train_sac.main(argv)
    return code, stderr.getvalue()

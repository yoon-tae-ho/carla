import contextlib
import io
import os
import subprocess
from pathlib import Path

from suspension_control.rl import train_sac
from suspension_control.rl.carla_backend import (
    LIVE_BACKEND_UNAVAILABLE_MESSAGE,
    LiveCarlaSuspensionBackend,
)


DOCKER_WRAPPER = Path("/home/yth/sim/docker/run_rl_suspension_train_carla.sh")


def test_live_backend_reset_raises_clear_dependency_error():
    backend = LiveCarlaSuspensionBackend(
        host="127.0.0.1",
        port=2000,
        timeout=5.0,
        routes="/workspace/routes.xml",
        routes_subset="00",
        team_agent="/workspace/agent.py",
        team_config="/workspace/checkpoint",
        baseline="skyhook",
        planning_provider="empty",
        traffic_manager_port=8000,
        fixed_delta_seconds=0.05,
        seed=100,
        role_name="hero",
        actor_id="",
        verify_suspension=True,
        carla_importer=lambda: (_ for _ in ()).throw(
            ImportError("carla missing")),
        required_route_modules=())

    assert backend.config_dict()["routes_subset"] == "00"
    assert backend.config_dict()["verify_suspension"] is True

    try:
        backend.reset(seed=100, route_id="00")
    except RuntimeError as error:
        assert LIVE_BACKEND_UNAVAILABLE_MESSAGE in str(error)
        assert "carla import failed" in str(error)
        assert "must not fall back to synthetic training" in str(error)
        assert "--env carla training" in str(error)
    else:
        raise AssertionError("live backend should fail without dependencies")


def test_train_sac_carla_propagates_live_backend_unavailable(tmp_path):
    stderr = io.StringIO()
    previous = os.environ.get("PHASE4_ROUTE_SCRIPT")
    os.environ["PHASE4_ROUTE_SCRIPT"] = str(tmp_path / "missing_route.sh")
    try:
        with contextlib.redirect_stderr(stderr):
            code = train_sac.main([
                "--env",
                "carla",
                "--routes-subset",
                "00",
                "--seed",
                "100",
                "--output-dir",
                str(tmp_path / "out"),
            ])
    finally:
        if previous is None:
            os.environ.pop("PHASE4_ROUTE_SCRIPT", None)
        else:
            os.environ["PHASE4_ROUTE_SCRIPT"] = previous

    assert code == 2
    message = stderr.getvalue()
    assert "LiveRouteProcessBackend dependencies are unavailable" in message
    assert (
        "route runner script missing" in message or
        "srunner import failed" in message or
        "leaderboard import failed" in message)
    assert "must not fall back to synthetic training" in message
    assert not (tmp_path / "out").exists()


def test_docker_training_wrapper_help_prints_usage():
    result = subprocess.run(
        ["bash", str(DOCKER_WRAPPER), "--help"],
        cwd="/home/yth/sim/docker",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False)

    assert result.returncode == 0
    assert "run_rl_suspension_train_carla.sh" in result.stdout
    assert "--env carla" in result.stdout
    assert "--allow-synthetic-training" in result.stdout
    assert result.stderr == ""


def test_docker_training_wrapper_translates_host_paths():
    result = subprocess.run(
        [
            "bash",
            str(DOCKER_WRAPPER),
            "--translate-path",
            "/home/yth/sim/e2e_models/outputs/example",
        ],
        cwd="/home/yth/sim/docker",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False)

    assert result.returncode == 0
    assert result.stdout.strip() == "/workspace/e2e_models/outputs/example"
    assert result.stderr == ""


def test_docker_training_wrapper_rejects_synthetic_env():
    result = subprocess.run(
        ["bash", str(DOCKER_WRAPPER), "--env", "synthetic"],
        cwd="/home/yth/sim/docker",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False)

    assert result.returncode == 2
    assert "only supports --env carla" in result.stderr

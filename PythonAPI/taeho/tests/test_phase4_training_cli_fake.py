import contextlib
import io
import json
import os

from suspension_control.rl import train_real_carla_sac


def test_phase4_training_cli_fake_either_trains_or_fails_on_missing_deps(tmp_path):
    output_dir = tmp_path / "fake_train"
    code, stderr = _run_train([
        "--backend",
        "fake",
        "--total-timesteps",
        "8",
        "--max-episode-steps",
        "4",
        "--output-dir",
        str(output_dir),
    ])

    assert "fake_backend_used=1" in stderr
    if code != 0:
        assert "Optional training dependencies are missing" in stderr
        return

    expected = (
        "sb3_model.zip",
        "policy.ts",
        "normalizer.json",
        "training_rollout.csv",
        "training_rollout.jsonl",
        "training_summary.json",
        "phase4_training_canary.csv",
        "phase4_training_canary.json",
    )
    for filename in expected:
        assert (output_dir / filename).is_file()

    with open(output_dir / "training_summary.json") as json_file:
        summary = json.load(json_file)
    assert summary["backend"] == "fake"
    assert summary["fake_backend_used"] == 1
    assert summary["synthetic_training"] == 0
    assert summary["policy_exported"] == 1
    assert summary["rollout_rows"] >= 1


def test_phase4_training_cli_live_fails_before_optional_training_import(tmp_path):
    output_dir = tmp_path / "live_train"
    previous = os.environ.get("PHASE4_ROUTE_SCRIPT")
    os.environ["PHASE4_ROUTE_SCRIPT"] = str(tmp_path / "missing_route.sh")
    try:
        code, stderr = _run_train([
            "--backend",
            "live",
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
    assert "route runner script missing" in stderr
    assert not output_dir.exists()


def _run_train(argv):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        code = train_real_carla_sac.main(argv)
    return code, stderr.getvalue()

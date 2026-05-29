#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m py_compile \
  suspension_control/controllers/base.py \
  suspension_control/controllers/rl_residual.py \
  suspension_control/rl/observations.py \
  suspension_control/rl/reward.py \
  suspension_control/rl/policy.py \
  suspension_control/runtime/planning_provider.py \
  suspension_control/rl/evaluate_policy.py \
  transfuser_suspension_control_suite.py

if python -c "import pytest" >/dev/null 2>&1; then
  python -m pytest -q \
    tests/test_rl_residual_controller.py \
    tests/test_rl_observations.py \
    tests/test_rl_reward.py \
    tests/test_planning_provider.py \
    tests/test_rl_policy_fallback.py
else
  python - <<'PY'
import importlib
import inspect
import tempfile
from pathlib import Path

modules = (
    "tests.test_rl_residual_controller",
    "tests.test_rl_observations",
    "tests.test_rl_reward",
    "tests.test_planning_provider",
    "tests.test_rl_policy_fallback",
)

count = 0
for module_name in modules:
    module = importlib.import_module(module_name)
    for name, function in sorted(vars(module).items()):
        if not name.startswith("test_") or not callable(function):
            continue
        kwargs = {}
        signature = inspect.signature(function)
        with tempfile.TemporaryDirectory() as temp_dir:
            if "tmp_path" in signature.parameters:
                kwargs["tmp_path"] = Path(temp_dir)
            function(**kwargs)
        count += 1
print("pytest is not installed; ran %d pytest-style tests with stdlib fallback" % count)
PY
fi

python -m suspension_control.rl.evaluate_policy \
  --dry-run \
  --steps 100 \
  --output-dir /tmp/rl_suspension_dry_run

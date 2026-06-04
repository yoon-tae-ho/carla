#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! python - <<'PY' >/dev/null 2>&1
import carla
import srunner
PY
then
  cat >&2 <<'EOF'
Phase 2-C online canary routes need the Docker runner environment.

Run from the host with:
  cd ~/sim/docker
  bash run_phase2c_nonzero_action.sh [suite args]

This in-tree script is intended to run inside the runner container where both
the custom CARLA Python wheel and scenario_runner/srunner are importable.
EOF
  exit 2
fi

python transfuser_suspension_control_suite.py \
  --scenarios rl_zero_residual_skyhook,rl_const_action_plus_0p25_skyhook,rl_const_action_minus_0p25_skyhook,rl_random_action_0p10_skyhook \
  --metric-warmup-seconds 3.0 \
  --verify-every 50 \
  --require-verification \
  --planning-provider empty \
  "$@"

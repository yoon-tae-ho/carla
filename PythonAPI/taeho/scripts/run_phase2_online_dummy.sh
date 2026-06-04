#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if ! python - <<'PY' >/dev/null 2>&1
import carla
import srunner
PY
then
  cat >&2 <<'EOF'
Phase 2 online dummy routes need the Docker runner environment.

Run from the host with:
  cd ~/sim/docker
  bash run_phase2_online_dummy.sh [suite args]

This in-tree script is intended to run inside the runner container where both
the custom CARLA Python wheel and scenario_runner/srunner are importable.
EOF
  exit 2
fi

python transfuser_suspension_control_suite.py \
  --scenarios stock,identity,pid,skyhook,rl_zero_residual_pid,rl_zero_residual_skyhook \
  --metric-warmup-seconds 3.0 \
  --verify-every 50 \
  --require-verification \
  --planning-provider empty \
  "$@"

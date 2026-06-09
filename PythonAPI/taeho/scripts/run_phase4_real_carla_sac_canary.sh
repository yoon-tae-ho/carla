#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

show_help() {
  cat <<'EOF'
Run Phase 4 real-CARLA SAC canary training.

This in-tree script is intended to run inside the Docker runner container.
From the host, prefer:
  cd ~/sim/docker
  bash run_phase4_real_carla_sac_canary.sh [training args]

Defaults:
  backend=live
  algorithm=sac
  total_timesteps=512
  max_episode_steps=0
  seed=100
  routes_subset=00
  planning_provider=empty
  verify_every=0
  eval_after_training=false
  timeout=120
  hero_timeout_seconds=900
  route_wait_timeout_seconds=120
  connect_retry_seconds=1.0
EOF
}

case "${1:-}" in
  -h|--help)
    show_help
    exit 0
    ;;
esac

if ! python - <<'PY' >/dev/null 2>&1
import carla
import gymnasium
import stable_baselines3
import srunner
PY
then
  cat >&2 <<'EOF'
Phase 4 real-CARLA SAC canary needs the Docker runner environment with
custom CARLA, scenario_runner/srunner, gymnasium, and stable-baselines3.

Run from the host with:
  cd ~/sim/docker
  bash run_phase4_real_carla_sac_canary.sh [training args]
EOF
  exit 2
fi

python -m suspension_control.rl.train_real_carla_sac \
  --backend live \
  --algorithm sac \
  --total-timesteps 512 \
  --max-episode-steps 0 \
  --seed 100 \
  --routes-subset 00 \
  --planning-provider empty \
  --verify-every 0 \
  --timeout 120 \
  --hero-timeout-seconds 900 \
  --route-wait-timeout-seconds 120 \
  --connect-retry-seconds 1.0 \
  --eval-after-training false \
  "$@"

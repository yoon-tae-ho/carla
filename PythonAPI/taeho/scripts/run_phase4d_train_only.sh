#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"

show_help() {
  cat <<'EOF'
Run Phase 4-D train-only live SAC and write accepted artifact metadata.

This in-tree script is intended to run inside the Docker runner container
after env_garage_2.sh has been sourced.

Usage:
  bash scripts/run_phase4d_train_only.sh \
    --seed 100 \
    --routes-subset 00 \
    --total-timesteps 1024 \
    --output-root /workspace/e2e_models/outputs/rl_suspension_phase4d_train_only

Stable live-training settings forced by this runner:
  --backend live
  --algorithm sac
  --baseline skyhook
  --planning-provider empty
  --total-timesteps 1024 by default
  --max-episode-steps 0
  --verify-every 0
  --eval-after-training false
  --timeout 120
  --hero-timeout-seconds 900
  --route-wait-timeout-seconds 120
  --connect-retry-seconds 1.0

Outputs:
  <output-root>/train/
  <output-root>/phase4d_train_acceptance.csv
  <output-root>/phase4d_train_acceptance.json
  <output-root>/phase4d_train_manifest.csv
  <output-root>/phase4d_train_manifest.json

Use --print-command to validate command assembly without running CARLA.
EOF
}

quote_cmd() {
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

sanitize_label() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9' '_'
}

phase4d_train_gate_allows_eval() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys

try:
    with open(sys.argv[1]) as json_file:
        row = json.load(json_file)
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if int(float(row.get("eval_allowed", 0) or 0)) == 1 else 1)
PY
}

seed="100"
routes_subset="00"
total_timesteps="1024"
total_timesteps_explicit=0
output_root=""
print_command=0
extra_train_args=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    --print-command)
      print_command=1
      shift
      ;;
    --seed=*)
      seed="${1#--seed=}"
      shift
      ;;
    --seed)
      if [ "$#" -lt 2 ]; then echo "missing value for --seed" >&2; exit 2; fi
      seed="$2"
      shift 2
      ;;
    --routes-subset=*)
      routes_subset="${1#--routes-subset=}"
      shift
      ;;
    --routes-subset)
      if [ "$#" -lt 2 ]; then echo "missing value for --routes-subset" >&2; exit 2; fi
      routes_subset="$2"
      shift 2
      ;;
    --total-timesteps=*)
      total_timesteps="${1#--total-timesteps=}"
      total_timesteps_explicit=1
      shift
      ;;
    --total-timesteps)
      if [ "$#" -lt 2 ]; then echo "missing value for --total-timesteps" >&2; exit 2; fi
      total_timesteps="$2"
      total_timesteps_explicit=1
      shift 2
      ;;
    --output-root=*)
      output_root="${1#--output-root=}"
      shift
      ;;
    --output-root)
      if [ "$#" -lt 2 ]; then echo "missing value for --output-root" >&2; exit 2; fi
      output_root="$2"
      shift 2
      ;;
    *)
      extra_train_args+=("$1")
      shift
      ;;
  esac
done

if [ "$total_timesteps_explicit" -eq 1 ] && [ "$total_timesteps" -gt 4096 ] 2>/dev/null; then
  echo "WARNING: Phase 4-D route00 training above 4096 steps needs multi-episode reset support." >&2
fi

if [ -z "$output_root" ]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  seed_label="$(sanitize_label "$seed")"
  route_label="$(sanitize_label "$routes_subset")"
  steps_label="$(sanitize_label "$total_timesteps")"
  output_root="/workspace/e2e_models/outputs/rl_suspension_phase4d_train_only_${steps_label}_seed${seed_label}_route${route_label}_${stamp}"
fi

train_dir="${output_root%/}/train"
acceptance_json="${output_root%/}/phase4d_train_acceptance.json"

train_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.train_real_carla_sac
  "${extra_train_args[@]}"
  --backend live
  --algorithm sac
  --baseline skyhook
  --total-timesteps "$total_timesteps"
  --max-episode-steps 0
  --seed "$seed"
  --routes-subset "$routes_subset"
  --planning-provider empty
  --verify-every 0
  --timeout 120
  --hero-timeout-seconds 900
  --route-wait-timeout-seconds 120
  --connect-retry-seconds 1.0
  --eval-after-training false
  --output-dir "$train_dir"
)

train_gate_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.phase4d_train_acceptance
  --training-output-dir "$train_dir"
  --output-dir "$output_root"
)

manifest_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.phase4d_train_manifest
  --training-output-dir "$train_dir"
  --train-acceptance "$acceptance_json"
  --output-dir "$output_root"
)

if [ "$print_command" -eq 1 ]; then
  echo "Phase 4-D train-only command:"
  printf 'TRAIN:'
  quote_cmd "${train_cmd[@]}"
  echo "Phase 4-D train acceptance command:"
  printf 'TRAIN_ACCEPTANCE:'
  quote_cmd "${train_gate_cmd[@]}" --training-return-code "<train_return_code>"
  echo "Phase 4-D train manifest command:"
  printf 'TRAIN_MANIFEST:'
  quote_cmd "${manifest_cmd[@]}"
  echo "Phase 4-D output root: $output_root"
  echo "Phase 4-D train dir: $train_dir"
  exit 0
fi

mkdir -p "$train_dir" "$output_root"
set +e
"${train_cmd[@]}"
train_return_code=$?
set -e
"${train_gate_cmd[@]}" --training-return-code "$train_return_code"
"${manifest_cmd[@]}"

if ! phase4d_train_gate_allows_eval "$acceptance_json"; then
  echo "Phase 4-D train-only gate failed; artifact is invalid_for_eval." >&2
  echo "Phase 4-D train acceptance JSON: $acceptance_json" >&2
  echo "Phase 4-D train manifest JSON: ${output_root%/}/phase4d_train_manifest.json" >&2
  if [ "$train_return_code" -ne 0 ]; then
    exit "$train_return_code"
  fi
  exit 1
fi

echo "Phase 4-D train-only status: accepted_for_eval"
echo "Phase 4-D output root: $output_root"
echo "Phase 4-D train dir: $train_dir"
echo "Phase 4-D train acceptance CSV: ${output_root%/}/phase4d_train_acceptance.csv"
echo "Phase 4-D train acceptance JSON: $acceptance_json"
echo "Phase 4-D train manifest JSON: ${output_root%/}/phase4d_train_manifest.json"

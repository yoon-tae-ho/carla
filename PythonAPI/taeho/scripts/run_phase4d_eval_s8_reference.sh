#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"

show_help() {
  cat <<'EOF'
Run Phase 4-D S8 zero-residual reference eval.

This in-tree script is intended to run inside the Docker runner container
after env_garage_2.sh has been sourced.

Usage:
  bash scripts/run_phase4d_eval_s8_reference.sh \
    --seed 100 \
    --routes-subset 00 \
    --output-dir /workspace/e2e_models/outputs/rl_suspension_phase4d_eval_s8_reference

Forced eval settings:
  --scenarios rl_zero_residual_skyhook
  --planning-provider empty
  --metric-warmup-seconds 3.0
  --timeout 120
  --tick-wait-timeout 120
  --sidecar-tick-mode poll
  --poll-seconds 0.05
  --verify-every 50
  --require-verification

This runner does not load a learned policy and does not run SAC training.
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

seed="100"
routes_subset="00"
output_dir=""
print_command=0
extra_args=()

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
    --output-dir=*)
      output_dir="${1#--output-dir=}"
      shift
      ;;
    --output-dir)
      if [ "$#" -lt 2 ]; then echo "missing value for --output-dir" >&2; exit 2; fi
      output_dir="$2"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [ -z "$output_dir" ]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  seed_label="$(sanitize_label "$seed")"
  route_label="$(sanitize_label "$routes_subset")"
  output_dir="/workspace/e2e_models/outputs/rl_suspension_phase4d_eval_s8_reference_seed${seed_label}_route${route_label}_${stamp}"
fi

route_cmd=(
  "$PYTHON_BIN" transfuser_suspension_control_suite.py
  --scenarios rl_zero_residual_skyhook
  --metric-warmup-seconds 3.0
  --timeout 120
  --tick-wait-timeout 120
  --sidecar-tick-mode poll
  --poll-seconds 0.05
  --verify-every 50
  --require-verification
  --planning-provider empty
  --routes-subset "$routes_subset"
  --seeds "$seed"
  --output-dir "$output_dir"
)
route_cmd+=("${extra_args[@]}")

accept_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.phase4d_eval_reference
  --output-dir "$output_dir"
  --suite-summary "$output_dir/suite_summary.csv"
  --routes-subset "$routes_subset"
  --seed "$seed"
)

if [ "$print_command" -eq 1 ]; then
  echo "Phase 4-D S8 reference command:"
  printf 'REFERENCE_S8:'
  quote_cmd "${route_cmd[@]}"
  echo "Phase 4-D S8 acceptance command:"
  printf 'ACCEPTANCE:'
  quote_cmd "${accept_cmd[@]}" --route-return-code "<route_return_code>" --fail-on-reject
  echo "Phase 4-D S8 reference output dir: $output_dir"
  echo "Phase 4-D S8 reference suite summary: $output_dir/suite_summary.csv"
  echo "Phase 4-D S8 reference scenario summary: $output_dir/seed${seed}_S8_rl_zero_residual_skyhook/summary.json"
  exit 0
fi

mkdir -p "$output_dir"
set +e
"${route_cmd[@]}"
route_return_code=$?
set -e
"${accept_cmd[@]}" --route-return-code "$route_return_code" --fail-on-reject

echo "Phase 4-D S8 reference output dir: $output_dir"
echo "Phase 4-D S8 reference suite summary CSV: $output_dir/suite_summary.csv"
echo "Phase 4-D S8 reference suite summary JSON: $output_dir/suite_summary.json"
echo "Phase 4-D S8 reference acceptance JSON: $output_dir/phase4d_eval_s8_acceptance.json"
echo "Phase 4-D S8 reference manifest JSON: $output_dir/phase4d_eval_s8_manifest.json"
echo "Phase 4-D S8 reference scenario summary: $output_dir/seed${seed}_S8_rl_zero_residual_skyhook/summary.json"

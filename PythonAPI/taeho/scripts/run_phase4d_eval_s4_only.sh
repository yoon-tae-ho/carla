#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"

show_help() {
  cat <<'EOF'
Run Phase 4-D S4-only eval from an accepted train artifact.

This in-tree script is intended to run inside the Docker runner container
after env_garage_2.sh has been sourced.

Usage:
  bash scripts/run_phase4d_eval_s4_only.sh \
    --train-dir /workspace/e2e_models/outputs/<TRAIN_RUN>/train \
    --output-dir /workspace/e2e_models/outputs/<S4_EVAL_RUN>

The runner preflights phase4d_train_acceptance.json and
phase4d_train_manifest.json before launching CARLA. Invalid train artifacts
write phase4d_eval_s4_acceptance.json with
eval_status=skipped_due_to_invalid_train_artifact and do not run CARLA.

Forced eval settings:
  --scenarios rl_residual_skyhook
  --planning-provider empty
  --metric-warmup-seconds 3.0
  --timeout 120
  --tick-wait-timeout 120
  --sidecar-tick-mode poll
  --poll-seconds 0.05
  --verify-every 50
  --require-verification

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

train_dir=""
train_acceptance=""
train_manifest=""
training_summary=""
policy_path=""
normalizer_path=""
routes_subset=""
seed=""
output_dir=""
print_command=0
extra_route_args=()

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
    --train-dir=*)
      train_dir="${1#--train-dir=}"
      shift
      ;;
    --train-dir)
      if [ "$#" -lt 2 ]; then echo "missing value for --train-dir" >&2; exit 2; fi
      train_dir="$2"
      shift 2
      ;;
    --train-acceptance=*)
      train_acceptance="${1#--train-acceptance=}"
      shift
      ;;
    --train-acceptance)
      if [ "$#" -lt 2 ]; then echo "missing value for --train-acceptance" >&2; exit 2; fi
      train_acceptance="$2"
      shift 2
      ;;
    --train-manifest=*)
      train_manifest="${1#--train-manifest=}"
      shift
      ;;
    --train-manifest)
      if [ "$#" -lt 2 ]; then echo "missing value for --train-manifest" >&2; exit 2; fi
      train_manifest="$2"
      shift 2
      ;;
    --training-summary=*)
      training_summary="${1#--training-summary=}"
      shift
      ;;
    --training-summary)
      if [ "$#" -lt 2 ]; then echo "missing value for --training-summary" >&2; exit 2; fi
      training_summary="$2"
      shift 2
      ;;
    --policy-path=*|--rl-policy=*)
      policy_path="${1#*=}"
      shift
      ;;
    --policy-path|--rl-policy)
      if [ "$#" -lt 2 ]; then echo "missing value for $1" >&2; exit 2; fi
      policy_path="$2"
      shift 2
      ;;
    --normalizer-path=*|--rl-normalizer=*)
      normalizer_path="${1#*=}"
      shift
      ;;
    --normalizer-path|--rl-normalizer)
      if [ "$#" -lt 2 ]; then echo "missing value for $1" >&2; exit 2; fi
      normalizer_path="$2"
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
    --seed=*|--seeds=*)
      seed="${1#*=}"
      shift
      ;;
    --seed|--seeds)
      if [ "$#" -lt 2 ]; then echo "missing value for $1" >&2; exit 2; fi
      seed="$2"
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
      extra_route_args+=("$1")
      shift
      ;;
  esac
done

if [ -z "$train_dir" ]; then
  echo "Phase 4-D S4 eval requires --train-dir" >&2
  exit 2
fi
if [ -z "$output_dir" ]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  route_label="$(sanitize_label "${routes_subset:-artifact}")"
  seed_label="$(sanitize_label "${seed:-artifact}")"
  output_dir="/workspace/e2e_models/outputs/rl_suspension_phase4d_eval_s4_only_seed${seed_label}_route${route_label}_${stamp}"
fi

prepare_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.phase4d_eval_artifact
  --action prepare
  --train-dir "$train_dir"
  --output-dir "$output_dir"
)
accept_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.phase4d_eval_artifact
  --action accept
  --train-dir "$train_dir"
  --output-dir "$output_dir"
  --suite-summary "$output_dir/suite_summary.csv"
)
if [ -n "$train_acceptance" ]; then
  prepare_cmd+=(--train-acceptance "$train_acceptance")
  accept_cmd+=(--train-acceptance "$train_acceptance")
fi
if [ -n "$train_manifest" ]; then
  prepare_cmd+=(--train-manifest "$train_manifest")
  accept_cmd+=(--train-manifest "$train_manifest")
fi
if [ -n "$training_summary" ]; then
  prepare_cmd+=(--training-summary "$training_summary")
  accept_cmd+=(--training-summary "$training_summary")
fi
if [ -n "$policy_path" ]; then
  prepare_cmd+=(--policy-path "$policy_path")
  accept_cmd+=(--policy-path "$policy_path")
fi
if [ -n "$normalizer_path" ]; then
  prepare_cmd+=(--normalizer-path "$normalizer_path")
  accept_cmd+=(--normalizer-path "$normalizer_path")
fi
if [ -n "$routes_subset" ]; then
  prepare_cmd+=(--routes-subset "$routes_subset")
  accept_cmd+=(--routes-subset "$routes_subset")
fi
if [ -n "$seed" ]; then
  prepare_cmd+=(--seed "$seed")
  accept_cmd+=(--seed "$seed")
fi

print_policy_path="${policy_path:-${train_dir%/}/policy.ts}"
print_normalizer_path="${normalizer_path:-${train_dir%/}/normalizer.json}"
print_routes_subset="${routes_subset:-<artifact_routes_subset>}"
print_seed="${seed:-<artifact_seed>}"

route_cmd=(
  "$PYTHON_BIN" transfuser_suspension_control_suite.py
  --scenarios rl_residual_skyhook
  --rl-policy "$print_policy_path"
  --rl-normalizer "$print_normalizer_path"
  --metric-warmup-seconds 3.0
  --timeout 120
  --tick-wait-timeout 120
  --sidecar-tick-mode poll
  --poll-seconds 0.05
  --verify-every 50
  --require-verification
  --planning-provider empty
  --routes-subset "$print_routes_subset"
  --seeds "$print_seed"
  --output-dir "$output_dir"
)
route_cmd+=("${extra_route_args[@]}")

if [ "$print_command" -eq 1 ]; then
  echo "Phase 4-D S4 eval preflight command:"
  printf 'PREFLIGHT:'
  quote_cmd "${prepare_cmd[@]}" --print-shell
  echo "Phase 4-D S4 route command:"
  printf 'EVAL_S4:'
  quote_cmd "${route_cmd[@]}"
  echo "Phase 4-D S4 acceptance command:"
  printf 'ACCEPTANCE:'
  quote_cmd "${accept_cmd[@]}" --route-return-code "<route_return_code>" --fail-on-reject
  echo "Phase 4-D S4 output dir: $output_dir"
  exit 0
fi

mkdir -p "$output_dir"
eval "$("${prepare_cmd[@]}" --print-shell)"
if [ "${PHASE4D_PREFLIGHT_OK:-0}" != "1" ]; then
  echo "Phase 4-D S4 eval skipped: invalid train artifact." >&2
  echo "Failed checks: ${PHASE4D_FAILED_CHECKS:-}" >&2
  echo "Acceptance JSON: $output_dir/phase4d_eval_s4_acceptance.json" >&2
  exit 1
fi

route_cmd=(
  "$PYTHON_BIN" transfuser_suspension_control_suite.py
  --scenarios rl_residual_skyhook
  --rl-policy "$PHASE4D_POLICY_PATH"
  --rl-normalizer "$PHASE4D_NORMALIZER_PATH"
  --metric-warmup-seconds 3.0
  --timeout 120
  --tick-wait-timeout 120
  --sidecar-tick-mode poll
  --poll-seconds 0.05
  --verify-every 50
  --require-verification
  --planning-provider empty
  --routes-subset "$PHASE4D_ROUTES_SUBSET"
  --seeds "$PHASE4D_SEED"
  --output-dir "$output_dir"
)
route_cmd+=("${extra_route_args[@]}")

set +e
"${route_cmd[@]}"
route_return_code=$?
set -e

"${accept_cmd[@]}" --route-return-code "$route_return_code" --fail-on-reject

echo "Phase 4-D S4 eval output dir: $output_dir"
echo "Phase 4-D S4 acceptance CSV: $output_dir/phase4d_eval_s4_acceptance.csv"
echo "Phase 4-D S4 acceptance JSON: $output_dir/phase4d_eval_s4_acceptance.json"
echo "Phase 4-D S4 manifest JSON: $output_dir/phase4d_eval_s4_manifest.json"

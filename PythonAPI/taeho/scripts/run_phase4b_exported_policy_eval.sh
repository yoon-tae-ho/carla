#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEFAULT_POLICY_DIR="/workspace/e2e_models/outputs/rl_suspension_phase4_real_carla_sac_canary_seed100_route00_20260604_113329"

show_help() {
  cat <<'EOF'
Run Phase 4-B exported policy route-suite evaluation.

This in-tree script is intended to run inside the Docker runner container
after env_garage_2.sh has been sourced.

From the host, prefer:
  cd ~/sim/docker
  bash run_phase4b_exported_policy_eval.sh \
    --policy-dir /home/yth/sim/e2e_models/outputs/rl_suspension_phase4_real_carla_sac_canary_seed100_route00_20260604_113329 \
    --routes-subset 00 \
    --seeds 100

Defaults:
  policy_dir=/workspace/e2e_models/outputs/rl_suspension_phase4_real_carla_sac_canary_seed100_route00_20260604_113329
  scenarios=rl_zero_residual_skyhook,rl_residual_skyhook
  routes_subset=00
  seeds=100
  metric_warmup_seconds=3.0
  verify_every=50
  require_verification=true
  planning_provider=empty
  baseline_scenario=S8_rl_zero_residual_skyhook

Use --print-command to validate command assembly without running CARLA.
EOF
}

quote_command() {
  local arg
  for arg in "$@"; do
    printf '%q ' "$arg"
  done
  printf '\n'
}

sanitize_label() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9' '_'
}

policy_dir="$DEFAULT_POLICY_DIR"
policy_path=""
normalizer_path=""
output_dir=""
routes_subset="00"
seeds="100"
scenarios="rl_zero_residual_skyhook,rl_residual_skyhook"
metric_warmup_seconds="3.0"
verify_every="50"
require_verification=1
planning_provider="empty"
baseline_scenario="S8_rl_zero_residual_skyhook"
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
    --policy-dir=*)
      policy_dir="${1#--policy-dir=}"
      shift
      ;;
    --policy-dir)
      if [ "$#" -lt 2 ]; then echo "missing value for --policy-dir" >&2; exit 2; fi
      policy_dir="$2"
      shift 2
      ;;
    --policy=*|--rl-policy=*)
      policy_path="${1#*=}"
      shift
      ;;
    --policy|--rl-policy)
      if [ "$#" -lt 2 ]; then echo "missing value for $1" >&2; exit 2; fi
      policy_path="$2"
      shift 2
      ;;
    --normalizer=*|--rl-normalizer=*)
      normalizer_path="${1#*=}"
      shift
      ;;
    --normalizer|--rl-normalizer)
      if [ "$#" -lt 2 ]; then echo "missing value for $1" >&2; exit 2; fi
      normalizer_path="$2"
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
    --routes-subset=*)
      routes_subset="${1#--routes-subset=}"
      shift
      ;;
    --routes-subset)
      if [ "$#" -lt 2 ]; then echo "missing value for --routes-subset" >&2; exit 2; fi
      routes_subset="$2"
      shift 2
      ;;
    --seeds=*)
      seeds="${1#--seeds=}"
      shift
      ;;
    --seeds)
      if [ "$#" -lt 2 ]; then echo "missing value for --seeds" >&2; exit 2; fi
      seeds="$2"
      shift 2
      ;;
    --scenarios=*)
      scenarios="${1#--scenarios=}"
      shift
      ;;
    --scenarios)
      if [ "$#" -lt 2 ]; then echo "missing value for --scenarios" >&2; exit 2; fi
      scenarios="$2"
      shift 2
      ;;
    --metric-warmup-seconds=*)
      metric_warmup_seconds="${1#--metric-warmup-seconds=}"
      shift
      ;;
    --metric-warmup-seconds)
      if [ "$#" -lt 2 ]; then echo "missing value for --metric-warmup-seconds" >&2; exit 2; fi
      metric_warmup_seconds="$2"
      shift 2
      ;;
    --verify-every=*)
      verify_every="${1#--verify-every=}"
      shift
      ;;
    --verify-every)
      if [ "$#" -lt 2 ]; then echo "missing value for --verify-every" >&2; exit 2; fi
      verify_every="$2"
      shift 2
      ;;
    --require-verification)
      require_verification=1
      shift
      ;;
    --no-require-verification)
      require_verification=0
      shift
      ;;
    --planning-provider=*)
      planning_provider="${1#--planning-provider=}"
      shift
      ;;
    --planning-provider)
      if [ "$#" -lt 2 ]; then echo "missing value for --planning-provider" >&2; exit 2; fi
      planning_provider="$2"
      shift 2
      ;;
    --baseline-scenario=*)
      baseline_scenario="${1#--baseline-scenario=}"
      shift
      ;;
    --baseline-scenario)
      if [ "$#" -lt 2 ]; then echo "missing value for --baseline-scenario" >&2; exit 2; fi
      baseline_scenario="$2"
      shift 2
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

if [ -z "$policy_path" ]; then
  policy_path="${policy_dir%/}/policy.ts"
fi
if [ -z "$normalizer_path" ]; then
  normalizer_path="${policy_dir%/}/normalizer.json"
fi
if [ -z "$output_dir" ]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  seed_label="$(sanitize_label "$seeds")"
  route_label="$(sanitize_label "$routes_subset")"
  output_dir="/workspace/e2e_models/outputs/rl_suspension_phase4b_policy_eval_seed${seed_label}_route${route_label}_${stamp}"
fi

if [ ! -f "$policy_path" ]; then
  echo "Phase 4-B exported policy eval missing policy file: $policy_path" >&2
  exit 2
fi
if [ ! -f "$normalizer_path" ]; then
  echo "Phase 4-B exported policy eval missing normalizer file: $normalizer_path" >&2
  exit 2
fi

route_cmd=(
  python transfuser_suspension_control_suite.py
  --scenarios "$scenarios"
  --rl-policy "$policy_path"
  --rl-normalizer "$normalizer_path"
  --metric-warmup-seconds "$metric_warmup_seconds"
  --verify-every "$verify_every"
  --planning-provider "$planning_provider"
  --routes-subset "$routes_subset"
  --seeds "$seeds"
  --baseline-scenario "$baseline_scenario"
  --output-dir "$output_dir"
)
if [ "$require_verification" -eq 1 ]; then
  route_cmd+=(--require-verification)
fi
route_cmd+=("${extra_args[@]}")

accept_cmd=(
  python -m suspension_control.rl.phase4_acceptance
  --phase4b-output-dir "$output_dir"
  --suite-summary "$output_dir/suite_summary.csv"
)

if [ "$print_command" -eq 1 ]; then
  echo "Phase 4-B route command:"
  quote_command "${route_cmd[@]}"
  echo "Phase 4-B acceptance command:"
  quote_command "${accept_cmd[@]}"
  echo "Phase 4-B output dir: $output_dir"
  exit 0
fi

mkdir -p "$output_dir"
"${route_cmd[@]}"
"${accept_cmd[@]}"

echo "Phase 4-B output dir: $output_dir"
echo "Phase 4-B acceptance CSV: $output_dir/phase4b_policy_eval_acceptance.csv"
echo "Phase 4-B acceptance JSON: $output_dir/phase4b_policy_eval_acceptance.json"

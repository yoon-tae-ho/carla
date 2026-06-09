#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

show_help() {
  cat <<'EOF'
Run Phase 4-C live-training horizon sweep inside the Docker runner container.

Default horizons:
  512,1024,2048

Stable live-training settings forced by this runner:
  --backend live
  --algorithm sac
  --max-episode-steps 0
  --verify-every 0
  --eval-after-training false
  --planning-provider empty
  --timeout 120
  --hero-timeout-seconds 900
  --route-wait-timeout-seconds 120
  --connect-retry-seconds 1.0

Usage:
  bash scripts/run_phase4c_training_horizon_sweep.sh [options] [extra train args]

Options:
  --steps 512,1024,2048      Override the default horizon list.
  --include-4096             Add 4096 steps to the horizon list.
  --eval-exported-policy     Run Phase 4-B exported-policy eval after each training run.
  --output-root PATH         Sweep output root.
  --seed VALUE               Traffic/training seed. Default: 100.
  --routes-subset VALUE      Route subset. Default: 00.
  --pause-between-runs SEC   Sleep between horizons. Default: 0.
  --print-command            Print commands without running CARLA training.
  -h, --help                 Show this help.

This runner does not include 4096 or exported-policy eval by default.
EOF
}

quote_cmd() {
  local arg
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

contains_step() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [ "$item" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

sanitize_label() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9' '_'
}

steps_csv="512,1024,2048"
include_4096=0
eval_exported_policy=0
output_root=""
seed="100"
routes_subset="00"
pause_between_runs=0
print_command=0
extra_train_args=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      show_help
      exit 0
      ;;
    --steps=*)
      steps_csv="${1#--steps=}"
      shift
      ;;
    --steps)
      if [ "$#" -lt 2 ]; then echo "missing value for --steps" >&2; exit 2; fi
      steps_csv="$2"
      shift 2
      ;;
    --include-4096)
      include_4096=1
      shift
      ;;
    --eval-exported-policy)
      eval_exported_policy=1
      shift
      ;;
    --no-eval-exported-policy)
      eval_exported_policy=0
      shift
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
    --pause-between-runs=*)
      pause_between_runs="${1#--pause-between-runs=}"
      shift
      ;;
    --pause-between-runs)
      if [ "$#" -lt 2 ]; then echo "missing value for --pause-between-runs" >&2; exit 2; fi
      pause_between_runs="$2"
      shift 2
      ;;
    --print-command)
      print_command=1
      shift
      ;;
    *)
      extra_train_args+=("$1")
      shift
      ;;
  esac
done

IFS=',' read -r -a horizon_steps <<< "$steps_csv"
if [ "$include_4096" -eq 1 ] && ! contains_step "4096" "${horizon_steps[@]}"; then
  horizon_steps+=("4096")
fi

stamp="$(date +%Y%m%d_%H%M%S)"
if [ -z "$output_root" ]; then
  seed_label="$(sanitize_label "$seed")"
  route_label="$(sanitize_label "$routes_subset")"
  output_root="/workspace/e2e_models/outputs/rl_suspension_phase4c_horizon_sweep_seed${seed_label}_route${route_label}_${stamp}"
fi

run_dirs=()
for horizon in "${horizon_steps[@]}"; do
  horizon="${horizon//[[:space:]]/}"
  if [ -z "$horizon" ]; then
    continue
  fi
  run_stamp="$(date +%Y%m%d_%H%M%S)"
  run_dir="${output_root}/rl_suspension_phase4c_train_${horizon}steps_seed${seed}_route${routes_subset}_${run_stamp}"
  run_dirs+=("$run_dir")

  train_cmd=(
    python -m suspension_control.rl.train_real_carla_sac
    "${extra_train_args[@]}"
    --backend live
    --algorithm sac
    --total-timesteps "$horizon"
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
    --output-dir "$run_dir"
  )
  health_cmd=(
    python -m suspension_control.rl.phase4_route_health
    --output-dir "$run_dir"
  )

  printf '\n=== Phase 4-C horizon: %s steps ===\n' "$horizon"
  printf 'Output: %s\n' "$run_dir"
  if [ "$print_command" -eq 1 ]; then
    printf 'TRAIN:'
    quote_cmd "${train_cmd[@]}"
    printf 'HEALTH:'
    quote_cmd "${health_cmd[@]}"
  else
    mkdir -p "$run_dir"
    "${train_cmd[@]}"
    "${health_cmd[@]}"
    if [ "$eval_exported_policy" -eq 1 ]; then
      eval_cmd=(
        bash scripts/run_phase4b_exported_policy_eval.sh
        --policy-dir "$run_dir"
        --routes-subset "$routes_subset"
        --seeds "$seed"
        --output-dir "$run_dir/phase4b_policy_eval"
      )
      "${eval_cmd[@]}"
    fi
    if [ "$pause_between_runs" != "0" ]; then
      sleep "$pause_between_runs"
    fi
  fi
done

run_dirs_csv="$(IFS=,; printf '%s' "${run_dirs[*]}")"
summary_cmd=(
  python -m suspension_control.rl.phase4c_horizon_sweep
  --output-root "$output_root"
  --run-dirs "$run_dirs_csv"
)

printf '\n=== Phase 4-C horizon sweep summary ===\n'
if [ "$print_command" -eq 1 ]; then
  printf 'SUMMARY:'
  quote_cmd "${summary_cmd[@]}"
else
  "${summary_cmd[@]}"
fi

printf 'phase4c sweep output root: %s\n' "$output_root"
printf 'phase4c sweep summary csv: %s\n' "$output_root/phase4c_horizon_sweep_summary.csv"
printf 'phase4c sweep summary json: %s\n' "$output_root/phase4c_horizon_sweep_summary.json"

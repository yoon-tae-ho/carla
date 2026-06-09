#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"

show_help() {
  cat <<'EOF'
Run Phase 4-D scripted residual sensitivity sweep.

This in-tree script is intended to run inside the Docker runner container
after env_garage_2.sh has been sourced.

Usage:
  bash scripts/run_phase4d_scripted_residual_sensitivity.sh \
    --eval-seeds 100,101,102 \
    --routes-subset 00 \
    --scripted-residuals zero,const_p0p05,const_m0p05,const_p0p10,const_m0p10 \
    --output-dir /workspace/e2e_models/outputs/<SCRIPTED_RESIDUAL_RUN>

This runner never trains and does not load a learned policy. It evaluates
skyhook plus scripted damper residuals, then compares each residual against
the same-seed zero residual reference.

Use --dry-run to write the scripted plan without starting CARLA.
Use --print-command to print assembled commands without running anything.
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

bool_value() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) printf '1' ;;
    0|false|FALSE|no|NO|off|OFF) printf '0' ;;
    *) echo "invalid boolean: $1" >&2; exit 2 ;;
  esac
}

scripted_value() {
  case "$1" in
    zero) printf '0.0' ;;
    const_p0p05) printf '0.05' ;;
    const_m0p05) printf -- '-0.05' ;;
    const_p0p10) printf '0.10' ;;
    const_m0p10) printf -- '-0.10' ;;
    *) echo "invalid scripted residual: $1" >&2; exit 2 ;;
  esac
}

write_scripted_config() {
  local config_path="$1"
  local residual="$2"
  local value
  value="$(scripted_value "$residual")"
  mkdir -p "$(dirname "$config_path")"
  cat > "$config_path" <<EOF
rl_residual_mode: scripted
rl_action_scale: 1.0
rl_residual_gain: 1.0
scripted_residual_kind: "$residual"
scripted_residual_value: $value
policy_path: ""
normalizer_path: ""
allow_untrained_policy: false
EOF
}

eval_seeds="100,101,102"
routes_subset="00"
baseline="skyhook"
planning_provider="empty"
scripted_residuals="zero,const_p0p05,const_m0p05,const_p0p10,const_m0p10"
stageA_report=""
output_dir=""
reuse_existing_evals=0
run_compare=1
dry_run=0
print_command=0
extra_eval_args=()

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
    --dry-run)
      dry_run=1
      shift
      ;;
    --eval-seeds=*)
      eval_seeds="${1#--eval-seeds=}"
      shift
      ;;
    --eval-seeds)
      if [ "$#" -lt 2 ]; then echo "missing value for --eval-seeds" >&2; exit 2; fi
      eval_seeds="$2"
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
    --baseline=*)
      baseline="${1#--baseline=}"
      shift
      ;;
    --baseline)
      if [ "$#" -lt 2 ]; then echo "missing value for --baseline" >&2; exit 2; fi
      baseline="$2"
      shift 2
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
    --scripted-residuals=*)
      scripted_residuals="${1#--scripted-residuals=}"
      shift
      ;;
    --scripted-residuals)
      if [ "$#" -lt 2 ]; then echo "missing value for --scripted-residuals" >&2; exit 2; fi
      scripted_residuals="$2"
      shift 2
      ;;
    --stageA-report=*)
      stageA_report="${1#--stageA-report=}"
      shift
      ;;
    --stageA-report)
      if [ "$#" -lt 2 ]; then echo "missing value for --stageA-report" >&2; exit 2; fi
      stageA_report="$2"
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
    --reuse-existing-evals=*)
      reuse_existing_evals="$(bool_value "${1#--reuse-existing-evals=}")"
      shift
      ;;
    --reuse-existing-evals)
      if [ "$#" -lt 2 ]; then echo "missing value for --reuse-existing-evals" >&2; exit 2; fi
      reuse_existing_evals="$(bool_value "$2")"
      shift 2
      ;;
    --compare=*)
      run_compare="$(bool_value "${1#--compare=}")"
      shift
      ;;
    --compare)
      if [ "$#" -lt 2 ]; then echo "missing value for --compare" >&2; exit 2; fi
      run_compare="$(bool_value "$2")"
      shift 2
      ;;
    *)
      extra_eval_args+=("$1")
      shift
      ;;
  esac
done

if [ "$baseline" != "skyhook" ]; then
  echo "scripted residual sensitivity currently requires --baseline skyhook" >&2
  exit 2
fi
if [ "$planning_provider" != "empty" ]; then
  echo "scripted residual sensitivity currently requires --planning-provider empty" >&2
  exit 2
fi
if [ -z "$output_dir" ]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  route_label="$(sanitize_label "$routes_subset")"
  seed_label="$(sanitize_label "$eval_seeds")"
  output_dir="/workspace/e2e_models/outputs/phase4d_scripted_residual_sensitivity_s${seed_label}_r${route_label}_${stamp}"
fi

plan_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.phase4d_scripted_residual_sensitivity
  --action plan
  --eval-seeds "$eval_seeds"
  --routes-subset "$routes_subset"
  --baseline "$baseline"
  --planning-provider "$planning_provider"
  --scripted-residuals "$scripted_residuals"
  --compare "$run_compare"
  --output-dir "$output_dir"
  --fail-on-preflight
)
if [ -n "$stageA_report" ]; then
  plan_cmd+=(--stageA-report "$stageA_report")
fi
if [ "$dry_run" -eq 1 ]; then
  plan_cmd+=(--dry-run)
fi

aggregate_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.phase4d_scripted_residual_sensitivity
  --action aggregate
  --output-dir "$output_dir"
)

IFS=',' read -r -a seed_array <<< "$eval_seeds"
IFS=',' read -r -a residual_array <<< "$scripted_residuals"

print_seed_commands() {
  local seed="$1"
  local seed_dir="$output_dir/eval_seed_${seed}"
  local residual
  for residual in "${residual_array[@]}"; do
    residual="${residual//[[:space:]]/}"
    if [ -z "$residual" ]; then
      continue
    fi
    local residual_dir="$seed_dir/$residual"
    local config_path="$residual_dir/scripted_${residual}.yaml"
    local route_cmd=(
      "$PYTHON_BIN" transfuser_suspension_control_suite.py
      --scenarios rl_residual_skyhook
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
      --rl-residual-config "$config_path"
      --output-dir "$residual_dir"
    )
    route_cmd+=("${extra_eval_args[@]}")
    printf 'SCRIPTED_CONFIG_SEED_%s_%s: write %s with scripted_residual_value=%s\n' \
      "$seed" "$residual" "$config_path" "$(scripted_value "$residual")"
    printf 'SCRIPTED_EVAL_SEED_%s_%s:' "$seed" "$residual"
    quote_cmd "${route_cmd[@]}"
  done
}

if [ "$print_command" -eq 1 ]; then
  echo "Phase 4-D scripted residual preflight/plan command:"
  printf 'SCRIPTED_PLAN:'
  quote_cmd "${plan_cmd[@]}"
  for seed in "${seed_array[@]}"; do
    seed="${seed//[[:space:]]/}"
    if [ -n "$seed" ]; then
      print_seed_commands "$seed"
    fi
  done
  printf 'SCRIPTED_AGGREGATE:'
  quote_cmd "${aggregate_cmd[@]}"
  echo "Phase 4-D scripted residual output dir: $output_dir"
  exit 0
fi

mkdir -p "$output_dir"
"${plan_cmd[@]}"

if [ "$dry_run" -eq 1 ]; then
  echo "Phase 4-D scripted residual dry-run complete."
  echo "Phase 4-D scripted residual plan: $output_dir/phase4d_scripted_residual_plan.json"
  exit 0
fi

status_csv="$output_dir/phase4d_scripted_residual_run_status.csv"
printf 'eval_seed,scripted_residual,scripted_residual_value,eval_status,eval_return_code,failed_stage,warnings\n' > "$status_csv"

for seed in "${seed_array[@]}"; do
  seed="${seed//[[:space:]]/}"
  if [ -z "$seed" ]; then
    continue
  fi
  seed_dir="$output_dir/eval_seed_${seed}"
  for residual in "${residual_array[@]}"; do
    residual="${residual//[[:space:]]/}"
    if [ -z "$residual" ]; then
      continue
    fi
    residual_dir="$seed_dir/$residual"
    config_path="$residual_dir/scripted_${residual}.yaml"
    mkdir -p "$residual_dir"
    write_scripted_config "$config_path" "$residual"
    eval_rc=0
    eval_status="skipped"
    failed_stage=""
    warnings=""
    if [ "$reuse_existing_evals" -eq 1 ] && [ -f "$residual_dir/suite_summary.csv" ]; then
      eval_status="reused"
    else
      set +e
      "$PYTHON_BIN" transfuser_suspension_control_suite.py \
        --scenarios rl_residual_skyhook \
        --metric-warmup-seconds 3.0 \
        --timeout 120 \
        --tick-wait-timeout 120 \
        --sidecar-tick-mode poll \
        --poll-seconds 0.05 \
        --verify-every 50 \
        --require-verification \
        --planning-provider empty \
        --routes-subset "$routes_subset" \
        --seeds "$seed" \
        --rl-residual-config "$config_path" \
        --output-dir "$residual_dir" \
        "${extra_eval_args[@]}"
      eval_rc=$?
      set -e
      if [ "$eval_rc" -eq 0 ]; then
        eval_status="completed"
      else
        eval_status="failed"
        failed_stage="eval"
      fi
    fi
    printf '%s,%s,%s,%s,%s,%s,%s\n' \
      "$seed" "$residual" "$(scripted_value "$residual")" \
      "$eval_status" "$eval_rc" "$failed_stage" "$warnings" >> "$status_csv"
  done
done

echo "Phase 4-D scripted residual output dir: $output_dir"
echo "Phase 4-D scripted residual plan: $output_dir/phase4d_scripted_residual_plan.json"
echo "Phase 4-D scripted residual run status: $status_csv"
"${aggregate_cmd[@]}"
echo "Phase 4-D scripted residual summary CSV: $output_dir/phase4d_scripted_residual_summary.csv"
echo "Phase 4-D scripted residual report JSON: $output_dir/phase4d_scripted_residual_report.json"

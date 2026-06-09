#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"

show_help() {
  cat <<'EOF'
Run Phase 4-D action authority sweep.

This in-tree script is intended to run inside the Docker runner container
after env_garage_2.sh has been sourced.

Usage:
  bash scripts/run_phase4d_action_authority_sweep.sh \
    --train-manifest /workspace/e2e_models/outputs/<TRAIN_RUN>/phase4d_train_manifest.json \
    --train-dir /workspace/e2e_models/outputs/<TRAIN_RUN>/train \
    --eval-seeds 100,101,102 \
    --routes-subset 00 \
    --action-scales 0.0,1.0,3.0,5.0,10.0 \
    --output-dir /workspace/e2e_models/outputs/<AUTHORITY_SWEEP_RUN>

This runner never trains. It runs S8 reference eval per seed, S4 learned
residual eval per action scale, per-gain S4-vs-S8 compare, then aggregates the
authority report.

Use --dry-run to write the authority plan without starting CARLA.
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

action_scale_label() {
  "$PYTHON_BIN" -c 'import sys
x=float(sys.argv[1])
t=("%.6f" % x).rstrip("0").rstrip(".")
if "." not in t:
    t += ".0"
print(t.replace("-", "m").replace(".", "p"))' "$1"
}

bool_value() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) printf '1' ;;
    0|false|FALSE|no|NO|off|OFF) printf '0' ;;
    *) echo "invalid boolean: $1" >&2; exit 2 ;;
  esac
}

write_gain_config() {
  local config_path="$1"
  local scale="$2"
  mkdir -p "$(dirname "$config_path")"
  cat > "$config_path" <<EOF
rl_residual_mode: learned_policy
rl_action_scale: $scale
rl_residual_gain: $scale
scripted_residual_kind: ""
scripted_residual_value: 0.0
EOF
}

train_manifest=""
train_dir=""
eval_seeds="100,101,102"
routes_subset="00"
baseline="skyhook"
planning_provider="empty"
action_scales="0.0,1.0,3.0,5.0,10.0"
stageA_report=""
output_dir=""
reuse_existing_evals=0
run_s8_reference=1
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
    --train-manifest=*)
      train_manifest="${1#--train-manifest=}"
      shift
      ;;
    --train-manifest)
      if [ "$#" -lt 2 ]; then echo "missing value for --train-manifest" >&2; exit 2; fi
      train_manifest="$2"
      shift 2
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
    --action-scales=*|--residual-gains=*)
      action_scales="${1#*=}"
      shift
      ;;
    --action-scales|--residual-gains)
      if [ "$#" -lt 2 ]; then echo "missing value for $1" >&2; exit 2; fi
      action_scales="$2"
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
    --run-s8-reference=*)
      run_s8_reference="$(bool_value "${1#--run-s8-reference=}")"
      shift
      ;;
    --run-s8-reference)
      if [ "$#" -lt 2 ]; then echo "missing value for --run-s8-reference" >&2; exit 2; fi
      run_s8_reference="$(bool_value "$2")"
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

if [ -z "$train_manifest" ]; then
  echo "authority sweep requires --train-manifest" >&2
  exit 2
fi
if [ -z "$train_dir" ]; then
  echo "authority sweep requires --train-dir" >&2
  exit 2
fi
if [ "$baseline" != "skyhook" ]; then
  echo "authority sweep currently requires --baseline skyhook" >&2
  exit 2
fi
if [ "$planning_provider" != "empty" ]; then
  echo "authority sweep currently requires --planning-provider empty" >&2
  exit 2
fi
if [ -z "$output_dir" ]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  route_label="$(sanitize_label "$routes_subset")"
  seed_label="$(sanitize_label "$eval_seeds")"
  output_dir="/workspace/e2e_models/outputs/phase4d_action_authority_sweep_s${seed_label}_r${route_label}_${stamp}"
fi

plan_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.phase4d_action_authority_sweep
  --action plan
  --train-manifest "$train_manifest"
  --train-dir "$train_dir"
  --eval-seeds "$eval_seeds"
  --routes-subset "$routes_subset"
  --baseline "$baseline"
  --planning-provider "$planning_provider"
  --action-scales "$action_scales"
  --run-s8-reference "$run_s8_reference"
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
  "$PYTHON_BIN" -m suspension_control.rl.phase4d_action_authority_sweep
  --action aggregate
  --output-dir "$output_dir"
)

IFS=',' read -r -a seed_array <<< "$eval_seeds"
IFS=',' read -r -a scale_array <<< "$action_scales"

print_seed_commands() {
  local seed="$1"
  local seed_dir="$output_dir/eval_seed_${seed}"
  local s8_dir="$seed_dir/s8_reference"
  local s8_cmd=(
    bash scripts/run_phase4d_eval_s8_reference.sh
    --routes-subset "$routes_subset"
    --seed "$seed"
    --output-dir "$s8_dir"
  )
  if [ "$run_s8_reference" -eq 1 ]; then
    printf 'AUTHORITY_S8_SEED_%s:' "$seed"
    quote_cmd "${s8_cmd[@]}"
  fi
  local scale
  for scale in "${scale_array[@]}"; do
    scale="${scale//[[:space:]]/}"
    if [ -z "$scale" ]; then
      continue
    fi
    local label
    label="$(action_scale_label "$scale")"
    local gain_dir="$seed_dir/gain_${label}"
    local config_path="$gain_dir/authority_gain_${label}.yaml"
    local compare_dir="$seed_dir/compare_gain_${label}"
    local s4_cmd=(
      bash scripts/run_phase4d_eval_s4_only.sh
      --train-manifest "$train_manifest"
      --train-dir "$train_dir"
      --routes-subset "$routes_subset"
      --seed "$seed"
      --output-dir "$gain_dir"
      --rl-residual-config "$config_path"
    )
    s4_cmd+=("${extra_eval_args[@]}")
    local compare_cmd=(
      "$PYTHON_BIN" -m suspension_control.rl.phase4d_compare_only
      --train-manifest "$train_manifest"
      --s4-summary "$gain_dir/suite_summary.csv"
      --s4-acceptance "$gain_dir/phase4d_eval_s4_acceptance.json"
      --s4-manifest "$gain_dir/phase4d_eval_s4_manifest.json"
      --s8-summary "$s8_dir/suite_summary.csv"
      --s8-acceptance "$s8_dir/phase4d_eval_s8_acceptance.json"
      --s8-manifest "$s8_dir/phase4d_eval_s8_manifest.json"
      --output-dir "$compare_dir"
    )
    printf 'AUTHORITY_GAIN_CONFIG_SEED_%s_%s: write %s with rl_action_scale=%s\n' \
      "$seed" "$label" "$config_path" "$scale"
    printf 'AUTHORITY_S4_SEED_%s_GAIN_%s:' "$seed" "$label"
    quote_cmd "${s4_cmd[@]}"
    if [ "$run_compare" -eq 1 ]; then
      printf 'AUTHORITY_COMPARE_SEED_%s_GAIN_%s:' "$seed" "$label"
      quote_cmd "${compare_cmd[@]}"
    fi
  done
}

if [ "$print_command" -eq 1 ]; then
  echo "Phase 4-D action authority preflight/plan command:"
  printf 'AUTHORITY_PLAN:'
  quote_cmd "${plan_cmd[@]}"
  for seed in "${seed_array[@]}"; do
    seed="${seed//[[:space:]]/}"
    if [ -n "$seed" ]; then
      print_seed_commands "$seed"
    fi
  done
  printf 'AUTHORITY_AGGREGATE:'
  quote_cmd "${aggregate_cmd[@]}"
  echo "Phase 4-D action authority output dir: $output_dir"
  exit 0
fi

mkdir -p "$output_dir"
"${plan_cmd[@]}"

if [ "$dry_run" -eq 1 ]; then
  echo "Phase 4-D action authority dry-run complete."
  echo "Phase 4-D action authority plan: $output_dir/phase4d_action_authority_plan.json"
  exit 0
fi

status_csv="$output_dir/phase4d_action_authority_run_status.csv"
printf 'eval_seed,action_scale,action_scale_label,s8_status,s8_return_code,s4_status,s4_return_code,compare_status,compare_return_code,failed_stage,warnings\n' > "$status_csv"

for seed in "${seed_array[@]}"; do
  seed="${seed//[[:space:]]/}"
  if [ -z "$seed" ]; then
    continue
  fi
  seed_dir="$output_dir/eval_seed_${seed}"
  s8_dir="$seed_dir/s8_reference"
  mkdir -p "$s8_dir"

  s8_rc=0
  s8_status="skipped"
  if [ "$run_s8_reference" -eq 1 ]; then
    if [ "$reuse_existing_evals" -eq 1 ] && [ -f "$s8_dir/phase4d_eval_s8_acceptance.json" ]; then
      s8_status="reused"
    else
      set +e
      bash scripts/run_phase4d_eval_s8_reference.sh \
        --routes-subset "$routes_subset" \
        --seed "$seed" \
        --output-dir "$s8_dir"
      s8_rc=$?
      set -e
      if [ "$s8_rc" -eq 0 ]; then
        s8_status="completed"
      else
        s8_status="failed"
      fi
    fi
  fi

  for scale in "${scale_array[@]}"; do
    scale="${scale//[[:space:]]/}"
    if [ -z "$scale" ]; then
      continue
    fi
    label="$(action_scale_label "$scale")"
    gain_dir="$seed_dir/gain_${label}"
    compare_dir="$seed_dir/compare_gain_${label}"
    config_path="$gain_dir/authority_gain_${label}.yaml"
    mkdir -p "$gain_dir" "$compare_dir"
    write_gain_config "$config_path" "$scale"

    s4_rc=0
    compare_rc=0
    s4_status="skipped"
    compare_status="skipped"
    failed_stage=""
    warnings=""

    if [ "$reuse_existing_evals" -eq 1 ] && [ -f "$gain_dir/phase4d_eval_s4_acceptance.json" ]; then
      s4_status="reused"
    else
      set +e
      bash scripts/run_phase4d_eval_s4_only.sh \
        --train-manifest "$train_manifest" \
        --train-dir "$train_dir" \
        --routes-subset "$routes_subset" \
        --seed "$seed" \
        --output-dir "$gain_dir" \
        --rl-residual-config "$config_path" \
        "${extra_eval_args[@]}"
      s4_rc=$?
      set -e
      if [ "$s4_rc" -eq 0 ]; then
        s4_status="completed"
      else
        s4_status="failed"
        failed_stage="${failed_stage:+$failed_stage;}s4"
      fi
    fi

    if [ "$run_compare" -eq 1 ] && [ "$run_s8_reference" -eq 1 ]; then
      if [ "$reuse_existing_evals" -eq 1 ] && [ -f "$compare_dir/phase4d_compare_acceptance.json" ]; then
        compare_status="reused"
      else
        set +e
        "$PYTHON_BIN" -m suspension_control.rl.phase4d_compare_only \
          --train-manifest "$train_manifest" \
          --s4-summary "$gain_dir/suite_summary.csv" \
          --s4-acceptance "$gain_dir/phase4d_eval_s4_acceptance.json" \
          --s4-manifest "$gain_dir/phase4d_eval_s4_manifest.json" \
          --s8-summary "$s8_dir/suite_summary.csv" \
          --s8-acceptance "$s8_dir/phase4d_eval_s8_acceptance.json" \
          --s8-manifest "$s8_dir/phase4d_eval_s8_manifest.json" \
          --output-dir "$compare_dir"
        compare_rc=$?
        set -e
        if [ "$compare_rc" -eq 0 ]; then
          compare_status="completed"
        else
          compare_status="failed"
          failed_stage="${failed_stage:+$failed_stage;}compare"
        fi
      fi
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$seed" "$scale" "$label" "$s8_status" "$s8_rc" "$s4_status" "$s4_rc" \
      "$compare_status" "$compare_rc" "$failed_stage" "$warnings" >> "$status_csv"
  done
done

echo "Phase 4-D action authority output dir: $output_dir"
echo "Phase 4-D action authority plan: $output_dir/phase4d_action_authority_plan.json"
echo "Phase 4-D action authority run status: $status_csv"
"${aggregate_cmd[@]}"
echo "Phase 4-D action authority summary CSV: $output_dir/phase4d_action_authority_summary.csv"
echo "Phase 4-D action authority report JSON: $output_dir/phase4d_action_authority_report.json"

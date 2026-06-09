#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"

show_help() {
  cat <<'EOF'
Run Phase 4-D Stage A same-artifact eval repeat.

This in-tree script is intended to run inside the Docker runner container
after env_garage_2.sh has been sourced.

Usage:
  bash scripts/run_phase4d_stageA_same_artifact_eval_repeat.sh \
    --train-manifest /workspace/e2e_models/outputs/<TRAIN_RUN>/phase4d_train_manifest.json \
    --train-dir /workspace/e2e_models/outputs/<TRAIN_RUN>/train \
    --eval-seeds 100,101,102 \
    --routes-subset 00 \
    --output-dir /workspace/e2e_models/outputs/<STAGE_A_RUN>

This runner never trains. It calls the existing S4 eval-only, S8 reference eval,
and compare-only entrypoints once per eval seed.

Use --dry-run to write the Stage A plan without starting CARLA.
Use --print-command to print the assembled commands without running anything.
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

train_manifest=""
train_dir=""
eval_seeds="100,101,102"
routes_subset="00"
baseline="skyhook"
planning_provider="empty"
output_dir=""
reuse_existing_evals=0
run_s4=1
run_s8=1
run_compare=1
dry_run=0
print_command=0

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
    --run-s4=*)
      run_s4="$(bool_value "${1#--run-s4=}")"
      shift
      ;;
    --run-s4)
      if [ "$#" -lt 2 ]; then echo "missing value for --run-s4" >&2; exit 2; fi
      run_s4="$(bool_value "$2")"
      shift 2
      ;;
    --run-s8=*)
      run_s8="$(bool_value "${1#--run-s8=}")"
      shift
      ;;
    --run-s8)
      if [ "$#" -lt 2 ]; then echo "missing value for --run-s8" >&2; exit 2; fi
      run_s8="$(bool_value "$2")"
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
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$train_manifest" ]; then
  echo "Stage A requires --train-manifest" >&2
  exit 2
fi
if [ -z "$train_dir" ]; then
  echo "Stage A requires --train-dir" >&2
  exit 2
fi
if [ "$baseline" != "skyhook" ]; then
  echo "Stage A currently requires --baseline skyhook" >&2
  exit 2
fi
if [ "$planning_provider" != "empty" ]; then
  echo "Stage A currently requires --planning-provider empty" >&2
  exit 2
fi
if [ -z "$output_dir" ]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  route_label="$(sanitize_label "$routes_subset")"
  seed_label="$(sanitize_label "$eval_seeds")"
  output_dir="/workspace/e2e_models/outputs/phase4d_stageA_same_artifact_repeat_s${seed_label}_r${route_label}_${stamp}"
fi

plan_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.phase4d_stageA_repeat
  --action plan
  --train-manifest "$train_manifest"
  --train-dir "$train_dir"
  --eval-seeds "$eval_seeds"
  --routes-subset "$routes_subset"
  --baseline "$baseline"
  --planning-provider "$planning_provider"
  --output-dir "$output_dir"
  --fail-on-preflight
)
if [ "$dry_run" -eq 1 ]; then
  plan_cmd+=(--dry-run)
fi

IFS=',' read -r -a seed_array <<< "$eval_seeds"

print_seed_commands() {
  local seed="$1"
  local seed_dir="$output_dir/eval_seed_${seed}"
  local s4_dir="$seed_dir/s4"
  local s8_dir="$seed_dir/s8"
  local compare_dir="$seed_dir/compare"
  local s4_cmd=(
    bash scripts/run_phase4d_eval_s4_only.sh
    --train-manifest "$train_manifest"
    --train-dir "$train_dir"
    --routes-subset "$routes_subset"
    --seed "$seed"
    --output-dir "$s4_dir"
  )
  local s8_cmd=(
    bash scripts/run_phase4d_eval_s8_reference.sh
    --routes-subset "$routes_subset"
    --seed "$seed"
    --output-dir "$s8_dir"
  )
  local compare_cmd=(
    "$PYTHON_BIN" -m suspension_control.rl.phase4d_compare_only
    --train-manifest "$train_manifest"
    --s4-summary "$s4_dir/suite_summary.csv"
    --s4-acceptance "$s4_dir/phase4d_eval_s4_acceptance.json"
    --s4-manifest "$s4_dir/phase4d_eval_s4_manifest.json"
    --s8-summary "$s8_dir/suite_summary.csv"
    --s8-acceptance "$s8_dir/phase4d_eval_s8_acceptance.json"
    --s8-manifest "$s8_dir/phase4d_eval_s8_manifest.json"
    --output-dir "$compare_dir"
  )
  if [ "$run_s4" -eq 1 ]; then
    printf 'STAGEA_S4_SEED_%s:' "$seed"
    quote_cmd "${s4_cmd[@]}"
  fi
  if [ "$run_s8" -eq 1 ]; then
    printf 'STAGEA_S8_SEED_%s:' "$seed"
    quote_cmd "${s8_cmd[@]}"
  fi
  if [ "$run_compare" -eq 1 ]; then
    printf 'STAGEA_COMPARE_SEED_%s:' "$seed"
    quote_cmd "${compare_cmd[@]}"
  fi
}

if [ "$print_command" -eq 1 ]; then
  echo "Phase 4-D Stage A preflight/plan command:"
  printf 'STAGEA_PLAN:'
  quote_cmd "${plan_cmd[@]}"
  for seed in "${seed_array[@]}"; do
    seed="${seed//[[:space:]]/}"
    if [ -n "$seed" ]; then
      print_seed_commands "$seed"
    fi
  done
  echo "Phase 4-D Stage A output dir: $output_dir"
  exit 0
fi

mkdir -p "$output_dir"
"${plan_cmd[@]}"

if [ "$dry_run" -eq 1 ]; then
  echo "Phase 4-D Stage A dry-run complete."
  echo "Phase 4-D Stage A plan: $output_dir/phase4d_stageA_repeat_plan.json"
  exit 0
fi

status_csv="$output_dir/phase4d_stageA_seed_status.csv"
printf 'eval_seed,seed_status,s4_status,s4_return_code,s8_status,s8_return_code,compare_status,compare_return_code,failed_stage,warnings\n' > "$status_csv"

for seed in "${seed_array[@]}"; do
  seed="${seed//[[:space:]]/}"
  if [ -z "$seed" ]; then
    continue
  fi
  seed_dir="$output_dir/eval_seed_${seed}"
  s4_dir="$seed_dir/s4"
  s8_dir="$seed_dir/s8"
  compare_dir="$seed_dir/compare"
  mkdir -p "$s4_dir" "$s8_dir" "$compare_dir"

  s4_rc=0
  s8_rc=0
  compare_rc=0
  s4_status="skipped"
  s8_status="skipped"
  compare_status="skipped"
  failed_stage=""
  warnings=""

  if [ "$reuse_existing_evals" -eq 1 ]; then
    set +e
    reuse_exports="$("$PYTHON_BIN" -m suspension_control.rl.phase4d_stageA_repeat \
      --action reuse-check \
      --train-manifest "$train_manifest" \
      --eval-seed "$seed" \
      --routes-subset "$routes_subset" \
      --baseline "$baseline" \
      --planning-provider "$planning_provider" \
      --output-dir "$output_dir" \
      --print-shell)"
    reuse_rc=$?
    set -e
    if [ "$reuse_rc" -eq 0 ]; then
      eval "$reuse_exports"
      if [ "${PHASE4D_STAGEA_REUSE_OK:-0}" = "1" ]; then
        printf '%s,reused,skipped,0,skipped,0,skipped,0,,reuse_existing_evals\n' "$seed" >> "$status_csv"
        continue
      fi
      warnings="${warnings:+$warnings;}reuse_rejected:${PHASE4D_STAGEA_REUSE_FAILED_CHECKS:-unknown}"
    else
      warnings="${warnings:+$warnings;}reuse_check_failed"
    fi
  fi

  if [ "$run_s4" -eq 1 ]; then
    set +e
    bash scripts/run_phase4d_eval_s4_only.sh \
      --train-manifest "$train_manifest" \
      --train-dir "$train_dir" \
      --routes-subset "$routes_subset" \
      --seed "$seed" \
      --output-dir "$s4_dir"
    s4_rc=$?
    set -e
    if [ "$s4_rc" -eq 0 ]; then
      s4_status="completed"
    else
      s4_status="failed"
      failed_stage="${failed_stage:+$failed_stage;}s4"
    fi
  fi

  if [ "$run_s8" -eq 1 ]; then
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
      failed_stage="${failed_stage:+$failed_stage;}s8"
    fi
  fi

  if [ "$run_compare" -eq 1 ]; then
    set +e
    "$PYTHON_BIN" -m suspension_control.rl.phase4d_compare_only \
      --train-manifest "$train_manifest" \
      --s4-summary "$s4_dir/suite_summary.csv" \
      --s4-acceptance "$s4_dir/phase4d_eval_s4_acceptance.json" \
      --s4-manifest "$s4_dir/phase4d_eval_s4_manifest.json" \
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

  if [ -z "$failed_stage" ]; then
    seed_status="completed"
  else
    seed_status="failed"
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$seed" "$seed_status" "$s4_status" "$s4_rc" "$s8_status" "$s8_rc" \
    "$compare_status" "$compare_rc" "$failed_stage" "$warnings" >> "$status_csv"
done

echo "Phase 4-D Stage A output dir: $output_dir"
echo "Phase 4-D Stage A plan: $output_dir/phase4d_stageA_repeat_plan.json"
echo "Phase 4-D Stage A seed status: $status_csv"
"$PYTHON_BIN" -m suspension_control.rl.phase4d_stageA_repeat \
  --action aggregate \
  --output-dir "$output_dir"
echo "Phase 4-D Stage A summary: $output_dir/phase4d_stageA_repeat_summary.csv"
echo "Phase 4-D Stage A stochasticity report: $output_dir/phase4d_stageA_stochasticity_report.json"

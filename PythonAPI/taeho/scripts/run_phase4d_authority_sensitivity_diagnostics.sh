#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"

show_help() {
  cat <<'EOF'
Run Phase 4-D authority sensitivity diagnostics.

This in-tree script is intended to run inside the Docker runner container
after env_garage_2.sh has been sourced.

Usage:
  bash scripts/run_phase4d_authority_sensitivity_diagnostics.sh \
    --train-manifest /workspace/e2e_models/outputs/<TRAIN_RUN>/phase4d_train_manifest.json \
    --train-dir /workspace/e2e_models/outputs/<TRAIN_RUN>/train \
    --stageA-report /workspace/e2e_models/outputs/<STAGEA_RUN>/phase4d_stageA_stochasticity_report.json \
    --eval-seeds 100,101,102 \
    --routes-subset 00 \
    --output-dir /workspace/e2e_models/outputs/<DIAGNOSTIC_RUN>

This is a convenience orchestrator. Official outputs remain split:
  <output-dir>/action_authority
  <output-dir>/scripted_residual
  <output-dir>/combined_report

Use --dry-run to write plans and an incomplete combined report without
starting CARLA. Use --print-command to print assembled commands only.

Use --report-only with --authority-dir and --scripted-dir to combine existing
aggregate outputs without running either sweep.
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
action_scales="0.0,1.0,3.0,5.0,10.0"
scripted_residuals="zero,const_p0p05,const_m0p05,const_p0p10,const_m0p10"
stageA_report=""
output_dir=""
authority_dir=""
scripted_dir=""
combined_dir=""
reuse_existing_evals=0
dry_run=0
print_command=0
report_only=0
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
    --report-only)
      report_only=1
      shift
      ;;
    --train-manifest=*) train_manifest="${1#--train-manifest=}"; shift ;;
    --train-manifest)
      if [ "$#" -lt 2 ]; then echo "missing value for --train-manifest" >&2; exit 2; fi
      train_manifest="$2"; shift 2 ;;
    --train-dir=*) train_dir="${1#--train-dir=}"; shift ;;
    --train-dir)
      if [ "$#" -lt 2 ]; then echo "missing value for --train-dir" >&2; exit 2; fi
      train_dir="$2"; shift 2 ;;
    --eval-seeds=*) eval_seeds="${1#--eval-seeds=}"; shift ;;
    --eval-seeds)
      if [ "$#" -lt 2 ]; then echo "missing value for --eval-seeds" >&2; exit 2; fi
      eval_seeds="$2"; shift 2 ;;
    --routes-subset=*) routes_subset="${1#--routes-subset=}"; shift ;;
    --routes-subset)
      if [ "$#" -lt 2 ]; then echo "missing value for --routes-subset" >&2; exit 2; fi
      routes_subset="$2"; shift 2 ;;
    --baseline=*) baseline="${1#--baseline=}"; shift ;;
    --baseline)
      if [ "$#" -lt 2 ]; then echo "missing value for --baseline" >&2; exit 2; fi
      baseline="$2"; shift 2 ;;
    --planning-provider=*) planning_provider="${1#--planning-provider=}"; shift ;;
    --planning-provider)
      if [ "$#" -lt 2 ]; then echo "missing value for --planning-provider" >&2; exit 2; fi
      planning_provider="$2"; shift 2 ;;
    --action-scales=*|--residual-gains=*) action_scales="${1#*=}"; shift ;;
    --action-scales|--residual-gains)
      if [ "$#" -lt 2 ]; then echo "missing value for $1" >&2; exit 2; fi
      action_scales="$2"; shift 2 ;;
    --scripted-residuals=*) scripted_residuals="${1#--scripted-residuals=}"; shift ;;
    --scripted-residuals)
      if [ "$#" -lt 2 ]; then echo "missing value for --scripted-residuals" >&2; exit 2; fi
      scripted_residuals="$2"; shift 2 ;;
    --stageA-report=*) stageA_report="${1#--stageA-report=}"; shift ;;
    --stageA-report)
      if [ "$#" -lt 2 ]; then echo "missing value for --stageA-report" >&2; exit 2; fi
      stageA_report="$2"; shift 2 ;;
    --output-dir=*) output_dir="${1#--output-dir=}"; shift ;;
    --output-dir)
      if [ "$#" -lt 2 ]; then echo "missing value for --output-dir" >&2; exit 2; fi
      output_dir="$2"; shift 2 ;;
    --authority-dir=*) authority_dir="${1#--authority-dir=}"; shift ;;
    --authority-dir)
      if [ "$#" -lt 2 ]; then echo "missing value for --authority-dir" >&2; exit 2; fi
      authority_dir="$2"; shift 2 ;;
    --scripted-dir=*) scripted_dir="${1#--scripted-dir=}"; shift ;;
    --scripted-dir)
      if [ "$#" -lt 2 ]; then echo "missing value for --scripted-dir" >&2; exit 2; fi
      scripted_dir="$2"; shift 2 ;;
    --combined-dir=*) combined_dir="${1#--combined-dir=}"; shift ;;
    --combined-dir)
      if [ "$#" -lt 2 ]; then echo "missing value for --combined-dir" >&2; exit 2; fi
      combined_dir="$2"; shift 2 ;;
    --reuse-existing-evals=*)
      reuse_existing_evals="$(bool_value "${1#--reuse-existing-evals=}")"; shift ;;
    --reuse-existing-evals)
      if [ "$#" -lt 2 ]; then echo "missing value for --reuse-existing-evals" >&2; exit 2; fi
      reuse_existing_evals="$(bool_value "$2")"; shift 2 ;;
    *)
      extra_eval_args+=("$1")
      shift
      ;;
  esac
done

if [ "$baseline" != "skyhook" ]; then
  echo "authority sensitivity diagnostics currently requires --baseline skyhook" >&2
  exit 2
fi
if [ "$planning_provider" != "empty" ]; then
  echo "authority sensitivity diagnostics currently requires --planning-provider empty" >&2
  exit 2
fi
if [ -z "$output_dir" ]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  route_label="$(sanitize_label "$routes_subset")"
  seed_label="$(sanitize_label "$eval_seeds")"
  output_dir="/workspace/e2e_models/outputs/phase4d_authority_sensitivity_s${seed_label}_r${route_label}_${stamp}"
fi
if [ -z "$authority_dir" ]; then
  authority_dir="$output_dir/action_authority"
fi
if [ -z "$scripted_dir" ]; then
  scripted_dir="$output_dir/scripted_residual"
fi
if [ -z "$combined_dir" ]; then
  combined_dir="$output_dir/combined_report"
fi

if [ "$report_only" -eq 0 ]; then
  if [ -z "$train_manifest" ]; then
    echo "authority sensitivity diagnostics requires --train-manifest unless --report-only is used" >&2
    exit 2
  fi
  if [ -z "$train_dir" ]; then
    echo "authority sensitivity diagnostics requires --train-dir unless --report-only is used" >&2
    exit 2
  fi
fi

authority_cmd=(
  bash scripts/run_phase4d_action_authority_sweep.sh
  --train-manifest "$train_manifest"
  --train-dir "$train_dir"
  --eval-seeds "$eval_seeds"
  --routes-subset "$routes_subset"
  --baseline "$baseline"
  --planning-provider "$planning_provider"
  --action-scales "$action_scales"
  --reuse-existing-evals "$reuse_existing_evals"
  --output-dir "$authority_dir"
)
scripted_cmd=(
  bash scripts/run_phase4d_scripted_residual_sensitivity.sh
  --eval-seeds "$eval_seeds"
  --routes-subset "$routes_subset"
  --baseline "$baseline"
  --planning-provider "$planning_provider"
  --scripted-residuals "$scripted_residuals"
  --reuse-existing-evals "$reuse_existing_evals"
  --output-dir "$scripted_dir"
)
report_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.phase4d_authority_sensitivity_report
  --authority-dir "$authority_dir"
  --scripted-dir "$scripted_dir"
  --output-dir "$combined_dir"
)
if [ -n "$stageA_report" ]; then
  authority_cmd+=(--stageA-report "$stageA_report")
  scripted_cmd+=(--stageA-report "$stageA_report")
  report_cmd+=(--stageA-report "$stageA_report")
fi
if [ "$dry_run" -eq 1 ]; then
  authority_cmd+=(--dry-run)
  scripted_cmd+=(--dry-run)
fi
authority_cmd+=("${extra_eval_args[@]}")
scripted_cmd+=("${extra_eval_args[@]}")

if [ "$print_command" -eq 1 ]; then
  echo "Phase 4-D authority sensitivity commands:"
  if [ "$report_only" -eq 0 ]; then
    printf 'COMBINED_AUTHORITY_SWEEP:'
    quote_cmd "${authority_cmd[@]}"
    printf 'COMBINED_SCRIPTED_RESIDUAL:'
    quote_cmd "${scripted_cmd[@]}"
  fi
  printf 'COMBINED_REPORT:'
  quote_cmd "${report_cmd[@]}"
  echo "Phase 4-D authority sensitivity output dir: $output_dir"
  exit 0
fi

mkdir -p "$output_dir" "$combined_dir"
authority_rc=0
scripted_rc=0
if [ "$report_only" -eq 0 ]; then
  set +e
  "${authority_cmd[@]}"
  authority_rc=$?
  "${scripted_cmd[@]}"
  scripted_rc=$?
  set -e
fi

"${report_cmd[@]}"
echo "Phase 4-D authority sensitivity output dir: $output_dir"
echo "Phase 4-D action authority dir: $authority_dir"
echo "Phase 4-D scripted residual dir: $scripted_dir"
echo "Phase 4-D combined report dir: $combined_dir"

if [ "$authority_rc" -ne 0 ] || [ "$scripted_rc" -ne 0 ]; then
  exit 1
fi

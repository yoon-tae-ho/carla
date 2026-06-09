#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"

show_help() {
  cat <<'EOF'
Run Phase 4-D no-planning live SAC training, S4-only eval, and split-reference reports.

This in-tree script is intended to run inside the Docker runner container
after env_garage_2.sh has been sourced.

Usage:
  bash scripts/run_phase4d_no_planning_train_eval.sh \
    --seed 100 \
    --routes-subset 00 \
    --total-timesteps 4096 \
    --reference-summary /workspace/e2e_models/outputs/<S8_RUN>/seed100_S8_rl_zero_residual_skyhook/summary.json

For route00 single-episode Phase 4-D canary, use 4096 steps.
8192 steps exceed the observed route00 single-episode lifetime and require
multi-episode reset support. Do not use 8192 for the main Phase 4-D split path
until route terminal/reset handling is implemented.

Stable live-training settings forced by this runner:
  --backend live
  --algorithm sac
  --planning-provider empty
  --max-episode-steps 0
  --verify-every 0
  --eval-after-training false
  --timeout 120
  --hero-timeout-seconds 900
  --route-wait-timeout-seconds 120
  --connect-retry-seconds 1.0

Route-suite evaluation settings:
  --scenarios rl_residual_skyhook
  --metric-warmup-seconds 3.0
  --timeout 120
  --tick-wait-timeout 120
  --poll-seconds 0.05
  --verify-every 50
  --require-verification
  --planning-provider empty

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

stabilize_carla_before_eval() {
  echo "Stabilizing CARLA before Phase 4-D S4 eval..."
  "${PYTHON_BIN}" - <<'PY' || true
import time

try:
    import carla
except Exception as error:
    print("CARLA eval preflight import failed: %s" % error)
    raise SystemExit(0)

try:
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(120.0)
    world = client.get_world()
    settings = world.get_settings()
    changed = False
    if getattr(settings, "synchronous_mode", False):
        settings.synchronous_mode = False
        changed = True
    if getattr(settings, "fixed_delta_seconds", None) is not None:
        settings.fixed_delta_seconds = None
        changed = True
    if changed:
        world.apply_settings(settings)
        time.sleep(1.0)
    try:
        client.get_trafficmanager(8000).set_synchronous_mode(False)
    except Exception:
        pass
    snapshot = world.get_snapshot()
    print(
        "CARLA eval preflight: map=%s frame=%s sync=%s fixed_delta=%s" %
        (
            world.get_map().name,
            getattr(snapshot, "frame", ""),
            getattr(settings, "synchronous_mode", ""),
            getattr(settings, "fixed_delta_seconds", ""),
        ))
except Exception as error:
    print("CARLA eval preflight failed: %s" % error)
PY
  sleep 30
}

write_final_summary() {
  "${PYTHON_BIN}" - "$output_root" "$train_dir" "$eval_dir" "$comparison_dir" "$reference_summary" <<'PY'
import csv
import json
import os
import sys

output_root, train_dir, eval_dir, comparison_dir, reference_summary = sys.argv[1:6]
selection_path = os.path.join(comparison_dir, "phase4d_policy_selection.json")
acceptance_path = os.path.join(comparison_dir, "phase4b_policy_eval_acceptance.json")
comparison_path = os.path.join(comparison_dir, "phase4b_split_comparison.json")
train_acceptance_path = os.path.join(comparison_dir, "phase4d_train_acceptance.json")
training_summary_path = os.path.join(train_dir, "training_summary.json")
training_canary_path = os.path.join(train_dir, "phase4_training_canary.json")
lifecycle_path = os.path.join(comparison_dir, "phase4d_lifecycle_summary.json")
lifecycle_csv_path = os.path.join(comparison_dir, "phase4d_lifecycle_summary.csv")

def read_json(path, fallback):
    try:
        with open(path) as json_file:
            return json.load(json_file)
    except Exception:
        return fallback

selection = read_json(selection_path, {})
selection_row = selection.get("summary", {}) if isinstance(selection, dict) else {}
acceptance = read_json(acceptance_path, [])
split_comparison = read_json(comparison_path, [])
train_acceptance = read_json(train_acceptance_path, {})
training_summary = read_json(training_summary_path, {})
training_canary = read_json(training_canary_path, {})
train_failed = (
    isinstance(train_acceptance, dict) and
    str(train_acceptance.get("invalid_for_eval", "")).strip() in ("1", "true", "True")
)

lifecycle_keys = (
    "backend",
    "status",
    "learn_error",
    "export_error",
    "hero_actor_id",
    "hero_actor_alive_last",
    "hero_destroy_detected",
    "hero_destroy_detected_step",
    "hero_destroy_detected_wall_time",
    "stale_actor_api_call_count",
    "actor_not_found_error_count",
    "last_suspension_api_call_step",
    "last_suspension_api_call_actor_id",
    "last_suspension_api_name",
    "last_suspension_api_error",
    "route_process_returncode",
    "route_record_status",
    "route_record_score_route",
    "route_record_score_composed",
    "route_record_duration_game",
    "route_checkpoint_progress",
    "route_entry_status",
    "route_finished_detected",
    "route_finished_detected_step",
    "route_finished_detected_wall_time",
    "terminal_reason",
    "episode_end_reason",
)
lifecycle = {
    "phase4d_status": (
        "failed_training"
        if train_failed else
        selection_row.get("phase4d_status", selection.get("phase4d_status", ""))),
    "eval_status": (
        train_acceptance.get("eval_status", "skipped_due_to_failed_training")
        if train_failed else
        "completed"
        if selection else
        ""),
    "invalid_for_eval": train_acceptance.get("invalid_for_eval", 0 if selection else ""),
    "phase4d_train_status": train_acceptance.get("phase4d_train_status", ""),
    "train_failed_checks": train_acceptance.get("failed_checks", ""),
    "training_summary": training_summary_path,
    "phase4_training_canary": training_canary_path,
    "phase4d_train_acceptance_json": train_acceptance_path,
    "phase4b_acceptance_json": acceptance_path,
    "phase4b_split_comparison_json": comparison_path,
    "phase4d_policy_selection_json": selection_path,
}
for key in lifecycle_keys:
    value = (
        training_summary.get(key, "")
        if isinstance(training_summary, dict) else
        "")
    if value == "" and isinstance(training_canary, dict):
        value = training_canary.get(key, "")
    if value == "" and isinstance(train_acceptance, dict):
        value = train_acceptance.get(key, "")
    lifecycle[key] = value

row = {
    "phase4d_status": (
        "failed_training"
        if train_failed else
        selection_row.get("phase4d_status", selection.get("phase4d_status", ""))),
    "engineering_pass": (
        0
        if train_failed else
        selection_row.get("engineering_pass", "")),
    "eval_status": (
        train_acceptance.get("eval_status", "skipped_due_to_failed_training")
        if train_failed else
        "completed"
        if selection else
        ""),
    "invalid_for_eval": train_acceptance.get("invalid_for_eval", 0 if selection else ""),
    "policy_tags": selection_row.get("policy_tags", ";".join(selection.get("policy_tags", []))),
    "failed_checks": (
        train_acceptance.get("failed_checks", "")
        if train_failed else
        selection_row.get("failed_checks", "")),
    "warnings": (
        train_acceptance.get("warnings", "")
        if train_failed else
        selection_row.get("warnings", "")),
    "train_failed_checks": train_acceptance.get("failed_checks", ""),
    "phase4d_train_status": train_acceptance.get("phase4d_train_status", ""),
    "total_timesteps_requested": train_acceptance.get("total_timesteps_requested", ""),
    "total_timesteps_collected": train_acceptance.get("total_timesteps_collected", ""),
    "terminal_reason": lifecycle.get("terminal_reason", ""),
    "episode_end_reason": lifecycle.get("episode_end_reason", ""),
    "hero_actor_id": lifecycle.get("hero_actor_id", ""),
    "hero_destroy_detected": lifecycle.get("hero_destroy_detected", ""),
    "actor_not_found_error_count": lifecycle.get("actor_not_found_error_count", ""),
    "stale_actor_api_call_count": lifecycle.get("stale_actor_api_call_count", ""),
    "route_process_returncode": lifecycle.get("route_process_returncode", ""),
    "route_record_status": lifecycle.get("route_record_status", ""),
    "route_record_score_route": lifecycle.get("route_record_score_route", ""),
    "route_checkpoint_progress": lifecycle.get("route_checkpoint_progress", ""),
    "reward_acceptable": selection_row.get("reward_acceptable", ""),
    "comfort_promising": selection_row.get("comfort_promising", ""),
    "stability_risk": selection_row.get("stability_risk", ""),
    "action_overuse_warning": selection_row.get("action_overuse_warning", ""),
    "output_root": output_root,
    "train_dir": train_dir,
    "eval_s4_dir": eval_dir,
    "comparison_dir": comparison_dir,
    "reference_summary": reference_summary,
    "training_summary": training_summary_path,
    "candidate_summary": os.path.join(eval_dir, "suite_summary.csv"),
    "phase4b_acceptance_json": acceptance_path,
    "phase4b_split_comparison_json": comparison_path,
    "phase4d_policy_selection_json": selection_path,
    "phase4d_train_acceptance_json": train_acceptance_path,
    "phase4d_lifecycle_summary_json": lifecycle_path,
    "acceptance_rows": len(acceptance) if isinstance(acceptance, list) else "",
    "split_comparison_rows": len(split_comparison) if isinstance(split_comparison, list) else "",
}
lifecycle_fields = list(lifecycle.keys())
os.makedirs(comparison_dir, exist_ok=True)
with open(lifecycle_csv_path, "w", newline="") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=lifecycle_fields)
    writer.writeheader()
    writer.writerow(lifecycle)
with open(lifecycle_path, "w") as json_file:
    json.dump(lifecycle, json_file, indent=2, sort_keys=True)
    json_file.write("\n")
fields = list(row.keys())
for directory in (output_root, comparison_dir):
    os.makedirs(directory, exist_ok=True)
    csv_path = os.path.join(directory, "phase4d_no_planning_summary.csv")
    json_path = os.path.join(directory, "phase4d_no_planning_summary.json")
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    with open(json_path, "w") as json_file:
        json.dump(row, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
print("phase4d no-planning summary csv: %s" % os.path.join(output_root, "phase4d_no_planning_summary.csv"))
print("phase4d no-planning summary json: %s" % os.path.join(output_root, "phase4d_no_planning_summary.json"))
print("phase4d no-planning comparison summary csv: %s" % os.path.join(comparison_dir, "phase4d_no_planning_summary.csv"))
print("phase4d no-planning comparison summary json: %s" % os.path.join(comparison_dir, "phase4d_no_planning_summary.json"))
print("phase4d lifecycle summary json: %s" % lifecycle_path)
PY
}

seed="100"
routes_subset="00"
total_timesteps="4096"
reference_summary=""
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
      shift
      ;;
    --total-timesteps)
      if [ "$#" -lt 2 ]; then echo "missing value for --total-timesteps" >&2; exit 2; fi
      total_timesteps="$2"
      shift 2
      ;;
    --reference-summary=*)
      reference_summary="${1#--reference-summary=}"
      shift
      ;;
    --reference-summary)
      if [ "$#" -lt 2 ]; then echo "missing value for --reference-summary" >&2; exit 2; fi
      reference_summary="$2"
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

if [ -z "$reference_summary" ]; then
  echo "Phase 4-D no-planning runner requires --reference-summary" >&2
  exit 2
fi

if [ -z "$output_root" ]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  seed_label="$(sanitize_label "$seed")"
  route_label="$(sanitize_label "$routes_subset")"
  steps_label="$(sanitize_label "$total_timesteps")"
  output_root="/workspace/e2e_models/outputs/rl_suspension_phase4d_no_planning_${steps_label}_seed${seed_label}_route${route_label}_${stamp}"
fi

train_dir="${output_root%/}/train"
eval_dir="${output_root%/}/eval_s4"
comparison_dir="${output_root%/}/comparison"

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

train_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.train_real_carla_sac
  "${extra_train_args[@]}"
  --backend live
  --algorithm sac
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
  --output-dir "$comparison_dir"
)

eval_cmd=(
  "$PYTHON_BIN" transfuser_suspension_control_suite.py
  --scenarios rl_residual_skyhook
  --rl-policy "$train_dir/policy.ts"
  --rl-normalizer "$train_dir/normalizer.json"
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
  --output-dir "$eval_dir"
)

accept_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.phase4b_policy_eval_acceptance
  --candidate-summary "$eval_dir/suite_summary.csv"
  --reference-summary "$reference_summary"
  --output-dir "$comparison_dir"
)

selection_cmd=(
  "$PYTHON_BIN" -m suspension_control.rl.phase4d_policy_selection
  --candidate-summary "$eval_dir/suite_summary.csv"
  --reference-summary "$reference_summary"
  --training-summary "$train_dir/training_summary.json"
  --output-dir "$comparison_dir"
)

if [ "$print_command" -eq 1 ]; then
  echo "Phase 4-D train command:"
  printf 'TRAIN:'
  quote_cmd "${train_cmd[@]}"
  echo "Phase 4-D train acceptance command:"
  printf 'TRAIN_ACCEPTANCE:'
  quote_cmd "${train_gate_cmd[@]}" --training-return-code "<train_return_code>"
  echo "Phase 4-D S4 eval command:"
  printf 'EVAL_S4:'
  quote_cmd "${eval_cmd[@]}"
  echo "Phase 4-D split-reference acceptance command:"
  printf 'ACCEPTANCE:'
  quote_cmd "${accept_cmd[@]}"
  echo "Phase 4-D policy selection command:"
  printf 'POLICY_SELECTION:'
  quote_cmd "${selection_cmd[@]}"
  echo "SUMMARY: write phase4d_no_planning_summary.csv/json and comparison/phase4d_lifecycle_summary.json from comparison outputs"
  echo "Phase 4-D output root: $output_root"
  echo "Phase 4-D train dir: $train_dir"
  echo "Phase 4-D eval_s4 dir: $eval_dir"
  echo "Phase 4-D comparison dir: $comparison_dir"
  exit 0
fi

mkdir -p "$train_dir" "$eval_dir" "$comparison_dir"
set +e
"${train_cmd[@]}"
train_return_code=$?
set -e
"${train_gate_cmd[@]}" --training-return-code "$train_return_code"
if ! phase4d_train_gate_allows_eval "$comparison_dir/phase4d_train_acceptance.json"; then
  write_final_summary
  echo "Phase 4-D training gate failed; S4 eval skipped." >&2
  echo "Phase 4-D train acceptance JSON: $comparison_dir/phase4d_train_acceptance.json" >&2
  if [ "$train_return_code" -ne 0 ]; then
    exit "$train_return_code"
  fi
  exit 1
fi
stabilize_carla_before_eval
"${eval_cmd[@]}"
"${accept_cmd[@]}"
"${selection_cmd[@]}"
write_final_summary

echo "Phase 4-D output root: $output_root"
echo "Phase 4-D train dir: $train_dir"
echo "Phase 4-D eval_s4 dir: $eval_dir"
echo "Phase 4-D comparison dir: $comparison_dir"
echo "Phase 4-D final summary CSV: $output_root/phase4d_no_planning_summary.csv"
echo "Phase 4-D final summary JSON: $output_root/phase4d_no_planning_summary.json"
echo "Phase 4-D lifecycle summary JSON: $comparison_dir/phase4d_lifecycle_summary.json"

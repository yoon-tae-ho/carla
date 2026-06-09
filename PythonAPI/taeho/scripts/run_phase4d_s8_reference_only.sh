#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  cat <<'EOF'
Run Phase 4-D S8 zero-residual reference route-suite evaluation only.

Compatibility wrapper for:
  bash scripts/run_phase4d_eval_s8_reference.sh

Use the new official split-workflow name for new scripts.
EOF
fi

exec bash scripts/run_phase4d_eval_s8_reference.sh "$@"

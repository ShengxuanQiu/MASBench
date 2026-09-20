#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-plan}
OUT=${PART4_OUTPUT_ROOT:-"$ROOT/results/part4"}/4_5
PYTHON_BIN=${MASBENCH_PYTHON:-"$HOME/.conda/envs/MAS/bin/python"}
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=$(command -v python3)
cd "$ROOT/mas_workflow"
if [[ "$ACTION" == plan ]]; then
  "$PYTHON_BIN" -m app.mixed_study --config configs/part4/4_5_capacity_mix.json --output "$OUT/run" --validate-only
  echo "4.5 plan validated; run 4.4 first, then use: $0 run"
  exit 0
fi
ARGS=(); [[ "$ACTION" == resume ]] && ARGS+=(--resume)
"$PYTHON_BIN" -m app.mixed_study --config configs/part4/4_5_capacity_mix.json --output "$OUT/run" "${ARGS[@]}"
echo "4.5 mix/capacity results: $OUT/run/results.json"

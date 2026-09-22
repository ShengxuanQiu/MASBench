#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-plan}
OUT=${PART4_OUTPUT_ROOT:-"$ROOT/results/part4"}/4_4
PYTHON_BIN=${MASBENCH_PYTHON:-"$HOME/.conda/envs/MAS/bin/python"}
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=$(command -v python3)
cd "$ROOT/mas_workflow"
if [[ "$ACTION" == plan ]]; then
  "$PYTHON_BIN" -m app.mixed_study --config configs/part4/4_4_heterogeneous.json --output "$OUT/run" --validate-only
  echo "4.4 plan validated; use: $0 run"
  exit 0
fi
ARGS=(); [[ "$ACTION" == resume ]] && ARGS+=(--resume)
"$PYTHON_BIN" -m app.mixed_study --config configs/part4/4_4_heterogeneous.json --output "$OUT/run" "${ARGS[@]}"
"$PYTHON_BIN" -m app.freeze_corpus --input "$OUT/run" --output "$OUT/frozen" --per-class 3
echo "4.4 results: $OUT/run/results.json"
echo "Frozen replay corpus for 4.5/4.6: $OUT/frozen/index.json"

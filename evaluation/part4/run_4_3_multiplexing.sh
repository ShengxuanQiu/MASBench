#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-plan}
OUT=${PART4_OUTPUT_ROOT:-"$ROOT/results/part4"}/4_3
PYTHON_BIN=${MASBENCH_PYTHON:-"$HOME/.conda/envs/MAS/bin/python"}
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=$(command -v python3)
cd "$ROOT/mas_workflow"
if [[ "$ACTION" == plan ]]; then
  TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
  "$PYTHON_BIN" -m app.publication --manifest configs/part4/4_3_multiplexing.json --output "$TMP"
  echo "4.3 plan validated; use: $0 run"
  exit 0
fi
ARGS=(); [[ "$ACTION" == resume ]] && ARGS+=(--resume)
"$PYTHON_BIN" -m app.publication --manifest configs/part4/4_3_multiplexing.json --output "$OUT/matrix" --execute "${ARGS[@]}"
"$PYTHON_BIN" -m app.part4_report --input "$OUT/matrix" --output "$OUT/report"
echo "4.3 outputs: $OUT/report/{workflow_rows.csv,case_rows.csv,report.json}"

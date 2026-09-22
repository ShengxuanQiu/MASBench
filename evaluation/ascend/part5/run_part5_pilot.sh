#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=${PART5_PILOT_ROOT:-"$ROOT/results/part5-ascend-pilot"}
PYTHON_BIN=${MASBENCH_PYTHON:-python3}
ACTION=${1:-all}
export PYTHONPATH="$ROOT/mas_workflow:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" "$ROOT/evaluation/ascend/part5/prepare_part5_pilot.py" --repo "$ROOT" --output "$OUT"
cd "$ROOT/mas_workflow"

run_52() {
  "$PYTHON_BIN" -m app.publication --manifest "$OUT/configs/5_2.json" --output "$OUT/5_2/matrix" --execute
  "$PYTHON_BIN" -m app.part4_report --input "$OUT/5_2/matrix" --output "$OUT/5_2/report"
}

run_53() {
  "$PYTHON_BIN" -m app.publication --manifest "$OUT/configs/5_3.json" --output "$OUT/5_3/matrix" --execute
  "$PYTHON_BIN" -m app.part4_report --input "$OUT/5_3/matrix" --output "$OUT/5_3/report"
}

run_54() {
  "$PYTHON_BIN" -m app.mixed_study --config "$OUT/configs/5_4.json" --output "$OUT/5_4/run"
  "$PYTHON_BIN" -m app.freeze_corpus --input "$OUT/5_4/run" --output "$OUT/5_4/frozen" --per-class 2
}

run_55() {
  test -f "$OUT/5_4/frozen/index.json" || { echo "Run 5_4 first" >&2; exit 3; }
  "$PYTHON_BIN" -m app.mixed_study --config "$OUT/configs/5_5.json" --output "$OUT/5_5/run"
}

run_56() {
  test -f "$OUT/5_4/frozen/index.json" || { echo "Run 5_4 first" >&2; exit 3; }
  "$PYTHON_BIN" -m app.mixed_study --config "$OUT/configs/5_6.json" --output "$OUT/5_6/run"
}

finalize() {
  bash "$ROOT/evaluation/ascend/part5/run_semantic_acceptance.sh"
  "$PYTHON_BIN" "$ROOT/evaluation/ascend/part5/report_part5_pilot.py" --input "$OUT" --output "$OUT/report"
}

case "$ACTION" in
  5_2) run_52 ;;
  5_3) run_53 ;;
  5_4) run_54 ;;
  5_5) run_55 ;;
  5_6) run_56 ;;
  finalize) finalize ;;
  all) run_52; run_53; run_54; run_55; run_56; finalize ;;
  *) echo "Usage: $0 {5_2|5_3|5_4|5_5|5_6|finalize|all}" >&2; exit 2 ;;
esac

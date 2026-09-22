#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=${PART5_PRESSURE_ROOT:-"$ROOT/results/part5-pressure-v2"}
PYTHON_BIN=${MASBENCH_PYTHON:-python3}
ACTION=${1:-all}
export PYTHONPATH="$ROOT/mas_workflow:$ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" "$ROOT/evaluation/ascend/part5_pressure/prepare_pressure_experiments.py" --repo "$ROOT" --output "$OUT"
cd "$ROOT/mas_workflow"

run_52() {
  "$PYTHON_BIN" -m app.mixed_study --config "$OUT/configs/5_2_shapes.json" --output "$OUT/5_2/run"
  "$PYTHON_BIN" -m app.freeze_corpus --input "$OUT/5_2/run" --output "$OUT/5_2/frozen" --per-class 1
}

run_53() {
  "$PYTHON_BIN" -m app.mixed_study --config "$OUT/configs/5_3_workflows.json" --output "$OUT/5_3/run"
}

run_54() {
  test -f "$OUT/5_2/frozen/index.json" || { echo "Run 5_2 first" >&2; exit 3; }
  "$PYTHON_BIN" -m app.mixed_study --config "$OUT/configs/5_4_load.json" --output "$OUT/5_4/run"
}

run_55() {
  test -f "$OUT/5_2/frozen/index.json" || { echo "Run 5_2 first" >&2; exit 3; }
  "$PYTHON_BIN" -m app.mixed_study --config "$OUT/configs/5_5_mix.json" --output "$OUT/5_5/run"
}

render() {
  "$PYTHON_BIN" "$ROOT/evaluation/ascend/part5_pressure/report_pressure_experiments.py" \
    --input "$OUT" --output "$OUT/report" --model-config /model/Qwen3-8B/config.json
}

case "$ACTION" in
  5_2) run_52 ;;
  5_3) run_53 ;;
  5_4) run_54 ;;
  5_5) run_55 ;;
  render) render ;;
  all) run_52; run_53; run_54; run_55; render ;;
  *) echo "Usage: $0 {5_2|5_3|5_4|5_5|render|all}" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

NUM_INSTANCES=5
LLM_MODE=mock
TOOL_MODE=synthetic
LATENCY_PROFILE=medium
LATENCY_SCALE=1.0
TRACE_DIR=traces

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-instances) NUM_INSTANCES="$2"; shift 2 ;;
    --llm-mode) LLM_MODE="$2"; shift 2 ;;
    --tool-mode) TOOL_MODE="$2"; shift 2 ;;
    --latency-profile) LATENCY_PROFILE="$2"; shift 2 ;;
    --latency-scale) LATENCY_SCALE="$2"; shift 2 ;;
    --trace-dir) TRACE_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

for topology in single independent centralized decentralized hybrid; do
  python -m app.main \
    --topology "$topology" \
    --task-source swebench_lite \
    --swebench-split test \
    --swebench-start-index 0 \
    --swebench-num-instances "$NUM_INSTANCES" \
    --llm-mode "$LLM_MODE" \
    --tool-mode "$TOOL_MODE" \
    --latency-profile "$LATENCY_PROFILE" \
    --latency-scale "$LATENCY_SCALE" \
    --trace-dir "$TRACE_DIR" \
    --random-seed 42
done

python -m app.analyze_trace "$TRACE_DIR" --out-dir "$TRACE_DIR" --summary-name summary_week1.json
echo "Summary: $TRACE_DIR/summary_week1.json"

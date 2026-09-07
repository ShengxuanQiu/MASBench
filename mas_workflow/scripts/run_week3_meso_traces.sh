#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TRACE_DIR="${TRACE_DIR:-traces/week3}"
REPEAT="${REPEAT:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND_BASE_URL="${BACKEND_BASE_URL:-http://127.0.0.1:8000/v1}"
BACKEND_METRICS_URL="${BACKEND_METRICS_URL:-http://127.0.0.1:8000/metrics}"
MODEL="${MODEL:-local-mas-model}"
LLM_MODE="${LLM_MODE:-openai_compatible}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-256}"
REPLAY_SNAPSHOT_DIR="${REPLAY_SNAPSHOT_DIR:-}"

mkdir -p "$TRACE_DIR"

tool_args=()
if [[ -n "$REPLAY_SNAPSHOT_DIR" && -d "$REPLAY_SNAPSHOT_DIR" ]]; then
  tool_args=(--tool-mode replay --search-provider recorded --replay-snapshot-dir "$REPLAY_SNAPSHOT_DIR" --tool-trace-replay-path "$REPLAY_SNAPSHOT_DIR")
else
  tool_args=(--tool-mode synthetic --search-provider synthetic --allow-synthetic-tools true)
fi

base_args=(
  --mode legacy_motif
  --task-source manual
  --llm-mode "$LLM_MODE"
  --backend-base-url "$BACKEND_BASE_URL"
  --backend-metrics-url "$BACKEND_METRICS_URL"
  --model "$MODEL"
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
  --max-concurrent-llm-calls "${MAX_CONCURRENT_LLM_CALLS:-8}"
  --trace-dir "$TRACE_DIR"
  --trace-level arch
  --export-trace-views true
  --collect-backend-metrics true
  --record-model-outputs false
)

run_workload() {
  local workload="$1"
  local query="$2"
  shift 2
  echo "[week3] workload=$workload args=$*"
  "$PYTHON_BIN" -m mas_workflow.app.main \
    "${base_args[@]}" \
    "${tool_args[@]}" \
    --workload "$workload" \
    --motif "$workload" \
    --query "$query" \
    "$@"
}

for ((r = 1; r <= REPEAT; r++)); do
  for width in 1 2 4; do
    for phase in overlap_reviewer overlap_finalizer; do
      run_workload tool_resume_contention_meso \
        "Week3 replayable tool resume contention run r=$r width=$width phase=$phase" \
        --tool-branch-width "$width" \
        --resume-phase-policy "$phase" \
        --critical-stage-marker "${CRITICAL_STAGE_MARKER:-reviewer}" \
        --background-resume-enabled true
    done
  done

  for groups in 2 4; do
    for agents in 2 4; do
      run_workload hierarchical_synthesis_pressure_meso \
        "Week3 hierarchical synthesis pressure run r=$r groups=$groups agents=$agents" \
        --group-count "$groups" \
        --agents-per-group "$agents"
    done
  done

  for agents in 2 4 6; do
    for rounds in 1 2; do
      run_workload debate_allgather_pressure_meso \
        "Week3 debate all-gather pressure run r=$r agents=$agents rounds=$rounds" \
        --num-agents "$agents" \
        --debate-rounds "$rounds"
    done
  done

  for depth in 1 2 3; do
    run_workload retry_debug_pressure_meso \
      "Week3 retry debug pressure run r=$r depth=$depth" \
      --max-retries "$depth"
  done

  for writers in 2 4; do
    for readers in 2 4; do
      run_workload shared_memory_fanin_meso \
        "Week3 shared memory fan-in pressure run r=$r writers=$writers readers=$readers" \
        --writer-count "$writers" \
        --reader-count "$readers"
    done
  done
done

echo "{\"trace_dir\":\"$TRACE_DIR\",\"repeat\":$REPEAT,\"tool_mode\":\"${tool_args[1]}\"}" > "$TRACE_DIR/week3_meso_trace_run_summary.json"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TRACE_DIR="${TRACE_DIR:-progress/week3/traces}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND_BASE_URL="${BACKEND_BASE_URL:-http://127.0.0.1:8000/v1}"
BACKEND_METRICS_URL="${BACKEND_METRICS_URL:-http://127.0.0.1:8000/metrics}"
MODEL="${MODEL:-local-mas-model}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-384}"
MAX_CONCURRENT_LLM_CALLS="${MAX_CONCURRENT_LLM_CALLS:-12}"
BACKEND_METRICS_INTERVAL_SEC="${BACKEND_METRICS_INTERVAL_SEC:-0.25}"

if [[ -z "${TAVILY_API_KEY:-}" ]]; then
  echo "ERROR: TAVILY_API_KEY is required; Week3 simulator-ready traces must not use synthetic tools." >&2
  exit 2
fi

mkdir -p "$TRACE_DIR"

base_args=(
  --task-source manual
  --llm-mode openai_compatible
  --backend-base-url "$BACKEND_BASE_URL"
  --backend-metrics-url "$BACKEND_METRICS_URL"
  --model "$MODEL"
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
  --max-concurrent-llm-calls "$MAX_CONCURRENT_LLM_CALLS"
  --trace-dir "$TRACE_DIR"
  --trace-level arch
  --export-trace-views true
  --collect-backend-metrics true
  --backend-metrics-interval-sec "$BACKEND_METRICS_INTERVAL_SEC"
  --record-model-outputs false
  --tool-mode live
  --search-provider tavily
  --allow-synthetic-tools false
)

run_topology() {
  local topology="$1"
  local query="$2"
  shift 2
  echo "[week3-sim] topology=$topology"
  "$PYTHON_BIN" -m mas_workflow.app.main \
    --mode topology \
    --topology "$topology" \
    --query "$query" \
    "${base_args[@]}" \
    "$@"
}

run_motif() {
  local motif="$1"
  local query="$2"
  shift 2
  echo "[week3-sim] motif=$motif"
  "$PYTHON_BIN" -m mas_workflow.app.main \
    --mode motif \
    --motif "$motif" \
    --workload "$motif" \
    --query "$query" \
    "${base_args[@]}" \
    "$@"
}

LONG_CONTEXT="Analyze simulator-ready MAS workflow tracing. Include graph dependencies, role-specific prompts, barriers, prefix/cache reuse opportunities, and serving bottlenecks. Keep enough context so later reviewer and finalizer prompts carry repeated shared material."

run_motif evidence_collection "$LONG_CONTEXT Independent multi-branch evidence collection with live web search in parallel branches." \
  --num-agents 4

run_motif tool_specialist_team "$LONG_CONTEXT Manager-worker tool-specialist coordination with live web search specialists." \
  --max-selected-agents 4

run_motif retry_debug_pressure_meso "$LONG_CONTEXT Review-loop run with three verifier/debugger/reviser iterations." \
  --max-retries 3

run_motif debate_allgather_pressure_meso "$LONG_CONTEXT Debate all-gather run with repeated peer-message synchronization." \
  --num-agents 6 \
  --debate-rounds 3

run_motif hierarchical_synthesis_pressure_meso "$LONG_CONTEXT Hierarchical manager-worker fan-in with live web context grounding." \
  --group-count 4 \
  --agents-per-group 4

run_motif tool_resume_contention_meso "$LONG_CONTEXT Live Tavily tool-resume contention run with four non-critical tool branches." \
  --tool-branch-width 4 \
  --resume-phase-policy overlap_reviewer \
  --critical-stage-marker reviewer \
  --background-resume-enabled true

run_motif shared_memory_fanin_meso "$LONG_CONTEXT Shared memory fan-in run with two live web writers and four readers." \
  --writer-count 2 \
  --reader-count 4

cat > "$TRACE_DIR/week3_simulator_ready_run_summary.json" <<JSON
{
  "trace_dir": "$TRACE_DIR",
  "llm_mode": "openai_compatible",
  "tool_mode": "live",
  "search_provider": "tavily",
  "synthetic_trace_used": false,
  "planned_live_tavily_calls": "bounded by workflow graph; every workflow includes live web search",
  "max_output_tokens": $MAX_OUTPUT_TOKENS,
  "max_concurrent_llm_calls": $MAX_CONCURRENT_LLM_CALLS
}
JSON

#!/usr/bin/env bash
set -u
set -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

if [[ -z "${TAVILY_API_KEY:-}" && -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND_BASE_URL="${BACKEND_BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-local-mas-model}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-1024}"
REACT_MAX_STEPS="${REACT_MAX_STEPS:-8}"
TRACE_DIR="${TRACE_DIR:-traces}"
REPLAY_SNAPSHOT_DIR="${REPLAY_SNAPSHOT_DIR:-}"
SUMMARY_PATH="$TRACE_DIR/motif_real_trace_summary.json"
mkdir -p "$TRACE_DIR"

if [[ -n "$REPLAY_SNAPSHOT_DIR" ]]; then
  TOOL_ARGS=(--tool-mode replay --search-provider recorded --replay-snapshot-dir "$REPLAY_SNAPSHOT_DIR")
else
  TOOL_ARGS=(--tool-mode live --search-provider tavily)
  if [[ -z "${TAVILY_API_KEY:-}" ]]; then
    cat > "$SUMMARY_PATH" <<JSON
{
  "success": false,
  "failure_reason": "TAVILY_API_KEY is not set and REPLAY_SNAPSHOT_DIR was not provided; refusing synthetic fallback for real trace tests.",
  "motifs": {}
}
JSON
    echo "ERROR: TAVILY_API_KEY is required for live Tavily tests, or set REPLAY_SNAPSHOT_DIR for replay mode." >&2
    echo "Summary: $SUMMARY_PATH" >&2
    exit 2
  fi
fi

MOTIFS=(
  planner_executor
  evidence_collection
  researcher_synthesizer
  generator_verifier
  coder_reviewer
  multi_coder_branch
  debate_reviewer
  tool_specialist_team
  all_gather_round
  shared_evidence_store
  retry_debug_loop
  router_handoff
)

declare -A QUERIES=(
  [planner_executor]="为 MAS benchmark trace 采集制定计划并执行"
  [evidence_collection]="收集 MAS benchmark 的相关证据"
  [researcher_synthesizer]="研究 MAS benchmark 的设计取舍并综合"
  [generator_verifier]="生成并验证一个 MAS benchmark 设计方案"
  [coder_reviewer]="编写并评审一个 MAS trace schema 扩展方案"
  [multi_coder_branch]="让多个 coder 给出 MAS workload runner 方案并选择"
  [debate_reviewer]="评审一个多智能体 benchmark 设计方案"
  [tool_specialist_team]="用工具团队收集 MAS benchmark 证据"
  [all_gather_round]="多智能体共享各自 MAS benchmark 观点并聚合"
  [shared_evidence_store]="多个写入者和读取者共享 MAS benchmark 证据"
  [retry_debug_loop]="执行、测试并调试一个 MAS trace 导出任务"
  [router_handoff]="根据任务路由到合适的 MAS specialist"
)

declare -A EXTRA_ARGS=(
  [debate_reviewer]="--num-agents 3 --debate-rounds 2 --communication-topology all_to_all"
  [all_gather_round]="--num-agents 3 --communication-topology all_to_all"
  [retry_debug_loop]="--max-retries 1"
  [coder_reviewer]="--max-retries 1"
  [generator_verifier]="--max-retries 1"
  [tool_specialist_team]="--max-selected-agents 4"
)

RESULTS_JSONL="$(mktemp)"

for motif in "${MOTIFS[@]}"; do
  log_file="$(mktemp)"
  query="${QUERIES[$motif]}"
  read -r -a extra <<< "${EXTRA_ARGS[$motif]:-}"
  echo "Running motif: $motif"
  set +e
  "$PYTHON_BIN" -m mas_workflow.app.main \
    --mode motif \
    --motif "$motif" \
    --task-source manual \
    --query "$query" \
    --llm-mode openai_compatible \
    --backend-base-url "$BACKEND_BASE_URL" \
    --model "$MODEL" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" \
    --agent-execution react \
    --react-max-steps "$REACT_MAX_STEPS" \
    "${TOOL_ARGS[@]}" \
    --trace-level arch \
    --export-trace-views true \
    --collect-backend-metrics true \
    --record-model-outputs true \
    "${extra[@]}" >"$log_file" 2>&1
  status=$?
  set -e
  trace_path="$(sed -n 's/.*trace=//p' "$log_file" | tail -1)"
  "$PYTHON_BIN" - "$motif" "$status" "$trace_path" "$log_file" "$RESULTS_JSONL" <<'PY'
import json
import sys
from pathlib import Path

motif, status_s, trace_s, log_s, out_s = sys.argv[1:6]
status = int(status_s)
trace_path = Path(trace_s) if trace_s else None
log_text = Path(log_s).read_text(encoding="utf-8", errors="replace")
result = {
    "motif": motif,
    "run_id": "",
    "trace_path": str(trace_path or ""),
    "success": False,
    "llm_event_count": 0,
    "tool_event_count": 0,
    "edge_event_count": 0,
    "total_input_tokens_est": 0,
    "total_output_tokens_est": 0,
    "total_tool_time": 0.0,
    "backend_metrics_exists": False,
    "failure_reason": "",
}
try:
    if status != 0:
        raise AssertionError(f"command failed with exit {status}: {log_text[-2000:]}")
    if trace_path is None or not trace_path.exists():
        raise AssertionError("jsonl trace does not exist")
    summary_path = trace_path.with_name(trace_path.stem + "_summary.json")
    viewer_path = trace_path.with_name(trace_path.stem + "_viewer.html")
    backend_path = trace_path.with_name(trace_path.stem + "_backend_metrics.json")
    for path, label in [(summary_path, "summary"), (viewer_path, "viewer"), (backend_path, "backend_metrics")]:
        if not path.exists():
            raise AssertionError(f"{label} file does not exist: {path}")
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result["run_id"] = events[0].get("run_id", "") if events else ""
    llm = [e for e in events if e.get("event_type") == "llm_request_end"]
    tools = [e for e in events if str(e.get("event_type", "")).startswith("tool_")]
    edges = [e for e in events if e.get("node_type") == "edge"]
    result.update(
        llm_event_count=len(llm),
        tool_event_count=len(tools),
        edge_event_count=len(edges),
        total_input_tokens_est=sum(int(e.get("input_tokens_est") or 0) for e in llm),
        total_output_tokens_est=sum(int(e.get("output_tokens_est") or 0) for e in llm),
        total_tool_time=round(sum(float(e.get("effective_duration_sec") or e.get("duration_sec") or 0) for e in tools), 6),
        backend_metrics_exists=backend_path.exists(),
    )
    if not any(e.get("motif_name") == motif and e.get("motif_instance_id") for e in events):
        raise AssertionError("motif_name or motif_instance_id missing")
    if llm and not all(int(e.get("input_tokens_est") or 0) > 0 and int(e.get("output_tokens_est") or 0) >= 0 for e in llm):
        raise AssertionError("LLM token fields missing")
    if motif in {"evidence_collection", "tool_specialist_team", "shared_evidence_store"} and not tools:
        raise AssertionError("tool-heavy motif has no tool event")
    if motif in {"debate_reviewer", "all_gather_round"} and not any(e.get("artifact_type") == "peer_message" or e.get("transfer_type") == "broadcast" for e in events):
        raise AssertionError("debate/all_gather motif has no peer_message or broadcast edge")
    if motif == "retry_debug_loop" and not any(int(e.get("retry_count") or 0) > 0 or int(e.get("debug_loop_count") or 0) > 0 for e in events):
        raise AssertionError("retry_debug_loop has no retry/debug loop marker")
    if motif == "shared_evidence_store" and not any(e.get("event_type") in {"memory_read", "memory_write"} or e.get("artifact_type") in {"memory_read", "memory_write"} for e in events):
        raise AssertionError("shared_evidence_store has no memory read/write event")
    if motif == "router_handoff" and not any(e.get("selected_route") and e.get("candidate_routes") for e in events):
        raise AssertionError("router_handoff missing route metadata")
    result["success"] = True
except Exception as exc:
    result["failure_reason"] = str(exc)

with Path(out_s).open("a", encoding="utf-8") as f:
    f.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
PY
  rm -f "$log_file"
done

"$PYTHON_BIN" - "$RESULTS_JSONL" "$SUMMARY_PATH" <<'PY'
import json
import sys
from pathlib import Path

rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if line.strip()]
payload = {
    "success": all(row.get("success") for row in rows),
    "motif_count": len(rows),
    "motifs": {row["motif"]: {k: v for k, v in row.items() if k != "motif"} for row in rows},
}
Path(sys.argv[2]).write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(f"Summary: {sys.argv[2]}")
if not payload["success"]:
    sys.exit(1)
PY

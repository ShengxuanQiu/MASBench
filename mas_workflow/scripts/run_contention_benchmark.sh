#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
REPO_PATH="${REPO_PATH:-}"
QUERY="${QUERY:-修复这个仓库中导致测试失败的 bug，并输出 root cause、patch 说明和验证建议}"
TEST_COMMAND="${TEST_COMMAND:-pytest -q}"
MAX_CONCURRENT_LLM_CALLS="${MAX_CONCURRENT_LLM_CALLS:-2}"
TOOL_DELAY_PROFILE="${TOOL_DELAY_PROFILE:-medium}"
BURST_TOOL_RETURN="${BURST_TOOL_RETURN:-true}"
COMMITTEE_WIDTH="${COMMITTEE_WIDTH:-3}"
MAX_RETRIES="${MAX_RETRIES:-1}"
ENABLE_ROUND2="${ENABLE_ROUND2:-true}"

if [[ -z "${REPO_PATH}" ]]; then
  cat <<'USAGE'
用法：
  REPO_PATH=/path/to/repo scripts/run_contention_benchmark.sh

可选环境变量：
  PYTHON_BIN=python
  QUERY="..."
  TEST_COMMAND="pytest -q"
  MAX_CONCURRENT_LLM_CALLS=2
  TOOL_DELAY_PROFILE=medium
  BURST_TOOL_RETURN=true
  COMMITTEE_WIDTH=3
  MAX_RETRIES=1
  ENABLE_ROUND2=true
USAGE
  exit 2
fi

cd "${ROOT_DIR}"

run_one() {
  local policy="$1"
  echo "===== 运行 ${policy} ====="
  "${PYTHON_BIN}" -m app.main \
    --repo "${REPO_PATH}" \
    --query "${QUERY}" \
    --test-command "${TEST_COMMAND}" \
    --max-concurrent-llm-calls "${MAX_CONCURRENT_LLM_CALLS}" \
    --dispatch-policy "${policy}" \
    --tool-delay-profile "${TOOL_DELAY_PROFILE}" \
    --burst-tool-return "${BURST_TOOL_RETURN}" \
    --committee-width "${COMMITTEE_WIDTH}" \
    --max-retries "${MAX_RETRIES}" \
    --enable-round2 "${ENABLE_ROUND2}"

  local trace_path
  trace_path="$(ls -t logs/run_*.json | head -n 1)"
  "${PYTHON_BIN}" -m app.analyze_trace "${trace_path}"
  local summary_path
  summary_path="logs/analysis_summary_${trace_path#logs/run_}"
  echo "===== ${policy} summary: ${summary_path} ====="
  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
summary = json.loads(Path("${summary_path}").read_text(encoding="utf-8"))
keys = [
    "end_to_end_latency",
    "total_llm_time",
    "total_tool_time",
    "total_queue_wait",
    "critical_queue_wait_time",
    "priority_inversion_time",
    "max_ready_queue_size",
    "contention_window_count",
    "total_barrier_wait",
    "redundant_prefill_tokens_estimated",
]
for key in keys:
    print(f"{key}: {summary.get(key)}")
PY
}

run_one fcfs
run_one criticality

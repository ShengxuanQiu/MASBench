#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${WORKFLOW_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/home/shegnxuanqiu/.conda/envs/MAS/bin/python}"
TRACE_DIR="${TRACE_DIR:-traces/week6_real/main}"
REPETITIONS="${REPETITIONS:-5}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_NAME="${MODEL_NAME:-local-mas-model}"
QUERY="Analyze how workflow graphs and tool-return traffic expose serving interference in multi-agent systems."

cd "${WORKFLOW_DIR}"

"${PYTHON_BIN}" - <<PY
from app.llm_client import LocalLLMClient
ok, detail = LocalLLMClient(model="${MODEL_NAME}", base_url="${BASE_URL}").is_available(timeout=5)
if not ok:
    raise SystemExit(f"vLLM endpoint unavailable: {detail}")
PY

for repetition in $(seq 1 "${REPETITIONS}"); do
  for policy in default_vllm critical_path_aware; do
    "${PYTHON_BIN}" -m app.main \
      --workload tool_resume_contention_meso \
      --query "${QUERY}" \
      --llm-mode openai_compatible \
      --tool-mode live \
      --search-provider tavily \
      --force-live-search-test true \
      --allow-synthetic-tools false \
      --model "${MODEL_NAME}" \
      --backend-base-url "${BASE_URL}" \
      --max-output-tokens 512 \
      --tool-branch-width 4 \
      --max-concurrent-llm-calls 8 \
      --admission-policy "${policy}" \
      --max-defer-sec 30 \
      --collect-backend-metrics true \
      --backend-metrics-interval-sec 0.1 \
      --record-model-outputs false \
      --trace-dir "${TRACE_DIR}"
  done
done

echo "Completed ${REPETITIONS} paired real-execution repetitions in ${REPO_DIR}/mas_workflow/${TRACE_DIR}."

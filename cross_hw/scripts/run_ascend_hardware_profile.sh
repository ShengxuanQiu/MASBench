#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

REPEATS="${REPEATS:-2}"
PROFILE_ID="${PROFILE_ID:-ascend_qwen3_8b_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="cross_hw/raw_runs/${PROFILE_ID}/runs"
LOG_DIR="cross_hw/raw_runs/${PROFILE_ID}"
mkdir -p "${OUTPUT_ROOT}"

WORKLOADS=(
  single_agent_control
  independent_fanin
  centralized_manager_worker
  debate_allgather_pressure_meso
  shared_memory_fanin_meso
  retry_debug_loop
  hierarchical_synthesis_pressure_meso
  issue_to_patch_workflow
)

jq -n \
  --arg profile_id "${PROFILE_ID}" \
  --arg platform "Ascend 910" \
  --arg model "Qwen3-8B" \
  --arg endpoint "http://127.0.0.1:8000/v1" \
  --arg tool_mode "recorded_tavily" \
  --argjson repeats "${REPEATS}" \
  '{profile_id:$profile_id,platform:$platform,model:$model,endpoint:$endpoint,tool_mode:$tool_mode,repeats:$repeats,modes:["raw"],started_at:(now|todate)}' \
  > "${LOG_DIR}/manifest.json"

for repeat in $(seq 1 "${REPEATS}"); do
  for workload in "${WORKLOADS[@]}"; do
    run_id="ascend_r${repeat}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    echo "[$(date -Is)] START repeat=${repeat} workload=${workload} run_id=${run_id}"
    python -m cross_hw.workflow \
      --config cross_hw/configs/ascend_qwen3_8b.json \
      --workload "${workload}" \
      --mode raw \
      --run-id "${run_id}" \
      --output-root "${OUTPUT_ROOT}"
    echo "[$(date -Is)] DONE repeat=${repeat} workload=${workload} run_id=${run_id}"
  done
done

jq '. + {completed_at:(now|todate)}' "${LOG_DIR}/manifest.json" > "${LOG_DIR}/manifest.tmp"
mv "${LOG_DIR}/manifest.tmp" "${LOG_DIR}/manifest.json"
echo "PROFILE_COMPLETE=${PROFILE_ID}"

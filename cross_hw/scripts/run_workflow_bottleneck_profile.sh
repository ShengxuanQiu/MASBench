#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PROFILE_ID="${PROFILE_ID:-ascend_bottleneck_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="cross_hw/raw_runs/${PROFILE_ID}"
REPEATS="${REPEATS:-1}"
mkdir -p "${OUTPUT_ROOT}"

sampler_pid=""
sampler_stop_file=""
cleanup_sampler() {
  if [[ -n "${sampler_pid}" ]] && kill -0 "${sampler_pid}" 2>/dev/null; then
    touch "${sampler_stop_file}"
    wait "${sampler_pid}" || true
  fi
}
trap cleanup_sampler EXIT INT TERM

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

for repeat in $(seq 1 "${REPEATS}"); do
  for workload in "${WORKLOADS[@]}"; do
    run_id="ascend_bottleneck_r${repeat}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
    run_root="${OUTPUT_ROOT}/${workload}/raw/${run_id}"
    telemetry_root="${OUTPUT_ROOT}/telemetry/${workload}/${run_id}"
    mkdir -p "${telemetry_root}"
    python cross_hw/scripts/xpu_telemetry.py \
      --output "${telemetry_root}/telemetry.csv" \
      --stop-file "${telemetry_root}/sampler.stop" \
      --ready-file "${telemetry_root}/sampler.ready" \
      --interval 0.25 &
    sampler_pid=$!
    sampler_stop_file="${telemetry_root}/sampler.stop"
    while [[ ! -f "${telemetry_root}/sampler.ready" ]]; do sleep 0.1; done
    python -m cross_hw.workflow \
      --config cross_hw/configs/ascend_qwen3_8b.json \
      --workload "${workload}" --mode raw --run-id "${run_id}" \
      --output-root "${OUTPUT_ROOT}"
    touch "${telemetry_root}/sampler.stop"
    wait "${sampler_pid}"
    sampler_pid=""
    sampler_stop_file=""
    mv "${telemetry_root}/telemetry.csv" "${run_root}/telemetry.csv"
    rm -f "${telemetry_root}/sampler.ready" "${telemetry_root}/sampler.stop"
    echo "[$(date -Is)] completed ${workload} repeat=${repeat}"
  done
done

printf '{"profile_id":"%s","platform":"Ascend 910C","repeats":%s,"completed_at":"%s"}\n' \
  "${PROFILE_ID}" "${REPEATS}" "$(date -Is)" > "${OUTPUT_ROOT}/manifest.json"
echo "PROFILE_COMPLETE=${PROFILE_ID}"

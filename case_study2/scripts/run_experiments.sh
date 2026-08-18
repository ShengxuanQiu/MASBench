#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

set -a
source .env
set +a

REPEATS="${REPEATS:-2}"
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
  if (( repeat % 2 == 1 )); then
    MODES=(raw producer)
  else
    MODES=(producer raw)
  fi
  for workload in "${WORKLOADS[@]}"; do
    for mode in "${MODES[@]}"; do
      run_id="r${repeat}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
      /opt/anaconda3/bin/conda run -n MAS --no-capture-output \
        python -m case_study2.workflow --workload "${workload}" --mode "${mode}" --run-id "${run_id}"
    done
  done
done

/opt/anaconda3/bin/conda run -n MAS --no-capture-output python -m case_study2.analyze

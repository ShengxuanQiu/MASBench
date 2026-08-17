#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

set -a
source .env
set +a

REPEATS="${REPEATS:-2}"
WORKLOADS=(
  debate_allgather_pressure_meso
  shared_memory_fanin_meso
  hierarchical_synthesis_pressure_meso
)

for repeat in $(seq 1 "${REPEATS}"); do
  for workload in "${WORKLOADS[@]}"; do
    for mode in raw structured; do
      run_id="r${repeat}_$(date +%Y%m%d_%H%M%S)_${RANDOM}"
      /opt/anaconda3/bin/conda run -n MAS --no-capture-output \
        python -m case_study2.workflow --workload "${workload}" --mode "${mode}" --run-id "${run_id}"
    done
  done
done

/opt/anaconda3/bin/conda run -n MAS --no-capture-output python -m case_study2.analyze

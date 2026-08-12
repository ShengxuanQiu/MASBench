#!/usr/bin/env bash
set -euo pipefail

CS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CS_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
OUT_DIR="${CS_DIR}/artifacts/multiconfig_3x/runs"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MAS
cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

# fanout_2 already has one pair in this server session. Collect two more pairs,
# then three complete fanout_4 pairs with alternating policy order.
entries=(
  "fanout_2.json default_vllm" "fanout_2.json critical_frontier"
  "fanout_4.json default_vllm" "fanout_4.json critical_frontier"
  "fanout_2.json critical_frontier" "fanout_2.json default_vllm"
  "fanout_4.json critical_frontier" "fanout_4.json default_vllm"
  "fanout_4.json default_vllm" "fanout_4.json critical_frontier"
)

for index in "${!entries[@]}"; do
  read -r config policy <<<"${entries[$index]}"
  printf 'CASE_STUDY1_PAPER_MOTIF %d/%d config=%s policy=%s\n' "$((index + 1))" "${#entries[@]}" "${config}" "${policy}"
  python -m case_study1.workflow \
    --policy "${policy}" \
    --config "${CS_DIR}/configs/${config}" \
    --out-dir "${OUT_DIR}"
done

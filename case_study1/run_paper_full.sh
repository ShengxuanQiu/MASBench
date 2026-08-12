#!/usr/bin/env bash
set -euo pipefail

CS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CS_DIR}/.." && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROOT_DIR}/.env"
  set +a
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MAS
cd "${ROOT_DIR}"

policies=(
  default_vllm critical_path_aware
  critical_path_aware default_vllm
  default_vllm critical_path_aware
)
for index in "${!policies[@]}"; do
  policy="${policies[$index]}"
  printf 'CASE_STUDY1_FULL %d/%d policy=%s\n' "$((index + 1))" "${#policies[@]}" "${policy}"
  python -m case_study1.run_full_workflow --policy "${policy}"
done

#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
mkdir -p "${LOG_DIR}"

ENV_FILE="${ROOT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# This container already provides vLLM, vLLM Ascend, torch-npu and CANN.
# Select one NPU by default; pass ASCEND_RT_VISIBLE_DEVICES=0,1 and TP_SIZE=2
# when a tensor-parallel experiment is needed.
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-${NPU_DEVICE:-0}}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-512}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE="${VLLM_ASCEND_ENABLE_DENSE_OPTIMIZE:-1}"

# Local API calls must bypass any proxy inherited by the container.
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-${NO_PROXY}}"

MODEL_PATH="${MODEL_PATH:-/model/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-mas-model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
BLOCK_SIZE="${BLOCK_SIZE:-128}"
MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.85}"
SEED="${SEED:-42}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-true}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-false}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}}"
ENFORCE_EAGER="${ENFORCE_EAGER:-false}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "Model not found: ${MODEL_PATH}/config.json" >&2
  echo "Download Qwen3-8B to /model/Qwen3-8B or set MODEL_PATH explicitly." >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not available in PATH." >&2
  exit 1
fi

(
cd /
python - <<'PY'
from importlib.metadata import version

import torch
import torch_npu  # noqa: F401
import vllm_ascend  # noqa: F401
from vllm.platforms import current_platform

if not torch.npu.is_available():
    raise SystemExit("torch.npu.is_available() is False")
print(f"vLLM {version('vllm')}; platform: {current_platform.device_type}; "
      f"visible NPU count: {torch.npu.device_count()}")
PY
)

export MAS_VLLM_LAUNCH_CONFIG_PATH="${MAS_VLLM_LAUNCH_CONFIG_PATH:-${LOG_DIR}/vllm_launch_config.json}"
python - <<PY
import json
from pathlib import Path

payload = {
    "backend_runtime": "vllm-ascend",
    "device_type": "npu",
    "vendor": "huawei",
    "device_model": "Ascend 910",
    "visible_devices": "${ASCEND_RT_VISIBLE_DEVICES}",
    "model_path": "${MODEL_PATH}",
    "served_model_name": "${SERVED_MODEL_NAME}",
    "tensor_parallel_size": int("${TP_SIZE}"),
    "pipeline_parallel_size": int("${PP_SIZE}"),
    "data_parallel_size": int("${DP_SIZE}"),
    "dtype": "${DTYPE}",
    "max_model_len": int("${MAX_MODEL_LEN}"),
    "max_num_seqs": int("${MAX_NUM_SEQS}"),
    "max_num_batched_tokens": int("${MAX_NUM_BATCHED_TOKENS}"),
    "block_size": int("${BLOCK_SIZE}"),
    "memory_utilization": float("${MEMORY_UTILIZATION}"),
    "launch_config_path": "${MAS_VLLM_LAUNCH_CONFIG_PATH}",
}
path = Path("${MAS_VLLM_LAUNCH_CONFIG_PATH}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
PY

CMD=(
  vllm serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --pipeline-parallel-size "${PP_SIZE}"
  --data-parallel-size "${DP_SIZE}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --block-size "${BLOCK_SIZE}"
  --gpu-memory-utilization "${MEMORY_UTILIZATION}"
  --seed "${SEED}"
  --reasoning-parser qwen3
  --enable-request-id-headers
  --compilation-config "${COMPILATION_CONFIG}"
)

if [[ "${ENABLE_PREFIX_CACHING}" == "true" ]]; then
  CMD+=(--enable-prefix-caching)
fi

if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "true" ]]; then
  CMD+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi

if [[ "${ENFORCE_EAGER}" == "true" ]]; then
  CMD+=(--enforce-eager)
fi

echo "Python: $(command -v python)"
python --version
echo "vLLM: $(command -v vllm)"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "PORT=${PORT}"
echo "TP/PP/DP=${TP_SIZE}/${PP_SIZE}/${DP_SIZE}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "BLOCK_SIZE=${BLOCK_SIZE}"
echo "MEMORY_UTILIZATION=${MEMORY_UTILIZATION}"
echo "MAS_VLLM_LAUNCH_CONFIG_PATH=${MAS_VLLM_LAUNCH_CONFIG_PATH}"
printf 'Launching:'
printf ' %q' "${CMD[@]}"
echo

# An uninitialized `vllm/` submodule exists at the repository root. Starting
# the console entry point from there can shadow the container's installed
# editable vLLM package, so launch from a neutral working directory.
cd /
"${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/vllm_ascend_server.log"

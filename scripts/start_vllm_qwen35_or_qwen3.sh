#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

ENV_FILE="${ROOT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# 优先激活 conda 环境 MAS；如果 conda 激活后的 PATH 被 base 覆盖，则手动把 MAS/bin 放到最前。
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -qx "MAS"; then
      conda activate MAS
    fi
  fi
fi

MAS_ENV_PREFIX="$(conda env list 2>/dev/null | awk '$1 == "MAS" {print $NF; exit}')"
if [[ -n "${MAS_ENV_PREFIX}" && -d "${MAS_ENV_PREFIX}/bin" ]]; then
  export PATH="${MAS_ENV_PREFIX}/bin:${PATH}"
elif [[ -f "${ROOT_DIR}/.venv-MAS/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv-MAS/bin/activate"
fi

DEFAULT_QWEN35="/data/models/Qwen3.5-35B-A3B"
FALLBACK_QWEN3="/data/models/Qwen3-8B"
LEGACY_QWEN35="/opt/models/Qwen3.5-4B/"
LEGACY_QWEN3="/data1/pretrained_models/Qwen3-8B/"

MODEL_PATH="${MODEL_PATH:-${DEFAULT_QWEN35}}"
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "默认模型路径不存在：${MODEL_PATH}"
  MODEL_PATH="${FALLBACK_QWEN3}"
  echo "切换到 fallback 模型：${MODEL_PATH}"
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  MODEL_PATH="${LEGACY_QWEN35}"
  echo "切换到 legacy fallback 模型：${MODEL_PATH}"
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  MODEL_PATH="${LEGACY_QWEN3}"
  echo "切换到 legacy fallback 模型：${MODEL_PATH}"
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "模型路径不存在：${MODEL_PATH}" >&2
  exit 1
fi

PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
HOST="${HOST:-0.0.0.0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-mas-model}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-false}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
DTYPE="${DTYPE:-bfloat16}"
KV_BLOCK_SIZE="${KV_BLOCK_SIZE:-16}"
GPU_COUNT="${GPU_COUNT:-$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)}"
export VLLM_MAS_TRACE_PATH="${VLLM_MAS_TRACE_PATH:-${LOG_DIR}/vllm_mas_backend_trace.jsonl}"
export MAS_VLLM_LAUNCH_CONFIG_PATH="${MAS_VLLM_LAUNCH_CONFIG_PATH:-${LOG_DIR}/vllm_launch_config.json}"

python - <<PY
import json
from pathlib import Path

payload = {
    "backend_runtime": "vllm",
    "device_type": "gpu",
    "vendor": "nvidia",
    "model_path": "${MODEL_PATH}",
    "served_model_name": "${SERVED_MODEL_NAME}",
    "tensor_parallel_size": int("${TP_SIZE}"),
    "pipeline_parallel_size": int("${PP_SIZE}"),
    "data_parallel_size": int("${DP_SIZE}"),
    "gpu_count": int("${GPU_COUNT}"),
    "dtype": "${DTYPE}",
    "max_model_len": int("${MAX_MODEL_LEN}"),
    "kv_block_size": int("${KV_BLOCK_SIZE}"),
    "gpu_memory_utilization": float("${GPU_MEMORY_UTILIZATION}"),
    "vllm_mas_trace_path": "${VLLM_MAS_TRACE_PATH}",
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
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  --block-size "${KV_BLOCK_SIZE}"
  --enable-prefix-caching
  --reasoning-parser qwen3
)

if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "true" ]]; then
  CMD+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_CALL_PARSER}")
fi

echo "Python: $(command -v python)"
python --version
echo "vLLM: $(command -v vllm)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设置}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "PORT=${PORT}"
echo "TP_SIZE=${TP_SIZE}"
echo "PP_SIZE=${PP_SIZE}"
echo "DP_SIZE=${DP_SIZE}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "DTYPE=${DTYPE}"
echo "KV_BLOCK_SIZE=${KV_BLOCK_SIZE}"
echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
echo "MAS_VLLM_LAUNCH_CONFIG_PATH=${MAS_VLLM_LAUNCH_CONFIG_PATH}"
echo "VLLM_MAS_TRACE_PATH=${VLLM_MAS_TRACE_PATH}"
echo "ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE}"
echo "TOOL_CALL_PARSER=${TOOL_CALL_PARSER}"
echo "启动命令："
printf ' %q' "${CMD[@]}"
echo

exec "${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/vllm_server.log"

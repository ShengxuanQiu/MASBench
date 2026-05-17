#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/home/shegnxuanqiu/mas-serving}"
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

if [[ -d "/data/home/shegnxuanqiu/.conda/envs/MAS/bin" ]]; then
  export PATH="/data/home/shegnxuanqiu/.conda/envs/MAS/bin:${PATH}"
elif [[ -f "${ROOT_DIR}/.venv-MAS/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.venv-MAS/bin/activate"
fi

DEFAULT_QWEN35="/opt/models/Qwen3.5-4B/"
FALLBACK_QWEN3="/data1/pretrained_models/Qwen3-8B/"

MODEL_PATH="${MODEL_PATH:-${DEFAULT_QWEN35}}"
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "默认模型路径不存在：${MODEL_PATH}"
  MODEL_PATH="${FALLBACK_QWEN3}"
  echo "切换到 fallback 模型：${MODEL_PATH}"
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "模型路径不存在：${MODEL_PATH}" >&2
  exit 1
fi

PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
HOST="${HOST:-0.0.0.0}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-mas-model}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"

CMD=(
  vllm serve "${MODEL_PATH}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --enable-prefix-caching
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser "${TOOL_CALL_PARSER}"
)

if [[ "${MODEL_PATH}" == *"Qwen3.5"* ]]; then
  CMD+=(--language-model-only)
fi

echo "Python: $(command -v python)"
python --version
echo "vLLM: $(command -v vllm)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-未设置}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "PORT=${PORT}"
echo "TP_SIZE=${TP_SIZE}"
echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "TOOL_CALL_PARSER=${TOOL_CALL_PARSER}"
echo "启动命令："
printf ' %q' "${CMD[@]}"
echo

exec "${CMD[@]}" 2>&1 | tee -a "${LOG_DIR}/vllm_server.log"

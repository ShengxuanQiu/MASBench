#!/usr/bin/env bash
set -euo pipefail

CS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${CS_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MAS

MODEL_PATH="${MODEL_PATH:-/data/home/shegnxuanqiu/deepthink-with-confidence-openai/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-8b-case-study1}"
GPU_ID="${GPU_ID:-1}"
PORT="${PORT:-8101}"
SESSION_DIR="${SESSION_DIR:-${CS_DIR}/artifacts/server}"
mkdir -p "${SESSION_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export VLLM_MAS_TRACE_PATH="${VLLM_MAS_TRACE_PATH:-${SESSION_DIR}/vllm_step_trace.jsonl}"

exec vllm serve "${MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.88 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --generation-config vllm \
  --reasoning-parser qwen3 \
  2>&1 | tee "${SESSION_DIR}/vllm_server.log"

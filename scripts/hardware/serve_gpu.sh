#!/usr/bin/env bash
set -euo pipefail
export MASBENCH_ENV=serving
source "$(dirname "$0")/env.sh"
: "${CUDA_VISIBLE_DEVICES:?Select an allocated, idle GPU explicitly}"
: "${MODEL_DIR:?Set the local model directory explicitly}"
[[ -f "$MODEL_DIR/config.json" ]] || { echo "Missing model config: $MODEL_DIR" >&2; exit 1; }
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "${SERVED_MODEL_NAME:-masbench-model}" \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-18200}" \
  --tensor-parallel-size "${TP_SIZE:-1}" \
  --dtype "${DTYPE:-bfloat16}" \
  --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --max-num-seqs "${MAX_NUM_SEQS:-32}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" "$@"

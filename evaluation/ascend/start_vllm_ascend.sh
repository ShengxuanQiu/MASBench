#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/model/Qwen3-8B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-local-mas-model}"
PORT="${PORT:-8000}"
NPU_DEVICE="${NPU_DEVICE:-0}"
PREFIX_CACHE_MODE="${PREFIX_CACHE_MODE:-enabled}"

case "${PREFIX_CACHE_MODE}" in
  enabled) cache_flag="--enable-prefix-caching" ;;
  disabled) cache_flag="--no-enable-prefix-caching" ;;
  *) echo "PREFIX_CACHE_MODE must be enabled or disabled" >&2; exit 2 ;;
esac

export ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICE}"
exec vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --block-size 128 \
  --gpu-memory-utilization 0.85 \
  --seed 42 \
  --reasoning-parser qwen3 \
  --enable-request-id-headers \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  "${cache_flag}"

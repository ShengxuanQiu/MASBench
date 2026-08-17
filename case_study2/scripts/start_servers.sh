#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/case_study2/results/server_logs"
mkdir -p "${LOG_DIR}"

nohup env CUDA_VISIBLE_DEVICES=0 /data/home/shegnxuanqiu/.conda/envs/MAS/bin/vllm \
  serve /data/home/shegnxuanqiu/deepthink-with-confidence-openai/Qwen3-8B \
  --served-model-name qwen3-8b-case-study2 \
  --host 127.0.0.1 --port 8201 --gpu-memory-utilization 0.82 \
  --max-model-len 32768 --max-num-batched-tokens 8192 \
  >"${LOG_DIR}/main_vllm.log" 2>&1 &
echo $! >"${LOG_DIR}/main_vllm.pid"
echo "Main vLLM server started. Producer-side memory uses the same generation request."

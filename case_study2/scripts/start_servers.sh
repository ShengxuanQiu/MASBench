#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${ROOT_DIR}/case_study2/results/server_logs"
mkdir -p "${LOG_DIR}"

export CUDA_VISIBLE_DEVICES=0
nohup /data/home/shegnxuanqiu/.conda/envs/MAS/bin/vllm \
  serve /data/home/shegnxuanqiu/deepthink-with-confidence-openai/Qwen3-8B \
  --served-model-name qwen3-8b-case-study2 \
  --host 127.0.0.1 --port 8201 --gpu-memory-utilization 0.82 \
  --max-model-len 32768 --max-num-batched-tokens 8192 \
  >"${LOG_DIR}/main_vllm.log" 2>&1 &
echo $! >"${LOG_DIR}/main_vllm.pid"

export CUDA_VISIBLE_DEVICES=1
nohup /data/home/shegnxuanqiu/.conda/envs/MAS/bin/vllm \
  serve /data1/pretrained_models/Qwen3.5-4B \
  --served-model-name qwen3.5-4b-memory \
  --host 127.0.0.1 --port 8202 --gpu-memory-utilization 0.72 \
  --max-model-len 32768 --max-num-batched-tokens 8192 \
  --reasoning-parser qwen3 --language-model-only \
  >"${LOG_DIR}/memory_vllm.log" 2>&1 &
echo $! >"${LOG_DIR}/memory_vllm.pid"

echo "Servers started. Logs: ${LOG_DIR}"

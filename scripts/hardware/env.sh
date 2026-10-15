#!/usr/bin/env bash
# Source this file from a checkout. It does not start a server or allocate a GPU.
MASBENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MASBENCH_ROOT
export MASBENCH_RUNTIME="${MASBENCH_RUNTIME:-$HOME/masbench-runtime}"
export MASBENCH_ENV="${MASBENCH_ENV:-client}"
if [[ ! -x "$MASBENCH_RUNTIME/envs/$MASBENCH_ENV/bin/python" ]]; then
  echo "Missing environment: $MASBENCH_RUNTIME/envs/$MASBENCH_ENV" >&2
  return 1 2>/dev/null || exit 1
fi
export PATH="$MASBENCH_RUNTIME/envs/$MASBENCH_ENV/bin:$MASBENCH_RUNTIME/bin:$PATH"
export PYTHONPATH="$MASBENCH_ROOT/mas_workflow${PYTHONPATH:+:$PYTHONPATH}"
export UV_CACHE_DIR="$MASBENCH_RUNTIME/cache/uv"
export HF_HOME="$MASBENCH_RUNTIME/cache/huggingface"
export XDG_CACHE_HOME="$MASBENCH_RUNTIME/cache"
export TMPDIR="$MASBENCH_RUNTIME/tmp"
export TORCHINDUCTOR_CACHE_DIR="$MASBENCH_RUNTIME/cache/torchinductor"
export TRITON_CACHE_DIR="$MASBENCH_RUNTIME/cache/triton"
export VLLM_CACHE_ROOT="$MASBENCH_RUNTIME/cache/vllm"
mkdir -p "$TMPDIR" "$MASBENCH_RUNTIME/logs"

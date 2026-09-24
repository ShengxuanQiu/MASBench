#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
PART5_ROOT=${PART5_PRESSURE_ROOT:-"$ROOT/results/part5-pressure-v2"}
OUT=${PHYSICAL_KV_ROOT:-"$PART5_ROOT/physical_kv"}
PYTHON_BIN=${MASBENCH_PYTHON:-python3}
ASCEND_PYTHONPATH=${PYTHONPATH:-}
export PYTHONPATH="$ROOT/mas_workflow:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
SERVICE_PID=

stop_service() {
  if [[ -n "$SERVICE_PID" ]] && kill -0 "$SERVICE_PID" 2>/dev/null; then
    kill -TERM "$SERVICE_PID" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "$SERVICE_PID" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$SERVICE_PID" 2>/dev/null; then
      kill -KILL "$SERVICE_PID" 2>/dev/null || true
    fi
  fi
  SERVICE_PID=
}
trap stop_service EXIT INT TERM

start_service() {
  local mode=$1 log=$2
  if curl --noproxy '*' -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then
    echo "Port 8001 already has a service; refusing to use it" >&2
    exit 3
  fi
  local prefix_mode=disabled
  [[ "$mode" == warm_cache_enabled ]] && prefix_mode=enabled
  mkdir -p "$(dirname "$log")"
  (cd /root && exec env PYTHONNOUSERSITE=1 PYTHONPATH="$ASCEND_PYTHONPATH" MODEL_PATH=/model/Qwen3-8B \
    SERVED_MODEL_NAME=masbench-qwen3-8b-pressure PORT=8001 NPU_DEVICE=1 \
    PREFIX_CACHE_MODE="$prefix_mode" "$ROOT/evaluation/ascend/start_vllm_ascend.sh") \
    >"$log" 2>&1 </dev/null &
  SERVICE_PID=$!
  for _ in {1..180}; do
    if curl --noproxy '*' -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then
      echo "Isolated NPU1 service ready: $mode"
      return
    fi
    if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
      tail -80 "$log" >&2
      exit 4
    fi
    sleep 1
  done
  tail -80 "$log" >&2
  echo "Service readiness timeout" >&2
  exit 5
}

run_mode() {
  local mode=$1 directory="$OUT/$1"
  "$PYTHON_BIN" "$ROOT/evaluation/ascend/part5_pressure/prepare_physical_kv.py" \
    --part5-root "$PART5_ROOT" --output "$directory/configs" --mode "$mode"
  if [[ -f "$directory/shapes/results.json" && -f "$directory/repeat_spawn/results.json" ]]; then
    echo "Existing completed profiles: $mode"
    return
  fi
  start_service "$mode" "$directory/service.log"
  cd "$ROOT/mas_workflow"
  "$PYTHON_BIN" -m app.mixed_study --config "$directory/configs/shapes.json" \
    --output "$directory/shapes"
  "$PYTHON_BIN" -m app.mixed_study --config "$directory/configs/repeat_spawn.json" \
    --output "$directory/repeat_spawn"
  stop_service
  sleep 5
}

run_mode cache_disabled
run_mode warm_cache_enabled
"$PYTHON_BIN" "$ROOT/evaluation/ascend/part5_pressure/plot_physical_kv.py" \
  --input "$OUT" --output "$PART5_ROOT/report"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASCEND_DIR="${ROOT_DIR}/evaluation/ascend"
CONFIG_DIR="${ASCEND_DIR}/configs"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT_DIR}/results/ascend-initial/${RUN_ID}}"
MODE="${1:-preflight}"

if [[ -f "${ASCEND_DIR}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ASCEND_DIR}/.env"
  set +a
fi

mkdir -p "${RESULT_ROOT}"
cd "${ROOT_DIR}/mas_workflow"
export PYTHONPATH="${ROOT_DIR}/mas_workflow:${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

preflight() {
  local output="${RESULT_ROOT}/preflight"
  mkdir -p "${output}"
  python3 - "${output}/endpoint.json" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=10) as response:
    data = json.loads(response.read())
models = [row.get("id") for row in data.get("data", [])]
if "local-mas-model" not in models:
    raise SystemExit("local-mas-model is not served by port 8000")
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(data, indent=2))
PY
  python3 - "${output}/npu_sample.json" <<'PY'
import json, sys
from app.backend_adapters import build_backend_trace_adapter
sample = build_backend_trace_adapter("ascend", {"npu_id": 0, "metadata": {"chip_id": 0}}).collect_device_metrics()
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(sample, indent=2))
if sample.get("status") != "success":
    raise SystemExit("Ascend telemetry is unavailable: " + json.dumps(sample))
PY
  python3 -m app.replay_protocol --model-dir /model/Qwen3-8B \
    --revision local-copy-revision-unavailable --output "${output}/model_identity.json"
  cd "${ROOT_DIR}"
  python3 -m pytest mas_workflow/tests case_study1/tests case_study2/tests \
    --import-mode=importlib -q | tee "${output}/pytest.log"
  cd "${ROOT_DIR}/mas_workflow"
  python3 -m app.publication --manifest "${CONFIG_DIR}/functional-manifest.json" \
    --output "${output}/functional-dry-run" | tee "${output}/functional-dry-run.log"
  python3 -m app.publication --manifest "${CONFIG_DIR}/pilot-manifest.json" \
    --output "${output}/pilot-dry-run" | tee "${output}/pilot-dry-run.log"
  git -C "${ROOT_DIR}" rev-parse HEAD > "${output}/git-head.txt"
  git -C "${ROOT_DIR}" status --short > "${output}/git-status.txt"
  npu-smi info -t board -i 0 -c 0 > "${output}/npu-board.txt"
  python3 -m pip show vllm vllm-ascend torch torch-npu > "${output}/software-versions.txt"
}

run_matrix() {
  local manifest="$1"
  local label="$2"
  local output="${RESULT_ROOT}/${label}"
  python3 -m app.publication --manifest "${CONFIG_DIR}/${manifest}" \
    --output "${output}" --execute | tee "${RESULT_ROOT}/${label}.log"
  python3 "${ASCEND_DIR}/summarize_initial_validation.py" \
    --matrix "${output}" --output "${output}/summary"
}

tavily_record_and_replay() {
  if [[ -z "${TAVILY_API_KEY:-}" ]]; then
    echo "TAVILY_API_KEY is missing. Copy evaluation/ascend/.env.example to .env and fill it." >&2
    return 2
  fi
  local source_root="${RESULT_ROOT}/tavily/source"
  local replay_root="${RESULT_ROOT}/tavily/replay"
  mkdir -p "${source_root}" "${replay_root}"
  python3 -m app.benchmark run --experiment "${CONFIG_DIR}/tavily-source.json" \
    --trace-dir "${source_root}" | tee "${RESULT_ROOT}/tavily/source.log"
  local source_trace
  source_trace="$(find "${source_root}" -type f -name '*.jsonl' | head -n 1)"
  if [[ -z "${source_trace}" ]]; then
    echo "No canonical source trace was produced" >&2
    return 3
  fi
  python3 -m app.benchmark replay --trace "${source_trace}" \
    --deployment "${CONFIG_DIR}/deployment-ascend-qwen3-8b.json" \
    --trace-dir "${replay_root}" | tee "${RESULT_ROOT}/tavily/replay.log"
  local replay_trace
  replay_trace="$(find "${replay_root}" -type f -name '*.jsonl' | head -n 1)"
  python3 -m app.replay_compare --source "${source_trace}" --replays "${replay_trace}" \
    --output "${RESULT_ROOT}/tavily/replay-comparison.json"
}

case "${MODE}" in
  preflight)
    preflight
    ;;
  functional)
    run_matrix functional-manifest.json functional
    ;;
  pilot)
    run_matrix pilot-manifest.json pilot
    ;;
  tavily)
    tavily_record_and_replay
    ;;
  quick)
    preflight
    run_matrix functional-manifest.json functional
    run_matrix pilot-manifest.json pilot
    ;;
  all)
    preflight
    run_matrix functional-manifest.json functional
    run_matrix pilot-manifest.json pilot
    tavily_record_and_replay
    ;;
  *)
    echo "Usage: $0 {preflight|functional|pilot|tavily|quick|all}" >&2
    exit 2
    ;;
esac

echo "RESULT_ROOT=${RESULT_ROOT}"

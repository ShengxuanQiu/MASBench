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
  python3 "${ASCEND_DIR}/prepare_live_source_matrix.py" \
    --manifest "${CONFIG_DIR}/source-manifest.json" \
    --output "${output}/source-dry-run" | tee "${output}/source-dry-run.log"
  git -C "${ROOT_DIR}" rev-parse HEAD > "${output}/git-head.txt"
  git -C "${ROOT_DIR}" status --short > "${output}/git-status.txt"
  npu-smi info -t board -i 0 -c 0 > "${output}/npu-board.txt"
  python3 -m pip show vllm vllm-ascend torch torch-npu > "${output}/software-versions.txt"
}

require_tavily() {
  if [[ -z "${TAVILY_API_KEY:-}" ]]; then
    echo "TAVILY_API_KEY is missing. Copy evaluation/ascend/.env.example to .env and fill it." >&2
    return 2
  fi
}

live_source() {
  require_tavily
  local output="${RESULT_ROOT}/source"
  python3 "${ASCEND_DIR}/prepare_live_source_matrix.py" \
    --manifest "${CONFIG_DIR}/source-manifest.json" \
    --output "${output}" --execute | tee "${RESULT_ROOT}/source.log"
  python3 "${ASCEND_DIR}/summarize_initial_validation.py" \
    --matrix "${output}" --output "${output}/summary"
}

replay_pilot() {
  local source="${SOURCE_MATRIX:-${RESULT_ROOT}/source}"
  if [[ ! -f "${source}/matrix_manifest.json" ]]; then
    echo "Recorded source matrix is missing at ${source}; run source first or set SOURCE_MATRIX." >&2
    return 3
  fi
  local output="${RESULT_ROOT}/replay-pilot"
  python3 "${ASCEND_DIR}/prepare_replay_pilot.py" \
    --source "${source}" --output "${output}" \
    --deployment "${CONFIG_DIR}/deployment-ascend-qwen3-8b.json" \
    --rates "${PILOT_RATES:-0.1,0.25}" --count "${PILOT_COUNT:-4}" \
    | tee "${RESULT_ROOT}/replay-pilot.log"
  python3 "${ASCEND_DIR}/summarize_initial_validation.py" \
    --matrix "${output}" --output "${output}/summary"
}

case "${MODE}" in
  preflight)
    preflight
    ;;
  functional)
    live_source
    ;;
  source)
    live_source
    ;;
  pilot)
    replay_pilot
    ;;
  tavily)
    live_source
    ;;
  quick)
    preflight
    live_source
    replay_pilot
    ;;
  all)
    preflight
    live_source
    replay_pilot
    ;;
  *)
    echo "Usage: $0 {preflight|source|functional|pilot|tavily|quick|all}" >&2
    exit 2
    ;;
esac

echo "RESULT_ROOT=${RESULT_ROOT}"

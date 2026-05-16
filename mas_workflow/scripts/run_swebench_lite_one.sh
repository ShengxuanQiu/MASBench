#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/home/shegnxuanqiu/mas-serving}"
WORKFLOW_DIR="${WORKFLOW_DIR:-${ROOT_DIR}/mas_workflow}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/swebench_repos}"
PYTHON_BIN="${PYTHON_BIN:-/data/home/shegnxuanqiu/.conda/envs/MAS/bin/python}"
INSTANCE_INDEX="${INSTANCE_INDEX:-0}"
MAX_RETRIES="${MAX_RETRIES:-0}"

export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"

mkdir -p "${REPO_ROOT}"

INSTANCE_JSON="$("${PYTHON_BIN}" - <<PY
import json
from datasets import load_dataset

idx = int("${INSTANCE_INDEX}")
ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test", streaming=True)
for i, item in enumerate(ds):
    if i == idx:
        print(json.dumps(item, ensure_ascii=False))
        break
else:
    raise SystemExit(f"找不到 INSTANCE_INDEX={idx}")
PY
)"

INSTANCE_ID="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["instance_id"])' "${INSTANCE_JSON}")"
REPO_FULL="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["repo"])' "${INSTANCE_JSON}")"
BASE_COMMIT="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["base_commit"])' "${INSTANCE_JSON}")"
PROBLEM="$("${PYTHON_BIN}" -c 'import json,sys; print(json.loads(sys.argv[1])["problem_statement"])' "${INSTANCE_JSON}")"
FAIL_TO_PASS="$("${PYTHON_BIN}" -c 'import json,sys; print(" ".join(json.loads(sys.argv[1]).get("FAIL_TO_PASS") or []))' "${INSTANCE_JSON}")"

LOCAL_REPO="${REPO_ROOT}/${INSTANCE_ID}"
if [[ ! -d "${LOCAL_REPO}/.git" ]]; then
  git clone --filter=blob:none "https://github.com/${REPO_FULL}.git" "${LOCAL_REPO}"
fi

git -C "${LOCAL_REPO}" fetch --depth 1 origin "${BASE_COMMIT}" || true
git -C "${LOCAL_REPO}" checkout "${BASE_COMMIT}"

QUERY="SWE-bench Lite instance ${INSTANCE_ID}
Repo: ${REPO_FULL}
Base commit: ${BASE_COMMIT}
Problem:
${PROBLEM}
FAIL_TO_PASS tests: ${FAIL_TO_PASS}"

TEST_COMMAND="${TEST_COMMAND:-pytest -q ${FAIL_TO_PASS}}"

cd "${WORKFLOW_DIR}"
"${PYTHON_BIN}" -m app.main \
  --repo "${LOCAL_REPO}" \
  --query "${QUERY}" \
  --test-command "${TEST_COMMAND}" \
  --max-retries "${MAX_RETRIES}"


#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${REPO_PATH:-}" ]]; then
  echo "用法："
  echo "  cd ~/mas-serving/mas_workflow"
  echo "  REPO_PATH=/path/to/repo QUERY='修复测试失败并输出 root cause 和 patch 说明' TEST_COMMAND='pytest -q' scripts/run_demo.sh"
  echo
  echo "说明："
  echo "  REPO_PATH 必填。"
  echo "  QUERY 默认：修复这个仓库中导致测试失败的 bug，并输出 root cause 和 patch 说明"
  echo "  TEST_COMMAND 默认：pytest -q"
  exit 1
fi

QUERY="${QUERY:-修复这个仓库中导致测试失败的 bug，并输出 root cause 和 patch 说明}"
TEST_COMMAND="${TEST_COMMAND:-pytest -q}"

python -m app.main \
  --repo "${REPO_PATH}" \
  --query "${QUERY}" \
  --test-command "${TEST_COMMAND}"


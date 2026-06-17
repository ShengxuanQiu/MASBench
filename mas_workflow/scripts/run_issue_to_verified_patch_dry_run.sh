#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}/mas_workflow"

python -m app.main \
  --mode full \
  --full-workflow issue_to_verified_patch \
  --task-source manual \
  --query "SWE-bench style issue: fix a regression where a parser rejects valid input after a recent refactor. Produce a patch plan, candidate patches, test result, and final report." \
  --repo "${ROOT_DIR}" \
  --num-agents 3 \
  --max-retries 1 \
  --llm-mode mock \
  --tool-mode synthetic \
  --search-provider local_repo \
  --dry-run-patch true \
  --dry-run-tests true \
  --trace-level arch \
  --export-trace-views false \
  --trace-dir traces/full_workflows

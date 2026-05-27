#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m app.analyze_week2_traces \
  --trace-root ../traces \
  --extra-trace-root traces \
  --output-dir ../traces/week2_characterization \
  --progress-dir ../progress/week2 \
  --include-backend-metrics true \
  --include-model-outputs false

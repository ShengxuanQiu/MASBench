#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ACTION=${1:-plan}; shift || true
OUT=${PART4_OUTPUT_ROOT:-"$ROOT/results/part4"}/4_6
PYTHON_BIN=${MASBENCH_PYTHON:-"$HOME/.conda/envs/MAS/bin/python"}
[[ -x "$PYTHON_BIN" ]] || PYTHON_BIN=$(command -v python3)
mkdir -p "$OUT"
BASE="$ROOT/mas_workflow/configs/part4/4_6_cross_hardware.json"
RESOLVED="$OUT/resolved-config.json"
"$PYTHON_BIN" - "$BASE" "$RESOLVED" "$@" <<'PY'
import json, pathlib, sys
base=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]).resolve()
cfg=json.loads(base.read_text())
cfg['classes']={k:{**v,**({'trace_corpus':str((base.parent/v['trace_corpus']).resolve())} if 'trace_corpus' in v else {})} for k,v in cfg['classes'].items()}
cfg['deployments']={k:str((base.parent/v).resolve()) for k,v in cfg['deployments'].items()}
for item in sys.argv[3:]:
    if '=' not in item: raise SystemExit('deployment argument must be name=/absolute/path.json')
    name,path=item.split('=',1); cfg['deployments'][name]=str(pathlib.Path(path).resolve())
out.write_text(json.dumps(cfg,indent=2))
PY
cd "$ROOT/mas_workflow"
if [[ "$ACTION" == plan ]]; then
  "$PYTHON_BIN" -m app.mixed_study --config "$RESOLVED" --output "$OUT/run" --validate-only
  echo "4.6 plan validated. Add devices as name=/absolute/deployment.json arguments."
  exit 0
fi
ARGS=(); [[ "$ACTION" == resume ]] && ARGS+=(--resume)
"$PYTHON_BIN" -m app.mixed_study --config "$RESOLVED" --output "$OUT/run" "${ARGS[@]}"
echo "4.6 cross-hardware results: $OUT/run/results.json"

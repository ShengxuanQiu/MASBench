#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
OUT=${PART5_PILOT_ROOT:-"$ROOT/results/part5-ascend-pilot"}
ACC="$OUT/semantic-acceptance"
TOKENIZER=${MASBENCH_TOKENIZER:-/model/Qwen3-8B}
MODEL=${MASBENCH_MODEL_ID:-masbench-qwen3-8b-part5}
BASE_URL=${MASBENCH_BASE_URL:-http://127.0.0.1:8001/v1}
mkdir -p "$ACC"

cd "$ROOT"
python3 - <<'PY' "$OUT" > "$ACC/source_path.txt"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])/"5_2/matrix"
manifest=json.load(open(root/"matrix_manifest.json"))
cell=next(c for c in manifest["cells"] if c["workload"]=="fork-join-width" and c["parameters"].get("width")==4)
paths=list((root/cell["id"]).rglob("traces/**/*.jsonl"))
if len(paths)!=1: raise SystemExit(f"expected one source trace, found {len(paths)}")
print(paths[0].resolve())
PY

SOURCE=$(cat "$ACC/source_path.txt")
rm -rf "$ACC/trace" "$ACC/scenario.json" "$ACC/system.json" "$ACC/official-run.json" "$ACC/official-metrics.json" "$ACC/validate.json"
cd "$ROOT/mas_workflow"
export PYTHONPATH=".:..${PYTHONPATH:+:$PYTHONPATH}"
python3 -m app.semantic.cli collect --source "$SOURCE" --output "$ACC/trace" \
  --workload-id part5-fork-join-width4 --workload-version pilot-v1 \
  --tokenizer "$TOKENIZER" --source-framework-version ascend-pilot
python3 - <<'PY' "$ACC/scenario-template.json" "$MODEL"
import json, pathlib, sys
d=json.load(open("configs/semantic/examples/scenario.template.json"))
d["scenario_id"]="part5-ascend-length-locked"
d["model_input_contract"]["model_identity"]=sys.argv[2]
d["model_input_contract"]["generation_parameters"]={"temperature":0.0,"max_tokens":96,"chat_template_kwargs":{"enable_thinking":False}}
d["measurement"].update({"task_count":1,"task_slo_sec":30.0})
pathlib.Path(sys.argv[1]).write_text(json.dumps(d,indent=2))
PY
python3 -m app.semantic.cli resolve-scenario --template "$ACC/scenario-template.json" \
  --trace "$ACC/trace" --tokenizer "$TOKENIZER" --output "$ACC/scenario.json"
python3 - <<'PY' "$ACC" "$MODEL"
import json, pathlib, subprocess, sys
root=pathlib.Path(sys.argv[1]); scenario=json.load(open(root/"scenario.json"))
version="unknown"
try:
    text=subprocess.check_output(["python3","-m","pip","show","vllm"],text=True)
    version=next(line.split(":",1)[1].strip() for line in text.splitlines() if line.startswith("Version:"))
except Exception: pass
d={"backend_name":"vllm-openai","backend_version":version,"hardware_model":"Huawei Ascend 910 V1",
   "accelerator_count":1,"model_identity":sys.argv[2],
   "tokenizer_hash":scenario["model_input_contract"]["tokenizer_hash"],
   "chat_template_hash":scenario["model_input_contract"]["chat_template_hash"],
   "batching_policy":{"max_num_seqs":32},"cache_implementation":{"prefix_cache":False},
   "scheduling_policy":{},"placement":{"visible_device":1,"npu_board":0,"chip":1},
   "parallelism":{"tp":1,"pp":1,"dp":1},
   "capabilities":{"supports_length_lock":True,"supports_token_lock":False}}
(root/"system.json").write_text(json.dumps(d,indent=2))
PY
python3 -m app.semantic.cli validate --trace "$ACC/trace" > "$ACC/validate.json"
python3 -m app.semantic.cli replay --trace "$ACC/trace" --scenario "$ACC/scenario.json" \
  --system "$ACC/system.json" --backend vllm-openai --base-url "$BASE_URL" --output "$ACC/official-run.json"
python3 -m app.semantic.cli measure --run "$ACC/official-run.json" --slo-sec 30 --accelerators 1 \
  --output "$ACC/official-metrics.json"
python3 - <<'PY' "$ACC"
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); run=json.load(open(root/"official-run.json")); metrics=json.load(open(root/"official-metrics.json"))
print(json.dumps({"run_valid":run["run_valid"],"invalid_reasons":run["invalid_reasons"],
 "operations":len(run["operations"]),"task_latency_p50_sec":metrics["task_latency_sec"]["p50"],
 "slo_attainment":metrics["slo_attainment"]},indent=2))
PY

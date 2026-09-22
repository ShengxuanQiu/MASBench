# Ascend Part 5 pilot

This directory runs a reduced real-backend acceptance matrix for the paper's
Part 5 experiment hierarchy. It uses an isolated Qwen3-8B vLLM endpoint with
prefix caching disabled. The pilot is designed to validate mechanisms and the
data path; its one-repetition cells are not publication measurements.

From the repository root:

```bash
bash evaluation/ascend/part5/run_part5_pilot.sh 5_2
bash evaluation/ascend/part5/run_part5_pilot.sh 5_3
bash evaluation/ascend/part5/run_part5_pilot.sh 5_4
bash evaluation/ascend/part5/run_part5_pilot.sh 5_5
bash evaluation/ascend/part5/run_part5_pilot.sh 5_6
bash evaluation/ascend/part5/run_part5_pilot.sh finalize
```

`5_4` must run before `5_5` and `5_6` because it freezes the replay corpus.
`finalize` performs the official Semantic Trace `length_locked` acceptance and
generates `results/part5-ascend-pilot/report/PART5_PILOT_REPORT.md` plus five
SVG figures.

The resolved pilot uses port 8001, served model `masbench-qwen3-8b-part5`, and
board 0/chip 1 (`ASCEND_RT_VISIBLE_DEVICES=1`). Start it outside the repository
root so the checked-out `vllm/` submodule does not shadow the installed package:

```bash
cd /tmp
ASCEND_RT_VISIBLE_DEVICES=1 NPU_DEVICE=1 PORT=8001 \
SERVED_MODEL_NAME=masbench-qwen3-8b-part5 PREFIX_CACHE_MODE=disabled \
nohup bash /root/mas-serving/evaluation/ascend/start_vllm_ascend.sh \
  >/root/mas-serving/logs/vllm_part5_npu1.log 2>&1 &
```

The full publication run should increase repetitions and task counts, add dense
load points around the observed capacity bracket, extend the dependency-heavy
right-censored range, and replay the identical frozen corpus on A6000 and H100.

# Preparing hardware experiments

Keep the benchmark client separate from the device serving runtime. The client
requires `mas_workflow/requirements-benchmark.txt`; analysis also uses pandas,
pyarrow, matplotlib, httpx, openai, psutil, and nvidia-ml-py. A client smoke test
checks trace/control behavior without generating performance evidence.

Use a private, writable data directory for environments, temporary files,
compiler caches, logs, and profiling output. The default is
`$HOME/masbench-runtime`. Do not modify the machine's shared Python installation,
driver, existing containers, or other users' jobs.

```bash
# After preparing envs/client and envs/serving in MASBENCH_RUNTIME:
source scripts/hardware/env.sh
python scripts/hardware/check_environment.py --output "$MASBENCH_RUNTIME/logs/inventory.json"
cd mas_workflow
python -m pytest tests/test_semantic_benchmark.py tests/test_benchmark_contracts.py -q
```

For a serving import check, set `MASBENCH_ENV=serving` before sourcing the file
and add `--serving-import`. `--model /path/to/model` checks the shard manifest
without loading weights. Neither check runs a workload or benchmarks a device.

The Pro5000 preparation uses the main branch; Ascend platforms use the ascend
branch. Record the exact Git SHA, Python package freeze, model/tokenizer identity,
driver/CUDA or CANN versions, scheduler/cache settings, and profiling permissions
when starting each study. Branch or platform names do not establish comparable
software or hardware conditions.

## GPU serving

The startup helper requires explicit device and model selection. Launch it only
after the experiment has an allocated idle device:

```bash
CUDA_VISIBLE_DEVICES=<allocated-device> MODEL_DIR=/path/to/model \
  bash scripts/hardware/serve_gpu.sh
```

Its default listener is loopback. Configuration values are initial launch
defaults, not a frozen experiment protocol. Validate actual model execution,
attention kernels, request accounting, and profiler collection on a free GPU
before accepting performance measurements. Installation/import checks alone
do not establish runtime compatibility.

## Ascend serving

Inspect the actual NPU product, host driver, CANN and torch-npu versions before
choosing a compatible vLLM-Ascend build. Do not reuse an A2/A3 wheel or container
for 950 by assumption. The upstream installation guide lists 950-specific CANN
images and source/image installation options:
https://docs.vllm.ai/projects/ascend/en/main/getting_started/installation.html

## Profiling boundaries

Inventory and serving metrics are not physical bandwidth/FLOP measurements.
Record whether Nsight Systems, Nsight Compute, msprof and their permissions are
available. Restricted device counters may need administrator configuration;
do not reload drivers or relax shared-host permissions during preparation.
PyTorch profiling and device telemetry are complementary to hardware counters.
Check tracing overhead separately before using profiled timings in a study.

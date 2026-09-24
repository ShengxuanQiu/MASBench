# Ascend Part 5 pressure experiments

This directory profiles graph shape separately from backend observations. It does not use live external tools during scored replay.

The pipeline records one native execution of each standard template (`Spawn`, `Fork--Join`, `Refinement Loop`, and `Debate`), freezes the realized traces, and replays them over a dense 0.25--20 user-QPS sweep. The refinement task intentionally requests revision until its configured limit so that the controlled template profile realizes the full loop depth.

Run against the isolated service configured on port 8001:

```bash
./evaluation/ascend/part5_pressure/run_pressure_experiments.sh 5_2
./evaluation/ascend/part5_pressure/run_pressure_experiments.sh 5_3
./evaluation/ascend/part5_pressure/run_pressure_experiments.sh 5_4
./evaluation/ascend/part5_pressure/run_pressure_experiments.sh 5_5
./evaluation/ascend/part5_pressure/run_pressure_experiments.sh render
```

The default output is `results/part5-pressure-v2/`. The Chinese report is `report/PART5_PRESSURE_REPORT_ZH.md`; every figure is emitted as PNG and PDF.

To measure backend-reported KV block occupancy at roughly 60 ms intervals, run the dedicated 5.2 profile. It starts and stops its own Qwen3-8B service on NPU1/port 8001, leaving the service on port 8000 untouched:

```bash
./evaluation/ascend/part5_pressure/run_physical_kv.sh
```

The profile records each of the four templates with prefix caching disabled, then runs two identical Spawn workflows with prefix caching off and on. It writes `report/figures/fig_5_2_backend_kv_occupancy.{png,pdf}`, `report/figures/fig_5_2_prefix_cache_retention.{png,pdf}`, and `report/PHYSICAL_KV_PROFILE_ZH.md`. The plotted metric is vLLM's `kv_cache_usage_perc`: `1 - free_blocks / total_blocks`. Cached prefix blocks can enter the free queue and still be reusable, so a zero reading between workflows does not imply that prefix-cache content was discarded. This is backend-reported occupied KV-block fraction, not directly measured NPU HBM bytes.

Metric boundaries are explicit:

- logical KV equivalent is estimated from observed token counts and model architecture;
- occupied KV-block fraction is backend-reported by vLLM; retained prefix-cache state and NPU HBM byte allocation are separate measurements;
- AICore and HBM-bandwidth utilization are observed through `npu-smi`;
- arithmetic intensity is unavailable unless synchronized FLOP and DRAM-byte profiler counters are supplied.

These short cohorts are for acceptance and hypothesis discovery. Publication runs should add repeated trials, confidence intervals, a stable measurement window, and the same frozen corpus on the NVIDIA backends.

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

Metric boundaries are explicit:

- logical KV equivalent is estimated from observed token counts and model architecture;
- physical KV is backend-reported by vLLM;
- AICore and HBM-bandwidth utilization are observed through `npu-smi`;
- arithmetic intensity is unavailable unless synchronized FLOP and DRAM-byte profiler counters are supplied.

These short cohorts are for acceptance and hypothesis discovery. Publication runs should add repeated trials, confidence intervals, a stable measurement window, and the same frozen corpus on the NVIDIA backends.

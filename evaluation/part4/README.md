# Part 4 experiment protocol

These five entry points implement the paper's experiment hierarchy without changing the canonical runner, trace, or replay semantics. Run them from the repository root. `plan` validates and expands a configuration; `run` executes it; `resume` only reuses cases whose fingerprints match.

```bash
./evaluation/part4/run_4_2_pressure_signatures.sh plan
./evaluation/part4/run_4_2_pressure_signatures.sh run
```

The checked-in GPU3 deployment expects the accepted Qwen3.5-4B vLLM endpoint at `127.0.0.1:18080`, served model name `qwen35-4b-mas-pilot`, A6000 GPU 0, temperature 0, and prefix cache disabled. Edit `mas_workflow/configs/part4/deployment-gpu3.json` only when the endpoint identity changes. The experiment output records the resolved deployment and source fingerprint.

Start the accepted GPU3 service in a separate shell before using `run` (the executable path may be overridden for another environment):

```bash
CUDA_VISIBLE_DEVICES=0 $HOME/.conda/envs/MAS/bin/vllm serve /data1/pretrained_models/Qwen3.5-4B \
  --host 127.0.0.1 --port 18080 --served-model-name qwen35-4b-mas-pilot \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 --max-num-seqs 16 \
  --gpu-memory-utilization 0.55 --language-model-only --no-enable-prefix-caching \
  --enforce-eager --default-chat-template-kwargs '{"enable_thinking":false}'
```

## 4.2 Single-Workflow Pressure Signatures

`run_4_2_pressure_signatures.sh` sweeps one factor at low offered load: ParallelAggregate width; EvaluateRefine revision depth; PeerExchange rounds, connectivity, and edge delivery. It writes one row per realized workflow and one row per load case.

Primary analysis fields are operation-DAG width/depth/fan-in/fan-out/density, critical-path ratio, delivered bytes/tokens, artifact reuse, delivery compression, request-context amplification, ready-frontier peak/area, barrier wait, and logical artifact lifetime/byte-seconds. Logical state is never labeled as physical KV/cache residency.

Expected use: establish a pressure-signature vector for each controlled structure before interpreting saturation behavior. Mock runs validate mechanics only; paper claims require the real endpoint.

## 4.3 Graph Pressure under Multiplexing

`run_4_3_multiplexing.sh` compares matched chains and matched parallel graphs with the same atomic request payload multiset, then sweeps representative ParallelAggregate, EvaluateRefine, and PeerExchange structures under increasing Poisson arrival rate.

Primary analysis fields add workflow p50/p95 E2E latency, goodput, SLO success, internal QPS, client ready-queue growth, maximum ready frontier/inflight requests, and endpoint-wide vLLM queue/cache/device samples. The matched pair isolates dependency release behavior; the motif cells test whether the same mechanism survives in canonical collaboration graphs.

Expected use: find the load level at which a low-load signature becomes queueing, interference, and latency degradation. A finite tested boundary is reported as an SLO bracket, not automatically as a hardware saturation diagnosis.

## 4.4 Heterogeneous Multi-Workflow Serving

`run_4_4_heterogeneous.sh` multiplexes three workflow classes on the same shared semaphore and backend:

- `burst_heavy`: width-8 ParallelAggregate;
- `context_heavy`: 4-peer, 3-round all-to-all PeerExchange with full edge delivery;
- `dependency_heavy`: an 8-request matched sequential chain.

The driver saves the exact class label, source index, arrival offset, seed, and schedule hash. It runs same-rate isolated class baselines and reports each class separately: latency, slowdown, SLO success, goodput, internal requests/QPS, prompt/output tokens, ready/barrier demand, delivered information, and logical state byte-seconds. Physical utilization, memory, bandwidth, and cache samples remain endpoint-wide because a shared device sample cannot be truthfully assigned to one class.

After a successful run, the script selects completed traces from the lowest-rate isolated cases and freezes three traces per class under `results/part4/4_4/frozen`. These traces are the fixed workload input for 4.5 and 4.6.

## 4.5 Workload Mix Determines Serving Capacity

Run 4.4 first, then run `run_4_5_capacity_mix.sh`. It replays the frozen traces under balanced, burst-dominated, context-dominated, and dependency-dominated mixes while sweeping total user QPS. Every cell saves an exact largest-remainder class count and a seeded shuffled arrival schedule, so equal-QPS composition comparisons are reproducible.

Primary analysis compares aggregate goodput/SLO capacity with per-class latency, slowdown, SLO success, internal-request share, token demand, information delivery, and logical state demand. The key test is whether equal user QPS produces different internal QPS and sustainable SLO behavior as composition changes.

## 4.6 Cross-Hardware Bottleneck Maps

`run_4_6_cross_hardware.sh` uses the same frozen traces, mixes, rates, seeds, and class schedules for every supplied deployment. On GPU3 it validates/runs the A6000 cell. In the next hardware round, add deployment files on the command line:

```bash
./evaluation/part4/run_4_6_cross_hardware.sh run \
  h100=/absolute/path/deployment-h100.json \
  ascend910c=/absolute/path/deployment-ascend.json
```

Compare workflow and class latency/slowdown, request queue, backend cache metrics, device utilization, memory/bandwidth, and power only where the adapter reports them. Every metric retains its observed/backend-reported/estimated/unavailable source. Interpret a bottleneck migration only when queue/runtime changes align with observed device counters; missing counters stay unavailable.

Cross-hardware replay is best-effort for output length. It fixes payloads, dependencies, tool snapshots, and control path, and verifies replay invariants. Output token sequence and strict output length are not claimed fixed. Keep prefix-cache protocol, model/tokenizer/template identity, concurrency policy, and output-length tolerance explicit in the deployment/results.

## Output map

```text
results/part4/
  4_2/matrix/ + 4_2/report/{workflow_rows.csv,case_rows.csv,report.json}
  4_3/matrix/ + 4_3/report/{workflow_rows.csv,case_rows.csv,report.json}
  4_4/run/results.json + 4_4/frozen/{index.json,<class>.json,<class>/*.jsonl}
  4_5/run/{results.json,capacity_results.json}
  4_6/run/{results.json,capacity_results.json}
```

The first paper plots should be built from these machine-readable outputs. Do not mix mock results, failed trace analyses, replay-ineligible runs, client-overflow cases, or unavailable hardware counters into performance conclusions.

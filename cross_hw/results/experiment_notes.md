# A6000 vs. Ascend 910: preliminary MASBench hardware profile

## Scope

- Model: Qwen3-8B, temperature 0, `/no_think` prompts.
- Hardware: one RTX A6000 versus one visible Ascend 910 NPU.
- Workloads: Single/Linear, Independent Fan-In, Manager–Worker, Debate All-Gather, Shared-Memory Fan-In, Retry Loop, Hierarchical Synthesis, and Issue-to-Patch.
- Mode: raw context propagation.
- Repetitions: two per workload and hardware platform.
- Tool treatment: figures use tool-excluded E2E latency. Ascend executions replay the frozen Tavily evidence from the A6000 experiment so that network search is not included in the hardware path.
- Timing source: OpenAI-compatible SSE stream observed by the MASBench client.

## Trace fields used

Request-level fields:

- `input_tokens`
- `output_tokens`
- `ttft_sec`
- `tpot_sec`
- `request_latency_sec`
- `request_ready_ts`
- `node_id`

Workflow-level fields:

- `result_latency_sec`
- `tool_latency_sec`
- `main_request_count`
- `workload`
- `run_id`

All 16 Ascend runs contain non-null TTFT, TPOT, and request latency for every LLM request.

## Preliminary observations

- Input-token totals differ by less than 0.7% for every workload; output-token totals differ by at most 2.3%, so the first comparison is well aligned at the token level.
- Across the eight workloads, the geometric-mean A6000/Ascend ratio is 1.46× for tool-excluded workflow latency and 1.49× for TPOT. In this configuration, lower Ascend TPOT is the most consistent source of its shorter end-to-end latency.
- TTFT is more graph- and request-stage-dependent than TPOT. Ascend is usually lower, but Hierarchical Synthesis is a counterexample at the run-level median, making TTFT behavior a more interesting topology-sensitive result than a simple platform ranking.
- Debate All-Gather and Issue-to-Patch have the largest absolute workflow latency because they contain the most LLM requests and synchronization stages.
- Graph amplification relative to Single/Linear is broadly similar across platforms, but Shared-Memory Fan-In and Debate All-Gather amplify Ascend latency somewhat more. This suggests graph structure and hardware/runtime behavior interact even when token counts are controlled.

## Interpretation limits

This is a strong preliminary comparison, not yet the final hardware-isolation result. The A6000 and Ascend services use different hardware backends and may use different vLLM/plugin versions, graph compilation strategies, memory settings, and maximum context settings. The figures should therefore be described as an end-to-end hardware-plus-serving-stack comparison until those software configurations are explicitly normalized and recorded.

Two repetitions are enough to inspect the visual shape but not enough for final confidence intervals. A publication-ready pass should use at least five repetitions and add a controlled concurrency sweep.

## Files

- `hardware_run_metrics.csv`: one row per hardware/workload/run.
- `hardware_request_metrics.csv`: one row per individual LLM request.
- `fig1_hardware_workflow_overview.*`: E2E, TTFT, and TPOT overview.
- `fig2_ttft_request_progression.*`: per-request TTFT along each graph.
- `fig3_tpot_request_progression.*`: per-request TPOT along each graph.
- `fig4_graph_amplification.*`: workflow latency normalized to Single/Linear.
- `a6000_raw_traces.tgz` and `ascend_qwen3_8b_raw_traces.tgz`: source trace and summary files.
- `ascend_profile_manifest.json`: Ascend run metadata.

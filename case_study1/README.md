# Case Study 1: Online MAS-aware critical-frontier prefill gating

This directory is the correctness-first real-execution case study. It is
independent of the older deterministic replay artifacts under `progress/`.

The current Chinese report, two-page PPT summary, and paper-style figures are
available at:

- `case_study1/完整汇报.md`
- `case_study1/PPT精简总结.md`
- `case_study1/artifacts/paper_configs/analysis/figure1_workflow_latency_breakdown.{png,pdf}`
- `case_study1/artifacts/paper_configs/analysis/figure2_multiconfig_metrics.{png,pdf}`

The current main experiment uses three corresponding configurations: a
2-agent controlled motif, a 4-agent controlled motif, and the fixed full
`issue_to_verified_patch` workflow. Each policy/configuration cell contains
three real-execution repetitions. The full workflow includes three eligible
windows around Evidence Synthesis, Diagnosis Consensus, and Final Report. Each
window launches three optional tool-using auditors whose real Tavily-return
continuations are not dependencies of the top-level result.

The main comparison is default vLLM versus an external online controller that
protects a graph-annotated structural critical frontier through result-ready.
The controller sees only request identity, graph criticality, cold/resume phase,
ready-time prompt tokens, and current frontier state. It never reads the future
evaluation trace or predicts output length/arrival time.

The controlled workflow uses real Qwen3-8B requests on one RTX A6000 and live
Tavily calls. A resume request contains the complete original conversation,
assistant pre-tool output, and the complete Tavily response. Tavily content is
not sliced or padded. The run fails if live Tavily credentials or vLLM are not
available; there is no synthetic fallback.

## Current correctness gate

Before producing paper figures, a run must establish all of the following:

1. client traces contain real SSE first-token and completion timing;
2. all tool returns are live Tavily and `result_truncated=false`;
3. vLLM scheduler traces contain per-request prefill/decode token work;
4. tool-resume requests show non-zero cached-prefix tokens;
5. baseline schedules non-critical prefill work during critical decode;
6. gating reduces that work across the complete reviewer-to-result frontier;
7. every deferred request completes within the maximum defer bound.

## Launch

Apply the scheduler-step instrumentation to the pinned vLLM submodule once,
then start the MAS-environment editable installation:

```bash
git -C vllm apply ../case_study1/vllm_step_trace.patch
bash case_study1/start_vllm.sh
```

Collect the controlled 2/4-agent motifs and the multi-window full workflow:

```bash
bash case_study1/run_paper_motifs.sh
bash case_study1/run_paper_full.sh
```

Regenerate the validated summaries and two paper figures from the retained raw
traces. The scheduler-step trace is stored as JSONL gzip to keep the repository
small; the analyzer reads it directly.

```bash
conda run -n MAS python -m case_study1.analyze_paper_configs
```

No experiment artifact contains the Tavily API key.

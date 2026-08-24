# Ascend prefill/decode and workflow bottleneck experiment

## Scope

This experiment keeps the MAS graph as the primary object of study.  The
controlled prefill/decode sweep characterizes the serving stack, while the
workflow bottleneck map aligns graph-node execution and xPU telemetry on the
same wall-clock axis.

- Hardware: one visible 64 GB die on Atlas 800I A3 / Ascend 910C
- Model: Qwen3-8B, BF16, tensor parallel size 1
- Serving stack: vLLM Ascend + CANN, OpenAI-compatible streaming endpoint
- Workflow mode: raw context propagation with frozen Tavily evidence
- Prefix-cache control in the synthetic sweep: a unique leading nonce for every request

## Controlled sweep

Prefill uses prompt lengths 128, 512, 1024, 2048, and 4096 with 16 forced
output tokens. Decode uses contexts 512, 2048, and 4096, concurrency 1, 2, 4,
and 8, and 128 forced output tokens. Every cell is repeated twice.

Observed Ascend trends:

- TTFT grows from about 45 ms at 128 prompt tokens to 346 ms at 4096 tokens.
- Effective prompt tokens per TTFT rises to roughly 11.8k token/s at 4096 tokens.
- At context 512, aggregate decode throughput rises from 63.7 to 407.9 token/s
  between concurrency 1 and 8, while median TPOT only rises from 15.3 to 17.1 ms/token.
- At context 4096, concurrency 8 reaches 185.9 token/s but TPOT rises to
  27.6 ms/token, exposing context/KV-pressure sensitivity.

These are serving-stack observations, not standalone peak-hardware claims.

## Workflow-aligned map

The detailed figure uses Hierarchical Synthesis because it exposes four
first-level agents, two parallel group synthesizers, a cross-group reviewer,
and a final global synthesizer. Each request is split at its measured SSE first
token timestamp into prefill and decode segments. AI Core utilization, HBM
bandwidth utilization, active prefill/decode counts, and board power are shown
against the same elapsed-time axis.

The first Ascend run shows high sustained utilization during LLM-active graph
stages and a visible power drop around the barrier between the first agent layer
and group synthesis. This is evidence that MAS graph pressure is observable in
the hardware timeline; causal bottleneck attribution still requires a matched
CUDA run and kernel-level profiler evidence.

## Measurement boundary

`npu-smi` sampling takes close to one second on this system. Individual prefill
segments are often only 50--350 ms, so the graph/SSE phase boundary is precise
but instantaneous NPU samples can miss short prefill intervals. The figure uses
markers to expose the real sample cadence. Do not interpret the sparse prefill
samples as a stable prefill utilization estimate. A later publication run
should add CANN/msprof kernel counters and a matched Nsight/DCGM CUDA collector.

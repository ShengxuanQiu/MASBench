# MASBench experiment pipeline: Sections 4.2–4.6

Run the commands below from `mas_workflow/`. All mock results are functional
checks, never device performance measurements. Existing run/replay commands and
plots in `evaluation/` remain available.

## 4.2: structure before performance

```bash
python -m app.structure_space --output ../results/structural-space \
  --widths 2 4 8 --rounds 0 1 2 --revisions 0 1 3 --record
```

This emits standalone separated-factor experiment JSONs, a corpus manifest,
canonical traces, and analysis JSONs. It reuses the actual motif runner with an
explicit **synthetic** accept-after-revision-limit policy. It does not invent a
second topology executor. Refinement limit is not actual revision count under a
real task; these generated control paths must be labeled as controlled synthetic
workloads. Repeated connectivity settings at zero rounds are intentionally
retained as factor cells, not independent real-system samples.

Surveyed MAS coverage and this generated space are distinct corpora. A literature
mapping does not automatically provide an executable trace. Keep system/paper,
task binding, structure, mapping evidence, and fidelity checks for each actual
system that is instantiated. Use arbitrary separated workflow configurations
with `app.benchmark run` to record composed/fan-out/fan-in workloads.

```bash
python -m app.benchmark run --experiment configs/benchmark/experiment.json \
  --trace-dir ../results/recordings
python -m app.workload_analysis TRACE1.jsonl TRACE2.jsonl \
  --output ../results/corpus-analysis
```

Analysis rejects incomplete operation graphs. The corpus report explicitly lists
excluded traces and errors, rather than silently retaining only successful runs.

### Metric definitions

| Metric | Definition and interpretation |
|---|---|
| Graph scope | Realized LLM + tool operations, including local artifact packing. Declared stage DAG is reported separately when present in the run manifest. |
| Width | Exact maximum antichain using reachability and bipartite matching. Beyond 2,000 nodes, exact width is unavailable; maximum level size is a separately named lower bound. |
| Depth | Longest path in number of operation nodes, not wall time. |
| Fan-in/out | Degree distribution over unique predecessor/successor operations. |
| Edge density | Unique ordered DAG arcs divided by `N(N-1)/2`; no transitive reduction. Data/control counts and overlap are separate. |
| Critical-path ratio (static) | Longest path node count / all operation nodes. |
| Artifact reuse | Distinct consuming operations per artifact identity, including zero-consumer artifacts. Also report consumed bytes / distinct consumed artifact bytes; identical text with different artifact IDs remains distinct. |
| LLM information consumption | Artifact bytes delivered to LLM operations, separately from bytes consumed by local packing tools. |

All these quantities are exact for the recorded graph, not for all possible task
control paths. Stage prerequisites currently connect all operations in prerequisite
stages to downstream calls; peer rounds have full synchronization even with sparse
message exchange. Consequently, total edge counts can stay unchanged while data
edges increase. Do not interpret a raw control-graph density as information flow.

## 4.3: trace dynamics

Each analysis JSON includes event-time series and the following definitions:

* `ready_waiting`: ready-event emitted but not submitted (client admission queue).
* `llm_inflight`: submitted through terminal event. This includes backend queuing;
  it is not the number of requests currently executing on a GPU.
* `scheduler_release_delay_sec`: dependency completion to emitted ready event.
* `client_admission_wait_sec`: ready to submit; server queue wait requires backend data.
* `critical_frontier`: dependency-ready but unfinished operations lying on an
  ex-post longest path weighted by observed submit/start-to-finish wall times.
  Includes waiting and executing operations; it is not an online scheduling oracle.
* Barrier arrival spread: last prerequisite completion minus first completion.
  Aggregate branch wait sums the individual early-finish-to-last-finish gaps.
  These are reconstructed synchronization exposure, not literal blocked-thread time.
* Context: message-byte distributions, artifact byte replication, and total message
  bytes / original task bytes. Serialization and role prompts are included. This
  is not a prefix-cache hit rate or model-token amplification ratio.
* Stage markers and operation intervals support phase inspection. No automatic
  semantic phase-transition or bottleneck label is inferred.

Request service weights are recomputed from observed canonical event timestamps;
mock latency model numbers are not used as measured elapsed time.

`KV_live` is unavailable from client traces. The optional in-flight token envelope
uses each request's final reported input+output token count throughout its lifetime;
it is an estimate, not resident KV bytes or a continuous decode-token history.
It omits sharing, allocation rounding, eviction, and retained idle prefixes. If any
request lacks token counts, the envelope is unavailable. Token counts from mock
backends are synthetic; analysis preserves that provenance.

## 4.4: top-level arrivals and serving capacity

```bash
python -m app.study --config configs/studies/run-smoke.json \
  --output ../results/run-smoke
python -m app.study --config configs/studies/run-smoke.json \
  --output ../results/run-smoke --resume
```

`experiments` is a round-robin workload mix; use one entry for an isolated motif
case or an explicit list for a fixed mixture. `count` is total offered workflows,
not per-entry count. The study driver owns arrivals; per-experiment `load` and
arrival fields are ignored. All workflows in a case share deployment concurrency.
Constant and seeded Poisson arrivals use the same schedule across deployments
within a repetition. Case order is randomized with a recorded seed.

The arrival clock never waits for workflow completion. At the configured maximum
in-flight workflows, arrivals are recorded as `client_overflow`, not silently
delayed. Overflow invalidates a claim about backend saturation; raise the client
budget or distribute load generation after measuring client overhead. The current
driver uses threads; it is not an unlimited-concurrency load generator.

E2E starts at the **planned arrival** and ends at canonical run completion. Launcher
lag is therefore included. Final serialization time is recorded separately. Failure
and overflow count against SLO success. p95 is explicitly the completed-workflow
distribution; always publish it with failure rate/SLO success, never alone.
Goodput counts successful workflows within the configured E2E SLO per accounting
window. The window is max(`count/rate`, elapsed through drain). Internal QPS uses
all submitted requests found in available traces, including failed workflows.
Missing traces make that count a lower bound, and are explicitly reported.

These outputs are finite cohorts plus drain, not automatic steady-state capacity
estimates. Choose long enough sweeps, inspect queue growth/drain and launch lag,
repeat independent trials, and define capacity using SLO plus error/overflow
criteria. Backend warmup is excluded but cache state persists. `warm_sequence`
requires a specified warmup count; `uncontrolled` makes no cache-state claim.
Cold-cache labels are rejected until an actual backend reset protocol is integrated.

Each case also writes `case_trace_dynamics.json`, aligning and summing analyzable
workflow timelines on the study clock. Its completeness flag prevents interpreting
missing/invalid traces as idle intervals. Client CPU time, sampled peak thread count,
launcher lag, and finalization distributions help detect load-generator pressure.
The sampled thread maximum is a lower bound, not a profiler measurement.

Every case writes an arrival journal, traces, offline analyses, summary, and an
attempt directory. The manifest stores resolved factors, trace content hashes,
source-code hashes (including untracked Python code), Git state, Python/platform,
and user-supplied deployment hardware metadata. Record the server model/tokenizer
revision, launch flags, driver/runtime versions and profiler configuration in
`DeploymentSpec.hardware`; a remote endpoint does not expose these automatically.

Run `--resume` only with an identical fingerprint. Finished cases are reused;
interrupted cases start a new attempt, preserving partial evidence. Individual
requests are not resumed inside a DAG. Study traces stream to `.partial` files
until an atomic final JSONL replaces them. This survives process interruption at
the last flushed line, not guaranteed storage/power failure. Journaling overhead
must be measured in the real-backend pilot.

## 4.5: fixed realized workload across deployments

First run the structural-space command above, then:

```bash
python -m app.study --config configs/studies/replay-smoke.json \
  --output ../results/replay-smoke

python -m app.replay_compare --source SOURCE.jsonl \
  --replays REPLAY_A.jsonl REPLAY_B.jsonl \
  --output ../results/replay-comparison.json
```

`trace_corpus` reads the generated manifest; `traces` alternatively supplies an
explicit list. For hardware experiments use recordings from the intended task/model
and deployment files for actual A6000/H100/NPU endpoints. No GPU is started or
allocated by this driver. Invariant checks compare operation mapping, data/control
edges, request payloads except model, tool snapshots, artifacts/provenance, and
recorded decisions. Timing changes cannot modify the recorded downstream requests.

Source trace parsing and validation happen before the measured arrival window;
immutable prepared recordings are reused across replay instances.
Replay has independent operation scheduler tasks with a shared LLM semaphore.
Tool snapshot delays do not occupy LLM slots. `max_replay_operations` is a preflight
resource guard; large traces and many simultaneous workflows require client
scalability validation. Replay tool timing reproduces recorded client duration;
it does not benchmark tool hardware or re-execute decision/control logic.

### Prefix cache and strictness

Current adapters explicitly report **best-effort** replay. They fix message text,
generation payload, dependencies, snapshots, and control decisions. They do not
enforce exact input token IDs, exact output length, or output token sequence. Even
with identical downstream payloads, a newly generated upstream output may not
create the same reusable prefix. The comparison report includes output-length
agreement and capability limitations. `--strict` rejects before submission.

For a strict prefix-cache hardware comparison, integrate and verify model/tokenizer
identity, exact input IDs, actual forward decoding with forced recorded output
tokens, and an explicit initial cold/warm protocol. Merely replacing returned text,
setting a maximum length, adding random per-request salts, or clearing cache every
stage does not implement that comparison. Without this extension, publish replay
as best-effort, report output-token variation, and use a backend-configured
cache-disabled baseline when isolating structure from prefix reuse. Record and
verify the server's cache setting; this client does not toggle it.

Set `collect_backend_metrics: true` for one endpoint-wide Prometheus sampler per
case. Raw labeled metrics and canonical `backend_measurement` observations are
preserved with backend-reported provenance. Missing/failed metrics remain errors,
not zero resource use. They are endpoint scope, not attributed to individual
workflows, and require an otherwise isolated endpoint. Unix timestamps align
these samples with the study's recorded clock origin; do not merge unrelated
relative clocks. No vLLM-specific adapter is presumed for an NPU endpoint.

Compute, bandwidth, memory capacity and device synchronization require GPU/NPU
profiling or validated backend counters. Client TTFT/latency alone cannot identify
them. `replay_compare` deliberately leaves bottleneck classification unset.

## 4.6 and acceptance before scaling

Derive implications only from established observations and their tested mechanism.
Use the same task/payload budgets and control path for structural interventions
where possible; report total requests, tokens and critical-path work when they
cannot be matched. Keep runtime bottleneck categories distinct from hardware
resource categories. Saturation-driven server queue growth can be an effect of
another bottleneck rather than its cause.

Before expensive sweeps, require: tests pass; real-backend small traces round-trip;
replay invariants pass; no client overflow; tolerable launch/finalization/trace
overhead; model/tokenizer/cache metadata complete; endpoint measurements validated;
and repeated pilots produce consistent distributions. There is no automatic
guarantee of hardware fidelity from a passing mock test.

```bash
python -m pytest tests -q
```

Compatibility note: typed data dependencies are no longer automatically duplicated
as control edges merely because they also occur in `parents`. Graph reachability
and scheduling are unchanged. Regenerate old analysis JSONs with the new analyzer;
old graph exports can contain inflated control-edge counts. Legacy summary fields
such as `critical_path_length` are retained for compatibility; use the explicitly
defined new analysis metrics for Sections 4.2–4.5.

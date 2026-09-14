# Publication experiment path and readiness

The canonical path is `WorkflowSpec → (MotifSpec | AtomicStage) → realized ExecutionGraph → trace analysis / fixed-workload replay → study → capacity`. The existing family runner, trace journal, replay scheduler and open-loop study driver are reused. Legacy topology/case-study runners remain available but are not the publication experiment path.

## Run and resume

From `mas_workflow/`, with `requirements-benchmark.txt` installed:

```bash
# Functional validation only; contains no hardware performance evidence.
python -m app.benchmark run --experiment configs/publication/adaptive.json --trace-dir ../results/adaptive
python -m app.publication --manifest configs/publication/matrix-functional.json --output ../results/publication-functional --execute
python -m app.publication --manifest configs/publication/matrix-functional.json --output ../results/publication-functional --execute --resume

# Copy and edit the templates before launching a real matrix.
# Without --execute this only expands the matrix into reviewable experiment cells.
python -m app.publication --manifest configs/publication/matrix.template.json --output ../results/main-matrix
python -m app.publication --manifest configs/publication/matrix.template.json --output ../results/main-matrix --execute --resume

# Replay the same recorded corpus across hardware, with output-length checks.
python -m app.capacity --config configs/publication/replay-capacity.template.json --output ../results/hardware-replay
python -m app.capacity --config configs/publication/replay-capacity.template.json --output ../results/hardware-replay --resume
python -m pytest -q
```

`matrix.template.json` declares workload presets, structural parameters, task files, deployments, QPS points, repetitions, SLO, cache policy and telemetry. The included rates/count/SLO are starting values, not measured recommendations. Add task files to the `tasks` map to cross every structure with every task. One capacity search runs per structure/task cell; deployments are compared within that cell. The template covers all four motifs, three compositions and matched dependency controls. A formal run requires real endpoint names, device identity and model identity; the templates intentionally do not invent them.

`study` uses finite arrival cohorts followed by drain. Choose a sufficiently long cohort after a pilot, check launch lag and client overflow, and verify stability with longer cohorts. It is not a steady-state simulator. An entire large matrix is not launched by this code change.

## Adaptive and atomic structure

`StageSpec.condition = {from_stage, field, equals}` (or `in`) evaluates a prior recorded control decision; if none exists, it selects a field from the prior stage's JSON result. That selector creates a prerequisite automatically. Activation and participant decisions are recorded. A false condition emits `stage_skip`; a skipped prerequisite propagates a skip unless `allow_skipped: true`. A join with this flag consumes only artifacts from executed branches. Selectors whose own decision source was skipped are not meaningful; route the join without such a selector.

`participants = {from_stage, field}` selects a nonempty list of unique worker indices within the declared width. It applies to DispatchExecute, ParallelAggregate and PeerDeliberation; Peer requires at least two selected peers. Connectivity is evaluated within that realized subset. Invalid selections fail rather than silently modifying the graph. Arbitrary self-modifying graphs are outside scope.

`atomic: {kind: llm|transform|router|tool, role: Worker}` replaces the motif reference. These stages emit formal LLM/tool operations and artifacts, with no collaboration motif identity. Deterministic task functions live in `TaskBinding.stage_bindings.<id>.parameters`: transforms support identity/constant/concat/json_field; routers support a constant object or a lookup/default map; tool-only stages call the configured search provider. An LLM stage can also supply JSON decisions to a later selector. This is not arbitrary Python execution or an autonomous tool loop.

`TaskBinding.stage_bindings` overrides task input, roles, criteria, routing/evaluation semantics, delivery instruction and function parameters. Topology and deployment overrides are rejected. Task-specific decision values affect which declared path is realized, not the set of legal stage edges.

## Artifact delivery

`StageSpec.delivery` declares a mode per `context`/`candidate` port:

| Mode | Execution | Provenance |
|---|---|---|
| full | Pass the original content | Original artifact and producer |
| selected | Task binding supplies artifact indices | Selected original artifact IDs |
| summarized | An additional LLM operation summarizes inputs | Summary artifact plus all source IDs |
| referenced | A tool operation produces `artifact://<id>` | Reference artifact plus source ID |
| retrieved | A tool operation fetches the recorded in-memory artifact | Retrieved snapshot plus source ID |

All consuming LLM/tool operations record producer, consumer, artifact, mode and source IDs. Delivery transformations add real operations to the DAG; they are not free preprocessing. `retrieved` currently means local artifact retrieval, not an external vector database or a RAG quality claim. `referenced` does not secretly resolve the reference inside the LLM request. Summary prompts and selection indices are task bindings. A summary uses the task-level `delivery_instruction` default, optionally overridden per stage. Candidate ports must deliver exactly one artifact.

Replay reuses the realized operations, dependencies, downstream payloads, tool snapshots, delivery provenance and control decisions. Newly generated text never determines a downstream payload or branch. Skip records are preserved as recorded logical outcomes, not retimed stage execution events.

## Structural controls and pressure metrics

Presets: `dispatch`, `parallel`, `refine`, `peer`, `parallel_refine`, `dispatch_parallel_refine`, `fanout_fanin`, `matched_chain`, `matched_parallel`. Peer `random_k` accepts either `k` or sparsity in [0,1]; degree is `round((1-sparsity)*(participants-1))`, capped to the active subset. Record actual data edges; integer rounding can produce identical graphs for neighboring sparsity values.

Matched chain/parallel contain N atomic LLM requests with an identical request-payload multiset and output caps, differing only in control dependencies. They do not pass generated answers downstream. With the same tokenizer/template their input token budgets match; actual output tokens and task quality are not guaranteed equal. Use them as mechanism controls alongside realistic composed workloads, not as a claim that all motifs perform equal useful work.

| Quantity | Output / definition |
|---|---|
| Structural width/depth, fan-in/out, critical path | Existing `structure` and `runtime` analysis |
| Work amplification | `pressure.work_amplification`: LLM request count per workflow relative to one atomic call; no quality normalization |
| Artifact amplification | `structure.artifact_byte_amplification`: consumed bytes / unique consumed artifact bytes; includes tool consumers; LLM-only consumed bytes are separate |
| Input-token amplification | Total backend prompt tokens / root LLM prompt tokens; unavailable if token counts are missing |
| Temporal pressure | Ready-wait area, peak ready and inflight; study also merges all workflows on a common clock |
| Synchronization exposure | Sum of branch waiting at recorded barriers |
| Logical state residency | Estimated unique artifact payload bytes from production until last consumer finishes; final outputs until run end |
| Physical cache/device observations | Separate canonical backend/device measurements; never derived from the logical state estimate |

Logical state excludes Python object overhead, allocator behavior, actual KV and Mamba state. A client token envelope is not physical residency. Artifact reference bytes, serialized prompt bytes and backend token counts are distinct quantities.

## Capacity and uncertainty

The controller runs the coarse rates, finds the first failing tested rate above a passing rate under the configured SLO/error criteria, and inserts `dense_points` inside that bracket. It aggregates independent repetitions, including mean/median/p95 and a bootstrap 95% CI of repetition means. A single repetition has no CI. This is a CI across repetitions, not a pooled-request p95 confidence interval.

`lambda_knee` is the upper tested bound of the refined SLO boundary; `rho = offered workflow QPS / lambda_knee`. The bracket is reported as well. All-pass/ all-fail/nonmonotonic/client-invalid results do not fabricate a knee. Client queue growth is a linear slope of ready-waiting requests sampled over the arrival window; it is not backend queue length. Raw backend waiting metrics remain separately available. Diagnose compute/bandwidth/capacity/synchronization bottlenecks only with supporting device and profiler observations.

An existing study output is resumable only if its resolved configuration, input trace hashes and application source hash match. Completed cells are skipped; interrupted attempts are preserved and retried. Each study saves resolved config, Git/environment metadata, application source snapshot, archived replay source corpus, arrival journal, canonical traces, device observations and analyses. Generated run traces are already inside the result directory. Archive the whole result directory, and separately record the serving engine build/launch configuration and model files; the client cannot reconstruct a remote server's binary from an endpoint URL.

## Telemetry and cross-hardware protocol

Deployment `telemetry` selects `adapter` (`vllm_gpu`/`ascend`/`generic`), `metrics_url`, an explicit GPU `device_id` (prefer UUID) or Ascend `npu_id`, and optional metadata. Study creates one endpoint sampler with this adapter. Device sampling continues if HTTP metrics fail. Missing counters are null/unavailable, including counter resets that make a delta invalid.

GPU sampling uses nvidia-smi for utilization, memory and power; cache counters are reported separately by the serving endpoint. Ascend invokes `npu-smi info -t usages -i <id> -c <chip>` and `-t power`. Optional `profiler_counters_path` reads a live JSON snapshot `{device_id, timestamp_unix, metrics}` with finite numeric counters and a maximum age of five seconds. A profiler/exporter must produce that file; this code does not start a vendor profiling session. Vendor usage/power command formats follow [Ascend usage documentation](https://www.hiascend.com/document/detail/zh/Atlas%20200I%20A2/253RC1/re/npu/npusmi_020.html) and [Huawei power documentation](https://info.support.huawei.com/enterprise/zh/doc/EDOC1100493502/ab5d7259). Firmware output varies; unsupported fields remain unavailable. The Ascend parser is fixture-tested, not validated on a live NPU in this change.

Output-length agreement compares recorded/replayed backend token counts, using relative error `abs(new-old)/max(1,old)` and `output_length_tolerance` (default zero). Missing counts or excess variation make that replay comparison ineligible. Even equal lengths do not prove exact token sequences, equal tokenizers, or identical kernel work; strict replay remains unsupported.

Compute verifiable local tokenizer/template evidence:

```bash
python -m app.replay_protocol --model-dir /path/to/model --revision EXACT_MODEL_REVISION --output model-identity.json
```

Copy the resulting `identity` object into the deployment, retain its evidence and attest that the endpoint uses those files without template overrides. `model_revision`, `tokenizer_sha256` and `chat_template_sha256` are compared automatically in replay. These are local file hashes and recorded metadata, not remote attestation of model weights. Record precision, engine version and launch flags alongside them.

`cache_disabled` requires endpoint metrics proving prefix caching is disabled. `warm_cache_enabled` requires enabled caching plus a positive `warmup_count`. Evidence comes from `enable_prefix_caching` configuration labels or an explicitly configured `metadata.prefix_cache_enabled_metric`. If unavailable or contradictory, the real experiment fails before requests. The driver does not restart servers or toggle flags. Run the disabled baseline on a disabled server; then launch the enabled sensitivity server and preserve its launch record. Warmup does not reset cache history or guarantee identical residency across repetitions. `warm_sequence` and `uncontrolled` remain compatible but unverified. Mock cache evidence is synthetic and cannot validate a real cache protocol.

## Main-experiment readiness checklist

Acceptance on gpu3 (2026-09-14): all 128 `mas_workflow` tests and 6 legacy case-study tests passed (134 total), using `PYTHONPATH=mas_workflow:. python -m pytest mas_workflow/tests case_study1/tests case_study2/tests --import-mode=importlib -q` from the repository root. Third-party vendored engine test suites are not included. The final mock matrix completed 20 load cases / 80 workflows across five structure cells and resumed without resubmitting completed cases. The formal template expanded into 43 cells without launching hardware experiments. Adaptive CLI run completed and exported canonical graph/trace views. Validation artifacts remain under `results/publication-functional-final`, `results/publication-main-plan` and `results/adaptive-final` on gpu3; generated results are not committed.

- [x] Canonical adaptive and atomic run/replay, delivery provenance and composed presets implemented.
- [x] Exact-payload matched dependency controls, parameterized peer degree, separated pressure measures implemented.
- [x] Coarse/dense capacity search, repetition aggregation, invalid/censored results, fingerprints and resume implemented.
- [x] Study adapter wiring, missing telemetry semantics, length checks and cache evidence checks implemented.
- [x] Functional unit/integration tests and executable mock matrix; no mock measurements used as hardware evidence.
- [ ] Set real deployments, exact model/tokenizer/template identity, precision and serving engine build/flags.
- [ ] Pilot the **new** adaptive/delivery/replay paths on each target GPU/NPU; older Qwen3.5-4B acceptance predates these additions.
- [ ] Verify live NPU field support, GPU UUID mapping, exclusive endpoint attribution and metric time alignment.
- [ ] Verify disabled/enabled prefix-cache evidence and document warmup/history protocol separately.
- [ ] Calibrate SLO, arrival duration, rates, concurrency and repetitions; rule out client launch/admission limits before assigning backend capacity.
- [ ] Freeze the source corpus, inspect output-length/identity agreement, and exclude ineligible comparisons.
- [ ] Collect profiler evidence before assigning hardware bottleneck mechanisms; validate coverage/quality with the paper's independently curated corpus.

The matrix execution machinery is ready for controlled pilots and subsequent scaling. Hardware conclusions and a fully validated cross-hardware publication protocol still require the unchecked steps above.

## Modified files and compatibility

Core changes: `app/specs.py`, `app/motifs/contracts.py`, `app/motifs/families.py`, new `app/motifs/adaptive.py`; `app/execution_graph.py`, `app/replay.py`, `app/replay_compare.py`, new `app/replay_protocol.py`; `app/workload_analysis.py`, new `app/pressure.py`; `app/study.py`, new `app/capacity.py`, `app/publication.py`; `app/backend_adapters.py`, `app/backend_metrics.py`; `configs/canonical_trace.schema.json`; new `configs/publication/*`, `tests/test_publication_features.py`, and this document/README.

Existing motif configuration and canonical replay remain compatible. New trace event types are additive within `masbench_execution_v1`; external enum validators must load the updated schema. `ExecutionGraph.to_dict()` now includes `deliveries`. Missing telemetry counter deltas/rates return null instead of a misleading zero. Existing `uncontrolled`/`warm_sequence` cache configurations are retained without becoming verified protocols. The previously authorized `progress/` archive deletion is included; historical analysis scripts and the locally modified vLLM submodule are not part of this refactor.

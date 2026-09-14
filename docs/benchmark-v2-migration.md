# Hierarchical MASBench: architecture, trace contract, and migration

## Audit and implementation boundary

The previous main path already implemented four motifs, role instances, artifact
transfer, admission control, search providers, backend adapters, and trace views.
The missing pieces were a workflow DAG scheduler, independently replaceable
experiment factors, a validated realized operation graph, and fixed-workload
replay. This revision extends those implementations in place. Historical
topology/full-workflow/case-study implementations remain explicit legacy paths;
they are not competing coordination frameworks in the new structure model.

`app/specs.py` defines the public factor contracts. `ExperimentConfig.compile()`
joins validated structure and task objects into the existing runner's private
stage dictionaries. These compiled dictionaries are execution plans, not public
factor definitions. `app/motifs/families.py` still owns the four motif algorithms.
`app/runtime.py` still owns LLM invocation, search, admission, and backend metrics.
Backend code receives requests and never receives or changes the dependency graph.

## Factors and roles

`StructureSpec` contains named `MotifSpec` objects and a `WorkflowSpec`. A stage
references one motif; `depends_on` declares control prerequisites and `inputs`
declares artifact edges. Artifact references also imply prerequisites. References
can point forward in the file. Duplicate IDs, missing references, self edges,
cycles, unsupported connectivity, and invalid bounds are rejected before execution.
The runtime releases each stage once all prerequisites complete successfully.
Independent stages run concurrently; there is no global layer-by-layer barrier.
Failure stops new stage launches while already active stages finish and are recorded.

The four families accept `DispatchExecute`, `ParallelAggregate`, `EvaluateRefine`,
and `PeerDeliberation` names (snake_case aliases remain valid). Connectivity and
coordination are validated motif properties. Dispatch/aggregation variants belong
to structure because they determine local operations. Routing/evaluation semantics
and criteria belong to task bindings. Motif protocol parsing validates the decision
format; it does not supply application-specific acceptance criteria.

`TaskBinding` owns task text, canonical role prompts, search bindings, role output
format/schema, and routing/evaluation criteria. The currently bound tool is `search`,
performed once before each call of a role that requests it. This reuses existing
providers; it is not an autonomous tool-calling loop. JSON Schema validation of
`RoleTask.io_schema` is applied to the parsed role output; `input_schema` validates
the assembled role input object (`task`, `role_index`, `inputs`, optional tool evidence).
Tools and schema never
select workflow edges. Default role prompts live in `motifs/task_defaults.py`.

Canonical roles map to old diagnostic slot names as follows:

| Canonical role | Compatibility slot |
| --- | --- |
| Coordinator | dispatcher |
| Worker | executor, worker, producer, peer |
| Reducer | collector |
| Reviewer | evaluator |

New `RoleTask` rejects model/endpoint fields. `DeploymentSpec` owns model, endpoint,
backend, generation fields, hardware metadata, and a global LLM concurrency limit.
Hardware describes the externally provisioned serving environment; the benchmark
does not launch or reconfigure GPU services. `ExperimentConfig` adds workflow count
(`load`) and deterministic inter-arrival spacing. Multiple runs share the LLM
concurrency limit, each with independent IDs and traces. Role configurations are
reusable; agent instances and mutable histories are created only at runtime.

## Canonical trace v1

Every new family run builds `trace.execution_graph` while executing and exports
`*_execution_graph.json`. Nodes are actual LLM calls and tool operations. Local
packing/selection of artifacts is an explicit local tool operation, not an extra
LLM request or a fictitious stage node. Stage/motif/agent identities annotate the
operations; stage and barrier events are not vertices in G.

The JSONL stream retains old `event_type`/flat fields for compatibility. Formal
records have `canonical_schema: masbench_execution_v1`, `canonical_type`, and three
attribute groups: `operation`, `information_flow`, `serving_observation`. The
machine-readable contract is `configs/canonical_trace.schema.json`. Additional
legacy diagnostic and simulator-view records can coexist in the stream; graph
reconstruction consumes only formal records.

| Canonical events | Contract |
| --- | --- |
| request_ready / request_submit / request_finish / request_fail | LLM lifecycle; `operation_phase` gives ready/started/finished/failed; submit stores the complete request payload |
| operation_start / operation_finish / operation_fail | Tool lifecycle, recorded snapshot and elapsed duration |
| dependency | src, dst and data/control edge kind |
| artifact_produce / artifact_consume | Stable artifact ID, exact content, hash, producer and consumer |
| control_decision | Parsed routing, selection, vote, or acceptance outcome |
| barrier_sync | Actual synchronization prerequisites |
| backend_measurement | Request measurements or backend telemetry |
| run_start / run_finish / stage_start / stage_finish | Run and stage lifetime |

Run, stage instance, motif instance, agent instance, operation, request, and artifact
IDs have independent identities. New entry-point runs use UUIDs; an explicitly
supplied legacy run ID remains the caller's responsibility. Every event snapshots
mutable values at emission. Stage prerequisites expand to control edges from the
completed prerequisite operations; peer round barriers expand to control edges
from all prior round peer calls, including peers that did not send data to a given
recipient. Ready is recorded after logical dependencies, before deployment capacity
waiting. Submit is recorded after capacity/admission. Backend timing is then observed.

Each measurement is `{value, source}`, with source `observed`, `backend-reported`,
`estimated`, or `unavailable`. Backend token counts are reported when supplied;
heuristic token counts are estimated. A backend's placeholder queue-wait zero is
normalized to unavailable. Request wall-clock/SSE timing is client observation,
not a claim about hardware kernels. Mock timing is only a mechanism smoke test.
Canonical operation graphs drive the existing architecture spans, HTML and OTel
views; old diagnostic tool events do not create duplicate canonical tool spans.

## Fixed-workload replay

`app.replay.replay_trace()` rebuilds and validates G from JSONL before submission.
It rejects dangling references, cycles, duplicate IDs, incomplete operations,
missing payloads/snapshots, and corrupted artifact contents. A bounded concurrent
scheduler releases an operation only after **all recorded parents** finish.

Replay never runs the motif/workflow algorithms, decision parsers, or live tools.
It submits each recorded downstream request payload directly. New model content
is measured but is never used to construct another prompt or choose a branch.
Artifacts retain recorded contents; new IDs are namespaced under the replay run
with source IDs recorded. Tool snapshots and recorded client durations are replayed.
Recorded decisions and synchronization are retained as trace evidence.

The replay deployment can change endpoint/backend/model and concurrency. Generation
parameters come from each recorded request; deployment generation settings are
ignored during replay. Changing models/tokenizers may change tokenization and is
an explicit experimental intervention, not an identical-model performance claim.
Keep model/tokenizer fixed for backend-only comparisons. The manifest records the
deployment and declares that only the payload's model field may differ.

Current mock and OpenAI-compatible adapters cannot guarantee exact recorded output
token counts. `max_tokens` is a cap, not an exact length requirement. Replay reports
`best-effort` and `output_length: best-effort`; `--strict` rejects the request before
any backend work. OpenAI-compatible canonical runs use the existing streaming
client directly, with no silent fallback that could discard generation parameters.
Request payloads support the current fixed-action system/user message pair. Legacy
ReAct traces do not meet this replay contract and require a fresh canonical recording.

## Configuration compatibility and examples

Start with `configs/benchmark/experiment.json` or `.yaml`. It references separate
`structure.json`, `motif.json`, `workflow.json`, `task.json`, and `deployment.json`.
References resolve relative to the experiment file. JSON needs only the standard
library for mock/HTTP work; YAML and optional JSON Schema validation need the
packages in `requirements-benchmark.txt`.

Old `python -m app.main --workload-config ...` commands and role backend overrides
remain supported. A legacy list with no `depends_on` fields retains its sequential
behavior, including implicit control edges. Add an explicit `depends_on` field
(including `[]`) to opt into DAG behavior for an old list. The new typed workflow
always uses explicit DAG semantics. Move task/roles/criteria into `TaskBinding`,
structural bounds/variants into `MotifSpec`, and backend/model fields into
`DeploymentSpec` when migrating. New factor constructors reject misplaced fields.

The old summary label `motif_families_v1` remains for consumers; the canonical trace
version separately identifies the execution contract. Do not infer new replay
support for old traces from that summary label. Existing user analysis files and
the vLLM submodule are outside this refactor.

## Validation and remaining empirical work

Regression tests cover factor isolation; reversed-order fan-out/fan-in;
simultaneous independent stages; trace round trips, IDs and provenance across all
motifs; replay dependency timing; tool snapshots; adversarial new acceptance
outputs; generation preservation; and strict/incomplete replay rejection.

These implementation tests do not establish empirical workload coverage, task
quality, or real GPU performance. The paper's R1 coverage claim still requires
mapping and evaluating representative real systems. Real backend experiments
must record model/tokenizer, hardware, generation, and capability differences.

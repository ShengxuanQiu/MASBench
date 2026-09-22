# MASBench Semantic Trace 1.0

MASBench benchmarks the serving efficiency of a realized multi-agent workload. Task correctness and agent capability are outside the primary score. A native framework execution is first collected, then canonicalized into a versioned Semantic Trace. Official replay consumes this trace and a resolved ScenarioManifest.

The Part 3 abstraction is frozen as a hierarchical workflow of collaboration stages. Each stage records Participants, Execution Control, and Context Construction. The four standard reference templates are Spawn, Fork--Join, Refinement Loop, and Debate. Shared state is an optional extension.

## Bundle layout

```text
trace/
  manifest.json
  operations.jsonl
  artifacts/
    sha256/<content-hash>
```

`manifest.json` contains TraceMetadata, TaskRecord, StageInstance, AgentSession, ArtifactRecord, and optional shared-state records. `operations.jsonl` contains one OperationRecord per line. Large prompts, tool results, and intermediate artifacts belong in the content-addressed store.

`source_observations` may contain source timestamps, queue time, TTFT, ITL, utilization, cache counters, and profiler data. This field is excluded from the workload hash and is never used as target service time or dependency-release time.

## Minimal operation

```json
{"operation_id":"op:worker","operation_type":"llm","task_id":"task:1","stage_instance_id":"stage:spawn:0","session_id":"session:worker","iteration":0,"control_dependencies":[],"data_dependencies":["op:dispatch"],"state_dependencies":[],"controlled_external_delay":0.0,"llm":{"canonical_request":{"model":"model","messages":[{"role":"user","content":"recorded input"}],"max_tokens":4},"input_token_ids":[1,2,3],"input_token_count":3,"tokenizer":{"name":"tokenizer","version":"1","hash":"..."},"chat_template":{"name":"template","hash":"..."},"generation_parameters":{"max_tokens":4},"recorded_output_token_ids":[4,5,6,7],"recorded_output_length":4,"recorded_output_ref":null,"context_artifact_ids":["artifact:dispatch"],"reuse_scope_id":"scope:worker"},"tool":null,"transform":null,"state":null,"source_observations":{}}
```

The checked-in golden bundles are the authoritative complete examples.

## Replay contract

An operation becomes ready only after every semantic predecessor finishes on the target backend, followed by its deterministic controlled external delay. Root task arrivals come from ScenarioManifest. Official tool replay reads recorded results and never calls a live external service.

`length_locked` fixes realized operations, dependencies, resolved input, and output length. Generated content is discarded and never changes a downstream request. `token_locked` additionally requires exact recorded output token IDs. Backends without this capability return `REPLAY_MODE_UNSUPPORTED`; no fallback is allowed.

Cache reuse is authorized by `reuse_scope_id`. Scenario resolution creates a deterministic salt for every scope from the scenario seed. Requests in different scopes receive different prefixes; requests in one scope receive the same prefix.

## Identity boundary

The workload/scenario hash includes trace content, replay mode, arrivals, cache policy, tokenizer, chat template, and workload generation parameters. SystemConfig contains backend, hardware, batching, cache implementation, scheduler, placement, and parallelism. Changing only SystemConfig leaves workload and scenario identities unchanged.

## Chakra boundary

Semantic Trace remains canonical. Chakra lowering requires real host/device/operator instrumentation annotated with semantic operation, task, and stage IDs plus an explicitly supplied official converter. The bridge never converts source request latency into a Chakra compute-node duration. The initial path preserves a fixed serving execution plan and does not claim scheduler or batching feedback.

## Commands

Use `python -m app.semantic.cli collect` to canonicalize a native JSONL trace with an exact local tokenizer. Use `resolve-scenario` to materialize deterministic cache-scope prefixes and freeze the exact request hashes and token IDs that the target will receive. `validate`, `replay`, `measure`, `capacity`, `features`, `coverage-workflow`, `coverage-cluster`, and `lower-chakra` implement the remaining pipeline stages. See the repository README for complete examples.

## Metric definitions

`task_latency` is target task finish minus ScenarioManifest root arrival minus deterministic external delay on the realized critical terminal path. Task goodput counts successful tasks meeting the configured SLO per measured second. SLO attainment divides good tasks by all attempted tasks, including failed and incomplete tasks. Sustainable capacity is the maximum tested arrival rate or active-task concurrency meeting the scenario's configured attainment target. Resource efficiency is task goodput divided by accelerator count. Token/request throughput and serving counters remain diagnostics.

Behavioral coverage features are computed from semantic dependencies and recorded model inputs, never hardware time. `context_growth` is the mean within-session range of exact input-token counts across iterations. `prefix_context_overlap` is the mean pairwise longest-common-prefix fraction within declared reuse scopes. `controlled_external_delay_fraction` is the fraction of semantic operations carrying a positive controlled delay. PCA normalization and basis fitting use held-out real workloads only; MASBench workloads are transformed into that fixed space before nearest-cluster assignment. Byte/token replication is reported as an operational delivery measure and is not called semantic-information duplication.


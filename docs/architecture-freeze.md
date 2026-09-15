# MASBench architecture freeze

The publication path implements Section 3 without adding another runner:

- `WorkflowSpec` realizes `W=(S,D,Ω)`: stages, artifact/control dependencies, and recorded activation/participant decisions.
- Each canonical `MotifSpec` realizes `G_s=(V_s,E_s,π_s,μ_s,τ_s)`: participants, producer-consumer connectivity, fixed family coordination, typed edge delivery, and completion condition.
- `AtomicStage` remains outside the four collaboration families. Its `llm`, `tool`, `transform`, and `router` operations retain their real operation kind.
- The public fourth family is `PeerExchange`. `PeerDeliberation` and `peer_deliberation` remain input aliases and compile to `peer_exchange`.

## Typed edge delivery

`DeliverySpec(mode, selector, transform)` supports `full`, `selected`, `summarized`, `referenced`, and `retrieved`. A stage input can use a separate policy on each edge:

```json
{"context": [
  {"source": "left.result", "delivery": {"mode": "full"}},
  {"source": "right.result", "delivery": {"mode": "selected", "selector": {"json_fields": ["answer"]}}}
]}
```

The older `delivery: {context: "summarized"}` form remains valid and applies once to the whole port. Legacy `parameters.delivery.<port>.indices` is accepted. New selection uses `artifact_indices` or deterministic `json_fields`.

The same implementation handles internal relations:

| Family | Delivery relation |
|---|---|
| DispatchExecute | `coordinator_to_worker` |
| ParallelAggregate | `worker_to_reducer` |
| EvaluateRefine | `producer_to_reviewer`, `reviewer_to_producer` |
| PeerExchange | `peer_to_peer` |

Configure these in `MotifSpec.delivery`. Summary creates an LLM operation; JSON selection/reference creates a transform operation; retrieval creates a retrieval operation. Every materialized relation records source and delivered artifacts, producer and consumer operations, mode, selector/transform metadata, and byte/token size. Reference is an artifact URI and retrieval is a recorded local snapshot; neither claims an external RAG system.

## Realized hierarchy and completion

`ExecutionGraph.to_dict()` exports operations/edges/artifacts plus `stages`, `decisions`, `deliveries`, and `hierarchy.operation_to_stage`. Each stage contains logical and instance IDs, motif or atomic identity, realized participants, dependencies, `E/π/μ/τ` semantics, activation/skip decisions, status, and completion reason.

Completion reasons are `selected_work_returned`, `required_branches_completed`, `accepted`/`revision_limit`, `fixed_rounds_completed`, atomic `operation_completed`/`router_decision_emitted`, skip reason, or `failed`. Replay copies recorded hierarchy for analysis but schedules only the flattened operation DAG. It never reruns orchestration or delivery transforms.

## Section 4.1 audit

The repository contains a marked example, not an invented 82-paper annotation:

```bash
cd mas_workflow
python -m app.coverage_audit --input ../evaluation/coverage_audit.example.json --output ../evaluation/coverage/example-metrics.json
python ../evaluation/plot_coverage.py --metrics ../evaluation/coverage/example-metrics.json --output-dir ../evaluation/coverage/example
```

The schema requires paper/system, stages, canonical and represented families, workflow dependencies, producer/consumer information relations, artifact semantics, delivery mode, and extension type. The calculator outputs `C_sub`, strict `C_wf^0`, and micro/macro `F_dep^spec` and `F_info^spec`. Plotting reads calculator output and contains no paper counts.

- `C_sub`: local stages whose annotated canonical family is represented by the benchmark family.
- `C_wf^0`: papers where every stage family is represented and the directed dependency set matches exactly.
- Dependency F1 uses `(paper, producer stage, consumer stage)`.
- Information F1 uses `(paper, producer, consumer, artifact semantics, delivery mode)`.
- Macro F1 averages per-paper F1; micro F1 pools relations.

## Quality and information metrics

Study `quality_evaluator` runs after the measured workload. Deterministic modes are exact text, numeric tolerance, and JSON-field numeric tolerance. Results go to `quality.json` and summary score/pass fields. They add zero benchmark requests, tokens, or latency. `register_evaluator()` is the explicit extension point for an external judge; no network judge is bundled or invoked by default.

Analysis reports delivered bytes/estimated tokens, unique source artifact bytes/tokens, reuse multiplicity, compression ratio, and reuse-inclusive context/information amplification. Compression compares materialized delivery with corresponding source payload; amplification compares all delivery payloads with unique source artifacts. These byte/token measures do not measure semantic-information duplication. Logical artifact residency remains separate from backend physical KV/cache observations.

## Publication templates

`app.publication` retains presets and adds `base` plus declarative `overrides`/Cartesian `factors` with validated dotted paths. Every repetition persists its exact arrival offsets and seed.

- `structural-sweep.json`
- `information-delivery-sweep.json`
- `hierarchical-phase-transition.json`
- `capacity-sweep.template.json`
- `cross-hardware-replay.template.json`

The delivery matrix fixes a PeerExchange graph while changing `peer_to_peer` delivery, and fixes a PA→ER workflow while changing its cross-stage edge delivery. Mock entries enable dry-run/functional validation only.

## Architecture-freeze checklist

- [x] One WorkflowSpec/MotifSpec/AtomicStage model maps directly to `W` and local `G_s`.
- [x] Typed per-edge and intra-motif delivery uses traced operations and complete provenance.
- [x] Stable family semantics and explicit completion reasons are exported.
- [x] Flattened replay remains fixed while the realized hierarchy is preserved.
- [x] Section 4.1 schema, validator, calculator, data-driven plot, and non-evidence example exist.
- [x] Out-of-band deterministic quality evaluation is separate from serving accounting.
- [x] Generic overrides and five templates exist; arrival schedules are persisted.
- [x] Information metrics state their byte/token definition and semantic limit.
- [x] CI runs tests, schema/audit validation, and publication dry-run.
- [ ] Fill and independently review real paper/system annotations and task references.
- [ ] Freeze publication task/config/corpus versions after pilots.
- [ ] Validate delivery policies and quality thresholds on real model outputs.
- [ ] Validate GPU/NPU telemetry, cache protocol, identity, output-length agreement, load generation, SLO boundary, and profiler attribution on each deployment.

The checked items freeze the core abstraction. Remaining work is data curation, configuration freeze, deployment validation, and Part 4 execution. Core changes should now be limited to defects or evidence-driven corrections.

Validation on gpu3 (2026-09-15): 147 first-party tests passed (141 under `mas_workflow/tests`, 6 legacy case-study tests). Section 4.1 example validation and data-driven PDF rendering passed. Publication dry-runs expanded 17 structural, 8 delivery, 21 hierarchical phase-transition, and 9 capacity cells. These are configuration/functionality checks, not coverage, task-quality, GPU, or NPU measurements.

## Files in this freeze

Core: `app/specs.py`, `app/motifs/contracts.py`, `app/motifs/task_defaults.py`, `app/motifs/families.py`, `app/motifs/adaptive.py`, `app/execution_graph.py`, `app/replay.py`, `app/replay_compare.py`, `app/workload_analysis.py`, `app/study.py`, `app/publication.py`; new `app/quality.py` and `app/coverage_audit.py`.

Evaluation/configuration: `evaluation/coverage_audit.schema.json`, `evaluation/coverage_audit.example.json`, `evaluation/plot_coverage.py`, publication templates, canonical trace schema, CI workflow, tests, README and readiness documents.

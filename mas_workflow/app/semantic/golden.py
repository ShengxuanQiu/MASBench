"""Synthetic golden traces for the four frozen reference templates."""
from __future__ import annotations

from pathlib import Path

from .canonicalize import ByteTokenizer
from .model import AgentSession, ArtifactRecord, OperationRecord, SemanticTrace, StageInstance, TaskRecord, TraceMetadata
from .scenario import ScenarioManifest, SystemConfig
from .store import TraceBundle

TEMPLATES = ("spawn", "fork_join", "refinement_loop", "debate")


def _llm(tokenizer: ByteTokenizer, oid: str, task: str, stage: str, session: str,
         deps: list[str], context: list[str], output: list[int], scope: str, iteration: int = 0) -> OperationRecord:
    request = {"model": "synthetic-conformance-model", "messages": [
        {"role": "system", "content": "MASBench synthetic golden trace."},
        {"role": "user", "content": f"Recorded downstream input for {oid}; dependencies={','.join(deps)}."}],
        "temperature": 0.0, "max_tokens": len(output)}
    ids = tokenizer.encode_request(request)
    return OperationRecord(oid, "llm", task, stage, session, iteration,
        data_dependencies=list(deps), llm={"canonical_request": request, "input_token_ids": ids, "input_token_count": len(ids),
        "tokenizer": tokenizer.identity, "chat_template": tokenizer.chat_template,
        "generation_parameters": {"temperature": 0.0, "max_tokens": len(output)},
        "recorded_output_token_ids": output, "recorded_output_token_source": "synthetic_exact", "recorded_output_length": len(output),
        "recorded_output_ref": None, "context_artifact_ids": context, "reuse_scope_id": scope})


def build_golden(template: str, destination: str | Path) -> SemanticTrace:
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template: {template}")
    root = Path(destination)
    bundle = TraceBundle(root)
    tokenizer = ByteTokenizer()
    trace_id, task_id, stage_id = f"golden:{template}", f"task:{template}", f"stage:{template}:0"
    operations: list[OperationRecord] = []
    artifacts: list[ArtifactRecord] = []
    sessions: list[AgentSession] = []

    def session(name: str, role: str, parent: str | None, relation: str, init: str, scope: str):
        sessions.append(AgentSession(name, role, parent, relation, init, scope))

    def add(oid: str, sid: str, deps: list[str], context: list[str], output: list[int], scope: str, iteration: int = 0):
        operations.append(_llm(tokenizer, oid, task_id, stage_id, sid, deps, context, output, scope, iteration))
        ref, digest = bundle.artifacts.put_json({"recorded_output_token_ids": output, "producer": oid})
        aid = f"artifact:{oid}"
        artifacts.append(ArtifactRecord(aid, oid, ref, digest, reuse_scope_id=scope))
        return aid

    if template == "spawn":
        session("session:coordinator", "Coordinator", None, "root", "fresh", "scope:spawn-root")
        session("session:worker", "Worker", "session:coordinator", "spawn", "explicit", "scope:spawn-worker")
        a = add("op:dispatch", "session:coordinator", [], [], [11, 12, 13], "scope:spawn-root")
        b = add("op:worker", "session:worker", ["op:dispatch"], [a], [21, 22, 23, 24], "scope:spawn-worker")
        participants = ["session:coordinator", "session:worker"]
    elif template == "fork_join":
        session("session:root", "Coordinator", None, "root", "fresh", "scope:fork-root")
        session("session:w1", "Worker", "session:root", "fork", "inherited", "scope:fork-shared")
        session("session:w2", "Worker", "session:root", "fork", "inherited", "scope:fork-shared")
        session("session:reducer", "Reducer", "session:root", "continuation", "explicit", "scope:join")
        a = add("op:w1", "session:w1", [], [], [31, 32], "scope:fork-shared")
        b = add("op:w2", "session:w2", [], [], [41, 42, 43], "scope:fork-shared")
        c = add("op:reduce", "session:reducer", ["op:w1", "op:w2"], [a, b], [51, 52], "scope:join")
        participants = [x.session_id for x in sessions]
    elif template == "refinement_loop":
        session("session:producer", "Worker", None, "root", "fresh", "scope:refine")
        session("session:reviewer", "Reviewer", "session:producer", "handoff", "explicit", "scope:review")
        a = add("op:draft", "session:producer", [], [], [61, 62], "scope:refine", 0)
        b = add("op:review", "session:reviewer", ["op:draft"], [a], [71, 72], "scope:review", 0)
        c = add("op:revision", "session:producer", ["op:review"], [b], [81, 82, 83], "scope:refine", 1)
        participants = [x.session_id for x in sessions]
    else:
        session("session:p1", "Worker", None, "root", "fresh", "scope:debate-p1")
        session("session:p2", "Worker", None, "root", "fresh", "scope:debate-p2")
        a = add("op:p1:r0", "session:p1", [], [], [91, 92], "scope:debate-p1", 0)
        b = add("op:p2:r0", "session:p2", [], [], [101, 102], "scope:debate-p2", 0)
        c = add("op:p1:r1", "session:p1", ["op:p2:r0"], [b], [111, 112], "scope:debate-p1", 1)
        d = add("op:p2:r1", "session:p2", ["op:p1:r0"], [a], [121, 122], "scope:debate-p2", 1)
        participants = [x.session_id for x in sessions]
    consumers = {artifact.artifact_id: [] for artifact in artifacts}
    for op in operations:
        for aid in (op.llm or {}).get("context_artifact_ids", []):
            consumers[aid].append(op.operation_id)
    for artifact in artifacts:
        artifact.consumer_operation_ids = consumers[artifact.artifact_id]
    children = {x.operation_id: set() for x in operations}
    for op in operations:
        for dep in op.dependencies:
            children[dep].add(op.operation_id)
    terminals = [x for x, value in children.items() if not value]
    stage = StageInstance(stage_id, template, template, participant_session_ids=participants,
        execution_control={"template": template, "realized": True},
        context_construction={"artifacts": [x.artifact_id for x in artifacts]},
        completion_reason={"spawn": "selected work returned", "fork_join": "required branches completed",
                           "refinement_loop": "revision limit", "debate": "fixed rounds"}[template])
    trace = SemanticTrace(TraceMetadata("golden-" + template, "1", trace_id, "masbench-synthetic", "1"),
        [TaskRecord(task_id, trace_id, 0.0, terminals, "golden-config", "completed")],
        [stage], sessions, operations, artifacts)
    bundle.write(trace)
    return trace


def golden_scenario(trace: SemanticTrace, *, mode: str = "length_locked") -> ScenarioManifest:
    first = next(x for x in trace.operations if x.llm).llm
    scopes = [x.reuse_scope_id for x in trace.sessions if x.reuse_scope_id]
    return ScenarioManifest("golden-scenario", "1", [trace.metadata.trace_id],
        {trace.metadata.trace_id: trace.metadata.workload_hash}, trace.metadata.workload_version,
        {"model_identity": "synthetic-conformance-model", "tokenizer_hash": first["tokenizer"]["hash"],
         "chat_template_hash": first["chat_template"]["hash"], "generation_parameters": {"temperature": 0.0}},
        {"fidelity_mode": mode, "tool_replay_policy": "recorded", "external_delay_policy": "deterministic",
         "cache_policy": "reuse_scope_isolated"},
        {"mode": "trace_offsets", "arrival_rate": None, "concurrency": 1, "random_seed": 17, "task_mix": {trace.metadata.trace_id: 1.0}},
        {"warmup_policy": "none", "task_count": 1, "task_slo_sec": 10.0, "slo_attainment_target": 1.0},
        {"error_handling": "invalidate", "cancellation_policy": "invalidate", "context_overflow_policy": "invalidate", "truncation_policy": "forbid"}).resolve(scopes, trace, ByteTokenizer().encode_request)


def golden_system() -> SystemConfig:
    tokenizer = ByteTokenizer()
    return SystemConfig("synthetic-conformance", "1", "none", 1,
        model_identity="synthetic-conformance-model", tokenizer_hash=tokenizer.identity["hash"],
        chat_template_hash=tokenizer.chat_template["hash"], capabilities=dict(MockExactBackend.capabilities))


from .replay import MockExactBackend  # imported late to avoid a circular type-only dependency

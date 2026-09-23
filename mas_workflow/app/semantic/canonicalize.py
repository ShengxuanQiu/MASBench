"""Adapters from native MASBench ExecutionGraph traces to Semantic Trace."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from app.execution_graph import ExecutionGraph
from app.tracing import stable_hash

from .model import (
    AgentSession, ArtifactRecord, OperationRecord, SemanticTrace, StageInstance,
    TaskRecord, TraceMetadata, sha256_json,
)
from .store import TraceBundle

TEMPLATE_ALIASES = {
    "dispatch_execute": "spawn", "DispatchExecute": "spawn", "spawn": "spawn",
    "parallel_aggregate": "fork_join", "ParallelAggregate": "fork_join", "Fork--Join": "fork_join", "fork_join": "fork_join",
    "evaluate_refine": "refinement_loop", "EvaluateRefine": "refinement_loop", "Refinement Loop": "refinement_loop", "refinement_loop": "refinement_loop",
    "peer_exchange": "debate", "PeerExchange": "debate", "PeerDeliberation": "debate", "Debate": "debate", "debate": "debate",
    "atomic": "atomic",
}


class TokenizerAdapter(Protocol):
    identity: dict[str, str]
    chat_template: dict[str, str]
    def encode_request(self, request: dict[str, Any]) -> list[int]: ...
    def encode_text(self, text: str) -> list[int]: ...


class ByteTokenizer:
    """Exact deterministic tokenizer for synthetic conformance traces only."""
    identity = {"name": "masbench-byte", "version": "1", "hash": sha256_json("masbench-byte-v1")}
    chat_template = {"name": "canonical-json", "hash": sha256_json("canonical-json-v1")}

    def encode_request(self, request: dict[str, Any]) -> list[int]:
        from .model import canonical_json
        return list(canonical_json(request).encode("utf-8"))

    def encode_text(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))


class TransformersTokenizer:
    def __init__(self, path: str, revision: str = "local"):
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        backend = getattr(self._tokenizer, "backend_tokenizer", None)
        serialized = backend.to_str() if backend is not None else self._tokenizer.get_vocab()
        self.identity = {"name": str(getattr(self._tokenizer, "name_or_path", path)), "version": revision,
                         "hash": sha256_json({"tokenizer": serialized,
                                              "special_tokens": self._tokenizer.special_tokens_map,
                                              "vocab_size": len(self._tokenizer)})}
        template = getattr(self._tokenizer, "chat_template", None) or ""
        self.chat_template = {"name": "tokenizer.chat_template", "hash": sha256_json(template)}

    def encode_request(self, request: dict[str, Any]) -> list[int]:
        messages = request.get("messages", [])
        template_kwargs = request.get("chat_template_kwargs") or {}
        if not isinstance(template_kwargs, dict):
            raise TypeError("chat_template_kwargs must be an object")
        encoded = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **template_kwargs)
        # Transformers 5 may return BatchEncoding instead of a flat list.
        # Iterating that object yields field names ("input_ids", ...), which
        # silently corrupts the benchmark input contract.
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], (list, tuple)):
            if len(encoded) != 1:
                raise ValueError("Tokenizer returned a batched input for one request")
            encoded = encoded[0]
        if not isinstance(encoded, (list, tuple)) or any(type(token) is not int for token in encoded):
            raise TypeError("Tokenizer must return a flat integer input_ids sequence")
        return list(encoded)

    def encode_text(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))


def _template_type(stage: dict[str, Any]) -> str:
    semantics = stage.get("semantics", {})
    family = semantics.get("canonical_family") or semantics.get("family") or semantics.get("motif") or stage.get("motif_instance_id") and "extension" or "atomic"
    return TEMPLATE_ALIASES.get(family, "extension")


def _workload_configuration(events: list[dict[str, Any]]) -> dict[str, Any]:
    native = next((e.get("extra", {}).get("config", {}) for e in events if e.get("canonical_type") == "run_start"), {})
    experiment = (native.get("extra") or {}).get("experiment") if isinstance(native, dict) else None
    if experiment:
        deployment = experiment.get("deployment", {})
        return {"structure": experiment.get("structure"), "task": experiment.get("task"),
                "model_input": {"model": deployment.get("model"), "generation": deployment.get("generation")}}
    excluded = {"llm_mode", "backend_base_url", "max_concurrent_llm_calls", "trace_dir",
                "collect_backend_metrics", "backend_metrics_url", "telemetry", "hardware"}
    return {key: value for key, value in native.items() if key not in excluded}


def canonicalize_native_trace(source: str | Path, destination: str | Path, *, tokenizer: TokenizerAdapter,
                              workload_id: str, workload_version: str = "1", source_framework_version: str = "unknown") -> SemanticTrace:
    source = Path(source)
    events = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    graph = ExecutionGraph.from_events(events, replay=True)
    observations: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        oid = event.get("operation_id") or event.get("node_id")
        if not oid or oid not in graph.operations:
            continue
        serving = (event.get("attributes") or {}).get("serving_observation", {})
        observations.setdefault(oid, []).append({
            "event_type": event.get("canonical_type", event.get("event_type")),
            "absolute_timestamp": event.get("timestamp"), "relative_time_sec": event.get("relative_time_sec"),
            "serving_observation": serving})
    bundle = TraceBundle(destination)
    task_id = f"task:{graph.run_id}"
    stage_records = []
    logical_to_instance = {stage["logical_stage_id"]: sid for sid, stage in graph.stages.items()}
    for sid, stage in graph.stages.items():
        stage_records.append(StageInstance(
            stage_instance_id=sid, logical_stage_id=stage["logical_stage_id"], template_type=_template_type(stage),
            participant_session_ids=sorted({op.identities.get("agent_instance_id") for op in graph.operations.values() if op.identities.get("stage_instance_id") == sid and op.identities.get("agent_instance_id")}),
            stage_dependencies=[logical_to_instance[x] for x in stage.get("dependencies", [])],
            execution_control={"coordination": stage.get("semantics", {}).get("coordination"),
                               "completion": stage.get("semantics", {}).get("completion")},
            context_construction={"connectivity": stage.get("semantics", {}).get("connectivity"),
                                  "delivery": stage.get("semantics", {}).get("delivery", {}),
                                  **stage.get("semantics", {}).get("context_construction", {})},
            completion_reason=stage.get("completion_reason", "")))
    agent_ids = sorted({op.identities.get("agent_instance_id") for op in graph.operations.values() if op.identities.get("agent_instance_id")})
    sessions = [AgentSession(aid,
        next((op.identities.get("role") or "Worker" for op in graph.operations.values() if op.identities.get("agent_instance_id") == aid), "Worker"),
        None, "root", "explicit",
        next((op.metadata.get("reuse_scope_id") for op in graph.operations.values()
              if op.identities.get("agent_instance_id") == aid and op.metadata.get("reuse_scope_id")), None))
        for aid in agent_ids]
    if not sessions:
        sessions = [AgentSession(f"session:{graph.run_id}", "Worker", None, "root", "explicit", None)]
    edges = {(src, dst): kind for src, dst, kind in graph.edges}
    artifact_records = []
    for aid, artifact in graph.artifacts.items():
        ref, digest = bundle.artifacts.put_json(artifact["content"])
        consumers = [consumer for artifact_id, consumer in graph.consumptions if artifact_id == aid]
        delivery = next((x for x in graph.deliveries if x.get("artifact_id") == aid or x.get("delivered_artifact_id") == aid), {})
        producer_scope = graph.operations[artifact["producer"]].metadata.get("reuse_scope_id") if artifact["producer"] in graph.operations else None
        artifact_records.append(ArtifactRecord(aid, artifact["producer"], ref, digest,
            selection_metadata=delivery.get("delivery_selector", {}), transformation_metadata=delivery.get("delivery_transform", {}),
            inheritance_relation=delivery.get("delivery_mode", "fresh"), reuse_scope_id=delivery.get("reuse_scope_id") or producer_scope, consumer_operation_ids=consumers))
    operations = []
    for oid, op in graph.operations.items():
        control = sorted(src for src in op.parents if edges.get((src, oid), "control") == "control")
        data = sorted(src for src in op.parents if edges.get((src, oid)) == "data")
        common = dict(operation_id=oid, operation_type="transform" if op.kind in {"router", "retrieval"} else op.kind,
                      task_id=task_id, stage_instance_id=op.identities.get("stage_instance_id") or stage_records[0].stage_instance_id,
                      session_id=op.identities.get("agent_instance_id") or sessions[0].session_id,
                      control_dependencies=control, data_dependencies=data,
                      source_observations={"duration_sec": op.duration_sec, "duration_kind": "source_observation",
                                           "events": observations.get(oid, [])})
        if op.kind == "llm":
            request = dict(op.payload or {})
            token_ids = tokenizer.encode_request(request)
            context_ids = [aid for aid, consumer in graph.consumptions if consumer == oid]
            produced = next(((aid, artifact) for aid, artifact in graph.artifacts.items() if artifact["producer"] == oid), None)
            output_text = str(produced[1]["content"]) if produced and produced[1].get("content") is not None else None
            retokenized_output = tokenizer.encode_text(output_text) if output_text is not None else None
            recorded_length = op.output_tokens if isinstance(op.output_tokens, int) else (len(retokenized_output) if retokenized_output is not None else None)
            output_ref = next((x.payload_ref for x in artifact_records if produced and x.artifact_id == produced[0]), None)
            payload = {"canonical_request": request, "input_token_ids": token_ids, "input_token_count": len(token_ids),
                       "tokenizer": tokenizer.identity, "chat_template": tokenizer.chat_template,
                       "generation_parameters": {k: request[k] for k in request if k not in {"messages", "model"}},
                       "recorded_output_token_ids": retokenized_output, "recorded_output_token_source": "retokenized_output_text" if retokenized_output is not None else "unavailable",
                       "recorded_output_length": recorded_length,
                       "recorded_output_ref": output_ref, "context_artifact_ids": context_ids,
                       "reuse_scope_id": op.metadata.get("reuse_scope_id")}
            operations.append(OperationRecord(**common, llm=payload))
        else:
            ref, digest = bundle.artifacts.put_json(op.snapshot)
            typed = {"identity": op.metadata.get("tool_name", op.kind), "input": op.metadata.get("tool_input"),
                     "recorded_result_ref": ref, "content_hash": digest, "deterministic_external_delay": op.duration_sec}
            # Source tool latency is an observation, never portable workload service time.
            controlled_delay = float(op.metadata.get("controlled_external_delay", 0.0))
            typed["deterministic_external_delay"] = controlled_delay
            common["controlled_external_delay"] = controlled_delay
            if op.kind == "tool": operations.append(OperationRecord(**common, tool=typed))
            else: operations.append(OperationRecord(**common, transform=typed))
    children = {oid: set() for oid in graph.operations}
    for op in operations:
        for parent in op.dependencies:
            children[parent].add(op.operation_id)
    terminals = sorted(oid for oid, descendants in children.items() if not descendants)
    trace_id = graph.run_id
    trace = SemanticTrace(
        TraceMetadata(workload_id, workload_version, trace_id, "masbench-native", source_framework_version,
                      configuration_hash=stable_hash(_workload_configuration(events)),
                      source_observations={"native_trace_sha256": stable_hash(source.read_bytes())}),
        [TaskRecord(task_id, trace_id, 0.0, terminals, "native-run-config", "completed")],
        stage_records, sessions, operations, artifact_records)
    bundle.write(trace)
    return trace

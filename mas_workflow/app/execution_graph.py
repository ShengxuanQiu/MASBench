"""Canonical trace schema and realized LLM/tool DAG, independent of deployment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA = "masbench_execution_v1"
EVENT_TYPES = {"operation_start", "operation_finish", "operation_fail", "dependency",
               "artifact_produce", "artifact_consume", "control_decision", "barrier_sync",
               "request_ready", "request_submit", "request_finish", "request_fail",
               "backend_measurement", "artifact_delivery", "stage_skip", "stage_start", "stage_finish", "run_start", "run_finish"}
ALIASES = {"llm_request_ready": "request_ready", "llm_request_start": "request_submit",
           "llm_request_end": "request_finish", "llm_request_error": "request_fail",
           "artifact_created": "artifact_produce", "barrier": "barrier_sync",
           "workflow_start": "run_start", "workflow_end": "run_finish"}


def normalize_event(event):
    """Keep flat compatibility fields; exporters consume this same canonical stream."""
    kind = ALIASES.get(event["event_type"], event["event_type"])
    if event["event_type"].endswith("metric_sample") or event["event_type"] == "cache_memory_sample":
        kind = "backend_measurement"
    if kind not in EVENT_TYPES:
        return
    event["canonical_schema"] = SCHEMA
    event["canonical_type"] = kind
    event["operation_id"] = event.get("operation_id") or (event.get("node_id", "") if kind.startswith(("request_", "operation_", "artifact_")) else "")
    event["request_id"] = event.get("request_id") or event.get("llm_request_id", "")
    for key in ("stage_instance_id", "motif_instance_id", "atomic_instance_id", "agent_instance_id", "artifact_id"):
        event.setdefault(key, "")
    event["operation_phase"] = {"request_ready": "ready", "request_submit": "started",
                                 "request_finish": "finished", "request_fail": "failed",
                                 "operation_start": "started", "operation_finish": "finished",
                                 "operation_fail": "failed"}.get(kind, "")
    operation_keys = {"operation_id", "request_id", "stage_instance_id", "motif_instance_id", "atomic_instance_id", "agent_instance_id",
                      "role", "operation_kind", "operation_phase", "request_payload", "request_metadata", "status", "error"}
    flow_keys = {"parents", "artifact_id", "source_artifact_id", "content", "producer_operation_id",
                 "src", "dst", "dependency_kind", "decision", "waiting_for_nodes", "tool_snapshot", "delivery_mode", "source_artifact_ids",
                 "delivered_artifact_id","consumer_operation_id","delivery_selector","delivery_transform","materialized_bytes","materialized_tokens_est"}
    measurements = {}
    for key in ("duration_sec", "request_ready_ts", "request_submit_ts", "response_end_ts", "ttft_sec", "tpot_sec",
                "backend_prompt_tokens", "backend_completion_tokens", "queue_wait_sec", "input_tokens_est", "output_tokens_est"):
        if key not in event:
            continue
        value = event[key]
        source = "observed"
        if value is None or value == "unavailable":
            source = "unavailable"
        elif key.endswith("_est") or event.get("duration_source") == "simulated_duration":
            source = "estimated"
        elif key.startswith("backend_"):
            source = "backend-reported"
        elif key == "queue_wait_sec":
            # Existing adapters return a placeholder zero, not measured server queuing.
            value, source = None, "unavailable"
        measurements[key] = {"value": value, "source": source}
    if kind == "backend_measurement":
        for key, value in event.items():
            if key in measurements or key in {"relative_time_sec", "random_seed", "attempt_id", "retry_count", "duration_sec"}:
                continue
            if type(value) in {int, float} or value is None:
                source = "unavailable" if value is None else ("estimated" if key.startswith("kv_cache_used_") else "backend-reported")
                measurements[key] = {"value": value, "source": source}
    event["attributes"] = {
        "operation": {k: event[k] for k in operation_keys if k in event},
        "information_flow": {k: event[k] for k in flow_keys if k in event},
        "serving_observation": measurements,
    }


@dataclass
class Operation:
    id: str
    kind: str
    parents: set[str] = field(default_factory=set)
    request_id: str = ""
    payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    snapshot: Any = None
    duration_sec: float = 0.0
    identities: dict[str, str] = field(default_factory=dict)
    output_tokens: int | None = None


@dataclass
class ExecutionGraph:
    run_id: str = ""
    operations: dict[str, Operation] = field(default_factory=dict)
    edges: set[tuple[str, str, str]] = field(default_factory=set)
    artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    consumptions: list[tuple[str, str]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    deliveries: list[dict[str, Any]] = field(default_factory=list)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)

    def ingest(self, event):
        if event.get("canonical_schema") != SCHEMA:
            return
        if self.run_id and event["run_id"] != self.run_id:
            raise ValueError("A graph cannot mix run IDs")
        self.run_id = event["run_id"]
        kind = event["canonical_type"]
        oid = event.get("operation_id", "")
        if kind in {"operation_start", "request_ready"}:
            if oid in self.operations:
                raise ValueError(f"Duplicate operation: {oid}")
            self.operations[oid] = Operation(oid, "llm" if kind == "request_ready" else event["operation_kind"],
                                             identities={k: event.get(k, "") for k in ("stage_instance_id", "motif_instance_id", "atomic_instance_id", "agent_instance_id", "role")})
        if kind == "stage_start":
            sid=event["stage_instance_id"]
            self.stages[sid]={"logical_stage_id":event.get("stage_id",""),"stage_instance_id":sid,
                "motif_instance_id":event.get("motif_instance_id",""),"atomic_instance_id":event.get("atomic_instance_id",""),
                "dependencies":event.get("stage_dependencies",[]),"semantics":event.get("stage_semantics",{}),
                "participants":event.get("realized_participants",[]),"activation":event.get("activation","active"),
                "status":"running","completion_reason":"","decisions":[]}
            self.stages[sid]["decisions"]=[e.get("decision",{}) for e in self.decisions
                if e.get("decision",{}).get("stage_id")==event.get("stage_id")]
        if kind == "stage_skip":
            sid=event["stage_instance_id"]
            self.stages[sid]={"logical_stage_id":event.get("stage_id",""),"stage_instance_id":sid,
                "motif_instance_id":"","atomic_instance_id":"","dependencies":event.get("stage_dependencies",[]),
                "semantics":event.get("stage_semantics",{}),"participants":[],"activation":"skipped",
                "status":"skipped","completion_reason":event.get("completion_reason",event.get("reason","")),"decisions":[]}
            self.stages[sid]["decisions"]=[e.get("decision",{}) for e in self.decisions
                if e.get("decision",{}).get("stage_id")==event.get("stage_id")]
        if kind == "stage_finish":
            sid=event["stage_instance_id"]
            if sid not in self.stages: raise ValueError("stage_finish without stage_start")
            self.stages[sid].update(status=event.get("status","completed"),
                                    completion_reason=event.get("completion_reason",""),
                                    participants=event.get("realized_participants",self.stages[sid]["participants"]),
                                    semantics=event.get("stage_semantics",self.stages[sid]["semantics"]))
        if kind in {"operation_start", "request_ready", "request_submit"}:
            op = self.operations[oid]
            if kind == "request_ready":
                op.request_id = event["request_id"]
            op.parents.update(event.get("parents", []))
        if kind == "request_submit":
            op = self.operations[oid]
            op.payload = event.get("request_payload")
            op.request_id = event["request_id"]
            op.metadata = event.get("request_metadata", {})
        if kind in {"operation_finish", "operation_fail", "request_finish", "request_fail"}:
            op = self.operations[oid]
            if op.status != "running":
                raise ValueError(f"Duplicate terminal event: {oid}")
            op.status = "failed" if kind.endswith("fail") else "completed"
            op.snapshot = event.get("tool_snapshot")
            op.duration_sec = event.get("duration_sec", 0.0)
            op.output_tokens = event.get("backend_completion_tokens")
        if kind == "dependency":
            self.edges.add((event["src"], event["dst"], event["dependency_kind"]))
        if kind == "artifact_produce":
            aid = event["artifact_id"]
            if aid in self.artifacts:
                raise ValueError(f"Duplicate artifact: {aid}")
            self.artifacts[aid] = {"producer": oid, "content": event.get("content"), "hash": event.get("output_hash")}
        if kind == "artifact_consume":
            self.consumptions.append((event["artifact_id"], oid))
        if kind == "artifact_delivery":
            if event.get("materialized_bytes") is None and event.get("artifact_id") in self.artifacts:
                from .tracing import estimate_tokens
                content=self.artifacts[event["artifact_id"]]["content"]
                event["materialized_bytes"]=len(str(content).encode('utf-8'))
                event["materialized_tokens_est"]=estimate_tokens(str(content))
                event["delivered_artifact_id"]=event.get("delivered_artifact_id") or event["artifact_id"]
                event["consumer_operation_id"]=event.get("consumer_operation_id") or oid
                event.setdefault("delivery_selector",{})
                event.setdefault("delivery_transform",{})
            self.deliveries.append(event)
        if kind == "control_decision":
            self.decisions.append(event)
            target=event.get("decision",{}).get("stage_id")
            for stage in self.stages.values():
                if (target and stage["logical_stage_id"]==target) or (not target and stage["stage_instance_id"]==event.get("stage_instance_id")):
                    stage["decisions"].append(event.get("decision",{}))

    def validate(self, *, replay=False):
        if not self.operations:
            raise ValueError("Trace contains no canonical operations; legacy traces need a fresh recording")
        parents = {oid: set(op.parents) for oid, op in self.operations.items()}
        # A scheduling parent is not automatically an additional control edge.
        # Preserve explicit data/control labels; classify only untyped legacy
        # parents as control dependencies after the full event stream is read.
        typed_pairs = {(a, b) for a, b, _ in self.edges}
        for oid, ps in parents.items():
            for parent in ps:
                if (parent, oid) not in typed_pairs:
                    self.edges.add((parent, oid, "control"))
        for src, dst, kind in self.edges:
            if src not in parents or dst not in parents or kind not in {"data", "control"}:
                raise ValueError(f"Dangling/invalid dependency: {src} -> {dst}")
            parents[dst].add(src)
        for aid, artifact in self.artifacts.items():
            if artifact["producer"] not in parents:
                raise ValueError(f"Unknown producer for {aid}")
            if replay:
                from .tracing import stable_hash
                if artifact["content"] is None or stable_hash(artifact["content"]) != artifact["hash"]:
                    raise ValueError(f"Artifact snapshot/hash mismatch: {aid}")
        for aid, oid in self.consumptions:
            if aid not in self.artifacts or oid not in parents:
                raise ValueError("Dangling artifact consumption")
            producer = self.artifacts[aid]["producer"]
            if (producer, oid, "data") not in self.edges:
                raise ValueError("Artifact consumption missing data dependency")
        for delivery in self.deliveries:
            if delivery["artifact_id"] not in self.artifacts or delivery["operation_id"] not in parents:
                raise ValueError("Dangling delivery provenance")
            if delivery.get("delivery_mode") not in {"full","selected","summarized","referenced","retrieved"}:
                raise ValueError("Invalid delivery mode")
            if any(a not in self.artifacts for a in delivery.get("source_artifact_ids",[])):
                raise ValueError("Unknown source artifact")
            if delivery.get("delivered_artifact_id") and delivery["delivered_artifact_id"] not in self.artifacts:
                raise ValueError("Unknown delivered artifact")
            if delivery.get("materialized_bytes") is None:
                raise ValueError("Delivery lacks materialized size")
        done = set()
        while len(done) < len(parents):
            ready = {oid for oid in parents if oid not in done and parents[oid] <= done}
            if not ready:
                raise ValueError("Execution graph contains a cycle or dangling parent")
            done.update(ready)
        logical={stage["logical_stage_id"] for stage in self.stages.values()}
        for stage in self.stages.values():
            if set(stage["dependencies"])-logical:
                raise ValueError("Stage hierarchy has a dangling dependency")
            if stage["status"] != "running" and not stage["completion_reason"]:
                raise ValueError("Completed stage lacks completion reason")
        if self.stages and any(op.identities.get("stage_instance_id") not in self.stages for op in self.operations.values()):
            raise ValueError("Operation lacks realized stage mapping")
        request_ids = [op.request_id for op in self.operations.values() if op.kind == "llm"]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Duplicate request IDs")
        for oid, op in self.operations.items():
            op.parents = parents[oid]
            if op.kind not in {"llm", "tool", "transform", "router", "retrieval"}:
                raise ValueError(f"Unsupported operation kind: {op.kind}")
            if replay and (op.status != "completed" or (op.kind == "llm" and not op.payload) or (op.kind != "llm" and op.snapshot is None)):
                raise ValueError(f"Incomplete operation cannot be replayed: {oid}")
            if replay and op.kind == "llm":
                messages = op.payload.get("messages", [])
                if (len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]
                        or any(not isinstance(m.get("content"), str) for m in messages)
                        or type(op.payload.get("max_tokens")) is not int or op.payload["max_tokens"] < 1):
                    raise ValueError("Unsupported or incomplete recorded request payload")
        return self

    @classmethod
    def from_events(cls, events, *, replay=False):
        graph = cls()
        for event in events:
            graph.ingest(event)
        return graph.validate(replay=replay)

    def to_dict(self):
        from dataclasses import asdict
        hierarchy={oid:op.identities.get("stage_instance_id","") for oid,op in self.operations.items()}
        return {"run_id": self.run_id, "operations": {oid: {**asdict(op), "parents": sorted(op.parents)} for oid, op in self.operations.items()},
                "edges": sorted(self.edges), "artifacts": self.artifacts, "consumptions": self.consumptions,
                "deliveries": self.deliveries,"decisions":self.decisions,"stages":self.stages,
                "hierarchy":{"operation_to_stage":hierarchy}}

"""Versioned, deployment-independent MASBench Semantic Trace model."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "masbench.semantic_trace/1.0.0"
TEMPLATE_TYPES = {"spawn", "fork_join", "refinement_loop", "debate", "atomic", "extension"}
OPERATION_TYPES = {"llm", "tool", "transform", "state"}
LIFECYCLE_RELATIONS = {"root", "spawn", "fork", "handoff", "continuation"}
CONTEXT_INITIALIZATIONS = {"fresh", "inherited", "explicit"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TraceMetadata:
    workload_id: str
    workload_version: str
    trace_id: str
    source_framework: str
    source_framework_version: str
    created_at: str = field(default_factory=utc_now)
    schema_version: str = SCHEMA_VERSION
    workload_hash: str = ""
    configuration_hash: str = ""
    content_hash: str = ""
    source_observations: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRecord:
    task_id: str
    trace_id: str
    root_arrival_offset: float
    terminal_operation_ids: list[str]
    workload_configuration_ref: str
    completion_status: str = "completed"


@dataclass
class StageInstance:
    stage_instance_id: str
    logical_stage_id: str
    template_type: str
    parent_stage_instance_id: str | None = None
    iteration: int = 0
    participant_session_ids: list[str] = field(default_factory=list)
    stage_dependencies: list[str] = field(default_factory=list)
    execution_control: dict[str, Any] = field(default_factory=dict)
    context_construction: dict[str, Any] = field(default_factory=dict)
    completion_reason: str = ""


@dataclass
class AgentSession:
    session_id: str
    role_id: str
    parent_session_id: str | None
    lifecycle_relation: str
    context_initialization: str
    reuse_scope_id: str | None = None


@dataclass
class OperationRecord:
    operation_id: str
    operation_type: str
    task_id: str
    stage_instance_id: str
    session_id: str | None
    iteration: int = 0
    control_dependencies: list[str] = field(default_factory=list)
    data_dependencies: list[str] = field(default_factory=list)
    state_dependencies: list[str] = field(default_factory=list)
    controlled_external_delay: float = 0.0
    llm: dict[str, Any] | None = None
    tool: dict[str, Any] | None = None
    transform: dict[str, Any] | None = None
    state: dict[str, Any] | None = None
    source_observations: dict[str, Any] = field(default_factory=dict)

    @property
    def dependencies(self) -> set[str]:
        return set(self.control_dependencies + self.data_dependencies + self.state_dependencies)


@dataclass
class ArtifactRecord:
    artifact_id: str
    producer_operation_id: str
    payload_ref: str
    content_hash: str
    version: int = 1
    selection_metadata: dict[str, Any] = field(default_factory=dict)
    ordering_metadata: dict[str, Any] = field(default_factory=dict)
    transformation_metadata: dict[str, Any] = field(default_factory=dict)
    inheritance_relation: str = "fresh"
    reuse_scope_id: str | None = None
    consumer_operation_ids: list[str] = field(default_factory=list)


@dataclass
class StateRecord:
    state_object_id: str
    version: int
    operation: str
    producer_operation_id: str | None
    consumer_operation_ids: list[str] = field(default_factory=list)


@dataclass
class SemanticTrace:
    metadata: TraceMetadata
    tasks: list[TaskRecord]
    stages: list[StageInstance]
    sessions: list[AgentSession]
    operations: list[OperationRecord]
    artifacts: list[ArtifactRecord]
    shared_state: list[StateRecord] = field(default_factory=list)

    def workload_projection(self) -> dict[str, Any]:
        """Return identity fields only. Source observations are deliberately absent."""
        metadata = {"schema_version": self.metadata.schema_version,
                    "workload_id": self.metadata.workload_id,
                    "workload_version": self.metadata.workload_version,
                    "configuration_hash": self.metadata.configuration_hash}
        operations = []
        for op in self.operations:
            item = asdict(op)
            item.pop("source_observations", None)
            operations.append(item)
        return {
            "metadata": metadata,
            "tasks": [{**asdict(x), "trace_id": ""} for x in self.tasks],
            "stages": [asdict(x) for x in self.stages],
            "sessions": [asdict(x) for x in self.sessions],
            "operations": operations,
            "artifacts": [asdict(x) for x in self.artifacts],
            "shared_state": [asdict(x) for x in self.shared_state],
        }

    def compute_workload_hash(self) -> str:
        return sha256_json(self.workload_projection())

    def finalize_hashes(self) -> "SemanticTrace":
        self.metadata.workload_hash = self.compute_workload_hash()
        self.metadata.content_hash = self.compute_content_hash()
        return self

    def compute_content_hash(self) -> str:
        return sha256_json({"manifest": self.to_manifest(include_content_hash=False),
                            "operations": [asdict(x) for x in self.operations]})

    def to_manifest(self, *, include_content_hash: bool = True) -> dict[str, Any]:
        metadata = asdict(self.metadata)
        if not include_content_hash:
            metadata["content_hash"] = ""
        return {
            "schema_version": SCHEMA_VERSION,
            "metadata": metadata,
            "tasks": [asdict(x) for x in self.tasks],
            "stages": [asdict(x) for x in self.stages],
            "sessions": [asdict(x) for x in self.sessions],
            "artifacts": [asdict(x) for x in self.artifacts],
            "shared_state": [asdict(x) for x in self.shared_state],
            "operations_file": "operations.jsonl",
        }

    @classmethod
    def from_parts(cls, manifest: dict[str, Any], operations: list[dict[str, Any]]) -> "SemanticTrace":
        return cls(
            TraceMetadata(**manifest["metadata"]),
            [TaskRecord(**x) for x in manifest.get("tasks", [])],
            [StageInstance(**x) for x in manifest.get("stages", [])],
            [AgentSession(**x) for x in manifest.get("sessions", [])],
            [OperationRecord(**x) for x in operations],
            [ArtifactRecord(**x) for x in manifest.get("artifacts", [])],
            [StateRecord(**x) for x in manifest.get("shared_state", [])],
        )

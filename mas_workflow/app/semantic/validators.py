"""Four validation layers used by official MASBench runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .model import (
    CONTEXT_INITIALIZATIONS, LIFECYCLE_RELATIONS, OPERATION_TYPES, SCHEMA_VERSION,
    TEMPLATE_TYPES, SemanticTrace, sha256_json,
)

REASON_CODES = {
    "SCHEMA_ERROR", "TRACE_HASH_MISMATCH", "TOKENIZER_MISMATCH",
    "CHAT_TEMPLATE_MISMATCH", "MISSING_DEPENDENCY", "INVALID_HIERARCHY",
    "INVALID_CONTEXT_REFERENCE", "CACHE_SCOPE_VIOLATION",
    "REPLAY_MODE_UNSUPPORTED", "OUTPUT_LENGTH_MISMATCH", "CONTEXT_OVERFLOW",
    "TRUNCATED_INPUT", "RUN_CANCELLED", "INCOMPLETE_TASK", "OUTPUT_TOKEN_MISMATCH",
}


@dataclass
class ValidationReport:
    run_valid: bool = True
    invalid_reasons: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)

    def add(self, code: str, message: str, **context: Any) -> None:
        if code not in REASON_CODES:
            raise ValueError(f"Unknown reason code: {code}")
        self.run_valid = False
        if code not in self.invalid_reasons:
            self.invalid_reasons.append(code)
        self.details.append({"code": code, "message": message, **context})

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        for detail in other.details:
            self.add(detail["code"], detail["message"], **{k: v for k, v in detail.items() if k not in {"code", "message"}})
        return self

    def to_dict(self) -> dict[str, Any]:
        return {"run_valid": self.run_valid, "invalid_reasons": self.invalid_reasons, "validation_details": self.details}


class SchemaValidator:
    def validate(self, trace: SemanticTrace) -> ValidationReport:
        report = ValidationReport()
        if trace.metadata.schema_version != SCHEMA_VERSION:
            report.add("SCHEMA_ERROR", "Unsupported Semantic Trace schema version")
        groups = {
            "task": [x.task_id for x in trace.tasks],
            "stage": [x.stage_instance_id for x in trace.stages],
            "session": [x.session_id for x in trace.sessions],
            "operation": [x.operation_id for x in trace.operations],
            "artifact": [x.artifact_id for x in trace.artifacts],
        }
        for kind, identifiers in groups.items():
            if any(not isinstance(x, str) or not x for x in identifiers) or len(identifiers) != len(set(identifiers)):
                report.add("SCHEMA_ERROR", f"{kind} IDs must be nonempty and unique")
        if not trace.tasks or not trace.operations:
            report.add("SCHEMA_ERROR", "Trace requires at least one task and operation")
        for stage in trace.stages:
            if stage.template_type not in TEMPLATE_TYPES or stage.iteration < 0:
                report.add("SCHEMA_ERROR", "Invalid stage template or iteration", stage_instance_id=stage.stage_instance_id)
        for session in trace.sessions:
            if session.lifecycle_relation not in LIFECYCLE_RELATIONS or session.context_initialization not in CONTEXT_INITIALIZATIONS:
                report.add("SCHEMA_ERROR", "Invalid session lifecycle/context initialization", session_id=session.session_id)
        for op in trace.operations:
            if op.operation_type not in OPERATION_TYPES or op.controlled_external_delay < 0:
                report.add("SCHEMA_ERROR", "Invalid operation type or external delay", operation_id=op.operation_id)
            typed = {"llm": op.llm, "tool": op.tool, "transform": op.transform, "state": op.state}
            if typed[op.operation_type] is None or sum(value is not None for value in typed.values()) != 1:
                report.add("SCHEMA_ERROR", "Operation requires exactly one matching typed payload", operation_id=op.operation_id)
        return report


class SemanticValidator:
    def validate(self, trace: SemanticTrace) -> ValidationReport:
        report = SchemaValidator().validate(trace)
        opmap = {x.operation_id: x for x in trace.operations}
        tasks = {x.task_id: x for x in trace.tasks}
        stages = {x.stage_instance_id: x for x in trace.stages}
        sessions = {x.session_id: x for x in trace.sessions}
        artifacts = {x.artifact_id: x for x in trace.artifacts}
        for stage in trace.stages:
            if stage.parent_stage_instance_id and stage.parent_stage_instance_id not in stages:
                report.add("INVALID_HIERARCHY", "Unknown parent stage", stage_instance_id=stage.stage_instance_id)
            if any(x not in stages for x in stage.stage_dependencies):
                report.add("MISSING_DEPENDENCY", "Unknown stage dependency", stage_instance_id=stage.stage_instance_id)
            if any(x not in sessions for x in stage.participant_session_ids):
                report.add("INVALID_HIERARCHY", "Unknown stage participant session", stage_instance_id=stage.stage_instance_id)
        for session in trace.sessions:
            if session.parent_session_id and session.parent_session_id not in sessions:
                report.add("INVALID_HIERARCHY", "Unknown parent session", session_id=session.session_id)
            if session.lifecycle_relation == "root" and session.parent_session_id:
                report.add("INVALID_HIERARCHY", "Root session cannot have a parent", session_id=session.session_id)
        parents: dict[str, set[str]] = {}
        for op in trace.operations:
            if op.task_id not in tasks or op.stage_instance_id not in stages or (op.session_id and op.session_id not in sessions):
                report.add("INVALID_HIERARCHY", "Operation has invalid task/stage/session mapping", operation_id=op.operation_id)
            parents[op.operation_id] = op.dependencies
            missing = op.dependencies - set(opmap)
            if missing:
                report.add("MISSING_DEPENDENCY", "Operation references missing predecessor", operation_id=op.operation_id, dependencies=sorted(missing))
            if op.llm:
                required = {"canonical_request", "input_token_ids", "input_token_count", "tokenizer", "chat_template", "generation_parameters", "recorded_output_length", "recorded_output_token_source", "context_artifact_ids", "reuse_scope_id"}
                if required - set(op.llm):
                    report.add("SCHEMA_ERROR", "LLM payload lacks replay fields", operation_id=op.operation_id, missing=sorted(required - set(op.llm)))
                if op.llm.get("input_token_count") != len(op.llm.get("input_token_ids", [])):
                    report.add("TOKENIZER_MISMATCH", "input_token_count does not match exact token IDs", operation_id=op.operation_id)
                if not isinstance(op.llm.get("recorded_output_length"), int) or op.llm.get("recorded_output_length", -1) < 0:
                    report.add("SCHEMA_ERROR", "LLM operation requires an exact recorded output length", operation_id=op.operation_id)
                for aid in op.llm.get("context_artifact_ids", []):
                    if aid not in artifacts or op.operation_id not in artifacts[aid].consumer_operation_ids:
                        report.add("INVALID_CONTEXT_REFERENCE", "Context artifact provenance is incomplete", operation_id=op.operation_id, artifact_id=aid)
        done: set[str] = set()
        while len(done) < len(parents):
            ready = {oid for oid, deps in parents.items() if oid not in done and deps <= done}
            if not ready:
                report.add("MISSING_DEPENDENCY", "Realized operation graph is cyclic or dangling")
                break
            done.update(ready)
        for artifact in trace.artifacts:
            if artifact.producer_operation_id not in opmap:
                report.add("INVALID_CONTEXT_REFERENCE", "Artifact producer does not exist", artifact_id=artifact.artifact_id)
            if any(x not in opmap for x in artifact.consumer_operation_ids):
                report.add("INVALID_CONTEXT_REFERENCE", "Artifact consumer does not exist", artifact_id=artifact.artifact_id)
        state_versions: dict[str, list[int]] = {}
        for state in trace.shared_state:
            if state.operation not in {"read", "write"} or state.version < 0:
                report.add("SCHEMA_ERROR", "Invalid shared-state operation/version", state_object_id=state.state_object_id)
            if state.producer_operation_id and state.producer_operation_id not in opmap:
                report.add("MISSING_DEPENDENCY", "Shared-state producer does not exist", state_object_id=state.state_object_id)
            for consumer in state.consumer_operation_ids:
                if consumer not in opmap or (state.producer_operation_id and state.producer_operation_id not in opmap[consumer].state_dependencies):
                    report.add("MISSING_DEPENDENCY", "Shared-state read lacks a state dependency", state_object_id=state.state_object_id, consumer_operation_id=consumer)
            state_versions.setdefault(state.state_object_id, []).append(state.version)
        for state_id, versions in state_versions.items():
            if len(versions) != len(set(versions)) or sorted(versions) != list(range(min(versions), max(versions) + 1)):
                report.add("INVALID_CONTEXT_REFERENCE", "Shared-state versions are duplicate or non-contiguous", state_object_id=state_id)
        for task in trace.tasks:
            if task.trace_id != trace.metadata.trace_id or any(x not in opmap for x in task.terminal_operation_ids):
                report.add("INVALID_HIERARCHY", "Task trace/terminal mapping is invalid", task_id=task.task_id)
        expected = trace.compute_workload_hash()
        if trace.metadata.workload_hash and trace.metadata.workload_hash != expected:
            report.add("TRACE_HASH_MISMATCH", "Workload identity hash mismatch")
        expected_content = trace.compute_content_hash()
        if trace.metadata.content_hash and trace.metadata.content_hash != expected_content:
            report.add("TRACE_HASH_MISMATCH", "Trace content hash mismatch")
        return report


class RunValidator:
    def validate(self, trace: SemanticTrace, scenario: Any, system: Any, run: dict[str, Any] | None = None) -> ValidationReport:
        report = SemanticValidator().validate(trace)
        if scenario.compute_hash() != scenario.scenario_hash:
            report.add("TRACE_HASH_MISMATCH", "Resolved scenario hash mismatch")
        if scenario.workload_version != trace.metadata.workload_version:
            report.add("TRACE_HASH_MISMATCH", "Scenario and trace workload versions differ")
        if not isinstance(scenario.root_arrival.get("random_seed"), int) or scenario.root_arrival.get("mode") not in {"trace_offsets", "fixed_rate", "poisson", "fixed_concurrency"}:
            report.add("SCHEMA_ERROR", "Invalid root arrival mode or random seed")
        mix = scenario.root_arrival.get("task_mix")
        if (not isinstance(mix, dict) or not mix or set(mix) - set(scenario.trace_set_ids)
                or any(not isinstance(x, (int, float)) or x < 0 for x in mix.values()) or sum(mix.values()) <= 0):
            report.add("SCHEMA_ERROR", "task_mix must assign nonnegative weight to known trace IDs")
        if not isinstance(scenario.measurement.get("task_slo_sec"), (int, float)) or scenario.measurement.get("task_slo_sec", -1) < 0:
            report.add("SCHEMA_ERROR", "Invalid task SLO")
        target = scenario.measurement.get("slo_attainment_target")
        if not isinstance(target, (int, float)) or not 0 <= target <= 1:
            report.add("SCHEMA_ERROR", "Invalid SLO-attainment target")
        if system.accelerator_count < 1:
            report.add("SCHEMA_ERROR", "SystemConfig accelerator_count must be positive")
        if trace.metadata.trace_id not in scenario.trace_set_ids or scenario.trace_hashes.get(trace.metadata.trace_id) != trace.metadata.workload_hash:
            report.add("TRACE_HASH_MISMATCH", "Scenario trace identity does not match bundle")
        contract = scenario.model_input_contract
        if system.model_identity != contract.get("model_identity"):
            report.add("TRACE_HASH_MISMATCH", "SUT model identity differs from the scenario model")
        if system.tokenizer_hash != contract.get("tokenizer_hash"):
            report.add("TOKENIZER_MISMATCH", "SUT tokenizer hash differs from the scenario")
        if system.chat_template_hash != contract.get("chat_template_hash"):
            report.add("CHAT_TEMPLATE_MISMATCH", "SUT chat-template hash differs from the scenario")
        for op in trace.operations:
            if not op.llm:
                continue
            if op.llm["tokenizer"].get("hash") != contract.get("tokenizer_hash"):
                report.add("TOKENIZER_MISMATCH", "Trace and scenario tokenizer hashes differ", operation_id=op.operation_id)
            if op.llm["chat_template"].get("hash") != contract.get("chat_template_hash"):
                report.add("CHAT_TEMPLATE_MISMATCH", "Trace and scenario chat-template hashes differ", operation_id=op.operation_id)
            for key, value in contract.get("generation_parameters", {}).items():
                if op.llm.get("generation_parameters", {}).get(key) != value:
                    report.add("TRACE_HASH_MISMATCH", "Trace and scenario generation parameters differ", operation_id=op.operation_id, field=key)
            resolved = scenario.resolved_request_contracts.get(op.operation_id)
            if not resolved or resolved.get("request_hash") != sha256_json(scenario.materialize_request(op)):
                report.add("CACHE_SCOPE_VIOLATION", "Operation lacks an exact resolved target-input contract", operation_id=op.operation_id)
        mode = scenario.replay.get("fidelity_mode")
        capability = "supports_token_lock" if mode == "token_locked" else "supports_length_lock"
        if mode not in {"length_locked", "token_locked"} or not system.capabilities.get(capability, False):
            report.add("REPLAY_MODE_UNSUPPORTED", f"SUT does not declare {mode} support")
        if mode == "token_locked":
            for op in trace.operations:
                if op.llm and (not isinstance(op.llm.get("recorded_output_token_ids"), list)
                               or len(op.llm["recorded_output_token_ids"]) != op.llm.get("recorded_output_length")
                               or op.llm.get("recorded_output_token_source") not in {"backend_token_ids", "synthetic_exact"}):
                    report.add("REPLAY_MODE_UNSUPPORTED", "Trace lacks exact recorded output token IDs", operation_id=op.operation_id)
        scopes = {x.reuse_scope_id for x in trace.sessions if x.reuse_scope_id} | {x.reuse_scope_id for x in trace.artifacts if x.reuse_scope_id}
        salts = scenario.resolved_cache_scope_salts
        if scopes - set(salts) or len({salts.get(x) for x in scopes}) != len(scopes):
            report.add("CACHE_SCOPE_VIOLATION", "Reuse scopes lack deterministic, distinct cache salts")
        if run:
            if run.get("cancelled"):
                report.add("RUN_CANCELLED", "Run was cancelled")
            for task in trace.tasks:
                if task.task_id not in run.get("completed_task_ids", []):
                    report.add("INCOMPLETE_TASK", "Attempted task did not complete", task_id=task.task_id)
            for mismatch in run.get("output_length_mismatches", []):
                report.add("OUTPUT_LENGTH_MISMATCH", "Backend output length differs from recorded length", **mismatch)
            if run.get("truncated_input"):
                report.add("TRUNCATED_INPUT", "SUT truncated a benchmark input")
            if run.get("context_overflow"):
                report.add("CONTEXT_OVERFLOW", "Benchmark input exceeded the configured context")
        return report


class LoweringValidator:
    def validate(self, trace: SemanticTrace, lowering_manifest: dict[str, Any]) -> ValidationReport:
        report = ValidationReport()
        mapping = lowering_manifest.get("semantic_to_chakra_nodes", {})
        lowered = set(lowering_manifest.get("lowered_operation_ids", []))
        for op in trace.operations:
            if op.operation_id in lowered and not mapping.get(op.operation_id):
                report.add("INVALID_CONTEXT_REFERENCE", "Lowered operation has no Chakra nodes", operation_id=op.operation_id)
        node_to_semantic = lowering_manifest.get("chakra_node_to_semantic", {})
        for opid, nodes in mapping.items():
            if any(node_to_semantic.get(str(node)) != opid for node in nodes):
                report.add("INVALID_CONTEXT_REFERENCE", "One-to-many Chakra provenance is not reversible", operation_id=opid)
        if lowered and not lowering_manifest.get("converter_version"):
            report.add("SCHEMA_ERROR", "Lowering manifest lacks converter version")
        chakra_edges = {tuple(map(str, edge)) for edge in lowering_manifest.get("chakra_edges", [])}
        opmap = {x.operation_id: x for x in trace.operations}
        for opid in lowered:
            if opid not in opmap: continue
            for parent in opmap[opid].dependencies & lowered:
                if not any((str(src), str(dst)) in chakra_edges for src in mapping.get(parent, []) for dst in mapping.get(opid, [])):
                    report.add("MISSING_DEPENDENCY", "Chakra lowering does not preserve a semantic dependency", src=parent, dst=opid)
        return report

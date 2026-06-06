"""MASBench-Arch week-1 trace, token, latency, and snapshot helpers."""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "masbench_arch_trace_v0.1"


class TraceHookManager:
    """Small hook bus for future workflow/motif composition.

    Exporters and probes can subscribe to all emitted events without changing
    topology code. The JSONL event stream remains the canonical trace.
    """

    def __init__(self) -> None:
        self._hooks: list[Any] = []

    def register(self, callback: Any) -> None:
        self._hooks.append(callback)

    def emit(self, event: dict[str, Any]) -> None:
        for callback in list(self._hooks):
            callback(event)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def utc_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def stable_hash(value: Any) -> str:
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8", errors="replace")
    else:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class TokenCounter:
    """Best-effort tokenizer adapter with an explicit fallback marker."""

    def __init__(self, model: str | None = None) -> None:
        self.source = "char_heuristic"
        self._encoder: Any = None
        try:
            import tiktoken  # type: ignore

            self._encoder = tiktoken.encoding_for_model(model or "gpt-4o-mini")
            self.source = "tokenizer"
        except Exception:
            try:
                from transformers import AutoTokenizer  # type: ignore

                self._encoder = AutoTokenizer.from_pretrained(model or "gpt2")
                self.source = "tokenizer"
            except Exception:
                self._encoder = None

    def count(self, text: str | None) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        return max(1, (len(text) + 3) // 4)


_DEFAULT_COUNTER = TokenCounter()


def estimate_tokens(text: str | None) -> int:
    return _DEFAULT_COUNTER.count(text)


def token_count_source() -> str:
    return _DEFAULT_COUNTER.source


def latency_delay(profile: str, rng: random.Random, scale: float = 1.0) -> float:
    profile = (profile or "none").lower()
    if profile == "none":
        delay = 0.0
    elif profile == "fast":
        delay = rng.uniform(0.1, 0.5)
    elif profile == "medium":
        delay = rng.uniform(1.0, 3.0)
    elif profile == "slow":
        delay = rng.uniform(5.0, 10.0)
    elif profile == "heavy_tail":
        delay = rng.uniform(10.0, 30.0) if rng.random() < 0.15 else rng.uniform(1.0, 3.0)
    else:
        delay = 0.0
    return max(0.0, delay * float(scale))


@dataclass
class TraceContext:
    run_id: str
    topology: str
    topology_role: str
    instance_id: str
    task_source: str
    workflow_id: str
    trace_path: Path
    summary_path: Path
    random_seed: int = 42
    trace_level: str = "arch"
    export_views: bool = True
    record_model_outputs: bool = False
    model_outputs_path: Path | None = None
    collect_backend_metrics: bool = False
    backend_metrics_path: Path | None = None
    mode: str = "topology"
    motif_name: str = ""
    motif_instance_id: str = ""
    parent_motif_id: str = ""
    composed_from_topologies: list[str] = field(default_factory=list)
    environment_id: str = field(default_factory=lambda: stable_hash({"cwd": str(Path.cwd()), "ts": now_ts()})[:16])
    start_perf: float = field(default_factory=time.perf_counter)
    events: list[dict[str, Any]] = field(default_factory=list)
    model_outputs: list[dict[str, Any]] = field(default_factory=list)
    backend_metrics_summary: dict[str, Any] = field(default_factory=dict)
    hooks: TraceHookManager = field(default_factory=TraceHookManager)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, **fields: Any) -> dict[str, Any]:
        with self.lock:
            event = {
                "schema_version": SCHEMA_VERSION,
                "event_id": fields.pop("event_id", str(uuid.uuid4())),
                "event_type": fields.pop("event_type", "unknown"),
                "timestamp": fields.pop("timestamp", utc_iso()),
                "relative_time_sec": round(time.perf_counter() - self.start_perf, 6),
                "run_id": self.run_id,
                "topology": self.topology,
                "topology_role": self.topology_role,
                "instance_id": self.instance_id,
                "task_source": self.task_source,
                "workflow_id": self.workflow_id,
                "workflow_name": fields.pop("workflow_name", self.motif_name or self.topology),
                "mode": fields.pop("mode", self.mode),
                "motif_name": fields.pop("motif_name", self.motif_name),
                "motif_instance_id": fields.pop("motif_instance_id", self.motif_instance_id),
                "parent_motif_id": fields.pop("parent_motif_id", self.parent_motif_id),
                "composed_from_topologies": fields.pop("composed_from_topologies", list(self.composed_from_topologies)),
                "node_id": fields.pop("node_id", ""),
                "node_name": fields.pop("node_name", ""),
                "node_type": fields.pop("node_type", "workflow"),
                "round_id": fields.pop("round_id", None),
                "manager_round_id": fields.pop("manager_round_id", None),
                "peer_round_id": fields.pop("peer_round_id", None),
                "attempt_id": fields.pop("attempt_id", 0),
                "retry_count": fields.pop("retry_count", 0),
                "parents": fields.pop("parents", []),
                "parent_node_ids": fields.pop("parent_node_ids", []),
                "children": fields.pop("children", []),
                "dependency_edges": fields.pop("dependency_edges", []),
                "motif_tags": fields.pop("motif_tags", []),
                "parallel_group": fields.pop("parallel_group", None),
                "criticality": fields.pop("criticality", "unknown"),
                "workload_role": fields.pop("workload_role", "unknown"),
                "critical_path_candidate": fields.pop("critical_path_candidate", False),
                "critical_stage": fields.pop("critical_stage", "none"),
                "fan_in_count": fields.pop("fan_in_count", 0),
                "fan_out_count": fields.pop("fan_out_count", 0),
                "status": fields.pop("status", "success"),
                "duration_sec": fields.pop("duration_sec", 0.0),
                "duration_source": fields.pop("duration_source", "mock_measured"),
                "replay_policy": fields.pop("replay_policy", "synthetic_only"),
                "environment_id": self.environment_id,
                "random_seed": self.random_seed,
                "trace_level": self.trace_level,
                "extra": fields.pop("extra", {}),
            }
            event.update(fields)
            if self.trace_level == "basic":
                self._redact_basic(event)
            self.events.append(event)
            self.hooks.emit(event)
        return event

    def record_model_output(
        self,
        *,
        event: dict[str, Any],
        output_text: str,
        input_text: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        response_metadata: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Store full model text outside the canonical JSONL event stream.

        The JSONL trace remains compact and simulation-oriented. When enabled,
        this sidecar keeps semantic outputs for inspection/replay research and
        links them back to the LLM event through stable ids and hashes.
        """
        if not self.record_model_outputs:
            return None
        output_hash = stable_hash(output_text)
        prompt_hash = stable_hash(input_text or "")
        artifact_id = f"model_output_{output_hash[:16]}"
        path = self.model_outputs_path or self.trace_path.with_name(f"{self.trace_path.stem}_model_outputs.json")
        record = {
            "schema_version": SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "artifact_type": "model_output",
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "run_id": self.run_id,
            "topology": self.topology,
            "topology_role": self.topology_role,
            "instance_id": self.instance_id,
            "task_source": self.task_source,
            "workflow_id": self.workflow_id,
            "node_id": event.get("node_id"),
            "node_name": event.get("node_name"),
            "node_type": event.get("node_type"),
            "agent_id": event.get("agent_id"),
            "agent_role": event.get("agent_role"),
            "round_id": event.get("round_id"),
            "manager_round_id": event.get("manager_round_id"),
            "peer_round_id": event.get("peer_round_id"),
            "parallel_group": event.get("parallel_group"),
            "llm_mode": event.get("llm_mode"),
            "backend_base_url": event.get("backend_base_url"),
            "model": event.get("model"),
            "llm_request_id": event.get("llm_request_id"),
            "request_id_for_backend": event.get("request_id_for_backend"),
            "prompt_template": event.get("prompt_template"),
            "prompt_hash": event.get("prompt_hash") or prompt_hash,
            "input_hash": prompt_hash,
            "output_hash": output_hash,
            "input_chars": len(input_text or ""),
            "output_chars": len(output_text or ""),
            "input_tokens_est": estimate_tokens(input_text or ""),
            "output_tokens_est": estimate_tokens(output_text),
            "token_count_source": token_count_source(),
            "backend_prompt_tokens": event.get("backend_prompt_tokens"),
            "backend_completion_tokens": event.get("backend_completion_tokens"),
            "backend_total_tokens": event.get("backend_total_tokens"),
            "backend_finish_reason": event.get("backend_finish_reason"),
            "timestamp": event.get("timestamp"),
            "relative_time_sec": event.get("relative_time_sec"),
            "duration_sec": event.get("duration_sec"),
            "duration_source": event.get("duration_source"),
            "replay_policy": event.get("replay_policy"),
            "output_text": output_text,
            "response_metadata": response_metadata or {},
            "extra": extra or {},
        }
        if system_prompt is not None:
            record["system_prompt_hash"] = stable_hash(system_prompt)
            record["system_prompt_chars"] = len(system_prompt)
            record["system_prompt_tokens_est"] = estimate_tokens(system_prompt)
        if user_prompt is not None:
            record["user_prompt_hash"] = stable_hash(user_prompt)
            record["user_prompt_chars"] = len(user_prompt)
            record["user_prompt_tokens_est"] = estimate_tokens(user_prompt)
        with self.lock:
            self.model_outputs.append(record)
            event["model_output_artifact_id"] = artifact_id
            event["model_output_path"] = str(path)
            event["model_output_hash"] = output_hash
        return record

    def save(self) -> Path:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_simulator_ready_records()
        with self.trace_path.open("w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        if self.record_model_outputs and self.model_outputs:
            path = self.model_outputs_path or self.trace_path.with_name(f"{self.trace_path.stem}_model_outputs.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "topology": self.topology,
                "instance_id": self.instance_id,
                "trace_path": str(self.trace_path),
                "record_count": len(self.model_outputs),
                "records": self.model_outputs,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return self.trace_path

    def _ensure_simulator_ready_records(self) -> None:
        """Append normalized records so raw JSONL is simulator-ready by default.

        These records duplicate selected event fields in a stable tabular shape.
        They are intentionally lightweight and deterministic; analysis scripts can
        still rebuild richer CSV views, but a simulator no longer has to infer the
        basic workflow/node/query/tool/barrier/prefix records from arbitrary event
        payloads.
        """
        if any(event.get("event_type") == "simulator_ready_manifest" for event in self.events):
            return
        source_events = [event for event in self.events if not str(event.get("event_type", "")).startswith("simulator_ready_")]
        llm_events = [event for event in source_events if event.get("event_type") == "llm_request_end"]
        tool_events = [event for event in source_events if str(event.get("event_type", "")).startswith("tool_")]
        barrier_events = [event for event in source_events if event.get("node_type") == "barrier" or event.get("event_type") == "barrier"]
        node_events = [
            event
            for event in source_events
            if event.get("node_id")
            and (
                event.get("event_type") in {"llm_request_end", "tool_search", "tool_controlled_delay", "barrier", "motif_control", "route_decision"}
                or event.get("node_type") in {"barrier", "edge"}
            )
        ]
        workflow_start = next((event for event in source_events if event.get("event_type") in {"workflow_start", "motif_start"}), source_events[0] if source_events else {})
        workflow_end = next((event for event in reversed(source_events) if event.get("event_type") in {"workflow_end", "motif_end"}), source_events[-1] if source_events else {})
        workflow_name = workflow_start.get("workflow_name") or self.motif_name or self.topology
        self._append_sim_record(
            "simulator_ready_manifest",
            record_type="manifest",
            workflow_run_id=self.workflow_id,
            workflow_name=workflow_name,
            raw_event_count=len(source_events),
            workflow_record_count=1,
            node_record_count=len(node_events),
            llm_query_record_count=len(llm_events),
            tool_call_record_count=len(tool_events),
            barrier_record_count=len(barrier_events),
            prefix_cache_record_count=len(llm_events),
            actual_backend_metric_source="vllm_metrics_sidecar" if self.collect_backend_metrics else "unavailable",
            potential_prefix_source="offline_prompt_hash_and_shared_block_hash",
        )
        self._append_sim_record(
            "simulator_ready_workflow",
            record_type="workflow",
            workflow_run_id=self.workflow_id,
            workflow_name=workflow_name,
            workflow_type=self.mode,
            motif_type=self.motif_name or self.topology,
            composed_subgraphs=list(self.composed_from_topologies),
            task_id=self.task_source,
            prompt_id=self.instance_id,
            start_time=workflow_start.get("relative_time_sec", 0.0),
            end_time=workflow_end.get("relative_time_sec", 0.0),
            end_to_end_latency=max(0.0, _safe_float(workflow_end.get("relative_time_sec")) - _safe_float(workflow_start.get("relative_time_sec"))),
            status=workflow_end.get("status", "unknown"),
            backend_base_url=_first_present(llm_events, "backend_base_url"),
            model_name=_first_present(llm_events, "model_name") or _first_present(llm_events, "model"),
            backend_metrics_summary=dict(self.backend_metrics_summary),
            actual_backend_fields_available=sorted(self.backend_metrics_summary.keys()),
        )
        child_map: dict[str, set[str]] = {}
        for event in source_events:
            dst = str(event.get("dst_node") or event.get("node_id") or "")
            for parent in event.get("parent_node_ids") or event.get("parents") or []:
                child_map.setdefault(str(parent), set()).add(dst)
        for event in node_events:
            node_id = str(event.get("node_id") or "")
            self._append_sim_record(
                "simulator_ready_node",
                record_type="node",
                source_event_id=event.get("event_id"),
                workflow_run_id=self.workflow_id,
                workflow_name=event.get("workflow_name") or workflow_name,
                node_id=node_id,
                agent_id=event.get("agent_id") or node_id,
                agent_role=event.get("agent_role") or event.get("node_type"),
                node_type=event.get("agent_role") or event.get("node_type"),
                parent_node_ids=event.get("parent_node_ids") or event.get("parents") or [],
                child_node_ids=sorted(child_map.get(node_id, set())),
                dependency_type=_dependency_type(event),
                branch_id=event.get("background_branch_id") or event.get("parallel_group") or "",
                round_id=event.get("round_id") or event.get("peer_round_id") or event.get("manager_round_id"),
                loop_iteration_id=event.get("retry_count") or event.get("debug_loop_count") or 0,
                is_critical_path=bool(event.get("critical_path_candidate") or event.get("criticality") == "critical"),
                criticality=event.get("criticality", "unknown"),
                start_time=_event_start(event),
                end_time=_safe_float(event.get("relative_time_sec")),
                duration_sec=event.get("duration_sec", 0.0),
            )
        previous_same_workflow: list[dict[str, Any]] = []
        for event in llm_events:
            request_id = event.get("request_id_for_backend") or event.get("llm_request_id") or event.get("event_id")
            prompt_segments = {
                "system_prompt_tokens": event.get("system_prompt_tokens_est"),
                "task_prompt_tokens": event.get("user_prompt_tokens_est"),
                "shared_context_tokens": event.get("shared_context_tokens_est") or event.get("round_shared_context_tokens"),
                "private_memory_tokens": event.get("private_history_tokens"),
                "retrieved_tool_context_tokens": event.get("retrieved_context_tokens_est") or event.get("shared_evidence_read_tokens_est"),
                "peer_message_tokens": event.get("peer_message_tokens_est"),
                "review_feedback_tokens": event.get("review_feedback_tokens_est"),
                "final_instruction_tokens": event.get("manager_instruction_tokens_est"),
            }
            self._append_sim_record(
                "simulator_ready_llm_query",
                record_type="llm_query",
                source_event_id=event.get("event_id"),
                request_id=request_id,
                workflow_run_id=self.workflow_id,
                workflow_name=event.get("workflow_name") or workflow_name,
                node_id=event.get("node_id"),
                agent_id=event.get("agent_id") or event.get("node_id"),
                role=event.get("agent_role"),
                node_type=event.get("agent_role") or event.get("node_type"),
                model_name=event.get("model_name") or event.get("model"),
                submit_time=event.get("request_submit_ts"),
                queue_start_time="unavailable",
                queue_end_time="unavailable",
                prefill_start_time="unavailable",
                prefill_end_time="unavailable",
                decode_start_time=event.get("response_start_ts"),
                decode_end_time=event.get("response_end_ts"),
                finish_time=event.get("response_end_ts") or event.get("relative_time_sec"),
                ttft=event.get("ttft_sec") if event.get("ttft_sec") is not None else "unavailable",
                tpot=event.get("tpot_sec") if event.get("tpot_sec") is not None else "unavailable",
                request_e2e_sec=event.get("request_e2e_sec") or event.get("duration_sec"),
                input_tokens=event.get("input_tokens") or event.get("input_tokens_est"),
                prefill_tokens=event.get("input_tokens") or event.get("input_tokens_est"),
                output_tokens=event.get("output_tokens") or event.get("output_tokens_est"),
                decode_tokens=event.get("output_tokens") or event.get("output_tokens_est"),
                reasoning_tokens=event.get("reasoning_tokens", "unavailable"),
                prompt_segments=prompt_segments,
                prompt_hash=event.get("prompt_hash"),
                segment_hashes=event.get("prompt_segment_hashes") or event.get("shared_block_hashes") or {},
                sampling_params={"max_output_tokens": event.get("max_output_tokens")},
                status=event.get("status", "success"),
                critical_path_candidate=bool(event.get("critical_path_candidate") or event.get("criticality") == "critical"),
            )
            prefix = _potential_prefix_for_event(event, previous_same_workflow)
            input_tokens = max(_safe_int(event.get("input_tokens") or event.get("input_tokens_est")), 1)
            dynamic_context_new_tokens = max(0, input_tokens - _safe_int(prefix.get("potential_prefix_match_tokens")))
            self._append_sim_record(
                "simulator_ready_prefix_cache",
                record_type="prefix_cache",
                source_event_id=event.get("event_id"),
                workflow_run_id=self.workflow_id,
                workflow_name=event.get("workflow_name") or workflow_name,
                request_id=request_id,
                node_id=event.get("node_id"),
                input_tokens=input_tokens,
                shared_context_reuse_tokens=sum(_safe_int(v) for v in (event.get("shared_block_tokens") or {}).values()),
                private_context_reuse_tokens=event.get("private_history_tokens") or 0,
                dynamic_context_new_tokens=dynamic_context_new_tokens,
                actual_prefix_cache_hit_tokens="unavailable",
                actual_prefix_cache_hit_rate="unavailable",
                actual_cached_blocks="unavailable",
                actual_new_blocks="unavailable",
                cache_eviction_count="unavailable",
                actual_metric_source="unavailable_per_request",
                potential_metric_source="offline_prompt_hash_and_shared_block_hash",
                **prefix,
            )
            previous_same_workflow.append(event)
        for event in tool_events:
            self._append_sim_record(
                "simulator_ready_tool_call",
                record_type="tool_call",
                source_event_id=event.get("event_id"),
                tool_call_id=event.get("tool_call_id") or event.get("event_id"),
                workflow_run_id=self.workflow_id,
                workflow_name=event.get("workflow_name") or workflow_name,
                node_id=event.get("node_id"),
                agent_id=event.get("agent_id") or event.get("node_id"),
                tool_name=event.get("tool_name"),
                start_time=_event_start(event),
                end_time=event.get("relative_time_sec"),
                absolute_start_time=event.get("tool_start_ts"),
                absolute_end_time=event.get("tool_end_ts") or event.get("tool_return_ts"),
                latency=event.get("tool_latency_sec") or event.get("effective_duration_sec") or event.get("duration_sec"),
                input_size_tokens=event.get("tool_query_tokens_est") or event.get("input_size_tokens"),
                input_size_chars=event.get("tool_query_chars") or event.get("input_size_chars"),
                output_size_tokens=event.get("tool_output_tokens_est") or event.get("output_tokens_est"),
                output_size_chars=event.get("tool_output_chars") or event.get("output_chars"),
                status=event.get("status", "success"),
                retry_count=event.get("retry_count") or 0,
                written_to_shared_context=bool(event.get("is_shared_context") or event.get("shared_context")),
            )
        node_end: dict[str, float] = {}
        for event in source_events:
            if str(event.get("event_type", "")).endswith("_end") and event.get("node_id"):
                node_end[str(event.get("node_id"))] = max(node_end.get(str(event.get("node_id")), 0.0), _safe_float(event.get("relative_time_sec")))
        for event in barrier_events:
            arrivals = event.get("arrived_nodes") or event.get("waiting_for_nodes") or []
            arrival_times = {str(node): node_end[str(node)] for node in arrivals if str(node) in node_end}
            earliest = min(arrival_times.values()) if arrival_times else ""
            latest = max(arrival_times.values()) if arrival_times else ""
            release = _safe_float(event.get("relative_time_sec"))
            waits = {node: max(0.0, release - arrival) for node, arrival in arrival_times.items()}
            self._append_sim_record(
                "simulator_ready_barrier",
                record_type="barrier",
                source_event_id=event.get("event_id"),
                barrier_id=event.get("barrier_id") or event.get("node_id"),
                workflow_run_id=self.workflow_id,
                workflow_name=event.get("workflow_name") or workflow_name,
                barrier_type=_barrier_type(event),
                participating_node_ids=arrivals,
                earliest_arrival_time=earliest,
                latest_arrival_time=latest,
                barrier_release_time=release,
                per_node_wait_time=waits,
                straggler_gap=(_safe_float(latest) - _safe_float(earliest)) if latest != "" and earliest != "" else event.get("straggler_gap_sec") or event.get("barrier_wait_sec"),
                downstream_node_ids=sorted(child_map.get(str(event.get("node_id")), set())),
            )

    def _append_sim_record(self, event_type: str, **fields: Any) -> None:
        record = {
            "schema_version": SCHEMA_VERSION,
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp": utc_iso(),
            "relative_time_sec": round(time.perf_counter() - self.start_perf, 6),
            "run_id": self.run_id,
            "topology": self.topology,
            "topology_role": self.topology_role,
            "instance_id": self.instance_id,
            "task_source": self.task_source,
            "workflow_id": self.workflow_id,
            "workflow_name": fields.pop("workflow_name", self.motif_name or self.topology),
            "mode": self.mode,
            "motif_name": self.motif_name,
            "motif_instance_id": self.motif_instance_id,
            "parent_motif_id": self.parent_motif_id,
            "composed_from_topologies": list(self.composed_from_topologies),
            "node_id": fields.get("node_id", ""),
            "node_type": "simulator_ready_record",
            "status": "success",
            "duration_sec": 0.0,
            "duration_source": "derived_trace_record",
            "replay_policy": "derived_from_raw_trace",
            "environment_id": self.environment_id,
            "random_seed": self.random_seed,
            "trace_level": self.trace_level,
        }
        record.update(fields)
        self.events.append(record)

    def _redact_basic(self, event: dict[str, Any]) -> None:
        """Keep simulator fields while dropping bulky/debug-only payloads."""
        for key in ("request_metadata",):
            if key in event:
                event[key] = {"redacted": True, "hash": stable_hash(event[key])}
        extra = event.get("extra")
        if isinstance(extra, dict) and extra:
            event["extra"] = {"redacted": True, "hash": stable_hash(extra)}


def make_event(
    *,
    agent_name: str,
    event_type: str,
    parents: list[str] | None = None,
    children: list[str] | None = None,
    input_text: str = "",
    output_text: str = "",
    duration_sec: float = 0.0,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Legacy compatibility for the old demo workflow."""
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "agent_name": agent_name,
        "event_type": event_type,
        "parents": parents or [],
        "children": children or [],
        "input_chars": len(input_text or ""),
        "output_chars": len(output_text or ""),
        "estimated_input_tokens": estimate_tokens(input_text),
        "estimated_output_tokens": estimate_tokens(output_text),
        "duration_sec": round(duration_sec, 6),
        "status": status,
        "extra": extra or {},
    }


def save_trace(log_dir: str | Path, trace_events: list[dict[str, Any]], run_id: str | None = None) -> Path:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / f"run_{run_id or now_ts()}.json"
    output_path.write_text(json.dumps(trace_events, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def save_run_result(log_dir: str | Path, result: dict[str, Any], run_id: str | None = None) -> Path:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / f"result_{run_id or now_ts()}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.duration = time.perf_counter() - self.start


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "" or value == "unavailable":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "" or value == "unavailable":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _first_present(events: list[dict[str, Any]], key: str) -> Any:
    return next((event.get(key) for event in events if event.get(key) not in {None, ""}), "")


def _event_start(event: dict[str, Any]) -> float:
    return max(0.0, _safe_float(event.get("relative_time_sec")) - _safe_float(event.get("duration_sec")))


def _dependency_type(event: dict[str, Any]) -> str:
    if event.get("dependency_type"):
        return str(event.get("dependency_type"))
    if event.get("transfer_type") == "broadcast" or event.get("peer_round_id") is not None:
        return "debate_round"
    if event.get("transfer_type") in {"aggregation", "late_non_blocking_context"} or _safe_int(event.get("fan_in_count")) > 1:
        return "fan_in"
    if event.get("tool_stalled") or event.get("tool_name"):
        return "tool_dependency"
    if _safe_int(event.get("retry_count")) > 0:
        return "review_loop"
    return "sequential"


def _barrier_type(event: dict[str, Any]) -> str:
    bid = str(event.get("barrier_id") or event.get("node_id") or "")
    if "debate" in bid or event.get("peer_round_id") is not None or "all_gather" in bid:
        return "debate_round_sync"
    if "manager" in bid:
        return "manager_wait"
    if "review" in bid:
        return "review_loop_wait"
    if "final" in bid:
        return "finalizer_wait"
    return "fan_in_merge"


def _shared_block_overlap(a: dict[str, Any], b: dict[str, Any]) -> int:
    tokens_a = a.get("shared_block_tokens") or {}
    tokens_b = b.get("shared_block_tokens") or {}
    hashes_a = a.get("shared_block_hashes") or {}
    hashes_b = b.get("shared_block_hashes") or {}
    inv_b = {str(v): k for k, v in hashes_b.items()}
    shared = 0
    for block_id, hsh in hashes_a.items():
        other = inv_b.get(str(hsh))
        if other is not None:
            shared += min(_safe_int(tokens_a.get(block_id)), _safe_int(tokens_b.get(other)))
    return shared


def _potential_prefix_for_event(event: dict[str, Any], previous: list[dict[str, Any]]) -> dict[str, Any]:
    best = 0
    source = "none"
    for other in previous:
        if event.get("prompt_hash") and event.get("prompt_hash") == other.get("prompt_hash"):
            candidate = min(_safe_int(event.get("input_tokens") or event.get("input_tokens_est")), _safe_int(other.get("input_tokens") or other.get("input_tokens_est")))
            if candidate > best:
                best = candidate
                source = "exact_prompt_hash"
        shared = _shared_block_overlap(event, other)
        if shared > best:
            best = shared
            source = "shared_block_hash"
        segment = _segment_reuse_proxy(event, other)
        if segment > best:
            best = segment
            source = "segment_token_proxy"
    input_tokens = max(_safe_int(event.get("input_tokens") or event.get("input_tokens_est")), 1)
    return {
        "potential_prefix_match_tokens": best,
        "potential_prefix_reuse_rate": best / input_tokens,
        "intra_workflow_prefix_match_tokens": best,
        "inter_workflow_prefix_match_tokens": 0,
        "potential_prefix_source": source,
    }


def _segment_reuse_proxy(event: dict[str, Any], other: dict[str, Any]) -> int:
    if event.get("workflow_id") != other.get("workflow_id"):
        return 0
    if event.get("prompt_template") != other.get("prompt_template"):
        return 0
    system = min(_safe_int(event.get("system_prompt_tokens_est")), _safe_int(other.get("system_prompt_tokens_est")))
    user = min(_safe_int(event.get("user_prompt_tokens_est") or event.get("shared_context_tokens_est")), _safe_int(other.get("user_prompt_tokens_est") or other.get("shared_context_tokens_est")))
    return system + int(0.75 * user)

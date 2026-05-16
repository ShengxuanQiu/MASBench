"""Unified topology interface and shared runtime helpers."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..llm_backends import MockLLM, OpenAICompatibleLLM
from ..search_providers import BaseSearchProvider, record_search_event
from ..tracing import TraceContext, estimate_tokens, stable_hash, token_count_source


@dataclass
class TopologyConfig:
    topology_name: str
    run_id: str
    task_id: str
    instance_id: str
    query: str
    max_rounds: int = 2
    num_agents: int = 3
    max_retries: int = 0
    stop_condition: str = "default"
    llm_mode: str = "mock"
    tool_mode: str = "synthetic"
    latency_profile: str = "none"
    latency_scale: float = 1.0
    random_seed: int = 42
    max_concurrent_llm_calls: int = 2
    dispatch_policy: str = "fcfs"
    backend_base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "local-mas-model"
    task_source: str = "manual"
    trace_dir: Path = Path("traces")
    repo_path: Path | None = None
    replay_snapshot_dir: Path | None = None
    record_tool_results: bool = True
    search_provider: str = "synthetic"
    manager_policy: str = "rule_based"
    aggregation_policy: str = "concat_summary"
    communication_topology: str = "all_to_all"
    debate_rounds: int = 2
    peer_rounds: int = 1
    force_centralized_rounds: int = 0
    allow_parallel_workers: bool = True
    force_live_search_test: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class RunnableTopology(Protocol):
    config: TopologyConfig
    motif_tags: list[str]

    def run(self) -> dict[str, Any]:
        ...

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        ...


class BaseTopology:
    motif_tags: list[str] = []

    def __init__(
        self,
        config: TopologyConfig,
        *,
        llm: MockLLM | OpenAICompatibleLLM,
        search_provider: BaseSearchProvider,
        trace: TraceContext,
    ) -> None:
        self.config = config
        self.llm = llm
        self.search_provider = search_provider
        self.trace = trace
        self.artifacts: dict[str, str] = {}

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        old_query = self.config.query
        self.config.query = str(payload.get("query") or old_query)
        try:
            return self.run()
        finally:
            self.config.query = old_query

    def workflow_start(self) -> None:
        self.trace.emit(
            event_type="workflow_start",
            node_id="START",
            node_name="START",
            node_type="workflow",
            motif_tags=self.motif_tags,
            status="start",
            extra={"query_hash": stable_hash(self.config.query), "config": self._config_dict()},
        )

    def workflow_end(self, final_answer: str) -> dict[str, Any]:
        self.trace.emit(
            event_type="workflow_end",
            node_id="END",
            node_name="END",
            node_type="workflow",
            motif_tags=self.motif_tags,
            status="success",
            output_hash=stable_hash(final_answer),
            output_chars=len(final_answer),
            output_tokens_est=estimate_tokens(final_answer),
            token_count_source=token_count_source(),
        )
        summary = self.build_summary(final_answer)
        self.trace.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        self.trace.save()
        return summary

    def call_llm(
        self,
        *,
        node_id: str,
        node_name: str,
        node_type: str,
        agent_role: str,
        prompt_template: str,
        system_prompt: str,
        user_prompt: str,
        round_id: int | None = None,
        manager_round_id: int | None = None,
        peer_round_id: int | None = None,
        parents: list[str] | None = None,
        parallel_group: str | None = None,
        criticality: str = "unknown",
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        metadata = {
            "agent_id": node_id,
            "agent_role": agent_role,
            "round_id": round_id,
            "manager_round_id": manager_round_id,
            "peer_round_id": peer_round_id,
            "max_rounds": self.config.max_rounds,
            "force_continue_rounds": self.config.force_centralized_rounds,
            "worker_ids": [f"worker_{i}" for i in range(1, self.config.num_agents + 1)],
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        prompt = system_prompt + "\n" + user_prompt
        self.trace.emit(
            event_type="llm_request_start",
            node_id=node_id,
            node_name=node_name,
            node_type="llm",
            round_id=round_id,
            manager_round_id=manager_round_id,
            peer_round_id=peer_round_id,
            parents=parents or [],
            motif_tags=self.motif_tags,
            parallel_group=parallel_group,
            criticality=criticality,
            status="start",
            agent_id=node_id,
            agent_role=agent_role,
            prompt_template=prompt_template,
            llm_mode=self.config.llm_mode,
            backend_base_url=self.config.backend_base_url,
            model=self.config.model,
            llm_request_id=str(uuid.uuid4()),
            input_chars=len(prompt),
            input_tokens_est=estimate_tokens(prompt),
            system_prompt_tokens_est=estimate_tokens(system_prompt),
            user_prompt_tokens_est=estimate_tokens(user_prompt),
            token_count_source=token_count_source(),
            shared_context_hash=stable_hash(user_prompt)[:16],
            prompt_hash=stable_hash(prompt),
            priority=1.0 if criticality == "critical" else 0.0,
            dispatch_policy=self.config.dispatch_policy,
            request_metadata=metadata,
        )
        result = self.llm.invoke(system_prompt, user_prompt, metadata)
        output = result.content
        self.trace.emit(
            event_type="llm_request_end",
            node_id=node_id,
            node_name=node_name,
            node_type="llm",
            round_id=round_id,
            manager_round_id=manager_round_id,
            peer_round_id=peer_round_id,
            parents=parents or [],
            motif_tags=self.motif_tags,
            parallel_group=parallel_group,
            criticality=criticality,
            status="success",
            duration_sec=round(result.duration_sec, 6),
            duration_source="mock_measured" if self.config.llm_mode == "mock" else "llm_backend_measured",
            replay_policy="live",
            agent_id=node_id,
            agent_role=agent_role,
            prompt_template=prompt_template,
            llm_mode=self.config.llm_mode,
            backend_base_url=self.config.backend_base_url,
            model=self.config.model,
            llm_request_id=result.request_id_for_backend,
            request_id_for_backend=result.request_id_for_backend,
            input_chars=len(prompt),
            output_chars=len(output),
            input_tokens_est=estimate_tokens(prompt),
            output_tokens_est=estimate_tokens(output),
            total_tokens_est=estimate_tokens(prompt) + estimate_tokens(output),
            system_prompt_tokens_est=estimate_tokens(system_prompt),
            user_prompt_tokens_est=estimate_tokens(user_prompt),
            shared_context_tokens_est=estimate_tokens(user_prompt),
            retrieved_context_tokens_est=0,
            peer_message_tokens_est=estimate_tokens(str(extra_metadata.get("peer_messages", ""))) if extra_metadata else 0,
            manager_instruction_tokens_est=estimate_tokens(str(extra_metadata.get("manager_instruction", ""))) if extra_metadata else 0,
            token_count_source=token_count_source(),
            queue_wait_sec=result.queue_wait_sec,
            dispatch_start_ts=result.dispatch_start_ts,
            dispatch_end_ts=result.dispatch_end_ts,
            generation_start_ts=result.generation_start_ts,
            generation_end_ts=result.generation_end_ts,
            shared_context_hash=stable_hash(user_prompt)[:16],
            prompt_hash=stable_hash(prompt),
            output_hash=stable_hash(output),
            priority=1.0 if criticality == "critical" else 0.0,
            dispatch_policy=self.config.dispatch_policy,
            request_metadata=result.request_metadata,
        )
        return output

    def search(self, *, node_id: str, node_name: str, query: str) -> list[dict[str, Any]]:
        result = self.search_provider.search(query)
        snapshot_dir = self.config.trace_dir / "snapshots" / self.config.run_id if self.config.record_tool_results else None
        replay_policy = {
            "live": "snapshot_result_recorded_latency",
            "replay": "snapshot_result_recorded_latency" if self.config.latency_profile == "none" else "snapshot_result_param_latency",
            "synthetic": "synthetic_only",
        }.get(self.config.tool_mode, "synthetic_only")
        record_search_event(
            self.trace,
            node_id=node_id,
            node_name=node_name,
            tool_mode=self.config.tool_mode,
            result=result,
            snapshot_dir=snapshot_dir,
            replay_policy=replay_policy,
            latency_profile=self.config.latency_profile,
        )
        return result.results

    def emit_edge(
        self,
        *,
        src_node: str,
        dst_node: str,
        artifact_type: str,
        content: str,
        transfer_type: str,
        round_id: int | None = None,
        manager_round_id: int | None = None,
        peer_round_id: int | None = None,
        parallel_group: str | None = None,
        fanout_count: int = 1,
        recipient_count: int = 1,
        duplicated_from_artifact_id: str | None = None,
    ) -> str:
        artifact_id = f"artifact_{stable_hash(src_node + dst_node + content)[:12]}"
        self.trace.emit(
            event_type="edge_dataflow",
            node_id=f"{src_node}->{dst_node}",
            node_name=f"{src_node}->{dst_node}",
            node_type="edge",
            round_id=round_id,
            manager_round_id=manager_round_id,
            peer_round_id=peer_round_id,
            motif_tags=self.motif_tags,
            parallel_group=parallel_group,
            src_node=src_node,
            dst_node=dst_node,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            artifact_version=1,
            artifact_hash=stable_hash(content),
            artifact_tokens_est=estimate_tokens(content),
            token_count_source=token_count_source(),
            is_shared_context=transfer_type in {"broadcast", "prompt_inclusion"},
            duplicated_from_artifact_id=duplicated_from_artifact_id,
            transfer_type=transfer_type,
            fanout_count=fanout_count,
            recipient_count=recipient_count,
        )
        return artifact_id

    def barrier(self, *, barrier_id: str, waiting_for_nodes: list[str], round_id: int | None = None, manager_round_id: int | None = None, peer_round_id: int | None = None) -> None:
        start = time.time()
        end = time.time()
        self.trace.emit(
            event_type="barrier",
            node_id=barrier_id,
            node_name=barrier_id,
            node_type="barrier",
            round_id=round_id,
            manager_round_id=manager_round_id,
            peer_round_id=peer_round_id,
            motif_tags=self.motif_tags,
            duration_sec=0.0,
            duration_source="measured_live",
            barrier_id=barrier_id,
            waiting_for_nodes=waiting_for_nodes,
            arrived_nodes=waiting_for_nodes,
            start_time=start,
            end_time=end,
            barrier_wait_sec=0.0,
            straggler_node=waiting_for_nodes[-1] if waiting_for_nodes else "",
            straggler_gap_sec=0.0,
        )

    def build_summary(self, final_answer: str) -> dict[str, Any]:
        events = self.trace.events
        total_llm = sum(float(e.get("duration_sec") or 0) for e in events if e.get("event_type") == "llm_request_end")
        tool_events = [e for e in events if str(e.get("event_type", "")).startswith("tool_")]
        total_tool = sum(float(e.get("effective_duration_sec") or e.get("duration_sec") or 0) for e in tool_events)
        input_tokens = sum(int(e.get("input_tokens_est") or 0) for e in events if e.get("event_type") == "llm_request_end")
        output_tokens = sum(int(e.get("output_tokens_est") or 0) for e in events if e.get("event_type") == "llm_request_end")
        artifact_tokens = sum(int(e.get("artifact_tokens_est") or 0) for e in events if e.get("node_type") == "edge")
        peer_tokens = sum(int(e.get("artifact_tokens_est") or 0) for e in events if e.get("artifact_type") == "peer_message")
        aggregation_tokens = sum(int(e.get("artifact_tokens_est") or 0) for e in events if e.get("artifact_type") in {"summary", "final"})
        groups = {e.get("parallel_group") for e in events if e.get("parallel_group")}
        summary = {
            "trace_path": str(self.trace.trace_path),
            "run_id": self.config.run_id,
            "topology": self.config.topology_name,
            "instance_id": self.config.instance_id,
            "end_to_end_latency": round(time.perf_counter() - self.trace.start_perf, 6),
            "total_llm_time": round(total_llm, 6),
            "total_tool_time": round(total_tool, 6),
            "measured_tool_time": round(sum(float(e.get("measured_duration_sec") or 0) for e in tool_events), 6),
            "injected_tool_delay_time": round(sum(float(e.get("injected_delay_sec") or 0) for e in tool_events), 6),
            "total_queue_wait": sum(float(e.get("queue_wait_sec") or 0) for e in events),
            "total_input_tokens_est": input_tokens,
            "total_output_tokens_est": output_tokens,
            "total_tokens_est": input_tokens + output_tokens,
            "total_artifact_tokens_est": artifact_tokens,
            "max_parallel_width": self.config.num_agents,
            "critical_path_length": round(total_llm + total_tool, 6),
            "barrier_wait_sum": sum(float(e.get("barrier_wait_sec") or 0) for e in events),
            "context_duplication_ratio": round(input_tokens / max(1, estimate_tokens(self.config.query)), 4),
            "all_gather_tokens_est": peer_tokens,
            "aggregation_tokens_est": aggregation_tokens,
            "peer_message_tokens_est": peer_tokens,
            "broadcast_tokens_est": sum(int(e.get("artifact_tokens_est") or 0) for e in events if e.get("transfer_type") == "broadcast"),
            "debate_rounds_actual": max([int(e.get("peer_round_id") or 0) for e in events] or [0]),
            "manager_rounds_actual": max([int(e.get("manager_round_id") or 0) for e in events] or [0]) + (1 if any(e.get("manager_round_id") == 0 for e in events) else 0),
            "peer_rounds_actual": max([int(e.get("peer_round_id") or 0) for e in events] or [0]),
            "retry_count_total": sum(int(e.get("retry_count") or 0) for e in events),
            "straggler_gap_by_parallel_group": {str(g): 0.0 for g in groups},
            "tool_time_by_tool_name": {},
            "tool_stall_events": len([e for e in tool_events if float(e.get("effective_duration_sec") or 0) > 0]),
            "critical_queue_wait_time": 0.0,
            "max_ready_queue_size": 0,
            "dispatch_policy": self.config.dispatch_policy,
            "slot_utilization_estimate": 0.0,
            "final_answer_hash": stable_hash(final_answer),
            "event_count": len(events),
        }
        for event in tool_events:
            name = str(event.get("tool_name") or "tool")
            summary["tool_time_by_tool_name"][name] = summary["tool_time_by_tool_name"].get(name, 0.0) + float(event.get("effective_duration_sec") or 0)
        summary.update(self.topology_summary(events))
        return summary

    def topology_summary(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        return {}

    def _config_dict(self) -> dict[str, Any]:
        data = dict(self.config.__dict__)
        for key in {"trace_dir", "repo_path", "replay_snapshot_dir"}:
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data


def parse_json_maybe(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {"text": text}

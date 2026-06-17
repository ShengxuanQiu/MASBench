"""Unified topology interface and shared runtime helpers."""

from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..llm_backends import MockLLM, OpenAICompatibleLLM
from ..search_providers import BaseSearchProvider, record_search_event
from ..trace_export import export_trace_views
from ..tracing import TraceContext, estimate_tokens, stable_hash, token_count_source
from ..backend_adapters import BackendTraceAdapter, build_backend_trace_adapter
from ..backend_metrics import BackendMetricsSampler, metrics_url_from_base_url, summarize_backend_metrics


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
    max_concurrent_llm_calls: int = 32
    dispatch_policy: str = "fcfs"
    backend_base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "local-mas-model"
    max_output_tokens: int = 4096
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
    allow_synthetic_tools: bool = False
    agent_execution: str = "fixed"
    react_max_steps: int = 4
    trace_level: str = "arch"
    export_trace_views: bool = True
    record_model_outputs: bool = False
    collect_backend_metrics: bool = False
    backend_metrics_url: str = ""
    backend_metrics_interval_sec: float = 0.5
    backend_trace_adapter: str = "vllm_gpu"
    agent_pool_size: int = 8
    min_selected_agents: int = 1
    max_selected_agents: int = 3
    orchestrator_stop_confidence: float = 0.78
    workload_name: str = ""
    tool_branch_width: int = 2
    controlled_tool_delay_sec: float = 3.0
    resume_phase_policy: str = "overlap_reviewer"
    critical_stage_marker: str = "reviewer"
    background_resume_enabled: bool = True
    contention_labeling: bool = True
    tool_trace_replay_path: Path | None = None
    group_count: int = 2
    agents_per_group: int = 2
    writer_count: int = 2
    reader_count: int = 2
    mode: str = "topology"
    motif_name: str = ""
    parent_motif_id: str = ""
    composed_from_topologies: list[str] = field(default_factory=list)
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
        self.backend_metrics_sampler: BackendMetricsSampler | None = None
        self.backend_trace_adapter: BackendTraceAdapter = build_backend_trace_adapter(config.backend_trace_adapter)

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        old_query = self.config.query
        self.config.query = str(payload.get("query") or old_query)
        try:
            return self.run()
        finally:
            self.config.query = old_query

    def workflow_start(self) -> None:
        self._start_backend_metrics_sampler()
        self.trace.emit(
            event_type="workflow_start",
            workflow_name=self.config.workload_name or self.config.motif_name or self.config.topology_name,
            node_id="START",
            node_name="START",
            node_type="workflow",
            motif_tags=self.motif_tags,
            status="start",
            extra={"query_hash": stable_hash(self.config.query), "config": self._config_dict()},
        )

    def workflow_end(self, final_answer: str) -> dict[str, Any]:
        self._annotate_request_overlap_metrics()
        self.trace.emit(
            event_type="workflow_end",
            workflow_name=self.config.workload_name or self.config.motif_name or self.config.topology_name,
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
        self._stop_backend_metrics_sampler()
        summary = self.build_summary(final_answer)
        self.trace.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace.save()
        export_paths = export_trace_views(self.trace.events, self.trace.trace_path, enabled=self.config.export_trace_views)
        summary["trace_exports"] = export_paths
        summary["trace_level"] = self.config.trace_level
        self.trace.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return summary

    def _start_backend_metrics_sampler(self) -> None:
        if not self.config.collect_backend_metrics or self.config.llm_mode != "openai_compatible":
            return
        url = self.config.backend_metrics_url or metrics_url_from_base_url(self.config.backend_base_url)
        path = self.trace.backend_metrics_path or self.trace.trace_path.with_name(f"{self.trace.trace_path.stem}_backend_metrics.json")
        sampler = BackendMetricsSampler(
            url=url,
            output_path=path,
            interval_sec=max(0.1, float(self.config.backend_metrics_interval_sec or 0.5)),
            start_perf=self.trace.start_perf,
            adapter=self.backend_trace_adapter,
        )
        self.backend_metrics_sampler = sampler
        sampler.start()
        self.trace.emit(
            event_type="backend_metrics_start",
            node_id="backend_metrics_sampler",
            node_name="Backend Metrics Sampler",
            node_type="dispatcher",
            motif_tags=self.motif_tags,
            status="start",
            backend_metrics_url=url,
            backend_metrics_path=str(path),
            backend_metrics_interval_sec=sampler.interval_sec,
        )
        self.trace.emit(
            event_type="backend_device",
            node_id="backend_device",
            node_name="Backend Device",
            node_type="backend",
            motif_tags=self.motif_tags,
            status="active",
            metrics_source="backend_trace_adapter",
            **self.backend_trace_adapter.backend_device_metadata(),
        )

    def _stop_backend_metrics_sampler(self) -> None:
        if self.backend_metrics_sampler is None:
            return
        self.backend_metrics_sampler.stop()
        for sample in self.backend_metrics_sampler.samples:
            metrics = sample.get("metrics") or {}
            def latest(*needles: str) -> float | None:
                values = [float(value) for name, value in metrics.items() if any(needle in name for needle in needles)]
                return max(values) if values else None

            def config_label(label: str) -> float | None:
                marker = f'{label}="'
                for name in metrics:
                    if marker not in name:
                        continue
                    value = name.split(marker, 1)[1].split('"', 1)[0]
                    try:
                        return float(value)
                    except ValueError:
                        return None
                return None

            usage = latest("kv_cache_usage_perc", "gpu_cache_usage_perc", "gpu_cache_usage")
            blocks = config_label("num_gpu_blocks")
            block_size = config_label("block_size")
            self.trace.emit(
                event_type="backend_metric_sample",
                node_id="vllm_backend",
                node_name="vLLM Backend Metric Sample",
                node_type="backend",
                status=sample.get("status", "error"),
                relative_time_sec=sample.get("relative_time_sec"),
                kv_cache_usage_perc=usage,
                num_requests_running=latest("num_requests_running", "num_running_requests"),
                num_requests_waiting=latest("num_requests_waiting", "num_waiting_requests"),
                num_gpu_blocks=blocks,
                kv_block_size_tokens=block_size,
                kv_cache_used_blocks=(usage * blocks) if usage is not None and blocks is not None else None,
                kv_cache_used_tokens_capacity=(usage * blocks * block_size) if usage is not None and blocks is not None and block_size is not None else None,
                metrics_source="vllm_prometheus_endpoint",
                evidence_scope="aggregate_server_observed",
                error=sample.get("error", ""),
            )
            serving_metrics = sample.get("serving_metrics") or self.backend_trace_adapter.serving_sample(metrics)
            self.trace.emit(
                event_type="serving_metric_sample",
                node_id="serving_backend",
                node_name="Serving Metric Sample",
                node_type="backend",
                status=sample.get("status", "error"),
                relative_time_sec=sample.get("relative_time_sec"),
                metrics_source="backend_trace_adapter",
                backend_runtime=self.backend_trace_adapter.runtime,
                device_type=self.backend_trace_adapter.device_type,
                **serving_metrics,
            )
            cache_metrics = sample.get("cache_memory_metrics") or self.backend_trace_adapter.cache_memory_sample(metrics)
            self.trace.emit(
                event_type="cache_memory_sample",
                node_id="cache_memory_backend",
                node_name="Cache/Memory Metric Sample",
                node_type="backend",
                status=sample.get("status", "error"),
                relative_time_sec=sample.get("relative_time_sec"),
                metrics_source="backend_trace_adapter",
                backend_runtime=self.backend_trace_adapter.runtime,
                device_type=self.backend_trace_adapter.device_type,
                **cache_metrics,
            )
            device_metrics = sample.get("device_metrics") or self.backend_trace_adapter.collect_device_metrics()
            device_payload = {key: value for key, value in device_metrics.items() if key not in {"status", "source"}}
            self.trace.emit(
                event_type="device_metric_sample",
                node_id="xpu_device",
                node_name="xPU Device Metric Sample",
                node_type="backend",
                status=device_metrics.get("status", sample.get("status", "error")),
                relative_time_sec=sample.get("relative_time_sec"),
                metrics_source=device_metrics.get("source", "backend_trace_adapter"),
                backend_runtime=self.backend_trace_adapter.runtime,
                device_type=self.backend_trace_adapter.device_type,
                **device_payload,
            )
        summary = summarize_backend_metrics(self.backend_metrics_sampler.samples)
        self.trace.backend_metrics_summary = summary
        self.trace.emit(
            event_type="backend_metrics_end",
            node_id="backend_metrics_sampler",
            node_name="Backend Metrics Sampler",
            node_type="dispatcher",
            motif_tags=self.motif_tags,
            status="success" if summary.get("backend_metrics_sample_count", 0) else "error",
            backend_metrics_url=self.backend_metrics_sampler.url,
            backend_metrics_path=str(self.backend_metrics_sampler.output_path),
            backend_metrics_summary=summary,
        )

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
        retry_count: int = 0,
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
            "workflow_name": self.config.workload_name or self.config.motif_name or self.config.topology_name,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        trace_metadata = {
            key: value
            for key, value in (extra_metadata or {}).items()
            if key
            not in {
                "agent_id",
                "agent_role",
                "round_id",
                "manager_round_id",
                "peer_round_id",
                "retry_count",
                "parents",
                "children",
                "criticality",
                "parallel_group",
                "status",
                "duration_sec",
                "duration_source",
                "node_id",
                "node_name",
                "node_type",
                "event_type",
                "workflow_name",
                "parent_node_ids",
                "dependency_edges",
                "fan_in_count",
                "fan_out_count",
                "model_name",
                "request_submit_ts",
                "response_start_ts",
                "response_end_ts",
            }
        }
        prompt = system_prompt + "\n" + user_prompt
        submit_ts = time.time()
        common_graph = {
            "workflow_name": self.config.workload_name or self.config.motif_name or self.config.topology_name,
            "parent_node_ids": parents or [],
            "dependency_edges": trace_metadata.get("dependency_edges") or [{"src": parent, "dst": node_id} for parent in (parents or [])],
            "fan_in_count": len(parents or []),
            "fan_out_count": int(trace_metadata.get("fan_out_count") or 0),
        }
        self.trace.emit(
            event_type="llm_request_start",
            node_id=node_id,
            node_name=node_name,
            node_type="llm",
            round_id=round_id,
            manager_round_id=manager_round_id,
            peer_round_id=peer_round_id,
            retry_count=retry_count,
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
            max_output_tokens=self.config.max_output_tokens,
            llm_request_id=str(uuid.uuid4()),
            input_chars=len(prompt),
            input_tokens_est=estimate_tokens(prompt),
            system_prompt_tokens_est=estimate_tokens(system_prompt),
            user_prompt_tokens_est=estimate_tokens(user_prompt),
            token_count_source=token_count_source(),
            shared_context_hash=stable_hash(user_prompt)[:16],
            prompt_hash=stable_hash(prompt),
            prompt_segment_hashes={
                "system_prompt": stable_hash(system_prompt)[:16],
                "user_prompt": stable_hash(user_prompt)[:16],
                "shared_context": stable_hash(user_prompt)[:16],
                "prompt_template": stable_hash(prompt_template)[:16],
            },
            priority=1.0 if criticality == "critical" else 0.0,
            dispatch_policy=self.config.dispatch_policy,
            request_metadata=metadata,
            request_submit_ts=submit_ts,
            start_ts=submit_ts,
            input_tokens=estimate_tokens(prompt),
            model_name=self.config.model,
            extra={"prompt_preview": prompt[:1000]} if self.config.trace_level == "detailed" else {},
            **common_graph,
            **trace_metadata,
        )
        result = self.llm.invoke(system_prompt, user_prompt, metadata)
        output = result.content
        event = self.trace.emit(
            event_type="llm_request_end",
            node_id=node_id,
            node_name=node_name,
            node_type="llm",
            round_id=round_id,
            manager_round_id=manager_round_id,
            peer_round_id=peer_round_id,
            retry_count=retry_count,
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
            max_output_tokens=self.config.max_output_tokens,
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
            prompt_segment_hashes={
                "system_prompt": stable_hash(system_prompt)[:16],
                "user_prompt": stable_hash(user_prompt)[:16],
                "shared_context": stable_hash(user_prompt)[:16],
                "prompt_template": stable_hash(prompt_template)[:16],
            },
            output_hash=stable_hash(output),
            priority=1.0 if criticality == "critical" else 0.0,
            dispatch_policy=self.config.dispatch_policy,
            request_metadata=result.request_metadata,
            backend_prompt_tokens=result.request_metadata.get("backend_prompt_tokens"),
            backend_completion_tokens=result.request_metadata.get("backend_completion_tokens"),
            backend_total_tokens=result.request_metadata.get("backend_total_tokens"),
            backend_finish_reason=result.request_metadata.get("backend_finish_reason"),
            backend_response_id=result.request_metadata.get("backend_response_id"),
            request_submit_ts=submit_ts,
            response_start_ts=result.generation_start_ts,
            response_end_ts=result.generation_end_ts,
            start_ts=submit_ts,
            end_ts=result.generation_end_ts,
            request_e2e_sec=round(result.duration_sec + float(result.queue_wait_sec or 0.0), 6),
            ttft_sec=result.request_metadata.get("ttft_sec"),
            tpot_sec=result.request_metadata.get("tpot_sec"),
            input_tokens=estimate_tokens(prompt),
            output_tokens=estimate_tokens(output),
            total_tokens=estimate_tokens(prompt) + estimate_tokens(output),
            model_name=self.config.model,
            nearby_background_request_count_1s=0,
            nearby_background_request_count_3s=0,
            overlapping_background_request_count=0,
            overlapping_background_tokens=0,
            extra={"output_preview": output[:1000]} if self.config.trace_level == "detailed" else {},
            **common_graph,
            **trace_metadata,
        )
        self.trace.record_model_output(
            event=event,
            output_text=output,
            input_text=prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_metadata=result.request_metadata,
            extra={"source": "fixed_llm_call"},
        )
        return output

    def search(self, *, node_id: str, node_name: str, query: str, trace_fields: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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
            trace_fields={
                "workflow_name": self.config.workload_name or self.config.motif_name or self.config.topology_name,
                "parent_node_ids": (trace_fields or {}).get("parent_node_ids") or (trace_fields or {}).get("parents") or [],
                "dependency_edges": (trace_fields or {}).get("dependency_edges") or [],
                "tool_trace_source": self.config.tool_mode,
                **(trace_fields or {}),
            },
        )
        return result.results

    def controlled_delay_tool(
        self,
        *,
        node_id: str,
        node_name: str,
        configured_delay_sec: float,
        delay_mode: str | None = None,
        trace_fields: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emit a deterministic tool-delay event without requiring a live tool."""
        configured = max(0.0, float(configured_delay_sec or 0.0))
        mode = delay_mode or ("actual" if self.config.llm_mode == "openai_compatible" and self.config.tool_mode == "live" else "simulated")
        start_wall = time.time()
        start_rel = time.perf_counter() - self.trace.start_perf
        if mode == "actual" and configured > 0:
            time.sleep(configured)
        observed = time.time() - start_wall if mode == "actual" else configured
        end_wall = time.time() if mode == "actual" else None
        event = self.trace.emit(
            event_type="tool_controlled_delay",
            node_id=node_id,
            node_name=node_name,
            node_type="tool",
            status="success",
            duration_sec=round(observed, 6),
            duration_source="measured_live" if mode == "actual" else "simulated_duration",
            replay_policy="controlled_delay_tool",
            tool_name="controlled_delay_tool",
            tool_mode=self.config.tool_mode,
            configured_delay_sec=round(configured, 6),
            observed_or_simulated_delay_sec=round(observed, 6),
            observed_tool_delay_sec=round(observed, 6),
            measured_duration_sec=round(observed if mode == "actual" else 0.0, 6),
            injected_delay_sec=round(0.0 if mode == "actual" else configured, 6),
            effective_duration_sec=round(observed, 6),
            delay_mode=mode,
            tool_start_ts=start_wall,
            tool_end_ts=end_wall or (start_wall + observed),
            tool_latency_sec=round(observed, 6),
            tool_return_ts=end_wall or (start_wall + observed),
            simulated_tool_return_ts=round(start_rel + configured, 6) if mode == "simulated" else None,
            tool_trace_source="live" if mode == "actual" else "replay",
            external_dependency="none",
            network_dependent=False,
            deterministic=True,
            **(trace_fields or {}),
        )
        return event

    def use_react_agents(self) -> bool:
        return self.config.agent_execution == "react"

    def call_agent(
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
        retry_count: int = 0,
        parents: list[str] | None = None,
        parallel_group: str | None = None,
        criticality: str = "unknown",
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        if not self.use_react_agents():
            return self.call_llm(
                node_id=node_id,
                node_name=node_name,
                node_type=node_type,
                agent_role=agent_role,
                prompt_template=prompt_template,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                round_id=round_id,
                manager_round_id=manager_round_id,
                peer_round_id=peer_round_id,
                retry_count=retry_count,
                parents=parents,
                parallel_group=parallel_group,
                criticality=criticality,
                extra_metadata=extra_metadata,
            )
        if self.config.llm_mode != "openai_compatible":
            raise ValueError("--agent-execution react currently requires --llm-mode openai_compatible")
        if self.config.tool_mode == "synthetic" and not self.config.allow_synthetic_tools:
            raise ValueError("--agent-execution react requires real/replay tools unless --allow-synthetic-tools true")
        from ..langgraph_agents.react_agent import run_react_agent

        return run_react_agent(
            config=self.config,
            trace=self.trace,
            search_provider=self.search_provider,
            motif_tags=self.motif_tags,
            node_id=node_id,
            node_name=node_name,
            node_type=node_type,
            agent_role=agent_role,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            round_id=round_id,
            manager_round_id=manager_round_id,
            peer_round_id=peer_round_id,
            parents=parents or [],
            parallel_group=parallel_group,
            criticality=criticality,
            extra_metadata=extra_metadata or {},
        )

    def run_parallel(self, items: list[Any], fn: Any) -> list[Any]:
        """Run independent topology branches concurrently and preserve item order."""
        if not items:
            return []
        if not self.config.allow_parallel_workers or len(items) == 1:
            return [fn(item) for item in items]
        max_workers = max(1, min(int(self.config.max_concurrent_llm_calls or 1), len(items)))
        results: list[Any] = [None] * len(items)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fn, item): idx for idx, item in enumerate(items)}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return results

    def agent_pool(self, *, prefix: str = "agent") -> list[dict[str, str]]:
        """Candidate specialist pool for orchestrator-driven topologies."""
        templates = [
            ("repo_search", "Repository Searcher", "find relevant files, symbols, and implementation locations"),
            ("issue_triage", "Issue Triage Analyst", "extract bug symptoms, constraints, and expected behavior"),
            ("code_localization", "Code Localizer", "map evidence to likely modules and functions"),
            ("api_doc", "API Documentation Analyst", "check public APIs, docs, and semantic contracts"),
            ("test_reasoning", "Test Reasoning Agent", "infer regression tests and failure modes"),
            ("patch_planning", "Patch Planner", "propose minimal implementation changes"),
            ("regression_risk", "Regression Risk Analyst", "identify compatibility and edge-case risks"),
            ("performance", "Performance Analyst", "reason about runtime, memory, and scaling impact"),
            ("tool_heavy", "Tool-Heavy Evidence Agent", "aggressively call tools to gather external evidence"),
            ("judge", "Judge / Synthesizer", "compare competing evidence and recommend stop or continue"),
            ("security", "Security Reviewer", "check unsafe behavior and security-adjacent regressions"),
            ("maintainer", "Maintainer Perspective Agent", "evaluate change scope and repository maintainability"),
        ]
        limit = max(1, min(int(self.config.agent_pool_size or len(templates)), len(templates)))
        pool = []
        for role, name, capability in templates[:limit]:
            pool.append(
                {
                    "id": f"{prefix}_{role}",
                    "role": role,
                    "name": name,
                    "capability": capability,
                    "system_prompt": f"You are {name}. Your specialty is to {capability}. Return concise evidence with confidence.",
                }
            )
        return pool

    def clamp_selected_agents(self, selected: list[str], pool: list[dict[str, str]]) -> list[str]:
        valid = [agent["id"] for agent in pool]
        seen: set[str] = set()
        cleaned = []
        for agent_id in selected:
            if agent_id in valid and agent_id not in seen:
                cleaned.append(agent_id)
                seen.add(agent_id)
        min_count = max(1, int(self.config.min_selected_agents or 1))
        max_count = max(min_count, int(self.config.max_selected_agents or min_count))
        for agent_id in valid:
            if len(cleaned) >= min_count:
                break
            if agent_id not in seen:
                cleaned.append(agent_id)
                seen.add(agent_id)
        return cleaned[:max_count]

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
        retry_count: int = 0,
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
            retry_count=retry_count,
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
        arrivals: dict[str, float] = {}
        for event in self.trace.events:
            node = str(event.get("node_id") or "")
            if node in waiting_for_nodes and str(event.get("event_type", "")).endswith("_end"):
                arrivals[node] = float(event.get("relative_time_sec") or 0.0)
        if arrivals:
            latest_node, latest = max(arrivals.items(), key=lambda item: item[1])
            earliest = min(arrivals.values())
            straggler_gap = max(0.0, latest - earliest)
        else:
            latest_node = waiting_for_nodes[-1] if waiting_for_nodes else ""
            straggler_gap = 0.0
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
            arrived_nodes=list(arrivals) or waiting_for_nodes,
            start_time=start,
            end_time=end,
            barrier_wait_sec=round(straggler_gap, 6),
            straggler_node=latest_node,
            straggler_gap_sec=round(straggler_gap, 6),
        )


    def _event_start(self, event: dict[str, Any]) -> float:
        return max(0.0, float(event.get("relative_time_sec") or 0.0) - float(event.get("duration_sec") or 0.0))

    def _annotate_request_overlap_metrics(self) -> None:
        llm = [e for e in self.trace.events if e.get("event_type") == "llm_request_end"]
        background = [e for e in llm if e.get("background_resume_request") or e.get("criticality") in {"background", "non_critical"}]
        critical = [e for e in llm if e.get("critical_path_candidate") or e.get("criticality") == "critical"]
        for event in llm:
            start = self._event_start(event)
            end = float(event.get("relative_time_sec") or 0.0)
            nearby_1 = []
            nearby_3 = []
            overlapping = []
            for bg in background:
                if bg is event:
                    continue
                bg_start = self._event_start(bg)
                bg_end = float(bg.get("relative_time_sec") or 0.0)
                if abs(bg_start - start) <= 1.0:
                    nearby_1.append(bg)
                if abs(bg_start - start) <= 3.0:
                    nearby_3.append(bg)
                if bg_start < end and bg_end > start:
                    overlapping.append(bg)
            event["nearby_background_request_count_1s"] = len(nearby_1)
            event["nearby_background_request_count_3s"] = len(nearby_3)
            event["overlapping_background_request_count"] = len(overlapping)
            event["overlapping_background_tokens"] = sum(int(e.get("total_tokens") or e.get("total_tokens_est") or 0) for e in overlapping)
            if event.get("background_resume_request"):
                event["actual_overlap_with_critical"] = any(self._event_start(c) < end and float(c.get("relative_time_sec") or 0.0) > start for c in critical)

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
        barriers = [e for e in events if e.get("node_type") == "barrier"]
        summary = {
            "trace_path": str(self.trace.trace_path),
            "run_id": self.config.run_id,
            "mode": self.config.mode,
            "motif_name": self.config.motif_name,
            "motif_instance_id": self.config.instance_id if self.config.mode == "motif" else "",
            "parent_motif_id": self.config.parent_motif_id,
            "composed_from_topologies": list(self.config.composed_from_topologies),
            "topology": self.config.topology_name,
            "instance_id": self.config.instance_id,
            "end_to_end_latency": round(time.perf_counter() - self.trace.start_perf, 6),
            "total_llm_time": round(total_llm, 6),
            "backend_prompt_tokens": sum(int(e.get("backend_prompt_tokens") or 0) for e in events if e.get("event_type") == "llm_request_end"),
            "backend_completion_tokens": sum(int(e.get("backend_completion_tokens") or 0) for e in events if e.get("event_type") == "llm_request_end"),
            "backend_total_tokens": sum(int(e.get("backend_total_tokens") or 0) for e in events if e.get("event_type") == "llm_request_end"),
            "total_tool_time": round(total_tool, 6),
            "measured_tool_time": round(sum(float(e.get("measured_duration_sec") or 0) for e in tool_events), 6),
            "injected_tool_delay_time": round(sum(float(e.get("injected_delay_sec") or 0) for e in tool_events), 6),
            "total_queue_wait": sum(float(e.get("queue_wait_sec") or 0) for e in events),
            "total_input_tokens_est": input_tokens,
            "total_output_tokens_est": output_tokens,
            "total_tokens_est": input_tokens + output_tokens,
            "total_artifact_tokens_est": artifact_tokens,
            "max_parallel_width": max(
                [int(e.get("selected_agent_count") or 0) for e in events if e.get("event_type") in {"manager_decision", "orchestrator_decision"}]
                or [self.config.num_agents]
            ),
            "critical_path_length": round(total_llm + total_tool, 6),
            "barrier_wait_sum": sum(float(e.get("barrier_wait_sec") or 0) for e in barriers),
            "context_duplication_ratio": round(input_tokens / max(1, estimate_tokens(self.config.query)), 4),
            "all_gather_tokens_est": peer_tokens,
            "aggregation_tokens_est": aggregation_tokens,
            "peer_message_tokens_est": peer_tokens,
            "broadcast_tokens_est": sum(int(e.get("artifact_tokens_est") or 0) for e in events if e.get("transfer_type") == "broadcast"),
            "debate_rounds_actual": max([int(e.get("peer_round_id") or 0) for e in events] or [0]),
            "manager_rounds_actual": max([int(e.get("manager_round_id") or 0) for e in events] or [0]) + (1 if any(e.get("manager_round_id") == 0 for e in events) else 0),
            "peer_rounds_actual": max([int(e.get("peer_round_id") or 0) for e in events] or [0]),
            "retry_count_total": sum(int(e.get("retry_count") or 0) for e in events),
            "straggler_gap_by_parallel_group": {str(e.get("barrier_id") or e.get("node_id")): float(e.get("straggler_gap_sec") or 0) for e in barriers},
            "tool_time_by_tool_name": {},
            "tool_stall_events": len([e for e in tool_events if float(e.get("effective_duration_sec") or 0) > 0]),
            "critical_queue_wait_time": 0.0,
            "max_ready_queue_size": 0,
            "dispatch_policy": self.config.dispatch_policy,
            "slot_utilization_estimate": 0.0,
            "final_answer_hash": stable_hash(final_answer),
            "event_count": len(events),
            "model_outputs_path": str(self.trace.model_outputs_path or self.trace.trace_path.with_name(f"{self.trace.trace_path.stem}_model_outputs.json")) if self.config.record_model_outputs else None,
            "model_output_record_count": len(self.trace.model_outputs),
            "backend_metrics_path": str(self.trace.backend_metrics_path or self.trace.trace_path.with_name(f"{self.trace.trace_path.stem}_backend_metrics.json")) if self.config.collect_backend_metrics else None,
        }
        if self.trace.backend_metrics_summary:
            summary.update(self.trace.backend_metrics_summary)
        for event in tool_events:
            name = str(event.get("tool_name") or "tool")
            summary["tool_time_by_tool_name"][name] = summary["tool_time_by_tool_name"].get(name, 0.0) + float(event.get("effective_duration_sec") or 0)
        summary.update(self.topology_summary(events))
        return summary

    def topology_summary(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        return {}

    def _config_dict(self) -> dict[str, Any]:
        data = dict(self.config.__dict__)
        for key in {"trace_dir", "repo_path", "replay_snapshot_dir", "tool_trace_replay_path"}:
            if data.get(key) is not None:
                data[key] = str(data[key])
        return data


def parse_json_maybe(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        return {"text": text}

"""Real/replay tools wrapped with MASBench-Arch trace events."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..search_providers import BaseSearchProvider, record_search_event
from ..tracing import TraceContext, stable_hash


class TracedToolRuntime:
    def __init__(
        self,
        *,
        config: Any,
        trace: TraceContext,
        search_provider: BaseSearchProvider,
        agent_id: str,
        agent_role: str,
        round_id: int | None,
        manager_round_id: int | None,
        peer_round_id: int | None,
        parallel_group: str | None,
        motif_tags: list[str],
    ) -> None:
        self.config = config
        self.trace = trace
        self.search_provider = search_provider
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.round_id = round_id
        self.manager_round_id = manager_round_id
        self.peer_round_id = peer_round_id
        self.parallel_group = parallel_group
        self.motif_tags = motif_tags
        self._lock = threading.Lock()
        self._counter = 0

    def search(self, query: str) -> str:
        with self._lock:
            self._counter += 1
            call_index = self._counter
        node_id = f"{self.agent_id}_tool_search_{call_index}"
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
            node_name=f"{self.agent_id}.search[{call_index}]",
            tool_mode=self.config.tool_mode,
            result=result,
            snapshot_dir=snapshot_dir,
            replay_policy=replay_policy,
            latency_profile=self.config.latency_profile,
            trace_fields={
                "round_id": self.round_id,
                "manager_round_id": self.manager_round_id,
                "peer_round_id": self.peer_round_id,
                "parallel_group": self.parallel_group,
                "motif_tags": self.motif_tags,
                "agent_id": self.agent_id,
                "agent_role": self.agent_role,
                "tool_call_id": f"{self.agent_id}:search:{call_index}",
            },
        )
        payload = {
            "answer": result.answer,
            "results": result.results,
            "result_hash": result.result_hash,
            "provider_name": result.provider_name,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def make_search_tools(runtime: TracedToolRuntime) -> list[Any]:
    """Create LangChain tools lazily so base installs without LangChain still import."""
    from langchain_core.tools import StructuredTool

    def web_or_repo_search(query: str) -> str:
        """Search for external web evidence or repository evidence relevant to the task."""
        return runtime.search(query)

    web_or_repo_search.__name__ = "search"
    return [
        StructuredTool.from_function(
            func=web_or_repo_search,
            name="search",
            description=(
                "Search for evidence before answering when the task needs current facts, "
                "repository evidence, issue details, APIs, errors, or confirmation. "
                "Input must be a concise search query."
            ),
        )
    ]


def tool_input_hash(tool_name: str, args: dict[str, Any]) -> str:
    return stable_hash({"tool_name": tool_name, "args": args})


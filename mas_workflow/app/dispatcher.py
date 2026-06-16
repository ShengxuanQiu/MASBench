"""统一 LLM 调度器。

第一版调度器运行在 workflow 上层，不修改 vLLM 内部源码。所有 agent 的 LLM
请求都会先进入本地 ready queue，再由调度策略决定何时提交给 vLLM endpoint。
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .llm_client import LocalLLMClient
from .tracing import make_event


@dataclass
class DispatcherResponse:
    """LLMDispatcher 返回给 agent node 的结果。"""

    content: str
    events: list[dict[str, Any]]
    record: dict[str, Any]


class LLMDispatcher:
    """带 ready queue 和并发上限的 LLM 请求调度器。"""

    def __init__(
        self,
        llm_client: LocalLLMClient,
        *,
        workflow_id: str,
        max_concurrent_llm_calls: int = 32,
        dispatch_policy: str = "fcfs",
    ) -> None:
        if dispatch_policy not in {"fcfs", "criticality"}:
            raise ValueError("dispatch_policy 只支持 fcfs 或 criticality")
        self.llm_client = llm_client
        self.workflow_id = workflow_id
        self.max_concurrent_llm_calls = max_concurrent_llm_calls
        self.dispatch_policy = dispatch_policy
        self._condition = threading.Condition()
        self._ready_queue: list[dict[str, Any]] = []
        self._running: dict[str, dict[str, Any]] = {}
        self._sequence = 0

    def submit(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        metadata: dict[str, Any],
        priority: float,
        max_tokens: int | None = None,
    ) -> DispatcherResponse:
        """提交一个 LLM 请求，并阻塞直到请求完成。"""
        request_id = str(uuid.uuid4())
        queue_enter_perf = time.perf_counter()
        metadata = dict(metadata)
        metadata.setdefault("workflow_id", self.workflow_id)
        metadata.setdefault("agent_name", agent_name)
        metadata.setdefault("priority", priority)
        metadata.setdefault("request_id", request_id)

        with self._condition:
            self._sequence += 1
            record = {
                "request_id": request_id,
                "workflow_id": metadata.get("workflow_id"),
                "agent_name": agent_name,
                "node_id": metadata.get("node_id", agent_name),
                "round_id": metadata.get("round_id", 1),
                "criticality": metadata.get("criticality", 0.0),
                "priority": priority,
                "shared_context_hash": metadata.get("shared_context_hash", ""),
                "queue_enter_time": queue_enter_perf,
                "dispatch_time": None,
                "finish_time": None,
                "queue_wait_time": None,
                "service_time": None,
                "status": "queued",
                "sequence": self._sequence,
                "is_critical": bool(metadata.get("is_critical", False)),
            }
            self._ready_queue.append(record)
            queue_event = self._event(
                "llm_queue_enter",
                agent_name,
                user_prompt,
                "",
                0.0,
                metadata,
                record,
            )
            while not self._can_dispatch(record):
                self._condition.wait(timeout=0.05)
            self._ready_queue.remove(record)
            dispatch_perf = time.perf_counter()
            record["dispatch_time"] = dispatch_perf
            record["queue_wait_time"] = dispatch_perf - queue_enter_perf
            record["status"] = "running"
            self._running[request_id] = record
            dispatch_event = self._event(
                "llm_dispatch",
                agent_name,
                user_prompt,
                "",
                record["queue_wait_time"],
                metadata,
                record,
            )
            llm_start_event = self._event(
                "llm_start",
                agent_name,
                user_prompt,
                "",
                0.0,
                metadata,
                record,
            )

        status = "ok"
        output = ""
        error_text = ""
        service_start = time.perf_counter()
        try:
            output = self.llm_client.invoke(
                system_prompt,
                user_prompt,
                metadata=metadata,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            status = "failed"
            error_text = str(exc)
            raise
        finally:
            finish_perf = time.perf_counter()
            service_time = finish_perf - service_start
            with self._condition:
                record["finish_time"] = finish_perf
                record["service_time"] = service_time
                record["status"] = status
                self._running.pop(request_id, None)
                llm_end_event = self._event(
                    "llm_end",
                    agent_name,
                    user_prompt,
                    output,
                    service_time,
                    metadata,
                    record,
                    status=status,
                    extra={"error": error_text} if error_text else None,
                )
                finish_event = self._event(
                    "llm_finish",
                    agent_name,
                    user_prompt,
                    output,
                    service_time,
                    metadata,
                    record,
                    status=status,
                )
                self._condition.notify_all()

        return DispatcherResponse(
            content=output,
            events=[queue_event, dispatch_event, llm_start_event, llm_end_event, finish_event],
            record=dict(record),
        )

    def _can_dispatch(self, record: dict[str, Any]) -> bool:
        if len(self._running) >= self.max_concurrent_llm_calls:
            return False
        if not self._ready_queue:
            return False
        if self.dispatch_policy == "fcfs":
            selected = min(self._ready_queue, key=lambda item: item["sequence"])
        else:
            selected = max(self._ready_queue, key=lambda item: (item["priority"], -item["sequence"]))
        return selected["request_id"] == record["request_id"]

    def _event(
        self,
        event_type: str,
        agent_name: str,
        input_text: str,
        output_text: str,
        duration: float,
        metadata: dict[str, Any],
        record: dict[str, Any],
        *,
        status: str = "ok",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        running_agents = [item["agent_name"] for item in self._running.values()]
        ready_agents = [item["agent_name"] for item in self._ready_queue]
        ready_records = [
            {
                "agent_name": item["agent_name"],
                "request_id": item["request_id"],
                "priority": item.get("priority", 0.0),
                "is_critical": bool(item.get("is_critical", False)),
                "criticality": item.get("criticality", 0.0),
            }
            for item in self._ready_queue
        ]
        running_records = [
            {
                "agent_name": item["agent_name"],
                "request_id": item["request_id"],
                "priority": item.get("priority", 0.0),
                "is_critical": bool(item.get("is_critical", False)),
                "criticality": item.get("criticality", 0.0),
            }
            for item in self._running.values()
        ]
        event_extra = {
            "request_id": record["request_id"],
            "request_metadata": metadata,
            "dispatcher_record": {
                key: value
                for key, value in record.items()
                if key
                in {
                    "request_id",
                    "workflow_id",
                    "agent_name",
                    "node_id",
                    "round_id",
                    "criticality",
                    "priority",
                    "shared_context_hash",
                    "queue_wait_time",
                    "service_time",
                    "status",
                    "is_critical",
                }
            },
            "ready_queue_size": len(self._ready_queue),
            "ready_agents": ready_agents,
            "ready_records": ready_records,
            "running_request_count": len(self._running),
            "running_agents": running_agents,
            "running_records": running_records,
            "max_concurrent_llm_calls": self.max_concurrent_llm_calls,
            "dispatch_policy": self.dispatch_policy,
        }
        if extra:
            event_extra.update(extra)
        return make_event(
            agent_name=agent_name,
            event_type=event_type,
            input_text=input_text,
            output_text=output_text,
            duration_sec=duration,
            status=status,
            extra=event_extra,
        )

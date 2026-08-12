"""Online MAS-aware admission gating for real workflow execution."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from .tracing import TraceContext


@dataclass(frozen=True)
class AdmissionDecision:
    policy: str
    decision: str
    reason: str
    ready_ts: float
    submit_ts: float
    defer_start_ts: float | None = None
    defer_end_ts: float | None = None
    defer_duration_sec: float = 0.0


class AdmissionController:
    """Gate only non-critical tool resumes during an observed critical decode.

    The controller is deliberately online: critical decode state is entered by
    the HTTP streaming first-token callback and left by request completion. It
    never consumes a completed trace or predicted arrival/output information.
    """

    DEFAULT = "default_vllm"
    CRITICAL_PATH_AWARE = "critical_path_aware"

    def __init__(
        self,
        *,
        policy: str,
        max_defer_sec: float,
        trace: TraceContext,
        prefill_budget_tokens: int = 0,
    ) -> None:
        if policy not in {self.DEFAULT, self.CRITICAL_PATH_AWARE}:
            raise ValueError(f"Unsupported admission policy: {policy}")
        self.policy = policy
        self.max_defer_sec = max(0.0, float(max_defer_sec))
        self.prefill_budget_tokens = max(0, int(prefill_budget_tokens))
        self.trace = trace
        self._condition = threading.Condition()
        self._active_critical_decodes: set[str] = set()
        self._active_structural_frontiers: set[str] = set()

    def begin_structural_frontier(self, frontier_id: str) -> None:
        with self._condition:
            self._active_structural_frontiers.add(frontier_id)
        self.trace.emit(
            event_type="critical_frontier_start",
            node_id=frontier_id,
            node_name=frontier_id,
            node_type="admission_controller",
            admission_policy=self.policy,
            frontier_id=frontier_id,
            duration_source="online_runtime_observed",
            replay_policy="live",
        )

    def end_structural_frontier(self, frontier_id: str) -> None:
        with self._condition:
            self._active_structural_frontiers.discard(frontier_id)
            self._condition.notify_all()
        self.trace.emit(
            event_type="critical_frontier_end",
            node_id=frontier_id,
            node_name=frontier_id,
            node_type="admission_controller",
            admission_policy=self.policy,
            frontier_id=frontier_id,
            duration_source="online_runtime_observed",
            replay_policy="live",
        )

    def admit(
        self,
        *,
        request_id: str,
        node_id: str,
        agent_role: str,
        critical_path_candidate: bool,
        tool_resumed: bool,
        ready_ts: float,
        prompt_tokens: int | None = None,
    ) -> AdmissionDecision:
        exceeds_budget = prompt_tokens is None or int(prompt_tokens) > self.prefill_budget_tokens
        should_consider = (
            self.policy == self.CRITICAL_PATH_AWARE
            and not critical_path_candidate
            and tool_resumed
            and exceeds_budget
        )
        defer_start: float | None = None
        timed_out = False
        if should_consider:
            with self._condition:
                if self._active_critical_decodes or self._active_structural_frontiers:
                    defer_start = time.time()
                    deadline = time.monotonic() + self.max_defer_sec
                    self._emit(
                        "admission_defer_start",
                        request_id=request_id,
                        node_id=node_id,
                        agent_role=agent_role,
                        ready_ts=ready_ts,
                        defer_start_ts=defer_start,
                        reason="non_critical_tool_resume_during_active_critical_decode",
                        prompt_tokens=prompt_tokens,
                        prefill_budget_tokens=self.prefill_budget_tokens,
                        active_critical_request_ids=sorted(self._active_critical_decodes),
                        active_structural_frontiers=sorted(self._active_structural_frontiers),
                    )
                    while self._active_critical_decodes or self._active_structural_frontiers:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            timed_out = True
                            break
                        self._condition.wait(timeout=remaining)

        submit_ts = time.time()
        defer_duration = max(0.0, submit_ts - defer_start) if defer_start is not None else 0.0
        if defer_start is None:
            decision = "admitted_immediately"
            reason = "baseline_no_mas_gating" if self.policy == self.DEFAULT else "no_active_critical_decode"
        elif timed_out:
            decision = "admitted_after_max_defer"
            reason = "maximum_defer_time_reached"
        else:
            decision = "admitted_after_defer"
            reason = "critical_decode_completed"
        if defer_start is not None:
            self._emit(
                "admission_defer_end",
                request_id=request_id,
                node_id=node_id,
                agent_role=agent_role,
                ready_ts=ready_ts,
                submit_ts=submit_ts,
                defer_start_ts=defer_start,
                defer_end_ts=submit_ts,
                defer_duration_sec=defer_duration,
                reason=reason,
            )
        self._emit(
            "admission_decision",
            request_id=request_id,
            node_id=node_id,
            agent_role=agent_role,
            ready_ts=ready_ts,
            submit_ts=submit_ts,
            decision=decision,
            reason=reason,
            critical_path_candidate=critical_path_candidate,
            tool_resumed=tool_resumed,
            prompt_tokens=prompt_tokens,
            prefill_budget_tokens=self.prefill_budget_tokens,
            defer_start_ts=defer_start,
            defer_end_ts=submit_ts if defer_start is not None else None,
            defer_duration_sec=defer_duration,
        )
        return AdmissionDecision(
            policy=self.policy,
            decision=decision,
            reason=reason,
            ready_ts=ready_ts,
            submit_ts=submit_ts,
            defer_start_ts=defer_start,
            defer_end_ts=submit_ts if defer_start is not None else None,
            defer_duration_sec=defer_duration,
        )

    def on_first_token(self, *, request_id: str, node_id: str, critical_path_candidate: bool, timestamp: float) -> None:
        if not critical_path_candidate:
            return
        with self._condition:
            self._active_critical_decodes.add(request_id)
        self._emit(
            "critical_decode_start",
            request_id=request_id,
            node_id=node_id,
            first_token_ts=timestamp,
            critical_path_candidate=True,
        )

    def on_complete(self, *, request_id: str, node_id: str, critical_path_candidate: bool, timestamp: float) -> None:
        if not critical_path_candidate:
            return
        was_active = False
        with self._condition:
            if request_id in self._active_critical_decodes:
                was_active = True
                self._active_critical_decodes.remove(request_id)
                self._condition.notify_all()
        if was_active:
            self._emit(
                "critical_decode_end",
                request_id=request_id,
                node_id=node_id,
                completion_ts=timestamp,
                critical_path_candidate=True,
            )

    def _emit(self, event_type: str, *, request_id: str, node_id: str, **fields: Any) -> None:
        self.trace.emit(
            event_type=event_type,
            node_id=node_id,
            node_name=node_id,
            node_type="admission_controller",
            admission_policy=self.policy,
            llm_request_id=request_id,
            request_id_for_backend=request_id,
            duration_source="online_runtime_observed",
            replay_policy="live",
            **fields,
        )

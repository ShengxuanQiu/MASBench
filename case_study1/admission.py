from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Decision:
    decision: str
    reason: str
    ready_ts: float
    submit_ts: float
    prompt_tokens: int
    defer_sec: float = 0.0


class CriticalFrontierAdmission:
    """Online whole-request gating using only ready-time information.

    During a critical frontier, short non-critical prefills may consume a small
    fixed token budget. Larger cold/resume prefills wait until result-ready or
    the maximum defer bound. Deferred drain can be serialized to avoid creating
    a second burst immediately after the protected frontier.
    """

    def __init__(
        self,
        *,
        policy: str,
        prefill_budget_tokens: int,
        max_defer_sec: float,
        serialize_deferred_drain: bool,
        emit: Callable[..., object],
    ) -> None:
        if policy not in {"default_vllm", "critical_frontier"}:
            raise ValueError(f"unsupported policy: {policy}")
        self.policy = policy
        self.prefill_budget_tokens = max(0, int(prefill_budget_tokens))
        self.max_defer_sec = max(0.0, float(max_defer_sec))
        self.serialize_deferred_drain = bool(serialize_deferred_drain)
        self.emit = emit
        self._cv = threading.Condition()
        self._frontier_active = False
        self._frontier_name = ""
        self._remaining_budget = self.prefill_budget_tokens
        self._deferred_fifo: list[str] = []
        self._drain_inflight: str | None = None

    def begin_frontier(self, name: str) -> None:
        with self._cv:
            self._frontier_active = True
            self._frontier_name = name
            self._remaining_budget = self.prefill_budget_tokens
        self.emit(
            "critical_frontier_start",
            frontier=name,
            prefill_budget_tokens=self.prefill_budget_tokens,
        )

    def transition(self, *, completed_node: str, ready_successor: str) -> None:
        with self._cv:
            if not self._frontier_active:
                raise RuntimeError("critical frontier transition without active frontier")
            self._frontier_name = ready_successor
        self.emit(
            "critical_frontier_transition",
            completed_node=completed_node,
            ready_successor=ready_successor,
        )

    def end_frontier(self, name: str) -> None:
        with self._cv:
            self._frontier_active = False
            self._frontier_name = ""
            self._cv.notify_all()
        self.emit("critical_frontier_end", frontier=name)

    def admit(
        self,
        *,
        request_id: str,
        node_id: str,
        phase: str,
        prompt_tokens: int,
        critical: bool,
        tool_resumed: bool,
        ready_ts: float,
    ) -> Decision:
        prompt_tokens = max(0, int(prompt_tokens))
        defer_start: float | None = None
        timeout = False
        budget_admitted = False
        frontier_at_ready = ""
        with self._cv:
            should_control = (
                self.policy == "critical_frontier"
                and self._frontier_active
                and not critical
                and phase in {"cold_prefill", "resume_prefill"}
            )
            frontier_at_ready = self._frontier_name
            if should_control and prompt_tokens <= self._remaining_budget:
                self._remaining_budget -= prompt_tokens
                budget_admitted = True
            elif should_control:
                defer_start = time.time()
                self._deferred_fifo.append(request_id)
                self.emit(
                    "admission_defer_start",
                    request_id=request_id,
                    node_id=node_id,
                    phase=phase,
                    tool_resumed=tool_resumed,
                    prompt_tokens=prompt_tokens,
                    frontier=frontier_at_ready,
                    remaining_prefill_budget_tokens=self._remaining_budget,
                    reason="noncritical_prefill_exceeds_frontier_budget",
                )
                deadline = time.monotonic() + self.max_defer_sec
                while True:
                    frontier_released = not self._frontier_active
                    at_head = bool(self._deferred_fifo) and self._deferred_fifo[0] == request_id
                    drain_available = (
                        not self.serialize_deferred_drain
                        or self._drain_inflight is None
                    )
                    if frontier_released and at_head and drain_available:
                        self._deferred_fifo.pop(0)
                        if self.serialize_deferred_drain:
                            self._drain_inflight = request_id
                        self._cv.notify_all()
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timeout = True
                        if request_id in self._deferred_fifo:
                            self._deferred_fifo.remove(request_id)
                        self._cv.notify_all()
                        break
                    self._cv.wait(timeout=remaining)

        submit_ts = time.time()
        defer_sec = 0.0 if defer_start is None else max(0.0, submit_ts - defer_start)
        if self.policy == "default_vllm":
            decision, reason = "admitted_immediately", "baseline_no_external_gating"
        elif critical:
            decision, reason = "admitted_immediately", "critical_frontier_request"
        elif budget_admitted:
            decision, reason = "admitted_with_prefill_budget", "fits_ready_time_prefill_budget"
        elif defer_start is None:
            decision, reason = "admitted_immediately", "no_active_critical_frontier"
        elif timeout:
            decision, reason = "admitted_after_max_defer", "maximum_defer_reached"
        else:
            decision, reason = "admitted_after_frontier", "top_level_result_ready"
        self.emit(
            "admission_decision",
            request_id=request_id,
            node_id=node_id,
            phase=phase,
            critical_path_candidate=critical,
            tool_resumed=tool_resumed,
            prompt_tokens=prompt_tokens,
            decision=decision,
            reason=reason,
            ready_ts=ready_ts,
            submit_ts=submit_ts,
            defer_sec=defer_sec,
            frontier_at_ready=frontier_at_ready,
        )
        return Decision(decision, reason, ready_ts, submit_ts, prompt_tokens, defer_sec)

    def on_request_complete(self, request_id: str) -> None:
        with self._cv:
            if self._drain_inflight == request_id:
                self._drain_inflight = None
                self._cv.notify_all()

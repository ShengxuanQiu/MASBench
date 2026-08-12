from __future__ import annotations

import threading
import time

from case_study1.admission import CriticalFrontierAdmission


def controller(events: list[tuple[str, dict]], **kwargs) -> CriticalFrontierAdmission:
    return CriticalFrontierAdmission(
        policy="critical_frontier",
        prefill_budget_tokens=kwargs.get("budget", 128),
        max_defer_sec=kwargs.get("max_defer", 1.0),
        serialize_deferred_drain=kwargs.get("serialize", True),
        emit=lambda event, **fields: events.append((event, fields)),
    )


def test_small_ready_prompt_uses_budget() -> None:
    events: list[tuple[str, dict]] = []
    gate = controller(events, budget=128)
    gate.begin_frontier("reviewer_to_result")
    decision = gate.admit(
        request_id="short",
        node_id="short",
        phase="resume_prefill",
        prompt_tokens=64,
        critical=False,
        tool_resumed=True,
        ready_ts=time.time(),
    )
    assert decision.decision == "admitted_with_prefill_budget"


def test_large_resume_waits_across_critical_transition() -> None:
    events: list[tuple[str, dict]] = []
    gate = controller(events, budget=128)
    gate.begin_frontier("reviewer_to_result")
    decisions = []

    def wait() -> None:
        decisions.append(
            gate.admit(
                request_id="long",
                node_id="long",
                phase="resume_prefill",
                prompt_tokens=4096,
                critical=False,
                tool_resumed=True,
                ready_ts=time.time(),
            )
        )

    thread = threading.Thread(target=wait)
    thread.start()
    time.sleep(0.02)
    gate.transition(completed_node="reviewer", ready_successor="finalizer")
    time.sleep(0.02)
    assert thread.is_alive()
    gate.end_frontier("reviewer_to_result")
    thread.join(timeout=1)
    assert decisions[0].decision == "admitted_after_frontier"


def test_max_defer_prevents_starvation() -> None:
    events: list[tuple[str, dict]] = []
    gate = controller(events, budget=0, max_defer=0.02)
    gate.begin_frontier("reviewer_to_result")
    decision = gate.admit(
        request_id="timeout",
        node_id="timeout",
        phase="cold_prefill",
        prompt_tokens=1024,
        critical=False,
        tool_resumed=False,
        ready_ts=time.time(),
    )
    assert decision.decision == "admitted_after_max_defer"

from __future__ import annotations

import threading
import time

from app.admission import AdmissionController


class RecordingTrace:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.lock = threading.Lock()

    def emit(self, **fields):
        with self.lock:
            self.events.append(fields)
        return fields


def test_default_policy_never_defers_tool_resume() -> None:
    trace = RecordingTrace()
    controller = AdmissionController(policy="default_vllm", max_defer_sec=1.0, trace=trace)  # type: ignore[arg-type]
    controller.on_first_token(request_id="critical", node_id="reviewer", critical_path_candidate=True, timestamp=time.time())
    ready = time.time()
    decision = controller.admit(
        request_id="resume",
        node_id="resume_agent",
        agent_role="evidence_processor",
        critical_path_candidate=False,
        tool_resumed=True,
        ready_ts=ready,
    )
    assert decision.decision == "admitted_immediately"
    assert decision.defer_duration_sec == 0.0


def test_critical_path_policy_releases_resume_after_decode() -> None:
    trace = RecordingTrace()
    controller = AdmissionController(policy="critical_path_aware", max_defer_sec=1.0, trace=trace)  # type: ignore[arg-type]
    controller.on_first_token(request_id="critical", node_id="reviewer", critical_path_candidate=True, timestamp=time.time())
    decisions = []

    def admit_resume() -> None:
        decisions.append(
            controller.admit(
                request_id="resume",
                node_id="resume_agent",
                agent_role="evidence_processor",
                critical_path_candidate=False,
                tool_resumed=True,
                ready_ts=time.time(),
            )
        )

    worker = threading.Thread(target=admit_resume)
    worker.start()
    time.sleep(0.03)
    assert worker.is_alive()
    controller.on_complete(request_id="critical", node_id="reviewer", critical_path_candidate=True, timestamp=time.time())
    worker.join(timeout=1.0)
    assert decisions[0].decision == "admitted_after_defer"
    assert decisions[0].defer_duration_sec >= 0.02
    assert {event["event_type"] for event in trace.events} >= {
        "critical_decode_start",
        "admission_defer_start",
        "admission_defer_end",
        "critical_decode_end",
    }


def test_max_defer_prevents_starvation() -> None:
    trace = RecordingTrace()
    controller = AdmissionController(policy="critical_path_aware", max_defer_sec=0.02, trace=trace)  # type: ignore[arg-type]
    controller.on_first_token(request_id="critical", node_id="reviewer", critical_path_candidate=True, timestamp=time.time())
    decision = controller.admit(
        request_id="resume",
        node_id="resume_agent",
        agent_role="evidence_processor",
        critical_path_candidate=False,
        tool_resumed=True,
        ready_ts=time.time(),
    )
    assert decision.decision == "admitted_after_max_defer"
    assert decision.defer_duration_sec >= 0.015

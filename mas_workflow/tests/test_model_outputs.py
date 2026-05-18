from __future__ import annotations

import json
from pathlib import Path

from app.tracing import TraceContext, stable_hash


def test_model_output_sidecar_links_to_llm_event(tmp_path: Path) -> None:
    trace_path = tmp_path / "run.jsonl"
    model_outputs_path = tmp_path / "run_model_outputs.json"
    trace = TraceContext(
        run_id="run",
        topology="single",
        topology_role="workflow",
        instance_id="manual_test",
        task_source="manual",
        workflow_id="single_run",
        trace_path=trace_path,
        summary_path=tmp_path / "run_summary.json",
        record_model_outputs=True,
        model_outputs_path=model_outputs_path,
    )
    event = trace.emit(
        event_type="llm_request_end",
        node_id="agent",
        node_name="Agent",
        node_type="llm",
        agent_id="agent",
        agent_role="worker",
        llm_request_id="request-1",
        request_id_for_backend="request-1",
        model="local-mas-model",
        output_hash=stable_hash("full answer"),
    )
    trace.record_model_output(
        event=event,
        output_text="full answer",
        input_text="prompt",
        system_prompt="system",
        user_prompt="user",
        response_metadata={"finish_reason": "stop"},
    )
    trace.save()

    assert trace_path.exists()
    payload = json.loads(model_outputs_path.read_text(encoding="utf-8"))
    assert payload["record_count"] == 1
    record = payload["records"][0]
    assert record["event_id"] == event["event_id"]
    assert record["output_text"] == "full answer"
    assert record["output_hash"] == stable_hash("full answer")
    saved_event = json.loads(trace_path.read_text(encoding="utf-8").splitlines()[0])
    assert saved_event["model_output_artifact_id"] == record["artifact_id"]
    assert saved_event["model_output_path"] == str(model_outputs_path)

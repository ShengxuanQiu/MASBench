from __future__ import annotations

import json
from pathlib import Path

from app.trace_export import events_to_arch_spans, export_trace_views


def test_trace_export_preserves_architecture_fields(tmp_path: Path) -> None:
    events = [
        {
            "event_id": "e1",
            "event_type": "llm_request_end",
            "relative_time_sec": 1.0,
            "duration_sec": 0.4,
            "node_id": "agent_1",
            "node_name": "Agent-1",
            "node_type": "llm",
            "parents": ["START"],
            "topology": "independent",
            "topology_role": "workflow",
            "parallel_group": "workers",
            "criticality": "near_critical",
            "round_id": 0,
            "input_tokens_est": 10,
            "output_tokens_est": 5,
            "duration_source": "mock_measured",
            "replay_policy": "live",
        },
        {
            "event_id": "e2",
            "event_type": "edge_dataflow",
            "relative_time_sec": 1.1,
            "duration_sec": 0,
            "node_id": "agent_1->aggregator",
            "node_name": "agent_1->aggregator",
            "node_type": "edge",
            "src_node": "agent_1",
            "dst_node": "aggregator",
            "artifact_type": "answer",
            "artifact_tokens_est": 5,
            "transfer_type": "aggregation",
            "topology": "independent",
            "topology_role": "workflow",
        },
    ]
    spans = events_to_arch_spans(events)
    assert spans[0]["tokens"]["total_tokens_est"] == 15
    assert spans[0]["parallel_group"] == "workers"
    assert spans[1]["artifact"]["artifact_type"] == "answer"

    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    paths = export_trace_views(events, trace_path)
    assert Path(paths["arch_spans"]).exists()
    assert Path(paths["otel_spans"]).exists()
    assert Path(paths["jaeger"]).exists()
    assert Path(paths["html_viewer"]).exists()

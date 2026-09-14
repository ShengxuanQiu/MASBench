"""Export MASBench-Arch traces into span, OTel/Jaeger, and HTML views.

The JSONL event stream is the canonical artifact for architecture simulation.
These exporters are derived views for inspection and external tooling.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from .tracing import stable_hash


SPAN_EVENT_TYPES = {
    "workflow_start",
    "workflow_end",
    "llm_request_end",
    "tool_search",
    "barrier",
    "manager_decision",
    "manager_instruction",
    "edge_dataflow",
}


def _duration(event: dict[str, Any]) -> float:
    if event.get("node_type") == "tool":
        return float(event.get("effective_duration_sec") or event.get("duration_sec") or 0.0)
    return float(event.get("duration_sec") or 0.0)


def _start_time(event: dict[str, Any]) -> float:
    return max(0.0, float(event.get("relative_time_sec") or 0.0) - _duration(event))


def _span_kind(event: dict[str, Any]) -> str:
    node_type = str(event.get("node_type") or "")
    if node_type == "llm":
        return "llm"
    if node_type == "tool":
        return "tool"
    if node_type == "edge":
        return "dataflow"
    if node_type == "barrier":
        return "barrier"
    if node_type == "manager":
        return "control"
    return "workflow" if node_type == "workflow" else node_type or "event"


def _tokens(event: dict[str, Any]) -> dict[str, int]:
    input_tokens = int(event.get("input_tokens_est") or 0)
    output_tokens = int(event.get("output_tokens_est") or 0)
    artifact_tokens = int(event.get("artifact_tokens_est") or 0)
    tool_tokens = int(event.get("output_size_tokens_est") or 0)
    return {
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "artifact_tokens_est": artifact_tokens,
        "tool_output_tokens_est": tool_tokens,
        "total_tokens_est": input_tokens + output_tokens + artifact_tokens + tool_tokens,
    }


def events_to_arch_spans(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert JSONL events into architecture-oriented spans.

    These spans preserve simulation fields: dependencies, token sizes, round IDs,
    duration source, replay policy, and communication/artifact metadata.
    """
    if any(e.get("canonical_type") in {"request_ready", "operation_start"} for e in events):
        # Derive views from canonical operations/dependencies, including local tools.
        # Historical tool_search/edge_dataflow events remain diagnostics, not a second graph.
        from .execution_graph import ExecutionGraph
        graph = ExecutionGraph.from_events(events)
        projected = []
        for event in events:
            kind = event.get("canonical_type")
            if kind in {"run_start", "run_finish", "barrier_sync"}:
                projected.append(event)
            elif kind in {"request_finish", "request_fail", "operation_finish", "operation_fail"}:
                op = graph.operations[event["operation_id"]]
                projected.append({**event, "event_type": "llm_request_end" if op.kind == "llm" else "tool_search",
                                  "node_type": op.kind, "parents": sorted(op.parents),
                                  "status": op.status})
        events = projected
    spans: list[dict[str, Any]] = []
    node_to_span: dict[str, str] = {}
    for index, event in enumerate(events):
        if event.get("event_type") not in SPAN_EVENT_TYPES:
            continue
        node_id = str(event.get("node_id") or event.get("event_id") or index)
        span_id = f"span_{stable_hash([event.get('event_id'), node_id, index])[:16]}"
        parent_nodes = [str(p) for p in (event.get("parents") or [])]
        parent_span_ids = [node_to_span[p] for p in parent_nodes if p in node_to_span]
        if not parent_span_ids and event.get("src_node") in node_to_span:
            parent_span_ids = [node_to_span[str(event.get("src_node"))]]
        tokens = _tokens(event)
        span = {
            "span_id": span_id,
            "event_id": event.get("event_id"),
            "name": event.get("node_name") or node_id,
            "kind": _span_kind(event),
            "node_id": node_id,
            "node_type": event.get("node_type"),
            "event_type": event.get("event_type"),
            "parent_span_ids": parent_span_ids,
            "dependency_node_ids": parent_nodes,
            "start_time_sec": round(_start_time(event), 6),
            "end_time_sec": round(float(event.get("relative_time_sec") or 0.0), 6),
            "duration_sec": round(_duration(event), 6),
            "duration_source": event.get("duration_source"),
            "replay_policy": event.get("replay_policy"),
            "status": event.get("status"),
            "criticality": event.get("criticality"),
            "parallel_group": event.get("parallel_group"),
            "round_id": event.get("round_id"),
            "manager_round_id": event.get("manager_round_id"),
            "peer_round_id": event.get("peer_round_id"),
            "topology": event.get("topology"),
            "topology_role": event.get("topology_role"),
            "mode": event.get("mode"),
            "motif_name": event.get("motif_name"),
            "motif_instance_id": event.get("motif_instance_id"),
            "motif_family": event.get("motif_family"),
            "role_slot": event.get("role_slot"),
            "role": event.get("role"),
            "stage_instance_id": event.get("stage_instance_id"),
            "canonical_attributes": event.get("attributes", {}),
            "role_index": event.get("role_index"),
            "agent_instance_id": event.get("agent_instance_id"),
            "parent_motif_id": event.get("parent_motif_id"),
            "composed_from_topologies": event.get("composed_from_topologies") or [],
            "motif_tags": event.get("motif_tags") or [],
            "tokens": tokens,
            "artifact": {
                "artifact_id": event.get("artifact_id"),
                "artifact_type": event.get("artifact_type"),
                "transfer_type": event.get("transfer_type"),
                "fanout_count": event.get("fanout_count"),
                "recipient_count": event.get("recipient_count"),
                "is_shared_context": event.get("is_shared_context"),
            },
            "tool": {
                "tool_name": event.get("tool_name"),
                "tool_mode": event.get("tool_mode"),
                "tool_result_hash": event.get("tool_result_hash"),
                "measured_duration_sec": event.get("measured_duration_sec"),
                "injected_delay_sec": event.get("injected_delay_sec"),
                "effective_duration_sec": event.get("effective_duration_sec"),
                "latency_profile": event.get("latency_profile"),
            },
            "llm": {
                "agent_id": event.get("agent_id"),
                "agent_role": event.get("agent_role"),
                "llm_mode": event.get("llm_mode"),
                "model": event.get("model"),
                "request_id_for_backend": event.get("request_id_for_backend"),
                "max_output_tokens": event.get("max_output_tokens"),
                "backend_prompt_tokens": event.get("backend_prompt_tokens"),
                "backend_completion_tokens": event.get("backend_completion_tokens"),
                "backend_total_tokens": event.get("backend_total_tokens"),
                "backend_finish_reason": event.get("backend_finish_reason"),
                "queue_wait_sec": event.get("queue_wait_sec"),
                "prompt_hash": event.get("prompt_hash"),
                "output_hash": event.get("output_hash"),
            },
        }
        spans.append(span)
        if event.get("node_type") != "edge":
            node_to_span[node_id] = span_id
    return spans


def spans_to_otel(spans: list[dict[str, Any]], *, trace_id: str) -> list[dict[str, Any]]:
    otel: list[dict[str, Any]] = []
    for span in spans:
        parent = span["parent_span_ids"][0] if span["parent_span_ids"] else None
        attrs = {
            "mas.topology": span.get("topology"),
            "mas.topology_role": span.get("topology_role"),
            "mas.mode": span.get("mode"),
            "mas.motif_name": span.get("motif_name"),
            "mas.motif_instance_id": span.get("motif_instance_id"),
            "mas.motif_family": span.get("motif_family"),
            "mas.role_slot": span.get("role_slot"),
            "mas.role": span.get("role"),
            "mas.stage_instance_id": span.get("stage_instance_id"),
            "mas.canonical_attributes": json.dumps(span.get("canonical_attributes", {}), ensure_ascii=False),
            "mas.role_index": span.get("role_index"),
            "mas.agent_instance_id": span.get("agent_instance_id"),
            "mas.parent_motif_id": span.get("parent_motif_id"),
            "mas.composed_from_topologies": ",".join(span.get("composed_from_topologies") or []),
            "mas.node_id": span.get("node_id"),
            "mas.node_type": span.get("node_type"),
            "mas.event_type": span.get("event_type"),
            "mas.parallel_group": span.get("parallel_group"),
            "mas.criticality": span.get("criticality"),
            "mas.round_id": span.get("round_id"),
            "mas.manager_round_id": span.get("manager_round_id"),
            "mas.peer_round_id": span.get("peer_round_id"),
            "mas.duration_source": span.get("duration_source"),
            "mas.replay_policy": span.get("replay_policy"),
            "mas.tokens.total_est": span["tokens"]["total_tokens_est"],
            "mas.tokens.input_est": span["tokens"]["input_tokens_est"],
            "mas.tokens.output_est": span["tokens"]["output_tokens_est"],
            "mas.tokens.artifact_est": span["tokens"]["artifact_tokens_est"],
            "mas.artifact_type": span["artifact"].get("artifact_type"),
            "mas.transfer_type": span["artifact"].get("transfer_type"),
            "mas.tool_name": span["tool"].get("tool_name"),
            "mas.llm_model": span["llm"].get("model"),
            "mas.llm_request_id": span["llm"].get("request_id_for_backend"),
            "mas.llm_max_output_tokens": span["llm"].get("max_output_tokens"),
            "mas.llm_backend_prompt_tokens": span["llm"].get("backend_prompt_tokens"),
            "mas.llm_backend_completion_tokens": span["llm"].get("backend_completion_tokens"),
            "mas.llm_backend_total_tokens": span["llm"].get("backend_total_tokens"),
            "mas.llm_backend_finish_reason": span["llm"].get("backend_finish_reason"),
        }
        start_us = int(float(span["start_time_sec"]) * 1_000_000)
        duration_us = int(float(span["duration_sec"]) * 1_000_000)
        otel.append(
            {
                "traceId": trace_id,
                "spanId": span["span_id"],
                "parentSpanId": parent,
                "operationName": span["name"],
                "startTime": start_us,
                "duration": duration_us,
                "kind": "CLIENT" if span["kind"] in {"llm", "tool"} else "INTERNAL",
                "tags": attrs,
                "logs": [],
                "references": [{"refType": "CHILD_OF", "spanID": parent, "traceID": trace_id}] if parent else [],
                "meta": span,
            }
        )
    return otel


def otel_to_jaeger(otel_spans: list[dict[str, Any]]) -> dict[str, Any]:
    trace_id = otel_spans[0]["traceId"] if otel_spans else "trace-empty"
    return {
        "data": [
            {
                "traceID": trace_id,
                "processes": {"p1": {"serviceName": "masbench-arch", "tags": []}},
                "spans": [
                    {
                        "traceID": span["traceId"],
                        "spanID": span["spanId"],
                        "operationName": span["operationName"],
                        "references": span.get("references") or [],
                        "startTime": span["startTime"],
                        "duration": span["duration"],
                        "processID": "p1",
                        "tags": [{"key": k, "type": "string", "value": str(v)} for k, v in span.get("tags", {}).items() if v is not None],
                        "logs": span.get("logs") or [],
                    }
                    for span in otel_spans
                ],
            }
        ]
    }


def write_html_viewer(spans: list[dict[str, Any]], output_path: Path) -> None:
    spans_json = json.dumps(spans, ensure_ascii=False)
    timeline_spans = [s for s in spans if s.get("event_type") != "workflow_start"]
    timeline_json = json.dumps(timeline_spans, ensure_ascii=False)
    rows = "\n".join(
        f"<tr><td>{escape(str(s['kind']))}</td><td>{escape(str(s['name']))}</td>"
        f"<td>{escape(str(s.get('parallel_group') or ''))}</td>"
        f"<td>{escape(str(s.get('manager_round_id') if s.get('manager_round_id') is not None else ''))}</td>"
        f"<td>{escape(str(s.get('peer_round_id') if s.get('peer_round_id') is not None else ''))}</td>"
        f"<td>{s['duration_sec']:.6f}</td><td>{s['tokens']['total_tokens_est']}</td></tr>"
        for s in spans
    )
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>MASBench-Arch Trace Viewer</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #202124; }}
h1 {{ font-size: 20px; margin-bottom: 6px; }}
.meta {{ color: #5f6368; margin-bottom: 18px; }}
.timeline {{ position: relative; border: 1px solid #dadce0; height: {max(180, 24 * len(spans) + 20)}px; overflow: auto; background: #fff; }}
.bar {{ position: absolute; height: 16px; border-radius: 3px; font-size: 11px; line-height: 16px; overflow: hidden; white-space: nowrap; color: #111; padding-left: 4px; box-sizing: border-box; }}
.marker {{ position: absolute; width: 6px; height: 16px; border-radius: 3px; box-sizing: border-box; border-left: 2px solid #444; font-size: 0; }}
.llm {{ background: #8ab4f8; }}
.tool {{ background: #fdd663; }}
.dataflow {{ background: #c7e8ca; }}
.barrier {{ background: #f6aea9; }}
.control {{ background: #d7aefb; }}
.workflow {{ background: #e8eaed; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 18px; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e8eaed; text-align: left; padding: 6px 8px; }}
th {{ background: #f8fafd; position: sticky; top: 0; }}
code {{ background: #f1f3f4; padding: 1px 4px; border-radius: 3px; }}
</style>
</head>
<body>
<h1>MASBench-Arch Trace Viewer</h1>
<div class="meta">Architecture view: spans preserve tokens, dependencies, rounds, parallel groups, artifact movement, tool latency, and replay policy.</div>
<div id="timeline" class="timeline"></div>
<table>
<thead><tr><th>kind</th><th>name</th><th>parallel group</th><th>manager round</th><th>peer round</th><th>duration sec</th><th>tokens est</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<script>
const allSpans = {spans_json};
const spans = {timeline_json};
const el = document.getElementById('timeline');
const maxEnd = Math.max(0.001, ...spans.map(s => s.end_time_sec));
const width = Math.max(900, el.clientWidth - 20);
spans.forEach((s, i) => {{
  const d = document.createElement('div');
  const isMarker = s.duration_sec <= 0.001 || s.kind === 'dataflow' || s.kind === 'control' || s.kind === 'workflow';
  d.className = (isMarker ? 'marker ' : 'bar ') + s.kind;
  d.style.left = (10 + (s.start_time_sec / maxEnd) * (width - 20)) + 'px';
  d.style.top = (10 + i * 24) + 'px';
  if (!isMarker) d.style.width = Math.max(3, (s.duration_sec / maxEnd) * (width - 20)) + 'px';
  d.title = JSON.stringify(s, null, 2);
  d.textContent = isMarker ? '' : s.name + '  ' + s.duration_sec.toFixed(3) + 's';
  el.appendChild(d);
}});
</script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def export_trace_views(events: list[dict[str, Any]], trace_path: Path, *, enabled: bool = True) -> dict[str, str]:
    if not enabled:
        return {}
    base = trace_path.with_suffix("")
    spans = events_to_arch_spans(events)
    trace_id = "trace_" + stable_hash([trace_path.name, events[0].get("run_id") if events else "empty"])[:24]
    otel = spans_to_otel(spans, trace_id=trace_id)
    paths = {
        "arch_spans": str(base) + "_spans.json",
        "otel_spans": str(base) + "_otel.json",
        "jaeger": str(base) + "_jaeger.json",
        "html_viewer": str(base) + "_viewer.html",
    }
    Path(paths["arch_spans"]).write_text(json.dumps({"trace_id": trace_id, "spans": spans}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    Path(paths["otel_spans"]).write_text(json.dumps({"resourceSpans": otel, "trace_id": trace_id}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    Path(paths["jaeger"]).write_text(json.dumps(otel_to_jaeger(otel), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_html_viewer(spans, Path(paths["html_viewer"]))
    return paths

"""Analyze MASBench-Arch JSONL traces and emit architecture metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MASBench-Arch trace JSONL")
    parser.add_argument("trace_path", help="Trace .jsonl file or directory")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--summary-name", default="")
    return parser.parse_args()


def read_events(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else data.get("events", [])


def summarize(events: list[dict[str, Any]], trace_path: Path | None = None) -> dict[str, Any]:
    if not events:
        return {"trace_path": str(trace_path or ""), "event_count": 0}
    llm = [e for e in events if e.get("event_type") == "llm_request_end"]
    tools = [e for e in events if str(e.get("event_type", "")).startswith("tool_")]
    edges = [e for e in events if e.get("node_type") == "edge"]
    barriers = [e for e in events if e.get("node_type") == "barrier"]
    groups = defaultdict(list)
    for e in events:
        if e.get("parallel_group"):
            groups[str(e.get("parallel_group"))].append(e)
    tool_by_name: dict[str, float] = defaultdict(float)
    for e in tools:
        tool_by_name[str(e.get("tool_name") or "tool")] += float(e.get("effective_duration_sec") or e.get("duration_sec") or 0)
    peer_edges = [e for e in edges if e.get("artifact_type") == "peer_message"]
    manager_decisions = [e for e in events if e.get("event_type") == "manager_decision"]
    topology = str(events[0].get("topology") or "")
    summary = {
        "trace_path": str(trace_path or ""),
        "topology": topology,
        "instance_id": events[0].get("instance_id"),
        "run_id": events[0].get("run_id"),
        "event_count": len(events),
        "end_to_end_latency": max(float(e.get("relative_time_sec") or 0) for e in events),
        "critical_path_length": sum(float(e.get("duration_sec") or 0) for e in llm) + sum(float(e.get("effective_duration_sec") or e.get("duration_sec") or 0) for e in tools),
        "max_parallel_width": max((len(v) for v in groups.values()), default=1),
        "barrier_wait_sum": sum(float(e.get("barrier_wait_sec") or 0) for e in barriers),
        "straggler_gap_by_parallel_group": {str(e.get("barrier_id") or e.get("node_id")): float(e.get("straggler_gap_sec") or 0) for e in barriers},
        "manager_rounds_actual": max([int(e.get("manager_round_id") or 0) for e in events] or [0]) + (1 if any(e.get("manager_round_id") == 0 for e in events) else 0),
        "debate_rounds_actual": max([int(e.get("peer_round_id") or 0) for e in events] or [0]),
        "peer_rounds_actual": max([int(e.get("peer_round_id") or 0) for e in events] or [0]),
        "total_artifact_tokens_est": sum(int(e.get("artifact_tokens_est") or 0) for e in edges),
        "total_input_tokens_est": sum(int(e.get("input_tokens_est") or 0) for e in llm),
        "total_output_tokens_est": sum(int(e.get("output_tokens_est") or 0) for e in llm),
        "context_duplication_ratio": 0.0,
        "aggregation_tokens_est": sum(int(e.get("artifact_tokens_est") or 0) for e in edges if e.get("artifact_type") in {"summary", "final", "vote"}),
        "all_gather_tokens_est": sum(int(e.get("artifact_tokens_est") or 0) for e in peer_edges),
        "peer_message_tokens_est": sum(int(e.get("artifact_tokens_est") or 0) for e in peer_edges),
        "broadcast_tokens_est": sum(int(e.get("artifact_tokens_est") or 0) for e in edges if e.get("transfer_type") == "broadcast"),
        "total_tool_time": sum(float(e.get("effective_duration_sec") or e.get("duration_sec") or 0) for e in tools),
        "measured_tool_time": sum(float(e.get("measured_duration_sec") or 0) for e in tools),
        "injected_tool_delay_time": sum(float(e.get("injected_delay_sec") or 0) for e in tools),
        "tool_time_by_tool_name": dict(tool_by_name),
        "tool_stall_events": len([e for e in tools if float(e.get("effective_duration_sec") or 0) > 0]),
        "total_llm_time": sum(float(e.get("duration_sec") or 0) for e in llm),
        "backend_prompt_tokens": sum(int(e.get("backend_prompt_tokens") or 0) for e in llm),
        "backend_completion_tokens": sum(int(e.get("backend_completion_tokens") or 0) for e in llm),
        "backend_total_tokens": sum(int(e.get("backend_total_tokens") or 0) for e in llm),
        "total_queue_wait": sum(float(e.get("queue_wait_sec") or 0) for e in llm),
        "critical_queue_wait_time": sum(float(e.get("queue_wait_sec") or 0) for e in llm if e.get("criticality") == "critical"),
        "max_ready_queue_size": 0,
        "dispatch_policy": next((e.get("dispatch_policy") for e in llm if e.get("dispatch_policy")), ""),
        "slot_utilization_estimate": 0.0,
    }
    if trace_path is not None:
        base = trace_path.with_suffix("")
        summary["trace_exports"] = {
            "arch_spans": str(base) + "_spans.json",
            "otel_spans": str(base) + "_otel.json",
            "jaeger": str(base) + "_jaeger.json",
            "html_viewer": str(base) + "_viewer.html",
        }
    summary["total_tokens_est"] = summary["total_input_tokens_est"] + summary["total_output_tokens_est"]
    base = max(1, min([int(e.get("input_tokens_est") or 0) for e in llm] or [1]))
    summary["context_duplication_ratio"] = round(summary["total_input_tokens_est"] / base, 4)
    if topology == "single":
        summary["single_baseline_tokens"] = summary["total_input_tokens_est"] + summary["total_output_tokens_est"]
        summary["single_baseline_time"] = summary["end_to_end_latency"]
    elif topology == "independent":
        summary["aggregation_input_tokens"] = sum(int(e.get("input_tokens_est") or 0) for e in llm if e.get("node_id") == "aggregator")
        worker_inputs = [int(e.get("input_tokens_est") or 0) for e in llm if str(e.get("node_id", "")).startswith("agent_")]
        summary["redundant_worker_input_tokens"] = max(0, sum(worker_inputs) - (max(worker_inputs) if worker_inputs else 0))
    elif topology == "centralized":
        summary["manager_decision_count"] = len(manager_decisions)
        summary["worker_fanout_count"] = sum(len(e.get("selected_workers") or []) for e in manager_decisions)
    elif topology == "decentralized":
        summary["peer_message_count"] = len(peer_edges)
        summary["all_to_all_tokens"] = summary["peer_message_tokens_est"]
    elif topology == "hybrid":
        summary["nested_round_count"] = len({(e.get("manager_round_id"), e.get("peer_round_id")) for e in events if e.get("peer_round_id") is not None})
        summary["manager_collection_barrier"] = len([e for e in barriers if "manager_collect" in str(e.get("node_id"))])
    return summary


def main() -> int:
    args = parse_args()
    path = Path(args.trace_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (path if path.is_dir() else path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        traces = sorted(path.rglob("*.jsonl"))
        summaries = [summarize(read_events(item), item) for item in traces]
        output = out_dir / (args.summary_name or "summary.json")
        output.write_text(json.dumps({"traces": summaries}, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    else:
        output = out_dir / (args.summary_name or f"{path.stem}_analysis_summary.json")
        output.write_text(json.dumps(summarize(read_events(path), path), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Summary: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

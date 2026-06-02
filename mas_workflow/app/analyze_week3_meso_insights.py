"""Analyze Week3 meso workflow traces and write systems insight artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.pyplot as plt


MESO_NAMES = {
    "tool_resume_contention_meso",
    "hierarchical_synthesis_pressure_meso",
    "debate_allgather_pressure_meso",
    "retry_debug_pressure_meso",
    "shared_memory_fanin_meso",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Week3 meso insight report")
    parser.add_argument("--trace-root", default="traces/week3")
    parser.add_argument("--progress-dir", default="progress/week3")
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def event_end(event: dict[str, Any]) -> float:
    return safe_float(event.get("relative_time_sec"))


def event_start(event: dict[str, Any]) -> float:
    return event_end(event) - safe_float(event.get("duration_sec"))


def discover_traces(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*.jsonl") if "snapshots" not in p.parts)


def workflow_name(events: list[dict[str, Any]], path: Path) -> str:
    for event in events:
        name = event.get("meso_workload_name") or event.get("workflow_name") or event.get("motif_name") or event.get("topology")
        if name:
            return str(name)
    return path.stem


def request_windows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "event": event,
            "start": event_start(event),
            "end": event_end(event),
            "tokens": safe_int(event.get("total_tokens") or event.get("total_tokens_est") or event.get("input_tokens_est"))
            + safe_int(event.get("output_tokens_est")),
        }
        for event in events
        if event.get("event_type") == "llm_request_end"
    ]


def overlaps(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a["start"] <= b["end"] and b["start"] <= a["end"]


def backend_summary(path: Path) -> dict[str, Any]:
    candidates = [
        path.with_name(path.stem + "_backend_metrics_summary.json"),
        path.with_name(path.stem + "_backend_summary.json"),
        path.with_name(path.stem + "_backend_metrics.json"),
    ]
    for item in candidates:
        if item.exists():
            try:
                payload = json.loads(item.read_text(encoding="utf-8"))
                return payload.get("summary") or payload
            except json.JSONDecodeError:
                return {}
    return {}


def summarize_run(path: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    name = workflow_name(events, path)
    llm = [e for e in events if e.get("event_type") == "llm_request_end"]
    tools = [e for e in events if str(e.get("event_type", "")).startswith("tool_")]
    edges = [e for e in events if e.get("node_type") == "edge"]
    barriers = [e for e in events if e.get("node_type") == "barrier"]
    windows = request_windows(events)
    critical = [
        w for w in windows
        if w["event"].get("criticality") == "critical"
        or w["event"].get("critical_path_candidate")
        or str(w["event"].get("critical_stage") or "").lower() in {"reviewer", "finalizer"}
    ]
    background_resume = [
        w for w in windows
        if w["event"].get("background_resume_request")
        or w["event"].get("resume_after_tool")
        or w["event"].get("workload_role") == "background_tool_branch"
    ]
    overlap_count = sum(1 for c in critical for b in background_resume if overlaps(c, b))
    critical_e2e = [safe_float(w["event"].get("request_e2e_sec") or w["event"].get("duration_sec")) for w in critical]
    background_e2e = [safe_float(w["event"].get("request_e2e_sec") or w["event"].get("duration_sec")) for w in windows if w not in critical]
    bsum = backend_summary(path)
    peer_tokens = sum(safe_int(e.get("artifact_tokens_est")) for e in edges if e.get("artifact_type") == "peer_message")
    broadcast_tokens = sum(safe_int(e.get("artifact_tokens_est")) for e in edges if e.get("transfer_type") == "broadcast")
    shared_context_tokens = sum(safe_int(e.get("round_shared_context_tokens")) for e in llm)
    duplicated_shared_tokens = sum(safe_int(e.get("duplicated_shared_context_tokens") or e.get("duplicated_context_tokens_est")) for e in events)
    shared_block_count = sum(len(e.get("shared_block_ids") or []) for e in llm)
    private_history_tokens = sum(safe_int(e.get("private_history_tokens")) for e in llm)
    estimated_kv_tokens = sum(safe_int(e.get("estimated_kv_tokens") or e.get("input_tokens") or e.get("input_tokens_est")) for e in llm)
    peak_context_tokens = 0
    for group in sorted({str(e.get("parallel_group")) for e in llm if e.get("parallel_group")}):
        peak_context_tokens = max(peak_context_tokens, sum(safe_int(e.get("input_tokens") or e.get("input_tokens_est")) for e in llm if e.get("parallel_group") == group))
    memory_write = [e for e in events if e.get("event_type") == "memory_write"]
    memory_read = [e for e in events if e.get("event_type") == "memory_read"]
    total_input = sum(safe_int(e.get("input_tokens") or e.get("input_tokens_est")) for e in llm)
    total_output = sum(safe_int(e.get("output_tokens") or e.get("output_tokens_est")) for e in llm)
    max_fanin = max([safe_int(e.get("fan_in_count")) for e in events] or [0])
    upstream_outputs = [safe_int(e.get("output_tokens") or e.get("output_tokens_est")) for e in llm if e.get("criticality") != "merge"]
    downstream_inputs = [safe_int(e.get("input_tokens") or e.get("input_tokens_est")) for e in llm if safe_int(e.get("fan_in_count")) > 1 or e.get("criticality") == "merge"]
    stalled_agents = len([e for e in tools if e.get("tool_stalled") or e.get("criticality") == "non_critical"])
    resume_inputs = [safe_int(w["event"].get("input_tokens") or w["event"].get("input_tokens_est")) for w in background_resume]
    return {
        "run_path": str(path),
        "workflow_name": name,
        "llm_request_count": len(llm),
        "tool_event_count": len(tools),
        "makespan_sec": max([event_end(e) for e in events] or [0.0]),
        "total_tokens": total_input + total_output,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "barrier_wait_sec": sum(safe_float(e.get("barrier_wait_sec")) for e in barriers),
        "background_resume_request_count": len(background_resume),
        "non_critical_stalled_agent_count": stalled_agents,
        "kv_idle_proxy_tokens": stalled_agents * (mean(resume_inputs) if resume_inputs else 0.0),
        "critical_request_count": len(critical),
        "critical_background_overlap_count": overlap_count,
        "critical_request_e2e_avg_sec": mean(critical_e2e) if critical_e2e else 0.0,
        "critical_request_e2e_max_sec": max(critical_e2e) if critical_e2e else 0.0,
        "background_request_e2e_median_sec": median(background_e2e) if background_e2e else 0.0,
        "critical_slowdown_ratio": (mean(critical_e2e) / max(median(background_e2e), 1e-9)) if critical_e2e and background_e2e else 0.0,
        "max_running_requests": bsum.get("max_num_requests_running", "unavailable"),
        "max_waiting_requests": bsum.get("max_num_requests_waiting", "unavailable"),
        "avg_ttft_sec": bsum.get("backend_avg_ttft_sec_from_metrics", "unavailable"),
        "avg_tpot_sec": bsum.get("backend_avg_tpot_sec_from_metrics", "unavailable"),
        "queue_time": "unavailable",
        "batch_membership": "unavailable",
        "kv_residency": "unavailable",
        "hierarchy_depth": max([safe_int(e.get("hierarchy_depth")) for e in events] or [0]),
        "group_count": max([safe_int(e.get("group_count")) for e in events] or [0]),
        "agents_per_group": max([safe_int(e.get("agents_per_group")) for e in events] or [0]),
        "fan_in_width": max_fanin,
        "avg_downstream_input_tokens": mean(downstream_inputs) if downstream_inputs else 0.0,
        "avg_upstream_output_tokens": mean(upstream_outputs) if upstream_outputs else 0.0,
        "context_amplification_ratio": (mean(downstream_inputs) / max(mean(upstream_outputs), 1.0)) if downstream_inputs and upstream_outputs else 0.0,
        "duplicated_context_tokens": sum(safe_int(e.get("duplicated_context_tokens_est")) for e in events),
        "private_history_tokens": private_history_tokens,
        "shared_block_count": shared_block_count,
        "round_shared_context_tokens": shared_context_tokens,
        "duplicated_shared_context_tokens": duplicated_shared_tokens,
        "estimated_kv_tokens": estimated_kv_tokens,
        "peak_concurrent_context_tokens": peak_context_tokens,
        "per_request_reuse_work_proxy": safe_int(max([safe_int(e.get("agent_count")) for e in events] or [0])) * shared_block_count,
        "collective_reuse_work_proxy": shared_block_count,
        "pairwise_shared_block_similarity_proxy": mean([safe_float(e.get("pairwise_shared_block_similarity_proxy")) for e in llm if e.get("pairwise_shared_block_similarity_proxy") is not None] or [0.0]),
        "agent_count": max([safe_int(e.get("agent_count") or e.get("num_agents")) for e in events] or [0]),
        "debate_rounds": max([safe_int(e.get("peer_round_id")) for e in events] or [0]),
        "peer_message_tokens": peer_tokens,
        "broadcast_tokens": broadcast_tokens,
        "retry_loop_depth": max([safe_int(e.get("debug_loop_count") or e.get("retry_count")) for e in events] or [0]),
        "extra_llm_time_sec": sum(safe_float(e.get("duration_sec")) for e in llm),
        "memory_write_count": len(memory_write),
        "memory_read_count": len(memory_read),
        "artifact_tokens": sum(safe_int(e.get("artifact_tokens_est")) for e in edges),
        "shared_evidence_write_tokens": sum(safe_int(e.get("shared_evidence_write_tokens_est")) for e in events),
        "shared_evidence_read_tokens": sum(safe_int(e.get("shared_evidence_read_tokens_est")) for e in events),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_placeholder(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def finish_plot(fig: Any, path: Path, caption: str) -> None:
    fig.text(0.01, 0.01, caption, fontsize=8, color="#444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tool_timeline(fig_dir: Path, trace_paths: list[Path]) -> None:
    selected: tuple[Path, list[dict[str, Any]]] | None = None
    for path in trace_paths:
        events = read_jsonl(path)
        if workflow_name(events, path) == "tool_resume_contention_meso":
            selected = (path, events)
            break
    out = fig_dir / "fig_tool_resume_overlap_timeline.png"
    if not selected:
        plot_placeholder(out, "Tool resume overlap timeline", "No tool_resume_contention_meso trace found.")
        return
    path, events = selected
    fig, ax = plt.subplots(figsize=(10, 4.5))
    y = 0
    for event in events:
        etype = event.get("event_type")
        if etype == "llm_request_end":
            start = event_start(event)
            dur = max(safe_float(event.get("duration_sec")), 0.01)
            role = str(event.get("workload_role") or event.get("agent_role") or event.get("node_id"))
            color = "#d62728" if event.get("criticality") == "critical" else ("#ff7f0e" if event.get("background_resume_request") else "#9aa0a6")
            ax.barh(y, dur, left=start, color=color, edgecolor="black", height=0.55)
            ax.text(start + dur / 2, y, str(event.get("node_id") or role), ha="center", va="center", fontsize=7)
            y += 1
        elif str(etype).startswith("tool_"):
            start = event_start(event)
            dur = max(safe_float(event.get("duration_sec") or event.get("tool_latency_sec")), 0.01)
            ax.barh(y, dur, left=start, color="#2ca02c", edgecolor="black", height=0.55)
            ax.text(start + dur / 2, y, "tool", ha="center", va="center", fontsize=7)
            y += 1
    ax.set_title("Tool resume burst overlaps critical stages")
    ax.set_xlabel("relative time (sec)")
    ax.set_yticks([])
    finish_plot(fig, out, f"Evidence tier: observed/proxy. Trace: {path.name}. Red=critical, orange=background resume, green=tool.")


def plot_scatter(rows: list[dict[str, Any]], fig_dir: Path, filename: str, title: str, x_key: str, y_key: str, xlabel: str, ylabel: str, filter_name: str = "") -> None:
    out = fig_dir / filename
    data = [r for r in rows if (not filter_name or r["workflow_name"] == filter_name)]
    if not data:
        plot_placeholder(out, title, "No matching trace rows.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = [safe_float(r.get(x_key)) for r in data]
    ys = [safe_float(r.get(y_key)) for r in data]
    labels = [r["workflow_name"].replace("_meso", "") for r in data]
    ax.scatter(xs, ys, s=60, color="#1f77b4")
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    finish_plot(fig, out, "Evidence tier: observed workload metrics; backend queue/KV fields remain unavailable unless vLLM exposes them.")


def plot_line(rows: list[dict[str, Any]], fig_dir: Path, filename: str, title: str, x_key: str, y_keys: list[str], xlabel: str, ylabel: str, filter_name: str) -> None:
    out = fig_dir / filename
    data = sorted([r for r in rows if r["workflow_name"] == filter_name], key=lambda r: safe_float(r.get(x_key)))
    if not data:
        plot_placeholder(out, title, "No matching trace rows.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = [safe_float(r.get(x_key)) for r in data]
    for key in y_keys:
        ax.plot(xs, [safe_float(r.get(key)) for r in data], marker="o", label=key)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    finish_plot(fig, out, "Evidence tier: observed/proxy. Trend is interpreted as workload pressure, not scheduler causality by itself.")


def plot_critical_path_contention_workflow(fig_dir: Path) -> None:
    out = fig_dir / "fig_critical_path_contention_workflow.png"
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axis("off")
    nodes = {
        "planner": (0.08, 0.72, "#d62728"),
        "critical_coder": (0.30, 0.72, "#d62728"),
        "critical_reviewer": (0.52, 0.72, "#d62728"),
        "finalizer": (0.76, 0.72, "#d62728"),
        "tool_agent_1": (0.25, 0.35, "#4c78a8"),
        "tool_wait": (0.48, 0.35, "#8da0cb"),
        "background_resume": (0.70, 0.35, "#4c78a8"),
        "late_merge": (0.86, 0.35, "#9aa0a6"),
    }
    for name, (x, y, color) in nodes.items():
        ax.text(x, y, name, ha="center", va="center", fontsize=10, color="white", bbox={"boxstyle": "round,pad=0.35", "fc": color, "ec": "black"})
    edges = [
        ("planner", "critical_coder", "#d62728"),
        ("critical_coder", "critical_reviewer", "#d62728"),
        ("critical_reviewer", "finalizer", "#d62728"),
        ("planner", "tool_agent_1", "#4c78a8"),
        ("tool_agent_1", "tool_wait", "#4c78a8"),
        ("tool_wait", "background_resume", "#4c78a8"),
        ("background_resume", "late_merge", "#4c78a8"),
    ]
    for src, dst, color in edges:
        x1, y1, _ = nodes[src]
        x2, y2, _ = nodes[dst]
        ax.annotate("", xy=(x2 - 0.055, y2), xytext=(x1 + 0.055, y1), arrowprops={"arrowstyle": "->", "lw": 2, "color": color})
    ax.text(0.47, 0.18, "non-critical lifecycle: LLM-1 -> function call / tool wait -> LLM-2 resume", ha="center", fontsize=10)
    ax.text(0.45, 0.94, "Critical-path contention DAG: red critical path, blue non-critical tool-stalled branches", ha="center", fontsize=12, weight="bold")
    finish_plot(fig, out, "Evidence tier: workload design. This DAG shows structure only; it does not prove backend queue contention.")


def selected_events(trace_paths: list[Path], name: str) -> tuple[Path, list[dict[str, Any]]] | None:
    for path in trace_paths:
        events = read_jsonl(path)
        if workflow_name(events, path) == name:
            return path, events
    return None


def plot_critical_overlap_timeline(fig_dir: Path, trace_paths: list[Path]) -> None:
    out = fig_dir / "fig_critical_overlap_timeline.png"
    selected = selected_events(trace_paths, "tool_resume_contention_meso")
    if not selected:
        plot_placeholder(out, "Critical overlap timeline", "No tool_resume_contention_meso trace found.")
        return
    path, events = selected
    fig, ax = plt.subplots(figsize=(10, 5))
    lanes = []
    for e in events:
        if e.get("event_type") == "llm_request_end" and (e.get("critical_stage") in {"reviewer", "finalizer"} or e.get("background_resume_request")):
            lanes.append(e)
        elif str(e.get("event_type", "")).startswith("tool_") and e.get("tool_stalled"):
            lanes.append(e)
    for y, e in enumerate(lanes):
        start = event_start(e)
        dur = max(safe_float(e.get("duration_sec") or e.get("tool_latency_sec")), 0.01)
        if e.get("criticality") == "critical":
            color = "#d62728"
        elif e.get("background_resume_request"):
            color = "#4c78a8"
        else:
            color = "#2ca02c"
        ax.barh(y, dur, left=start, color=color, edgecolor="black", alpha=0.9)
        ax.text(start + dur / 2, y, str(e.get("node_id")), ha="center", va="center", fontsize=7, color="white")
    ax.set_title("Observed overlap window between non-critical resume and critical requests")
    ax.set_xlabel("relative time (sec)")
    ax.set_yticks([])
    finish_plot(fig, out, f"Evidence tier: observed overlap. Trace: {path.name}. Red=critical reviewer/finalizer, blue=non-critical resume, green=tool wait.")


def plot_critical_latency_overlap(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    data = [r for r in rows if r["workflow_name"] == "tool_resume_contention_meso"]
    out = fig_dir / "fig_critical_latency_vs_background_overlap.png"
    if not data:
        plot_placeholder(out, "Critical latency vs background overlap", "No tool resume rows.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = [safe_float(r.get("critical_background_overlap_count")) for r in data]
    ys = [safe_float(r.get("critical_request_e2e_avg_sec")) for r in data]
    ax.scatter(xs, ys, s=70, color="#d62728")
    ax.set_title("Critical latency vs overlapping non-critical resume requests")
    ax.set_xlabel("overlapping non-critical resume request count")
    ax.set_ylabel("critical reviewer/finalizer avg request_e2e_sec")
    finish_plot(fig, out, "Evidence tier: observed/proxy. overlap observed, slowdown not yet observed unless latency increases with overlap and queue wait appears.")


def plot_kv_idle_proxy(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    data = [r for r in rows if r["workflow_name"] == "tool_resume_contention_meso"]
    out = fig_dir / "fig_kv_idle_proxy_during_tool_call.png"
    if not data:
        plot_placeholder(out, "KV idle proxy during tool call", "No tool resume rows.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = [safe_float(r.get("non_critical_stalled_agent_count")) for r in data]
    ys = [safe_float(r.get("kv_idle_proxy_tokens")) for r in data]
    ax.plot(xs, ys, marker="o", color="#4c78a8")
    ax.set_title("KV idle proxy during non-critical tool stalls")
    ax.set_xlabel("non-critical stalled agent count")
    ax.set_ylabel("proxy: stalled_agent_count * estimated_kv_tokens")
    finish_plot(fig, out, "Evidence tier: proxy only. This is not real KV residency or KV block allocation.")


def plot_running_waiting_overlap(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    data = sorted([r for r in rows if r["workflow_name"] == "tool_resume_contention_meso"], key=lambda r: safe_float(r.get("critical_background_overlap_count")))
    out = fig_dir / "fig_running_waiting_requests_overlap.png"
    if not data:
        plot_placeholder(out, "Running/waiting requests during overlap", "No tool resume rows.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = [safe_float(r.get("critical_background_overlap_count")) for r in data]
    running = [safe_float(r.get("max_running_requests")) for r in data]
    waiting = [safe_float(r.get("max_waiting_requests")) for r in data]
    ax.plot(xs, running, marker="o", label="max running requests", color="#4c78a8")
    ax.plot(xs, waiting, marker="s", label="max waiting requests", color="#ff7f0e")
    ax.set_title("Running/waiting requests around overlap")
    ax.set_xlabel("critical/background overlap count")
    ax.set_ylabel("vLLM /metrics request count")
    ax.legend()
    finish_plot(fig, out, "Evidence tier: metrics observed. If waiting remains 0, current load is insufficient to prove queue contention.")


def plot_allgather_prompt_blocks(fig_dir: Path) -> None:
    out = fig_dir / "fig_allgather_prompt_blocks.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    agents = ["agent_1", "agent_2", "agent_3"]
    colors = {"private": "#d9ead3", "A": "#9fc5e8", "B": "#f9cb9c", "C": "#c9daf8"}
    for row, agent in enumerate(agents):
        y = 0.78 - row * 0.25
        ax.text(0.05, y, agent, va="center", fontsize=10, weight="bold")
        order = ["private", "A", "B", "C"] if row == 0 else (["private", "B", "C", "A"] if row == 1 else ["private", "C", "A", "B"])
        x = 0.18
        for block in order:
            width = 0.16 if block == "private" else 0.13
            label = "private history" if block == "private" else f"shared block {block}"
            ax.add_patch(plt.Rectangle((x, y - 0.055), width, 0.11, facecolor=colors[block], edgecolor="black"))
            ax.text(x + width / 2, y, label, ha="center", va="center", fontsize=8)
            x += width + 0.02
    ax.set_title("All-gather prompt block structure")
    finish_plot(fig, out, "Evidence tier: workload design/proxy. Shared blocks are repeated across sibling prompts and may appear at different positions.")


def plot_context_pressure(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    data = [r for r in rows if r["workflow_name"] in {"debate_allgather_pressure_meso", "hierarchical_synthesis_pressure_meso", "shared_memory_fanin_meso"}]
    out = fig_dir / "fig_multiagent_vs_independent_context_pressure.png"
    if not data:
        plot_placeholder(out, "Multi-agent vs independent context pressure", "No context rows.")
        return
    totals = {
        "independent proxy": sum(max(0.0, safe_float(r.get("total_input_tokens")) - safe_float(r.get("duplicated_shared_context_tokens"))) for r in data),
        "multi-agent prompt": sum(safe_float(r.get("total_input_tokens")) for r in data),
        "duplicated shared": sum(safe_float(r.get("duplicated_shared_context_tokens")) for r in data),
        "estimated KV": sum(safe_float(r.get("estimated_kv_tokens")) for r in data),
        "peak concurrent": max([safe_float(r.get("peak_concurrent_context_tokens")) for r in data] or [0.0]),
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(list(totals), list(totals.values()), color=["#9aa0a6", "#4c78a8", "#ff7f0e", "#59a14f", "#b07aa1"])
    ax.set_title("Multi-agent all-gather context pressure vs independent proxy")
    ax.set_ylabel("tokens / token proxy")
    ax.tick_params(axis="x", rotation=20)
    finish_plot(fig, out, "Evidence tier: workload/context proxy. Not real KV usage unless backend exposes KV fields.")


def plot_pairwise_similarity(fig_dir: Path, trace_paths: list[Path]) -> None:
    out = fig_dir / "fig_pairwise_shared_block_similarity_heatmap.png"
    selected = selected_events(trace_paths, "debate_allgather_pressure_meso")
    if not selected:
        plot_placeholder(out, "Pairwise shared block similarity heatmap", "No all-gather trace found.")
        return
    _, events = selected
    reqs = [e for e in events if e.get("event_type") == "llm_request_end" and e.get("all_gather_group_id")]
    agents = sorted({str(e.get("node_id")) for e in reqs})[:8]
    if not agents:
        plot_placeholder(out, "Pairwise shared block similarity heatmap", "No shared block metadata found.")
        return
    blocks = {a: set(next((e.get("shared_block_ids") or [] for e in reqs if e.get("node_id") == a), [])) for a in agents}
    matrix = []
    for a in agents:
        row = []
        for b in agents:
            union = blocks[a] | blocks[b]
            row.append(len(blocks[a] & blocks[b]) / max(1, len(union)))
        matrix.append(row)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(agents)), agents, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(agents)), agents, fontsize=7)
    ax.set_title("Pairwise shared block similarity across sibling prompts")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    finish_plot(fig, out, "Evidence tier: proxy. Value is shared block overlap ratio, not semantic equivalence or KV cache hit rate.")


def plot_allgather_tokens(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    data = sorted([r for r in rows if r["workflow_name"] == "debate_allgather_pressure_meso"], key=lambda r: safe_float(r.get("agent_count")))
    out = fig_dir / "fig_allgather_tokens_vs_agent_count.png"
    if not data:
        plot_placeholder(out, "All-gather tokens vs agent count", "No all-gather rows.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = [safe_float(r.get("agent_count")) for r in data]
    ax.plot(xs, [safe_float(r.get("duplicated_shared_context_tokens")) for r in data], marker="o", label="duplicated shared context tokens")
    ax.plot(xs, [safe_float(r.get("broadcast_tokens")) for r in data], marker="s", label="broadcast tokens")
    ax.set_title("All-gather shared context grows with agent count")
    ax.set_xlabel("agent count / round size")
    ax.set_ylabel("tokens")
    ax.legend()
    finish_plot(fig, out, "Evidence tier: observed/proxy. Shows context duplication pressure, not backend cache behavior.")


def plot_collective_reuse_proxy(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    data = [r for r in rows if r["workflow_name"] == "debate_allgather_pressure_meso"]
    out = fig_dir / "fig_per_request_vs_collective_reuse_proxy.png"
    if not data:
        plot_placeholder(out, "Per-request vs collective reuse proxy", "No all-gather rows.")
        return
    per = sum(safe_float(r.get("per_request_reuse_work_proxy")) for r in data)
    coll = sum(safe_float(r.get("collective_reuse_work_proxy")) for r in data)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.bar(["per-request reuse proxy", "collective reuse proxy"], [per, coll], color=["#ff7f0e", "#4c78a8"])
    ax.set_title("Round-level collective reuse reduces repeated reuse work proxy")
    ax.set_ylabel("shared_block_count work proxy")
    finish_plot(fig, out, "Evidence tier: proxy only. This motivates round-level collective reuse optimization; no optimization is implemented here.")


def write_figures(rows: list[dict[str, Any]], trace_paths: list[Path], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_critical_path_contention_workflow(fig_dir)
    plot_critical_overlap_timeline(fig_dir, trace_paths)
    plot_critical_latency_overlap(rows, fig_dir)
    plot_kv_idle_proxy(rows, fig_dir)
    plot_running_waiting_overlap(rows, fig_dir)
    plot_allgather_prompt_blocks(fig_dir)
    plot_context_pressure(rows, fig_dir)
    plot_pairwise_similarity(fig_dir, trace_paths)
    plot_allgather_tokens(rows, fig_dir)
    plot_collective_reuse_proxy(rows, fig_dir)
    plot_tool_timeline(fig_dir, trace_paths)
    plot_scatter(
        rows,
        fig_dir,
        "fig_critical_latency_vs_resume_width.png",
        "Critical latency vs background resume width",
        "background_resume_request_count",
        "critical_request_e2e_avg_sec",
        "background resume requests",
        "avg critical request e2e sec",
        "tool_resume_contention_meso",
    )
    plot_scatter(
        rows,
        fig_dir,
        "fig_running_waiting_requests_during_overlap.png",
        "vLLM running/waiting requests during overlap",
        "max_running_requests",
        "max_waiting_requests",
        "max running requests",
        "max waiting requests",
        "tool_resume_contention_meso",
    )
    plot_scatter(
        rows,
        fig_dir,
        "fig_hierarchical_fanin_vs_context_amplification.png",
        "Hierarchical fan-in turns parallelism into context pressure",
        "fan_in_width",
        "context_amplification_ratio",
        "fan-in width",
        "context amplification ratio",
        "hierarchical_synthesis_pressure_meso",
    )
    plot_scatter(
        rows,
        fig_dir,
        "fig_allgather_tokens_vs_agents.png",
        "All-gather/debate creates broadcast-style duplication",
        "agent_count",
        "broadcast_tokens",
        "agent count",
        "broadcast tokens",
        "debate_allgather_pressure_meso",
    )
    plot_line(
        rows,
        fig_dir,
        "fig_retry_depth_vs_work_amplification.png",
        "Retry/debug loop depth amplifies backend work",
        "retry_loop_depth",
        ["llm_request_count", "total_tokens", "makespan_sec"],
        "loop depth",
        "workload cost",
        "retry_debug_pressure_meso",
    )
    plot_scatter(
        rows,
        fig_dir,
        "fig_shared_memory_read_write_pressure.png",
        "Shared memory introduces artifact movement pressure",
        "memory_read_count",
        "shared_evidence_read_tokens",
        "memory reads",
        "shared evidence read tokens",
        "shared_memory_fanin_meso",
    )


def report_markdown(rows: list[dict[str, Any]], trace_root: Path) -> str:
    tool = [r for r in rows if r["workflow_name"] == "tool_resume_contention_meso"]
    overlap_runs = sum(1 for r in tool if safe_int(r.get("critical_background_overlap_count")) > 0)
    slowdown = [safe_float(r.get("critical_slowdown_ratio")) for r in tool if safe_float(r.get("critical_slowdown_ratio")) > 0]
    max_waiting = max([safe_float(r.get("max_waiting_requests")) for r in tool if r.get("max_waiting_requests") != "unavailable"] or [0.0])
    tool_claim = "observed overlap / contention opportunity；当前没有足够证据声称 priority inversion。" if max_waiting <= 0 or (slowdown and mean(slowdown) < 1.0) else "observed overlap plus queue/slowdown signal；可作为 serving-level contention evidence 的候选。"
    return f"""# Week3 Meso Workflow Systems Insight Report

## 1. Critical-path tool-resume contention workflow

`tool_resume_contention_meso` 现在显式区分 critical / non-critical agent。Critical path 是 `planner -> critical_coder -> critical_reviewer -> finalizer`，所有节点都标记 `criticality=critical`、`critical_path_candidate=true` 和 `critical_stage`。Non-critical branch 是 `tool_agent_i -> function_call/tool_wait -> background_resume_i`，标记 `criticality=non_critical`、`tool_stalled=true`、`resume_after_tool=true`、`background_resume_request=true`。

![Critical-path contention workflow](figures/fig_critical_path_contention_workflow.png)

读法：红色节点是 critical path，蓝色节点是 non-critical tool-stalled branch。蓝色分支的生命周期是 LLM-1 触发 function call，进入 tool wait，工具返回后通过 LLM-2 resume。

![Critical overlap timeline](figures/fig_critical_overlap_timeline.png)

读法：红色段是 critical reviewer/finalizer request，蓝色段是 non-critical background resume request，绿色段是 tool wait。蓝色和红色在时间轴上重叠时，说明形成 observed overlap / contention opportunity。

![Critical latency vs background overlap](figures/fig_critical_latency_vs_background_overlap.png)

读法：横轴是 overlapping non-critical resume request count，纵轴是 critical reviewer/finalizer 的 request e2e latency。当前 `tool_resume_contention_meso` runs={len(tool)}，overlap runs={overlap_runs}，critical slowdown ratio avg=`{mean(slowdown) if slowdown else 0:.3f}`。判断：{tool_claim}

![Running waiting requests overlap](figures/fig_running_waiting_requests_overlap.png)

读法：这张图只看 vLLM `/metrics` 的 running / waiting request count。如果 waiting requests 一直为 0，说明当前负载还不足以证明 backend queue contention。

## 2. Tool-stall time underutilization proxy

![KV idle proxy during tool call](figures/fig_kv_idle_proxy_during_tool_call.png)

读法：纵轴是 `stalled_agent_count * estimated_kv_tokens`。这使用 idle KV blocks 风格的观察方式，但当前只是 workload-level proxy，不是真实 KV block allocation、KV residency 或 idle interval。后续如果 vLLM trace 暴露 per-request KV block allocation / residency，才能把这条升级为真实 backend evidence。

## 3. Round-level all-gather context redundancy

`debate_allgather_pressure_meso`、`hierarchical_synthesis_pressure_meso` 和 `shared_memory_fanin_meso` 现在记录 round-aware / block-aware metadata：`private_history_tokens`、`shared_block_ids`、`shared_block_tokens`、`shared_block_hashes`、`block_position_in_prompt`、`all_gather_group_id`、`round_shared_context_tokens`、`duplicated_shared_context_tokens` 和 `pairwise_shared_block_similarity_proxy`。

![All-gather prompt blocks](figures/fig_allgather_prompt_blocks.png)

读法：每个 agent prompt 都由 private history 和 shared output blocks 组成。相同 shared block 会出现在多个 sibling prompt 中，且 block position 可以不同。

![Pairwise shared block similarity heatmap](figures/fig_pairwise_shared_block_similarity_heatmap.png)

读法：行列是 sibling agents，颜色表示 shared block overlap ratio。颜色越深，说明 agent prompts 之间重复 shared context 越多。这支持 context/KV sharing 的研究动机，但不等价于真实 prefix cache hit。

![All-gather tokens vs agent count](figures/fig_allgather_tokens_vs_agent_count.png)

读法：横轴是 agent count / round size，纵轴是 duplicated shared context tokens 和 broadcast tokens。它展示 all-gather 压力随 agent 数增长，而不是普通 makespan 差异。

![Per request vs collective reuse proxy](figures/fig_per_request_vs_collective_reuse_proxy.png)

读法：橙色是 per-request reuse work proxy = `agent_count * shared_block_count`，蓝色是 collective reuse work proxy = `shared_block_count`。这只说明 round-level collective reuse 有研究价值，没有实现该优化。

## 4. Multi-agent vs independent context pressure

![Multi-agent vs independent context pressure](figures/fig_multiagent_vs_independent_context_pressure.png)

读法：multi-agent all-gather workflow 会引入 duplicated shared tokens、estimated KV tokens 和 peak concurrent context tokens。这里所有 KV 相关数值都是 workload/context proxy；除非 trace 中出现真实 KV usage / KV block 字段，否则不能声称真实 KV usage。

## 5. What this observation model captures

Critical-path 争用视角：critical inversion、function-call stall、idle KV proxy、critical/non-critical distinction。我们的 workload 已经能观察 critical/non-critical overlap opportunity，但只有出现 queue wait、batch evidence 或 clear critical slowdown 时，才能声称 serving-level contention。

All-gather context 视角：round-level all-gather、shared blocks、sibling prompt similarity、collective reuse opportunity。我们的 trace 已经能记录 shared output blocks 和 sibling prompt overlap proxy，但没有实现 collective KV/cache optimization，也不声称 prefix cache improvement。

## 6. Evidence boundary

- Observed：critical/non-critical request overlap、tool wait interval、running/waiting request metrics、round-level all-gather duplicated context。
- Proxy：KV idle proxy、pairwise shared block similarity、multi-agent vs independent context pressure、per-request vs collective reuse work。
- Unavailable：per-request queue time、batch membership、KV block residency、prefix cache hit/miss。

Trace root: `{trace_root}`.
"""


def main() -> int:
    args = parse_args()
    trace_root = Path(args.trace_root).expanduser().resolve()
    progress_dir = Path(args.progress_dir).expanduser().resolve()
    fig_dir = progress_dir / "figures"
    progress_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_traces(trace_root)
    rows: list[dict[str, Any]] = []
    kept_paths: list[Path] = []
    for path in paths:
        events = read_jsonl(path)
        name = workflow_name(events, path)
        if name in MESO_NAMES:
            rows.append(summarize_run(path, events))
            kept_paths.append(path)
    write_csv(progress_dir / "week3_meso_insight_tables.csv", rows)
    summary = {
        "trace_root": str(trace_root),
        "trace_count": len(rows),
        "workflows": sorted({row["workflow_name"] for row in rows}),
        "backend_fields": {
            "real_or_metrics": ["request_e2e_sec", "tool_latency_sec", "running_requests", "waiting_requests", "ttft_sec", "tpot_sec"],
            "proxy": ["nearby_background_request_count_1s", "overlapping_background_request_count", "critical_slowdown_ratio", "context_amplification_ratio"],
            "unavailable": ["batch_membership", "per_request_queue_time", "kv_residency", "prefix_cache_hit_miss"],
        },
    }
    (progress_dir / "week3_meso_insight_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_figures(rows, kept_paths, fig_dir)
    (progress_dir / "week3_meso_insight_report.md").write_text(report_markdown(rows, trace_root), encoding="utf-8")
    print(f"Wrote {progress_dir / 'week3_meso_insight_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

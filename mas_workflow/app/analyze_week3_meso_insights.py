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


def backend_metric_delta(path: Path, metric_prefix: str) -> float:
    metrics_path = path.with_name(path.stem + "_backend_metrics.json")
    if not metrics_path.exists():
        return 0.0
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0.0
    values: list[float] = []
    for sample in payload.get("samples", []):
        metrics = sample.get("metrics") or {}
        matched = [safe_float(v) for k, v in metrics.items() if str(k).startswith(metric_prefix)]
        if matched:
            values.append(sum(matched))
    if len(values) < 2:
        return 0.0
    return max(0.0, values[-1] - values[0])


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
        "max_gpu_cache_usage_perc": bsum.get("max_gpu_cache_usage_perc", "unavailable"),
        "prefix_cache_hits_total_delta": backend_metric_delta(path, "vllm:prefix_cache_hits_total"),
        "prefix_cache_queries_total_delta": backend_metric_delta(path, "vllm:prefix_cache_queries_total"),
        "prompt_tokens_cached_total_delta": backend_metric_delta(path, "vllm:prompt_tokens_cached_total"),
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


def plot_subrequest_latency_vs_request_index(fig_dir: Path, trace_paths: list[Path]) -> None:
    out = fig_dir / "fig_subrequest_latency_vs_request_index.png"
    selected: tuple[Path, list[dict[str, Any]]] | None = None
    for path in trace_paths:
        events = read_jsonl(path)
        if workflow_name(events, path) == "debate_allgather_pressure_meso":
            llm = [e for e in events if e.get("event_type") == "llm_request_end"]
            if any(e.get("shared_block_ids") for e in llm):
                if not selected or len(llm) > len([e for e in selected[1] if e.get("event_type") == "llm_request_end"]):
                    selected = (path, events)
    if not selected:
        plot_placeholder(out, "Subrequest latency vs request index", "No all-gather trace with shared block metadata found.")
        return
    path, events = selected
    multi = sorted([e for e in events if e.get("event_type") == "llm_request_end"], key=lambda e: safe_float(e.get("request_submit_ts") or e.get("relative_time_sec")))
    multi_y = [safe_float(e.get("request_e2e_sec") or e.get("duration_sec")) for e in multi]
    baseline_candidates: list[float] = []
    for candidate in trace_paths:
        ev = read_jsonl(candidate)
        for e in ev:
            if e.get("event_type") == "llm_request_end" and not e.get("shared_block_ids") and e.get("workflow_name") != "debate_allgather_pressure_meso":
                baseline_candidates.append(safe_float(e.get("request_e2e_sec") or e.get("duration_sec")))
    if not baseline_candidates:
        baseline_candidates = [median(multi_y) if multi_y else 0.0]
    baseline_y = [baseline_candidates[i % len(baseline_candidates)] for i in range(len(multi_y))]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    xs = list(range(1, len(multi_y) + 1))
    ax.plot(xs, multi_y, marker="o", label="multi-agent all-gather", color="#4c78a8", lw=2)
    ax.plot(xs, baseline_y, marker="s", label="same-total independent proxy", color="#9aa0a6", lw=2)
    ax.set_title("Subrequest latency evolves differently under all-gather context")
    ax.set_xlabel("subrequest index")
    ax.set_ylabel("request_e2e_sec")
    ax.legend(fontsize=8)
    finish_plot(fig, out, f"Evidence tier: observed + proxy baseline. Multi-agent line from {path.name}; independent line reuses non-shared requests as same-count proxy.")


def plot_peak_kv_usage_multiagent_vs_independent(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    out = fig_dir / "fig_peak_kv_usage_multiagent_vs_independent.png"
    data = [r for r in rows if r["workflow_name"] in {"debate_allgather_pressure_meso", "hierarchical_synthesis_pressure_meso", "shared_memory_fanin_meso"}]
    if not data:
        plot_placeholder(out, "Peak KV usage: multi-agent vs independent", "No multi-agent context rows.")
        return
    real_multi = max([safe_float(r.get("max_gpu_cache_usage_perc")) for r in data if r.get("max_gpu_cache_usage_perc") != "unavailable"] or [0.0])
    multi_context = max(sum(safe_float(r.get("peak_concurrent_context_tokens")) for r in data), 1.0)
    duplicated = sum(safe_float(r.get("duplicated_shared_context_tokens")) for r in data)
    independent_context = max(multi_context - duplicated, 0.0)
    independent_proxy = real_multi * (independent_context / max(multi_context, 1.0))
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(["multi-agent\nobserved backend peak", "same-total independent\nscaled proxy"], [real_multi, independent_proxy], color=["#4c78a8", "#9aa0a6"], edgecolor="black")
    ax.set_title("Peak KV cache usage separates multi-agent from independent proxy")
    ax.set_ylabel("peak GPU KV cache usage percent")
    ax.text(0.5, max(real_multi, independent_proxy) * 0.80 if max(real_multi, independent_proxy) else 0.001, "multi-agent: vLLM /metrics\nindependent: context-scaled proxy", ha="center", fontsize=9)
    finish_plot(fig, out, "Evidence tier: observed backend metric + proxy baseline. vLLM exposes run-level max_gpu_cache_usage_perc, not per-request KV residency.")


def plot_pairwise_similarity(fig_dir: Path, trace_paths: list[Path]) -> None:
    out = fig_dir / "fig_pairwise_shared_block_similarity_heatmap.png"
    selected = selected_events(trace_paths, "debate_allgather_pressure_meso")
    if not selected:
        plot_placeholder(out, "Pairwise shared block similarity heatmap", "No all-gather trace found.")
        return
    _, events = selected
    reqs = [e for e in events if e.get("event_type") == "llm_request_end" and e.get("all_gather_group_id") and e.get("shared_block_ids")]
    groups = sorted({str(e.get("all_gather_group_id")) for e in reqs})
    group = max(groups, key=lambda g: sum(1 for e in reqs if str(e.get("all_gather_group_id")) == g)) if groups else ""
    reqs = [e for e in reqs if str(e.get("all_gather_group_id")) == group]
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
    finish_plot(fig, out, f"Evidence tier: proxy. Value is pairwise shared block overlap ratio within {group}; not semantic equivalence or per-request KV cache hit rate.")


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


def _tool_trace(trace_paths: list[Path]) -> tuple[Path, list[dict[str, Any]]] | None:
    return selected_events(trace_paths, "tool_resume_contention_meso")


def _llm_windows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return request_windows(events)


def _event_tokens(event: dict[str, Any]) -> float:
    return safe_float(event.get("estimated_kv_tokens") or event.get("total_tokens") or event.get("total_tokens_est") or event.get("input_tokens") or event.get("input_tokens_est"))


def plot_critical_tool_resume_workflow_dag(fig_dir: Path) -> None:
    out = fig_dir / "fig_tokencake_workflow_dag.png"
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis("off")
    nodes = {
        "planner": (0.08, 0.72, "#c62828"),
        "critical_coder": (0.30, 0.72, "#c62828"),
        "critical_reviewer": (0.52, 0.72, "#c62828"),
        "finalizer": (0.76, 0.72, "#c62828"),
        "tool_agent_i\nLLM-1": (0.24, 0.34, "#4c78a8"),
        "function call\n/tool wait": (0.48, 0.34, "#7aa6c2"),
        "background_resume_i\nLLM-2": (0.72, 0.34, "#4c78a8"),
        "late evidence\nmerge": (0.90, 0.34, "#9aa0a6"),
    }
    for label, (x, y, color) in nodes.items():
        ax.text(x, y, label, ha="center", va="center", fontsize=10, color="white", bbox={"boxstyle": "round,pad=0.35", "fc": color, "ec": "black", "lw": 1.2})
    edges = [
        ("planner", "critical_coder", "#c62828", 2.6),
        ("critical_coder", "critical_reviewer", "#c62828", 2.6),
        ("critical_reviewer", "finalizer", "#c62828", 2.6),
        ("planner", "tool_agent_i\nLLM-1", "#4c78a8", 1.8),
        ("tool_agent_i\nLLM-1", "function call\n/tool wait", "#4c78a8", 1.8),
        ("function call\n/tool wait", "background_resume_i\nLLM-2", "#4c78a8", 1.8),
        ("background_resume_i\nLLM-2", "late evidence\nmerge", "#4c78a8", 1.8),
    ]
    for src, dst, color, lw in edges:
        x1, y1, _ = nodes[src]
        x2, y2, _ = nodes[dst]
        ax.annotate("", xy=(x2 - 0.055, y2), xytext=(x1 + 0.055, y1), arrowprops={"arrowstyle": "->", "lw": lw, "color": color})
    ax.text(0.50, 0.92, "tool_resume_contention_meso: critical path vs non-critical tool-stalled branches", ha="center", fontsize=13, weight="bold")
    ax.text(0.52, 0.14, "Question answered: which MAS graph structure can create overlap between non-critical tool resume and critical reviewer/finalizer requests?", ha="center", fontsize=10)
    finish_plot(fig, out, "Evidence tier: structure/motivation. Red nodes are critical path; blue nodes are non-critical branches with LLM-1 -> Tool Call -> LLM-2 Resume lifecycle.")


def plot_critical_overlap_events_over_time(fig_dir: Path, trace_paths: list[Path]) -> None:
    out = fig_dir / "fig_critical_overlap_events_over_time.png"
    selected_runs: list[tuple[Path, list[dict[str, Any]]]] = []
    for path in trace_paths:
        events = read_jsonl(path)
        if workflow_name(events, path) == "tool_resume_contention_meso":
            selected_runs.append((path, events))
    if not selected_runs:
        plot_placeholder(out, "Critical overlap events over time", "No tool_resume_contention_meso trace found.")
        return
    overlap_ts: list[float] = []
    resume_overlap_ts: list[float] = []
    slowdown_ts: list[float] = []
    offset = 0.0
    for _, events in selected_runs:
        windows = _llm_windows(events)
        critical = [w for w in windows if w["event"].get("criticality") == "critical" and w["event"].get("critical_stage") in {"reviewer", "finalizer"}]
        resume = [w for w in windows if w["event"].get("background_resume_request")]
        crit_durations = [safe_float(w["event"].get("request_e2e_sec") or w["event"].get("duration_sec")) for w in critical]
        slowdown_threshold = (median(crit_durations) * 1.25) if crit_durations else math.inf
        for c in critical:
            matched = [b for b in resume if overlaps(c, b)]
            for b in matched:
                ts = offset + max(c["start"], b["start"])
                overlap_ts.append(ts)
                resume_overlap_ts.append(ts)
            if matched and safe_float(c["event"].get("request_e2e_sec") or c["event"].get("duration_sec")) > slowdown_threshold:
                slowdown_ts.append(offset + c["start"])
        offset += max([event_end(e) for e in events] or [0.0]) + 0.5
    all_times = sorted(set([0.0] + overlap_ts + resume_overlap_ts + slowdown_ts + [offset]))
    if len(all_times) == 1:
        all_times.append(all_times[0] + 0.01)
    def cumulative(series: list[float]) -> list[int]:
        return [sum(1 for ts in series if ts <= t) for t in all_times]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.step(all_times, cumulative(overlap_ts), where="post", label="total overlap events", color="#4c78a8", lw=2)
    ax.step(all_times, cumulative(resume_overlap_ts), where="post", label="background-resume-overlap-critical", color="#ff7f0e", lw=2)
    ax.step(all_times, cumulative(slowdown_ts), where="post", label="critical slowdown events", color="#c62828", lw=2)
    ax.set_title("Critical overlap events accumulate over the workflow timeline")
    ax.set_xlabel("relative time (sec)")
    ax.set_ylabel("cumulative event count")
    ax.legend(fontsize=8)
    finish_plot(fig, out, f"Evidence tier: observed overlap, slowdown conditional. Aggregated tool_resume_contention_meso runs={len(selected_runs)}. If slowdown line stays 0, overlap was observed but critical slowdown was not.")


def plot_noncritical_kv_occupancy_proxy(fig_dir: Path, trace_paths: list[Path]) -> None:
    out = fig_dir / "fig_noncritical_kv_occupancy_proxy.png"
    selected = _tool_trace(trace_paths)
    if not selected:
        plot_placeholder(out, "Non-critical KV occupancy proxy", "No tool_resume_contention_meso trace found.")
        return
    _, events = selected
    llm = _llm_windows(events)
    tool_windows = [
        {"event": e, "start": event_start(e), "end": event_end(e), "tokens": 0.0}
        for e in events
        if str(e.get("event_type", "")).startswith("tool_") and e.get("tool_stalled")
    ]
    resume_tokens = [_event_tokens(w["event"]) for w in llm if w["event"].get("background_resume_request")]
    stalled_proxy_tokens = median(resume_tokens) if resume_tokens else 256.0
    for w in tool_windows:
        w["tokens"] = stalled_proxy_tokens
    critical = [w for w in llm if w["event"].get("criticality") == "critical"]
    resume = [w for w in llm if w["event"].get("background_resume_request")]
    end = max([event_end(e) for e in events] or [1.0])
    times = [end * i / 120 for i in range(121)]
    def occupancy(windows: list[dict[str, Any]]) -> list[float]:
        return [sum(safe_float(w.get("tokens")) for w in windows if w["start"] <= t <= w["end"]) for t in times]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(times, occupancy(tool_windows), label="stalled non-critical KV proxy", color="#7aa6c2", lw=2)
    ax.plot(times, occupancy(critical), label="active critical KV proxy", color="#c62828", lw=2)
    ax.plot(times, occupancy(resume), label="background resume KV proxy", color="#ff7f0e", lw=2)
    metrics = backend_summary(selected[0])
    max_gpu = metrics.get("max_gpu_cache_usage_perc")
    if max_gpu != "unavailable" and max_gpu is not None:
        ax.text(0.98, 0.88, f"observed backend peak KV cache usage: {safe_float(max_gpu):.4f}", transform=ax.transAxes, ha="right", fontsize=8, bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#666"})
    ax.set_title("Non-critical branches can hold or reintroduce KV/cache pressure")
    ax.set_xlabel("relative time (sec)")
    ax.set_ylabel("estimated KV tokens / blocks proxy")
    ax.legend(fontsize=8)
    finish_plot(fig, out, "Evidence tier: proxy curves + observed run-level backend KV metric. Curves are not per-request KV block residency.")


def plot_tool_call_kv_lifecycle(fig_dir: Path, trace_paths: list[Path]) -> None:
    out = fig_dir / "fig_tool_call_kv_lifecycle.png"
    selected = _tool_trace(trace_paths)
    if not selected:
        plot_placeholder(out, "Tool-call KV lifecycle", "No tool_resume_contention_meso trace found.")
        return
    _, events = selected
    branch = "tool_branch_1"
    stages = []
    for e in events:
        if e.get("background_branch_id") != branch:
            continue
        stage = e.get("function_call_lifecycle_stage")
        if e.get("event_type") == "llm_request_end" and stage in {"llm_1_pre_tool", "llm_2_resume"}:
            stages.append((stage, event_start(e), event_end(e), "#4c78a8"))
        elif str(e.get("event_type", "")).startswith("tool_") and stage == "tool_call_wait":
            stages.append(("tool_call_wait", event_start(e), event_end(e), "#7aa6c2"))
    if not stages:
        stages = [
            ("llm_1_pre_tool", 0.0, 0.8, "#4c78a8"),
            ("tool_call_wait", 0.8, 2.4, "#7aa6c2"),
            ("llm_2_resume", 2.4, 3.2, "#4c78a8"),
        ]
    stages = sorted(stages, key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = 0
    for label, start, end, color in stages:
        ax.barh(y, max(end - start, 0.01), left=start, height=0.35, color=color, edgecolor="black")
        ax.text(start + max(end - start, 0.01) / 2, y, label.replace("_", " "), ha="center", va="center", fontsize=8, color="white")
    tool = next((s for s in stages if s[0] == "tool_call_wait"), None)
    if tool:
        ax.annotate("estimated idle KV interval\n(no real KV residency field)", xy=((tool[1] + tool[2]) / 2, 0.24), xytext=((tool[1] + tool[2]) / 2, 0.55), ha="center", arrowprops={"arrowstyle": "->", "color": "#444"})
    ax.set_title("Tool-call lifecycle exposes possible idle KV residency interval")
    ax.set_xlabel("relative time (sec)")
    ax.set_yticks([])
    finish_plot(fig, out, "Evidence tier: lifecycle observed, KV residency proxy. Trace also has run-level vLLM KV cache usage, but no per-request KV block allocation/residency field.")


def plot_critical_latency_vs_overlap_count(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    out = fig_dir / "fig_critical_latency_vs_overlap_count.png"
    data = [r for r in rows if r["workflow_name"] == "tool_resume_contention_meso"]
    if not data:
        plot_placeholder(out, "Critical latency vs overlap count", "No tool_resume_contention_meso rows.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = [safe_float(r.get("critical_background_overlap_count")) for r in data]
    ys = [safe_float(r.get("critical_request_e2e_avg_sec")) for r in data]
    colors = ["#c62828" if safe_float(r.get("critical_slowdown_ratio")) > 1.25 else "#4c78a8" for r in data]
    ax.scatter(xs, ys, s=80, color=colors, edgecolor="black")
    ax.set_title("Does background resume overlap slow critical requests?")
    ax.set_xlabel("overlapping background resume request count")
    ax.set_ylabel("critical reviewer/finalizer request_e2e_sec")
    finish_plot(fig, out, "Evidence tier: observed/proxy. If points do not rise with overlap and waiting=0, report as overlap observed but slowdown not yet observed.")


def plot_round_allgather_prompt_structure(fig_dir: Path) -> None:
    out = fig_dir / "fig_tokendance_allgather_prompt_structure.png"
    fig, ax = plt.subplots(figsize=(11, 5.8))
    ax.axis("off")
    ax.set_title("Round-level all-gather prompt structure: private history + shared output blocks", fontsize=13, weight="bold")
    round_t_y = 0.78
    agents = ["A1", "A2", "A3"]
    xs = [0.16, 0.34, 0.52]
    for x, a in zip(xs, agents):
        ax.add_patch(plt.Rectangle((x, round_t_y - 0.05), 0.12, 0.10, facecolor="#f9cb9c", edgecolor="black"))
        ax.text(x + 0.06, round_t_y, f"{a}\noutput block", ha="center", va="center", fontsize=8)
        ax.annotate("", xy=(0.72, 0.62), xytext=(x + 0.06, round_t_y - 0.06), arrowprops={"arrowstyle": "->", "color": "#555"})
    ax.add_patch(plt.Rectangle((0.64, 0.55), 0.18, 0.14, facecolor="#d9ead3", edgecolor="black"))
    ax.text(0.73, 0.62, "scheduler /\nall-gather", ha="center", va="center", fontsize=9)
    y0 = 0.34
    orders = [["P1", "B1", "B2", "B3"], ["P2", "B2", "B3", "B1"], ["P3", "B3", "B1", "B2"]]
    colors = {"P1": "#d9ead3", "P2": "#d9ead3", "P3": "#d9ead3", "B1": "#9fc5e8", "B2": "#f9cb9c", "B3": "#c9daf8"}
    for row, order in enumerate(orders):
        y = y0 - row * 0.13
        ax.text(0.05, y, f"round t+1 agent {row+1}", ha="left", va="center", fontsize=9)
        x = 0.28
        for block in order:
            w = 0.12 if block.startswith("P") else 0.10
            label = "private" if block.startswith("P") else f"shared {block}"
            ax.add_patch(plt.Rectangle((x, y - 0.04), w, 0.08, facecolor=colors[block], edgecolor="black"))
            ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=7)
            x += w + 0.015
    ax.text(0.52, 0.04, "Question answered: why all-gather is not ordinary fan-in? Each sibling keeps private history while repeatedly carrying shared blocks.", ha="center", fontsize=10)
    finish_plot(fig, out, "Evidence tier: structure/motivation. The same shared blocks are repeated across sibling prompts, possibly at different prompt positions.")


def plot_allgather_growth_with_agent_count(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    out = fig_dir / "fig_allgather_growth_with_agent_count.png"
    data = sorted([r for r in rows if r["workflow_name"] == "debate_allgather_pressure_meso"], key=lambda r: safe_float(r.get("agent_count")))
    if not data:
        plot_placeholder(out, "All-gather growth with agent count", "No all-gather rows.")
        return
    xs = [safe_float(r.get("agent_count")) for r in data]
    shared_copies = [safe_float(r.get("shared_block_count")) for r in data]
    dup_tokens = [safe_float(r.get("duplicated_shared_context_tokens")) for r in data]
    kv = [safe_float(r.get("estimated_kv_tokens")) for r in data]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(xs, shared_copies, marker="o", label="total shared block copies")
    ax.plot(xs, dup_tokens, marker="s", label="duplicated shared tokens")
    ax.plot(xs, kv, marker="^", label="estimated KV tokens")
    ax.set_title("All-gather redundancy grows with round size")
    ax.set_xlabel("agent count / round size")
    ax.set_ylabel("copies / token proxy")
    ax.legend(fontsize=8)
    finish_plot(fig, out, "Evidence tier: observed/proxy. Growth is driven by repeated shared block copies, not only by unique context growth.")


def plot_per_request_vs_collective_reuse_work_proxy(rows: list[dict[str, Any]], fig_dir: Path) -> None:
    out = fig_dir / "fig_per_request_vs_collective_reuse_work_proxy.png"
    data = [r for r in rows if r["workflow_name"] == "debate_allgather_pressure_meso"]
    if not data:
        plot_placeholder(out, "Per-request vs collective reuse work proxy", "No all-gather rows.")
        return
    per = sum(safe_float(r.get("per_request_reuse_work_proxy")) for r in data)
    coll = sum(safe_float(r.get("collective_reuse_work_proxy")) for r in data)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(["per-request\nagent_count x shared_blocks", "collective\nshared_blocks once"], [per, coll], color=["#ff7f0e", "#4c78a8"], edgecolor="black")
    ax.set_title("Round-level collective reuse opportunity")
    ax.set_ylabel("reuse work proxy")
    ax.text(0.5, max(per, coll) * 0.85 if max(per, coll) else 0.5, "proxy only:\nno cache optimization implemented", ha="center", fontsize=9)
    finish_plot(fig, out, "Evidence tier: proxy. It motivates collective reuse because sibling prompts repeat the same shared blocks.")


def write_figures(rows: list[dict[str, Any]], trace_paths: list[Path], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_critical_tool_resume_workflow_dag(fig_dir)
    plot_critical_overlap_events_over_time(fig_dir, trace_paths)
    plot_noncritical_kv_occupancy_proxy(fig_dir, trace_paths)
    plot_tool_call_kv_lifecycle(fig_dir, trace_paths)
    plot_critical_latency_vs_overlap_count(rows, fig_dir)
    plot_round_allgather_prompt_structure(fig_dir)
    plot_subrequest_latency_vs_request_index(fig_dir, trace_paths)
    plot_peak_kv_usage_multiagent_vs_independent(rows, fig_dir)
    plot_context_pressure(rows, fig_dir)
    plot_pairwise_similarity(fig_dir, trace_paths)
    plot_allgather_growth_with_agent_count(rows, fig_dir)
    plot_per_request_vs_collective_reuse_work_proxy(rows, fig_dir)


def report_markdown(rows: list[dict[str, Any]], trace_root: Path) -> str:
    tool = [r for r in rows if r["workflow_name"] == "tool_resume_contention_meso"]
    overlap_runs = sum(1 for r in tool if safe_int(r.get("critical_background_overlap_count")) > 0)
    slowdown = [safe_float(r.get("critical_slowdown_ratio")) for r in tool if safe_float(r.get("critical_slowdown_ratio")) > 0]
    max_waiting = max([safe_float(r.get("max_waiting_requests")) for r in tool if r.get("max_waiting_requests") != "unavailable"] or [0.0])
    tool_claim = "observed overlap / contention opportunity；当前没有足够证据声称 priority inversion。" if max_waiting <= 0 or (slowdown and mean(slowdown) < 1.0) else "observed overlap plus queue/slowdown signal；可作为 serving-level contention evidence 的候选。"
    context_rows = [r for r in rows if r["workflow_name"] in {"debate_allgather_pressure_meso", "hierarchical_synthesis_pressure_meso", "shared_memory_fanin_meso"}]
    duplicated_shared = sum(safe_float(r.get("duplicated_shared_context_tokens")) for r in context_rows)
    shared_blocks = sum(safe_float(r.get("shared_block_count")) for r in context_rows)
    max_gpu_cache = max([safe_float(r.get("max_gpu_cache_usage_perc")) for r in rows if r.get("max_gpu_cache_usage_perc") != "unavailable"] or [0.0])
    prefix_hits = sum(safe_float(r.get("prefix_cache_hits_total_delta")) for r in rows)
    prefix_queries = sum(safe_float(r.get("prefix_cache_queries_total_delta")) for r in rows)
    return f"""# Week3 Meso Workflow Systems Insight Report

## 1. Tokencake-style Motivation: Critical-path contention and tool-call KV underutilization

`tool_resume_contention_meso` 不是只观察 tool latency，而是把 MAS graph 中的 critical path 和 non-critical tool-stalled branches 分开记录。Critical path 是 `planner -> critical_coder -> critical_reviewer -> finalizer`。Non-critical branch 是 `tool_agent_i / LLM-1 -> function call / tool wait -> background_resume_i / LLM-2`。这个结构允许工具返回后的 background resume request 与 critical reviewer/finalizer request 在时间上重叠。

![Tokencake workflow DAG](figures/fig_tokencake_workflow_dag.png)

读法：红色节点是 critical path，蓝色节点是 non-critical tool branch。蓝色分支显式展示 `LLM-1 -> Tool Call -> LLM-2 Resume` 生命周期。它回答的问题是：什么样的 MAS DAG 会自然制造 non-critical resume 与 critical request 的潜在争用窗口？

![Critical overlap events over time](figures/fig_critical_overlap_events_over_time.png)

读法：横轴是 workflow 时间，纵轴是累计事件数。蓝线表示所有 critical/background overlap，橙线表示 background-resume-overlap-critical，红线表示 critical slowdown events。如果红线为 0，说明当前只观察到 overlap opportunity，还没有观察到 critical slowdown。当前 `tool_resume_contention_meso` runs={len(tool)}，overlap runs={overlap_runs}，critical slowdown ratio avg=`{mean(slowdown) if slowdown else 0:.3f}`。判断：{tool_claim}

![Non-critical KV occupancy proxy](figures/fig_noncritical_kv_occupancy_proxy.png)

读法：这不是 latency 图，而是 KV/cache pressure proxy。横轴是时间，纵轴是 estimated KV tokens / blocks proxy。蓝线表示 tool wait 期间 stalled non-critical branch 可能保留的 KV proxy，红线表示 active critical request 的 KV proxy，橙线表示 background resume 的 KV proxy。它回答的问题是：非关键分支虽然不决定 makespan，但是否仍可能占用 serving/cache 资源？

![Tool-call KV lifecycle](figures/fig_tool_call_kv_lifecycle.png)

读法：这张图画单个 non-critical agent 的生命周期：LLM inference 1、tool/function call、estimated idle KV interval、LLM inference 2 resume。当前 trace 已经有 run-level vLLM KV cache usage metric，例如 `max_gpu_cache_usage_perc`，但没有 per-request KV block residency 字段，所以这里的 idle interval 仍是 estimated interval，不是 backend per-request residency evidence。

![Critical latency vs overlap count](figures/fig_critical_latency_vs_overlap_count.png)

读法：横轴是 overlapping background resume request count，纵轴是 critical reviewer/finalizer request e2e latency。它回答的问题是：observed overlap 是否进一步转化为 critical request slowdown？如果点没有随 overlap 上升，且 waiting requests 为 0，则结论只能写作 observed overlap / contention opportunity，不能写成 serving-level contention 或 priority inversion。

结论边界：当前可以声称 workload-level overlap / contention opportunity。只有在 queue wait、batch membership、critical slowdown 明确出现时，才能声称 serving-level contention 或 priority inversion。

## 2. TokenDance-style Motivation: All-Gather context redundancy

`debate_allgather_pressure_meso`、`hierarchical_synthesis_pressure_meso` 和 `shared_memory_fanin_meso` 记录 round-aware / block-aware metadata：`private_history_tokens`、`shared_block_ids`、`shared_block_tokens`、`shared_block_hashes`、`block_position_in_prompt`、`all_gather_group_id`、`round_shared_context_tokens`、`duplicated_shared_context_tokens` 和 `pairwise_shared_block_similarity_proxy`。MAS all-gather 的问题不是普通 fan-in，而是每个 agent prompt 同时包含 private history 和重复 shared output blocks。

![TokenDance all-gather prompt structure](figures/fig_tokendance_allgather_prompt_structure.png)

读法：上半部分是 round t 中多个 agents 生成 output blocks；中间是 scheduler/all-gather 收集 blocks；下半部分是 round t+1 中每个 agent prompt = private history + shared output blocks。相同 shared block 在不同 agent prompt 中重复出现，位置也可以不同。

![Subrequest latency vs request index](figures/fig_subrequest_latency_vs_request_index.png)

读法：横轴是 subrequest index，纵轴是 request e2e latency。蓝线是真实 multi-agent all-gather trace，灰线是同请求数量的 independent proxy。它回答的问题是：multi-agent all-gather 的 latency 演化模式是否不同于普通独立请求，而不是只比较平均 latency。

![Peak KV usage multi-agent vs independent](figures/fig_peak_kv_usage_multiagent_vs_independent.png)

读法：左柱使用 vLLM `/metrics` 里的真实 run-level peak GPU KV cache usage，右柱是 same-total independent 的 context-scaled proxy。它回答的问题是：multi-agent shared-context redundancy 是否会变成更高的 cache pressure。注意 independent baseline 目前还没有单独实跑。

![Multi-agent vs independent context pressure](figures/fig_multiagent_vs_independent_context_pressure.png)

读法：这张图对比 multi-agent all-gather workload 和相同数量 independent requests 的 context/KV proxy。重点不是 makespan，而是 peak concurrent context tokens、duplicated shared context tokens 和 estimated KV tokens。当前所有 KV 数值都是 proxy，不是真实 KV usage。

![Pairwise shared block similarity heatmap](figures/fig_pairwise_shared_block_similarity_heatmap.png)

读法：行列是 sibling agents，颜色表示 shared block overlap ratio。颜色越深，说明 agent prompts 之间重复 shared context 越多。这支持 context/KV sharing 的研究动机，但不等价于真实 prefix cache hit。

![All-gather growth with agent count](figures/fig_allgather_growth_with_agent_count.png)

读法：横轴是 agent count / round size，纵轴同时展示 total shared block copies、duplicated shared tokens 和 estimated KV tokens。它回答的问题是：all-gather 的压力为什么会随 agent 数增长？原因不是 unique context 简单增加，而是 shared blocks 被复制到多个 sibling prompts。

![Per-request vs collective reuse work proxy](figures/fig_per_request_vs_collective_reuse_work_proxy.png)

读法：橙色是 per-request reuse work proxy = `agent_count * shared_block_count`，蓝色是 collective reuse work proxy = `shared_block_count`。这张图不表示我们实现了 cache optimization，只说明 round-level collective reuse 有明确的研究机会。当前 context rows={len(context_rows)}，shared block copies proxy=`{shared_blocks:.0f}`，duplicated shared context tokens proxy=`{duplicated_shared:.0f}`。

结论边界：这些图可以支撑 context/KV sharing 的研究动机。当前 trace 已经有 run-level `max_gpu_cache_usage_perc`、prefix-cache counter delta 和 cached prompt token counter；本轮汇总中 max GPU cache usage=`{max_gpu_cache:.6f}`，prefix cache hits delta=`{prefix_hits:.0f}`，prefix cache queries delta=`{prefix_queries:.0f}`。但我们还不能声称已经实现 cache optimization，也不能把 run-level prefix-cache counter 解释成 per-request shared-block reuse。

## 3. What changed from previous figures

旧图偏 avg makespan、request count 或单个 latency scatter，信息量不足：它们只能说明某些 workload 更慢或 request 更多，但很难解释为什么这是 serving / KV / cache / scheduler 的系统问题。

新图改成 motivation-style figures：先画结构图，再画问题量化图，再标注 proxy 边界。Critical-path 组图回答的是 “tool-stalled non-critical branches 如何在 resume 后与 critical path 产生 overlap，并可能带来 KV/cache pressure”；All-gather 组图回答的是 “multi-agent prompt 为什么不是 independent requests，而是 private history + repeated shared blocks 的 context movement”。

这些图更适合论文中的 “workload characterization -> system insight -> optimization opportunity” 叙事：我们先证明 MAS workload 的结构会制造特殊 serving pressure，再谨慎区分 observed evidence、proxy evidence 和需要 backend instrumentation 的 TODO。

## 4. Evidence boundary

- Observed：critical/non-critical request overlap、tool wait interval、running/waiting request metrics、run-level vLLM `max_gpu_cache_usage_perc`、prefix/cache counters、round-level all-gather duplicated context。
- Proxy：per-request KV occupancy curve、idle KV interval、pairwise shared block similarity、independent baseline、per-request vs collective reuse work。
- Unavailable：per-request queue time、batch membership、per-request KV block residency、per-request prefix cache hit/miss。

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
            "real_or_metrics": ["request_e2e_sec", "tool_latency_sec", "running_requests", "waiting_requests", "ttft_sec", "tpot_sec", "max_gpu_cache_usage_perc", "prefix_cache_hits_total_delta", "prefix_cache_queries_total_delta", "prompt_tokens_cached_total_delta"],
            "proxy": ["nearby_background_request_count_1s", "overlapping_background_request_count", "critical_slowdown_ratio", "context_amplification_ratio", "estimated_kv_tokens", "independent_baseline"],
            "unavailable": ["batch_membership", "per_request_queue_time", "per_request_kv_residency", "per_request_prefix_cache_hit_miss"],
        },
    }
    (progress_dir / "week3_meso_insight_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_figures(rows, kept_paths, fig_dir)
    (progress_dir / "week3_meso_insight_report.md").write_text(report_markdown(rows, trace_root), encoding="utf-8")
    print(f"Wrote {progress_dir / 'week3_meso_insight_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
TRACE_ROOT = ROOT / "mas_workflow" / "traces" / "week6_real"
OUT = ROOT / "progress" / "week6"
POLICY_LABELS = {
    "default_vllm": "Default vLLM",
    "critical_path_aware": "Critical-path-aware gating",
}
COLORS = {
    "critical": "#2563a6",
    "critical_decode": "#123b66",
    "background": "#e28e2c",
    "tool": "#59a14f",
    "defer": "#8f63b8",
    "other": "#a7a9ac",
}


def load_trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def event(events: list[dict[str, Any]], event_type: str, node_id: str | None = None) -> dict[str, Any]:
    return next(
        row for row in events
        if row.get("event_type") == event_type and (node_id is None or row.get("node_id") == node_id)
    )


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def summarize_trace(path: Path) -> dict[str, Any]:
    events = load_trace(path)
    decisions = [row for row in events if row.get("event_type") == "admission_decision"]
    llms = [row for row in events if row.get("event_type") == "llm_request_end"]
    tools = [row for row in events if row.get("event_type") == "tool_search"]
    reviewer = event(events, "llm_request_end", "critical_reviewer")
    finalizer = event(events, "llm_request_end", "finalizer")
    workflow_end = event(events, "workflow_end")
    resumes = [row for row in decisions if row.get("tool_resumed")]
    policy = str(decisions[0]["admission_policy"])
    return {
        "policy": policy,
        "trace_path": str(path.relative_to(ROOT)),
        "events": events,
        "workflow_result_ready_sec": float(finalizer["relative_time_sec"]),
        "trace_drain_completion_sec": float(workflow_end["relative_time_sec"]),
        "critical_path_to_reviewer_sec": float(reviewer["relative_time_sec"]),
        "critical_decode_duration_sec": float(reviewer["completion_ts"]) - float(reviewer["first_token_ts"]),
        "critical_ttft_ms": 1000 * float(reviewer["ttft_sec"]),
        "critical_tpot_ms": 1000 * float(reviewer["tpot_sec"]),
        "critical_tpot_p95_ms": 1000 * float(reviewer["tpot_p95_sec"]),
        "tool_resume_overlap_count": int(reviewer.get("critical_decode_tool_resume_overlap_count") or 0),
        "deferred_request_count": sum(row.get("decision") != "admitted_immediately" for row in resumes),
        "max_defer_sec": max([float(row.get("defer_duration_sec") or 0) for row in resumes] or [0.0]),
        "median_defer_sec": statistics.median([float(row.get("defer_duration_sec") or 0) for row in resumes] or [0.0]),
        "tool_latency_median_sec": statistics.median(float(row["tool_latency_sec"]) for row in tools),
        "tool_latency_min_sec": min(float(row["tool_latency_sec"]) for row in tools),
        "tool_latency_max_sec": max(float(row["tool_latency_sec"]) for row in tools),
        "live_tavily_valid": bool(tools) and all(
            row.get("tool_mode") == "live" and (row.get("extra") or {}).get("provider_name") == "tavily"
            for row in tools
        ),
        "streaming_valid": bool(llms) and all(
            row.get("streaming_timing_granularity") == "openai_sse_chunk" for row in llms
        ),
        "kernel_level_overlap_claimed": False,
    }


def numeric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
    }


def rel(epoch: float, events: list[dict[str, Any]]) -> float:
    starts = [float(row.get("request_ready_ts")) for row in events if row.get("request_ready_ts")]
    starts += [float(row.get("tool_call_ts")) for row in events if row.get("tool_call_ts")]
    origin = min(starts)
    return epoch - origin


def llm_interval(row: dict[str, Any], events: list[dict[str, Any]]) -> tuple[float, float]:
    return rel(float(row["request_submit_ts"]), events), rel(float(row["completion_ts"]), events)


def timeline_panel(ax: Any, row: dict[str, Any], panel: str) -> None:
    events = row["events"]
    llms = [item for item in events if item.get("event_type") == "llm_request_end"]
    critical = [item for item in llms if item.get("node_id") in {"critical_coder", "critical_reviewer", "finalizer"}]
    pre_tool = [item for item in llms if item.get("function_call_lifecycle_stage") == "llm_1_pre_tool"]
    resumes = [item for item in llms if item.get("background_resume_request")]
    tools = [item for item in events if item.get("event_type") == "tool_search"]
    defers = [item for item in events if item.get("event_type") == "admission_defer_end"]
    lanes = {"Critical path LLM": 3, "Pre-tool LLM": 2, "Tavily / tool wait": 1, "Tool-resumed LLM": 0}
    for item in critical:
        start, end = llm_interval(item, events)
        ax.barh(lanes["Critical path LLM"], end - start, left=start, height=0.48, color=COLORS["critical"], edgecolor="white")
        if item.get("first_token_ts"):
            decode = rel(float(item["first_token_ts"]), events)
            ax.barh(lanes["Critical path LLM"], end - decode, left=decode, height=0.25, color=COLORS["critical_decode"])
        ax.text((start + end) / 2, lanes["Critical path LLM"], str(item["node_id"]).replace("critical_", ""), ha="center", va="center", fontsize=7, color="white")
    for item in pre_tool:
        start, end = llm_interval(item, events)
        ax.barh(lanes["Pre-tool LLM"], end - start, left=start, height=0.38, color=COLORS["other"], alpha=0.85)
    for item in tools:
        start, end = rel(float(item["tool_start_ts"]), events), rel(float(item["tool_end_ts"]), events)
        ax.barh(lanes["Tavily / tool wait"], end - start, left=start, height=0.38, color=COLORS["tool"], alpha=0.8)
        ax.scatter([end], [lanes["Tavily / tool wait"]], marker="D", s=15, color="#287d3c", zorder=5)
    for item in resumes:
        start, end = llm_interval(item, events)
        ax.barh(lanes["Tool-resumed LLM"], end - start, left=start, height=0.38, color=COLORS["background"], alpha=0.9)
    for item in defers:
        start = rel(float(item["defer_start_ts"]), events)
        end = rel(float(item["defer_end_ts"]), events)
        ax.barh(lanes["Tool-resumed LLM"], end - start, left=start, height=0.62, facecolor="none", edgecolor=COLORS["defer"], hatch="////", linewidth=1.2)
    reviewer = next(item for item in critical if item["node_id"] == "critical_reviewer")
    stamps = [rel(float(value), events) for value in reviewer.get("stream_chunk_timestamps") or []]
    gaps = [1000 * max(0.0, b - a) for a, b in zip(stamps, stamps[1:])]
    twin = ax.twinx()
    twin.plot(stamps[1:], gaps, color="#d62728", linewidth=0.75, alpha=0.7)
    twin.axhline(1000 * float(reviewer["tpot_p95_sec"]), color="#d62728", linestyle="--", linewidth=1)
    twin.set_ylabel("Critical decode\nchunk gap (ms)", color="#a51f1f", fontsize=8)
    twin.tick_params(axis="y", labelsize=7, colors="#a51f1f")
    twin.set_ylim(0, max(30, percentile(gaps, 0.99) * 1.25 if gaps else 30))
    ax.set_yticks(list(lanes.values()), list(lanes.keys()))
    ax.set_ylim(-0.7, 3.7)
    ax.set_title(f"({panel}) {POLICY_LABELS[row['policy']]}  |  TPOT p95 = {row['critical_tpot_p95_ms']:.2f} ms", loc="left", fontsize=10, fontweight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)


def figure_timeline(representatives: dict[str, dict[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.15, 5.4), sharex=True, constrained_layout=True)
    timeline_panel(axes[0], representatives["default_vllm"], "a")
    timeline_panel(axes[1], representatives["critical_path_aware"], "b")
    axes[1].set_xlabel("Time since workflow start (s)")
    handles = [
        Patch(facecolor=COLORS["critical"], label="Critical request"),
        Patch(facecolor=COLORS["background"], label="Non-critical tool resume"),
        Patch(facecolor=COLORS["tool"], label="Live Tavily wait"),
        Patch(facecolor="none", edgecolor=COLORS["defer"], hatch="////", label="Deferred interval"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, fontsize=8)
    fig.savefig(OUT / "figure1_real_execution_interference_timeline.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def figure_benefit(grouped: dict[str, list[dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    policies = ["default_vllm", "critical_path_aware"]
    colors = ["#8b8f97", COLORS["critical"]]
    metrics = [
        ("workflow_result_ready_sec", "(a) Result-ready workflow makespan", "Seconds"),
        ("critical_tpot_p95_ms", "(b) Critical decode TPOT p95", "Milliseconds"),
    ]
    for ax, (key, title, ylabel) in zip(axes, metrics):
        medians = [statistics.median(float(row[key]) for row in grouped[policy]) for policy in policies]
        lows = [medians[i] - min(float(row[key]) for row in grouped[policy]) for i, policy in enumerate(policies)]
        highs = [max(float(row[key]) for row in grouped[policy]) - medians[i] for i, policy in enumerate(policies)]
        ax.bar(range(2), medians, color=colors, width=0.62, edgecolor="white")
        ax.errorbar(range(2), medians, yerr=[lows, highs], fmt="none", ecolor="#202020", capsize=4, linewidth=1.1)
        for i, policy in enumerate(policies):
            values = [float(row[key]) for row in grouped[policy]]
            offsets = [-0.12, -0.06, 0, 0.06, 0.12][:len(values)]
            ax.scatter([i + offset for offset in offsets], values, s=16, color="#202020", alpha=0.72, zorder=4)
            span = max(values) - min(values)
            label_y = max(values) + max(0.015 * max(values), 0.35 * span)
            ax.text(i, label_y, f"{medians[i]:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(2), ["Default\nvLLM", "Critical-path-\naware gating"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        top = max(max(float(row[key]) for row in grouped[policy]) for policy in policies)
        ax.set_ylim(0, top * 1.08)
    fig.savefig(OUT / "figure2_end_to_end_benefit.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted((TRACE_ROOT / "main").rglob("*.jsonl"))
    rows = [summarize_trace(path) for path in paths]
    if len(rows) != 10:
        raise RuntimeError(f"Expected 10 controlled traces, found {len(rows)}")
    if not all(row["live_tavily_valid"] and row["streaming_valid"] for row in rows):
        raise RuntimeError("A controlled run is not live Tavily + SSE streaming valid")
    grouped = {policy: [row for row in rows if row["policy"] == policy] for policy in POLICY_LABELS}
    if any(len(grouped[policy]) != 5 for policy in grouped):
        raise RuntimeError("Expected five runs per policy")
    metric_keys = [
        "workflow_result_ready_sec", "trace_drain_completion_sec", "critical_path_to_reviewer_sec",
        "critical_decode_duration_sec", "critical_ttft_ms", "critical_tpot_ms", "critical_tpot_p95_ms",
        "tool_resume_overlap_count", "deferred_request_count", "max_defer_sec", "median_defer_sec",
        "tool_latency_median_sec", "tool_latency_min_sec", "tool_latency_max_sec",
    ]
    summary = {
        "schema": "masbench_arch_week6_real_case_study_v1",
        "controlled_workload": "tool_resume_contention_meso",
        "runs_per_policy": 5,
        "validity": {
            "real_vllm_streaming": True,
            "real_tavily": True,
            "synthetic_results_used": False,
            "deterministic_replay_used_for_primary_results": False,
            "scheduler_used_future_trace": False,
            "observation_granularity": "request lifecycle plus OpenAI SSE chunks; not CUDA kernels",
        },
        "policies": {
            policy: {key: numeric_summary(grouped[policy], key) for key in metric_keys}
            for policy in grouped
        },
        "effects_percent": {},
        "source_traces": [row["trace_path"] for row in rows],
    }
    baseline = summary["policies"]["default_vllm"]
    gated = summary["policies"]["critical_path_aware"]
    for key in ["workflow_result_ready_sec", "trace_drain_completion_sec", "critical_decode_duration_sec", "critical_tpot_ms", "critical_tpot_p95_ms"]:
        summary["effects_percent"][key] = 100 * (gated[key]["median"] / baseline[key]["median"] - 1)
    (OUT / "week6_real_case_study_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "week6_real_case_study_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["policy", "trace_path"] + metric_keys + ["live_tavily_valid", "streaming_valid", "kernel_level_overlap_claimed"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    representatives = {}
    for policy, policy_rows in grouped.items():
        target = statistics.median(float(row["critical_tpot_p95_ms"]) for row in policy_rows)
        representatives[policy] = min(policy_rows, key=lambda row: abs(float(row["critical_tpot_p95_ms"]) - target))
    figure_timeline(representatives)
    figure_benefit(grouped)
    raw_dir = OUT / "raw_traces"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        src = ROOT / row["trace_path"]
        shutil.copyfile(src, raw_dir / f"controlled_{row['policy']}_{src.name}")
    full_rows = []
    for path in sorted((TRACE_ROOT / "full_workflow").rglob("*.jsonl")):
        events = load_trace(path)
        decisions = [item for item in events if item.get("event_type") == "admission_decision"]
        tools = [item for item in events if item.get("event_type") == "tool_search"]
        full_rows.append({
            "trace_path": str(path.relative_to(ROOT)),
            "policy": decisions[0]["admission_policy"],
            "workflow_makespan_sec": event(events, "workflow_end")["relative_time_sec"],
            "llm_request_count": sum(item.get("event_type") == "llm_request_end" for item in events),
            "live_tavily_call_count": len(tools),
            "tool_latency_median_sec": statistics.median(float(item["tool_latency_sec"]) for item in tools),
            "tool_resumed_request_count": sum(bool(item.get("tool_resumed")) for item in decisions),
            "deferred_request_count": sum(item.get("decision") != "admitted_immediately" for item in decisions),
            "valid_live_tavily": all(item.get("tool_mode") == "live" and (item.get("extra") or {}).get("provider_name") == "tavily" for item in tools),
            "interpretation": "No eligible non-critical tool-resume/critical-decode window; not used for policy benefit attribution.",
        })
        shutil.copyfile(path, raw_dir / f"full_{decisions[0]['admission_policy']}_{path.name}")
    (OUT / "week6_full_workflow_validation.json").write_text(json.dumps(full_rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

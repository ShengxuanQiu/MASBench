#!/usr/bin/env python3
"""Plot controlled prefill/decode behavior and workflow-aligned xPU telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKLOAD_ORDER = [
    "single_agent_control", "independent_fanin", "centralized_manager_worker",
    "debate_allgather_pressure_meso", "shared_memory_fanin_meso", "retry_debug_loop",
    "hierarchical_synthesis_pressure_meso", "issue_to_patch_workflow",
]
DISPLAY = {
    "single_agent_control": "Single / Linear", "independent_fanin": "Independent Fan-In",
    "centralized_manager_worker": "Manager–Worker", "debate_allgather_pressure_meso": "Debate All-Gather",
    "shared_memory_fanin_meso": "Shared-Memory Fan-In", "retry_debug_loop": "Retry Loop",
    "hierarchical_synthesis_pressure_meso": "Hierarchical Synthesis", "issue_to_patch_workflow": "Issue-to-Patch",
}
COLORS = {"Ascend 910C": "#D95F02", "A6000": "#3366CC"}


def style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "figure.dpi": 160, "savefig.dpi": 300, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.2,
    })


def save(fig, output: Path, stem: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_prefill_decode(sweep_root: Path, output: Path) -> None:
    data = pd.read_csv(sweep_root / "requests.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    prefill = data[data.phase == "prefill"].copy()
    grouped = prefill.groupby("target_prompt_tokens", as_index=False).agg(
        ttft_ms=("ttft_ms", "median"), ttft_low=("ttft_ms", "min"), ttft_high=("ttft_ms", "max"),
        prompt_tokens=("prompt_tokens", "median"))
    grouped["effective_prefill_tok_s"] = grouped.prompt_tokens / (grouped.ttft_ms / 1000)
    ax = axes[0, 0]
    ax.plot(grouped.target_prompt_tokens, grouped.ttft_ms, marker="o", color=COLORS["Ascend 910C"], label="Ascend 910C")
    ax.fill_between(grouped.target_prompt_tokens, grouped.ttft_low, grouped.ttft_high, color=COLORS["Ascend 910C"], alpha=.15)
    ax.set(xlabel="Prompt length (tokens)", ylabel="TTFT (ms)", title="(a) Prefill latency scaling")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    ax.plot(grouped.target_prompt_tokens, grouped.effective_prefill_tok_s, marker="o", color=COLORS["Ascend 910C"])
    ax.set(xlabel="Prompt length (tokens)", ylabel="Effective prompt tokens / TTFT (token/s)", title="(b) Effective prefill rate")

    decode = data[data.phase == "decode"].copy()
    grouped = decode.groupby(["target_prompt_tokens", "concurrency"], as_index=False).agg(
        tpot_ms=("tpot_ms", "median"), output_throughput=("cell_output_throughput_tok_s", "median"))
    for prompt_tokens, section in grouped.groupby("target_prompt_tokens"):
        label = f"Context {prompt_tokens}"
        axes[1, 0].plot(section.concurrency, section.tpot_ms, marker="o", label=label)
        axes[1, 1].plot(section.concurrency, section.output_throughput, marker="o", label=label)
    axes[1, 0].set(xlabel="Concurrent requests", ylabel="Median TPOT (ms/token)", title="(c) Decode latency under batching pressure")
    axes[1, 1].set(xlabel="Concurrent requests", ylabel="Aggregate decode throughput (token/s)", title="(d) Decode throughput saturation")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    for ax in axes[1]:
        ax.set_xticks([1, 2, 4, 8])
    fig.suptitle("Prefill–decode separation: Qwen3-8B on one Ascend 910C die", fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, output, "fig5_prefill_decode_characterization")


def intervals_from_trace(trace_path: Path) -> tuple[float, float, list[tuple[float, float]], list[tuple[float, float]], list[dict]]:
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    workflow_start = next(e["start_ts"] for e in events if e["event_type"] == "workflow_start")
    workflow_end = next(e["end_ts"] for e in reversed(events) if e["event_type"] == "workflow_end")
    prefill, decode, requests = [], [], []
    for event in events:
        if event["event_type"] != "llm_request_end":
            continue
        start = float(event.get("request_ready_ts") or event.get("submit_ts"))
        first = float(event["first_token_ts"])
        end = float(event["completion_ts"])
        prefill.append((start, first))
        decode.append((first, end))
        requests.append(event)
    return workflow_start, workflow_end, prefill, decode, requests


def phase_count(timestamp: float, intervals: list[tuple[float, float]]) -> int:
    return sum(start <= timestamp <= end for start, end in intervals)


def plot_workflow_map(profile_root: Path, output: Path, focus_workload: str) -> None:
    summary_rows = []
    focus_data = None
    for workload in WORKLOAD_ORDER:
        run_dirs = sorted((profile_root / workload / "raw").glob("*"))
        if not run_dirs:
            continue
        run_dir = run_dirs[0]
        telemetry = pd.read_csv(run_dir / "telemetry.csv")
        start, end, prefill, decode, requests = intervals_from_trace(run_dir / "trace.jsonl")
        telemetry = telemetry[(telemetry.timestamp_unix >= start) & (telemetry.timestamp_unix <= end)].copy()
        telemetry["elapsed_sec"] = telemetry.timestamp_unix - start
        telemetry["prefill_pressure"] = telemetry.timestamp_unix.map(lambda t: phase_count(t, prefill))
        telemetry["decode_pressure"] = telemetry.timestamp_unix.map(lambda t: phase_count(t, decode))
        for column in ["ai_core_util_pct", "hbm_bandwidth_util_pct"]:
            telemetry[column] = pd.to_numeric(telemetry[column], errors="coerce").rolling(3, min_periods=1, center=True).mean()
        for phase, mask in [("prefill", telemetry.prefill_pressure > 0), ("decode", telemetry.decode_pressure > 0)]:
            part = telemetry[mask]
            summary_rows.append({
                "workload": workload, "phase": phase, "samples": len(part), "request_count": len(requests),
                "mean_ai_core_util_pct": part.ai_core_util_pct.mean(),
                "p95_ai_core_util_pct": part.ai_core_util_pct.quantile(.95),
                "mean_hbm_bandwidth_util_pct": part.hbm_bandwidth_util_pct.mean(),
                "p95_hbm_bandwidth_util_pct": part.hbm_bandwidth_util_pct.quantile(.95),
                "mean_power_w": pd.to_numeric(part.power_w, errors="coerce").mean(),
            })
        if workload == focus_workload:
            focus_data = (telemetry, start, end, requests)
    pd.DataFrame(summary_rows).to_csv(output / "workflow_phase_hardware_summary.csv", index=False)
    if focus_data is None:
        raise RuntimeError(f"no profiled run found for focus workload {focus_workload}")

    telemetry, start, end, requests = focus_data
    requests = sorted(requests, key=lambda event: (float(event.get("request_ready_ts") or 0), str(event.get("node_id") or "")))
    fig, axes = plt.subplots(3, 1, figsize=(13.5, 9.0), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1.4, 1.25]})

    # Panel 1 is the MAS graph execution itself: every LLM node is split into
    # prefill and decode, so the hardware signals below retain graph meaning.
    gantt = axes[0]
    labels = []
    for ordinal, event in enumerate(requests):
        ready = float(event.get("request_ready_ts") or event.get("submit_ts")) - start
        first = float(event["first_token_ts"]) - start
        completed = float(event["completion_ts"]) - start
        gantt.barh(ordinal, first - ready, left=ready, height=.7, color="#E9C46A", label="Prefill" if ordinal == 0 else None)
        gantt.barh(ordinal, completed - first, left=first, height=.7, color="#4EA8DE", label="Decode" if ordinal == 0 else None)
        labels.append(f"{ordinal + 1:02d}  {event.get('node_id', 'request')}")
    gantt.set_yticks(range(len(labels)), labels)
    gantt.invert_yaxis()
    gantt.set_ylabel("MAS graph node / request")
    gantt.set_title("(a) Graph execution timeline (measured SSE phase boundaries)")
    gantt.legend(frameon=False, ncol=2, loc="upper right")

    hardware = axes[1]
    hardware.plot(telemetry.elapsed_sec, telemetry.ai_core_util_pct, color="#7B2CBF", linewidth=1.6,
                  marker="o", markersize=3.2, label="AI Core utilization (sampled)")
    hardware.plot(telemetry.elapsed_sec, telemetry.hbm_bandwidth_util_pct, color="#008C95", linewidth=1.6,
                  marker="o", markersize=3.2, label="HBM bandwidth utilization (sampled)")
    hardware.set_ylabel("Observed utilization (%)")
    hardware.set_ylim(0, 105)
    hardware.set_title("(b) CANN/NPU hardware behavior aligned to the same clock")
    hardware.legend(frameon=False, ncol=2, loc="upper right")

    pressure = axes[2]
    pressure.step(telemetry.elapsed_sec, telemetry.prefill_pressure, where="mid", color="#C58A00", linewidth=1.7, label="Concurrent prefill")
    pressure.step(telemetry.elapsed_sec, telemetry.decode_pressure, where="mid", color="#247BA0", linewidth=1.7, label="Concurrent decode")
    pressure.set_ylabel("Active requests")
    pressure.set_xlabel("Elapsed workflow time (s)")
    pressure.set_title("(c) Graph pressure and board power")
    power = pressure.twinx()
    power.plot(telemetry.elapsed_sec, telemetry.power_w, color="#666666", alpha=.65, linewidth=1.2, label="Power")
    power.set_ylabel("Power (W)")
    handles1, labels1 = pressure.get_legend_handles_labels()
    handles2, labels2 = power.get_legend_handles_labels()
    pressure.legend(handles1 + handles2, labels1 + labels2, frameon=False, ncol=3, loc="upper right")

    for ax in axes:
        ax.set_xlim(0, end - start)
    fig.suptitle(f"Workflow bottleneck map — {DISPLAY[focus_workload]} on one Ascend 910C die", fontsize=13, y=.995)
    fig.tight_layout()
    save(fig, output, "fig6_workflow_bottleneck_map")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--workflow-profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--focus-workload", choices=WORKLOAD_ORDER, default="hierarchical_synthesis_pressure_meso")
    args = parser.parse_args()
    style()
    plot_prefill_decode(args.sweep_root, args.output)
    plot_workflow_map(args.workflow_profile_root, args.output, args.focus_workload)


if __name__ == "__main__":
    main()

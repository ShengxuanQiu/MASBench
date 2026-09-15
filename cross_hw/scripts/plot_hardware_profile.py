from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


WORKLOAD_ORDER = [
    "single_agent_control",
    "independent_fanin",
    "centralized_manager_worker",
    "debate_allgather_pressure_meso",
    "shared_memory_fanin_meso",
    "retry_debug_loop",
    "hierarchical_synthesis_pressure_meso",
    "issue_to_patch_workflow",
]

DISPLAY = {
    "single_agent_control": "Single / Linear",
    "independent_fanin": "Independent\nFan-In",
    "centralized_manager_worker": "Manager–Worker",
    "debate_allgather_pressure_meso": "Debate\nAll-Gather",
    "shared_memory_fanin_meso": "Shared-Memory\nFan-In",
    "retry_debug_loop": "Retry Loop",
    "hierarchical_synthesis_pressure_meso": "Hierarchical\nSynthesis",
    "issue_to_patch_workflow": "Issue-to-Patch",
}

COLORS = {"A6000": "#3366CC", "Ascend 910": "#D95F02"}


def percentile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q))


def load_platform(root: Path, hardware: str) -> tuple[list[dict], list[dict]]:
    run_rows: list[dict] = []
    request_rows: list[dict] = []
    for summary_path in sorted(root.rglob("summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("mode") != "raw" or summary.get("workload") not in WORKLOAD_ORDER:
            continue
        trace_path = summary_path.with_name("trace.jsonl")
        if not trace_path.exists():
            continue
        events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        requests = [event for event in events if event.get("event_type") == "llm_request_end"]
        requests.sort(key=lambda event: (float(event.get("request_ready_ts") or 0), str(event.get("node_id") or "")))
        ttfts = [float(event["ttft_sec"]) for event in requests if event.get("ttft_sec") is not None]
        tpots = [float(event["tpot_sec"]) for event in requests if event.get("tpot_sec") is not None]
        latencies = [float(event["request_latency_sec"]) for event in requests if event.get("request_latency_sec") is not None]
        run_rows.append(
            {
                "hardware": hardware,
                "workload": summary["workload"],
                "run_id": summary["run_id"],
                "request_count": len(requests),
                "result_latency_sec": float(summary["result_latency_sec"]),
                "tool_latency_sec": float(summary.get("tool_latency_sec") or 0),
                "tool_excluded_e2e_sec": float(summary["result_latency_sec"]) - float(summary.get("tool_latency_sec") or 0),
                "median_ttft_ms": 1000 * float(np.median(ttfts)),
                "p95_ttft_ms": 1000 * percentile(pd.Series(ttfts), 0.95),
                "median_tpot_ms": 1000 * float(np.median(tpots)),
                "median_request_latency_sec": float(np.median(latencies)),
                "total_input_tokens": sum(int(event.get("input_tokens") or 0) for event in requests),
                "total_output_tokens": sum(int(event.get("output_tokens") or 0) for event in requests),
            }
        )
        for ordinal, event in enumerate(requests, start=1):
            request_rows.append(
                {
                    "hardware": hardware,
                    "workload": summary["workload"],
                    "run_id": summary["run_id"],
                    "request_ordinal": ordinal,
                    "node_id": event.get("node_id"),
                    "input_tokens": int(event.get("input_tokens") or 0),
                    "output_tokens": int(event.get("output_tokens") or 0),
                    "ttft_ms": 1000 * float(event.get("ttft_sec") or 0),
                    "tpot_ms": 1000 * float(event.get("tpot_sec") or 0),
                    "request_latency_sec": float(event.get("request_latency_sec") or 0),
                }
            )
    return run_rows, request_rows


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
        }
    )


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overview(runs: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("tool_excluded_e2e_sec", "Tool-excluded E2E latency (s)"),
        ("median_ttft_ms", "Median request TTFT (ms)"),
        ("median_tpot_ms", "Median TPOT (ms/token)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    x = np.arange(len(WORKLOAD_ORDER))
    for ax, (metric, ylabel) in zip(axes, metrics):
        for hardware in ["A6000", "Ascend 910"]:
            means, lows, highs = [], [], []
            for workload in WORKLOAD_ORDER:
                values = runs[(runs.hardware == hardware) & (runs.workload == workload)][metric]
                means.append(values.mean())
                lows.append(values.min())
                highs.append(values.max())
            means_array = np.asarray(means, dtype=float)
            ax.plot(x, means_array, marker="o", linewidth=2.0, markersize=4.5, color=COLORS[hardware], label=hardware)
            ax.fill_between(x, lows, highs, alpha=0.12, color=COLORS[hardware])
        ax.set_xticks(x, [DISPLAY[item] for item in WORKLOAD_ORDER], rotation=28, ha="right")
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Qwen3-8B MAS workflow behavior across hardware platforms (raw propagation)", y=1.03, fontsize=13)
    fig.tight_layout()
    save_figure(fig, output, "fig1_hardware_workflow_overview")


def plot_progression(requests: pd.DataFrame, output: Path, metric: str, ylabel: str, stem: str) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.2), sharey=False)
    for ax, workload in zip(axes.flat, WORKLOAD_ORDER):
        subset = requests[requests.workload == workload]
        for hardware in ["A6000", "Ascend 910"]:
            hw = subset[subset.hardware == hardware]
            grouped = hw.groupby("request_ordinal")[metric].agg(["mean", "min", "max"]).reset_index()
            x = grouped["request_ordinal"].to_numpy()
            ax.plot(x, grouped["mean"], marker="o", markersize=3.5, linewidth=1.8, color=COLORS[hardware], label=hardware)
            ax.fill_between(x, grouped["min"], grouped["max"], alpha=0.12, color=COLORS[hardware])
        ax.set_title(DISPLAY[workload].replace("\n", " "))
        ax.set_xlabel("LLM request ordinal")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Per-request {ylabel} along each MAS graph", y=1.01, fontsize=13)
    fig.tight_layout()
    save_figure(fig, output, stem)


def plot_amplification(runs: pd.DataFrame, output: Path) -> None:
    means = runs.groupby(["hardware", "workload"], as_index=False)["tool_excluded_e2e_sec"].mean()
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(WORKLOAD_ORDER))
    for hardware in ["A6000", "Ascend 910"]:
        hw = means[means.hardware == hardware].set_index("workload")
        baseline = float(hw.loc["single_agent_control", "tool_excluded_e2e_sec"])
        values = [float(hw.loc[workload, "tool_excluded_e2e_sec"]) / baseline for workload in WORKLOAD_ORDER]
        ax.plot(x, values, marker="o", linewidth=2.0, color=COLORS[hardware], label=hardware)
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(x, [DISPLAY[item] for item in WORKLOAD_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Graph amplification vs. Single / Linear (×)")
    ax.legend(frameon=False)
    ax.set_title("How MAS graph structure amplifies hardware-level latency")
    fig.tight_layout()
    save_figure(fig, output, "fig4_graph_amplification")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a6000-root", type=Path, required=True)
    parser.add_argument("--ascend-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    run_rows, request_rows = load_platform(args.a6000_root, "A6000")
    ascend_runs, ascend_requests = load_platform(args.ascend_root, "Ascend 910")
    runs = pd.DataFrame(run_rows + ascend_runs)
    requests = pd.DataFrame(request_rows + ascend_requests)
    if runs.groupby(["hardware", "workload"]).size().min() < 2:
        raise RuntimeError("expected at least two runs per hardware/workload")
    runs.to_csv(args.output / "hardware_run_metrics.csv", index=False)
    requests.to_csv(args.output / "hardware_request_metrics.csv", index=False)
    style()
    plot_overview(runs, args.output)
    plot_progression(requests, args.output, "ttft_ms", "TTFT (ms)", "fig2_ttft_request_progression")
    plot_progression(requests, args.output, "tpot_ms", "TPOT (ms/token)", "fig3_tpot_request_progression")
    plot_amplification(runs, args.output)
    print(runs.groupby(["hardware", "workload"])[["tool_excluded_e2e_sec", "median_ttft_ms", "median_tpot_ms"]].mean().to_string())


if __name__ == "__main__":
    main()

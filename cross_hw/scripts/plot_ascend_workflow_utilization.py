#!/usr/bin/env python3
"""Clean NPU-only utilization view for matched MAS workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


WORKLOADS = [
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
METRICS = [
    ("ai_core_util_pct", "AI Core", "#7B2CBF", "-"),
    ("ai_cube_util_pct", "AI Cube", "#D95F02", "-."),
    ("hbm_bandwidth_util_pct", "HBM bandwidth", "#008C95", "--"),
]


def load_candidate(run_dir: Path) -> tuple[pd.DataFrame, float]:
    events = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next(float(event["start_ts"]) for event in events if event["event_type"] == "workflow_start")
    end = next(float(event["end_ts"]) for event in reversed(events) if event["event_type"] == "workflow_end")
    data = pd.read_csv(run_dir / "telemetry.csv")
    data = data[(data.timestamp_unix >= start) & (data.timestamp_unix <= end)].copy()
    data["elapsed_sec"] = data.timestamp_unix - start
    return data, end - start


def runs_for(roots: list[Path], workload: str) -> list[Path]:
    result = []
    for root in roots:
        result.extend(path for path in sorted((root / workload / "raw").glob("*"))
                      if (path / "telemetry.csv").exists() and (path / "trace.jsonl").exists())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": .18, "savefig.dpi": 300,
    })
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 7.4), sharey=True)
    summaries = []
    for ax, workload in zip(axes.flat, WORKLOADS):
        candidates = []
        for run_dir in runs_for(args.profile_root, workload):
            data, duration = load_candidate(run_dir)
            candidates.append((duration, run_dir, data))
            for column, _, _, _ in METRICS:
                values = pd.to_numeric(data[column], errors="coerce")
                summaries.append({
                    "workload": workload, "run_id": run_dir.name, "duration_sec": duration,
                    "metric": column, "samples": int(values.notna().sum()),
                    "mean_pct": values.mean(), "p95_pct": values.quantile(.95),
                })
        if not candidates:
            raise RuntimeError(f"no telemetry runs found for {workload}")
        candidates.sort(key=lambda item: item[0])
        _, _, representative = candidates[len(candidates) // 2]
        for column, label, color, linestyle in METRICS:
            values = pd.to_numeric(representative[column], errors="coerce").rolling(3, center=True, min_periods=1).mean()
            ax.plot(representative.elapsed_sec, values, color=color, linestyle=linestyle,
                    linewidth=1.8, marker="o", markersize=2.5, label=label)
        ax.set_title(DISPLAY[workload])
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Utilization (%)")
        ax.set_ylim(0, 105)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.suptitle("Ascend 910C utilization across MAS workflows", fontsize=13, y=1.055)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(args.output / f"fig7_ascend_workflow_utilization.{suffix}", bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(args.output / "ascend_workflow_utilization_runs.csv", index=False)


if __name__ == "__main__":
    main()

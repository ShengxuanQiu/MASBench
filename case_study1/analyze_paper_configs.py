from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from case_study1.analyze import analyze_run, read_jsonl


ROOT = Path(__file__).resolve().parent
MOTIF_RUNS = ROOT / "artifacts/multiconfig_3x/runs"
VLLM_TRACE = ROOT / "artifacts/multiconfig_3x/server/vllm_step_trace.jsonl.gz"
FULL_RUNS = (
    ROOT
    / "artifacts/paper_configs/full_workflow_multiwindow_frontier_v2/issue_to_verified_patch"
    / "astropy__astropy-12907"
)
OUT = ROOT / "artifacts/paper_configs/analysis"

CONFIGS = ["2-agent motif", "4-agent motif", "Full workflow"]
POLICIES = ["Default vLLM", "Online MAS-aware gating"]
LIGHT_BLUE = "#B0D7E6"
BLUE = "#475EA4"
GRAY = "#ABB2BC"


def quantiles(values: list[float]) -> dict[str, float]:
    return {
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def event_one(events: list[dict[str, Any]], kind: str, node: str | None = None) -> dict[str, Any]:
    rows = [
        row
        for row in events
        if row.get("event_type") == kind and (node is None or row.get("node_id") == node)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one {kind}/{node}, found {len(rows)}")
    return rows[0]


def motif_rows() -> list[dict[str, Any]]:
    backend = read_jsonl(VLLM_TRACE)
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(MOTIF_RUNS.glob("cs1_*")):
        config_path = run_dir / "experiment_config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        fanout = int(config["experiment"]["resume_fanout"])
        if fanout not in {2, 4}:
            continue
        run = analyze_run(run_dir, backend)
        events = run["events"]
        reviewer = event_one(events, "llm_request_end", "critical_reviewer")
        result = event_one(events, "workflow_result_ready")
        first = float(reviewer["first_token_ts"])
        complete = float(reviewer["completion_ts"])
        result_ts = float(result["result_ready_ts"])
        result_latency = float(run["result_ready_sec"])
        critical_decode = complete - first
        post_critical = result_ts - complete
        rows.append(
            {
                "configuration": f"{fanout}-agent motif",
                "policy": "Default vLLM" if run["policy"] == "default_vllm" else "Online MAS-aware gating",
                "run_id": run["run_id"],
                "result_latency_sec": result_latency,
                # result_ready_sec begins at runner start, slightly before the
                # workflow_start trace event; use the exact residual so the
                # three mutually exclusive phases sum to E2E latency.
                "pre_critical_sec": result_latency - critical_decode - post_critical,
                "critical_decode_sec": critical_decode,
                "post_critical_sec": post_critical,
                "unprotected_work_sec": result_latency - critical_decode,
                "critical_tpot_ms": float(run["reviewer_tpot_ms"]),
                "critical_tpot_p95_ms": float(run["reviewer_tpot_p95_ms"]),
                "overlap_resume_prefill_tokens": int(run["reviewer_background_prefill_tokens"]),
                "overlap_resume_prefill_steps": int(run["reviewer_background_prefill_steps"]),
                "protected_decode_total_sec": critical_decode,
                "evidence_window_overlap_tokens": 0,
                "diagnosis_window_overlap_tokens": 0,
                "final_window_overlap_tokens": 0,
                "critical_ttft_ms": float(run["reviewer_ttft_ms"]),
                "tool_latency_sec": float(run["tool_latency_median_sec"]),
                "max_defer_sec": float(run["max_defer_sec"]),
                "deferred_count": int(run["deferred_count"]),
                "defer_timeout_count": 0,
                "all_background_completed": bool(run["valid"]["all_background_completed"]),
                "real_vllm_sse": bool(run["valid"]["client_sse"]),
                "real_untruncated_tavily": bool(run["valid"]["live_untruncated_tavily"]),
                "vllm_step_evidence": bool(run["valid"]["vllm_scheduler_step_trace"]),
                "eligible_resume_window": True,
            }
        )
    return rows


def full_rows() -> list[dict[str, Any]]:
    backend = read_jsonl(VLLM_TRACE)
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(FULL_RUNS.glob("*.jsonl")):
        events = read_jsonl(trace_path)
        start = event_one(events, "workflow_start")
        end = event_one(events, "workflow_end")
        result_ready = event_one(events, "workflow_result_ready")
        final = event_one(events, "llm_request_end", "final_report")
        summary_path = trace_path.with_name(f"{trace_path.stem}_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        policy_raw = start["extra"]["config"]["admission_policy"]
        decisions = [row for row in events if row.get("event_type") == "admission_decision"]
        llm_ends = [row for row in events if row.get("event_type") == "llm_request_end"]
        tool_searches = [row for row in events if row.get("event_type") == "tool_search"]
        request_ids = {str(row["request_id_for_backend"]) for row in llm_ends}
        server_request_ids = {
            str(request_id)
            for row in backend
            if row.get("event_type") == "scheduler_batch"
            for request_id in (row.get("request_phase_details") or {})
        }
        snapshots = [
            json.loads(Path(row["result_snapshot_id"]).read_text(encoding="utf-8"))
            for row in tool_searches
        ]
        # Reconstruct the run's absolute zero from a request event carrying both clocks.
        zero_ts = float(final["completion_ts"]) - float(final["relative_time_sec"])
        result_latency = float(result_ready["relative_time_sec"])
        first = float(final["first_token_ts"])
        complete = float(final["completion_ts"])
        windows = [row for row in events if row.get("event_type") == "case_study_eligible_window"]
        resumed = [row for row in decisions if row.get("tool_resumed")]
        deferred = [row for row in decisions if str(row.get("decision", "")).startswith("admitted_after")]
        resumed_ends = [row for row in llm_ends if row.get("tool_resumed")]
        overlap_tokens = 0
        overlap_steps = 0
        protected_decode_sec = 0.0
        window_overlap_tokens: dict[str, int] = {}
        window_overlap_steps: dict[str, int] = {}
        for window in windows:
            window_id = str(window["node_id"])
            critical_node = str(window["critical_node"])
            resume_nodes = {str(node) for node in window.get("resume_nodes", [])}
            critical = event_one(events, "llm_request_end", critical_node)
            decode_start = float(critical["first_token_ts"])
            decode_end = float(critical["completion_ts"])
            protected_decode_sec += decode_end - decode_start
            window_resume_ids = {
                str(row["request_id_for_backend"])
                for row in llm_ends
                if str(row.get("node_id")) in resume_nodes
            }
            tokens = 0
            steps = 0
            for server_row in backend:
                if server_row.get("event_type") != "scheduler_batch":
                    continue
                timestamp = float(server_row["timestamp_unix"])
                if not decode_start <= timestamp <= decode_end:
                    continue
                step_tokens = sum(
                    int(detail.get("prefill_tokens") or 0)
                    for server_request_id, detail in (server_row.get("request_phase_details") or {}).items()
                    if any(request_id in server_request_id for request_id in window_resume_ids)
                )
                tokens += step_tokens
                steps += int(step_tokens > 0)
            window_overlap_tokens[window_id] = tokens
            window_overlap_steps[window_id] = steps
            overlap_tokens += tokens
            overlap_steps += steps
        rows.append(
            {
                "configuration": "Full workflow",
                "policy": "Default vLLM" if policy_raw == "default_vllm" else "Online MAS-aware gating",
                "run_id": trace_path.stem,
                "result_latency_sec": result_latency,
                "pre_critical_sec": first - zero_ts,
                "critical_decode_sec": complete - first,
                "post_critical_sec": max(0.0, result_latency - (complete - zero_ts)),
                "critical_tpot_ms": 1000.0 * float(final["tpot_sec"]),
                "critical_tpot_p95_ms": 1000.0 * float(final["tpot_p95_sec"]),
                "overlap_resume_prefill_tokens": overlap_tokens,
                "overlap_resume_prefill_steps": overlap_steps,
                "protected_decode_total_sec": protected_decode_sec,
                "unprotected_work_sec": result_latency - protected_decode_sec,
                "evidence_window_overlap_tokens": window_overlap_tokens.get("evidence_window", 0),
                "diagnosis_window_overlap_tokens": window_overlap_tokens.get("diagnosis_window", 0),
                "final_window_overlap_tokens": window_overlap_tokens.get("final_window", 0),
                "critical_ttft_ms": 1000.0 * float(final["ttft_sec"]),
                "tool_latency_sec": float(summary["measured_tool_time"]),
                "max_defer_sec": max([float(row.get("defer_duration_sec") or 0.0) for row in decisions] or [0.0]),
                "deferred_count": len(deferred),
                "defer_timeout_count": sum(
                    row.get("decision") == "admitted_after_max_defer" for row in decisions
                ),
                "all_background_completed": len(resumed_ends) == 9 and all(
                    float(row["relative_time_sec"]) <= float(end["relative_time_sec"])
                    for row in resumed_ends
                ),
                "real_vllm_sse": all(
                    row.get("request_metadata", {}).get("streaming_timing_granularity") == "openai_sse_chunk"
                    for row in llm_ends
                ),
                "real_untruncated_tavily": len(tool_searches) >= 11 and all(
                    row.get("tool_trace_source") == "live"
                    and row.get("extra", {}).get("provider_name") == "tavily"
                    for row in tool_searches
                ) and all(
                    snapshot.get("provider_name") == "tavily"
                    and snapshot.get("output", {}).get("results")
                    and all("raw_content" in result for result in snapshot["output"]["results"])
                    for snapshot in snapshots
                ),
                "vllm_step_evidence": all(
                    any(request_id in server_request_id for server_request_id in server_request_ids)
                    for request_id in request_ids
                ),
                "eligible_resume_window": len(windows) == 3 and len(resumed) == 9,
            }
        )
    return rows


def grouped(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[(row["configuration"], row["policy"])].append(row)
    return result


def style_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", width=0.8, color="#333333")
    ax.grid(axis="y", color="#D9DDE3", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")


def figure1(groups: dict[tuple[str, str], list[dict[str, Any]]]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.45))
    x = np.arange(len(CONFIGS), dtype=float)
    width = 0.31
    offsets = [-width / 1.75, width / 1.75]
    phase_keys = ["unprotected_work_sec", "protected_decode_total_sec"]
    phase_labels = ["Unprotected workflow work", "Protected decode work"]
    colors = [LIGHT_BLUE, BLUE]

    for policy_idx, policy in enumerate(POLICIES):
        bottoms = np.zeros(len(CONFIGS))
        xpos = x + offsets[policy_idx]
        for key, color in zip(phase_keys, colors):
            vals = np.array([median([float(r[key]) for r in groups[(config, policy)]]) for config in CONFIGS])
            ax.bar(
                xpos,
                vals,
                width,
                bottom=bottoms,
                color=color,
                edgecolor="white",
                linewidth=0.7,
            )
            bottoms += vals
        for idx, config in enumerate(CONFIGS):
            totals = [float(r["result_latency_sec"]) for r in groups[(config, policy)]]
            center = median(totals)
            ax.errorbar(
                xpos[idx], center,
                yerr=[[center - min(totals)], [max(totals) - center]],
                fmt="none", ecolor="#30343B", capsize=2.5, linewidth=0.8, zorder=5,
            )
            ax.text(xpos[idx], bottoms[idx] + 0.65, f"{center:.1f}", ha="center", va="bottom", fontsize=8)

    for idx, config in enumerate(CONFIGS):
        base = median([float(r["result_latency_sec"]) for r in groups[(config, POLICIES[0])]])
        gated = median([float(r["result_latency_sec"]) for r in groups[(config, POLICIES[1])]])
        label = f"{base / gated:.2f}×"
        ymax = max(base, gated)
        ax.text(x[idx], ymax + 1.2, label, ha="center", va="bottom", color=BLUE, fontsize=9, fontweight="bold")

    ax.set_xticks(np.ravel(np.column_stack((x + offsets[0], x + offsets[1]))))
    ax.set_xticklabels(["Default", "Gating"] * len(CONFIGS), fontsize=8)
    for idx, config in enumerate(CONFIGS):
        display = {
            "2-agent motif": "Motif 1",
            "4-agent motif": "Motif 2",
            "Full workflow": "Full workflow",
        }[config]
        ax.text(x[idx], -0.16, display, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
    ax.set_ylabel("Workflow result latency (s)")
    ymax = max(
        float(r["result_latency_sec"])
        for config in CONFIGS
        for policy in POLICIES
        for r in groups[(config, policy)]
    )
    ax.set_ylim(0, ymax * 1.14)
    ax.set_title("End-to-end Latency Breakdown", fontsize=11, pad=10)
    phase_handles = [Patch(facecolor=c, edgecolor="white", label=l) for c, l in zip(colors, phase_labels)]
    ax.legend(handles=phase_handles, ncol=3, loc="upper center", frameon=False, fontsize=8)
    style_axis(ax)
    fig.subplots_adjust(bottom=0.22, top=0.83, left=0.10, right=0.985)
    save_both(fig, "figure1_workflow_latency_breakdown")
    plt.close(fig)


def figure2(groups: dict[tuple[str, str], list[dict[str, Any]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.76))
    x = np.arange(len(CONFIGS))
    policy_style = {
        "Default vLLM": (GRAY, "s"),
        "Online MAS-aware gating": (BLUE, "o"),
    }
    for ax in axes[:1]:
        for policy in POLICIES:
            stats = [quantiles([float(r["result_latency_sec"]) for r in groups[(config, policy)]]) for config in CONFIGS]
            med = np.array([s["median"] for s in stats])
            low = med - np.array([s["min"] for s in stats])
            high = np.array([s["max"] for s in stats]) - med
            color, marker = policy_style[policy]
            ax.plot(x, med, color=color, marker=marker, markersize=5.5, linewidth=1.8, label=("Online MAS gating" if policy == "Online MAS-aware gating" else policy), zorder=3)
            ax.errorbar(x, med, yerr=np.vstack([low, high]), fmt="none", ecolor=color, capsize=3, linewidth=0.9, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(["Motif 1", "Motif 2", "Full workflow"], fontsize=8)
        ax.set_ylabel("Workflow result latency (s)", fontsize=8.5)
        ax.set_title("(a) End-to-end latency", fontsize=9.5, pad=8)
        style_axis(ax)

    # Report the directly attributable scheduling benefit as a latency. Each
    # range uses every Default/Gating cross-run difference (3x3), avoiding a
    # cherry-picked pairing while preserving the observed run-to-run range.
    saved_stats = []
    for config in CONFIGS:
        default_latency = [float(r["result_latency_sec"]) for r in groups[(config, "Default vLLM")]]
        gated_latency = [float(r["result_latency_sec"]) for r in groups[(config, "Online MAS-aware gating")]]
        cross_run_differences = [
            default - gated for default in default_latency for gated in gated_latency
        ]
        saved_stats.append(
            {
                "median": median(default_latency) - median(gated_latency),
                "min": min(cross_run_differences),
                "max": max(cross_run_differences),
            }
        )
    saved = np.array([stat["median"] for stat in saved_stats])
    saved_low = saved - np.array([stat["min"] for stat in saved_stats])
    saved_high = np.array([stat["max"] for stat in saved_stats]) - saved
    axes[1].bar(x, saved, width=0.54, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    axes[1].errorbar(
        x,
        saved,
        yerr=np.vstack([saved_low, saved_high]),
        fmt="none",
        ecolor="#30343B",
        capsize=3,
        linewidth=0.9,
        zorder=4,
    )
    for index, value in enumerate(saved):
        axes[1].text(index, value + max(saved) * 0.035, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Motif 1", "Motif 2", "Full workflow"], fontsize=8)
    axes[1].set_ylabel("Workflow latency saved (s)", fontsize=8.5)
    axes[1].set_title("(b) End-to-end scheduling benefit", fontsize=9.5, pad=8)
    axes[1].set_ylim(0, max(saved + saved_high) * 1.18)
    style_axis(axes[1])
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.subplots_adjust(wspace=0.40, bottom=0.22, top=0.84, left=0.065, right=0.985)
    save_both(fig, "figure2_multiconfig_metrics")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = motif_rows() + full_rows()
    groups = grouped(rows)
    for config in CONFIGS:
        for policy in POLICIES:
            count = len(groups[(config, policy)])
            if count != 3:
                raise RuntimeError(f"{config}/{policy}: expected 3 runs, found {count}")
    if not all(r["real_vllm_sse"] and r["real_untruncated_tavily"] and r["vllm_step_evidence"] for r in rows):
        raise RuntimeError("real-execution validity gate failed")
    if not all(r["all_background_completed"] for r in rows):
        raise RuntimeError("a deferred/background request did not complete")
    if any(r["defer_timeout_count"] for r in rows):
        raise RuntimeError("a deferred request reached the maximum-defer fail-safe")
    if not all(r["eligible_resume_window"] for r in rows if r["configuration"] == "Full workflow"):
        raise RuntimeError("full workflow did not expose the registered tool-resume window")
    for config in CONFIGS:
        if any(r["overlap_resume_prefill_tokens"] <= 0 for r in groups[(config, "Default vLLM")]):
            raise RuntimeError(f"{config}: default run missing resume-prefill overlap")
        if any(r["overlap_resume_prefill_tokens"] != 0 for r in groups[(config, "Online MAS-aware gating")]):
            raise RuntimeError(f"{config}: gating failed to remove resume-prefill overlap")
    full_window_fields = [
        "evidence_window_overlap_tokens",
        "diagnosis_window_overlap_tokens",
        "final_window_overlap_tokens",
    ]
    if any(
        float(r[field]) <= 0
        for r in groups[("Full workflow", "Default vLLM")]
        for field in full_window_fields
    ):
        raise RuntimeError("full workflow default run missing overlap in an eligible window")
    if any(
        float(r[field]) != 0
        for r in groups[("Full workflow", "Online MAS-aware gating")]
        for field in full_window_fields
    ):
        raise RuntimeError("full workflow gating left overlap in an eligible window")

    fields = list(rows[0])
    with (OUT / "per_run_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "experiment": {
            "configurations": CONFIGS,
            "policies": POLICIES,
            "repetitions_per_cell": 3,
            "model": "Qwen3-8B BF16",
            "gpu": "NVIDIA RTX A6000 48 GB",
            "runtime": "MAS environment vLLM 0.20.1rc1.dev2+gc245d35ff.precompiled",
            "tool": "live Tavily Advanced Search; raw response retained without application-side text slicing",
            "observation_granularity": "vLLM scheduler step plus OpenAI SSE request lifecycle",
        },
        "cells": {},
    }
    numeric = [
        "result_latency_sec", "pre_critical_sec", "critical_decode_sec", "post_critical_sec",
        "critical_tpot_ms", "critical_tpot_p95_ms", "overlap_resume_prefill_tokens",
        "overlap_resume_prefill_steps", "critical_ttft_ms", "tool_latency_sec", "max_defer_sec",
        "protected_decode_total_sec", "evidence_window_overlap_tokens",
        "unprotected_work_sec",
        "diagnosis_window_overlap_tokens", "final_window_overlap_tokens",
    ]
    for config in CONFIGS:
        summary["cells"][config] = {}
        for policy in POLICIES:
            cell = groups[(config, policy)]
            summary["cells"][config][policy] = {
                "run_ids": [r["run_id"] for r in cell],
                **{key: quantiles([float(r[key]) for r in cell]) for key in numeric},
            }
        base = summary["cells"][config][POLICIES[0]]["result_latency_sec"]["median"]
        gated = summary["cells"][config][POLICIES[1]]["result_latency_sec"]["median"]
        summary["cells"][config]["effect"] = {
            "workflow_speedup_x": base / gated,
            "workflow_latency_reduction_pct": 100.0 * (1.0 - gated / base),
            "workflow_latency_saved_sec": base - gated,
            "critical_decode_latency_saved_sec": (
                summary["cells"][config][POLICIES[0]]["critical_decode_sec"]["median"]
                - summary["cells"][config][POLICIES[1]]["critical_decode_sec"]["median"]
            ),
        }
    (OUT / "paper_config_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    figure1(groups)
    figure2(groups)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

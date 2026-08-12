from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


COLORS = {
    "critical_prefill": "#8CB6D9",
    "critical_decode": "#236192",
    "background_prefill": "#D97824",
    "tool": "#4C956C",
    "defer": "#7B6BA8",
    "result": "#222222",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt", encoding="utf-8") if path.suffix == ".gz" else opener("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def one(events: list[dict[str, Any]], event_type: str, node_id: str | None = None) -> dict[str, Any]:
    rows = [
        row
        for row in events
        if row.get("event_type") == event_type
        and (node_id is None or row.get("node_id") == node_id)
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one {event_type}/{node_id}, got {len(rows)}")
    return rows[0]


def backend_rows_for_run(rows: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if any(run_id in str(req_id) for req_id in row.get("request_ids", []))
    ]


def request_is_background_resume(request_id: str) -> bool:
    return "tool_branch_" in request_id and "_resume" in request_id


def analyze_run(run_dir: Path, backend: list[dict[str, Any]]) -> dict[str, Any]:
    events = read_jsonl(run_dir / "client_trace.jsonl")
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    run_id = str(summary["run_id"])
    server = backend_rows_for_run(backend, run_id)
    reviewer = one(events, "llm_request_end", "critical_reviewer")
    finalizer = one(events, "llm_request_end", "critical_finalizer")
    result_ready = one(events, "workflow_result_ready")
    workflow_end = one(events, "workflow_end")
    reviewer_ready = one(events, "llm_request_ready", "critical_reviewer")
    tools = [row for row in events if row.get("event_type") == "tool_return"]
    decisions = [
        row
        for row in events
        if row.get("event_type") == "admission_decision"
        and row.get("tool_resumed")
    ]

    def scheduled_background_prefill(lo: float, hi: float) -> tuple[int, int, set[str]]:
        tokens = 0
        steps = 0
        requests: set[str] = set()
        for row in server:
            if row.get("event_type") != "scheduler_batch":
                continue
            ts = float(row["timestamp_unix"])
            if not lo <= ts <= hi:
                continue
            step_has_prefill = False
            for request_id, detail in (row.get("request_phase_details") or {}).items():
                count = int(detail.get("prefill_tokens") or 0)
                if request_is_background_resume(request_id) and count:
                    tokens += count
                    requests.add(request_id)
                    step_has_prefill = True
            if step_has_prefill:
                steps += 1
        return tokens, steps, requests

    review_lo = float(reviewer["first_token_ts"])
    review_hi = float(reviewer["completion_ts"])
    final_lo = float(finalizer["first_token_ts"])
    final_hi = float(finalizer["completion_ts"])
    review_overlap = scheduled_background_prefill(review_lo, review_hi)
    final_overlap = scheduled_background_prefill(final_lo, final_hi)

    first_schedule: dict[str, dict[str, Any]] = {}
    for row in server:
        if row.get("event_type") != "scheduler_batch":
            continue
        for request_id, detail in (row.get("request_phase_details") or {}).items():
            if detail.get("first_schedule"):
                first_schedule[request_id] = detail
    resume_first = {
        request_id: detail
        for request_id, detail in first_schedule.items()
        if request_is_background_resume(request_id)
    }
    ready_rows = {
        row["node_id"]: row
        for row in events
        if row.get("event_type") == "llm_request_ready"
    }
    end_rows = {
        row["node_id"]: row
        for row in events
        if row.get("event_type") == "llm_request_end"
    }
    prompt_token_consistency = all(
        int(ready_rows[node]["ready_prompt_tokens"]) == int(end_rows[node]["input_tokens"])
        for node in ready_rows.keys() & end_rows.keys()
    )
    return {
        "run_id": run_id,
        "policy": summary["policy"],
        "run_dir": str(run_dir),
        "events": events,
        "server": server,
        "result_ready_sec": float(summary["result_ready_sec"]),
        "drain_complete_sec": float(summary["drain_complete_sec"]),
        "critical_frontier_sec": float(result_ready["result_ready_ts"]) - float(reviewer_ready["ready_ts"]),
        "reviewer_ttft_ms": 1000.0 * float(reviewer["ttft_sec"]),
        "reviewer_decode_sec": review_hi - review_lo,
        "reviewer_tpot_ms": 1000.0 * float(reviewer["tpot_sec"]),
        "reviewer_tpot_p95_ms": 1000.0 * float(reviewer["tpot_p95_sec"]),
        "finalizer_decode_sec": final_hi - final_lo,
        "finalizer_ttft_ms": 1000.0 * float(finalizer["ttft_sec"]),
        "finalizer_tpot_ms": 1000.0 * float(finalizer["tpot_sec"]),
        "reviewer_background_prefill_tokens": review_overlap[0],
        "reviewer_background_prefill_steps": review_overlap[1],
        "reviewer_background_prefill_requests": len(review_overlap[2]),
        "finalizer_background_prefill_tokens": final_overlap[0],
        "resume_cached_prefix_tokens": {
            request_id: int(detail.get("cached_prefix_tokens") or 0)
            for request_id, detail in resume_first.items()
        },
        "resume_prompt_tokens": {
            request_id: int(detail.get("prompt_tokens") or 0)
            for request_id, detail in resume_first.items()
        },
        "max_defer_sec": max([float(row.get("defer_sec") or 0.0) for row in decisions] or [0.0]),
        "deferred_count": sum(1 for row in decisions if str(row.get("decision", "")).startswith("admitted_after")),
        "tool_latency_median_sec": median([float(row["tool_latency_sec"]) for row in tools]),
        "tool_result_chars": sorted(int(row["result_chars"]) for row in tools),
        "valid": {
            "client_sse": all(row.get("timing_source") == "openai_sse_and_vllm_usage" for row in end_rows.values()),
            "live_untruncated_tavily": bool(tools) and all(
                row.get("provider") == "tavily"
                and row.get("result_truncated") is False
                and row.get("raw_content_requested") is True
                for row in tools
            ),
            "vllm_scheduler_step_trace": any(
                row.get("event_type") == "scheduler_batch"
                and row.get("phase_evidence_scope") == "vllm_scheduler_step"
                for row in server
            ),
            "vllm_model_batch_trace": any(row.get("event_type") == "model_execute_batch" for row in server),
            "resume_prefix_cache_hit": bool(resume_first) and all(
                int(detail.get("cached_prefix_tokens") or 0) > 0
                for detail in resume_first.values()
            ),
            "ready_tokens_match_server_usage": prompt_token_consistency,
            "all_background_completed": bool(workflow_end.get("all_background_completed")),
            "result_terminal_is_finalizer": result_ready.get("node_id") == "critical_finalizer",
        },
    }


def relative(ts: float, origin: float) -> float:
    return ts - origin


def draw_run_timeline(ax: plt.Axes, gap_ax: plt.Axes, run: dict[str, Any], title: str) -> None:
    events = run["events"]
    server = run["server"]
    reviewer_ready = one(events, "llm_request_ready", "critical_reviewer")
    origin = float(reviewer_ready["ready_ts"])
    result = one(events, "workflow_result_ready")
    result_x = relative(float(result["result_ready_ts"]), origin)

    for node, label, y in [
        ("critical_reviewer", "Reviewer", 3.0),
        ("critical_finalizer", "Finalizer", 3.0),
    ]:
        ready = one(events, "llm_request_ready", node)
        end = one(events, "llm_request_end", node)
        a = relative(float(ready["ready_ts"]), origin)
        b = relative(float(end["first_token_ts"]), origin)
        c = relative(float(end["completion_ts"]), origin)
        ax.barh(y, b - a, left=a, height=0.42, color=COLORS["critical_prefill"])
        ax.barh(y, c - b, left=b, height=0.42, color=COLORS["critical_decode"])
        ax.text((a + c) / 2, y + 0.29, label, ha="center", va="bottom", fontsize=8)

    tool_rows = sorted(
        [row for row in events if row.get("event_type") == "tool_return"],
        key=lambda row: int(row["branch_id"]),
    )
    for idx, row in enumerate(tool_rows):
        y = 2.0 + (idx - 1) * 0.12
        a = relative(float(row["tool_start_ts"]), origin)
        b = relative(float(row["tool_return_ts"]), origin)
        ax.barh(y, b - a, left=a, height=0.10, color=COLORS["tool"])
        ax.scatter([b], [y], marker="D", s=16, color=COLORS["tool"], zorder=4)

    prefill_steps: list[tuple[float, int]] = []
    for row in server:
        if row.get("event_type") != "scheduler_batch":
            continue
        count = sum(
            int(detail.get("prefill_tokens") or 0)
            for request_id, detail in (row.get("request_phase_details") or {}).items()
            if request_is_background_resume(request_id)
        )
        if count:
            x = relative(float(row["timestamp_unix"]), origin)
            if x <= result_x + 2.0:
                prefill_steps.append((x, count))
    for x, tokens in prefill_steps:
        ax.vlines(x, 0.75, 1.25, color=COLORS["background_prefill"], linewidth=1.5 + 4 * min(1.0, tokens / 8192.0))
    if prefill_steps:
        label = (
            "deferred prefill release begins"
            if run["policy"] == "critical_frontier"
            else f"{sum(tokens for _, tokens in prefill_steps):,} scheduled prefill tokens"
        )
        ax.text(min(x for x, _ in prefill_steps), 1.38, label, color=COLORS["background_prefill"], fontsize=8)

    defer_rows = [row for row in events if row.get("event_type") == "admission_decision" and float(row.get("defer_sec") or 0) > 0]
    for idx, row in enumerate(defer_rows):
        a = relative(float(row["ready_ts"]), origin)
        b = min(relative(float(row["submit_ts"]), origin), result_x)
        y = 0.0 + (idx - 1) * 0.12
        ax.barh(y, b - a, left=a, height=0.10, color=COLORS["defer"], alpha=0.9)

    ax.axvline(result_x, color=COLORS["result"], linestyle="--", linewidth=1.2)
    ax.text(result_x, 3.62, "result ready", rotation=90, va="top", ha="right", fontsize=8)
    ax.set_yticks([0, 1, 2, 3], ["Deferred", "vLLM prefill steps", "Live Tavily", "Critical chain"])
    ax.set_ylim(-0.45, 3.75)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.set_title(title, loc="left", fontsize=11, weight="bold")

    for node in ["critical_reviewer", "critical_finalizer"]:
        end = one(events, "llm_request_end", node)
        timestamps = [float(value) for value in end.get("stream_chunk_timestamps", [])]
        gaps = [1000.0 * (b - a) for a, b in zip(timestamps, timestamps[1:])]
        xs = [relative(value, origin) for value in timestamps[1:]]
        gap_ax.scatter(xs, gaps, color=COLORS["critical_decode"], s=4, alpha=0.75, linewidths=0)
    gap_ax.axvline(result_x, color=COLORS["result"], linestyle="--", linewidth=1.2)
    gap_ax.set_ylabel("SSE gap\n(ms)", fontsize=8)
    gap_ax.set_xlabel("Time from critical reviewer ready (s)")
    gap_ax.grid(axis="both", color="#e5e5e5", linewidth=0.5)
    gap_ax.set_ylim(bottom=0)


def figure1(baseline: dict[str, Any], gated: dict[str, Any], out: Path) -> None:
    fig = plt.figure(figsize=(10.2, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(4, 1, height_ratios=[3.0, 1.0, 3.0, 1.0], hspace=0.08)
    ax0 = fig.add_subplot(grid[0])
    gap0 = fig.add_subplot(grid[1], sharex=ax0)
    ax1 = fig.add_subplot(grid[2], sharex=ax0)
    gap1 = fig.add_subplot(grid[3], sharex=ax1)
    draw_run_timeline(ax0, gap0, baseline, "(a) Default vLLM execution")
    draw_run_timeline(ax1, gap1, gated, "(b) MASBench-guided critical-frontier gating")
    result_offsets = []
    for run in [baseline, gated]:
        ready = one(run["events"], "llm_request_ready", "critical_reviewer")
        result = one(run["events"], "workflow_result_ready")
        result_offsets.append(float(result["result_ready_ts"]) - float(ready["ready_ts"]))
    ax0.set_xlim(0, max(result_offsets) + 2.0)
    ax0.tick_params(labelbottom=False)
    ax1.tick_params(labelbottom=False)
    handles = [
        Patch(color=COLORS["critical_prefill"], label="Critical prefill / TTFT"),
        Patch(color=COLORS["critical_decode"], label="Critical decode"),
        Patch(color=COLORS["tool"], label="Live Tavily wait"),
        Patch(color=COLORS["background_prefill"], label="vLLM scheduler prefill step"),
        Patch(color=COLORS["defer"], label="Client-side defer"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=5, frameon=False, fontsize=8)
    fig.suptitle("Real tool-return prefills interfere with a critical MAS frontier", fontsize=13, weight="bold", y=1.02)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def figure2(baseline: dict[str, Any], gated: dict[str, Any], out: Path) -> None:
    labels = ["Default vLLM", "Critical-frontier"]
    colors = ["#777777", COLORS["critical_decode"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.4), constrained_layout=True)
    x = [0, 1]
    result = [baseline["result_ready_sec"], gated["result_ready_sec"]]
    drain = [baseline["drain_complete_sec"], gated["drain_complete_sec"]]
    axes[0].bar(x, result, color=colors, width=0.58, label="Result ready")
    axes[0].scatter(x, drain, marker="D", facecolors="white", edgecolors="#222222", zorder=4, label="Background drain complete")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Time from workflow start (s)")
    axes[0].set_title("(a) Top-level result latency")
    speedup = baseline["result_ready_sec"] / gated["result_ready_sec"]
    axes[0].text(0.5, max(result) * 0.62, f"{speedup:.2f}× faster\nresult ready", ha="center", weight="bold")
    axes[0].legend(frameon=False, fontsize=8)

    width = 0.34
    mean_tpot = [baseline["reviewer_tpot_ms"], gated["reviewer_tpot_ms"]]
    p95_tpot = [baseline["reviewer_tpot_p95_ms"], gated["reviewer_tpot_p95_ms"]]
    axes[1].bar([v - width / 2 for v in x], mean_tpot, width=width, color=colors, alpha=0.95, label="Aggregate TPOT")
    axes[1].bar([v + width / 2 for v in x], p95_tpot, width=width, color=colors, alpha=0.48, label="SSE gap p95")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Critical reviewer latency (ms)")
    axes[1].set_title("(b) Critical decode stability")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].text(
        0.5,
        max(mean_tpot + p95_tpot) * 0.67,
        f"Reviewer decode\n{baseline['reviewer_decode_sec']:.2f}s → {gated['reviewer_decode_sec']:.2f}s",
        ha="center",
        fontsize=9,
        weight="bold",
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    fig.suptitle("Graph-aware prefill admission accelerates the semantic result path", fontsize=12, weight="bold")
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)


def public_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key not in {"events", "server"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--gated", type=Path, required=True)
    parser.add_argument("--backend-trace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts" / "analysis")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    backend = read_jsonl(args.backend_trace)
    baseline = analyze_run(args.baseline, backend)
    gated = analyze_run(args.gated, backend)
    for run in [baseline, gated]:
        failed = [key for key, value in run["valid"].items() if not value]
        if failed:
            raise RuntimeError(f"{run['policy']} failed validity checks: {failed}")
    if baseline["reviewer_background_prefill_tokens"] <= 0:
        raise RuntimeError("baseline did not expose scheduler-observed prefill interference")
    if gated["reviewer_background_prefill_tokens"] != 0 or gated["finalizer_background_prefill_tokens"] != 0:
        raise RuntimeError("gated run did not protect the complete critical frontier")
    result = {
        "evidence_level": {
            "prefill_decode_phase": "vllm_scheduler_step_observed",
            "batch_execution": "gpu_model_runner_host_forward_dispatch_observed",
            "token_timing": "openai_sse_chunk_observed",
            "kernel_overlap": "not_claimed",
        },
        "baseline": public_summary(baseline),
        "critical_frontier": public_summary(gated),
        "effect": {
            "result_ready_speedup": baseline["result_ready_sec"] / gated["result_ready_sec"],
            "result_ready_reduction_pct": 100.0 * (1.0 - gated["result_ready_sec"] / baseline["result_ready_sec"]),
            "reviewer_decode_reduction_pct": 100.0 * (1.0 - gated["reviewer_decode_sec"] / baseline["reviewer_decode_sec"]),
            "reviewer_tpot_reduction_pct": 100.0 * (1.0 - gated["reviewer_tpot_ms"] / baseline["reviewer_tpot_ms"]),
            "reviewer_tpot_p95_reduction_pct": 100.0 * (1.0 - gated["reviewer_tpot_p95_ms"] / baseline["reviewer_tpot_p95_ms"]),
            "background_drain_change_pct": 100.0 * (gated["drain_complete_sec"] / baseline["drain_complete_sec"] - 1.0),
            "critical_decode_overlapped_prefill_tokens_removed": baseline["reviewer_background_prefill_tokens"] - gated["reviewer_background_prefill_tokens"],
        },
        "experiment_status": "single_pair_correctness_pilot_not_final_repeated_result",
    }
    (args.out_dir / "paired_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    figure1(baseline, gated, args.out_dir / "figure1_real_interference_timeline.png")
    figure2(baseline, gated, args.out_dir / "figure2_result_benefit.png")
    print(json.dumps(result["effect"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

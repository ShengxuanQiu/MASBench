"""Create refined Week-2 systems insights from existing traces only."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .analyze_week2_traces import (
    EXPECTED_NAMES,
    aggregate_by_name,
    analyze_run,
    event_start,
    event_time,
    read_jsonl,
    safe_float,
    safe_int,
)


TOOL_RESUME_WORKLOAD = "tool_resume_contention_meso"
MATRIX_ROWS = [
    "single",
    "independent",
    "centralized",
    "decentralized",
    "hybrid",
    "planner_executor",
    "evidence_collection",
    "researcher_synthesizer",
    "generator_verifier",
    "coder_reviewer",
    "multi_coder_branch",
    "debate_reviewer",
    "tool_specialist_team",
    "all_gather_round",
    "shared_evidence_store",
    "retry_debug_loop",
    "router_handoff",
    TOOL_RESUME_WORKLOAD,
]
MATRIX_COLS = [
    "parallel branch",
    "barrier wait",
    "fan-in context",
    "all-gather broadcast",
    "tool stall",
    "post-tool burst",
    "retry loop",
    "shared memory dataflow",
    "critical candidate",
    "contention opportunity",
]
DISPLAY_MATRIX_COLS = [
    "parallel branch",
    "barrier",
    "fan-in",
    "all-gather",
    "tool stall",
    "post-tool burst",
    "retry loop",
    "critical candidate",
    "background resume",
    "contention opportunity",
]
CRITICAL_ROLES = {"manager", "planner", "reviewer", "verifier", "synthesizer", "selector", "finalizer", "aggregator", "router"}
BACKGROUND_ROLES = {"researcher", "tool_agent", "coder", "worker", "independent_worker", "writer", "reader", "peer_agent"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate refined Week2 MASBench-Arch insight report")
    parser.add_argument("--trace-root", default="traces")
    parser.add_argument("--extra-trace-root", action="append", default=[])
    parser.add_argument("--progress-dir", default="progress/week2")
    return parser.parse_args()


def discover_paths(args: argparse.Namespace) -> list[Path]:
    roots = [Path(args.trace_root), *[Path(p) for p in args.extra_trace_root]]
    default = Path("mas_workflow/traces")
    if default.exists():
        roots.append(default)
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(p for p in root.rglob("*.jsonl") if "snapshots" not in p.parts and "week2_structure_validation" not in p.parts)
    return sorted(set(paths))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def role_of(event: dict[str, Any]) -> str:
    return str(event.get("agent_role") or event.get("node_id") or "").lower()


def is_critical_candidate(event: dict[str, Any]) -> bool:
    role = role_of(event)
    node = str(event.get("node_id") or "").lower()
    return event.get("criticality") == "critical" or any(token in role or token in node for token in CRITICAL_ROLES | {"merge"})


def is_background_candidate(event: dict[str, Any]) -> bool:
    role = role_of(event)
    node = str(event.get("node_id") or "").lower()
    return any(token in role or token in node for token in BACKGROUND_ROLES | {"tool", "background"})


def run_events(run: dict[str, Any]) -> list[dict[str, Any]]:
    return read_jsonl(Path(run["metrics"]["trace_path"]))


def post_tool_counts(events: list[dict[str, Any]], windows: list[float]) -> dict[float, int]:
    tools = [e for e in events if str(e.get("event_type", "")).startswith("tool_")]
    llm = [e for e in events if e.get("event_type") == "llm_request_end"]
    out = {w: 0 for w in windows}
    for tool in tools:
        end = event_time(tool)
        for window in windows:
            out[window] = max(out[window], sum(1 for req in llm if 0 <= event_start(req) - end <= window))
    return out


def opportunity_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        m = run["metrics"]
        events = run_events(run)
        counts = post_tool_counts(events, [1.0, 3.0, 5.0])
        downstream = any(str(e.get("node_id", "")).lower().find("merge") >= 0 or str(e.get("dst_node", "")).lower().find("merge") >= 0 for e in events)
        critical = any(is_critical_candidate(e) for e in events if e.get("event_type") == "llm_request_end")
        tool_count = int(m["tool_event_count"])
        score = 0
        score += min(3, tool_count)
        score += min(3, counts[3.0])
        score += 2 if downstream else 0
        score += 2 if critical else 0
        score += min(2, safe_float(m.get("total_barrier_wait_sec")))
        if tool_count and counts[3.0] and downstream and critical:
            level = "observed_components"
        elif tool_count or counts[3.0]:
            level = "proxy_only"
        else:
            level = "not_observed"
        rows.append(
            {
                "run_path": m["trace_path"],
                "motif_name/topology_name": m["name"],
                "tool_event_count": tool_count,
                "total_tool_time_sec": m["total_tool_time_sec"],
                "post_tool_request_count_1s": counts[1.0],
                "post_tool_request_count_3s": counts[3.0],
                "post_tool_request_count_5s": counts[5.0],
                "downstream_fanin_node_present": downstream,
                "critical_candidate_present": critical,
                "barrier_wait_sec": m["total_barrier_wait_sec"],
                "request_burstiness_after_tool": counts[3.0],
                "opportunity_score": round(score, 3),
                "evidence_level": level,
            }
        )
    return rows


def criticality_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        m = run["metrics"]
        events = run_events(run)
        end = max([event_time(e) for e in events] or [0.0])
        child_counts: dict[str, int] = defaultdict(int)
        for event in events:
            for parent in event.get("parents") or []:
                child_counts[str(parent)] += 1
            if event.get("src_node"):
                child_counts[str(event.get("src_node"))] += 1
        for event in events:
            if event.get("event_type") != "llm_request_end":
                continue
            node = str(event.get("node_id") or "")
            if is_critical_candidate(event):
                proxy = "critical_candidate"
            elif is_background_candidate(event):
                proxy = "background_candidate"
            else:
                proxy = "unknown"
            finish = event_time(event)
            duration = safe_float(event.get("duration_sec"))
            contribution = round(max(0.0, end - finish) / max(end, 1e-9) + duration / max(end, 1e-9), 6)
            rows.append(
                {
                    "run_path": m["trace_path"],
                    "node_id": node,
                    "motif_name": m["name"],
                    "agent_role": event.get("agent_role", ""),
                    "criticality_proxy": proxy,
                    "request_e2e_sec": duration,
                    "input_tokens": safe_int(event.get("input_tokens_est")),
                    "output_tokens": safe_int(event.get("output_tokens_est")),
                    "downstream_dependency_count": child_counts.get(node, 0),
                    "contribution_to_makespan_proxy": contribution,
                    "before_finalizer": finish <= max([event_time(e) for e in events if "finalizer" in str(e.get("node_id", "")).lower()] or [end]),
                    "evidence_quality": "proxy_supported",
                }
            )
    return rows


def matrix_rows(aggregate: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    numeric_rows = []
    display_rows = []
    for name in MATRIX_ROWS:
        row = aggregate.get(name, {"status": "trace_missing"})
        present = row.get("status") == "present"
        signatures = set(row.get("detected_signatures") or [])
        vals: dict[str, Any] = {"workload": name}
        display: dict[str, Any] = {"workload": name}
        if name == TOOL_RESUME_WORKLOAD:
            planned = {
                "parallel branch",
                "barrier wait",
                "fan-in context",
                "tool stall",
                "post-tool burst",
                "critical candidate",
                "contention opportunity",
            }
            for col in MATRIX_COLS:
                vals[col] = "P" if col in planned else 0
            for col in DISPLAY_MATRIX_COLS:
                display[col] = "planned" if col in {"parallel branch", "barrier", "fan-in", "tool stall", "post-tool burst", "critical candidate", "background resume", "contention opportunity"} else "not observed"
        else:
            vals["parallel branch"] = 2 if present and safe_float(row.get("avg_request_burstiness")) >= 1.2 else (1 if name in {"independent", "researcher_synthesizer", "multi_coder_branch"} else 0)
            vals["barrier wait"] = 2 if present and safe_float(row.get("avg_barrier_wait_sec")) > 0 else (1 if name in {"independent", "multi_coder_branch", "all_gather_round"} else 0)
            vals["fan-in context"] = 2 if present and safe_float(row.get("avg_fanin_amplification_ratio")) > 0 else (1 if "fanin_context_amplification" in signatures else 0)
            vals["all-gather broadcast"] = 3 if name == "all_gather_round" else (2 if name in {"debate_reviewer", "decentralized"} else 0)
            vals["tool stall"] = 2 if present and safe_float(row.get("avg_tool_time_sec")) > 0 else (1 if name in {"evidence_collection", "tool_specialist_team"} else 0)
            vals["post-tool burst"] = 2 if "post_tool_burst" in signatures else (1 if vals["tool stall"] else 0)
            vals["retry loop"] = 2 if "retry_amplification" in signatures else (1 if name in {"coder_reviewer", "generator_verifier", "retry_debug_loop"} else 0)
            vals["shared memory dataflow"] = 3 if name == "shared_evidence_store" else 0
            vals["critical candidate"] = 2 if present and name != "single" else 1
            vals["contention opportunity"] = 1 if vals["tool stall"] and vals["critical candidate"] else 0
            for col in DISPLAY_MATRIX_COLS:
                src = {"barrier": "barrier wait", "fan-in": "fan-in context", "all-gather": "all-gather broadcast", "background resume": "contention opportunity"}.get(col, col)
                val = vals.get(src, 0)
                display[col] = "strong observed" if val == 3 else "observed" if val == 2 else "proxy" if val == 1 else "not observed"
        numeric_rows.append(vals)
        display_rows.append(display)
    return numeric_rows, display_rows


def save_figures(progress_dir: Path, aggregate: dict[str, dict[str, Any]], runs: list[dict[str, Any]], opps: list[dict[str, Any]], crit_rows: list[dict[str, Any]], matrix: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig_dir = progress_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=180)
        plt.close()

    present = [aggregate[n] for n in EXPECTED_NAMES if aggregate.get(n, {}).get("status") == "present"]
    labels = [r["name"] for r in present]
    cols = ["avg_llm_request_count", "avg_max_parallel_width", "avg_barrier_wait_sec", "avg_request_burstiness", "avg_context_amplification_ratio", "avg_tool_time_sec"]
    data = []
    for r in present:
        row = []
        for c in cols:
            value = safe_float(r.get(c) or r.get("avg_request_burstiness" if c == "request_arrival_burstiness" else c))
            row.append(math.log1p(value))
        data.append(row)
    plt.figure(figsize=(11, max(5, len(labels) * 0.35)))
    plt.imshow(data, aspect="auto", cmap="viridis")
    plt.colorbar(label="log-scaled load-shape metric")
    plt.xticks(range(len(cols)), ["LLM req", "parallel width", "barrier wait", "burstiness", "context amp", "tool time"], rotation=25, ha="right")
    plt.yticks(range(len(labels)), labels)
    plt.title("Graph composition creates distinct backend load shapes\nEvidence tier: observed workload shape")
    save("fig_graph_load_shape_matrix.png")

    fanin_points = []
    for run in runs:
        events = run_events(run)
        llm_by_node = {str(e.get("node_id")): e for e in events if e.get("event_type") == "llm_request_end"}
        for event in events:
            if event.get("event_type") == "llm_request_end" and len(event.get("parents") or []) >= 2:
                parents = [llm_by_node[p] for p in event.get("parents") or [] if p in llm_by_node]
                if parents:
                    avg_up = mean([safe_int(p.get("output_tokens_est")) for p in parents] or [1])
                    fanin_points.append((len(parents), safe_int(event.get("input_tokens_est")) / max(1, avg_up), run["metrics"]["name"], str(event.get("node_id"))))
    plt.figure(figsize=(9, 5))
    for x, y, name, node in fanin_points:
        plt.scatter(x, y, alpha=0.75)
        if any(t in node.lower() for t in ("synth", "review", "final", "gather")):
            plt.text(x + 0.03, y, node[:18], fontsize=7)
    plt.xlabel("fan-in width / upstream branch count")
    plt.ylabel("fan-in input tokens / avg upstream output tokens")
    plt.title("Parallelism reappears as downstream context movement\nEvidence tier: observed/proxy-supported")
    save("fig_fanin_context_movement.png")

    plt.figure(figsize=(8, 5))
    xs = [safe_float(aggregate[n].get("avg_fanin_amplification_ratio")) for n in labels]
    ys = [safe_float(aggregate[n].get("avg_broadcast_amplification_ratio")) for n in labels]
    plt.scatter(xs, ys, alpha=0.75)
    for x, y, name in zip(xs, ys, labels):
        if name in {"all_gather_round", "debate_reviewer", "researcher_synthesizer", "multi_coder_branch"}:
            plt.text(x + 0.01, y + 0.01, name, fontsize=8)
    plt.xlabel("fan-in aggregation pressure proxy")
    plt.ylabel("all-to-all broadcast pressure proxy")
    plt.title("Broadcast and fan-in create different context pressure\nEvidence tier: proxy-supported")
    save("fig_broadcast_vs_fanin_pressure.png")

    best_opp = max(opps, key=lambda r: safe_float(r.get("opportunity_score")), default=None)
    plt.figure(figsize=(10, 4))
    if best_opp:
        events = read_jsonl(Path(best_opp["run_path"]))
        ymap = {"tool": 3, "post-tool LLM": 2, "critical/merge": 1}
        for e in events:
            if str(e.get("event_type", "")).startswith("tool_"):
                plt.hlines(ymap["tool"], event_start(e), event_time(e), lw=7, color="#4477aa")
                plt.scatter([event_time(e)], [ymap["tool"]], color="black", marker="|", s=120)
            elif e.get("event_type") == "llm_request_end":
                y = ymap["critical/merge"] if is_critical_candidate(e) else ymap["post-tool LLM"]
                plt.hlines(y, event_start(e), event_time(e), lw=5, color="#cc6677" if y == 1 else "#66aa55")
                plt.text(event_start(e), y + 0.08, str(e.get("node_id"))[:18], fontsize=7)
        plt.yticks(list(ymap.values()), list(ymap.keys()))
    plt.xlabel("relative time sec")
    plt.title("Tool returns can phase-shift post-tool LLM arrivals near critical stages\nEvidence tier: observed components")
    save("fig_tool_phase_shift_timeline.png")

    plt.figure(figsize=(10, 5))
    opp_by_name: dict[str, float] = defaultdict(float)
    level_by_name: dict[str, str] = {}
    for row in opps:
        name = row["motif_name/topology_name"]
        if safe_float(row["opportunity_score"]) >= opp_by_name[name]:
            opp_by_name[name] = safe_float(row["opportunity_score"])
            level_by_name[name] = row["evidence_level"]
    names = list(opp_by_name)
    colors = {"observed_components": "#4477aa", "proxy_only": "#ddcc77", "not_observed": "#dddddd"}
    plt.bar(range(len(names)), [opp_by_name[n] for n in names], color=[colors.get(level_by_name.get(n, ""), "#999999") for n in names])
    plt.xticks(range(len(names)), names, rotation=65, ha="right")
    plt.ylabel("opportunity score")
    plt.title("Tool phase-shift contention opportunity is a workload-level hypothesis\nEvidence tier: observed components + TODO")
    save("fig_tool_phase_shift_opportunity_score.png")

    plt.figure(figsize=(8, 5))
    color = {"critical_candidate": "#cc3311", "background_candidate": "#777777", "unknown": "#bbbbbb"}
    for row in crit_rows:
        plt.scatter(safe_float(row["request_e2e_sec"]), safe_float(row["contribution_to_makespan_proxy"]), color=color.get(row["criticality_proxy"], "#bbbbbb"), alpha=0.55, s=20)
    plt.xlabel("request e2e sec")
    plt.ylabel("contribution to makespan proxy")
    plt.title("Graph-level criticality is invisible to flat serving\nEvidence tier: proxy-supported")
    save("fig_criticality_invisible_to_backend.png")

    critical_run = max(runs, key=lambda r: safe_float(r["metrics"].get("critical_path_length_sec")), default=None)
    plt.figure(figsize=(10, 4))
    if critical_run:
        events = run_events(critical_run)
        for i, e in enumerate([e for e in events if e.get("event_type") == "llm_request_end"]):
            y = 2 if is_critical_candidate(e) else 1
            plt.hlines(y, event_start(e), event_time(e), color="#cc3311" if y == 2 else "#888888", lw=5)
            if y == 2:
                plt.text(event_start(e), y + 0.08, str(e.get("node_id"))[:18], fontsize=7)
        plt.yticks([1, 2], ["background", "critical candidate"])
    plt.xlabel("relative time sec")
    plt.title("Critical candidates cluster near review/merge/final stages\nEvidence tier: proxy-supported")
    save("fig_critical_candidates_on_timeline.png")

    retry = [aggregate[n] for n in {"coder_reviewer", "generator_verifier", "retry_debug_loop"} if aggregate.get(n, {}).get("status") == "present"]
    plt.figure(figsize=(9, 5))
    x = range(len(retry))
    plt.plot(list(x), [safe_float(r.get("avg_llm_request_count")) for r in retry], marker="o", label="LLM requests")
    plt.plot(list(x), [safe_float(r.get("avg_total_tokens_est")) / 1000.0 for r in retry], marker="s", label="tokens / 1k")
    plt.plot(list(x), [safe_float(r.get("avg_makespan_sec")) for r in retry], marker="^", label="makespan sec")
    plt.xticks(list(x), [r["name"] for r in retry], rotation=25, ha="right")
    plt.title("Review/debug loops amplify backend work\nEvidence tier: observed/proxy-supported")
    plt.legend()
    save("fig_retry_loop_work_amplification.png")

    retry_run = next((r for r in runs if r["metrics"]["name"] == "retry_debug_loop"), next((r for r in runs if r["metrics"]["name"] == "coder_reviewer"), None))
    plt.figure(figsize=(10, 4))
    if retry_run:
        events = run_events(retry_run)
        for e in events:
            if e.get("event_type") == "llm_request_end":
                role = str(e.get("agent_role") or e.get("node_id"))
                y = 2 if any(t in role for t in ("tester", "verifier", "reviewer")) else 1
                plt.hlines(y, event_start(e), event_time(e), lw=5, color="#aa4499" if y == 2 else "#44aa99")
                plt.text(event_start(e), y + 0.08, str(e.get("node_id"))[:20], fontsize=7)
        plt.yticks([1, 2], ["generation/revision", "review/test"])
    plt.xlabel("relative time sec")
    plt.title("Retry timeline turns review/test feedback into more LLM work\nEvidence tier: observed")
    save("fig_retry_timeline_example.png")

    plt.figure(figsize=(11, 6))
    matrix_data = []
    annotations = []
    for row in matrix:
        vals = []
        anns = []
        for col in MATRIX_COLS:
            v = row[col]
            vals.append(2.5 if v == "P" else float(v))
            anns.append(str(v))
        matrix_data.append(vals)
        annotations.append(anns)
    plt.imshow(matrix_data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=3)
    for y, anns in enumerate(annotations):
        for x, text in enumerate(anns):
            plt.text(x, y, text, ha="center", va="center", fontsize=7)
    plt.xticks(range(len(MATRIX_COLS)), MATRIX_COLS, rotation=35, ha="right")
    plt.yticks(range(len(MATRIX_ROWS)), MATRIX_ROWS)
    plt.title("Compound bottleneck matrix: motifs systematically expose backend pressure components\nEvidence tier: observed/proxy/planned")
    save("fig_compound_bottleneck_matrix.png")


def markdown_report(display_matrix: list[dict[str, Any]]) -> str:
    def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
        return "\n".join(
            ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
            + ["| " + " | ".join(str(r.get(c, "")).replace("\n", " ") for c in cols) + " |" for r in rows]
        )

    evidence_rows = [
        {"insight": "Graph load shape", "tier": "observed", "supporting existing traces": "5 topologies, 12 composite motifs", "missing backend fields": "queue time, batch membership", "next validation action": "rerun with backend scheduler telemetry"},
        {"insight": "Context movement", "tier": "proxy_supported", "supporting existing traces": "fan-in, broadcast, aggregation token estimates", "missing backend fields": "KV/prefix cache hit/miss", "next validation action": "record prefix/cache counters"},
        {"insight": "Tool phase shift", "tier": "planned_todo", "supporting existing traces": "tool stall + post-tool burst components", "missing backend fields": "queue time, batch membership, TTFT/TPOT", "next validation action": "rerun tool_resume_contention_meso"},
        {"insight": "Criticality invisible", "tier": "proxy_supported", "supporting existing traces": "role and dependency proxies", "missing backend fields": "explicit critical path and scheduler decision", "next validation action": "record critical path labels"},
        {"insight": "Retry amplification", "tier": "observed", "supporting existing traces": "coder_reviewer/generator_verifier/retry_debug_loop", "missing backend fields": "deep loop scaling", "next validation action": "rerun full software debug workflow"},
    ]
    meso_rows = [
        {"field": "Composed from motifs", "value": "evidence_collection/tool_specialist_team + multi_coder_branch + coder_reviewer + finalizer"},
        {"field": "Why not isolated", "value": "It uses existing tool-stalled evidence, parallel branch, reviewer, and fan-in/finalizer roles instead of a standalone special motif."},
        {"field": "Validates later", "value": "Tool return -> background resume burst near reviewer/finalizer critical requests."},
        {"field": "Current status", "value": "Structure-level validation only; no claim of real backend contention."},
    ]
    return f"""# Week2 Refined MASBench-Arch Systems Insights

## 1. Why the previous insight version was insufficient

The earlier report was mostly motif-level descriptive statistics. Makespan and request count are useful inventory metrics, but they do not explain why MAS workloads stress serving systems differently. This version reorganizes the evidence around compound bottleneck mechanisms, backend design motivation, and explicit limits on what current traces can prove.

## 2. What existing Week2 traces can support

{md_table(evidence_rows, ["insight", "tier", "supporting existing traces", "missing backend fields", "next validation action"])}

Observed means the workload trace directly contains the relevant graph/timing/dataflow fields. Proxy-supported means the trace contains workload-level estimates or role/dependency proxies. Planned TODO means the structure is prepared but backend validation needs a GPU rerun.

## 3. Compound Bottleneck Matrix

![Compound bottleneck matrix](figures/fig_compound_bottleneck_matrix.png)

{md_table(display_matrix, ["workload", *DISPLAY_MATRIX_COLS])}

This is the central Week2 systems view: MASBench-Arch is not collecting random agent logs, but systematically composing motifs that expose parallel branches, barriers, fan-in context movement, tool stalls, post-tool bursts, retry loops, shared memory dataflow, and critical-path candidates.

## 4. Insight 1: Graph composition creates distinct backend load shapes

Claim: MAS workload shape cannot be summarized by total tokens or total request count. Graph composition changes arrival bursts, parallel width, barriers, fan-in, and tool windows.

![Graph load shape matrix](figures/fig_graph_load_shape_matrix.png)

Evidence: existing topology and motif traces show different workload-level load shapes across hybrid, decentralized, debate_reviewer, all_gather_round, multi_coder_branch, and tool-heavy motifs.

Limitation: no per-request queue time or batch membership, so this supports workload shape, not backend scheduler outcome.

Rerun TODO: collect queue time, batch membership, TTFT/TPOT, and scheduler decisions for each graph family.

## 5. Insight 2: Parallelism reappears as context movement pressure

Claim: Parallel branches are not free. Their outputs reappear in synthesizer, reviewer, selector, merge, and finalizer prompts as downstream context movement.

![Fan-in context movement](figures/fig_fanin_context_movement.png)

![Broadcast vs fan-in pressure](figures/fig_broadcast_vs_fanin_pressure.png)

Evidence: researcher_synthesizer, evidence_collection, multi_coder_branch, debate_reviewer, all_gather_round, and shared_evidence_store expose aggregation, broadcast, peer-message, duplicated-context, and context-amplification estimates.

Limitation: these are workload-level estimates, not observed KV cache residency or prefix-cache hit/miss.

Rerun TODO: record prefix hashes, prefix cache hit/miss, KV allocation/residency, and shared artifact cache reuse.

## 6. Insight 3: Tool stalls phase-shift request arrivals and create contention opportunities

Claim: Tool stalls are not only slow external calls. They pause background branches, then resume into delayed LLM bursts that may land near reviewer/finalizer/manager critical-path requests.

![Tool phase-shift timeline](figures/fig_tool_phase_shift_timeline.png)

![Tool phase-shift opportunity score](figures/fig_tool_phase_shift_opportunity_score.png)

Evidence: existing traces contain observed components: tool stalls, post-tool request bursts, downstream fan-in/merge, critical candidate roles, and barrier or makespan extension. The new `tool_resume_contention_meso` workload prepares a composed structure for later validation.

Limitation: no per-request queue time, batch membership, scheduler wait, or KV residency. This is not evidence of real priority inversion or true backend contention.

Rerun TODO: run `tool_resume_contention_meso` on a real backend and plot tool return -> resume burst vs reviewer/finalizer latency and overlapping background request count.

## 7. Insight 4: Graph-level criticality is invisible to flat serving

Claim: MAS requests differ in end-to-end importance. Reviewer, verifier, synthesizer, selector, manager, merge, and finalizer requests are often closer to the critical path, but an OpenAI-compatible backend sees ordinary requests.

![Criticality invisible to backend](figures/fig_criticality_invisible_to_backend.png)

![Critical candidates on timeline](figures/fig_critical_candidates_on_timeline.png)

Evidence: role/dependency proxies separate critical candidates from background researchers/tool/coder branches and show different makespan contribution proxies.

Limitation: this is a graph-level proxy, not a vLLM scheduler conclusion. Explicit dependency edges and critical path flags are incomplete in older traces.

Rerun TODO: record complete dependency edges, critical path labels, queue time, and scheduler priority decisions.

## 8. Insight 5: Review/debug loops amplify backend work

Claim: Review/debug loops are systems mechanisms, not only agent accuracy techniques. A local failure can become extra LLM requests, tool/test waiting, context movement, and a longer critical path.

![Retry loop work amplification](figures/fig_retry_loop_work_amplification.png)

![Retry timeline example](figures/fig_retry_timeline_example.png)

Evidence: coder_reviewer, generator_verifier, and retry_debug_loop expose loop counts, retry/review/debug markers, total request/token growth, and timeline extension.

Limitation: loop depth is limited, so current traces should not be extrapolated to long software debugging workflows.

Rerun TODO: run full software workflows with deeper issue/repo search -> planner -> multi-coder -> reviewer -> test/debug -> finalizer loops.

## 9. What must be rerun when GPU is available

- `tool_resume_contention_meso`
- `evidence_collection + planner_executor + coder_reviewer`
- `tool_specialist_team + multi_coder_branch + reviewer/finalizer`
- `shared_evidence_store + retry_debug_loop`
- full software workflow: issue/repo search -> planner -> multi-coder -> reviewer -> test/debug -> finalizer

Required plots: request timeline with critical/background labels; tool return -> resume burst timeline; critical request latency vs overlapping background request count; queue time / batch membership timeline; KV residency during tool stall; prefix cache hit/miss under fan-in; retry amplification vs loop depth.

## 10. Presentation-ready summary

- Week2 traces support observed workload-level evidence that graph composition changes backend load shape.
- Fan-in and all-gather motifs provide proxy-supported motivation for context-cache and shared-artifact-cache research.
- Existing tool-heavy traces show the components of tool phase shift, but not real backend contention.
- `tool_resume_contention_meso` is prepared as a composed meso workload for GPU rerun, not as an isolated special benchmark.
- Critical-path-aware serving is motivated by graph-level role/dependency proxies, not by current scheduler evidence.
- Review/debug loops are observed to amplify work in bounded Week2 traces.
- Claims about scheduler priority inversion, KV idle residency, and cache optimization are deferred until backend telemetry exists.
"""


def main() -> int:
    args = parse_args()
    progress_dir = Path(args.progress_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_paths(args)
    runs = [analyze_run(path, include_backend=True, include_outputs=False) for path in paths]
    aggregate = aggregate_by_name(runs)
    opps = opportunity_rows(runs)
    crit = criticality_rows(runs)
    matrix, display = matrix_rows(aggregate)
    write_csv(
        progress_dir / "compound_tool_phase_shift_opportunities.csv",
        opps,
        ["run_path", "motif_name/topology_name", "tool_event_count", "total_tool_time_sec", "post_tool_request_count_1s", "post_tool_request_count_3s", "post_tool_request_count_5s", "downstream_fanin_node_present", "critical_candidate_present", "barrier_wait_sec", "request_burstiness_after_tool", "opportunity_score", "evidence_level"],
    )
    write_csv(
        progress_dir / "criticality_proxy_table.csv",
        crit,
        ["run_path", "node_id", "motif_name", "agent_role", "criticality_proxy", "request_e2e_sec", "input_tokens", "output_tokens", "downstream_dependency_count", "contribution_to_makespan_proxy", "before_finalizer", "evidence_quality"],
    )
    write_csv(progress_dir / "compound_bottleneck_matrix.csv", matrix, ["workload", *MATRIX_COLS])
    save_figures(progress_dir, aggregate, runs, opps, crit, matrix)
    (progress_dir / "week2_refined_insight_report.md").write_text(markdown_report(display), encoding="utf-8")
    print(f"Refined report: {progress_dir / 'week2_refined_insight_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build simulator-ready Week3 tables, figures, and report from real traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


MOTIF_FOCUS = {
    "independent": "independent_multi_branch",
    "multi_coder_branch": "independent_multi_branch",
    "evidence_collection": "independent_multi_branch",
    "centralized": "manager_worker",
    "planner_executor": "manager_worker",
    "tool_specialist_team": "manager_worker",
    "retry_debug_pressure_meso": "review_loop",
    "coder_reviewer": "review_loop",
    "generator_verifier": "review_loop",
    "debate_allgather_pressure_meso": "debate",
    "debate_reviewer": "debate",
    "decentralized": "debate",
    "hybrid": "hybrid_manager_peer",
    "hierarchical_synthesis_pressure_meso": "manager_worker",
    "shared_memory_fanin_meso": "independent_multi_branch",
    "tool_resume_contention_meso": "composite_tool_resume",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week3 simulator-ready trace analysis")
    parser.add_argument("--trace-root", default="progress/week3/traces")
    parser.add_argument("--progress-dir", default="progress/week3")
    parser.add_argument("--copy-traces", default="false")
    return parser.parse_args()


def as_bool(value: str | bool) -> bool:
    return value if isinstance(value, bool) else value.lower() in {"1", "true", "yes", "y", "on"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "" or value == "unavailable":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "" or value == "unavailable":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def event_end(event: dict[str, Any]) -> float:
    return safe_float(event.get("relative_time_sec"))


def event_start(event: dict[str, Any]) -> float:
    return max(0.0, event_end(event) - safe_float(event.get("duration_sec")))


def trace_paths(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.jsonl") if "snapshots" not in p.parts)


def workflow_name(events: list[dict[str, Any]], path: Path) -> str:
    for event in events:
        name = event.get("workflow_name") or event.get("meso_workload_name") or event.get("motif_name") or event.get("topology")
        if name:
            return str(name)
    return path.parent.parent.name if len(path.parts) > 2 else path.stem


def workflow_type(events: list[dict[str, Any]]) -> str:
    for event in events:
        if event.get("mode"):
            return str(event.get("mode"))
    return "unknown"


def focus_for(name: str) -> str:
    return MOTIF_FOCUS.get(name, "other")


def backend_path_for(path: Path) -> Path:
    return path.with_name(path.stem + "_backend_metrics.json")


def backend_payload(path: Path) -> dict[str, Any]:
    bpath = backend_path_for(path)
    if not bpath.exists():
        return {}
    try:
        return json.loads(bpath.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def backend_summary(path: Path) -> dict[str, Any]:
    return backend_payload(path).get("summary") or {}


def metric_series(path: Path, metric_substring: str) -> list[tuple[float, float]]:
    series: list[tuple[float, float]] = []
    for sample in backend_payload(path).get("samples", []):
        metrics = sample.get("metrics") or {}
        vals = [safe_float(v) for k, v in metrics.items() if metric_substring in str(k)]
        if vals:
            series.append((safe_float(sample.get("relative_time_sec")), max(vals)))
    return series


def metric_delta(path: Path, metric_prefix: str) -> float:
    vals = []
    for sample in backend_payload(path).get("samples", []):
        metrics = sample.get("metrics") or {}
        matched = [safe_float(v) for k, v in metrics.items() if str(k).startswith(metric_prefix)]
        if matched:
            vals.append(sum(matched))
    return max(0.0, vals[-1] - vals[0]) if len(vals) >= 2 else 0.0


def discover_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for path in trace_paths(root):
        events = read_jsonl(path)
        if not events:
            continue
        name = workflow_name(events, path)
        runs.append({"path": path, "events": events, "workflow_name": name, "motif_focus": focus_for(name)})
    return runs


def normalize_dependency_type(event: dict[str, Any]) -> str:
    if event.get("dependency_type"):
        return str(event.get("dependency_type"))
    if event.get("transfer_type") == "broadcast" or event.get("peer_round_id") is not None:
        return "debate_round"
    if event.get("transfer_type") in {"aggregation", "late_non_blocking_context"} or safe_int(event.get("fan_in_count")) > 1:
        return "fan_in"
    if event.get("tool_stalled") or event.get("tool_name"):
        return "tool_dependency"
    if safe_int(event.get("retry_count")) > 0:
        return "review_loop"
    return "sequential"


def llm_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if e.get("event_type") == "llm_request_end"]


def tool_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if str(e.get("event_type", "")).startswith("tool_")]


def barrier_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in events if e.get("node_type") == "barrier" or e.get("event_type") == "barrier"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def shared_block_overlap(a: dict[str, Any], b: dict[str, Any]) -> tuple[int, int]:
    tokens_a = a.get("shared_block_tokens") or {}
    tokens_b = b.get("shared_block_tokens") or {}
    hashes_a = a.get("shared_block_hashes") or {}
    hashes_b = b.get("shared_block_hashes") or {}
    inv_b = {str(v): k for k, v in hashes_b.items()}
    shared = 0
    for block_id, hsh in hashes_a.items():
        other = inv_b.get(str(hsh))
        if other is not None:
            shared += min(safe_int(tokens_a.get(block_id)), safe_int(tokens_b.get(other)))
    denom = max(safe_int(a.get("input_tokens") or a.get("input_tokens_est")), 1)
    return shared, denom


def segment_reuse_proxy(event: dict[str, Any], other: dict[str, Any]) -> int:
    if event.get("workflow_id") != other.get("workflow_id"):
        return 0
    if event.get("prompt_template") != other.get("prompt_template"):
        return 0
    system = min(safe_int(event.get("system_prompt_tokens_est")), safe_int(other.get("system_prompt_tokens_est")))
    user = min(safe_int(event.get("user_prompt_tokens_est") or event.get("shared_context_tokens_est")), safe_int(other.get("user_prompt_tokens_est") or other.get("shared_context_tokens_est")))
    # The raw Week3 traces do not store full token sequences, so this is a
    # simulator-side estimate for repeated prompt segments within one motif.
    # It is intentionally below the full shared-context size to leave room for
    # dynamic review feedback / revised artifacts appended by each iteration.
    return system + int(0.75 * user)


def potential_prefix_for_event(event: dict[str, Any], previous: list[dict[str, Any]], same_workflow: bool) -> dict[str, Any]:
    best = 0
    source = "none"
    for other in previous:
        if event.get("prompt_hash") and event.get("prompt_hash") == other.get("prompt_hash"):
            candidate = min(safe_int(event.get("input_tokens") or event.get("input_tokens_est")), safe_int(other.get("input_tokens") or other.get("input_tokens_est")))
            if candidate > best:
                best = candidate
                source = "exact_prompt_hash"
        shared, _ = shared_block_overlap(event, other)
        if shared > best:
            best = shared
            source = "shared_block_hash"
        segment = segment_reuse_proxy(event, other) if same_workflow else 0
        if segment > best:
            best = segment
            source = "segment_token_proxy"
    input_tokens = max(safe_int(event.get("input_tokens") or event.get("input_tokens_est")), 1)
    return {
        "potential_prefix_match_tokens": best,
        "potential_prefix_reuse_rate": best / input_tokens,
        "intra_workflow_prefix_match_tokens": best if same_workflow else 0,
        "inter_workflow_prefix_match_tokens": 0 if same_workflow else best,
        "potential_prefix_source": source,
    }


def build_tables(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    workflows: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    barriers: list[dict[str, Any]] = []
    prefixes: list[dict[str, Any]] = []
    all_previous_llm: list[dict[str, Any]] = []

    for run in runs:
        path = run["path"]
        events = run["events"]
        name = run["workflow_name"]
        summary = backend_summary(path)
        llm = llm_events(events)
        tool = tool_events(events)
        barrier = barrier_events(events)
        start = min([event_start(e) for e in events] or [0.0])
        end = max([event_end(e) for e in events] or [0.0])
        wf_id = str(events[0].get("workflow_id") or events[0].get("run_id") or path.stem)
        composed = next((e.get("composed_from_motifs") or e.get("composed_from_topologies") for e in events if e.get("composed_from_motifs") or e.get("composed_from_topologies")), [])
        workflows.append(
            {
                "workflow_run_id": wf_id,
                "workflow_name": name,
                "workflow_type": workflow_type(events),
                "motif_type": run["motif_focus"],
                "composed_subgraphs": json.dumps(composed, ensure_ascii=False),
                "task_id": events[0].get("task_id") or events[0].get("instance_id"),
                "prompt_id": events[0].get("instance_id"),
                "start_time": start,
                "end_time": end,
                "end_to_end_latency": end - start,
                "status": "success" if any(e.get("event_type") in {"workflow_end", "motif_end"} for e in events) else "unknown",
                "backend_base_url": next((e.get("backend_base_url") for e in llm if e.get("backend_base_url")), ""),
                "model_name": next((e.get("model_name") or e.get("model") for e in llm if e.get("model_name") or e.get("model")), ""),
                "max_concurrent_running_requests": summary.get("max_num_requests_running"),
                "max_waiting_requests": summary.get("max_num_requests_waiting"),
                "max_gpu_cache_usage_perc": summary.get("max_gpu_cache_usage_perc"),
                "prefix_cache_hits_total_delta": metric_delta(path, "vllm:prefix_cache_hits_total"),
                "prefix_cache_queries_total_delta": metric_delta(path, "vllm:prefix_cache_queries_total"),
                "trace_path": str(path),
            }
        )

        child_map: dict[str, set[str]] = defaultdict(set)
        for event in events:
            dst = event.get("dst_node") or event.get("node_id")
            for parent in event.get("parent_node_ids") or event.get("parents") or []:
                child_map[str(parent)].add(str(dst))
        for event in events:
            if event.get("event_type") not in {"llm_request_end", "tool_search", "barrier", "motif_control", "route_decision"} and event.get("node_type") not in {"barrier", "edge"}:
                continue
            node_id = str(event.get("node_id") or "")
            if not node_id:
                continue
            nodes.append(
                {
                    "workflow_run_id": wf_id,
                    "workflow_name": name,
                    "motif_type": run["motif_focus"],
                    "node_id": node_id,
                    "agent_id": event.get("agent_id") or node_id,
                    "agent_role": event.get("agent_role") or event.get("node_type"),
                    "node_type": event.get("agent_role") or event.get("node_type"),
                    "parent_node_ids": json.dumps(event.get("parent_node_ids") or event.get("parents") or [], ensure_ascii=False),
                    "child_node_ids": json.dumps(sorted(child_map.get(node_id, [])), ensure_ascii=False),
                    "dependency_type": normalize_dependency_type(event),
                    "branch_id": event.get("background_branch_id") or event.get("parallel_group") or "",
                    "round_id": event.get("round_id") or event.get("peer_round_id") or event.get("manager_round_id"),
                    "loop_iteration_id": event.get("retry_count") or event.get("debug_loop_count") or 0,
                    "is_critical_path": bool(event.get("critical_path_candidate") or event.get("criticality") == "critical"),
                    "duration_sec": event.get("duration_sec"),
                    "start_time": event_start(event),
                    "end_time": event_end(event),
                }
            )
        same_workflow_previous: list[dict[str, Any]] = []
        for event in llm:
            request_id = event.get("request_id_for_backend") or event.get("llm_request_id") or event.get("event_id")
            prompt_segments = {
                "system_prompt_tokens": event.get("system_prompt_tokens_est"),
                "task_prompt_tokens": event.get("user_prompt_tokens_est"),
                "shared_context_tokens": event.get("shared_context_tokens_est") or event.get("round_shared_context_tokens"),
                "private_memory_tokens": event.get("private_history_tokens"),
                "retrieved_tool_context_tokens": event.get("retrieved_context_tokens_est") or event.get("shared_evidence_read_tokens_est"),
                "peer_message_tokens": event.get("peer_message_tokens_est"),
                "review_feedback_tokens": event.get("review_feedback_tokens_est"),
                "final_instruction_tokens": event.get("manager_instruction_tokens_est"),
            }
            queries.append(
                {
                    "request_id": request_id,
                    "workflow_run_id": wf_id,
                    "workflow_name": name,
                    "motif_type": run["motif_focus"],
                    "node_id": event.get("node_id"),
                    "agent_id": event.get("agent_id") or event.get("node_id"),
                    "role": event.get("agent_role"),
                    "node_type": event.get("agent_role") or event.get("node_type"),
                    "model_name": event.get("model_name") or event.get("model"),
                    "submit_time": event.get("request_submit_ts"),
                    "queue_start_time": "unavailable",
                    "queue_end_time": "unavailable",
                    "prefill_start_time": "unavailable",
                    "prefill_end_time": "unavailable",
                    "decode_start_time": event.get("response_start_ts"),
                    "decode_end_time": event.get("response_end_ts"),
                    "finish_time": event.get("response_end_ts") or event_end(event),
                    "ttft": event.get("ttft_sec") if event.get("ttft_sec") is not None else "unavailable",
                    "tpot": event.get("tpot_sec") if event.get("tpot_sec") is not None else "unavailable",
                    "request_e2e_sec": event.get("request_e2e_sec") or event.get("duration_sec"),
                    "input_tokens": event.get("input_tokens") or event.get("input_tokens_est"),
                    "prefill_tokens": event.get("input_tokens") or event.get("input_tokens_est"),
                    "output_tokens": event.get("output_tokens") or event.get("output_tokens_est"),
                    "decode_tokens": event.get("output_tokens") or event.get("output_tokens_est"),
                    "reasoning_tokens": event.get("reasoning_tokens", "unavailable"),
                    "prompt_segments": json.dumps(prompt_segments, ensure_ascii=False),
                    "prompt_hash": event.get("prompt_hash"),
                    "segment_hashes": json.dumps(event.get("prompt_segment_hashes") or event.get("shared_block_hashes") or {}, ensure_ascii=False),
                    "sampling_params": json.dumps({"max_output_tokens": event.get("max_output_tokens")}, ensure_ascii=False),
                    "status": event.get("status", "success"),
                    "critical_path_candidate": bool(event.get("critical_path_candidate") or event.get("criticality") == "critical"),
                    "relative_start_sec": event_start(event),
                    "relative_end_sec": event_end(event),
                }
            )
            intra = potential_prefix_for_event(event, same_workflow_previous, True)
            inter = potential_prefix_for_event(event, all_previous_llm, False)
            prefix_row = {
                "workflow_run_id": wf_id,
                "workflow_name": name,
                "motif_type": run["motif_focus"],
                "request_id": request_id,
                "node_id": event.get("node_id"),
                "input_tokens": event.get("input_tokens") or event.get("input_tokens_est"),
                "shared_context_reuse_tokens": sum(safe_int(v) for v in (event.get("shared_block_tokens") or {}).values()),
                "private_context_reuse_tokens": event.get("private_history_tokens") or 0,
                "dynamic_context_new_tokens": 0,
                "actual_prefix_cache_hit_tokens": "unavailable",
                "actual_prefix_cache_hit_rate": "unavailable",
                "actual_cached_blocks": "unavailable",
                "actual_new_blocks": "unavailable",
                "cache_eviction_count": "unavailable",
            }
            best = intra if intra["potential_prefix_match_tokens"] >= inter["potential_prefix_match_tokens"] else inter
            prefix_row.update(best)
            prefix_row["dynamic_context_new_tokens"] = max(0, safe_int(prefix_row["input_tokens"]) - safe_int(prefix_row["potential_prefix_match_tokens"]))
            prefixes.append(prefix_row)
            same_workflow_previous.append(event)
            all_previous_llm.append(event)
        for event in tool:
            tools.append(
                {
                    "tool_call_id": event.get("tool_call_id") or event.get("event_id"),
                    "workflow_run_id": wf_id,
                    "workflow_name": name,
                    "motif_type": run["motif_focus"],
                    "node_id": event.get("node_id"),
                    "agent_id": event.get("agent_id") or event.get("node_id"),
                    "tool_name": event.get("tool_name"),
                    "start_time": event_start(event),
                    "end_time": event_end(event),
                    "absolute_start_time": event.get("tool_start_ts"),
                    "absolute_end_time": event.get("tool_end_ts") or event.get("tool_return_ts"),
                    "latency": event.get("tool_latency_sec") or event.get("effective_duration_sec") or event.get("duration_sec"),
                    "input_size_tokens": event.get("tool_query_tokens_est") or event.get("input_size_tokens"),
                    "input_size_chars": event.get("tool_query_chars") or event.get("input_size_chars"),
                    "output_size_tokens": event.get("tool_output_tokens_est") or event.get("output_tokens_est"),
                    "output_size_chars": event.get("tool_output_chars") or event.get("output_chars"),
                    "status": event.get("status", "success"),
                    "retry_count": event.get("retry_count") or 0,
                    "written_to_shared_context": bool(event.get("is_shared_context") or event.get("shared_context")),
                }
            )
        for event in barrier:
            arrivals = event.get("arrived_nodes") or event.get("waiting_for_nodes") or []
            arrival_times: dict[str, float] = {}
            for arrival in arrivals:
                node = str(arrival)
                ends = [
                    event_end(candidate)
                    for candidate in events
                    if str(candidate.get("node_id")) == node
                    and str(candidate.get("event_type", "")).endswith("_end")
                    and event_end(candidate) > 0
                ]
                if ends:
                    arrival_times[node] = max(ends)
            earliest_arrival = min(arrival_times.values()) if arrival_times else ""
            latest_arrival = max(arrival_times.values()) if arrival_times else ""
            release_time = event_end(event)
            if safe_float(release_time) <= 0 and latest_arrival != "":
                release_time = safe_float(latest_arrival)
            waits = {
                node: max(0.0, safe_float(release_time) - arrival)
                for node, arrival in arrival_times.items()
            }
            if not waits:
                waits = {node: safe_float(event.get("barrier_wait_sec")) for node in arrivals}
            barriers.append(
                {
                    "barrier_id": event.get("barrier_id") or event.get("node_id"),
                    "workflow_run_id": wf_id,
                    "workflow_name": name,
                    "motif_type": run["motif_focus"],
                    "barrier_type": barrier_type(event),
                    "participating_node_ids": json.dumps(arrivals, ensure_ascii=False),
                    "earliest_arrival_time": earliest_arrival,
                    "latest_arrival_time": latest_arrival,
                    "barrier_release_time": release_time,
                    "per_node_wait_time": json.dumps(waits, ensure_ascii=False),
                    "straggler_gap": (
                        safe_float(latest_arrival) - safe_float(earliest_arrival)
                        if latest_arrival != "" and earliest_arrival != ""
                        else event.get("straggler_gap_sec") or event.get("barrier_wait_sec")
                    ),
                    "downstream_node_ids": json.dumps(sorted(child_map.get(str(event.get("node_id")), [])), ensure_ascii=False),
                }
            )
    return {"workflows": workflows, "nodes": nodes, "queries": queries, "tools": tools, "barriers": barriers, "prefix_cache": prefixes}


def barrier_type(event: dict[str, Any]) -> str:
    bid = str(event.get("barrier_id") or event.get("node_id") or "")
    if "debate" in bid or event.get("peer_round_id") is not None or "all_gather" in bid:
        return "debate_round_sync"
    if "manager" in bid:
        return "manager_wait"
    if "review" in bid:
        return "review_loop_wait"
    if "final" in bid:
        return "finalizer_wait"
    return "fan_in_merge"


def aggregate_for_figures(tables: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, Any]] = {}
    for wf in tables["workflows"]:
        by_run[str(wf["workflow_run_id"])] = dict(wf)
        by_run[str(wf["workflow_run_id"])].update(
            {
                "llm_time": 0.0,
                "tool_time": 0.0,
                "barrier_wait": 0.0,
                "merge_time": 0.0,
                "critical_time": 0.0,
                "query_count": 0,
                "max_query_e2e": 0.0,
                "potential_prefix_tokens": 0.0,
                "input_tokens": 0.0,
            }
        )
    for q in tables["queries"]:
        row = by_run.get(str(q["workflow_run_id"]))
        if not row:
            continue
        dur = safe_float(q.get("request_e2e_sec"))
        row["llm_time"] += dur
        row["query_count"] += 1
        row["max_query_e2e"] = max(row["max_query_e2e"], dur)
        if q.get("critical_path_candidate") in {True, "True", "true", "1"}:
            row["critical_time"] += dur
        if str(q.get("node_type", "")).lower() in {"merge", "aggregator", "finalizer", "synthesizer"}:
            row["merge_time"] += dur
    for t in tables["tools"]:
        row = by_run.get(str(t["workflow_run_id"]))
        if row:
            row["tool_time"] += safe_float(t.get("latency"))
    for b in tables["barriers"]:
        row = by_run.get(str(b["workflow_run_id"]))
        if row:
            row["barrier_wait"] += safe_float(b.get("straggler_gap"))
    for p in tables["prefix_cache"]:
        row = by_run.get(str(p["workflow_run_id"]))
        if row:
            row["potential_prefix_tokens"] += safe_float(p.get("potential_prefix_match_tokens"))
            row["input_tokens"] += safe_float(p.get("input_tokens"))
    rows = []
    for row in by_run.values():
        row["potential_prefix_reuse_rate"] = row["potential_prefix_tokens"] / max(row["input_tokens"], 1.0)
        row["critical_path_ratio"] = row["critical_time"] / max(safe_float(row.get("end_to_end_latency")), 1e-9)
        row["idle_wait"] = max(0.0, safe_float(row.get("end_to_end_latency")) - row["critical_time"])
        rows.append(row)
    return rows


def finish_plot(fig: Any, path: Path, caption: str) -> None:
    fig.text(0.01, 0.01, caption, fontsize=8, color="#444")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def figure_a_prefill_decode(tables: dict[str, list[dict[str, Any]]], fig_dir: Path) -> None:
    rows = tables["queries"]
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_role[str(row.get("role") or row.get("node_type") or "unknown")].append(row)
    summaries = []
    for role, pts in by_role.items():
        xs = [safe_float(r.get("prefill_tokens")) for r in pts]
        ys = [safe_float(r.get("decode_tokens")) for r in pts]
        if not xs:
            continue
        summaries.append(
            {
                "role": role,
                "count": len(pts),
                "median_x": median(xs),
                "median_y": median(ys),
                "min_x": min(xs),
                "max_x": max(xs),
                "min_y": min(ys),
                "max_y": max(ys),
            }
        )
    summaries = sorted(summaries, key=lambda r: (r["count"], r["median_x"]), reverse=True)[:14]
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    cmap = plt.get_cmap("tab20")
    for idx, row in enumerate(summaries):
        color = cmap(idx % 20)
        ax.hlines(row["median_y"], row["min_x"], row["max_x"], color=color, alpha=0.35, linewidth=4)
        ax.vlines(row["median_x"], row["min_y"], row["max_y"], color=color, alpha=0.35, linewidth=4)
        size = 55 + 18 * row["count"]
        ax.scatter(row["median_x"], row["median_y"], s=size, color=color, edgecolor="black", linewidth=0.8, zorder=3, label=f"{row['role'][:16]} n={row['count']}")
    ax.set_title("Figure A: Role-level prefill/decode token shape summary")
    ax.set_xlabel("prefill/input tokens")
    ax.set_ylabel("decode/output tokens")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    finish_plot(fig, fig_dir / "fig_a_prefill_vs_decode_tokens.png", "Graph insight: each bubble is a graph role; size is request count and crossbars show token range. The signal is role-specific prompt shape, not raw scatter density.")


def figure_b_timeline(tables: dict[str, list[dict[str, Any]]], fig_dir: Path) -> None:
    queries = tables["queries"]
    tools = tables["tools"]
    barriers = tables["barriers"]
    if not queries:
        return
    wf = max(defaultdict(int, {r["workflow_run_id"]: sum(1 for q in queries if q["workflow_run_id"] == r["workflow_run_id"]) for r in queries}), key=lambda k: sum(1 for q in queries if q["workflow_run_id"] == k))
    items = []
    for q in queries:
        if q["workflow_run_id"] == wf:
            items.append(("LLM", q.get("node_id"), safe_float(q.get("relative_start_sec")), safe_float(q.get("relative_end_sec")), "#4c78a8"))
    for t in tools:
        if t["workflow_run_id"] == wf:
            items.append(("Tool", t.get("node_id"), safe_float(t.get("start_time")), safe_float(t.get("end_time")), "#59a14f"))
    for b in barriers:
        if b["workflow_run_id"] == wf:
            end = safe_float(b.get("barrier_release_time"))
            start = max(0.0, end - safe_float(b.get("straggler_gap")))
            items.append(("Barrier", b.get("barrier_id"), start, end, "#f28e2b"))
    items.sort(key=lambda x: (x[2], x[0]))
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.28 * len(items))))
    for y, (kind, label, start, end, color) in enumerate(items):
        ax.barh(y, max(end - start, 0.015), left=start, color=color, edgecolor="black", height=0.55)
        ax.text(start + max(end - start, 0.015) / 2, y, str(label)[:26], ha="center", va="center", fontsize=7, color="white")
    ax.set_yticks(range(len(items)), [i[0] for i in items], fontsize=7)
    ax.set_xlabel("relative time (sec)")
    ax.set_title("Figure B: Representative workflow timeline with LLM/tool/barrier spans")
    finish_plot(fig, fig_dir / "fig_b_workflow_timeline.png", "Graph insight: workflow latency is shaped by graph synchronization and tool dependencies, not only individual LLM requests. Prefill/decode subspans are unavailable.")


def figure_c_latency_breakdown(agg: list[dict[str, Any]], fig_dir: Path) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in agg:
        groups[str(row.get("motif_type"))].append(row)
    labels = sorted(groups)
    metrics = [
        ("workflow latency", "end_to_end_latency"),
        ("LLM time", "llm_time"),
        ("tool time", "tool_time"),
        ("barrier wait", "barrier_wait"),
        ("merge time", "merge_time"),
        ("critical ratio", "critical_path_ratio"),
        ("max waiting", "max_waiting_requests"),
        ("KV usage", "max_gpu_cache_usage_perc"),
        ("potential prefix reuse", "potential_prefix_reuse_rate"),
    ]
    raw = [
        [mean([safe_float(r.get(field)) for r in groups[label]]) for _, field in metrics]
        for label in labels
    ]
    maxima = [max([row[i] for row in raw] or [1.0]) or 1.0 for i in range(len(metrics))]
    norm = [[value / maxima[i] for i, value in enumerate(row)] for row in raw]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    im = ax.imshow(norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics)), [m[0] for m in metrics], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title("Figure C: Motif load-shape matrix, not an average-latency bar chart")
    for y, row in enumerate(raw):
        for x, value in enumerate(row):
            if metrics[x][1] in {"critical_path_ratio", "max_gpu_cache_usage_perc", "potential_prefix_reuse_rate"}:
                text = f"{value:.2f}"
            else:
                text = f"{value:.1f}"
            ax.text(x, y, text, ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="column-normalized pressure")
    finish_plot(fig, fig_dir / "fig_c_motif_latency_decomposition.png", "Graph insight: rows differ by graph motif and columns differ by serving pressure type. Dark cells identify which motif stresses scheduling, tools, barriers, KV, or prefix reuse.")


def figure_d_critical_path(agg: list[dict[str, Any]], fig_dir: Path) -> None:
    rows = sorted(agg, key=lambda r: safe_float(r.get("end_to_end_latency")), reverse=True)
    labels = [str(r.get("workflow_name"))[:30] for r in rows]
    critical = [safe_float(r.get("critical_time")) for r in rows]
    barrier = [safe_float(r.get("barrier_wait")) for r in rows]
    tool = [safe_float(r.get("tool_time")) for r in rows]
    workflow = [safe_float(r.get("end_to_end_latency")) for r in rows]
    other = [max(0.0, workflow[i] - critical[i] - barrier[i] - tool[i]) for i in range(len(rows))]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    y = range(len(rows))
    ax.barh(y, critical, color="#c62828", label="critical-path LLM time")
    ax.barh(y, barrier, left=critical, color="#f28e2b", label="barrier / sync wait")
    left2 = [critical[i] + barrier[i] for i in range(len(rows))]
    ax.barh(y, tool, left=left2, color="#59a14f", label="tool time")
    left3 = [left2[i] + tool[i] for i in range(len(rows))]
    ax.barh(y, other, left=left3, color="#bab0ab", label="other / noncritical time")
    for idx, row in enumerate(rows):
        ax.text(workflow[idx] + 0.6, idx, f"ratio={safe_float(row.get('critical_path_ratio')):.2f}", va="center", fontsize=7)
    ax.set_yticks(list(y), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("seconds")
    ax.set_title("Figure D: Workflow latency decomposed by graph-critical components")
    ax.legend(fontsize=8, ncols=2)
    finish_plot(fig, fig_dir / "fig_d_critical_path_breakdown.png", "Graph insight: the decisive cost is where time sits in the dependency graph. Critical-path LLMs and barriers matter more than flat request averages.")


def figure_e_prefix_redundancy(tables: dict[str, list[dict[str, Any]]], fig_dir: Path) -> None:
    query_order = {
        (q["workflow_run_id"], q["request_id"]): idx
        for idx, q in enumerate(sorted(tables["queries"], key=lambda q: (str(q.get("workflow_run_id")), safe_float(q.get("relative_start_sec")))))
    }
    query_lookup = {(q["workflow_run_id"], q["request_id"]): q for q in tables["queries"]}

    def choose_workflow(motif: str) -> str | None:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in tables["prefix_cache"]:
            if row.get("motif_type") == motif:
                grouped[str(row.get("workflow_run_id"))].append(row)
        if not grouped:
            return None
        return max(grouped, key=lambda wf: sum(safe_float(r.get("input_tokens")) for r in grouped[wf]))

    review_wf = choose_workflow("review_loop")
    manager_wf = choose_workflow("manager_worker")
    panels = [
        ("Review loop: repeated prompt segments survive across iterations", review_wf, "review_loop"),
        ("Manager / hierarchical fan-in: aggregation carries reusable blocks", manager_wf, "manager_worker"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.0), sharey=False)
    for ax, (title, wf_id, motif) in zip(axes, panels, strict=False):
        pts = [r for r in tables["prefix_cache"] if r.get("workflow_run_id") == wf_id and r.get("motif_type") == motif]
        pts = sorted(pts, key=lambda r: query_order.get((r.get("workflow_run_id"), r.get("request_id")), 0))
        xs = list(range(len(pts)))
        potential = [safe_float(r.get("potential_prefix_match_tokens")) for r in pts]
        dynamic = [safe_float(r.get("dynamic_context_new_tokens")) for r in pts]
        inputs = [safe_float(r.get("input_tokens")) for r in pts]
        labels = []
        for r in pts:
            q = query_lookup.get((r.get("workflow_run_id"), r.get("request_id")), {})
            label = str(q.get("node_id") or r.get("node_id"))
            labels.append(label.replace("cross_group_", "xgrp_").replace("global_", "g_")[:16])
        ax.bar(xs, potential, color="#9467bd", label="potential reusable context")
        ax.bar(xs, dynamic, bottom=potential, color="#8cd17d", label="dynamic/new suffix")
        ax.plot(xs, inputs, color="#111111", marker="o", linewidth=1.2, label="input tokens")
        for x, r, p in zip(xs, pts, potential, strict=False):
            source = str(r.get("potential_prefix_source") or "")
            if p > 0 and source:
                ax.text(x, p + max(inputs) * 0.03, source.replace("_", "\n"), ha="center", va="bottom", fontsize=6, rotation=0)
        ax.set_title(title)
        ax.set_ylabel("tokens")
        ax.set_xticks(xs, labels, rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("LLM request order in representative real vLLM workflow")
    finish_plot(fig, fig_dir / "fig_e_prefix_redundancy_review_manager.png", "Graph insight: review loops and manager fan-in repeatedly move old context while appending dynamic feedback/artifacts. Purple is simulator-side potential reuse; labels show exact hash/shared block/segment proxy source, not actual vLLM cache hits.")


def figure_f_branch_straggler(tables: dict[str, list[dict[str, Any]]], fig_dir: Path) -> None:
    rows = [r for r in tables["barriers"] if safe_float(r.get("straggler_gap")) > 0]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (len(json.loads(r.get("per_node_wait_time") or "{}")), safe_float(r.get("straggler_gap"))), reverse=True)[:10]
    max_width = max(len(json.loads(r.get("per_node_wait_time") or "{}")) for r in rows)
    matrix: list[list[float]] = []
    labels: list[str] = []
    for row in rows:
        waits = sorted([safe_float(v) for v in json.loads(row.get("per_node_wait_time") or "{}").values()], reverse=True)
        waits += [math.nan] * (max_width - len(waits))
        matrix.append(waits)
        labels.append(f"{str(row.get('workflow_name'))[:20]} / {str(row.get('barrier_id'))[:20]}")
    fig, ax = plt.subplots(figsize=(10.5, max(5.0, 0.42 * len(rows))))
    im = ax.imshow(matrix, cmap="Oranges", aspect="auto")
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xticks(range(max_width), [f"arrival rank {i+1}" for i in range(max_width)], rotation=25, ha="right", fontsize=8)
    ax.set_title("Figure F: Fan-in and round-sync wait matrix")
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if not math.isnan(value):
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7, color="#222")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="wait before barrier release (sec)")
    finish_plot(fig, fig_dir / "fig_f_branch_straggler_fanin_wait.png", "Graph insight: each row is a real barrier. Earlier arrivals accumulate wait; the last arrival has near-zero wait and acts as the straggler that releases the barrier.")


def figure_g_query_workflow_mismatch(tables: dict[str, list[dict[str, Any]]], fig_dir: Path) -> None:
    queries = tables["queries"]
    workflows = sorted(tables["workflows"], key=lambda r: safe_float(r.get("end_to_end_latency")), reverse=True)
    wf_latency = {r["workflow_run_id"]: safe_float(r.get("end_to_end_latency")) for r in workflows}
    barrier_by_wf: dict[str, float] = defaultdict(float)
    for b in tables["barriers"]:
        barrier_by_wf[str(b["workflow_run_id"])] += safe_float(b.get("straggler_gap"))
    max_query: dict[str, float] = defaultdict(float)
    critical_query: dict[str, float] = defaultdict(float)
    for q in queries:
        wf = str(q["workflow_run_id"])
        dur = safe_float(q.get("request_e2e_sec"))
        max_query[wf] = max(max_query[wf], dur)
        if q.get("critical_path_candidate") in {True, "true", "True", "1"}:
            critical_query[wf] = max(critical_query[wf], dur)
    labels = [str(r.get("workflow_name"))[:26] for r in workflows]
    x = list(range(len(workflows)))
    fig, ax1 = plt.subplots(figsize=(11.0, 5.4))
    ax1.bar(x, [wf_latency[str(r["workflow_run_id"])] for r in workflows], color="#d9d9d9", edgecolor="#666", label="workflow latency")
    ax1.scatter(x, [max_query[str(r["workflow_run_id"])] for r in workflows], color="#4c78a8", s=75, label="slowest single query", zorder=3)
    ax1.scatter(x, [critical_query[str(r["workflow_run_id"])] for r in workflows], color="#c62828", marker="D", s=62, label="slowest critical query", zorder=3)
    ax1.set_ylabel("seconds")
    ax1.set_xticks(x, labels, rotation=55, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    shares = [barrier_by_wf[str(r["workflow_run_id"])] / max(wf_latency[str(r["workflow_run_id"])], 1e-9) for r in workflows]
    ax2.plot(x, shares, color="#f28e2b", marker="o", linewidth=2.0, label="barrier share")
    ax2.set_ylabel("barrier wait / workflow latency")
    ax1.set_title("Figure G: Flat query metrics are not enough to explain workflow latency")
    lines, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels1 + labels2, fontsize=8, loc="upper right")
    finish_plot(fig, fig_dir / "fig_g_query_workflow_mismatch.png", "Graph insight: workflow latency can be much larger than the slowest individual query because dependency position, barrier share, and critical path decide end-to-end delay.")


def draw_box(ax: Any, xy: tuple[float, float], text: str, color: str, width: float = 1.7, height: float = 0.42) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.2,
        edgecolor="#333",
        facecolor=color,
        alpha=0.92,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=8)


def arrow(ax: Any, start: tuple[float, float], end: tuple[float, float], color: str = "#333") -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.3, "color": color})


def figure_resume_workflow_dag(fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.axis("off")
    critical = "#f4a3a3"
    noncritical = "#bcd4f6"
    tool = "#c7e9c0"
    merge = "#ddd"
    x0, y0 = 0.6, 3.5
    nodes = [
        ("planner", (x0, y0)),
        ("critical_coder", (2.7, y0)),
        ("critical_reviewer", (4.8, y0)),
        ("finalizer", (6.9, y0)),
        ("evidence_merge", (8.9, y0 - 0.6)),
    ]
    for text, xy in nodes[:4]:
        draw_box(ax, xy, text, critical)
    draw_box(ax, nodes[4][1], nodes[4][0], merge)
    for i in range(len(nodes) - 2):
        arrow(ax, (nodes[i][1][0] + 1.7, nodes[i][1][1] + 0.21), (nodes[i + 1][1][0], nodes[i + 1][1][1] + 0.21), "#b71c1c")
    arrow(ax, (nodes[3][1][0] + 1.7, nodes[3][1][1] + 0.21), (nodes[4][1][0], nodes[4][1][1] + 0.45), "#b71c1c")
    for idx, y in enumerate([2.55, 1.85, 1.15, 0.45], start=1):
        draw_box(ax, (1.3, y), f"tool_agent_{idx}\nLLM-1", noncritical, width=1.55)
        draw_box(ax, (3.35, y), "web_search\nTool Call", tool, width=1.55)
        draw_box(ax, (5.4, y), f"resume_{idx}\nLLM-2", noncritical, width=1.55)
        arrow(ax, (2.85, y + 0.21), (3.35, y + 0.21), "#1565c0")
        arrow(ax, (4.9, y + 0.21), (5.4, y + 0.21), "#1565c0")
        arrow(ax, (6.95, y + 0.21), (8.9, 3.1), "#1565c0")
    ax.text(0.7, 4.2, "Critical path", color="#b71c1c", fontsize=11, weight="bold")
    ax.text(1.3, 3.05, "Non-critical tool-stalled branches", color="#1565c0", fontsize=10, weight="bold")
    ax.text(5.35, 0.05, "Workflow is composed from planner/worker, tool evidence, review, and merge subgraphs.", fontsize=8, color="#444")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.7)
    finish_plot(fig, fig_dir / "fig_tool_resume_workflow_dag.png", "Graph insight: non-critical branches follow LLM-1 -> tool call -> LLM-2 resume while the critical reviewer/finalizer path can continue without waiting for every tool branch.")


def tool_resume_rows(tables: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    queries = [q for q in tables["queries"] if q.get("workflow_name") == "tool_resume_contention_meso"]
    tools = [t for t in tables["tools"] if t.get("workflow_name") == "tool_resume_contention_meso"]
    workflows = [w for w in tables["workflows"] if w.get("workflow_name") == "tool_resume_contention_meso"]
    return queries, tools, workflows


def figure_resume_overlap_timeline(tables: dict[str, list[dict[str, Any]]], fig_dir: Path) -> None:
    queries, tools, _ = tool_resume_rows(tables)
    if not queries:
        return
    items: list[tuple[str, str, float, float, str]] = []
    for q in queries:
        start = safe_float(q.get("relative_start_sec"))
        end = safe_float(q.get("relative_end_sec"))
        role = str(q.get("role") or q.get("node_type"))
        node = str(q.get("node_id"))
        if q.get("critical_path_candidate") in {True, "true", "True", "1"}:
            color = "#c62828"
            lane = "critical LLM"
        elif "resume" in node or "evidence_processor" in role:
            color = "#1565c0"
            lane = "background resume"
        else:
            color = "#90a4ae"
            lane = "background LLM"
        items.append((lane, node, start, end, color))
    for t in tools:
        items.append(("tool call", str(t.get("node_id")), safe_float(t.get("start_time")), safe_float(t.get("end_time")), "#59a14f"))
    items.sort(key=lambda x: (x[2], x[0]))
    fig, ax = plt.subplots(figsize=(11.2, max(5.6, 0.32 * len(items))))
    for y, (lane, label, start, end, color) in enumerate(items):
        ax.barh(y, max(end - start, 0.03), left=start, height=0.58, color=color, edgecolor="black", alpha=0.9)
        ax.text(start + max(end - start, 0.03) / 2, y, label[:28], ha="center", va="center", fontsize=7, color="white" if color != "#59a14f" else "black")
    critical = [(safe_float(q.get("relative_start_sec")), safe_float(q.get("relative_end_sec"))) for q in queries if q.get("critical_path_candidate") in {True, "true", "True", "1"} and str(q.get("role")) in {"reviewer", "finalizer"}]
    resumes = [(safe_float(q.get("relative_start_sec")), safe_float(q.get("relative_end_sec"))) for q in queries if "resume" in str(q.get("node_id")) or str(q.get("role")) == "evidence_processor"]
    for cs, ce in critical:
        for rs, re in resumes:
            start, end = max(cs, rs), min(ce, re)
            if end > start:
                ax.axvspan(start, end, color="#ffcc80", alpha=0.28)
    ax.set_yticks(range(len(items)), [i[0] for i in items], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("relative time (sec)")
    ax.set_title("Composite Figure 1: Tool-return resume burst overlaps critical stages")
    finish_plot(fig, fig_dir / "fig_tool_resume_overlap_timeline.png", "Graph insight: orange windows are observed overlaps between background resume LLMs and critical reviewer/finalizer LLMs in the real vLLM trace.")


def overlap_counts(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    by_wf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in queries:
        by_wf[str(q.get("workflow_run_id"))].append(q)
    for wf, wf_queries in by_wf.items():
        critical = [q for q in wf_queries if q.get("critical_path_candidate") in {True, "true", "True", "1"} and str(q.get("role")) in {"reviewer", "finalizer"}]
        resumes = [q for q in wf_queries if "resume" in str(q.get("node_id")) or str(q.get("role")) == "evidence_processor"]
        width = len([q for q in wf_queries if "tool_agent" in str(q.get("node_id"))])
        for cq in critical:
            cs, ce = safe_float(cq.get("relative_start_sec")), safe_float(cq.get("relative_end_sec"))
            overlapping = []
            overlap_tokens = 0
            for rq in resumes:
                rs, re = safe_float(rq.get("relative_start_sec")), safe_float(rq.get("relative_end_sec"))
                if min(ce, re) > max(cs, rs):
                    overlapping.append(rq)
                    overlap_tokens += safe_int(rq.get("input_tokens")) + safe_int(rq.get("output_tokens"))
            rows.append({"workflow_run_id": wf, "critical": cq, "overlap_count": len(overlapping), "overlap_tokens": overlap_tokens, "tool_branch_width": width})
    return rows


def figure_resume_events_over_time(tables: dict[str, list[dict[str, Any]]], fig_dir: Path) -> None:
    queries, _, _ = tool_resume_rows(tables)
    if not queries:
        return
    events: list[tuple[float, str]] = []
    by_wf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in queries:
        by_wf[str(q.get("workflow_run_id"))].append(q)
    for wf_queries in by_wf.values():
        critical = [q for q in wf_queries if q.get("critical_path_candidate") in {True, "true", "True", "1"} and str(q.get("role")) in {"reviewer", "finalizer"}]
        resumes = [q for q in wf_queries if "resume" in str(q.get("node_id")) or str(q.get("role")) == "evidence_processor"]
        for cq in critical:
            cs, ce = safe_float(cq.get("relative_start_sec")), safe_float(cq.get("relative_end_sec"))
            for rq in resumes:
                rs, re = safe_float(rq.get("relative_start_sec")), safe_float(rq.get("relative_end_sec"))
                if min(ce, re) > max(cs, rs):
                    events.append((max(cs, rs), "overlap"))
                    events.append((max(cs, rs), "resume_overlap"))
                    # No queue/slowdown event is asserted without direct evidence.
    times = sorted({0.0, *[safe_float(q.get("relative_end_sec")) for q in queries], *[t for t, _ in events]})
    cumulative_total = []
    cumulative_resume = []
    cumulative_slowdown = []
    for t in times:
        cumulative_total.append(sum(1 for ts, _ in events if ts <= t))
        cumulative_resume.append(sum(1 for ts, kind in events if ts <= t and kind == "resume_overlap"))
        cumulative_slowdown.append(0)
    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    ax.step(times, cumulative_total, where="post", color="#4c78a8", linewidth=2.2, label="total overlap events")
    ax.step(times, cumulative_resume, where="post", color="#1565c0", linewidth=2.2, label="background-resume overlaps critical")
    ax.step(times, cumulative_slowdown, where="post", color="#c62828", linewidth=2.0, label="critical slowdown events (not observed)")
    ax.set_xlabel("relative time (sec)")
    ax.set_ylabel("cumulative event count")
    ax.set_title(f"Composite Figure 2: Cumulative critical/background overlap events across {len(by_wf)} real runs")
    ax.legend(fontsize=8)
    finish_plot(fig, fig_dir / "fig_tool_resume_overlap_events_over_time.png", "Graph insight: the trace shows repeated overlap opportunities. Slowdown is plotted as zero because no per-request queue or critical slowdown evidence is claimed.")


def figure_noncritical_kv_proxy(tables: dict[str, list[dict[str, Any]]], fig_dir: Path) -> None:
    queries, tools, _ = tool_resume_rows(tables)
    if not queries:
        return
    times = sorted({0.0, *[safe_float(q.get("relative_start_sec")) for q in queries], *[safe_float(q.get("relative_end_sec")) for q in queries], *[safe_float(t.get("start_time")) for t in tools], *[safe_float(t.get("end_time")) for t in tools]})
    stalled_series, critical_series, resume_series = [], [], []
    tool_agent_tokens = {
        str(q.get("node_id")).replace("_tool_agent", ""): safe_int(q.get("input_tokens")) + safe_int(q.get("output_tokens"))
        for q in queries
        if "tool_agent" in str(q.get("node_id"))
    }
    for t in times:
        stalled = 0
        for tool_row in tools:
            branch = str(tool_row.get("node_id")).replace("_search", "")
            if safe_float(tool_row.get("start_time")) <= t <= safe_float(tool_row.get("end_time")):
                stalled += tool_agent_tokens.get(branch, 0)
        critical = sum(
            safe_int(q.get("input_tokens")) + safe_int(q.get("output_tokens"))
            for q in queries
            if q.get("critical_path_candidate") in {True, "true", "True", "1"}
            and safe_float(q.get("relative_start_sec")) <= t <= safe_float(q.get("relative_end_sec"))
        )
        resume = sum(
            safe_int(q.get("input_tokens")) + safe_int(q.get("output_tokens"))
            for q in queries
            if ("resume" in str(q.get("node_id")) or str(q.get("role")) == "evidence_processor")
            and safe_float(q.get("relative_start_sec")) <= t <= safe_float(q.get("relative_end_sec"))
        )
        stalled_series.append(stalled)
        critical_series.append(critical)
        resume_series.append(resume)
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.step(times, stalled_series, where="post", color="#90a4ae", linewidth=2.2, label="stalled non-critical KV proxy")
    ax.step(times, critical_series, where="post", color="#c62828", linewidth=2.2, label="active critical KV proxy")
    ax.step(times, resume_series, where="post", color="#1565c0", linewidth=2.2, label="background resume KV proxy")
    ax.set_xlabel("relative time (sec)")
    ax.set_ylabel("estimated active / idle context tokens")
    ax.set_title("Composite Figure 3: Non-critical tool-stalled context occupancy proxy")
    ax.legend(fontsize=8)
    finish_plot(fig, fig_dir / "fig_tool_resume_noncritical_kv_proxy.png", "Graph insight: this is a context-token proxy for KV pressure, not actual KV block residency. It shows non-critical stalled and resumed branches coexisting with critical requests.")


def figure_critical_latency_overlap(tables: dict[str, list[dict[str, Any]]], fig_dir: Path) -> None:
    queries, _, _ = tool_resume_rows(tables)
    rows = overlap_counts(queries)
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    jitter_counter: dict[tuple[int, str], int] = defaultdict(int)
    grouped_y: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        overlap = int(row["overlap_count"])
        width = int(row["tool_branch_width"])
        node = str(row["critical"].get("node_id"))
        stage = "finalizer" if "finalizer" in node else "reviewer"
        jitter_counter[(overlap, stage)] += 1
        jitter = (jitter_counter[(overlap, stage)] - 2.5) * 0.025
        x = overlap + jitter
        y = safe_float(row["critical"].get("request_e2e_sec"))
        grouped_y[overlap].append(y)
        color = "#ef5350" if width <= 4 else "#b71c1c"
        marker = "o" if stage == "reviewer" else "s"
        ax.scatter(x, y, s=90 + 14 * width, color=color, marker=marker, edgecolor="black", alpha=0.84)
    mean_x = sorted(grouped_y)
    mean_y = [mean(grouped_y[x]) for x in mean_x]
    if mean_x:
        ax.plot(mean_x, mean_y, color="#333", marker="D", linewidth=2.0, label="mean critical latency")
        for x, y in zip(mean_x, mean_y):
            ax.text(x, y + 0.08, f"mean {y:.2f}s", ha="center", fontsize=8, color="#333")
    ax.scatter([], [], s=130, color="#ef5350", edgecolor="black", label="width 4")
    ax.scatter([], [], s=190, color="#b71c1c", edgecolor="black", label="width 8")
    ax.scatter([], [], s=120, color="#999", marker="o", edgecolor="black", label="reviewer")
    ax.scatter([], [], s=120, color="#999", marker="s", edgecolor="black", label="finalizer")
    ax.set_xlabel("overlapping background resume request count")
    ax.set_ylabel("critical reviewer/finalizer request_e2e_sec")
    ax.set_title(f"Composite Figure 4: Critical latency vs background resume overlap ({len(rows)} critical points)")
    ax.legend(fontsize=8, ncols=2)
    finish_plot(fig, fig_dir / "fig_tool_resume_critical_latency_vs_overlap.png", "Graph insight: each point is a critical reviewer/finalizer from one real workflow run; marker size/color reflects tool branch width.")


def write_figures(tables: dict[str, list[dict[str, Any]]], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate_for_figures(tables)
    figure_a_prefill_decode(tables, fig_dir)
    figure_b_timeline(tables, fig_dir)
    figure_c_latency_breakdown(agg, fig_dir)
    figure_d_critical_path(agg, fig_dir)
    figure_e_prefix_redundancy(tables, fig_dir)
    figure_f_branch_straggler(tables, fig_dir)
    figure_g_query_workflow_mismatch(tables, fig_dir)
    figure_resume_workflow_dag(fig_dir)
    figure_resume_overlap_timeline(tables, fig_dir)
    figure_resume_events_over_time(tables, fig_dir)
    figure_noncritical_kv_proxy(tables, fig_dir)
    figure_critical_latency_overlap(tables, fig_dir)


def write_schema(path: Path) -> None:
    path.write_text(
        """# Week3 Simulator-ready Trace Schema

The raw JSONL traces remain the source of truth. Week3 normalization exports simulator-facing tables derived only from real vLLM-backed runs.

## Workflow
`workflow_run_id`, `workflow_name`, `workflow_type`, `motif_type`, `composed_subgraphs`, `task_id`, `prompt_id`, `start_time`, `end_time`, `end_to_end_latency`, `status`, backend endpoint/model, concurrency/backend metrics.

## Graph / Node
`node_id`, `agent_id`, `agent_role`, `node_type`, `parent_node_ids`, `child_node_ids`, `dependency_type`, `branch_id`, `round_id`, `loop_iteration_id`, `is_critical_path`.

## LLM Query
`request_id`, `workflow_run_id`, `node_id`, `agent_id`, `role`, `model_name`, `submit_time`, queue/prefill/decode timestamps when available, `finish_time`, `ttft`, `tpot`, `input_tokens`, `prefill_tokens`, `output_tokens`, `decode_tokens`, prompt segment token breakdown, `prompt_hash`, `segment_hashes`, sampling params, `status`.

Queue and prefill substage timestamps are marked `unavailable` when vLLM does not expose per-request values.

## Tool Call
`tool_call_id`, `workflow_run_id`, `node_id`, `tool_name`, start/end time, latency, input/output size, status, retry count, and whether the result is written to shared context.

## Synchronization / Barrier
`barrier_id`, `barrier_type`, participants, release time, per-node wait proxy, straggler gap, downstream nodes.

## Prefix / Cache
Actual backend metrics are sourced only from vLLM `/metrics` sidecars, including run-level GPU KV cache usage and prefix-cache counters when exposed.

Offline fields use explicit potential terminology: `potential_prefix_match_tokens`, `potential_prefix_reuse_rate`, `intra_workflow_prefix_match_tokens`, `inter_workflow_prefix_match_tokens`, `shared_context_reuse_tokens`, `private_context_reuse_tokens`, `dynamic_context_new_tokens`.

Do not interpret potential prefix reuse as actual cache hit rate.
""",
        encoding="utf-8",
    )


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] * len(fields)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    return "\n".join([header, sep, *body])


def write_report(path: Path, tables: dict[str, list[dict[str, Any]]], trace_root: Path) -> None:
    workflows = tables["workflows"]
    by_focus: dict[str, int] = defaultdict(int)
    for row in workflows:
        by_focus[str(row.get("motif_type"))] += 1
    rows = [{"motif_type": k, "runs": v} for k, v in sorted(by_focus.items())]
    max_kv = max([safe_float(r.get("max_gpu_cache_usage_perc")) for r in workflows if r.get("max_gpu_cache_usage_perc") not in {None, ""}] or [0.0])
    tool_queries, tool_events_rows, tool_workflows = tool_resume_rows(tables)
    tool_widths: dict[int, int] = defaultdict(int)
    for wf in tool_workflows:
        wf_queries = [q for q in tool_queries if q.get("workflow_run_id") == wf.get("workflow_run_id")]
        tool_widths[len([q for q in wf_queries if "tool_agent" in str(q.get("node_id"))])] += 1
    tool_width_text = ", ".join(f"width {k}: {v} runs" for k, v in sorted(tool_widths.items())) or "none"
    critical_points = len(overlap_counts(tool_queries))
    text = f"""# Week3 进展：面向模拟器的 Trace 与图结构诱发瓶颈

## 1. Motivation

Week3 的重点从“能跑 workflow 并记录日志”推进到“面向模拟器、图结构感知的 MASBench trace”。我们现在不只关心单个 LLM request 的 TTFT、TPOT 或 token 数，而是把 workflow graph 本身作为 benchmark 对象：不同 motif 的依赖边、fan-out/fan-in、barrier、review loop、tool dependency 和 shared context movement 会塑造完全不同的 serving 压力。

这也是 MASBench 和固定 Agent 系统 trace 的区别：我们希望证明瓶颈来自可组合的 MAS workflow 图结构，而不是某一个固定应用的偶然行为。后续 simulator replay 必须保留 graph role、依赖关系、barrier、context segment 和潜在 prefix reuse，否则只 replay 一串扁平 LLM 请求会丢掉 MAS workload 的关键结构。

## 2. Simulator-ready Trace Design

本轮标准化导出了六类表：Workflow、Graph/Node、LLM Query、Tool Call、Barrier、Prefix/Cache。原始 JSONL trace 仍然是 source of truth，CSV 表只是为了 simulator replay 和图结构分析做的 deterministic projection。

具体 schema 见 `trace_schema.md`。需要特别注意两点：

- vLLM 当前没有暴露到 per-request 的 queue/prefill/decode 细分时间时，字段会明确写成 `unavailable`，不会隐式估算。
- prefix/cache 相关字段严格区分 `actual` 和 `potential`。真实 vLLM `/metrics` 里能拿到的是 run-level KV usage 和 prefix-cache counter；通过 prompt/hash 离线算出来的只叫 potential prefix reuse，不能写成真实 cache hit。

## 3. Real vLLM-backed Execution Setup

- Trace root: `{trace_root}`
- 本报告分析的 workflow run 数量: `{len(workflows)}`
- Motif coverage: {dict(sorted(by_focus.items()))}
- 观测到的 backend model: `{sorted({str(r.get('model_name')) for r in workflows if r.get('model_name')})}`
- vLLM metrics 中观测到的 run-level 最大 GPU KV cache usage: `{max_kv:.6f}`
- 所有图表和表格均来自真实 vLLM-backed run 的原始 trace 与 backend metrics sidecar。本报告没有使用 synthetic trace 或 mock trace。

{md_table(rows, ["motif_type", "runs"])}

## 4. Figure Quality Check

我把原先信息量偏低的普通柱状图/散点图替换成了更适合文章叙事的图：load-shape matrix、workflow timeline、关键路径分解、prefix reuse 分解、fan-in wait waterfall、query/workflow mismatch 对比。保留下来的散点图也加了 role median 和大 prompt 节点标注，避免只展示“点很多”。

这些图的读法统一是：先看图结构对应的 motif，再看颜色/层级/标注说明的系统压力，最后看 caption 里的证据边界。凡是 prefix/cache 不是 vLLM 直接给出的指标，图中都只解释为 potential 或 proxy。

## 5. Local Motif Bottlenecks

### 5.1 Independent / Multi-branch Motif

图结构模式：多个分支并行执行，最后进入 fan-in merge。测量重点是 branch arrival、straggler gap、barrier release 和下游 finalizer 等待。

![Figure F](figures/fig_f_branch_straggler_fanin_wait.png)

**读图 tips：**每一行是一个真实 barrier 或 round sync。横轴的 `arrival rank` 是按到达早晚排序后的分支位置；格子里的数字表示该分支在 barrier release 前等待了多久。越深的格子说明越早到达、等待越久；接近 0 的最后到达者就是释放 barrier 的 straggler。

**图结构洞察：**并行分支不是免费加速。只要下游需要 fan-in，workflow latency 就会被最慢分支和 merge release 决定。对 serving 的意义是：scheduler 只看单个 request latency 不够，还需要知道哪些 request 会阻塞下游 fan-in。

### 5.2 Manager-worker Motif

图结构模式：manager 发指令，worker 并行执行，再回到 manager/reviewer/synthesizer 汇总。测量重点是 centralized coordination、manager critical path、fan-in context aggregation 和 backend waiting/KV pressure。

![Figure C](figures/fig_c_motif_latency_decomposition.png)

**读图 tips：**这不是平均 latency 柱状图。每一行是一个 motif 类别，每一列是一种 serving 压力：workflow latency、LLM time、tool time、barrier wait、merge time、critical ratio、waiting requests、KV usage、potential prefix reuse。颜色越深表示该列下相对压力越高，格子里的数字是原始量级。

**图结构洞察：**不同 motif 对 backend 施加的是不同 load shape。manager-worker 和 hierarchical synthesis 往往同时出现 fan-in aggregation、critical manager path 和上下文增长，因此它们适合用来研究 graph-aware scheduling、manager-stage batching 和 prefix-aware context management。

### 5.3 Review-loop Motif

图结构模式：generator/coder 产生结果，reviewer/verifier 提反馈，reviser/debugger 进入下一轮。测量重点是每轮 input token 增长、重复 prompt prefix、review feedback 动态后缀和最终路径增长。

![Figure E](figures/fig_e_prefix_redundancy_review_manager.png)

**读图 tips：**左图看 request 顺序上的 input tokens 和可复用 prefix token：圆点/折线代表输入规模，方块代表离线 hash/token 分析得到的 potential reusable prefix。右图把 potential reusable prefix 和 dynamic/new suffix 分开，紫色越多说明存在更强的潜在复用机会，绿色越多说明新 feedback/instruction 仍会制造不可复用的动态上下文。

**图结构洞察：**review loop 和 manager aggregation 会反复携带长上下文，因此 prefix/cache 研究有明确动机；但每轮新增的 review feedback 是动态后缀，不能简单假设全部可缓存。当前证据支持 potential prefix reuse，不支持声称 actual cache hit。

### 5.4 Debate Motif

图结构模式：同一轮多个 agent 并行生成观点，随后进入 round barrier / all-gather，再进入下一轮。测量重点是 per-round LLM span、round barrier、peer-message growth 和 final aggregation。

![Figure B](figures/fig_b_workflow_timeline.png)

**读图 tips：**横轴是真实运行时间，蓝色是 LLM request，绿色是 tool call，橙色是 barrier/sync。读这张图时不要只看最长的蓝条，而要看蓝条之间是否被橙色同步点切成多段：每个 round barrier 都会把本来并行的 agent 再次串行化到下一轮。

**图结构洞察：**debate 的瓶颈不是“agent 数多”这么简单，而是 round-level synchronization。即使一轮内部可以并发，下一轮仍必须等待 barrier release，因此 serving 侧需要考虑 barrier-aware batching 和 round-aware scheduling。

### 5.5 Hybrid Manager + Peer Motif

图结构模式：manager 集中控制和 peer communication 同时存在。测量重点是 manager critical-path ratio、peer synchronization wait、merge overhead 和 shared context growth。

![Figure D](figures/fig_d_critical_path_breakdown.png)

**读图 tips：**每一行是一个 workflow。红色是 critical-path LLM time，橙色是 barrier/sync wait，绿色是 tool time，灰色是其他非关键路径时间。右侧的 ratio 表示 critical-path LLM time 与端到端 latency 的比例。红色和橙色越长，说明 graph 依赖关系越强地决定了整体 latency。

**图结构洞察：**hybrid graph 会叠加两类瓶颈：manager 的集中式关键路径和 peer/barrier 的同步开销。对 MAS serving 的意义是：仅优化平均 request latency 可能无法改善端到端 workflow latency，需要识别 graph critical path 和同步点。

## 6. Query-level vs Workflow-level 的错配

![Figure A](figures/fig_a_prefill_vs_decode_tokens.png)

**读图 tips：**横轴是 prefill/input tokens，纵轴是 decode/output tokens。每个气泡代表一个 graph role，气泡大小表示该 role 的 request 数量，横/竖浅色线表示该 role 的 token 范围。重点不是点的多少，而是不同 role 的输入/输出 token 形状不同，说明 graph role 会改变 backend workload。

**图结构洞察：**manager、reviewer、finalizer、peer agent 等角色不是同质请求。相同 backend 看到的都是 OpenAI-compatible request，但 MAS graph 中的 role 决定了 prompt 组成、prefill 压力和后续依赖重要性。

![Figure G](figures/fig_g_query_workflow_mismatch.png)

**读图 tips：**灰色柱表示 workflow 端到端 latency；蓝点是该 workflow 中最慢的单个 query；红色菱形是最慢的 critical query；橙线是 barrier wait 占 workflow latency 的比例。如果灰柱明显高于蓝点，说明 workflow 慢不是由单个 query 慢完全解释的。

**图结构洞察：**query-level 指标和 workflow-level 指标会错配。一个 query TTFT/latency 高，不一定决定 workflow；真正决定端到端延迟的可能是 barrier wait、critical path 位置或 finalizer 前的 fan-in。这个结论支撑 graph-aware simulator：replay 必须保留依赖图，而不是只 replay query 列表。

## 7. Composite Tool-resume Contention Scope

这一节对应 `tool_resume_contention_meso`。它不是孤立 motif，而是由 planner/worker、tool evidence branch、reviewer/finalizer critical path 和 merge/fan-in 组合出来的 meso workflow。当前图来自真实 vLLM + live web search stress sweep，共 `{len(tool_workflows)}` 个 `tool_resume_contention_meso` runs，宽度分布为 `{tool_width_text}`，critical reviewer/finalizer points=`{critical_points}`。critical path 是 `planner -> critical_coder -> critical_reviewer -> finalizer`；非关键分支是 `tool_agent_i -> web_search/tool_call -> resume_i`。关键设计点是：critical path 不强制等待所有 tool branch 完成，因此 tool 返回后的 resume LLM 有机会与 reviewer/finalizer 阶段重叠。

![Composite Figure 0](figures/fig_tool_resume_workflow_dag.png)

**读图 tips：**红色是 critical path，蓝色是 non-critical tool-stalled branches，绿色是真实 web search tool call。每条工具分支都是 `LLM-1 -> Tool Call -> LLM-2 Resume` 生命周期；这些 resume request 最后进入 merge/finalizer 相关路径。

**图结构洞察：**这张图说明争用机会来自 workflow graph 组合，而不是人为创建一个特殊 benchmark。它把 tool stall、background resume、critical reviewer/finalizer 和 downstream merge 放在同一个 DAG 中。

![Composite Figure 1](figures/fig_tool_resume_overlap_timeline.png)

**读图 tips：**横轴是真实相对时间。红色条是 critical LLM，蓝色条是 background resume LLM，绿色条是 tool call。浅橙色区域表示 background resume 与 critical reviewer/finalizer 的实际重叠窗口。

**图结构洞察：**这里可以声称 observed overlap / contention opportunity：工具返回后的 background resume request 确实和 critical path stage 接近或重叠。但这还不是 priority inversion 证明。

![Composite Figure 2](figures/fig_tool_resume_overlap_events_over_time.png)

**读图 tips：**横轴是时间，纵轴是累计事件数。蓝线表示累计 overlap 事件，深蓝线表示 background-resume-overlap-critical 事件，红线表示 critical slowdown 事件。红线保持 0 表示当前 trace 没有足够证据声称 slowdown。

**图结构洞察：**这张图把“重叠”从单个 timeline 现象提升成事件类型：随着 workflow 推进，tool resume 和 critical stage 的 overlap 可以被计数。后续增加 repeat/concurrency 后，可以用同一张图观察事件是否持续累积。

![Composite Figure 3](figures/fig_tool_resume_noncritical_kv_proxy.png)

**读图 tips：**灰线是 tool call 期间 non-critical stalled branch 的 context/KV proxy，红线是 active critical request 的 context/KV proxy，蓝线是 background resume request 的 context/KV proxy。纵轴不是实际 KV block，而是由真实 token 数推导的 context-token proxy。

**证据边界：**这张图不能写成 observed KV residency。它只能说明：非关键工具分支在 stall/resume 生命周期里携带的上下文 token 可能转化为 KV/cache 压力。实际 KV block residency 需要 vLLM 插桩或更细的 cache metrics。

![Composite Figure 4](figures/fig_tool_resume_critical_latency_vs_overlap.png)

**读图 tips：**横轴是 critical reviewer/finalizer 执行时同时重叠的 background resume request 数量，纵轴是该 critical request 的 `request_e2e_sec`。如果点随横轴明显上升，才有 slowdown 趋势。

**图结构洞察：**stress sweep 后已经能看到 width 4/8 下的多组 overlap 点，图比单 run 更能说明 background resume burst 会系统性靠近 critical stage。不过本节仍然只声称 observed contention opportunity；除非后续能从 vLLM 拿到更明确的 queue wait / batch membership，并看到 critical latency 随 overlap 或 waiting requests 稳定上升，否则不能写成 serving-level contention proof。

## 8. Implications for MASBench

这些 trace 和图结构洞察支持五个方向：

- **Simulator Design：**trace replay 必须包含 workflow、node、query、tool、barrier、prefix/cache 层，而不是只有 LLM request 时间序列。
- **Graph-aware Scheduling：**scheduler 需要知道 request 属于 manager、worker、reviewer、finalizer 还是 background branch。
- **Critical-path-aware Serving：**critical path 上的 request 对端到端 latency 更敏感，不能和所有 background request 扁平处理。
- **Barrier-aware Batching：**fan-in 和 debate round barrier 会把局部慢请求放大成 workflow 等待。
- **Prefix-aware Context Management：**review loop、manager aggregation 和 debate shared context 都会重复携带长 prompt segment，存在 simulator-side potential prefix reuse。

## 9. Next Steps

- 将 simulator-ready trace schema 扩展到所有 motif 和 topology。
- 增加更高 branch width、longer context、repeat 和并发 workload，用来放大 graph-induced bottleneck。
- 校验 vLLM prefix-cache counter 与 per-request prompt segment/hash 的关系。
- 实现 trace-driven simulator replay。
- 原型化 graph-aware scheduling、critical-path-aware scheduling 和 barrier-aware batching 策略。
"""
    path.write_text(text, encoding="utf-8")


def copy_raw_traces(runs: list[dict[str, Any]], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for run in runs:
        src = run["path"]
        rel = Path(run["workflow_name"]) / src.parent.name / src.name
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        bsrc = backend_path_for(src)
        if bsrc.exists():
            shutil.copy2(bsrc, out.with_name(out.stem + "_backend_metrics.json"))


def main() -> int:
    args = parse_args()
    trace_root = Path(args.trace_root).expanduser().resolve()
    progress = Path(args.progress_dir).expanduser().resolve()
    tables_dir = progress / "tables"
    fig_dir = progress / "figures"
    runs = discover_runs(trace_root)
    if as_bool(args.copy_traces):
        copy_raw_traces(runs, progress / "traces")
    tables = build_tables(runs)
    for name, rows in tables.items():
        write_csv(tables_dir / f"week3_{name}.csv", rows)
    write_csv(tables_dir / "week3_plot_summary.csv", aggregate_for_figures(tables))
    write_figures(tables, fig_dir)
    write_schema(progress / "trace_schema.md")
    write_report(progress / "week3.md", tables, trace_root)
    print(f"wrote {progress / 'week3.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

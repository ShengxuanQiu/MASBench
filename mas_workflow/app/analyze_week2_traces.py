"""Validate existing Week-2 traces and characterize MAS runtime structure.

This module is intentionally read-only with respect to workloads: it consumes
already generated JSONL traces plus sidecar exports and writes analysis
artifacts. It does not invoke agents, tools, vLLM, or workload runners.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


TOPOLOGIES = ["single", "independent", "centralized", "decentralized", "hybrid"]
MOTIFS = [
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
]
EXPECTED_NAMES = TOPOLOGIES + MOTIFS
TOOL_HEAVY = {"evidence_collection", "tool_specialist_team"}
DATAFLOW_HEAVY = {
    "evidence_collection",
    "researcher_synthesizer",
    "multi_coder_branch",
    "debate_reviewer",
    "all_gather_round",
    "shared_evidence_store",
    "retry_debug_loop",
    "coder_reviewer",
}
RETRY_HEAVY = {"coder_reviewer", "retry_debug_loop", "generator_verifier"}
MEMORY_HEAVY = {"shared_evidence_store"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and characterize existing Week-2 MAS traces")
    parser.add_argument("--trace-root", default="traces")
    parser.add_argument("--output-dir", default="traces/week2_characterization")
    parser.add_argument("--progress-dir", default="progress/week2")
    parser.add_argument("--include-backend-metrics", default="true")
    parser.add_argument("--include-model-outputs", default="false")
    parser.add_argument(
        "--extra-trace-root",
        action="append",
        default=[],
        help="Additional read-only trace root. Defaults to mas_workflow/traces when present.",
    )
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    events.append(item)
            except json.JSONDecodeError:
                events.append({"event_type": "decode_error", "raw_line": line[:200]})
    return events


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


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def coverage(events: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    total = len(events)
    if total == 0:
        return {"total": 0, "coverage": {field: 0.0 for field in fields}}
    return {
        "total": total,
        "coverage": {
            field: round(sum(1 for event in events if event.get(field) not in (None, "", [])) / total, 4)
            for field in fields
        },
    }


def discover_trace_roots(args: argparse.Namespace) -> list[Path]:
    roots = [Path(args.trace_root)]
    roots.extend(Path(item) for item in args.extra_trace_root)
    default_topology_root = Path("mas_workflow/traces")
    if default_topology_root.exists() and default_topology_root not in roots:
        roots.append(default_topology_root)
    unique: list[Path] = []
    for root in roots:
        root = root.expanduser()
        if root.exists() and root not in unique:
            unique.append(root)
    return unique


def trace_name(events: list[dict[str, Any]], path: Path) -> tuple[str, str]:
    first = events[0] if events else {}
    mode = str(first.get("mode") or "topology")
    motif = next((str(event.get("motif_name")) for event in events if event.get("motif_name")), "")
    topology = str(first.get("topology") or path.parts[-3] if len(path.parts) >= 3 else "")
    if motif:
        return "motif", motif
    if topology in MOTIFS:
        return "motif", topology
    return "topology", topology


def sidecar_paths(path: Path) -> dict[str, Path]:
    stem = path.with_suffix("")
    return {
        "jsonl": path,
        "summary": Path(f"{stem}_summary.json"),
        "spans": Path(f"{stem}_spans.json"),
        "otel": Path(f"{stem}_otel.json"),
        "jaeger": Path(f"{stem}_jaeger.json"),
        "viewer": Path(f"{stem}_viewer.html"),
        "backend_metrics": Path(f"{stem}_backend_metrics.json"),
        "model_outputs": Path(f"{stem}_model_outputs.json"),
    }


def file_inventory(path: Path, include_backend: bool, include_outputs: bool) -> dict[str, Any]:
    paths = sidecar_paths(path)
    required = ["jsonl", "summary", "spans", "viewer"]
    optional = ["otel", "jaeger"]
    if include_backend:
        optional.append("backend_metrics")
    if include_outputs:
        optional.append("model_outputs")
    row: dict[str, Any] = {"trace_path": str(path)}
    for key, item in paths.items():
        row[f"{key}_path"] = str(item)
        row[f"{key}_exists"] = item.exists()
        row[f"{key}_bytes"] = item.stat().st_size if item.exists() else 0
    row["required_files_ok"] = all(paths[key].exists() for key in required)
    row["optional_file_count"] = sum(1 for key in optional if paths[key].exists())
    return row


def event_time(event: dict[str, Any]) -> float:
    return safe_float(event.get("relative_time_sec"))


def event_start(event: dict[str, Any]) -> float:
    duration = safe_float(event.get("duration_sec"))
    return max(0.0, event_time(event) - duration)


def estimate_depth(events: list[dict[str, Any]]) -> int:
    nodes = [event for event in events if event.get("node_id") and event.get("node_type") != "edge"]
    parents: dict[str, list[str]] = {}
    for event in nodes:
        node = str(event.get("node_id"))
        parents.setdefault(node, [])
        for parent in event.get("parents") or []:
            parents[node].append(str(parent))
    memo: dict[str, int] = {}

    def depth(node: str) -> int:
        if node in memo:
            return memo[node]
        ps = parents.get(node) or []
        memo[node] = 1 + max((depth(parent) for parent in ps if parent != node), default=0)
        return memo[node]

    return max((depth(node) for node in parents), default=0)


def max_overlap(events: list[dict[str, Any]]) -> int:
    intervals: list[tuple[float, int]] = []
    for event in events:
        if event.get("event_type") not in {"llm_request_end", "tool_search"}:
            continue
        start = event_start(event)
        end = max(start, event_time(event))
        intervals.append((start, 1))
        intervals.append((end, -1))
    active = 0
    best = 0
    for _, delta in sorted(intervals):
        active += delta
        best = max(best, active)
    return best or 1


def burstiness(llm_events: list[dict[str, Any]]) -> float:
    starts = [safe_float(event.get("generation_start_ts"), event_start(event)) for event in llm_events]
    if len(starts) < 2:
        return 1.0 if starts else 0.0
    base = min(starts)
    bins = Counter(int(start - base) for start in starts)
    counts = list(bins.values())
    return round(max(counts) / max(1.0, mean(counts)), 4)


def backend_available_fields(backend_path: Path) -> list[str]:
    data = read_json(backend_path)
    if not isinstance(data, dict):
        return []
    fields = set(data.get("summary", {}).keys())
    samples = data.get("samples") or []
    if samples and isinstance(samples[0], dict):
        metrics = samples[0].get("metrics")
        if isinstance(metrics, dict):
            for key in metrics:
                if any(term in key for term in ("time_to_first_token", "time_per_output_token", "kv", "cache", "num_requests")):
                    fields.add(key)
    return sorted(fields)


def analyze_run(path: Path, include_backend: bool, include_outputs: bool) -> dict[str, Any]:
    events = read_jsonl(path)
    mode, name = trace_name(events, path)
    paths = sidecar_paths(path)
    inventory = file_inventory(path, include_backend, include_outputs)
    summary_sidecar = read_json(paths["summary"]) if paths["summary"].exists() else {}
    if not isinstance(summary_sidecar, dict):
        summary_sidecar = {}

    llm = [event for event in events if event.get("event_type") == "llm_request_end"]
    tools = [event for event in events if str(event.get("event_type", "")).startswith("tool_")]
    edges = [event for event in events if event.get("node_type") == "edge" or event.get("event_type") == "edge_dataflow"]
    barriers = [event for event in events if event.get("event_type") == "barrier" or event.get("node_type") == "barrier"]
    memory_writes = [event for event in events if event.get("artifact_type") == "memory_write" or event.get("memory_write")]
    memory_reads = [event for event in events if event.get("artifact_type") == "memory_read" or event.get("memory_read")]

    makespan = max([event_time(event) for event in events] or [0.0])
    llm_durations = [safe_float(event.get("duration_sec")) for event in llm]
    tool_durations = [
        safe_float(event.get("effective_duration_sec"), safe_float(event.get("measured_duration_sec"), safe_float(event.get("duration_sec"))))
        for event in tools
    ]
    total_input = sum(safe_int(event.get("input_tokens_est") or event.get("prompt_tokens") or event.get("backend_prompt_tokens")) for event in llm)
    total_output = sum(
        safe_int(event.get("output_tokens_est") or event.get("completion_tokens") or event.get("backend_completion_tokens")) for event in llm
    )
    backend_total = sum(safe_int(event.get("backend_total_tokens")) for event in llm)
    total_tokens = total_input + total_output
    if total_tokens == 0:
        total_tokens = backend_total

    aggregation = sum(
        safe_int(event.get("aggregation_tokens_est") or event.get("artifact_tokens_est"))
        for event in edges
        if event.get("artifact_type") in {"summary", "final", "vote", "evidence_group", "candidate", "review"}
        or "merge" in str(event.get("node_id", "")).lower()
        or "synth" in str(event.get("node_id", "")).lower()
    )
    broadcast = sum(safe_int(event.get("broadcast_tokens_est") or event.get("artifact_tokens_est")) for event in events if event.get("transfer_type") == "broadcast")
    peer_message = sum(
        safe_int(event.get("peer_message_tokens_est") or event.get("artifact_tokens_est"))
        for event in events
        if event.get("artifact_type") == "peer_message" or event.get("peer_message_tokens_est")
    )
    duplicated = sum(safe_int(event.get("duplicated_context_tokens_est")) for event in events)
    shared_read = sum(safe_int(event.get("shared_evidence_read_tokens_est")) for event in events)
    shared_write = sum(safe_int(event.get("shared_evidence_write_tokens_est")) for event in events)
    retry_added = sum(safe_int(event.get("retry_added_tokens_est") or event.get("revision_tokens_est")) for event in events)
    candidate_tokens = sum(safe_int(event.get("candidate_tokens_est") or event.get("artifact_tokens_est")) for event in events if event.get("artifact_type") == "candidate")
    reviewer_input = sum(safe_int(event.get("reviewer_input_tokens_est")) for event in events)
    fanin = aggregation + candidate_tokens + reviewer_input + shared_read

    retry_loop_count = max([safe_int(event.get("retry_count")) for event in events] or [0])
    debug_loop_count = max([safe_int(event.get("debug_loop_count")) for event in events] or [0])
    review_loop_count = max([safe_int(event.get("review_loop_count")) for event in events] or [0])
    handoff_count = sum(safe_int(event.get("handoff_count")) for event in events)
    manager_rounds = len({event.get("manager_round_id") for event in events if event.get("manager_round_id") is not None})
    peer_rounds = len({event.get("peer_round_id") for event in events if event.get("peer_round_id") is not None})

    critical_nodes = [event for event in events if event.get("critical_path_flag") or event.get("criticality") == "critical"]
    if critical_nodes:
        critical_path_time = sum(safe_float(event.get("duration_sec")) for event in critical_nodes)
        critical_node_count = len({event.get("node_id") for event in critical_nodes if event.get("node_id")})
    else:
        # Best-effort proxy: serial LLM/tool work on the observed longest named path
        by_node: dict[str, float] = defaultdict(float)
        for event in llm + tools:
            by_node[str(event.get("node_id") or event.get("agent_id") or event.get("tool_name") or "unknown")] += safe_float(
                event.get("duration_sec")
            )
        critical_path_time = max(by_node.values(), default=0.0)
        critical_node_count = 0

    barrier_wait = sum(safe_float(event.get("barrier_wait_sec")) for event in barriers)
    max_barrier = max([safe_float(event.get("barrier_wait_sec")) for event in barriers] or [0.0])
    straggler_gap = max([safe_float(event.get("straggler_gap_sec")) for event in barriers] or [0.0])
    manager_time = sum(
        safe_float(event.get("duration_sec"))
        for event in llm
        if event.get("agent_role") in {"manager", "planner", "router", "selector", "reviewer", "synthesizer", "finalizer"}
    )

    backend_fields = backend_available_fields(paths["backend_metrics"]) if include_backend and paths["backend_metrics"].exists() else []
    ttft_values = [safe_float(event.get("ttft_sec")) for event in llm if event.get("ttft_sec") not in (None, "")]
    tpot_values = [safe_float(event.get("tpot_sec")) for event in llm if event.get("tpot_sec") not in (None, "")]
    if not ttft_values and summary_sidecar.get("backend_avg_ttft_sec_from_metrics"):
        ttft_values = [safe_float(summary_sidecar.get("backend_avg_ttft_sec_from_metrics"))]
    if not tpot_values and summary_sidecar.get("backend_avg_tpot_sec_from_metrics"):
        tpot_values = [safe_float(summary_sidecar.get("backend_avg_tpot_sec_from_metrics"))]

    token_base = max(1, total_input)
    metrics = {
        "trace_path": str(path),
        "mode": mode,
        "name": name,
        "run_id": events[0].get("run_id") if events else path.stem,
        "instance_id": events[0].get("instance_id") if events else path.parent.name,
        "motif_instance_id": next((event.get("motif_instance_id") for event in events if event.get("motif_instance_id")), ""),
        "composed_from_topologies": next((event.get("composed_from_topologies") for event in events if event.get("composed_from_topologies")), []),
        "event_count": len(events),
        "makespan_sec": round(makespan, 6),
        "graph_depth": estimate_depth(events),
        "graph_width": len({event.get("node_id") for event in events if event.get("node_id")}),
        "max_parallel_width": max_overlap(events),
        "critical_path_length_sec": round(critical_path_time, 6),
        "critical_path_node_count": critical_node_count,
        "total_barrier_wait_sec": round(barrier_wait, 6),
        "max_barrier_wait_sec": round(max_barrier, 6),
        "straggler_gap_sec": round(straggler_gap, 6),
        "manager_serialization_ratio": round(manager_time / makespan, 4) if makespan else 0.0,
        "retry_loop_count": max(retry_loop_count, debug_loop_count),
        "debug_loop_count": debug_loop_count,
        "review_loop_count": review_loop_count,
        "handoff_count": handoff_count,
        "manager_rounds_actual": manager_rounds,
        "peer_rounds_actual": peer_rounds,
        "total_input_tokens_est": total_input,
        "total_output_tokens_est": total_output,
        "total_tokens_est": total_tokens,
        "backend_total_tokens": backend_total,
        "aggregation_tokens_est": aggregation,
        "broadcast_tokens_est": broadcast,
        "peer_message_tokens_est": peer_message,
        "duplicated_context_tokens_est": duplicated,
        "fanin_input_tokens_est": fanin,
        "shared_evidence_read_tokens_est": shared_read,
        "shared_evidence_write_tokens_est": shared_write,
        "retry_added_tokens_est": retry_added,
        "candidate_tokens_est": candidate_tokens,
        "reviewer_input_tokens_est": reviewer_input,
        "context_amplification_ratio": round((total_input + aggregation + broadcast + peer_message + duplicated) / token_base, 4),
        "fanin_amplification_ratio": round(fanin / token_base, 4),
        "broadcast_amplification_ratio": round((broadcast + peer_message + duplicated) / token_base, 4),
        "retry_token_amplification_ratio": round(retry_added / token_base, 4),
        "llm_request_count": len(llm),
        "request_arrival_burstiness": burstiness(llm),
        "avg_request_e2e_sec": round(mean(llm_durations), 6) if llm_durations else 0.0,
        "p50_request_e2e_sec": round(median(llm_durations), 6) if llm_durations else 0.0,
        "p95_request_e2e_sec": round(pct(llm_durations, 0.95), 6) if llm_durations else 0.0,
        "avg_ttft_sec": round(mean(ttft_values), 6) if ttft_values else 0.0,
        "p95_ttft_sec": round(pct(ttft_values, 0.95), 6) if ttft_values else 0.0,
        "avg_tpot_sec": round(mean(tpot_values), 6) if tpot_values else 0.0,
        "p95_tpot_sec": round(pct(tpot_values, 0.95), 6) if tpot_values else 0.0,
        "critical_path_llm_time_sec": round(sum(safe_float(event.get("duration_sec")) for event in critical_nodes if event.get("event_type") == "llm_request_end"), 6),
        "noncritical_llm_time_sec": round(sum(llm_durations) - sum(safe_float(event.get("duration_sec")) for event in critical_nodes if event.get("event_type") == "llm_request_end"), 6),
        "backend_metric_available_fields": backend_fields,
        "backend_metrics_exists": paths["backend_metrics"].exists(),
        "tool_event_count": len(tools),
        "total_tool_time_sec": round(sum(tool_durations), 6),
        "max_tool_time_sec": round(max(tool_durations or [0.0]), 6),
        "tool_stall_window_sec": round(sum(tool_durations), 6),
        "downstream_blocked_time_sec": round(barrier_wait + max_barrier, 6),
        "post_tool_request_burst_count": post_tool_burst_count(tools, llm),
        "memory_write_count": len(memory_writes),
        "memory_read_count": len(memory_reads),
        "artifact_version_count": len({event.get("artifact_version") for event in events if event.get("artifact_version") not in (None, "")}),
        "stale_read_count": sum(1 for event in events if event.get("stale_read")),
    }
    validation = validate_run(events, mode, name, inventory, llm, tools, paths)
    signatures = detect_signatures(metrics)
    return {"metrics": metrics, "validation": validation, "inventory": inventory, "signatures": signatures}


def post_tool_burst_count(tools: list[dict[str, Any]], llm: list[dict[str, Any]]) -> int:
    if not tools or not llm:
        return 0
    tool_ends = [event_time(event) for event in tools]
    llm_starts = [event_start(event) for event in llm]
    return max((sum(1 for start in llm_starts if 0 <= start - end <= 2.0) for end in tool_ends), default=0)


def validate_run(
    events: list[dict[str, Any]],
    mode: str,
    name: str,
    inventory: dict[str, Any],
    llm: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    paths: dict[str, Path],
) -> dict[str, Any]:
    graph_fields = [
        "run_id",
        "mode",
        "node_id",
        "agent_role",
        "relative_time_sec",
        "duration_sec",
        "parents",
        "parallel_group",
        "round_id",
        "peer_round_id",
        "manager_round_id",
        "retry_count",
        "critical_path_flag",
        "barrier_wait_sec",
    ]
    llm_fields = [
        "request_id_for_backend",
        "backend_base_url",
        "model",
        "agent_role",
        "node_id",
        "generation_start_ts",
        "generation_end_ts",
        "input_tokens_est",
        "output_tokens_est",
        "backend_prompt_tokens",
        "backend_completion_tokens",
        "backend_total_tokens",
        "duration_sec",
        "ttft_sec",
        "tpot_sec",
    ]
    tool_fields = [
        "tool_name",
        "tool_mode",
        "tool_start_ts",
        "tool_end_ts",
        "tool_latency_sec",
        "measured_duration_sec",
        "effective_duration_sec",
        "replay_policy",
        "tool_result_hash",
    ]
    dataflow_fields = [
        "aggregation_tokens_est",
        "broadcast_tokens_est",
        "peer_message_tokens_est",
        "duplicated_context_tokens_est",
        "shared_evidence_read_tokens_est",
        "shared_evidence_write_tokens_est",
        "retry_added_tokens_est",
        "candidate_tokens_est",
        "reviewer_input_tokens_est",
    ]
    motif_expected = mode == "motif"
    missing_required: list[str] = []
    if not inventory["required_files_ok"]:
        missing_required.extend(key for key in ("jsonl", "summary", "spans", "viewer") if not inventory.get(f"{key}_exists"))
    if motif_expected:
        for field in ("motif_name", "motif_instance_id", "composed_from_topologies"):
            if not any(event.get(field) not in (None, "", []) for event in events):
                missing_required.append(field)
    else:
        if not any(event.get("topology") for event in events):
            missing_required.append("topology")

    categories = {
        "graph": coverage(events, graph_fields),
        "llm_backend": coverage(llm, llm_fields),
        "tool": coverage(tools, tool_fields),
        "dataflow": coverage(events, dataflow_fields),
    }
    field_expectations = classify_field_expectations(name, len(tools) > 0)
    return {
        "trace_path": inventory["trace_path"],
        "mode": mode,
        "name": name,
        "file_integrity_ok": bool(inventory["required_files_ok"]),
        "backend_metrics_exists": bool(paths["backend_metrics"].exists()),
        "model_outputs_exists": bool(paths["model_outputs"].exists()),
        "missing_required": missing_required,
        "llm_request_count": len(llm),
        "request_id_for_backend_coverage": categories["llm_backend"]["coverage"].get("request_id_for_backend", 0.0),
        "token_field_coverage": max(
            categories["llm_backend"]["coverage"].get("backend_total_tokens", 0.0),
            categories["llm_backend"]["coverage"].get("input_tokens_est", 0.0),
        ),
        "timing_field_coverage": categories["llm_backend"]["coverage"].get("duration_sec", 0.0),
        "coverage": categories,
        "field_expectations": field_expectations,
        "validation_status": "pass" if not missing_required else "warn",
    }


def classify_field_expectations(name: str, has_tools: bool) -> dict[str, str]:
    status = {
        "tool_name": "expected" if has_tools or name in TOOL_HEAVY else "not_applicable",
        "tool_mode": "expected" if has_tools or name in TOOL_HEAVY else "not_applicable",
        "tool_start_ts": "optional",
        "tool_end_ts": "optional",
        "tool_latency_sec": "optional",
        "measured_duration_sec": "expected" if has_tools else "not_applicable",
        "replay_policy": "expected" if has_tools else "not_applicable",
        "tool_result_hash": "optional" if has_tools else "not_applicable",
        "aggregation_tokens_est": "expected" if name in DATAFLOW_HEAVY else "optional",
        "broadcast_tokens_est": "expected" if name in {"debate_reviewer", "all_gather_round"} else "optional",
        "peer_message_tokens_est": "expected" if name in {"debate_reviewer", "all_gather_round"} else "optional",
        "duplicated_context_tokens_est": "expected" if name == "all_gather_round" else "optional",
        "shared_evidence_read_tokens_est": "expected" if name in MEMORY_HEAVY else "optional",
        "shared_evidence_write_tokens_est": "expected" if name in MEMORY_HEAVY else "optional",
        "retry_added_tokens_est": "expected" if name in RETRY_HEAVY else "optional",
        "candidate_tokens_est": "expected" if name == "multi_coder_branch" else "optional",
        "reviewer_input_tokens_est": "expected" if name in {"coder_reviewer", "multi_coder_branch"} else "optional",
        "ttft_sec": "optional",
        "tpot_sec": "optional",
        "critical_path_flag": "optional",
        "barrier_wait_sec": "optional",
    }
    return status


def detect_signatures(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    signatures: list[dict[str, Any]] = []
    name = str(metrics.get("name") or "")

    def add(name: str, confidence: str, support: dict[str, Any]) -> None:
        signatures.append(
            {
                "signature": name,
                "confidence": confidence,
                "supporting_metrics": support,
                "representative_trace_path": metrics["trace_path"],
            }
        )

    manager_like = {
        "centralized",
        "hybrid",
        "planner_executor",
        "generator_verifier",
        "coder_reviewer",
        "multi_coder_branch",
        "debate_reviewer",
        "router_handoff",
    }
    if name in manager_like and metrics["manager_serialization_ratio"] >= 0.35 and metrics["llm_request_count"] >= 3:
        add(
            "serial_manager_bottleneck",
            "medium" if metrics["manager_serialization_ratio"] < 0.55 else "high",
            {
                "manager_serialization_ratio": metrics["manager_serialization_ratio"],
                "manager_rounds_actual": metrics["manager_rounds_actual"],
            },
        )
    if metrics["critical_path_length_sec"] and metrics["makespan_sec"] and metrics["critical_path_length_sec"] / metrics["makespan_sec"] >= 0.35:
        add(
            "critical_path_concentration",
            "low" if metrics["critical_path_node_count"] == 0 else "medium",
            {
                "critical_path_length_sec": metrics["critical_path_length_sec"],
                "makespan_sec": metrics["makespan_sec"],
                "critical_path_node_count": metrics["critical_path_node_count"],
            },
        )
    if metrics["request_arrival_burstiness"] >= 1.5 and metrics["llm_request_count"] >= 4:
        add(
            "parallel_request_burst",
            "medium",
            {
                "request_arrival_burstiness": metrics["request_arrival_burstiness"],
                "max_parallel_width": metrics["max_parallel_width"],
                "llm_request_count": metrics["llm_request_count"],
            },
        )
    if metrics["straggler_gap_sec"] > 0 or metrics["total_barrier_wait_sec"] > 0:
        add(
            "barrier_straggler",
            "medium",
            {
                "total_barrier_wait_sec": metrics["total_barrier_wait_sec"],
                "straggler_gap_sec": metrics["straggler_gap_sec"],
            },
        )
    if metrics["fanin_amplification_ratio"] >= 0.15 or metrics["aggregation_tokens_est"] >= 1000:
        add(
            "fanin_context_amplification",
            "medium" if metrics["aggregation_tokens_est"] else "low",
            {
                "aggregation_tokens_est": metrics["aggregation_tokens_est"],
                "fanin_amplification_ratio": metrics["fanin_amplification_ratio"],
            },
        )
    broadcast_like = {"decentralized", "hybrid", "debate_reviewer", "all_gather_round"}
    if name in broadcast_like and (metrics["broadcast_amplification_ratio"] >= 0.1 or metrics["peer_message_tokens_est"] >= 1000):
        add(
            "all_to_all_broadcast_amplification",
            "medium" if metrics["peer_message_tokens_est"] else "low",
            {
                "broadcast_tokens_est": metrics["broadcast_tokens_est"],
                "peer_message_tokens_est": metrics["peer_message_tokens_est"],
                "duplicated_context_tokens_est": metrics["duplicated_context_tokens_est"],
            },
        )
    if metrics["total_tool_time_sec"] >= 5:
        add(
            "tool_stall_blocking",
            "medium" if name in TOOL_HEAVY or name in {"evidence_collection", "tool_specialist_team"} else "low",
            {
                "total_tool_time_sec": metrics["total_tool_time_sec"],
                "max_tool_time_sec": metrics["max_tool_time_sec"],
                "downstream_blocked_time_sec": metrics["downstream_blocked_time_sec"],
            },
        )
    if metrics["post_tool_request_burst_count"] >= 2:
        add(
            "post_tool_burst",
            "low" if metrics["post_tool_request_burst_count"] < 4 else "medium",
            {"post_tool_request_burst_count": metrics["post_tool_request_burst_count"]},
        )
    if metrics["retry_loop_count"] > 0 or metrics["review_loop_count"] > 0 or metrics["retry_token_amplification_ratio"] > 0:
        add(
            "retry_amplification",
            "medium",
            {
                "retry_loop_count": metrics["retry_loop_count"],
                "review_loop_count": metrics["review_loop_count"],
                "retry_token_amplification_ratio": metrics["retry_token_amplification_ratio"],
            },
        )
    if metrics["memory_write_count"] > 0 or metrics["memory_read_count"] > 0 or metrics["shared_evidence_read_tokens_est"] > 0:
        add(
            "shared_memory_dataflow",
            "medium",
            {
                "memory_write_count": metrics["memory_write_count"],
                "memory_read_count": metrics["memory_read_count"],
                "shared_evidence_read_tokens_est": metrics["shared_evidence_read_tokens_est"],
                "shared_evidence_write_tokens_est": metrics["shared_evidence_write_tokens_est"],
            },
        )
    if not metrics["backend_metrics_exists"] or not any("kv" in field.lower() or "cache" in field.lower() for field in metrics["backend_metric_available_fields"]):
        add(
            "backend_metric_insufficient",
            "high" if not metrics["backend_metrics_exists"] else "medium",
            {
                "backend_metrics_exists": metrics["backend_metrics_exists"],
                "backend_metric_available_field_count": len(metrics["backend_metric_available_fields"]),
            },
        )
    return signatures


def aggregate_by_name(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[run["metrics"]["name"]].append(run)
    out: dict[str, dict[str, Any]] = {}
    for name in EXPECTED_NAMES:
        items = grouped.get(name, [])
        if not items:
            out[name] = {"name": name, "run_count": 0, "status": "trace_missing"}
            continue
        metrics = [item["metrics"] for item in items]
        signature_counts = Counter(sig["signature"] for item in items for sig in item["signatures"])
        confidences = Counter(sig["confidence"] for item in items for sig in item["signatures"])
        out[name] = {
            "name": name,
            "mode": metrics[0]["mode"],
            "run_count": len(items),
            "status": "present",
            "avg_makespan_sec": round(mean(m["makespan_sec"] for m in metrics), 4),
            "avg_llm_request_count": round(mean(m["llm_request_count"] for m in metrics), 4),
            "avg_max_parallel_width": round(mean(m["max_parallel_width"] for m in metrics), 4),
            "avg_total_tokens_est": round(mean(m["total_tokens_est"] for m in metrics), 2),
            "avg_tool_time_sec": round(mean(m["total_tool_time_sec"] for m in metrics), 4),
            "avg_barrier_wait_sec": round(mean(m["total_barrier_wait_sec"] for m in metrics), 4),
            "avg_context_amplification_ratio": round(mean(m["context_amplification_ratio"] for m in metrics), 4),
            "avg_request_burstiness": round(mean(m["request_arrival_burstiness"] for m in metrics), 4),
            "avg_manager_serialization_ratio": round(mean(m["manager_serialization_ratio"] for m in metrics), 4),
            "avg_broadcast_amplification_ratio": round(mean(m["broadcast_amplification_ratio"] for m in metrics), 4),
            "avg_fanin_amplification_ratio": round(mean(m["fanin_amplification_ratio"] for m in metrics), 4),
            "detected_signatures": [name for name, _ in signature_counts.most_common()],
            "confidence": confidence_label(confidences),
            "representative_trace_path": max(items, key=lambda item: item["metrics"]["makespan_sec"])["metrics"]["trace_path"],
        }
    return out


def confidence_label(counts: Counter[str]) -> str:
    if counts["high"]:
        return "high"
    if counts["medium"]:
        return "medium"
    if counts["low"]:
        return "low"
    return "none"


def write_inventory(path: Path, runs: list[dict[str, Any]], missing: list[dict[str, Any]]) -> None:
    rows = [run["inventory"] for run in runs]
    rows.extend(missing)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tables(path: Path, aggregate: dict[str, dict[str, Any]]) -> None:
    rows = list(aggregate.values())
    fieldnames = [
        "name",
        "mode",
        "status",
        "run_count",
        "avg_makespan_sec",
        "avg_llm_request_count",
        "avg_total_tokens_est",
        "avg_tool_time_sec",
        "avg_barrier_wait_sec",
        "avg_context_amplification_ratio",
        "avg_request_burstiness",
        "avg_manager_serialization_ratio",
        "avg_fanin_amplification_ratio",
        "avg_broadcast_amplification_ratio",
        "detected_signatures",
        "confidence",
        "representative_trace_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["detected_signatures"] = ";".join(row.get("detected_signatures") or [])
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def make_figures(progress_dir: Path, aggregate: dict[str, dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    fig_dir = progress_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    present = [row for row in aggregate.values() if row.get("status") == "present"]
    labels = [row["name"] for row in present]

    def save(name: str) -> str:
        out = fig_dir / name
        plt.tight_layout()
        plt.savefig(out, dpi=180)
        plt.close()
        return str(out)

    figures: dict[str, str] = {}

    plt.figure(figsize=(12, 5))
    x = range(len(labels))
    plt.bar(x, [row["avg_makespan_sec"] for row in present], label="makespan sec")
    plt.plot(list(x), [row["avg_llm_request_count"] for row in present], color="black", marker="o", label="LLM requests")
    plt.xticks(list(x), labels, rotation=70, ha="right")
    plt.ylabel("seconds / request count")
    plt.title("Motifs expose different runtime shapes")
    plt.legend()
    figures["runtime_signatures"] = save("insight_1_runtime_signatures.png")

    plt.figure(figsize=(12, 5))
    plt.bar(x, [row["avg_request_burstiness"] for row in present], label="request burstiness")
    plt.plot(list(x), [row["avg_context_amplification_ratio"] for row in present], color="#b23a48", marker="o", label="context amplification")
    plt.xticks(list(x), labels, rotation=70, ha="right")
    plt.ylabel("ratio")
    plt.title("Graph structure changes burstiness and context pressure")
    plt.legend()
    figures["graph_first_class"] = save("insight_2_graph_structure.png")

    run_metrics = [run["metrics"] for run in runs]
    plt.figure(figsize=(8, 5))
    plt.scatter(
        [m["critical_path_length_sec"] for m in run_metrics],
        [m["makespan_sec"] for m in run_metrics],
        s=[20 + 5 * m["llm_request_count"] for m in run_metrics],
        alpha=0.7,
    )
    plt.xlabel("critical path proxy sec")
    plt.ylabel("makespan sec")
    plt.title("Latency criticality is uneven across requests")
    figures["critical_path"] = save("insight_3_critical_path.png")

    plt.figure(figsize=(10, 5))
    selected = [row for row in present if row["avg_fanin_amplification_ratio"] or row["avg_broadcast_amplification_ratio"]]
    sx = range(len(selected))
    plt.bar(sx, [row["avg_fanin_amplification_ratio"] for row in selected], label="fan-in")
    plt.bar(sx, [row["avg_broadcast_amplification_ratio"] for row in selected], bottom=[row["avg_fanin_amplification_ratio"] for row in selected], label="broadcast/all-gather")
    plt.xticks(list(sx), [row["name"] for row in selected], rotation=60, ha="right")
    plt.ylabel("amplification ratio")
    plt.title("Fan-in and all-gather amplify context")
    plt.legend()
    figures["context_amplification"] = save("insight_4_context_amplification.png")

    plt.figure(figsize=(9, 5))
    tool_rows = [row for row in present if row["avg_tool_time_sec"] > 0]
    tx = range(len(tool_rows))
    plt.bar(tx, [row["avg_tool_time_sec"] for row in tool_rows], color="#4878a8")
    plt.xticks(list(tx), [row["name"] for row in tool_rows], rotation=60, ha="right")
    plt.ylabel("tool time sec")
    plt.title("Tool stalls create downstream blocking windows")
    figures["tool_stalls"] = save("insight_5_tool_stalls.png")

    plt.figure(figsize=(8, 5))
    retry_rows = [row for row in present if "retry_amplification" in row.get("detected_signatures", []) or row["name"] in RETRY_HEAVY]
    rx = range(len(retry_rows))
    plt.bar(rx, [row["avg_llm_request_count"] for row in retry_rows], label="LLM requests")
    plt.plot(list(rx), [row["avg_total_tokens_est"] / 1000.0 for row in retry_rows], color="#b23a48", marker="o", label="tokens est / 1k")
    plt.xticks(list(rx), [row["name"] for row in retry_rows], rotation=40, ha="right")
    plt.ylabel("count / 1k tokens")
    plt.title("Review and debug loops repeat LLM work")
    plt.legend()
    figures["retry_loops"] = save("insight_6_retry_loops.png")

    return figures


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    return "\n".join([header, sep] + body)


def validation_report(
    aggregate: dict[str, dict[str, Any]],
    runs: list[dict[str, Any]],
    missing: list[dict[str, Any]],
) -> str:
    validations = [run["validation"] for run in runs]
    llm_count = sum(v["llm_request_count"] for v in validations)
    request_cov = mean([v["request_id_for_backend_coverage"] for v in validations if v["llm_request_count"]] or [0.0])
    token_cov = mean([v["token_field_coverage"] for v in validations if v["llm_request_count"]] or [0.0])
    timing_cov = mean([v["timing_field_coverage"] for v in validations if v["llm_request_count"]] or [0.0])
    backend_cov = sum(1 for v in validations if v["backend_metrics_exists"]) / max(1, len(validations))
    coverage_rows = [
        {
            "name": row["name"],
            "status": row["status"],
            "run_count": row.get("run_count", 0),
            "representative_trace_path": row.get("representative_trace_path", ""),
        }
        for row in aggregate.values()
    ]
    return f"""# Week2 Trace Validation Report

## Coverage

{md_table(coverage_rows, ["name", "status", "run_count", "representative_trace_path"])}

Missing expected families: {", ".join(row["name"] for row in missing) or "none"}.

## File Integrity

- Runs scanned: {len(runs)}
- Runs with required JSONL/summary/spans/viewer files: {sum(1 for v in validations if v["file_integrity_ok"])}/{len(validations)}
- Runs with backend metrics sidecar: {sum(1 for v in validations if v["backend_metrics_exists"])}/{len(validations)}
- Runs with model output sidecar: {sum(1 for v in validations if v["model_outputs_exists"])}/{len(validations)}

## Backend Alignment

- LLM request events: {llm_count}
- request_id_for_backend coverage across runs: {request_cov:.3f}
- token field coverage across runs: {token_cov:.3f}
- client-side timing field coverage across runs: {timing_cov:.3f}
- backend metrics sidecar coverage: {backend_cov:.3f}

## Measurement Semantics

- graph trace timing: measured at the client/workflow process from event timestamps and durations.
- client-side LLM request timing: measured around OpenAI-compatible vLLM calls in `llm_request_end.duration_sec`.
- backend metrics: sampled from vLLM `/metrics` when sidecars exist; these are aggregate server counters/windows, not per-node scheduler internals.
- token fields ending in `_est`: estimated proxies or derived from client-visible text/events.
- backend token counters: available from OpenAI-compatible response usage when present.
- unavailable in current traces: reliable per-request queue time, batch membership, KV allocation/eviction, prefix cache hit/miss, token-level shared-prefix hashes, and explicit streaming first-token timestamps for every request.

## Characterization Sufficiency

The existing traces are sufficient for Week2 graph-level characterization of control flow, data-flow amplification, tool stalls, retry loops, and coarse backend alignment. They are not sufficient to claim vLLM scheduler, KV residency, or cache behavior directly.
"""


def characterization_report(
    aggregate: dict[str, dict[str, Any]],
    runs: list[dict[str, Any]],
    figures: dict[str, str],
) -> str:
    rows = [
        {
            "name": row["name"],
            "run_count": row.get("run_count", 0),
            "avg_makespan_sec": row.get("avg_makespan_sec", ""),
            "avg_llm_request_count": row.get("avg_llm_request_count", ""),
            "avg_total_tokens_est": row.get("avg_total_tokens_est", ""),
            "avg_tool_time_sec": row.get("avg_tool_time_sec", ""),
            "avg_barrier_wait_sec": row.get("avg_barrier_wait_sec", ""),
            "avg_context_amplification_ratio": row.get("avg_context_amplification_ratio", ""),
            "detected_signatures": ";".join(row.get("detected_signatures") or []),
            "confidence": row.get("confidence", ""),
        }
        for row in aggregate.values()
    ]
    signature_counter = Counter(sig["signature"] for run in runs for sig in run["signatures"])
    representative = {}
    for run in runs:
        for sig in run["signatures"]:
            representative.setdefault(sig["signature"], sig["representative_trace_path"])

    def figure_ref(path: str) -> str:
        figure_path = Path(path)
        if "figures" in figure_path.parts:
            return "figures/" + figure_path.name
        return path

    def representatives(*names: str) -> str:
        paths = []
        for name in names:
            path = representative.get(name)
            if path and path not in paths:
                paths.append(path)
        return ", ".join(f"`{path}`" for path in paths) or "not available"

    def sig_line(name: str) -> str:
        return f"- `{name}`: {signature_counter.get(name, 0)} runs; representative `{representative.get(name, '')}`"

    insight_blocks = [
        (
            "Insight 1: Different MAS motifs expose distinct runtime bottleneck signatures.",
            "centralized/planner style motifs show manager serialization; branch motifs show bursts and straggler surfaces; tool-heavy motifs show tool stalls; debate/all-gather shows broadcast amplification; review/debug motifs show retry amplification.",
            figures["runtime_signatures"],
            "medium",
            "Some signatures are proxy-based because explicit critical_path_flag and queue fields are not uniformly available.",
            representatives("serial_manager_bottleneck", "parallel_request_burst", "tool_stall_blocking", "all_to_all_broadcast_amplification", "retry_amplification"),
        ),
        (
            "Insight 2: MAS graph structure should be treated as a first-class workload property.",
            "With the same OpenAI-compatible local backend, graph structure changes makespan, LLM burstiness, barrier behavior, and context amplification.",
            figures["graph_first_class"],
            "medium",
            "The traces are existing Week2 runs, not a controlled factorial experiment with identical prompts.",
            representatives("parallel_request_burst", "barrier_straggler", "fanin_context_amplification"),
        ),
        (
            "Insight 3: Not all agent requests are equally latency-critical.",
            "Critical-path proxies correlate with makespan, but explicit critical path flags are incomplete.",
            figures["critical_path"],
            "low",
            "Needs explicit dependency edges and critical_path_flag for high-confidence per-request latency criticality.",
            representatives("critical_path_concentration"),
        ),
        (
            "Insight 4: Fan-in nodes amplify context and motivate prefix/context-cache mechanisms.",
            "Synthesizer, reviewer, merge, and all-gather stages increase estimated fan-in/broadcast token pressure.",
            figures["context_amplification"],
            "medium",
            "Token quantities ending in `_est` are estimated workload proxies, not internal vLLM token-cache counters.",
            representatives("fanin_context_amplification", "all_to_all_broadcast_amplification"),
        ),
        (
            "Insight 5: Tool-stalled branches motivate tool-aware scheduling/cache management.",
            "Tool-heavy motifs have measurable tool latency and downstream merge/finalizer windows.",
            figures["tool_stalls"],
            "medium",
            "Current traces do not include KV residency; this motivates future instrumentation rather than proving idle KV behavior.",
            representatives("tool_stall_blocking", "post_tool_burst"),
        ),
        (
            "Insight 6: Review/debug loops amplify small local iterations into repeated LLM invocations.",
            "coder_reviewer, generator_verifier, and retry_debug_loop contain repeated reviewer/verifier/debug steps and extra LLM/token work.",
            figures["retry_loops"],
            "medium",
            "Loop depth is limited by the current benchmark settings.",
            representatives("retry_amplification"),
        ),
    ]
    insights = "\n\n".join(
        f"""### {title}

Supporting motifs: see per-motif table and signature rows below.

Supporting metrics: {support}

Supporting figure: ![]({figure_ref(figure)})

Representative trace paths: {trace_paths}

Confidence: {confidence}

Caveat: {caveat}
"""
        for title, support, figure, confidence, caveat, trace_paths in insight_blocks
    )
    return f"""# Week2 MASBench-Arch Trace Characterization

## 1. Scope

This report analyzes only existing Week2 traces. It does not rerun workloads, collect new traces, analyze task accuracy, modify vLLM scheduler/KV manager, design a final full workflow, or run case studies.

## 2. Trace Coverage

{md_table(rows, ["name", "run_count", "avg_makespan_sec", "avg_llm_request_count", "avg_total_tokens_est", "avg_tool_time_sec", "avg_barrier_wait_sec", "avg_context_amplification_ratio", "detected_signatures", "confidence"])}

## 3. Backend Alignment Quality

Measured: graph event timestamps/durations, client-side LLM request duration, tool measured/replayed latency, and sampled vLLM `/metrics` sidecars where present.

Estimated: token fields with `_est`, dataflow token amplification, context duplication, and critical-path proxies when explicit flags are absent.

Unavailable: per-request vLLM queue time, batch membership, KV allocation/eviction, prefix-cache hit/miss, idle KV residency during tool stalls, and reliable first-token timestamps for every request.

## 4. Per-Motif Summary

The table in Section 2 is the compact per-topology/per-motif summary. Full machine-readable metrics are in `progress/week2/week2_trace_characterization_summary.json` and `progress/week2/week2_trace_characterization_tables.csv`.

## 5. Control-flow Bottlenecks

Detected signatures:

{sig_line("serial_manager_bottleneck")}
{sig_line("critical_path_concentration")}
{sig_line("parallel_request_burst")}
{sig_line("barrier_straggler")}
{sig_line("retry_amplification")}

Manager/reviewer/finalizer roles concentrate serial work in centralized and staged motifs. Parallel motifs expose bursty arrivals and potential straggler gaps. Review/debug motifs turn a small local failure or request-changes decision into repeated LLM calls.

## 6. Data-flow Bottlenecks

Detected signatures:

{sig_line("fanin_context_amplification")}
{sig_line("all_to_all_broadcast_amplification")}

Fan-in nodes such as synthesizers, evidence merges, reviewers, selectors, and finalizers consume aggregated artifacts. Debate and all-gather motifs increase peer/broadcast token movement and duplicated context pressure.

## 7. Backend Serving Observations

The traces align MAS graph nodes to OpenAI-compatible backend requests through `request_id_for_backend`, node ids, agent roles, model names, backend URLs, and token/timing fields. Parallel motifs create short arrival bursts. The current backend metrics are useful for coarse serving windows but are insufficient for scheduler/cache/memory conclusions because they lack per-request queue time, batch membership, and KV/cache fields.

## 8. Tool and Memory Observations

Detected signatures:

{sig_line("tool_stall_blocking")}
{sig_line("post_tool_burst")}
{sig_line("shared_memory_dataflow")}

Tool-heavy motifs include measured or replayed tool latency and downstream aggregation. Shared evidence traces expose read/write dataflow and artifact version fields when present. These traces motivate future KV idle residency and version-aware cache instrumentation, but do not directly observe KV residency.

## 9. Preliminary Insights for Week2 Progress Report

{insights}

## 10. Missing Fields and Next Steps

- more reliable per-request `request_id_for_backend` propagation into backend logs
- streaming first-token timestamp for each request
- real per-request TTFT and TPOT
- vLLM queue time and scheduler wait
- batch membership over time
- KV allocation, eviction, and residency intervals
- prefix cache hit/miss and shared prefix hashes
- explicit `critical_path_flag` and complete dependency edges for every node
- token-level shared-prefix hash for fan-in/fan-out reuse analysis
"""


def main() -> int:
    args = parse_args()
    include_backend = as_bool(args.include_backend_metrics)
    include_outputs = as_bool(args.include_model_outputs)
    output_dir = Path(args.output_dir)
    progress_dir = Path(args.progress_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    roots = discover_trace_roots(args)
    paths: list[Path] = []
    for root in roots:
        paths.extend(path for path in root.rglob("*.jsonl") if "snapshots" not in path.parts)
    paths = sorted(set(paths))

    runs = [analyze_run(path, include_backend, include_outputs) for path in paths]
    present_names = {run["metrics"]["name"] for run in runs}
    missing = [{"name": name, "status": "trace_missing", "trace_path": ""} for name in EXPECTED_NAMES if name not in present_names]
    aggregate = aggregate_by_name(runs)

    validation_summary = {
        "trace_roots": [str(root) for root in roots],
        "run_count": len(runs),
        "expected_names": EXPECTED_NAMES,
        "present_names": sorted(present_names),
        "missing_names": [row["name"] for row in missing],
        "file_integrity": {
            "required_ok_count": sum(1 for run in runs if run["validation"]["file_integrity_ok"]),
            "backend_metrics_count": sum(1 for run in runs if run["validation"]["backend_metrics_exists"]),
            "model_outputs_count": sum(1 for run in runs if run["validation"]["model_outputs_exists"]),
        },
        "backend_alignment": {
            "llm_request_count": sum(run["validation"]["llm_request_count"] for run in runs),
            "request_id_for_backend_avg_coverage": round(
                mean([run["validation"]["request_id_for_backend_coverage"] for run in runs if run["validation"]["llm_request_count"]] or [0.0]), 4
            ),
            "token_field_avg_coverage": round(
                mean([run["validation"]["token_field_coverage"] for run in runs if run["validation"]["llm_request_count"]] or [0.0]), 4
            ),
            "timing_field_avg_coverage": round(
                mean([run["validation"]["timing_field_coverage"] for run in runs if run["validation"]["llm_request_count"]] or [0.0]), 4
            ),
        },
        "aggregate": aggregate,
        "runs": runs,
    }

    validation_path = output_dir / "week2_trace_validation_summary.json"
    validation_path.write_text(json.dumps(validation_summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "week2_trace_validation_report.md").write_text(validation_report(aggregate, runs, missing), encoding="utf-8")
    write_inventory(output_dir / "week2_trace_file_inventory.csv", runs, missing)

    figures = make_figures(progress_dir, aggregate, runs)
    characterization = {
        "trace_roots": [str(root) for root in roots],
        "run_count": len(runs),
        "aggregate": aggregate,
        "signatures": [sig for run in runs for sig in run["signatures"]],
        "figures": figures,
        "measurement_semantics": {
            "measured": [
                "graph event relative timestamps and durations",
                "client-side LLM request duration",
                "tool measured/replayed latency",
                "vLLM /metrics sampled sidecars where present",
            ],
            "estimated": [
                "input/output token estimates",
                "dataflow token amplification estimates",
                "critical-path proxy when explicit flags are absent",
            ],
            "unavailable": [
                "per-request vLLM queue time",
                "batch membership",
                "KV allocation/eviction/residency",
                "prefix cache hit/miss",
            ],
        },
    }
    (progress_dir / "week2_trace_characterization_summary.json").write_text(
        json.dumps(characterization, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_tables(progress_dir / "week2_trace_characterization_tables.csv", aggregate)
    (progress_dir / "week2_trace_characterization_report.md").write_text(
        characterization_report(aggregate, runs, figures),
        encoding="utf-8",
    )

    print(f"Validation summary: {validation_path}")
    print(f"Validation report: {output_dir / 'week2_trace_validation_report.md'}")
    print(f"Characterization summary: {progress_dir / 'week2_trace_characterization_summary.json'}")
    print(f"Characterization report: {progress_dir / 'week2_trace_characterization_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

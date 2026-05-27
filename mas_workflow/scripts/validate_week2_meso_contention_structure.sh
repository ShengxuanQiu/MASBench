#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import json
import sys
from pathlib import Path

from mas_workflow.app.motifs import build_workflow
from mas_workflow.app.topologies.base import TopologyConfig
from mas_workflow.app.tracing import now_ts

out_dir = Path("traces/week2_structure_validation")
out_dir.mkdir(parents=True, exist_ok=True)

config = TopologyConfig(
    topology_name="tool_resume_contention_meso",
    run_id=now_ts(),
    task_id="week2_structure_validation",
    instance_id="tool_resume_contention_meso_structure",
    query="Validate a composed MAS workload with delayed tool resume and critical reviewer/finalizer stages.",
    mode="motif",
    motif_name="tool_resume_contention_meso",
    llm_mode="mock",
    tool_mode="synthetic",
    search_provider="synthetic",
    trace_dir=out_dir / "raw",
    export_trace_views=False,
    max_concurrent_llm_calls=8,
    tool_branch_width=2,
    controlled_tool_delay_sec=2.5,
    resume_phase_policy="overlap_reviewer",
    critical_stage_marker="reviewer",
    background_resume_enabled=True,
    contention_labeling=True,
)

summary = build_workflow(config).run()
trace_path = Path(summary["trace_path"])
events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]

def has_event(predicate):
    return any(predicate(event) for event in events)

llm_events = [event for event in events if event.get("event_type") == "llm_request_end"]
tool_events = [event for event in events if str(event.get("event_type", "")).startswith("tool_")]
edge_events = [event for event in events if event.get("event_type") == "edge_dataflow"]
plan_events = [event for event in events if event.get("event_type") == "meso_trace_plan"]

checks = {
    "can_import_and_build": True,
    "critical_path_branch": has_event(lambda e: e.get("workload_role") == "critical_path"),
    "background_tool_stalled_branch": has_event(lambda e: e.get("workload_role") == "background_tool_branch" and e.get("tool_stalled") is True),
    "resume_processor_node": has_event(lambda e: "resume_evidence_processor" in str(e.get("node_id", "")) and e.get("resume_after_tool") is True),
    "critical_reviewer_or_finalizer": has_event(lambda e: e.get("critical_request_marker") in {"reviewer", "finalizer"}),
    "merge_fanin_edge": has_event(lambda e: e.get("dst_node") in {"evidence_merge", "finalizer"} and e.get("transfer_type") == "aggregation"),
    "metadata_criticality": has_event(lambda e: e.get("criticality") in {"critical", "background", "merge"}),
    "metadata_critical_path_candidate": has_event(lambda e: e.get("critical_path_candidate") is True),
    "metadata_tool_stalled": has_event(lambda e: e.get("tool_stalled") is True),
    "metadata_resume_after_tool": has_event(lambda e: e.get("resume_after_tool") is True),
    "metadata_resume_phase_policy": has_event(lambda e: e.get("resume_phase_policy") == "overlap_reviewer"),
    "metadata_expected_overlap_target": has_event(lambda e: e.get("expected_overlap_target") == "reviewer"),
    "dry_run_critical_candidate": any(e.get("critical_request_marker") in {"reviewer", "finalizer"} for e in llm_events),
    "dry_run_background_resume_candidate": any(e.get("background_resume_request") is True for e in llm_events),
    "dry_run_expected_overlap_window": has_event(lambda e: float(e.get("expected_overlap_window_sec") or 0.0) > 0.0),
    "composed_from_motifs_nonempty": has_event(lambda e: bool(e.get("composed_from_motifs"))),
    "controlled_delay_tool_event": any(e.get("tool_name") == "controlled_delay_tool" for e in tool_events),
}

validation = {
    "workload": "tool_resume_contention_meso",
    "status": "pass" if all(checks.values()) else "fail",
    "checks": checks,
    "summary": {
        "trace_path": str(trace_path),
        "llm_event_count": len(llm_events),
        "tool_event_count": len(tool_events),
        "edge_event_count": len(edge_events),
        "plan_event_count": len(plan_events),
        "critical_candidates": [e.get("node_id") for e in llm_events if e.get("critical_request_marker") in {"reviewer", "finalizer"}],
        "background_resume_candidates": [e.get("node_id") for e in llm_events if e.get("background_resume_request") is True],
        "expected_overlap_targets": sorted({str(e.get("expected_overlap_target")) for e in events if e.get("expected_overlap_target")}),
        "composed_from_motifs": next((e.get("composed_from_motifs") for e in events if e.get("composed_from_motifs")), []),
        "composed_from_topologies": next((e.get("composed_from_topologies") for e in events if e.get("composed_from_topologies")), []),
    },
}

plan = {
    "workload": "tool_resume_contention_meso",
    "trace_path": str(trace_path),
    "composed_from_motifs": validation["summary"]["composed_from_motifs"],
    "composed_from_topologies": validation["summary"]["composed_from_topologies"],
    "nodes": [
        {
            "node_id": e.get("node_id"),
            "event_type": e.get("event_type"),
            "agent_role": e.get("agent_role"),
            "workload_role": e.get("workload_role"),
            "criticality": e.get("criticality"),
            "critical_path_candidate": e.get("critical_path_candidate"),
            "critical_stage": e.get("critical_stage"),
            "tool_stalled": e.get("tool_stalled"),
            "resume_after_tool": e.get("resume_after_tool"),
            "expected_overlap_target": e.get("expected_overlap_target"),
            "expected_overlap_window_sec": e.get("expected_overlap_window_sec"),
            "parent_node_ids": e.get("parent_node_ids") or e.get("parents"),
            "dependency_edges": e.get("dependency_edges"),
        }
        for e in events
        if e.get("event_type") in {"llm_request_end", "tool_controlled_delay", "meso_trace_plan", "barrier"}
    ],
    "edges": [
        {
            "src_node": e.get("src_node"),
            "dst_node": e.get("dst_node"),
            "artifact_type": e.get("artifact_type"),
            "transfer_type": e.get("transfer_type"),
            "parallel_group": e.get("parallel_group"),
        }
        for e in edge_events
    ],
}

(out_dir / "tool_resume_contention_meso_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
(out_dir / "tool_resume_contention_meso_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

rows = "\n".join(f"| {name} | {'pass' if ok else 'fail'} |" for name, ok in checks.items())
md = f"""# tool_resume_contention_meso Structure Validation

Status: **{validation['status']}**

Trace: `{trace_path}`

| Check | Result |
| --- | --- |
{rows}

Critical candidates: {', '.join(validation['summary']['critical_candidates'])}

Background resume candidates: {', '.join(validation['summary']['background_resume_candidates'])}

Expected overlap targets: {', '.join(validation['summary']['expected_overlap_targets'])}
"""
(out_dir / "tool_resume_contention_meso_validation.md").write_text(md, encoding="utf-8")

print(json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True))
if validation["status"] != "pass":
    sys.exit(1)
PY

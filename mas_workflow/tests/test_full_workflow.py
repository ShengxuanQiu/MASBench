from __future__ import annotations

import json
from pathlib import Path

from app.full_workflows import build_workflow
from app.topologies.base import TopologyConfig


def test_issue_to_verified_patch_dry_run_trace(tmp_path: Path) -> None:
    config = TopologyConfig(
        topology_name="issue_to_verified_patch",
        run_id="test_full_workflow",
        task_id="manual_test",
        instance_id="manual_test",
        query="Fix a small parser regression.",
        num_agents=2,
        max_retries=1,
        llm_mode="mock",
        tool_mode="synthetic",
        search_provider="local_repo",
        repo_path=Path.cwd(),
        trace_dir=tmp_path,
        export_trace_views=False,
        mode="full",
        workload_name="issue_to_verified_patch",
    )
    config.extra["dry_run_patch"] = True
    config.extra["dry_run_tests"] = True
    config.extra["enable_case_study_resume_window"] = True
    config.extra["critical_prefill_budget_tokens"] = 512
    config.extra["case_study_critical_output_tokens"] = 512
    workflow = build_workflow(config)
    summary = workflow.run()

    trace_path = Path(summary["trace_path"])
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    event_types = {event["event_type"] for event in events}
    assert "full_workflow_start" in event_types
    assert "stage_summary" in event_types
    assert "candidate_set_summary" in event_types
    assert "patch_selection_summary" in event_types
    assert "tool_resume_burst_summary" in event_types
    assert "debug_loop_prefix_summary" in event_types
    assert "workflow_result_ready" in event_types
    resume_ready = [
        event
        for event in events
        if event.get("event_type") == "llm_request_ready" and event.get("tool_resumed")
    ]
    assert {event["node_id"] for event in resume_ready} == {
        "evidence_window_auditor_1_resume",
        "evidence_window_auditor_2_resume",
        "diagnosis_window_auditor_1_resume",
        "diagnosis_window_auditor_2_resume",
        "regression_auditor_1_resume",
        "regression_auditor_2_resume",
    }
    eligible_windows = [event for event in events if event.get("event_type") == "case_study_eligible_window"]
    assert {event["node_id"] for event in eligible_windows} == {
        "evidence_window",
        "diagnosis_window",
        "final_window",
    }
    assert summary["mode"] == "full"
    assert summary["full_workflow_name"] == "issue_to_verified_patch"

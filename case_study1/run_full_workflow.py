from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from mas_workflow.app.full_workflows import build_workflow
from mas_workflow.app.topologies.base import TopologyConfig


ROOT = Path(__file__).resolve().parents[1]
CS_DIR = Path(__file__).resolve().parent
TASK_TRACE = (
    ROOT
    / "mas_workflow/traces/week6_real/full_workflow/issue_to_verified_patch"
    / "astropy__astropy-12907/20260808_210351_015313.jsonl"
)


def fixed_task() -> dict[str, object]:
    for line in TASK_TRACE.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        if event.get("event_type") == "workflow_start":
            return dict(event["extra"]["config"]["extra"]["swebench"])
    raise RuntimeError("fixed SWE-bench task metadata not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["default_vllm", "critical_path_aware"], required=True)
    args = parser.parse_args()
    task = fixed_task()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    config = TopologyConfig(
        topology_name="issue_to_verified_patch",
        run_id=run_id,
        task_id=str(task["instance_id"]),
        instance_id=str(task["instance_id"]),
        query=str(task["problem_statement"]),
        max_rounds=2,
        # The full workflow keeps its two core evidence agents (repo + tests).
        # Adding history/api agents can return a single >200 KB raw web page,
        # which cannot fit the model context without forbidden text slicing.
        num_agents=2,
        max_retries=0,
        llm_mode="openai_compatible",
        tool_mode="live",
        max_concurrent_llm_calls=8,
        backend_base_url="http://127.0.0.1:8101/v1",
        model="qwen3-8b-case-study1",
        max_output_tokens=128,
        task_source="swebench_local",
        trace_dir=CS_DIR / "artifacts/paper_configs/full_workflow_multiwindow_frontier_v2",
        repo_path=ROOT / "swebench_repos/astropy__astropy-12907",
        record_tool_results=True,
        search_provider="tavily",
        force_live_search_test=True,
        allow_synthetic_tools=False,
        trace_level="arch",
        export_trace_views=False,
        admission_policy=args.policy,
        max_defer_sec=90.0,
        mode="full",
        workload_name="issue_to_verified_patch",
    )
    config.extra.update(
        {
            "swebench": task,
            "test_command": "",
            "dry_run_patch": True,
            "dry_run_tests": True,
            "paper_configuration": "Full workflow",
            "physical_gpu_id": 4,
            "enable_case_study_resume_window": True,
            "critical_prefill_budget_tokens": 512,
            "case_study_critical_output_tokens": 512,
            "case_study_resume_agents_per_window": 3,
            "protect_case_study_structural_frontier": True,
            "search_options": {
                # Keep the provider payload complete while fitting Qwen3-8B's
                # 40,960-token context: one full raw result, never text slicing.
                "max_results": 1,
                "include_answer": True,
                "include_raw_content": True,
                "search_depth": "advanced",
            },
        }
    )
    summary = build_workflow(config).run()
    print(json.dumps({"run_id": run_id, "policy": args.policy, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

from app.topologies import TopologyConfig, build_motif, build_workflow


def _config(tmp_path: Path, topology: str) -> TopologyConfig:
    return TopologyConfig(
        topology_name=topology,
        run_id=f"test_{topology}",
        task_id=f"manual_{topology}",
        instance_id=f"manual_{topology}",
        query="验证 orchestrator 从更大的 agent pool 中按轮次选择不同专家",
        llm_mode="mock",
        tool_mode="synthetic",
        search_provider="synthetic",
        latency_profile="none",
        trace_dir=tmp_path,
        max_rounds=4,
        peer_rounds=1,
        agent_pool_size=10,
        min_selected_agents=1,
        max_selected_agents=4,
        max_concurrent_llm_calls=4,
        export_trace_views=False,
    )


def test_centralized_workflow_selects_dynamic_agent_subsets(tmp_path: Path) -> None:
    workflow = build_workflow(_config(tmp_path, "centralized"))
    summary = workflow.run()
    decisions = [e for e in workflow.trace.events if e.get("event_type") == "manager_decision"]
    continuing = [e for e in decisions if e.get("decision") == "continue"]

    assert decisions
    assert summary["available_agent_count"] == 10
    assert all(1 <= int(e["selected_agent_count"]) <= 4 for e in continuing)
    assert len({tuple(e.get("selected_agent_roles") or []) for e in continuing}) > 1
    assert summary["max_parallel_width"] <= 4


def test_hybrid_motif_uses_same_dynamic_orchestrator_interface(tmp_path: Path) -> None:
    motif = build_motif(_config(tmp_path, "hybrid"))
    summary = motif.invoke({"query": "验证 motif 模式也支持动态专家选择"})
    decisions = [e for e in motif.trace.events if e.get("event_type") == "manager_decision"]
    selected_counts = [int(e.get("selected_agent_count") or 0) for e in decisions]

    assert motif.trace.topology_role == "motif"
    assert summary["available_agent_count"] == 10
    assert len(selected_counts) >= 2
    assert max(selected_counts) <= 4
    assert len(set(selected_counts)) > 1
    assert any(e.get("peer_round_id") == 1 for e in motif.trace.events)

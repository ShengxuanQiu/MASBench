"""Registry for composite MAS motifs."""

from __future__ import annotations

from pathlib import Path

from ..llm_backends import build_llm_backend
from ..search_providers import build_search_provider
from ..topologies.base import TopologyConfig
from ..topologies.registry import build_trace_context
from .all_gather_round import build_motif as all_gather_round_motif
from .all_gather_round import build_workflow as all_gather_round_workflow
from .coder_reviewer import build_motif as coder_reviewer_motif
from .coder_reviewer import build_workflow as coder_reviewer_workflow
from .debate_reviewer import build_motif as debate_reviewer_motif
from .debate_reviewer import build_workflow as debate_reviewer_workflow
from .evidence_collection import build_motif as evidence_collection_motif
from .evidence_collection import build_workflow as evidence_collection_workflow
from .generator_verifier import build_motif as generator_verifier_motif
from .generator_verifier import build_workflow as generator_verifier_workflow
from .multi_coder_branch import build_motif as multi_coder_branch_motif
from .multi_coder_branch import build_workflow as multi_coder_branch_workflow
from .planner_executor import build_motif as planner_executor_motif
from .planner_executor import build_workflow as planner_executor_workflow
from .researcher_synthesizer import build_motif as researcher_synthesizer_motif
from .researcher_synthesizer import build_workflow as researcher_synthesizer_workflow
from .retry_debug_loop import build_motif as retry_debug_loop_motif
from .retry_debug_loop import build_workflow as retry_debug_loop_workflow
from .router_handoff import build_motif as router_handoff_motif
from .router_handoff import build_workflow as router_handoff_workflow
from .shared_evidence_store import build_motif as shared_evidence_store_motif
from .shared_evidence_store import build_workflow as shared_evidence_store_workflow
from .tool_specialist_team import build_motif as tool_specialist_team_motif
from .tool_specialist_team import build_workflow as tool_specialist_team_workflow


MOTIF_NAMES = [
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

WORKFLOW_BUILDERS = {
    "planner_executor": planner_executor_workflow,
    "evidence_collection": evidence_collection_workflow,
    "researcher_synthesizer": researcher_synthesizer_workflow,
    "generator_verifier": generator_verifier_workflow,
    "coder_reviewer": coder_reviewer_workflow,
    "multi_coder_branch": multi_coder_branch_workflow,
    "debate_reviewer": debate_reviewer_workflow,
    "tool_specialist_team": tool_specialist_team_workflow,
    "all_gather_round": all_gather_round_workflow,
    "shared_evidence_store": shared_evidence_store_workflow,
    "retry_debug_loop": retry_debug_loop_workflow,
    "router_handoff": router_handoff_workflow,
}

MOTIF_BUILDERS = {
    "planner_executor": planner_executor_motif,
    "evidence_collection": evidence_collection_motif,
    "researcher_synthesizer": researcher_synthesizer_motif,
    "generator_verifier": generator_verifier_motif,
    "coder_reviewer": coder_reviewer_motif,
    "multi_coder_branch": multi_coder_branch_motif,
    "debate_reviewer": debate_reviewer_motif,
    "tool_specialist_team": tool_specialist_team_motif,
    "all_gather_round": all_gather_round_motif,
    "shared_evidence_store": shared_evidence_store_motif,
    "retry_debug_loop": retry_debug_loop_motif,
    "router_handoff": router_handoff_motif,
}


def _deps(config: TopologyConfig, topology_role: str):
    config.mode = "motif"
    config.motif_name = config.topology_name
    trace = build_trace_context(config, topology_role=topology_role)
    llm = build_llm_backend(
        config.llm_mode,
        model=config.model,
        backend_base_url=config.backend_base_url,
        max_output_tokens=config.max_output_tokens,
    )
    provider = build_search_provider(
        provider_name=config.search_provider,
        tool_mode=config.tool_mode,
        replay_snapshot_dir=config.replay_snapshot_dir,
        latency_profile=config.latency_profile,
        latency_scale=config.latency_scale,
        random_seed=config.random_seed,
        repo_path=config.repo_path,
        force_live_search_test=config.force_live_search_test,
        allow_synthetic_fallback=config.allow_synthetic_tools,
    )
    return {"llm": llm, "search_provider": provider, "trace": trace}


def build_workflow(config: TopologyConfig):
    name = config.topology_name
    if name not in WORKFLOW_BUILDERS:
        raise ValueError(f"Unknown motif: {name}")
    return WORKFLOW_BUILDERS[name](config, **_deps(config, "workflow"))


def build_motif(config: TopologyConfig):
    name = config.topology_name
    if name not in MOTIF_BUILDERS:
        raise ValueError(f"Unknown motif: {name}")
    return MOTIF_BUILDERS[name](config, **_deps(config, "motif"))

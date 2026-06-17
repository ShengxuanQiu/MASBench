"""Registry for end-to-end full workflows."""

from __future__ import annotations

from ..llm_backends import build_llm_backend
from ..search_providers import build_search_provider
from ..topologies.base import TopologyConfig
from ..topologies.registry import build_trace_context
from .issue_to_verified_patch import build_workflow as issue_to_verified_patch_workflow


FULL_WORKFLOW_NAMES = ["issue_to_verified_patch"]

WORKFLOW_BUILDERS = {
    "issue_to_verified_patch": issue_to_verified_patch_workflow,
}


def _deps(config: TopologyConfig, topology_role: str):
    config.mode = "full"
    config.motif_name = ""
    trace = build_trace_context(config, topology_role=topology_role)
    trace.mode = "full"
    trace.motif_name = ""
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
        raise ValueError(f"Unknown full workflow: {name}")
    return WORKFLOW_BUILDERS[name](config, **_deps(config, "full_workflow"))

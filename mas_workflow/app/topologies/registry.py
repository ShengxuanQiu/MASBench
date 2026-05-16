"""Topology registry for workflow and motif construction."""

from __future__ import annotations

from pathlib import Path

from ..llm_backends import build_llm_backend
from ..search_providers import build_search_provider
from ..tracing import TraceContext
from .base import TopologyConfig
from .centralized import build_motif as centralized_motif
from .centralized import build_workflow as centralized_workflow
from .decentralized_debate import build_motif as decentralized_motif
from .decentralized_debate import build_workflow as decentralized_workflow
from .hybrid import build_motif as hybrid_motif
from .hybrid import build_workflow as hybrid_workflow
from .independent import build_motif as independent_motif
from .independent import build_workflow as independent_workflow
from .single_agent import build_motif as single_motif
from .single_agent import build_workflow as single_workflow


WORKFLOW_BUILDERS = {
    "single": single_workflow,
    "independent": independent_workflow,
    "centralized": centralized_workflow,
    "decentralized": decentralized_workflow,
    "hybrid": hybrid_workflow,
}

MOTIF_BUILDERS = {
    "single": single_motif,
    "independent": independent_motif,
    "centralized": centralized_motif,
    "decentralized": decentralized_motif,
    "hybrid": hybrid_motif,
}


def build_trace_context(config: TopologyConfig, *, topology_role: str = "workflow") -> TraceContext:
    trace_dir = Path(config.trace_dir)
    topology_dir = trace_dir / config.topology_name
    trace_path = topology_dir / f"{config.instance_id}_{config.run_id}.jsonl"
    summary_path = topology_dir / f"{config.instance_id}_{config.run_id}_summary.json"
    return TraceContext(
        run_id=config.run_id,
        topology=config.topology_name,
        topology_role=topology_role,
        instance_id=config.instance_id,
        task_source=config.task_source,
        workflow_id=f"{config.topology_name}_{config.run_id}",
        trace_path=trace_path,
        summary_path=summary_path,
        random_seed=config.random_seed,
    )


def _deps(config: TopologyConfig, topology_role: str):
    trace = build_trace_context(config, topology_role=topology_role)
    llm = build_llm_backend(config.llm_mode, model=config.model, backend_base_url=config.backend_base_url)
    provider = build_search_provider(
        provider_name=config.search_provider,
        tool_mode=config.tool_mode,
        replay_snapshot_dir=config.replay_snapshot_dir,
        latency_profile=config.latency_profile,
        latency_scale=config.latency_scale,
        random_seed=config.random_seed,
        repo_path=config.repo_path,
        force_live_search_test=config.force_live_search_test,
    )
    return {"llm": llm, "search_provider": provider, "trace": trace}


def build_workflow(config: TopologyConfig):
    name = config.topology_name
    if name not in WORKFLOW_BUILDERS:
        raise ValueError(f"Unknown topology: {name}")
    return WORKFLOW_BUILDERS[name](config, **_deps(config, "workflow"))


def build_motif(config: TopologyConfig):
    name = config.topology_name
    if name not in MOTIF_BUILDERS:
        raise ValueError(f"Unknown topology: {name}")
    return MOTIF_BUILDERS[name](config, **_deps(config, "motif"))

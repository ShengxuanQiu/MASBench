"""Primary family registry and explicit historical experiment compatibility."""
from __future__ import annotations

import warnings
from copy import deepcopy

from ..llm_backends import build_llm_backend
from ..search_providers import build_search_provider
from ..runtime_factory import build_trace_context
from .families import FamilyWorkload, MOTIF_NAMES
from .presets import PRESETS, preset

TEMPLATE_NAMES = {"Spawn":"dispatch_execute", "spawn":"dispatch_execute",
                  "ForkJoin":"parallel_aggregate", "fork_join":"parallel_aggregate",
                  "RefinementLoop":"evaluate_refine", "refinement_loop":"evaluate_refine",
                  "Debate":"peer_exchange", "debate":"peer_exchange"}

LEGACY_WORKLOAD_NAMES = [
    "shared_evidence_store", "tool_resume_contention_meso",
    "hierarchical_synthesis_pressure_meso", "debate_allgather_pressure_meso",
    "retry_debug_pressure_meso", "shared_memory_fanin_meso",
]
WORKLOAD_NAMES = MOTIF_NAMES + list(TEMPLATE_NAMES) + list(PRESETS) + LEGACY_WORKLOAD_NAMES


def build_workflow(config):
    if config.mode == "legacy_motif" or config.topology_name in LEGACY_WORKLOAD_NAMES:
        if "workload_spec" in config.extra:
            raise ValueError("A family workload specification cannot be combined with a historical workload")
        from .legacy_registry import build_workflow as legacy
        warnings.warn("Executing historical motif implementation; not a canonical motif family.", FutureWarning)
        return legacy(config)
    spec = deepcopy(config.extra.get("workload_spec"))
    name = config.topology_name
    if spec is None:
        if name in PRESETS:
            warnings.warn(f"{name} now uses a family task binding. Use --mode legacy_motif for the historical trace semantics.", FutureWarning)
            spec = preset(name)
        elif name in MOTIF_NAMES:
            spec = {"family": name}
        elif name in TEMPLATE_NAMES:
            spec = {"family": TEMPLATE_NAMES[name]}
        else:
            raise ValueError(f"Unknown motif family or preset: {name}")
    config.mode = "motif"
    if config.agent_execution != "fixed":
        raise ValueError("Family bindings currently support fixed actions; use legacy_motif for historical ReAct workloads")
    config.motif_name = "composition" if "stages" in spec else spec["family"]
    if "workload_spec" in config.extra:
        config.topology_name = config.motif_name
        config.workload_name = config.motif_name
    trace = build_trace_context(config)
    llm = build_llm_backend(config.llm_mode, model=config.model,
                            backend_base_url=config.backend_base_url, max_output_tokens=config.max_output_tokens)
    provider = build_search_provider(
        provider_name=config.search_provider, tool_mode=config.tool_mode,
        replay_snapshot_dir=config.replay_snapshot_dir, latency_profile=config.latency_profile,
        latency_scale=config.latency_scale, random_seed=config.random_seed, repo_path=config.repo_path,
        force_live_search_test=config.force_live_search_test, allow_synthetic_fallback=config.allow_synthetic_tools,
    )
    return FamilyWorkload(config, spec=spec, llm=llm, search_provider=provider, trace=trace)


build_motif = build_workflow

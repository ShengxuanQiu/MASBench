from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import PlannerExecutorMotif


def build_workflow(config: TopologyConfig, **deps) -> PlannerExecutorMotif:
    return PlannerExecutorMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> PlannerExecutorMotif:
    return PlannerExecutorMotif(config, **deps)

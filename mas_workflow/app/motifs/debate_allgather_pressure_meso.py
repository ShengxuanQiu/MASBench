"""Week3 meso workload: debate and all-gather pressure."""

from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import DebateAllGatherPressureMesoMotif


def build_workflow(config: TopologyConfig, **deps) -> DebateAllGatherPressureMesoMotif:
    return DebateAllGatherPressureMesoMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> DebateAllGatherPressureMesoMotif:
    return DebateAllGatherPressureMesoMotif(config, **deps)

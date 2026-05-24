from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import AllGatherRoundMotif


def build_workflow(config: TopologyConfig, **deps) -> AllGatherRoundMotif:
    return AllGatherRoundMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> AllGatherRoundMotif:
    return AllGatherRoundMotif(config, **deps)

from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import RouterHandoffMotif


def build_workflow(config: TopologyConfig, **deps) -> RouterHandoffMotif:
    return RouterHandoffMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> RouterHandoffMotif:
    return RouterHandoffMotif(config, **deps)

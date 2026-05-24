from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import MultiCoderBranchMotif


def build_workflow(config: TopologyConfig, **deps) -> MultiCoderBranchMotif:
    return MultiCoderBranchMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> MultiCoderBranchMotif:
    return MultiCoderBranchMotif(config, **deps)

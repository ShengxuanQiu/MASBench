"""Week3 meso workload: hierarchical synthesis pressure."""

from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import HierarchicalSynthesisPressureMesoMotif


def build_workflow(config: TopologyConfig, **deps) -> HierarchicalSynthesisPressureMesoMotif:
    return HierarchicalSynthesisPressureMesoMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> HierarchicalSynthesisPressureMesoMotif:
    return HierarchicalSynthesisPressureMesoMotif(config, **deps)

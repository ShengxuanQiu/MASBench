from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import CriticalPathToolResumeContentionMesoMotif


def build_workflow(config: TopologyConfig, **deps) -> CriticalPathToolResumeContentionMesoMotif:
    return CriticalPathToolResumeContentionMesoMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> CriticalPathToolResumeContentionMesoMotif:
    return CriticalPathToolResumeContentionMesoMotif(config, **deps)

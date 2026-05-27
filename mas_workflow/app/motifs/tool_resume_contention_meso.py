from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import ToolResumeContentionMesoMotif


def build_workflow(config: TopologyConfig, **deps) -> ToolResumeContentionMesoMotif:
    return ToolResumeContentionMesoMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> ToolResumeContentionMesoMotif:
    return ToolResumeContentionMesoMotif(config, **deps)

from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import ResearcherSynthesizerMotif


def build_workflow(config: TopologyConfig, **deps) -> ResearcherSynthesizerMotif:
    return ResearcherSynthesizerMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> ResearcherSynthesizerMotif:
    return ResearcherSynthesizerMotif(config, **deps)

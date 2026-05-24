from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import ToolSpecialistTeamMotif


def build_workflow(config: TopologyConfig, **deps) -> ToolSpecialistTeamMotif:
    return ToolSpecialistTeamMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> ToolSpecialistTeamMotif:
    return ToolSpecialistTeamMotif(config, **deps)

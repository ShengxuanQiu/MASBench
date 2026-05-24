from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import DebateReviewerMotif


def build_workflow(config: TopologyConfig, **deps) -> DebateReviewerMotif:
    return DebateReviewerMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> DebateReviewerMotif:
    return DebateReviewerMotif(config, **deps)

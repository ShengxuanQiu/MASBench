from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import CoderReviewerMotif


def build_workflow(config: TopologyConfig, **deps) -> CoderReviewerMotif:
    return CoderReviewerMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> CoderReviewerMotif:
    return CoderReviewerMotif(config, **deps)

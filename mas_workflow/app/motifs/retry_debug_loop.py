from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import RetryDebugLoopMotif


def build_workflow(config: TopologyConfig, **deps) -> RetryDebugLoopMotif:
    return RetryDebugLoopMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> RetryDebugLoopMotif:
    return RetryDebugLoopMotif(config, **deps)

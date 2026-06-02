"""Week3 meso workload: retry/debug pressure."""

from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import RetryDebugPressureMesoMotif


def build_workflow(config: TopologyConfig, **deps) -> RetryDebugPressureMesoMotif:
    return RetryDebugPressureMesoMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> RetryDebugPressureMesoMotif:
    return RetryDebugPressureMesoMotif(config, **deps)

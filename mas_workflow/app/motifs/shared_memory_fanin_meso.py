"""Week3 meso workload: shared memory fan-in pressure."""

from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import SharedMemoryFaninMesoMotif


def build_workflow(config: TopologyConfig, **deps) -> SharedMemoryFaninMesoMotif:
    return SharedMemoryFaninMesoMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> SharedMemoryFaninMesoMotif:
    return SharedMemoryFaninMesoMotif(config, **deps)

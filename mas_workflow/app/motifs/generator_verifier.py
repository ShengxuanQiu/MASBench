from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import GeneratorVerifierMotif


def build_workflow(config: TopologyConfig, **deps) -> GeneratorVerifierMotif:
    return GeneratorVerifierMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> GeneratorVerifierMotif:
    return GeneratorVerifierMotif(config, **deps)

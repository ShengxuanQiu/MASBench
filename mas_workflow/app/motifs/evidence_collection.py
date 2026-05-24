from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import EvidenceCollectionMotif


def build_workflow(config: TopologyConfig, **deps) -> EvidenceCollectionMotif:
    return EvidenceCollectionMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> EvidenceCollectionMotif:
    return EvidenceCollectionMotif(config, **deps)

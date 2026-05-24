from __future__ import annotations

from ..topologies.base import TopologyConfig
from .base import SharedEvidenceStoreMotif


def build_workflow(config: TopologyConfig, **deps) -> SharedEvidenceStoreMotif:
    return SharedEvidenceStoreMotif(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> SharedEvidenceStoreMotif:
    return SharedEvidenceStoreMotif(config, **deps)

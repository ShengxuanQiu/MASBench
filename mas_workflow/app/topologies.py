"""Import compatibility for historical ``app.topologies`` callers.

Implementations live in legacy_topologies. New workloads use runtime and motifs.
The package search path keeps old submodule imports working without duplicating
the implementation directory.
"""
from . import legacy_topologies as _legacy
from .legacy_topologies import TopologyConfig, build_motif, build_workflow

__path__ = _legacy.__path__
__all__ = ["TopologyConfig", "build_motif", "build_workflow"]

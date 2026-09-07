"""Compatibility imports for the original collaboration-regime runners.

New motif implementations use app.runtime directly. These names remain for
historical experiments and downstream imports; they do not define graph shapes.
"""
from ..runtime import (
    WorkloadConfig as TopologyConfig,
    WorkloadRuntime as BaseTopology,
    RunnableWorkload as RunnableTopology,
    parse_json_maybe,
)

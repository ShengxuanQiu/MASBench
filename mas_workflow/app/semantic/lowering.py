"""Optional provenance-preserving bridge to the official Chakra toolchain."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

from .model import SemanticTrace


class ChakraUnavailable(RuntimeError):
    pass


def chakra_available() -> bool:
    return bool(importlib.util.find_spec("chakra"))


def lower_with_official_converter(trace: SemanticTrace, instrumented_execution: str | Path,
                                  output_trace: str | Path, *, converter_command: list[str],
                                  converter_version: str, model_or_cost_version: str) -> dict[str, Any]:
    """Run an explicitly supplied official converter on real instrumented execution.

    The instrumented input must carry `masbench_semantic_op_id`, task and stage
    annotations. No source request latency is synthesized into COMP nodes.
    """
    source = Path(instrumented_execution)
    annotations = json.loads(source.read_text(encoding="utf-8"))
    records = annotations.get("nodes", [])
    semantic_ids = {x.operation_id for x in trace.operations}
    lowered_ids = {str(x["masbench_semantic_op_id"]) for x in records if x.get("masbench_semantic_op_id")}
    unknown = lowered_ids - semantic_ids
    if unknown:
        raise ValueError(f"Instrumented trace references unknown semantic operations: {sorted(unknown)}")
    for node in records:
        if node.get("masbench_semantic_op_id") and (not node.get("masbench_task_id") or not node.get("masbench_stage_instance_id")):
            raise ValueError("Every annotated Chakra node requires semantic operation, task, and stage provenance")
    command = [part.format(input=str(source), output=str(output_trace)) for part in converter_command]
    if not command:
        raise ChakraUnavailable("An official Chakra converter command is required")
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise ChakraUnavailable("Official Chakra converter failed: " + completed.stderr[-1000:])
    mapping: dict[str, list[str]] = {oid: [] for oid in sorted(lowered_ids)}
    reverse = {}
    for node in records:
        opid = node.get("masbench_semantic_op_id")
        node_id = str(node.get("chakra_node_id", node.get("node_id", "")))
        if opid and node_id:
            mapping[str(opid)].append(node_id)
            reverse[node_id] = str(opid)
    chakra_edges = [[str(x["src"]), str(x["dst"])] for x in annotations.get("edges", [])]
    return {"schema_version": "masbench.chakra_lowering/1.0.0", "chakra_trace": str(output_trace),
            "converter_version": converter_version, "model_or_cost_version": model_or_cost_version,
            "lowered_operation_ids": sorted(lowered_ids), "semantic_to_chakra_nodes": mapping,
            "chakra_node_to_semantic": reverse, "fixed_serving_execution_plan": True,
            "chakra_edges": chakra_edges,
            "scheduler_feedback_supported": False, "source_request_latency_used_as_node_duration": False}

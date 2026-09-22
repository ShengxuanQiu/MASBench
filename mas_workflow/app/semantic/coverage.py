"""Coverage is an abstraction audit, separate from performance scoring."""
from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from .model import SemanticTrace

FEATURES = ["workflow_depth", "stage_depth", "request_count", "fan_out_mean", "fan_out_max",
    "fan_in_mean", "fan_in_max", "peak_ready_parallelism", "spawn_count", "join_count",
    "refinement_rounds", "debate_rounds", "input_tokens_mean", "input_tokens_p95",
    "output_tokens_mean", "output_tokens_p95", "context_growth", "prefix_context_overlap",
    "tool_operation_fraction", "controlled_external_delay_fraction"]


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values); index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[index])


def extract_feature_vector(trace: SemanticTrace, workload_id: str | None = None) -> dict[str, Any]:
    """Extract architecture-independent features from a realized semantic DAG."""
    operations = {x.operation_id: x for x in trace.operations}
    children = {oid: [] for oid in operations}
    depth: dict[str, int] = {}
    pending = set(operations)
    levels: dict[int, int] = {}
    while pending:
        ready = [oid for oid in pending if operations[oid].dependencies <= set(depth)]
        if not ready:
            raise ValueError("Cannot extract features from a cyclic trace")
        for oid in ready:
            d = 1 + max((depth[parent] for parent in operations[oid].dependencies), default=0)
            depth[oid] = d; levels[d] = levels.get(d, 0) + 1; pending.remove(oid)
            for parent in operations[oid].dependencies: children[parent].append(oid)
    stage_map = {x.stage_instance_id: x for x in trace.stages}
    stage_depth: dict[str, int] = {}
    pending_stages = set(stage_map)
    while pending_stages:
        ready = [sid for sid in pending_stages if set(stage_map[sid].stage_dependencies) <= set(stage_depth)]
        if not ready: raise ValueError("Stage hierarchy is cyclic")
        for sid in ready:
            stage_depth[sid] = 1 + max((stage_depth[x] for x in stage_map[sid].stage_dependencies), default=0)
            pending_stages.remove(sid)
    fan_out = [len(children[x]) for x in operations]
    fan_in = [len(x.dependencies) for x in operations.values()]
    llms = [x for x in operations.values() if x.llm]
    input_tokens = [float(x.llm["input_token_count"]) for x in llms]
    output_tokens = [float(x.llm["recorded_output_length"]) for x in llms]
    by_session: dict[str, list[tuple[int, int]]] = {}
    for op in llms:
        by_session.setdefault(op.session_id or "", []).append((op.iteration, op.llm["input_token_count"]))
    growth = [max(v for _, v in rows) - min(v for _, v in rows) for rows in by_session.values() if len(rows) > 1]
    overlaps = []
    by_scope: dict[str, list[list[int]]] = {}
    for op in llms:
        if op.llm.get("reuse_scope_id"):
            by_scope.setdefault(op.llm["reuse_scope_id"], []).append(op.llm["input_token_ids"])
    for rows in by_scope.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                common = 0
                for a, b in zip(rows[i], rows[j]):
                    if a != b: break
                    common += 1
                overlaps.append(common / max(1, min(len(rows[i]), len(rows[j]))))
    templates = [x.template_type for x in trace.stages]
    return {"workload_id": workload_id or trace.metadata.workload_id,
        "workflow_depth": float(max(stage_depth.values(), default=0)), "stage_depth": float(max(depth.values(), default=0)),
        "request_count": float(len(llms)), "fan_out_mean": sum(fan_out) / len(fan_out) if fan_out else 0.0,
        "fan_out_max": float(max(fan_out, default=0)), "fan_in_mean": sum(fan_in) / len(fan_in) if fan_in else 0.0,
        "fan_in_max": float(max(fan_in, default=0)), "peak_ready_parallelism": float(max(levels.values(), default=0)),
        "spawn_count": float(templates.count("spawn")), "join_count": float(templates.count("fork_join")),
        "refinement_rounds": float(max((x.iteration for x in llms if stage_map[x.stage_instance_id].template_type == "refinement_loop"), default=0)),
        "debate_rounds": float(max((x.iteration for x in llms if stage_map[x.stage_instance_id].template_type == "debate"), default=0)),
        "input_tokens_mean": sum(input_tokens) / len(input_tokens) if input_tokens else 0.0, "input_tokens_p95": _quantile(input_tokens, .95),
        "output_tokens_mean": sum(output_tokens) / len(output_tokens) if output_tokens else 0.0, "output_tokens_p95": _quantile(output_tokens, .95),
        "context_growth": sum(growth) / len(growth) if growth else 0.0,
        "prefix_context_overlap": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "tool_operation_fraction": sum(x.operation_type == "tool" for x in operations.values()) / len(operations) if operations else 0.0,
        "controlled_external_delay_fraction": sum(x.controlled_external_delay > 0 for x in operations.values()) / len(operations) if operations else 0.0}


def validate_corpus(data: dict[str, Any]) -> None:
    if data.get("schema_version") != "masbench.coverage_corpus/1.0.0":
        raise ValueError("Unsupported coverage corpus schema")
    keys = set()
    for item in data.get("systems", []):
        key = (item.get("system"), item.get("version"), item.get("canonical_configuration"))
        if any(not x for x in key) or key in keys:
            raise ValueError("Corpus unit must be unique system/framework x version x canonical configuration")
        keys.add(key)
        if item.get("split") not in {"discovery", "held_out"}:
            raise ValueError("Corpus split must be discovery or held_out")
        if item.get("relevance") != "causal_multi_agent" or not item.get("inspectability_sources") or not item.get("frozen_reference"):
            raise ValueError("Corpus entry fails relevance/inspectability/versionability requirements")
        impact = item.get("external_impact_snapshot")
        impact_fields = {"captured_at", "github_stars", "citation_count", "publication_or_release_date",
                         "organization_or_vendor", "maintenance_status", "downstream_adoption_evidence"}
        if not isinstance(impact, dict) or impact_fields - set(impact):
            raise ValueError("Corpus entry requires a complete external-impact snapshot")
        assessment = item.get("workflow_assessment")
        if not isinstance(assessment, dict) or not isinstance(assessment.get("stages"), list) or not assessment["stages"]:
            raise ValueError("Coverage entry requires an auditable workflow_assessment")
        computed = (all(stage.get("template_type") in {"spawn", "fork_join", "refinement_loop", "debate"}
                        and stage.get("standard_semantics") is True and not stage.get("requires_extension")
                        and not stage.get("wrapped_as_atomic") for stage in assessment["stages"])
                    and assessment.get("standard_composition_expressible") is True
                    and assessment.get("arbitrary_control_callback_required") is False)
        if item.get("zero_extension_covered") != computed:
            raise ValueError("zero_extension_covered disagrees with the auditable assessment")


def workflow_coverage(data: dict[str, Any]) -> dict[str, Any]:
    validate_corpus(data)
    held = [x for x in data["systems"] if x["split"] == "held_out"]
    covered = [x for x in held if x.get("zero_extension_covered") is True]
    residuals = [{"unit_id": x.get("unit_id"), "failure_reason": x.get("failure_reason"),
                  "missing_semantics": x.get("missing_semantics", []), "residual_category": x.get("residual_category")}
                 for x in held if x.get("zero_extension_covered") is not True]
    return {"metric": "C_workflow", "numerator": len(covered), "denominator": len(held),
            "value": len(covered) / len(held) if held else None, "residuals": residuals,
            "corpus_split_version": data.get("corpus_split_version"), "abstraction_version": data.get("abstraction_version")}


def _normalize(real_rows: list[list[float]], benchmark_rows: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    columns = list(zip(*real_rows))
    means = [sum(x) / len(x) for x in columns]
    stds = [math.sqrt(sum((v - means[i]) ** 2 for v in columns[i]) / len(real_rows)) or 1.0 for i in range(len(columns))]
    transform = lambda rows: [[(v - means[i]) / stds[i] for i, v in enumerate(row)] for row in rows]
    return transform(real_rows), transform(benchmark_rows)


def _pca(real_rows: list[list[float]], benchmark_rows: list[list[float]], dimensions: int) -> tuple[list[list[float]], list[list[float]]]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Behavioral cluster coverage requires numpy") from exc
    real = np.asarray(real_rows, dtype=float)
    benchmark = np.asarray(benchmark_rows, dtype=float)
    _, _, vt = np.linalg.svd(real, full_matrices=False)
    basis = vt[:min(dimensions, vt.shape[0])].T
    return (real @ basis).tolist(), (benchmark @ basis).tolist()


def _kmeans(rows: list[list[float]], k: int, seed: int, iterations: int = 100):
    rng = random.Random(seed)
    centers = [list(rows[i]) for i in rng.sample(range(len(rows)), k)]
    labels = [0] * len(rows)
    for _ in range(iterations):
        new_labels = [min(range(k), key=lambda j: sum((a-b)**2 for a,b in zip(row, centers[j]))) for row in rows]
        new_centers = []
        for j in range(k):
            members = [row for row, label in zip(rows, new_labels) if label == j]
            new_centers.append([sum(values) / len(values) for values in zip(*members)] if members else centers[j])
        if new_labels == labels:
            break
        labels, centers = new_labels, new_centers
    return labels, centers


def cluster_coverage(real_features: list[dict[str, Any]], benchmark_features: list[dict[str, Any]], *, clusters: int, seed: int = 42, pca_dimensions: int = 8) -> dict[str, Any]:
    if not real_features or not benchmark_features or clusters < 1 or clusters > len(real_features):
        raise ValueError("Cluster coverage needs real/benchmark rows and a valid cluster count")
    real_rows = [[float(x[name]) for name in FEATURES] for x in real_features]
    benchmark_rows = [[float(x[name]) for name in FEATURES] for x in benchmark_features]
    real_rows, benchmark_rows = _normalize(real_rows, benchmark_rows)
    real_projected, benchmark_projected = _pca(real_rows, benchmark_rows, pca_dimensions)
    labels, centers = _kmeans(real_projected, clusters, seed)
    represented = set()
    distances = []
    for row, item in zip(benchmark_projected, benchmark_features):
        label = min(range(clusters), key=lambda j: sum((a-b)**2 for a,b in zip(row, centers[j])))
        represented.add(label)
        distances.append({"workload_id": item.get("workload_id"), "cluster": label,
                          "distance": math.sqrt(sum((a-b)**2 for a,b in zip(row, centers[label])))})
    return {"metric": "C_cluster", "numerator": len(represented), "denominator": clusters,
            "value": len(represented) / clusters, "represented_clusters": sorted(represented),
            "pca_dimensions": min(pca_dimensions, len(real_projected[0])), "random_seed": seed,
            "diagnostics": {"nearest_cluster_distances": distances, "real_cluster_assignments": labels}}


def read_feature_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

#!/usr/bin/env python3
"""Create compact, auditable tables from an Ascend publication pilot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def nested(data, *keys):
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = args.matrix.resolve()
    manifest = json.loads((matrix / "matrix_manifest.json").read_text(encoding="utf-8"))
    cell_map = {row["id"]: row for row in manifest["cells"]}
    point_rows, trace_rows, backend_rows = [], [], []
    summaries = []

    for cell_id, cell in cell_map.items():
        result_root = matrix / cell_id / "results"
        capacity_path = result_root / "capacity_results.json"
        if capacity_path.exists():
            capacity = json.loads(capacity_path.read_text(encoding="utf-8"))
            for deployment, result in capacity.items():
                for point in result.get("points", []):
                    metrics = point.get("metrics", {})
                    point_rows.append({
                        "cell_id": cell_id,
                        "workload": cell.get("workload"),
                        "parameters": json.dumps(cell.get("parameters", {}), sort_keys=True),
                        "deployment": deployment,
                        "capacity_status": result.get("status"),
                        "lambda_knee": result.get("lambda_knee"),
                        "rate": point.get("rate"),
                        "slo_pass": point.get("slo_pass"),
                        "valid_client": point.get("valid_client"),
                        "e2e_p95_sec": nested(metrics, "e2e_p95_sec", "mean"),
                        "goodput_qps": nested(metrics, "goodput_qps", "mean"),
                        "internal_qps": nested(metrics, "internal_qps", "mean"),
                        "slo_success_fraction": nested(metrics, "slo_success_fraction", "mean"),
                    })
        for summary_path in result_root.glob("**/summary.json"):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if "rate" not in summary or "attempt_path" not in summary:
                continue
            summaries.append(summary)
            attempt = Path(summary["attempt_path"])
            backend_path = attempt / "backend_metrics.json"
            if backend_path.exists():
                backend = json.loads(backend_path.read_text(encoding="utf-8")).get("summary", {})
                backend_rows.append({
                    "cell_id": cell_id,
                    "workload": cell.get("workload"),
                    "parameters": json.dumps(cell.get("parameters", {}), sort_keys=True),
                    "rate": summary.get("rate"),
                    "repetition": summary.get("repetition"),
                    **backend,
                })
            for analysis_path in (attempt / "analysis").glob("*.json"):
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                structure = analysis.get("structure", {})
                runtime = analysis.get("runtime", {})
                pressure = analysis.get("pressure", {})
                flow = analysis.get("information_flow", {})
                trace_rows.append({
                    "cell_id": cell_id,
                    "workload": cell.get("workload"),
                    "parameters": json.dumps(cell.get("parameters", {}), sort_keys=True),
                    "rate": summary.get("rate"),
                    "run_id": analysis.get("run_id"),
                    "nodes": structure.get("nodes"),
                    "edges": structure.get("edges"),
                    "depth_nodes": structure.get("depth_nodes"),
                    "width_max_antichain": structure.get("width_max_antichain"),
                    "dag_edge_density": structure.get("dag_edge_density"),
                    "critical_path_node_ratio": structure.get("critical_path_node_ratio"),
                    "artifact_reuse_mean": nested(structure, "artifact_reuse_multiplicity", "mean"),
                    "artifact_byte_amplification": structure.get("artifact_byte_amplification"),
                    "llm_requests": runtime.get("llm_requests"),
                    "e2e_sec": runtime.get("e2e_sec"),
                    "weighted_cp_over_e2e": runtime.get("weighted_cp_over_e2e"),
                    "barrier_wait_sec": pressure.get("synchronization_exposure_sec"),
                    "peak_ready_waiting": nested(pressure, "temporal_pressure", "peak_ready_waiting"),
                    "peak_llm_inflight": nested(pressure, "temporal_pressure", "peak_llm_inflight"),
                    "delivered_bytes": flow.get("delivered_bytes"),
                    "unique_source_artifact_bytes": flow.get("unique_source_artifact_bytes"),
                    "delivery_compression_ratio_bytes": flow.get("delivery_compression_ratio_bytes"),
                    "context_information_amplification_bytes": flow.get("context_information_amplification_bytes"),
                })

    write_csv(args.output / "capacity_points.csv", point_rows)
    write_csv(args.output / "trace_metrics.csv", trace_rows)
    write_csv(args.output / "backend_points.csv", backend_rows)
    readiness = {
        "schema": "masbench_ascend_initial_validation_v1",
        "cells_planned": len(cell_map),
        "capacity_points": len(point_rows),
        "workflow_traces_analyzed": len(trace_rows),
        "study_cases": len(summaries),
        "failed_or_rejected": sum(int(row.get("failed_or_rejected", 0)) for row in summaries),
        "analysis_error_count": sum(len(row.get("analysis_errors", [])) for row in summaries),
        "client_overflow": sum(int(row.get("client_overflow", 0)) for row in summaries),
        "backend_samples": sum(int(row.get("backend_metrics_sample_count", 0)) for row in backend_rows),
        "interpretation": "Acceptance and rate-bracketing evidence only; not a cross-hardware or steady-state conclusion.",
    }
    (args.output / "readiness.json").write_text(json.dumps(readiness, indent=2), encoding="utf-8")
    print(json.dumps(readiness, indent=2))


if __name__ == "__main__":
    main()

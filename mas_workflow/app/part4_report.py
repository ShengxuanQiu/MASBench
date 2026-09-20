"""Create machine-readable Section 4.2/4.3 tables from publication traces."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .study import atomic_json


def _get(obj, path, default=None):
    for part in path.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return default
        obj = obj[part]
    return obj


def _flat_analysis(analysis):
    return {
        "run_id": analysis["run_id"],
        "operation_nodes": _get(analysis, "structure.nodes"),
        "operation_edges": _get(analysis, "structure.edges"),
        "width_max_antichain": _get(analysis, "structure.width_max_antichain"),
        "critical_path_depth_nodes": _get(analysis, "structure.depth_nodes"),
        "fan_out_mean": _get(analysis, "structure.fan_out.mean"),
        "fan_out_max": _get(analysis, "structure.fan_out.max"),
        "fan_in_mean": _get(analysis, "structure.fan_in.mean"),
        "fan_in_max": _get(analysis, "structure.fan_in.max"),
        "edge_density": _get(analysis, "structure.dag_edge_density"),
        "critical_path_node_ratio": _get(analysis, "structure.critical_path_node_ratio"),
        "artifact_reuse_mean": _get(analysis, "information_flow.artifact_reuse_multiplicity.mean"),
        "delivered_bytes": _get(analysis, "information_flow.delivered_bytes"),
        "delivered_tokens_est": _get(analysis, "information_flow.delivered_tokens_est"),
        "delivery_compression_ratio_bytes": _get(analysis, "information_flow.delivery_compression_ratio_bytes"),
        "context_information_amplification_bytes": _get(analysis, "information_flow.context_information_amplification_bytes"),
        "request_context_amplification": _get(analysis, "runtime.total_input_bytes_per_task_byte"),
        "weighted_critical_path_sec": _get(analysis, "runtime.weighted_critical_path_sec"),
        "critical_path_work_ratio": _get(analysis, "runtime.critical_path_work_ratio"),
        "ready_wait_seconds": _get(analysis, "pressure.temporal_pressure.ready_wait_seconds"),
        "peak_ready_frontier": _get(analysis, "pressure.temporal_pressure.peak_ready_waiting"),
        "peak_llm_inflight": _get(analysis, "pressure.temporal_pressure.peak_llm_inflight"),
        "barrier_wait_seconds": _get(analysis, "pressure.synchronization_exposure_sec"),
        "logical_state_peak_bytes": _get(analysis, "pressure.logical_state_residency.peak_artifact_bytes"),
        "logical_state_byte_seconds": _get(analysis, "pressure.logical_state_residency.byte_seconds"),
        "artifact_lifetime_mean_sec": _get(analysis, "pressure.logical_state_residency.artifact_lifetime_sec.mean"),
        "artifact_lifetime_p95_sec": _get(analysis, "pressure.logical_state_residency.artifact_lifetime_sec.p95"),
    }


def collect_publication(root, output):
    root = Path(root).resolve()
    manifest = json.loads((root / "matrix_manifest.json").read_text())
    cells = {row["id"]: row for row in manifest["cells"]}
    workflow_rows, case_rows = [], []
    for cell_id, cell in cells.items():
        cell_root = root / cell_id
        for summary_path in cell_root.rglob("summary.json"):
            summary = json.loads(summary_path.read_text())
            if "case_id" not in summary:
                continue
            base = {"cell_id": cell_id, "workload": cell["workload"], "task": cell["task"],
                    "parameters": json.dumps(cell["parameters"], sort_keys=True),
                    "deployment": summary["deployment"], "rate": summary["rate"],
                    "repetition": summary["repetition"], "case_id": summary["case_id"]}
            attempt = Path(summary["attempt_path"])
            backend_path = attempt / "backend_metrics.json"
            backend = json.loads(backend_path.read_text()).get("summary", {}) if backend_path.exists() else {}
            case_rows.append({**base,
                "offered": summary["offered"], "completed": summary["completed"],
                "p50_e2e_sec": _get(summary, "completed_e2e_sec.p50"),
                "p95_e2e_sec": _get(summary, "completed_e2e_sec.p95"),
                "goodput_qps": summary["goodput_qps"], "slo_success_fraction": summary["slo_success_fraction"],
                "internal_qps": summary["internal_qps"], "queue_growth_client_per_sec": summary.get("queue_growth_client_per_sec"),
                "max_client_ready_waiting": summary.get("max_client_ready_waiting"),
                "max_llm_inflight": summary.get("max_llm_inflight"),
                "backend_max_requests_running": backend.get("max_num_requests_running"),
                "backend_max_requests_waiting": backend.get("max_num_requests_waiting"),
                "backend_max_gpu_cache_usage_percent": backend.get("max_gpu_cache_usage_perc"),
                "backend_prompt_tokens_per_sec": backend.get("backend_prompt_tokens_per_sec_window"),
                "backend_generation_tokens_per_sec": backend.get("backend_generation_tokens_per_sec_window"),
                "backend_metrics_path": str(backend_path) if backend_path.exists() else None,
                "analysis_error_count": len(summary.get("analysis_errors", []))})
            for analysis_path in (attempt / "analysis").glob("*.json"):
                workflow_rows.append({**base, **_flat_analysis(json.loads(analysis_path.read_text()))})
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", {"schema": "masbench_part4_report_v1",
        "workflow_rows": workflow_rows, "case_rows": case_rows,
        "definitions": {
            "pressure_signature": "realized DAG structure plus ready/barrier/context/logical-state dynamics at low offered load",
            "state_lifetime": "artifact produce time to last consumer finish, or run finish when unconsumed",
            "context_amplification": "serialized request message bytes divided by original task UTF-8 bytes; not semantic duplication",
            "physical_kv": "unavailable from client traces; backend cache metrics remain separate"}})
    for name, rows in (("workflow_rows.csv", workflow_rows), ("case_rows.csv", case_rows)):
        if not rows:
            continue
        with (output / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    return {"workflow_rows": len(workflow_rows), "case_rows": len(case_rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(collect_publication(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()

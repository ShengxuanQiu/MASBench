#!/usr/bin/env python3
"""Replay every recorded source workflow under a small controlled load sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.capacity import run_capacity
from app.study import atomic_json


def source_trace(cell_root: Path) -> Path:
    candidates = sorted(cell_root.glob("results/**/traces/**/*.jsonl"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one source trace in {cell_root}, found {len(candidates)}")
    return candidates[0].resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--rates", default="0.1,0.25")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((source / "matrix_manifest.json").read_text(encoding="utf-8"))
    rates = [float(value) for value in args.rates.split(",") if value.strip()]
    if not rates or args.count < 1:
        raise ValueError("rates and count must be nonempty and positive")
    replay_manifest = {
        **source_manifest,
        "replay_source_matrix": str(source),
        "replay_protocol": "fixed operation DAG, dependencies, tool snapshots, request payloads and decisions",
    }
    atomic_json(output / "matrix_manifest.json", replay_manifest)
    deployment = args.deployment.resolve()
    for cell in source_manifest["cells"]:
        cell_out = output / cell["id"]
        cell_out.mkdir(exist_ok=True)
        trace = source_trace(source / cell["id"])
        study = {
            "mode": "replay",
            "traces": [str(trace)],
            "deployments": {"ascend": str(deployment)},
            "rates": rates,
            "count": args.count,
            "repetitions": 1,
            "seed": 42,
            "arrival": "poisson",
            "max_inflight_workflows": 16,
            "slo_sec": 180,
            "cache_protocol": "warm_cache_enabled",
            "warmup_count": 1,
            "collect_backend_metrics": True,
            "backend_metrics_interval_sec": 2.0,
            "strict_replay": False,
            "output_length_tolerance": 0.25,
            "require_identity_match": False,
        }
        atomic_json(cell_out / "study.json", study)
        atomic_json(cell_out / "capacity.json", {
            "study": "study.json",
            "capacity": {"dense_points": 1, "min_slo_success": 0.95, "max_failure_fraction": 0.01},
        })
        run_capacity(cell_out / "capacity.json", cell_out / "results", resume=args.resume)
    print(json.dumps({"cells": len(source_manifest["cells"]), "rates": rates, "count": args.count}))


if __name__ == "__main__":
    main()

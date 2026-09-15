#!/usr/bin/env python3
"""Materialize one live Tavily evidence stage for every source workflow cell."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.capacity import run_capacity
from app.publication import generate
from app.specs import experiment_from_dict
from app.study import atomic_json


def prerequisites(stage: dict) -> set[str]:
    parents = set(stage.get("depends_on", []))
    for selector in (stage.get("condition"), stage.get("participants")):
        if selector:
            parents.add(selector["from_stage"])
    for raw in stage.get("inputs", {}).values():
        for edge in raw if isinstance(raw, list) else [raw]:
            source = edge.get("source") if isinstance(edge, dict) else edge
            parents.add(source.split(".", 1)[0])
    return parents


def add_evidence_input(stage: dict) -> None:
    edge = {"source": "live_evidence.result", "delivery": {"mode": "full"}}
    inputs = stage.setdefault("inputs", {})
    current = inputs.get("context")
    if current is None:
        inputs["context"] = [edge]
    elif isinstance(current, list):
        current.append(edge)
    else:
        inputs["context"] = [current, edge]


def inject_live_evidence(experiment: dict) -> dict:
    stages = experiment["structure"]["workflow"]["stages"]
    if any(stage["id"] == "live_evidence" for stage in stages):
        raise ValueError("live_evidence is a reserved source-materialization stage id")
    roots = [stage for stage in stages if not prerequisites(stage)]
    matched_atomic = stages and all((stage.get("atomic") or {}).get("kind") == "llm" for stage in stages)
    targets = stages if matched_atomic else roots
    for stage in targets:
        add_evidence_input(stage)
    stages.insert(0, {
        "id": "live_evidence",
        "atomic": {"kind": "tool", "role": "Worker"},
    })
    experiment["task"]["tool_provider"] = "tavily"
    experiment["task"]["tool_mode"] = "live"
    experiment_from_dict(experiment)
    return experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cells = generate(args.manifest, args.output, resume=args.resume)
    for cell in cells:
        path = args.output / cell["id"] / "experiment.json"
        experiment = json.loads(path.read_text(encoding="utf-8"))
        atomic_json(path, inject_live_evidence(experiment))
    manifest_path = args.output / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_materialization"] = {
        "provider": "tavily",
        "policy": "one explicit live evidence operation per workflow source run",
        "replay_policy": "recorded operation DAG, tool snapshot, downstream payload and control path",
    }
    atomic_json(manifest_path, manifest)
    if args.execute:
        for cell in cells:
            run_capacity(
                args.output / cell["id"] / "capacity.json",
                args.output / cell["id"] / "results",
                resume=args.resume,
            )
    print(json.dumps({"cells": len(cells), "live_tavily_source_runs": len(cells), "executed": args.execute}))


if __name__ == "__main__":
    main()

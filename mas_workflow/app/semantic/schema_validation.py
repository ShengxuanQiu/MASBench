"""Validate checked-in Semantic Trace, Scenario, result, and coverage fixtures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jsonschema


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_repository(root: Path) -> dict[str, int]:
    semantic = root / "mas_workflow" / "configs" / "semantic"
    trace_schema = load(semantic / "semantic_trace.schema.json")
    operation_schema = load(semantic / "semantic_operation.schema.json")
    scenario_schema = load(semantic / "scenario.schema.json")
    result_schema = load(semantic / "benchmark_result.schema.json")
    coverage_schema = load(root / "evaluation" / "coverage" / "coverage_corpus.schema.json")
    for schema in (trace_schema, operation_schema, scenario_schema, result_schema, coverage_schema):
        jsonschema.Draft202012Validator.check_schema(schema)
    counts = {"schemas": 5, "trace_manifests": 0, "operations": 0, "scenarios": 0, "coverage_corpora": 0}
    for directory in (root / "mas_workflow" / "tests" / "golden" / "semantic").iterdir():
        if not directory.is_dir(): continue
        jsonschema.validate(load(directory / "manifest.json"), trace_schema); counts["trace_manifests"] += 1
        for line in (directory / "operations.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip(): jsonschema.validate(json.loads(line), operation_schema); counts["operations"] += 1
    for path in (semantic / "examples").glob("scenario*.json"):
        if ".template." in path.name: continue
        jsonschema.validate(load(path), scenario_schema); counts["scenarios"] += 1
    for path in (root / "evaluation" / "coverage").glob("coverage_corpus*.json"):
        if path.name.endswith("schema.json"): continue
        jsonschema.validate(load(path), coverage_schema); counts["coverage_corpora"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--repo-root", default=".")
    result = validate_repository(Path(parser.parse_args().repo_root).resolve()); print(json.dumps(result))


if __name__ == "__main__": main()

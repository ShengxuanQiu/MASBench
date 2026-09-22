"""Freeze representative canonical traces from a mixed run for later replay."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .study import atomic_json


def freeze(input_dir, output, per_class=3):
    input_dir = Path(input_dir).resolve()
    results = json.loads((input_dir / "results.json").read_text())
    classes = sorted({name for row in results for name in row.get("per_class", {})})
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for name in classes:
        candidates = [row for row in results if row.get("mix") == "isolated__" + name]
        if not candidates:
            candidates = [row for row in results if name in row.get("per_class", {})]
        candidates.sort(key=lambda row: (row["rate"], row["repetition"], row["case_id"]))
        selected = []
        for case in candidates:
            records = json.loads((Path(case["attempt_path"]) / "records.json").read_text())
            for record in records:
                if record.get("workload_class") != name or record.get("status") not in {"completed", "accepted"}:
                    continue
                source = Path(record["trace_path"])
                sha = hashlib.sha256(source.read_bytes()).hexdigest()
                target = output / name / (sha + ".jsonl")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source, target)
                selected.append({"trace": str(target.relative_to(output)), "sha256": sha,
                                 "source_case_id": case["case_id"], "source_rate": case["rate"]})
                if len(selected) == per_class:
                    break
            if len(selected) == per_class:
                break
        if not selected:
            raise ValueError("No completed trace for class " + name)
        path = output / (name + ".json")
        atomic_json(path, {"schema": "masbench_frozen_class_corpus_v1", "class": name,
                           "selection": "lowest-rate isolated completed traces first",
                           "runs": selected})
        manifests[name] = str(path)
    atomic_json(output / "index.json", {"schema": "masbench_frozen_corpus_index_v1", "classes": manifests})
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(freeze(args.input, args.output, args.per_class), indent=2))


if __name__ == "__main__":
    main()

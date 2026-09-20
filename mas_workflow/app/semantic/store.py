"""Content-addressed storage and streaming Semantic Trace bundles."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .model import SemanticTrace, canonical_json


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put_bytes(self, payload: bytes) -> tuple[str, str]:
        digest = hashlib.sha256(payload).hexdigest()
        rel = Path("sha256") / digest
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("CAS collision or corruption")
        return rel.as_posix(), digest

    def put_json(self, value: Any) -> tuple[str, str]:
        return self.put_bytes(canonical_json(value).encode("utf-8"))

    def get_bytes(self, ref: str, expected_hash: str | None = None) -> bytes:
        path = (self.root / ref).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError("Artifact reference escapes the store")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_hash and digest != expected_hash:
            raise ValueError("Artifact hash mismatch")
        return payload

    def get_json(self, ref: str, expected_hash: str | None = None) -> Any:
        return json.loads(self.get_bytes(ref, expected_hash))


class TraceBundle:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.artifacts = ArtifactStore(self.root / "artifacts")

    def write(self, trace: SemanticTrace) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        trace.finalize_hashes()
        (self.root / "manifest.json").write_text(
            json.dumps(trace.to_manifest(), ensure_ascii=False, indent=2), encoding="utf-8")
        with (self.root / "operations.jsonl").open("w", encoding="utf-8") as handle:
            for operation in trace.operations:
                handle.write(canonical_json(asdict(operation)) + "\n")

    def read(self) -> SemanticTrace:
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        operations = [json.loads(line) for line in (self.root / manifest.get("operations_file", "operations.jsonl")).read_text(encoding="utf-8").splitlines() if line.strip()]
        trace = SemanticTrace.from_parts(manifest, operations)
        for artifact in trace.artifacts:
            self.artifacts.get_bytes(artifact.payload_ref, artifact.content_hash)
        return trace

"""Small SWE-bench instance loader for full software workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SWEBENCH_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "test_patch",
    "patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)


def normalize_swebench_instance(row: dict[str, Any]) -> dict[str, Any]:
    """Return a stable subset while preserving unknown fields under extra."""
    normalized = {field: row.get(field, "" if field not in {"FAIL_TO_PASS", "PASS_TO_PASS"} else []) for field in SWEBENCH_FIELDS}
    normalized["instance_id"] = str(normalized.get("instance_id") or "manual_swebench_instance")
    normalized["problem_statement"] = str(normalized.get("problem_statement") or "")
    normalized["repo"] = str(normalized.get("repo") or "")
    normalized["base_commit"] = str(normalized.get("base_commit") or "")
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = normalized.get(key)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                normalized[key] = parsed if isinstance(parsed, list) else [value]
            except Exception:
                normalized[key] = [value] if value else []
        elif value is None:
            normalized[key] = []
    normalized["extra_fields"] = {key: value for key, value in row.items() if key not in SWEBENCH_FIELDS}
    return normalized


def load_swebench_instances(path: str | Path, *, instance_ids: list[str] | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """Load SWE-bench rows from local .json or .jsonl."""
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"SWE-bench task file does not exist: {source}")
    rows: list[dict[str, Any]] = []
    if source.suffix.lower() == ".jsonl":
        for line in source.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict) and isinstance(payload.get("instances"), list):
            rows = payload["instances"]
        elif isinstance(payload, dict):
            rows = [payload]
        else:
            raise ValueError(f"Unsupported SWE-bench JSON payload in {source}")
    wanted = set(instance_ids or [])
    if wanted:
        rows = [row for row in rows if str(row.get("instance_id")) in wanted]
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return [normalize_swebench_instance(row) for row in rows]

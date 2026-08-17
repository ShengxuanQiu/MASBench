from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, path: Path, *, run_id: str, workload: str, mode: str) -> None:
        self.path = path
        self.run_id = run_id
        self.workload = workload
        self.mode = mode
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        row = {
            "schema_version": "masbench_case_study2.v1",
            "event_type": event_type,
            "timestamp_unix": time.time(),
            "run_id": self.run_id,
            "workflow_id": f"{self.workload}_{self.run_id}",
            "workload": self.workload,
            "mode": self.mode,
            **fields,
        }
        payload = json.dumps(row, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        return row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

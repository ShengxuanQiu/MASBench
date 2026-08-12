from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class TraceWriter:
    """Small thread-safe JSONL writer for online workflow events."""

    def __init__(self, path: Path, *, run_id: str, policy: str) -> None:
        self.path = path
        self.run_id = run_id
        self.policy = policy
        self.started_wall = time.time()
        self.started_perf = time.perf_counter()
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    def emit(self, event_type: str, **fields: Any) -> dict[str, Any]:
        record = {
            "schema_version": "masbench_case_study1.v1",
            "event_type": event_type,
            "run_id": self.run_id,
            "workflow_id": self.run_id,
            "policy": self.policy,
            "timestamp_unix": time.time(),
            "relative_time_sec": time.perf_counter() - self.started_perf,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return record

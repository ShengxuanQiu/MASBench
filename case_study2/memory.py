from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from .trace import TraceWriter


RECORD_TYPES = {"claim", "evidence", "constraint", "keyword_summary"}


def terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text)
    }


def pairwise_distinctiveness(term_sets: list[set[str]]) -> float | None:
    """One minus mean pairwise Jaccard similarity; 1.0 means fully distinct."""
    if len(term_sets) < 2:
        return None
    similarities: list[float] = []
    for index, left in enumerate(term_sets):
        for right in term_sets[index + 1 :]:
            union = left | right
            similarities.append(len(left & right) / len(union) if union else 1.0)
    return 1.0 - sum(similarities) / len(similarities)


class TaskMemoryStore:
    """Task-scoped append-only structured records produced in the source request."""

    def __init__(self, root: Path, task_id: str, trace: TraceWriter) -> None:
        self.task_id = task_id
        self.path = root / f"{task_id}.json"
        self.trace = trace
        self._lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {
            "task_id": task_id,
            "state": "active",
            "materialization": "producer_same_generation",
            "created_at": time.time(),
            "records": [],
        }
        self._persist()
        trace.emit("memory_store_created", task_id=task_id, memory_path=str(self.path))

    def _persist(self) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def append(self, source: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        with self._lock:
            if self._data["state"] != "active":
                raise RuntimeError("cannot append to a frozen memory store")
            appended: list[dict[str, Any]] = []
            for raw in records:
                if raw.get("type") not in RECORD_TYPES or not str(raw.get("content", "")).strip():
                    raise ValueError("invalid producer-side memory record")
                memory_id = f"m_{len(self._data['records']) + 1:04d}"
                row = {
                    "memory_id": memory_id,
                    "source_agent": source["source_agent"],
                    "source_artifact_id": source["artifact_id"],
                    "timestamp": time.time(),
                    "type": raw["type"],
                    "content": str(raw["content"]).strip(),
                    "keywords": list(raw.get("keywords", [])),
                    "evidence_refs": list(raw.get("evidence_refs", [])),
                    "token_count": int(raw["token_count"]),
                    "graph_reason": source.get("graph_reason"),
                }
                self._data["records"].append(row)
                appended.append(row)
            self._persist()
        self.trace.emit(
            "memory_append",
            source_agent=source["source_agent"],
            source_artifact_id=source["artifact_id"],
            memory_ids=[row["memory_id"] for row in appended],
            record_count=len(appended),
            structured_tokens=sum(row["token_count"] for row in appended),
        )
        return appended

    def retrieve(
        self,
        *,
        query: str,
        source_artifact_ids: list[str],
        top_k: int,
    ) -> list[dict[str, Any]]:
        with self._lock:
            candidates = [
                dict(row)
                for row in self._data["records"]
                if row["source_artifact_id"] in source_artifact_ids
            ]
        query_terms = terms(query)

        def score(row: dict[str, Any]) -> tuple[int, float]:
            searchable = " ".join([row["content"], *row.get("keywords", [])])
            return len(query_terms & terms(searchable)), float(row["timestamp"])

        ranked = sorted(candidates, key=score, reverse=True)
        # Graph provenance is a hard constraint: retain at least one record from
        # every legal upstream source before filling remaining top-k slots.
        selected: list[dict[str, Any]] = []
        for source_id in source_artifact_ids:
            source_rows = [row for row in ranked if row["source_artifact_id"] == source_id]
            if source_rows:
                selected.append(source_rows[0])
        limit = max(len(selected), top_k)
        for row in ranked:
            if row not in selected and len(selected) < limit:
                selected.append(row)
        return selected[:limit]

    def freeze(self) -> None:
        with self._lock:
            self._data["state"] = "frozen"
            self._data["frozen_at"] = time.time()
            self._persist()
        self.trace.emit(
            "memory_store_frozen",
            task_id=self.task_id,
            record_count=len(self._data["records"]),
        )

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._data["records"]]

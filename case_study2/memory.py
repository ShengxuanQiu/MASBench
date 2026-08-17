from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from mas_workflow.app.llm_client import LocalLLMClient

from .trace import TraceWriter


RECORD_TYPES = {"claim", "evidence", "constraint", "keyword_summary"}


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text)
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else cleaned
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("memory agent did not return a JSON object")
    parsed = json.loads(candidate[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("memory agent JSON root must be an object")
    return parsed


class TaskMemoryStore:
    """Task-scoped append-only logical record store backed by one JSON file."""

    def __init__(self, root: Path, task_id: str, trace: TraceWriter) -> None:
        self.task_id = task_id
        self.path = root / f"{task_id}.json"
        self.trace = trace
        self._lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {
            "task_id": task_id,
            "state": "active",
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
                memory_id = f"m_{len(self._data['records']) + 1:04d}"
                row = {
                    "memory_id": memory_id,
                    "source_agent": source["source_agent"],
                    "source_artifact_id": source["artifact_id"],
                    "timestamp": time.time(),
                    "type": raw["type"],
                    "content": raw["content"],
                    "keywords": raw["keywords"],
                    "evidence_refs": raw["evidence_refs"],
                    "token_count": int(raw["token_count"]),
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
        query_terms = _terms(query)

        def score(row: dict[str, Any]) -> tuple[int, float]:
            searchable = " ".join([row["content"], *row.get("keywords", [])])
            return len(query_terms & _terms(searchable)), float(row["timestamp"])

        ranked = sorted(candidates, key=score, reverse=True)
        # Preserve graph coverage: select one record per requested source before
        # filling remaining top-k slots by lexical relevance.
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


class AsyncMemoryAgent:
    """A small LLM that only compresses artifacts into structured records."""

    def __init__(
        self,
        *,
        client: LocalLLMClient,
        store: TaskMemoryStore,
        trace: TraceWriter,
        token_count: Callable[[str], int],
        max_workers: int,
        max_tokens: int,
        seed: int,
    ) -> None:
        self.client = client
        self.store = store
        self.trace = trace
        self.token_count = token_count
        self.max_tokens = max_tokens
        self.seed = seed
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[str, Future[list[dict[str, Any]]]] = {}
        self._lock = threading.Lock()

    def schedule(self, artifact: dict[str, Any], *, objective: str) -> Future[list[dict[str, Any]]]:
        artifact_id = str(artifact["artifact_id"])
        with self._lock:
            if artifact_id in self._futures:
                return self._futures[artifact_id]
            future = self.executor.submit(self._construct, artifact, objective)
            self._futures[artifact_id] = future
        self.trace.emit(
            "memory_construction_scheduled",
            source_artifact_id=artifact_id,
            source_agent=artifact["source_agent"],
            source_tokens=artifact["token_count"],
        )
        return future

    def _invoke(self, request_id: str, messages: list[dict[str, str]]) -> tuple[str, dict[str, Any]]:
        ready = time.time()
        self.trace.emit("memory_llm_ready", request_id=request_id, ready_ts=ready)
        content, metadata = self.client.invoke_messages_streaming_with_metadata(
            messages,
            metadata={
                "request_id_for_backend": request_id,
                "agent_id": "memory_construction_agent",
                "priority": "background",
                "_request_payload": {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "memory_record",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "records": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 1,
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "type": {
                                                    "type": "string",
                                                    "enum": sorted(RECORD_TYPES),
                                                },
                                                "content": {"type": "string"},
                                                "keywords": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                    "maxItems": 6,
                                                },
                                                "evidence_refs": {
                                                    "type": "array",
                                                    "items": {"type": "string"},
                                                    "maxItems": 4,
                                                },
                                            },
                                            "required": [
                                                "type",
                                                "content",
                                                "keywords",
                                                "evidence_refs",
                                            ],
                                            "additionalProperties": False,
                                        },
                                    }
                                },
                                "required": ["records"],
                                "additionalProperties": False,
                            },
                        },
                    },
                },
            },
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        usage = metadata.get("usage") or {}
        self.trace.emit(
            "memory_llm_end",
            request_id=request_id,
            ready_ts=ready,
            first_token_ts=metadata.get("first_token_ts"),
            completion_ts=metadata.get("completion_ts"),
            ttft_sec=metadata.get("ttft_sec"),
            tpot_sec=metadata.get("tpot_sec"),
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
        )
        return content, metadata

    def _validated_records(self, response: str, source_id: str) -> list[dict[str, Any]]:
        parsed = _parse_json_object(response)
        rows = parsed.get("records")
        if not isinstance(rows, list) or not rows:
            raise ValueError("memory response must contain non-empty records")
        validated: list[dict[str, Any]] = []
        for row in rows[:1]:
            if not isinstance(row, dict):
                continue
            record_type = str(row.get("type", "")).strip()
            content = str(row.get("content", "")).strip()
            if record_type not in RECORD_TYPES or not content or content in {"...", "…"}:
                continue
            keywords = [str(item).strip() for item in row.get("keywords", []) if str(item).strip()]
            refs = [str(item).strip() for item in row.get("evidence_refs", []) if str(item).strip()]
            if source_id not in refs:
                refs.insert(0, source_id)
            validated.append(
                {
                    "type": record_type,
                    "content": content,
                    "keywords": keywords[:8],
                    "evidence_refs": refs[:8],
                    "token_count": self.token_count(content),
                }
            )
        if not validated:
            raise ValueError("memory response contained no valid record")
        return validated

    def _construct(self, artifact: dict[str, Any], objective: str) -> list[dict[str, Any]]:
        artifact_id = str(artifact["artifact_id"])
        start = time.time()
        self.trace.emit(
            "memory_construction_start",
            source_artifact_id=artifact_id,
            source_agent=artifact["source_agent"],
            start_ts=start,
        )
        system = (
            "You are a memory construction agent, not a reasoning or planning agent. "
            "Compress one source artifact into exactly 1 factual structured record. Use only "
            "types claim, evidence, constraint, keyword_summary. Each content field must "
            "be at most 16 words. Return JSON only as {\"records\":[{\"type\":...,"
            "\"content\":...,\"keywords\":[...],\"evidence_refs\":[...]}]}. /no_think"
        )
        user = (
            f"Task objective:\n{objective}\n\nSource artifact id: {artifact_id}\n"
            f"Source agent: {artifact['source_agent']}\nSource artifact:\n{artifact['content']}"
        )
        request_id = f"{self.trace.run_id}__memory__{artifact_id}"
        response, _ = self._invoke(request_id, [{"role": "system", "content": system}, {"role": "user", "content": user}])
        try:
            records = self._validated_records(response, artifact_id)
        except (ValueError, json.JSONDecodeError) as exc:
            self.trace.emit(
                "memory_parse_retry",
                request_id=request_id,
                source_artifact_id=artifact_id,
                error=str(exc),
            )
            repair = (
                f"The previous response was invalid. Return only the required JSON object.\n"
                f"Previous response:\n{response}"
            )
            response, _ = self._invoke(
                request_id + "__repair",
                [{"role": "system", "content": system}, {"role": "user", "content": repair}],
            )
            records = self._validated_records(response, artifact_id)
        appended = self.store.append(artifact, records)
        end = time.time()
        self.trace.emit(
            "memory_construction_end",
            source_artifact_id=artifact_id,
            start_ts=start,
            end_ts=end,
            duration_sec=end - start,
            source_tokens=artifact["token_count"],
            structured_tokens=sum(row["token_count"] for row in appended),
            record_count=len(appended),
        )
        return appended

    def wait_for(self, artifact_ids: list[str]) -> float:
        start = time.time()
        with self._lock:
            futures = [self._futures[artifact_id] for artifact_id in artifact_ids]
        for future in futures:
            future.result()
        return time.time() - start

    def wait_all(self) -> None:
        with self._lock:
            futures = list(self._futures.values())
        for future in futures:
            future.result()

    def close(self) -> None:
        self.executor.shutdown(wait=True)

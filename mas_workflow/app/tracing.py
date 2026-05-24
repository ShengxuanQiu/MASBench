"""MASBench-Arch week-1 trace, token, latency, and snapshot helpers."""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "masbench_arch_trace_v0.1"


class TraceHookManager:
    """Small hook bus for future workflow/motif composition.

    Exporters and probes can subscribe to all emitted events without changing
    topology code. The JSONL event stream remains the canonical trace.
    """

    def __init__(self) -> None:
        self._hooks: list[Any] = []

    def register(self, callback: Any) -> None:
        self._hooks.append(callback)

    def emit(self, event: dict[str, Any]) -> None:
        for callback in list(self._hooks):
            callback(event)


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def utc_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def stable_hash(value: Any) -> str:
    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8", errors="replace")
    else:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class TokenCounter:
    """Best-effort tokenizer adapter with an explicit fallback marker."""

    def __init__(self, model: str | None = None) -> None:
        self.source = "char_heuristic"
        self._encoder: Any = None
        try:
            import tiktoken  # type: ignore

            self._encoder = tiktoken.encoding_for_model(model or "gpt-4o-mini")
            self.source = "tokenizer"
        except Exception:
            try:
                from transformers import AutoTokenizer  # type: ignore

                self._encoder = AutoTokenizer.from_pretrained(model or "gpt2")
                self.source = "tokenizer"
            except Exception:
                self._encoder = None

    def count(self, text: str | None) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass
        return max(1, (len(text) + 3) // 4)


_DEFAULT_COUNTER = TokenCounter()


def estimate_tokens(text: str | None) -> int:
    return _DEFAULT_COUNTER.count(text)


def token_count_source() -> str:
    return _DEFAULT_COUNTER.source


def latency_delay(profile: str, rng: random.Random, scale: float = 1.0) -> float:
    profile = (profile or "none").lower()
    if profile == "none":
        delay = 0.0
    elif profile == "fast":
        delay = rng.uniform(0.1, 0.5)
    elif profile == "medium":
        delay = rng.uniform(1.0, 3.0)
    elif profile == "slow":
        delay = rng.uniform(5.0, 10.0)
    elif profile == "heavy_tail":
        delay = rng.uniform(10.0, 30.0) if rng.random() < 0.15 else rng.uniform(1.0, 3.0)
    else:
        delay = 0.0
    return max(0.0, delay * float(scale))


@dataclass
class TraceContext:
    run_id: str
    topology: str
    topology_role: str
    instance_id: str
    task_source: str
    workflow_id: str
    trace_path: Path
    summary_path: Path
    random_seed: int = 42
    trace_level: str = "arch"
    export_views: bool = True
    record_model_outputs: bool = False
    model_outputs_path: Path | None = None
    collect_backend_metrics: bool = False
    backend_metrics_path: Path | None = None
    mode: str = "topology"
    motif_name: str = ""
    motif_instance_id: str = ""
    parent_motif_id: str = ""
    composed_from_topologies: list[str] = field(default_factory=list)
    environment_id: str = field(default_factory=lambda: stable_hash({"cwd": str(Path.cwd()), "ts": now_ts()})[:16])
    start_perf: float = field(default_factory=time.perf_counter)
    events: list[dict[str, Any]] = field(default_factory=list)
    model_outputs: list[dict[str, Any]] = field(default_factory=list)
    backend_metrics_summary: dict[str, Any] = field(default_factory=dict)
    hooks: TraceHookManager = field(default_factory=TraceHookManager)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, **fields: Any) -> dict[str, Any]:
        with self.lock:
            event = {
                "schema_version": SCHEMA_VERSION,
                "event_id": fields.pop("event_id", str(uuid.uuid4())),
                "event_type": fields.pop("event_type", "unknown"),
                "timestamp": fields.pop("timestamp", utc_iso()),
                "relative_time_sec": round(time.perf_counter() - self.start_perf, 6),
                "run_id": self.run_id,
                "topology": self.topology,
                "topology_role": self.topology_role,
                "instance_id": self.instance_id,
                "task_source": self.task_source,
                "workflow_id": self.workflow_id,
                "mode": fields.pop("mode", self.mode),
                "motif_name": fields.pop("motif_name", self.motif_name),
                "motif_instance_id": fields.pop("motif_instance_id", self.motif_instance_id),
                "parent_motif_id": fields.pop("parent_motif_id", self.parent_motif_id),
                "composed_from_topologies": fields.pop("composed_from_topologies", list(self.composed_from_topologies)),
                "node_id": fields.pop("node_id", ""),
                "node_name": fields.pop("node_name", ""),
                "node_type": fields.pop("node_type", "workflow"),
                "round_id": fields.pop("round_id", None),
                "manager_round_id": fields.pop("manager_round_id", None),
                "peer_round_id": fields.pop("peer_round_id", None),
                "attempt_id": fields.pop("attempt_id", 0),
                "retry_count": fields.pop("retry_count", 0),
                "parents": fields.pop("parents", []),
                "children": fields.pop("children", []),
                "motif_tags": fields.pop("motif_tags", []),
                "parallel_group": fields.pop("parallel_group", None),
                "criticality": fields.pop("criticality", "unknown"),
                "status": fields.pop("status", "success"),
                "duration_sec": fields.pop("duration_sec", 0.0),
                "duration_source": fields.pop("duration_source", "mock_measured"),
                "replay_policy": fields.pop("replay_policy", "synthetic_only"),
                "environment_id": self.environment_id,
                "random_seed": self.random_seed,
                "trace_level": self.trace_level,
                "extra": fields.pop("extra", {}),
            }
            event.update(fields)
            if self.trace_level == "basic":
                self._redact_basic(event)
            self.events.append(event)
            self.hooks.emit(event)
        return event

    def record_model_output(
        self,
        *,
        event: dict[str, Any],
        output_text: str,
        input_text: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        response_metadata: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Store full model text outside the canonical JSONL event stream.

        The JSONL trace remains compact and simulation-oriented. When enabled,
        this sidecar keeps semantic outputs for inspection/replay research and
        links them back to the LLM event through stable ids and hashes.
        """
        if not self.record_model_outputs:
            return None
        output_hash = stable_hash(output_text)
        prompt_hash = stable_hash(input_text or "")
        artifact_id = f"model_output_{output_hash[:16]}"
        path = self.model_outputs_path or self.trace_path.with_name(f"{self.trace_path.stem}_model_outputs.json")
        record = {
            "schema_version": SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "artifact_type": "model_output",
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "run_id": self.run_id,
            "topology": self.topology,
            "topology_role": self.topology_role,
            "instance_id": self.instance_id,
            "task_source": self.task_source,
            "workflow_id": self.workflow_id,
            "node_id": event.get("node_id"),
            "node_name": event.get("node_name"),
            "node_type": event.get("node_type"),
            "agent_id": event.get("agent_id"),
            "agent_role": event.get("agent_role"),
            "round_id": event.get("round_id"),
            "manager_round_id": event.get("manager_round_id"),
            "peer_round_id": event.get("peer_round_id"),
            "parallel_group": event.get("parallel_group"),
            "llm_mode": event.get("llm_mode"),
            "backend_base_url": event.get("backend_base_url"),
            "model": event.get("model"),
            "llm_request_id": event.get("llm_request_id"),
            "request_id_for_backend": event.get("request_id_for_backend"),
            "prompt_template": event.get("prompt_template"),
            "prompt_hash": event.get("prompt_hash") or prompt_hash,
            "input_hash": prompt_hash,
            "output_hash": output_hash,
            "input_chars": len(input_text or ""),
            "output_chars": len(output_text or ""),
            "input_tokens_est": estimate_tokens(input_text or ""),
            "output_tokens_est": estimate_tokens(output_text),
            "token_count_source": token_count_source(),
            "backend_prompt_tokens": event.get("backend_prompt_tokens"),
            "backend_completion_tokens": event.get("backend_completion_tokens"),
            "backend_total_tokens": event.get("backend_total_tokens"),
            "backend_finish_reason": event.get("backend_finish_reason"),
            "timestamp": event.get("timestamp"),
            "relative_time_sec": event.get("relative_time_sec"),
            "duration_sec": event.get("duration_sec"),
            "duration_source": event.get("duration_source"),
            "replay_policy": event.get("replay_policy"),
            "output_text": output_text,
            "response_metadata": response_metadata or {},
            "extra": extra or {},
        }
        if system_prompt is not None:
            record["system_prompt_hash"] = stable_hash(system_prompt)
            record["system_prompt_chars"] = len(system_prompt)
            record["system_prompt_tokens_est"] = estimate_tokens(system_prompt)
        if user_prompt is not None:
            record["user_prompt_hash"] = stable_hash(user_prompt)
            record["user_prompt_chars"] = len(user_prompt)
            record["user_prompt_tokens_est"] = estimate_tokens(user_prompt)
        with self.lock:
            self.model_outputs.append(record)
            event["model_output_artifact_id"] = artifact_id
            event["model_output_path"] = str(path)
            event["model_output_hash"] = output_hash
        return record

    def save(self) -> Path:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("w", encoding="utf-8") as f:
            for event in self.events:
                f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        if self.record_model_outputs and self.model_outputs:
            path = self.model_outputs_path or self.trace_path.with_name(f"{self.trace_path.stem}_model_outputs.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_id,
                "topology": self.topology,
                "instance_id": self.instance_id,
                "trace_path": str(self.trace_path),
                "record_count": len(self.model_outputs),
                "records": self.model_outputs,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return self.trace_path

    def _redact_basic(self, event: dict[str, Any]) -> None:
        """Keep simulator fields while dropping bulky/debug-only payloads."""
        for key in ("request_metadata",):
            if key in event:
                event[key] = {"redacted": True, "hash": stable_hash(event[key])}
        extra = event.get("extra")
        if isinstance(extra, dict) and extra:
            event["extra"] = {"redacted": True, "hash": stable_hash(extra)}


def make_event(
    *,
    agent_name: str,
    event_type: str,
    parents: list[str] | None = None,
    children: list[str] | None = None,
    input_text: str = "",
    output_text: str = "",
    duration_sec: float = 0.0,
    status: str = "ok",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Legacy compatibility for the old demo workflow."""
    return {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "agent_name": agent_name,
        "event_type": event_type,
        "parents": parents or [],
        "children": children or [],
        "input_chars": len(input_text or ""),
        "output_chars": len(output_text or ""),
        "estimated_input_tokens": estimate_tokens(input_text),
        "estimated_output_tokens": estimate_tokens(output_text),
        "duration_sec": round(duration_sec, 6),
        "status": status,
        "extra": extra or {},
    }


def save_trace(log_dir: str | Path, trace_events: list[dict[str, Any]], run_id: str | None = None) -> Path:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / f"run_{run_id or now_ts()}.json"
    output_path.write_text(json.dumps(trace_events, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def save_run_result(log_dir: str | Path, result: dict[str, Any], run_id: str | None = None) -> Path:
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / f"result_{run_id or now_ts()}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.duration = time.perf_counter() - self.start

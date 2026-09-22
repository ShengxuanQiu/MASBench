"""Official dependency-aware Semantic Trace replay."""
from __future__ import annotations

import copy
import json
import threading
import time
import urllib.request
import random
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .model import OperationRecord, SemanticTrace, canonical_json, sha256_json
from .scenario import ScenarioManifest, SystemConfig
from .store import TraceBundle
from .validators import RunValidator, ValidationReport


@dataclass
class BackendResult:
    output_text: str
    output_token_ids: list[int] | None
    output_length: int
    observations: dict[str, Any] = field(default_factory=dict)


class ReplayBackend(Protocol):
    capabilities: dict[str, bool]
    def generate(self, request: dict[str, Any], operation: OperationRecord, mode: str) -> BackendResult: ...


class MockExactBackend:
    """Synthetic conformance backend; never represents real hardware support."""
    capabilities = {"supports_length_lock": True, "supports_token_lock": True}

    def generate(self, request: dict[str, Any], operation: OperationRecord, mode: str) -> BackendResult:
        recorded = list(operation.llm.get("recorded_output_token_ids") or [])
        length = int(operation.llm["recorded_output_length"] or len(recorded))
        tokens = recorded if mode == "token_locked" else [((i * 31) + 7) % 251 for i in range(length)]
        return BackendResult("synthetic-conformance-output", tokens, len(tokens), {"source": "synthetic"})


class VLLMOpenAIBackend:
    """vLLM OpenAI-compatible adapter with verified length-lock requests.

    `ignore_eos` and `min_tokens == max_tokens` are vLLM extensions. The run is
    still invalidated if the server reports a different completion length.
    Token locking is intentionally unsupported.
    """
    capabilities = {"supports_length_lock": True, "supports_token_lock": False}

    def __init__(self, base_url: str, api_key: str = "EMPTY", timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, request: dict[str, Any], operation: OperationRecord, mode: str) -> BackendResult:
        if mode != "length_locked":
            raise NotImplementedError("vLLM OpenAI adapter does not support token_locked replay")
        target = int(operation.llm["recorded_output_length"])
        payload = copy.deepcopy(request)
        payload.update({"stream": False, "max_tokens": target, "min_tokens": target, "ignore_eos": True})
        # vLLM rejects stream_options on non-streaming requests. The source
        # trace may legitimately record them, but replay controls streaming.
        payload.pop("stream_options", None)
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/chat/completions", data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.api_key})
        started = time.perf_counter()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=self.timeout) as response:
            value = json.loads(response.read())
        elapsed = time.perf_counter() - started
        usage = value.get("usage") or {}
        text = ((value.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        length = int(usage.get("completion_tokens") or 0)
        return BackendResult(text, None, length, {"wall_time_sec": elapsed, "usage": usage, "backend_response_id": value.get("id")})


def materialize_request(operation: OperationRecord, scenario: ScenarioManifest) -> tuple[dict[str, Any], str]:
    request = scenario.materialize_request(operation)
    request_hash = sha256_json(request)
    contract = scenario.resolved_request_contracts.get(operation.operation_id, {})
    if contract.get("request_hash") != request_hash:
        raise ValueError(f"Resolved request contract mismatch for {operation.operation_id}")
    return request, request_hash


class ReplayExecutor:
    def __init__(self, trace: SemanticTrace, scenario: ScenarioManifest, system: SystemConfig,
                 backend: ReplayBackend, artifact_store: Any | None = None, max_workers: int = 64):
        self.trace, self.scenario, self.system, self.backend = trace, scenario, system, backend
        self.artifact_store, self.max_workers = artifact_store, max_workers

    def run(self) -> dict[str, Any]:
        self.system.capabilities = dict(self.backend.capabilities)
        preflight = RunValidator().validate(self.trace, self.scenario, self.system)
        if not preflight.run_valid:
            return {"schema_version": "masbench.run/1.0.0", **preflight.to_dict(), "scenario_hash": self.scenario.scenario_hash,
                    "workload_hash": self.trace.metadata.workload_hash, "system_hash": self.system.system_hash(), "operations": []}
        operations = {x.operation_id: x for x in self.trace.operations}
        tasks = {x.task_id: x for x in self.trace.tasks}
        ordered_tasks = sorted(tasks)
        arrival_mode = self.scenario.root_arrival["mode"]
        task_arrivals: dict[str, float | None] = {}
        if arrival_mode == "trace_offsets":
            task_arrivals = {task_id: tasks[task_id].root_arrival_offset for task_id in tasks}
        elif arrival_mode == "fixed_rate":
            rate = float(self.scenario.root_arrival.get("arrival_rate") or 0)
            if rate <= 0: raise ValueError("fixed_rate requires a positive arrival_rate")
            task_arrivals = {task_id: index / rate for index, task_id in enumerate(ordered_tasks)}
        elif arrival_mode == "poisson":
            rate = float(self.scenario.root_arrival.get("arrival_rate") or 0)
            if rate <= 0: raise ValueError("poisson requires a positive arrival_rate")
            rng = random.Random(int(self.scenario.root_arrival["random_seed"])); current = 0.0
            for index, task_id in enumerate(ordered_tasks):
                if index: current += rng.expovariate(rate)
                task_arrivals[task_id] = current
        else:
            task_arrivals = {task_id: None for task_id in tasks}
        admitted_tasks: set[str] = set()
        active_tasks: set[str] = set()
        pending = set(operations)
        completed: set[str] = set()
        active: dict[Any, str] = {}
        result_records: list[dict[str, Any]] = []
        external_path_delay: dict[str, float] = {}
        task_start: dict[str, float] = {}
        task_finish: dict[str, float] = {}
        operation_finish: dict[str, float] = {}
        run_start = time.perf_counter()
        lock = threading.Lock()

        def execute(op: OperationRecord, path_external_delay: float) -> dict[str, Any]:
            predecessor_finished = time.perf_counter()
            if op.controlled_external_delay:
                time.sleep(op.controlled_external_delay)
            ready = time.perf_counter()
            with lock:
                task_start.setdefault(op.task_id, ready)
            record: dict[str, Any] = {"operation_id": op.operation_id, "task_id": op.task_id,
                "ready_time": ready - run_start, "controlled_external_delay": op.controlled_external_delay,
                "controlled_external_delay_on_path": path_external_delay,
                "source_observations_used_for_scheduling": False}
            started = time.perf_counter()
            if op.operation_type == "llm":
                request, request_hash = materialize_request(op, self.scenario)
                response = self.backend.generate(request, op, self.scenario.replay["fidelity_mode"])
                record.update({"request_hash": request_hash, "output_length": response.output_length,
                               "target_observations": response.observations})
                expected = int(op.llm["recorded_output_length"])
                if response.output_length != expected:
                    record["output_length_mismatch"] = {"operation_id": op.operation_id, "expected": expected, "actual": response.output_length}
                if self.scenario.replay["fidelity_mode"] == "token_locked" and response.output_token_ids != op.llm.get("recorded_output_token_ids"):
                    record["token_mismatch"] = True
                record["generated_output_used_downstream"] = False
            elif op.operation_type in {"tool", "transform"}:
                typed = op.tool or op.transform
                if self.artifact_store and typed.get("recorded_result_ref"):
                    self.artifact_store.get_bytes(typed["recorded_result_ref"], typed.get("content_hash"))
                record["recorded_result_used"] = True
                record["live_external_service_called"] = False
            else:
                record["recorded_state_transition_used"] = True
            finished = time.perf_counter()
            record.update({"start_time": started - run_start, "finish_time": finished - run_start,
                           "target_duration_sec": finished - started, "predecessor_release_time": predecessor_finished - run_start})
            return record

        error: Exception | None = None
        with ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, len(operations)))) as pool:
            while pending or active:
                now = time.perf_counter() - run_start
                if arrival_mode == "fixed_concurrency":
                    limit = int(self.scenario.root_arrival.get("concurrency") or 0)
                    if limit < 1: raise ValueError("fixed_concurrency requires positive concurrency")
                    for task_id in ordered_tasks:
                        if len(active_tasks) >= limit: break
                        if task_id not in admitted_tasks:
                            admitted_tasks.add(task_id); active_tasks.add(task_id); task_arrivals[task_id] = now
                else:
                    for task_id, arrival in task_arrivals.items():
                        if arrival is not None and now >= arrival:
                            admitted_tasks.add(task_id); active_tasks.add(task_id)
                for oid in sorted(list(pending)):
                    op = operations[oid]
                    if op.dependencies <= completed and op.task_id in admitted_tasks:
                        path_delay = max((external_path_delay[x] for x in op.dependencies), default=0.0) + op.controlled_external_delay
                        external_path_delay[oid] = path_delay
                        active[pool.submit(execute, op, path_delay)] = oid
                        pending.remove(oid)
                if not active:
                    if pending:
                        future_arrivals = [float(task_arrivals[operations[x].task_id]) for x in pending
                                           if task_arrivals[operations[x].task_id] is not None
                                           and operations[x].task_id not in admitted_tasks]
                        if future_arrivals:
                            time.sleep(min(0.01, max(0.0, min(future_arrivals) - now)))
                        else:
                            time.sleep(0.001)
                        continue
                    break
                finished, _ = wait(active, return_when=FIRST_COMPLETED, timeout=0.05)
                for future in finished:
                    oid = active.pop(future)
                    try:
                        record = future.result()
                    except Exception as exc:
                        error = exc
                        continue
                    result_records.append(record)
                    completed.add(oid)
                    operation_finish[oid] = run_start + record["finish_time"]
                    task_id = operations[oid].task_id
                    if set(tasks[task_id].terminal_operation_ids) <= completed:
                        task_finish[task_id] = max(operation_finish[x] for x in tasks[task_id].terminal_operation_ids)
                        active_tasks.discard(task_id)
                if error:
                    for future in active:
                        future.cancel()
                    break
        mismatches = [x["output_length_mismatch"] for x in result_records if "output_length_mismatch" in x]
        completed_tasks = [task_id for task_id, task in tasks.items() if set(task.terminal_operation_ids) <= completed]
        run_info = {"cancelled": False, "completed_task_ids": completed_tasks,
                    "output_length_mismatches": mismatches, "truncated_input": False, "context_overflow": False}
        validation = RunValidator().validate(self.trace, self.scenario, self.system, run_info)
        if any(x.get("token_mismatch") for x in result_records):
            validation.add("OUTPUT_TOKEN_MISMATCH", "Token-locked output IDs differ from the trace")
        task_records = []
        controlled_by_task = {}
        for task_id, task in tasks.items():
            finished_terminals = [oid for oid in task.terminal_operation_ids if oid in operation_finish]
            critical_terminal = max(finished_terminals, key=operation_finish.get) if finished_terminals else None
            controlled_by_task[task_id] = external_path_delay.get(critical_terminal, 0.0) if critical_terminal else 0.0
        for task_id in tasks:
            arrival_offset = float(task_arrivals[task_id] or 0.0)
            arrival = run_start + arrival_offset
            finish = task_finish.get(task_id)
            raw = finish - arrival if finish else None
            task_records.append({"task_id": task_id, "attempted": True, "successful": task_id in completed_tasks and error is None,
                                 "arrival_time": arrival_offset, "finish_time": finish - run_start if finish else None,
                                 "raw_wall_clock_latency_sec": raw,
                                 "controlled_external_delay_sec": controlled_by_task[task_id],
                                 "task_latency_sec": max(0.0, raw - controlled_by_task[task_id]) if raw is not None else None})
        return {"schema_version": "masbench.run/1.0.0", **validation.to_dict(), "scenario_hash": self.scenario.scenario_hash,
                "workload_hash": self.trace.metadata.workload_hash, "system_hash": self.system.system_hash(),
                "operations": sorted(result_records, key=lambda x: x["operation_id"]), "tasks": task_records,
                "measurement_duration_sec": time.perf_counter() - run_start,
                "error": str(error) if error else None,
                "replay_invariants": {"dependency_release_uses_target_completion": True,
                    "source_timestamps_used": False, "generated_output_used_downstream": False,
                    "live_external_tools_called": False},
                "resolved_root_arrival": {"mode": arrival_mode, "random_seed": self.scenario.root_arrival["random_seed"],
                                          "task_arrival_offsets": task_arrivals}}

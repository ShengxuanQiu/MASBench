"""Fixed realized work: schedule recorded requests, never rerun task decisions."""
from __future__ import annotations

import json
import time
import threading
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from .execution_graph import ExecutionGraph
from .llm_backends import build_llm_backend
from .runtime import WorkloadConfig, WorkloadRuntime
from .runtime_factory import build_trace_context
from .search_providers import build_search_provider


@dataclass(frozen=True)
class PreparedReplay:
    path: str
    events: tuple
    graph: ExecutionGraph
    sha256: str


def prepare_replay(path, *, max_operations=4096):
    raw = Path(path).read_bytes()
    events = tuple(json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip())
    graph = ExecutionGraph.from_events(events, replay=True)
    if len(graph.operations) > max_operations:
        raise ValueError("Replay graph exceeds configured operation limit")
    return PreparedReplay(str(path), events, graph, hashlib.sha256(raw).hexdigest())


def replay_capabilities(backend):
    # No current adapter enforces exact output tokens (max_tokens is just a cap).
    return {"mode": "best-effort", "request_payload": "fixed_except_model",
            "dependencies": "fixed", "control_path": "recorded", "tool_results": "snapshots",
            "tool_duration": "recorded_client_duration", "output_length": "best-effort",
            "reason": "Current adapters cannot enforce the recorded exact output token count.",
            "output_token_sequence": "uncontrolled", "input_token_ids": "unavailable",
            "prefix_cache": "request_text_fixed; generated-prefix equivalence not guaranteed",
            "cache_initial_state": "uncontrolled", "model_tokenizer_equivalence": "not_verified",
            "backend": getattr(backend, "mode", type(backend).__name__)}


def replay_trace(path, deployment, *, trace_dir="traces/replay", strict=False, backend=None,
                 llm_slots=None, export_views=True, max_operations=4096, journal=False, prepared=None):
    prepared = prepared or prepare_replay(path, max_operations=max_operations)
    if Path(prepared.path).resolve() != Path(path).resolve():
        raise ValueError("Prepared replay path mismatch")
    events, graph = prepared.events, prepared.graph
    if len(graph.operations) > max_operations:
        raise ValueError("Replay graph exceeds configured operation limit")
    backend = backend or build_llm_backend(deployment.backend, model=deployment.model,
                                           backend_base_url=deployment.endpoint,
                                           max_output_tokens=deployment.generation.get("max_tokens", 256))
    capability = replay_capabilities(backend)
    if strict:
        raise ValueError("Strict replay unavailable: " + capability["reason"])
    run_id = "replay_" + uuid4().hex
    config = WorkloadConfig(topology_name="replay", run_id=run_id, task_id="recorded", instance_id="recorded",
                            query="", llm_mode=deployment.backend, model=deployment.model,
                            backend_base_url=deployment.endpoint, trace_dir=Path(trace_dir),
                            max_concurrent_llm_calls=deployment.concurrency, export_trace_views=export_views)
    trace = build_trace_context(config)
    trace.journal_enabled = journal
    trace.canonical_enabled = True
    trace.execution_graph = ExecutionGraph()
    runtime = WorkloadRuntime(config, llm=backend, search_provider=build_search_provider(
        provider_name="synthetic", tool_mode="synthetic", replay_snapshot_dir=None,
        latency_profile="none", latency_scale=1.0, random_seed=42), trace=trace)
    # Readiness is exposed before this shared LLM admission limit. Tool snapshots
    # never occupy LLM slots; each DAG operation has an independent scheduler task.
    runtime._llm_slots = llm_slots if llm_slots is not None else threading.BoundedSemaphore(deployment.concurrency)
    ids = {oid: run_id + ":" + oid for oid in graph.operations}
    artifact_ids = {aid: run_id + ":" + aid for aid in graph.artifacts}
    def identities(op):
        return {key: run_id + ":" + value if value and key.endswith("_id") else value for key, value in op.identities.items()}
    runtime.workflow_start()
    trace.emit(event_type="replay_manifest", source_run_id=graph.run_id, capabilities=capability,
               deployment=asdict(deployment), source_trace=str(path), source_operation_ids=ids,
               generation_policy="recorded_request_generation; deployment generation ignored",
               source_control_decisions=graph.decisions,
               source_trace_sha256=prepared.sha256)

    incoming, consumed, produced, decisions = (defaultdict(list) for _ in range(4))
    for src, dst, kind in sorted(graph.edges):
        incoming[dst].append((src, kind))
    for aid, consumer in graph.consumptions:
        consumed[consumer].append(aid)
    for aid, artifact in graph.artifacts.items():
        produced[artifact["producer"]].append((aid, artifact))
    for event in graph.decisions:
        decisions[event.get("node_id")].append(event)

    def execute(oid):
        op = graph.operations[oid]
        identity = identities(op)
        parents = [ids[p] for p in sorted(op.parents)]
        for src, kind in incoming[oid]:
            trace.emit(event_type="dependency", src=ids[src], dst=ids[oid], dependency_kind=kind, **identity)
        for aid in consumed[oid]:
            trace.emit(event_type="artifact_consume", node_id=ids[oid], artifact_id=artifact_ids[aid], **identity)
        if op.kind == "llm":
            payload = deepcopy(op.payload)
            payload["model"] = deployment.model
            # Use the full recorded wire payload, including all generation fields.
            messages = payload["messages"]
            if len(messages) != 2 or [m["role"] for m in messages] != ["system", "user"]:
                raise ValueError("This adapter requires a recorded system/user message pair")
            class RecordedBackend:
                model = deployment.model
                base_url = deployment.endpoint
                def invoke(self, system, user, metadata):
                    metadata["_request_payload"] = payload
                    metadata["_fixed_payload"] = True
                    return backend.invoke(system, user, metadata)
            runtime.call_llm(node_id=ids[oid], node_name=op.metadata.get("role_slot", "replay"), node_type="llm",
                             agent_role=op.metadata.get("agent_role", "replay"), prompt_template="recorded_payload",
                             system_prompt=messages[0]["content"], user_prompt=messages[1]["content"], parents=parents,
                             llm_override=RecordedBackend(),
                             extra_metadata={**{k: v for k, v in op.metadata.items() if k not in {"request_id_for_backend", "agent_id"}},
                                             **identity, "request_max_output_tokens": payload["max_tokens"],
                                             "recorded_request_payload": payload,
                                             "source_operation_id": oid, "source_request_id": op.request_id,
                                             "recorded_output_tokens": op.output_tokens})
        else:
            trace.emit(event_type="operation_start", node_id=ids[oid], operation_kind="tool", parents=parents, **identity)
            started = time.perf_counter()
            time.sleep(max(0.0, op.duration_sec))
            trace.emit(event_type="operation_finish", node_id=ids[oid], tool_snapshot=op.snapshot,
                       duration_sec=time.perf_counter() - started, source_operation_id=oid, **identity)
        for aid, artifact in produced[oid]:
            trace.emit(event_type="artifact_created", node_id=ids[oid], artifact_id=artifact_ids[aid],
                       source_artifact_id=aid, content=artifact["content"], output_hash=artifact["hash"],
                       replay_content_source="recorded_snapshot", **identity)
        for event in decisions[oid]:
            trace.emit(event_type="control_decision", node_id=ids[oid], decision=event["decision"],
                       source_event_id=event["event_id"], replay_content_source="recorded_decision", **identity)

    done, pending, active = set(), set(graph.operations), {}
    error = None
    barriers = [e for e in events if e.get("canonical_type") == "barrier_sync"]
    emitted_barriers = set()
    try:
        with ThreadPoolExecutor(max_workers=max(1, len(graph.operations))) as pool:
            while pending or active:
                if error is None:
                    for oid in sorted(pending):
                        if graph.operations[oid].parents <= done:
                            active[pool.submit(execute, oid)] = oid
                            pending.remove(oid)
                if not active:
                    if pending and error is None:
                        error = RuntimeError("Unreleased dependencies remain")
                    break
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    oid = active.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        error = error or exc
                    else:
                        done.add(oid)
                for barrier in barriers:
                    if barrier["event_id"] not in emitted_barriers and set(barrier.get("waiting_for_nodes", [])) <= done:
                        emitted_barriers.add(barrier["event_id"])
                        trace.emit(event_type="barrier", node_type="barrier", node_id=run_id + ":" + barrier["node_id"],
                                   waiting_for_nodes=[ids[p] for p in barrier.get("waiting_for_nodes", [])],
                                   source_event_id=barrier["event_id"], replay_content_source="recorded_sync")
    finally:
        summary = runtime.workflow_end("", status="failed" if error else "completed")
    if error:
        raise RuntimeError(f"Replay failed; partial trace saved to {trace.trace_path}") from error
    trace.execution_graph.validate(replay=True)
    if len(done) != len(graph.operations):
        raise RuntimeError("Replay did not complete all recorded operations")
    graph_path = trace.trace_path.with_name(run_id + "_execution_graph.json")
    graph_path.write_text(json.dumps(trace.execution_graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    summary.update(replay_capabilities=capability, replayed_operations=len(done), source_run_id=graph.run_id,
                   execution_graph_path=str(graph_path), canonical_trace_schema="masbench_execution_v1")
    trace.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, trace

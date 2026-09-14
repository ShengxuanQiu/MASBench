from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from dataclasses import replace

import pytest

from app.benchmark import build_experiment
from app.execution_graph import ExecutionGraph, SCHEMA
from app.llm_backends import MockLLM
from app.replay import replay_trace
from app.specs import (DeploymentSpec, ExperimentConfig, MotifSpec, RoleTask, StageSpec,
                       StructureSpec, TaskBinding, WorkflowSpec, load_experiment)


def experiment(family="ParallelAggregate", width=2, stages=None, task=None, **motif):
    return ExperimentConfig(
        StructureSpec({"m": MotifSpec(family, width=width, **motif)},
                      WorkflowSpec(tuple(stages or [StageSpec("one", "m")]))),
        task or TaskBinding("Solve the test task."), DeploymentSpec(generation={"max_tokens": 77, "temperature": 0.7, "top_p": 0.9, "seed": 12}))


def record(tmp_path, exp=None, backend=None):
    runner = build_experiment(exp or experiment(), trace_dir=tmp_path)
    runner.config.export_trace_views = False
    if backend:
        runner.llm = backend
    summary = runner.run()
    assert summary["status"] in {"completed", "accepted"}
    return runner


def test_independent_factor_replacement(tmp_path):
    base = experiment()
    changed_task = replace(base, task=TaskBinding("Different domain", {"Worker": RoleTask("Think about agriculture")}))
    changed_structure = replace(base, structure=replace(base.structure, motifs={"m": MotifSpec("ParallelAggregate", width=4)}))
    changed_deployment = replace(base, deployment=replace(base.deployment, model="another", concurrency=1))
    runners = [record(tmp_path, exp) for exp in (base, changed_task, changed_structure, changed_deployment)]
    assert [len(r.trace.execution_graph.operations) for r in runners] == [3, 3, 5, 3]
    assert base.structure == changed_task.structure == changed_deployment.structure
    assert base.task == changed_structure.task == changed_deployment.task
    assert changed_deployment.compile() == base.compile()
    assert any("Different domain" in json.dumps(op.payload) for op in runners[1].trace.execution_graph.operations.values())
    assert all(op.payload["model"] == "another" for op in runners[3].trace.execution_graph.operations.values())


@pytest.mark.parametrize("factory,kwargs", [
    (MotifSpec, {"family": "ParallelAggregate", "prompt": "forbidden"}),
    (MotifSpec, {"family": "ParallelAggregate", "backend": "mock"}),
    (TaskBinding, {"task_input": "x", "width": 4}),
    (RoleTask, {"model": "forbidden"}),
    (DeploymentSpec, {"depends_on": ["a"]}),
    (DeploymentSpec, {"generation": {"messages": []}}),
])
def test_factor_boundaries_reject_leaks(factory, kwargs):
    with pytest.raises((ValueError, TypeError)):
        factory(**kwargs)


def test_dag_fan_out_fan_in_and_multiple_artifacts(tmp_path):
    # Reverse declaration order proves execution does not depend on list order.
    stages = [StageSpec("join", "m", ("left", "right"), {"context": ["left.result", "right.result"]}),
              StageSpec("right", "m", ("root",)), StageSpec("left", "m", ("root",)), StageSpec("root", "m")]
    runner = record(tmp_path, experiment(width=1, stages=stages))
    events = runner.trace.events
    starts = {e["stage_id"]: e for e in events if e["event_type"] == "stage_start"}
    finishes = {e["stage_instance_id"]: e for e in events if e["event_type"] == "stage_finish"}
    for child, parents in {"join": ["left", "right"], "left": ["root"], "right": ["root"]}.items():
        assert all(starts[child]["relative_time_sec"] >= finishes[starts[p]["stage_instance_id"]]["relative_time_sec"] for p in parents)
    join_calls = [op for op in runner.trace.execution_graph.operations.values() if op.identities["stage_instance_id"] == starts["join"]["stage_instance_id"]]
    inputs = json.loads(join_calls[0].payload["messages"][1]["content"])["inputs"]
    assert len(inputs) == 2
    assert len(join_calls[0].parents) == 4  # all operations of both prerequisite stages


def test_independent_stages_actually_overlap(tmp_path):
    rendezvous = threading.Barrier(2)
    class Overlap(MockLLM):
        def invoke(self, system, user, metadata):
            if metadata["role_slot"] == "worker":
                rendezvous.wait(timeout=3)
            return super().invoke(system, user, metadata)
    record(tmp_path, experiment(width=1, stages=[StageSpec("a", "m"), StageSpec("b", "m")]), Overlap())


@pytest.mark.parametrize("stages", [
    [StageSpec("x", "m", ("missing",))],
    [StageSpec("x", "m", ("y",)), StageSpec("y", "m", ("x",))],
    [StageSpec("x", "m"), StageSpec("x", "m")],
])
def test_invalid_workflow_rejected(stages):
    with pytest.raises(ValueError):
        experiment(stages=stages)


@pytest.mark.parametrize("family,options", [("DispatchExecute", {"dispatch": "route"}),
    ("ParallelAggregate", {"aggregation": "vote"}), ("EvaluateRefine", {}),
    ("PeerDeliberation", {"rounds": 2, "connectivity": "ring"})])
def test_trace_round_trip_provenance_and_ids(tmp_path, family, options):
    task = TaskBinding("Research", {"Worker": RoleTask(tools=("search",))})
    runner = record(tmp_path, experiment(family, task=task, **options))
    events = [json.loads(line) for line in runner.trace.trace_path.read_text().splitlines()]
    graph = ExecutionGraph.from_events(events, replay=True)
    assert graph.to_dict() == runner.trace.execution_graph.to_dict()
    assert all(op.kind in {"llm", "tool"} and op.identities["stage_instance_id"] and op.identities["motif_instance_id"] for op in graph.operations.values())
    assert all(op.identities["agent_instance_id"] for op in graph.operations.values() if op.kind == "llm")
    assert len({op.request_id for op in graph.operations.values() if op.kind == "llm"}) == sum(op.kind == "llm" for op in graph.operations.values())
    assert all(artifact["content"] is not None and artifact["producer"] in graph.operations for artifact in graph.artifacts.values())
    assert any(op.kind == "tool" and op.snapshot is not None for op in graph.operations.values())
    assert graph.consumptions
    for event in events:
        if event.get("canonical_schema") == SCHEMA:
            assert set(event["attributes"]) == {"operation", "information_flow", "serving_observation"}
            assert all(m["source"] in {"observed", "backend-reported", "estimated", "unavailable"} for m in event["attributes"]["serving_observation"].values())


def test_replay_waits_for_all_dependencies_and_ignores_new_decisions(tmp_path):
    runner = record(tmp_path / "record", experiment("EvaluateRefine", stages=[
        StageSpec("a", "m"), StageSpec("b", "m"), StageSpec("join", "m", ("a", "b"))]))
    graph = runner.trace.execution_graph
    source_calls = {oid: op for oid, op in graph.operations.items() if op.kind == "llm"}
    calls = {}
    lock = threading.Lock()
    class Adversarial(MockLLM):
        def invoke(self, system, user, metadata):
            oid = metadata["source_operation_id"]
            with lock:
                calls[oid] = {"start": time.perf_counter(), "payload": deepcopy(metadata["_request_payload"])}
            time.sleep(0.012 if metadata["role_slot"] == "evaluator" else 0.003)
            result = super().invoke(system, user, metadata)
            with lock:
                calls[oid]["end"] = time.perf_counter()
            return replace(result, content='{"decision":"revise","feedback":"NEW BRANCH MUST NOT RUN"}')
    summary, trace = replay_trace(runner.trace.trace_path, DeploymentSpec(),
                                  trace_dir=tmp_path / "replay", backend=Adversarial())
    assert set(calls) == set(source_calls)
    for oid, call in calls.items():
        expected = {**source_calls[oid].payload, "model": "mock-mas-model"}
        assert call["payload"] == expected
        for parent in source_calls[oid].parents:
            if parent in calls:
                assert call["start"] >= calls[parent]["end"]
    replay_graph = ExecutionGraph.from_events(trace.events, replay=True)
    assert len(replay_graph.operations) == len(graph.operations)
    assert len(replay_graph.edges) == len(graph.edges)
    assert summary["replay_capabilities"]["output_length"] == "best-effort"
    assert all("NEW BRANCH" not in str(op.payload) for op in replay_graph.operations.values())
    assert {a["content"] for a in replay_graph.artifacts.values()} == {a["content"] for a in graph.artifacts.values()}


def test_replay_tool_snapshot_and_sync(tmp_path):
    runner = record(tmp_path / "record", experiment("PeerDeliberation", width=3, connectivity="pairwise", rounds=1,
                    task=TaskBinding("Search", {"Worker": RoleTask(tools=("search",))})))
    _, trace = replay_trace(runner.trace.trace_path, DeploymentSpec(), trace_dir=tmp_path / "replay")
    graph = ExecutionGraph.from_events(trace.events, replay=True)
    finished = {e["operation_id"]: e["relative_time_sec"] for e in trace.events if e.get("canonical_type") in {"request_finish", "operation_finish"}}
    starts = {e["operation_id"]: e["relative_time_sec"] for e in trace.events if e.get("canonical_type") in {"request_ready", "operation_start"}}
    for oid, op in graph.operations.items():
        assert all(starts[oid] >= finished[parent] for parent in op.parents)
    for oid, op in runner.trace.execution_graph.operations.items():
        if op.kind == "tool":
            assert next(other for key, other in graph.operations.items() if key.endswith(":" + oid)).snapshot == op.snapshot


def test_strict_and_incomplete_replay_rejected_before_submission(tmp_path):
    runner = record(tmp_path)
    with pytest.raises(ValueError, match="Strict replay unavailable"):
        replay_trace(runner.trace.trace_path, DeploymentSpec(), strict=True)
    events = deepcopy(runner.trace.events)
    events = [e for e in events if e.get("canonical_type") != "request_finish"]
    with pytest.raises(ValueError, match="Incomplete"):
        ExecutionGraph.from_events(events, replay=True)


def test_provenance_corruption_rejected(tmp_path):
    runner = record(tmp_path)
    events = deepcopy(runner.trace.events)
    events = [e for e in events if e.get("canonical_type") != "artifact_produce"]
    with pytest.raises(ValueError, match="Dangling artifact"):
        ExecutionGraph.from_events(events, replay=True)


def test_generation_is_recorded_exactly(tmp_path):
    runner = record(tmp_path)
    for op in runner.trace.execution_graph.operations.values():
        assert op.payload["max_tokens"] == 77
        assert op.payload["temperature"] == 0.7
        assert op.payload["top_p"] == 0.9
        assert op.payload["seed"] == 12


def test_candidate_singleton_list_and_task_io_validation(tmp_path):
    exp = experiment("EvaluateRefine", stages=[StageSpec("first", "m"), StageSpec("second", "m", inputs={"candidate": ["first.result"]})])
    runner = record(tmp_path, exp)
    assert sum(op.metadata.get("role_slot") == "producer" for op in runner.trace.execution_graph.operations.values()) == 1
    bad = replace(experiment(), task=TaskBinding("x", {"Worker": RoleTask(input_schema={"required": ["not_present"]})}))
    runner = build_experiment(bad, trace_dir=tmp_path / "invalid")
    assert runner.run()["status"] == "failed"
    assert not runner.trace.execution_graph.operations


def test_ready_precedes_capacity_wait_and_limit_is_global(tmp_path):
    lock = threading.Lock()
    running, peak = 0, 0
    class Slow(MockLLM):
        def invoke(self, system, user, metadata):
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.025)
            try:
                return super().invoke(system, user, metadata)
            finally:
                with lock:
                    running -= 1
    exp = experiment(width=1, stages=[StageSpec("a", "m"), StageSpec("b", "m"), StageSpec("c", "m")])
    exp = replace(exp, deployment=replace(exp.deployment, concurrency=1))
    runner = record(tmp_path, exp, Slow())
    assert peak == 1
    submits = [e for e in runner.trace.events if e.get("canonical_type") == "request_submit"]
    assert any(e["request_submit_ts"] - e["request_ready_ts"] > 0.015 for e in submits)


def test_examples_and_canonical_json_schema(tmp_path):
    from pathlib import Path
    import jsonschema
    base = Path(__file__).parents[1] / "configs"
    exp = load_experiment(base / "benchmark/experiment.json")
    assert exp == load_experiment(base / "benchmark/experiment.yaml")
    runner = record(tmp_path, exp)
    schema = json.loads((base / "canonical_trace.schema.json").read_text())
    for event in runner.trace.events:
        if event.get("canonical_schema"):
            jsonschema.validate(event, schema)


def test_export_contains_exact_canonical_operations(tmp_path):
    from app.trace_export import events_to_arch_spans, spans_to_otel
    runner = record(tmp_path, experiment("PeerDeliberation", task=TaskBinding("Search", {"Worker": RoleTask(tools=("search",))})))
    spans = events_to_arch_spans(runner.trace.events)
    operations = [s for s in spans if s["kind"] in {"llm", "tool"}]
    assert {s["node_id"] for s in operations} == set(runner.trace.execution_graph.operations)
    assert len(operations) == len(runner.trace.execution_graph.operations)
    assert all(s["stage_instance_id"] for s in operations)
    assert len(spans_to_otel(spans, trace_id="a" * 32)) == len(spans)


def test_tool_failure_keeps_valid_partial_trace(tmp_path):
    runner = build_experiment(experiment(task=TaskBinding("Search", {"Worker": RoleTask(tools=("search",))})), trace_dir=tmp_path)
    class Broken:
        def search(self, *args, **kwargs):
            raise RuntimeError("tool failed")
    runner.search_provider = Broken()
    assert runner.run()["status"] == "failed"
    graph = ExecutionGraph.from_events(runner.trace.events)
    assert all(op.status == "failed" for op in graph.operations.values())


def test_openai_wire_payload_is_preserved_in_run_and_replay(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    bodies = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunk = {"id": "test", "choices": [{"delta": {"content": "different generated content"}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}}
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode())
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        deployment = DeploymentSpec(backend="openai_compatible", endpoint=f"http://127.0.0.1:{server.server_port}/v1",
                                    generation={"max_tokens": 17, "temperature": 0.71, "top_p": 0.83, "seed": 2026, "stop": ["STOP"]})
        runner = record(tmp_path / "record", replace(experiment(width=1), deployment=deployment))
        expected = [op.payload for op in runner.trace.execution_graph.operations.values()]
        assert bodies == expected
        bodies.clear()
        _, trace = replay_trace(runner.trace.trace_path, replace(deployment, generation={"max_tokens": 999, "temperature": 0}), trace_dir=tmp_path / "replay")
        assert bodies == expected
        assert [op.payload for op in trace.execution_graph.operations.values()] == expected
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

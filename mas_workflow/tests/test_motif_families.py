from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.llm_backends import MockLLM
from app.motifs import MOTIF_NAMES, build_workflow
from app.motifs.contracts import AgentInstance, RoleBinding, RoleSlot
from app.motifs.families import FamilyWorkload
from app.motifs.presets import PRESETS
from app.runtime import WorkloadConfig


def runner(tmp_path, spec, **kwargs):
    config = WorkloadConfig(topology_name=spec.get("family", "composition"), run_id="test", task_id="task",
                            instance_id="input", query="Compare two approaches to water conservation.",
                            trace_dir=tmp_path, search_provider="synthetic", export_trace_views=False,
                            num_agents=3, **kwargs)
    config.extra["workload_spec"] = spec
    return build_workflow(config)


class Scripted(MockLLM):
    def __init__(self, responses):
        super().__init__()
        self.responses = responses
        self.calls = []

    def invoke(self, system_prompt, user_prompt, metadata):
        self.calls.append((metadata.copy(), json.loads(user_prompt)))
        result = super().invoke(system_prompt, user_prompt, metadata)
        role = metadata["role_slot"]
        if role in self.responses:
            value = self.responses[role]
            value = value.pop(0) if isinstance(value, list) else value
            if isinstance(value, Exception):
                raise value
            return replace(result, content=value)
        return result


@pytest.mark.parametrize("family", MOTIF_NAMES)
def test_families_execute_and_identify_calls(tmp_path, family):
    workflow = runner(tmp_path, {"family": family})
    summary = workflow.run()
    assert summary["status"] in {"completed", "accepted"}
    assert summary["workload_schema"] == "motif_families_v1"
    calls = [e for e in workflow.trace.events if e["event_type"] == "llm_request_end"]
    assert len({e["node_id"] for e in calls}) == len(calls)
    assert all(e["agent_id"] == e["agent_instance_id"] != e["node_id"] for e in calls)
    assert all(e["role_slot"] and e["motif_instance_id"].startswith("motif_") for e in calls)
    assert summary["composed_from_topologies"] == []


def test_same_binding_is_not_shared_state():
    slot, binding = RoleSlot("worker", "Work"), RoleBinding("Work")
    a, b = AgentInstance(slot, binding, "a"), AgentInstance(slot, binding, "b")
    a.history.append({"output": "private"})
    assert a.instance_id != b.instance_id and b.history == []


@pytest.mark.parametrize("arguments,mode", [([], "motif"), (["--topology", "independent"], "topology"),
                                           (["--topology=single"], "topology"),
                                           (["--mode", "legacy_motif", "--motif", "coder_reviewer"], "legacy_motif")])
def test_cli_default_and_legacy_selection(monkeypatch, arguments, mode):
    from app.main import parse_args
    monkeypatch.setattr("sys.argv", ["app.main"] + arguments)
    assert parse_args().mode == mode


def test_composition_reuses_candidate_but_not_agents(tmp_path):
    workflow = runner(tmp_path, {"stages": [
        {"id": "first", "family": "parallel_aggregate", "width": 2},
        {"id": "second", "family": "evaluate_refine", "inputs": {"candidate": "first.result"}},
        {"id": "third", "family": "parallel_aggregate", "width": 2, "inputs": {"context": "second.result"}},
    ]})
    backend = Scripted({})
    workflow.llm = backend
    summary = workflow.run()
    assert summary["topology"] == "composition"
    assert len(summary["motif_results"]) == 3
    assert not any(m["role_slot"] == "producer" for m, _ in backend.calls)
    first_artifact = summary["motif_results"][0]["outputs"]["result"]
    evaluation = next((m, p) for m, p in backend.calls if m["role_slot"] == "evaluator")
    assert first_artifact["artifact_id"] in evaluation[0]["input_artifact_ids"]
    assert evaluation[1]["inputs"][0]["content"] == first_artifact["content"]
    workers = [m for m, _ in backend.calls if m["role_slot"] == "worker"]
    assert len({m["agent_instance_id"] for m in workers}) == 4
    assert len({m["motif_instance_id"] for m in workers}) == 2


@pytest.mark.parametrize("shape,peer_count", [("ring", 1), ("all_to_all", 3), ("random_k", 2), ("pairwise", 1)])
def test_peer_delivery_and_identity_across_rounds(tmp_path, shape, peer_count):
    workflow = runner(tmp_path, {"family": "peer_deliberation", "width": 4, "rounds": 2, "connectivity": shape})
    backend = Scripted({})
    workflow.llm = backend
    assert workflow.run()["status"] == "completed"
    for metadata, payload in backend.calls:
        if metadata["round_id"]:
            assert sum(i["kind"] == "peer_message" for i in payload["inputs"]) == peer_count
            assert sum(i["kind"] == "own_state" for i in payload["inputs"]) == 1
    identities = {}
    for metadata, _ in backend.calls:
        identities.setdefault(metadata["role_index"], set()).add(metadata["agent_instance_id"])
    assert len(identities) == 4 and all(len(ids) == 1 for ids in identities.values())


def test_revision_limit_and_failure_stop_composition(tmp_path):
    workflow = runner(tmp_path, {"stages": [
        {"id": "review", "family": "evaluate_refine", "max_revisions": 2},
        {"family": "parallel_aggregate"},
    ]})
    backend = Scripted({"evaluator": '{"decision":"revise","feedback":"fix it"}'})
    workflow.llm = backend
    summary = workflow.run()
    assert summary["status"] == "max_revisions"
    assert len(summary["motif_results"]) == 1
    assert summary["motif_results"][0]["iterations"] == 2
    assert sum(m["role_slot"] == "producer" for m, _ in backend.calls) == 3
    assert sum(m["role_slot"] == "evaluator" for m, _ in backend.calls) == 3


@pytest.mark.parametrize("response", ['{"decision":"accept"}', 'not JSON', '{"decision":"unacceptable","feedback":"x"}', RuntimeError("backend failed")])
def test_invalid_decisions_or_backend_failure_are_not_acceptance(tmp_path, response):
    workflow = runner(tmp_path, {"family": "evaluate_refine"})
    workflow.llm = Scripted({"evaluator": response})
    summary = workflow.run()
    assert summary["status"] == "failed"
    assert summary["motif_results"][0]["error"]
    saved = json.loads(Path(workflow.trace.summary_path).read_text())
    assert saved["status"] == "failed"
    assert any(e["event_type"] == "workflow_end" and e["status"] == "failed" for e in workflow.trace.events)


@pytest.mark.parametrize("spec", [
    {"family": "evaluate_refine", "max_revisions": -1},
    {"family": "peer_deliberation", "connectivity": "tree"},
    {"family": "parallel_aggregate", "roles": {"worker": {"tools": ["shell"]}}},
    {"family": "parallel_aggregate", "roles": {"coder": {}}},
    {"stages": [{"family": "evaluate_refine", "inputs": {"candidate": "later.result"}}]},
])
def test_invalid_bindings_rejected_before_execution(tmp_path, spec):
    with pytest.raises(ValueError):
        runner(tmp_path, spec)


def test_judge_output_is_selected_candidate(tmp_path):
    workflow = runner(tmp_path, {"family": "parallel_aggregate", "width": 1, "aggregation": "judge"})
    workflow.llm = Scripted({"worker": "candidate", "collector": '{"selected_index":0,"reason":"best"}'})
    summary = workflow.run()
    assert summary["motif_results"][0]["outputs"]["result"]["content"] == "candidate"


def test_dispatch_route_executes_only_selected_slot(tmp_path):
    workflow = runner(tmp_path, {"family": "dispatch_execute", "dispatch": "route", "width": 4})
    backend = Scripted({"dispatcher": '{"worker_index":2,"instruction":"solve"}'})
    workflow.llm = backend
    assert workflow.run()["status"] == "completed"
    executors = [m for m, _ in backend.calls if m["role_slot"] == "executor"]
    assert len(executors) == 1 and executors[0]["role_index"] == 2


def test_task_bound_tool_and_output_contract(tmp_path):
    workflow = runner(tmp_path, {"family": "parallel_aggregate", "width": 1,
                                "roles": {"worker": {"tools": ["search"], "output_format": "json"}}})
    workflow.llm = Scripted({"worker": "invalid JSON"})
    assert workflow.run()["status"] == "failed"
    assert any(e["event_type"].startswith("tool_") for e in workflow.trace.events)


def test_export_preserves_slot_instance_mapping(tmp_path):
    from app.trace_export import events_to_arch_spans, spans_to_otel
    workflow = runner(tmp_path, {"family": "peer_deliberation", "width": 2})
    workflow.run()
    spans = events_to_arch_spans(workflow.trace.events)
    calls = [s for s in spans if s.get("agent_instance_id") and s.get("role_slot") == "peer"]
    assert calls
    assert all(s["motif_instance_id"].startswith("motif_") for s in calls)
    otel = spans_to_otel(spans, trace_id="1" * 32)
    assert otel


def test_role_backend_override_does_not_mutate_other_roles(tmp_path, monkeypatch):
    import app.motifs.families as families
    special = Scripted({"worker": "worker output"})
    special.model = "worker-model"
    created = []
    def make_backend(mode, **kwargs):
        created.append(kwargs)
        return special
    monkeypatch.setattr(families, "build_llm_backend", make_backend)
    workflow = runner(tmp_path, {"family": "parallel_aggregate", "width": 2,
                                "roles": {"worker": {"model": "worker-model"}}})
    default = Scripted({})
    workflow.llm = default
    assert workflow.run()["status"] == "completed"
    assert len(special.calls) == 2 and len(default.calls) == 1
    assert all(c["model"] == "worker-model" for c in created)
    calls = [e for e in workflow.trace.events if e["event_type"] == "llm_request_end"]
    assert all(e["model"] == "worker-model" for e in calls if e["role_slot"] == "worker")
    assert workflow.llm is default


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_use_family_runtime(tmp_path, name):
    config = WorkloadConfig(topology_name=name, run_id="test", task_id="t", instance_id="i", query="task",
                            trace_dir=tmp_path, export_trace_views=False)
    with pytest.warns(FutureWarning):
        workflow = build_workflow(config)
    assert isinstance(workflow, FamilyWorkload)
    assert workflow.run()["status"] in {"completed", "accepted"}

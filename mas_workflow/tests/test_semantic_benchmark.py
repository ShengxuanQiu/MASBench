import copy
import json
import time
from dataclasses import asdict

import pytest
import jsonschema
from pathlib import Path

from app.semantic.coverage import FEATURES, cluster_coverage, workflow_coverage
from app.semantic.canonicalize import ByteTokenizer, canonicalize_native_trace
from app.semantic.golden import TEMPLATES, build_golden, golden_scenario, golden_system
from app.semantic.metrics import sustainable_capacity, task_metrics
from app.semantic.replay import BackendResult, MockExactBackend, ReplayExecutor, materialize_request
from app.semantic.scenario import SystemConfig
from app.semantic.store import TraceBundle
from app.semantic.validators import LoweringValidator, RunValidator, SemanticValidator
from app.benchmark import build_experiment
from app.specs import DeploymentSpec, ExperimentConfig, MotifSpec, StageSpec, StructureSpec, TaskBinding, WorkflowSpec


@pytest.mark.parametrize("template", TEMPLATES)
def test_golden_template_full_pipeline(tmp_path, template):
    path = tmp_path / template
    trace = build_golden(template, path)
    assert SemanticValidator().validate(trace).run_valid
    scenario = golden_scenario(trace)
    system = golden_system()
    run = ReplayExecutor(trace, scenario, system, MockExactBackend(), TraceBundle(path).artifacts).run()
    assert run["run_valid"], run
    assert all(not x.get("generated_output_used_downstream") for x in run["operations"])
    assert all(not x["source_observations_used_for_scheduling"] for x in run["operations"])
    metrics = task_metrics(run, slo_sec=10, accelerator_count=1)
    schema = json.loads((Path(__file__).parents[1] / "configs" / "semantic" / "benchmark_result.schema.json").read_text())
    jsonschema.validate(metrics, schema)
    assert metrics["slo_attainment"] == 1.0
    assert metrics["task_goodput"] > 0


def test_native_trace_canonicalization_pipeline(tmp_path):
    experiment = ExperimentConfig(StructureSpec({"m": MotifSpec("Spawn", width=1)},
        WorkflowSpec((StageSpec("spawn", "m", reuse_scope_id="scope:spawn"),))), TaskBinding("test"), DeploymentSpec())
    runner = build_experiment(experiment, trace_dir=tmp_path / "native")
    runner.config.export_trace_views = False
    assert runner.run()["status"] == "completed"
    trace = canonicalize_native_trace(runner.trace.trace_path, tmp_path / "semantic", tokenizer=ByteTokenizer(),
        workload_id="native-mock", source_framework_version="test")
    report = SemanticValidator().validate(trace)
    assert report.run_valid, report.to_dict()
    assert all(op.source_observations.get("duration_kind") == "source_observation" for op in trace.operations)
    assert {session.reuse_scope_id for session in trace.sessions} == {"scope:spawn"}


def test_source_observations_and_sut_do_not_change_workload_identity(tmp_path):
    trace = build_golden("spawn", tmp_path / "trace")
    original = trace.compute_workload_hash()
    trace.metadata.source_observations["source_gpu_latency"] = 999999
    trace.operations[0].source_observations["absolute_timestamp"] = "2099-01-01T00:00:00Z"
    assert trace.compute_workload_hash() == original
    scenario = golden_scenario(trace)
    scenario_hash = scenario.scenario_hash
    a = SystemConfig("vllm", "1", "A6000", 1)
    b = SystemConfig("vllm", "2", "H100", 8)
    assert a.system_hash() != b.system_hash()
    assert scenario.scenario_hash == scenario_hash


def test_benchmark_factors_change_scenario_hash(tmp_path):
    trace = build_golden("spawn", tmp_path / "trace")
    base = golden_scenario(trace)
    for field, key, value in [
        ("replay", "fidelity_mode", "token_locked"),
        ("root_arrival", "random_seed", 99),
        ("model_input_contract", "tokenizer_hash", "different"),
        ("model_input_contract", "chat_template_hash", "different"),
    ]:
        changed = copy.deepcopy(base)
        getattr(changed, field)[key] = value
        changed.resolve([x.reuse_scope_id for x in trace.sessions if x.reuse_scope_id])
        assert changed.scenario_hash != base.scenario_hash


def test_dependency_release_uses_target_completion_and_recorded_downstream_input(tmp_path):
    trace = build_golden("spawn", tmp_path / "trace")
    trace.operations[0].source_observations["finish_timestamp"] = 9999999999.0
    trace.finalize_hashes()
    downstream_before = copy.deepcopy(trace.operations[1].llm["canonical_request"])

    class SlowBackend(MockExactBackend):
        def generate(self, request, operation, mode):
            if operation.operation_id == "op:dispatch":
                time.sleep(.03)
            return super().generate(request, operation, mode)

    scenario = golden_scenario(trace)
    system = golden_system()
    run = ReplayExecutor(trace, scenario, system, SlowBackend(), TraceBundle(tmp_path / "trace").artifacts).run()
    records = {x["operation_id"]: x for x in run["operations"]}
    assert records["op:worker"]["ready_time"] >= records["op:dispatch"]["finish_time"]
    expected_hash = materialize_request(trace.operations[1], scenario)[1]
    assert records["op:worker"]["request_hash"] == expected_hash
    assert trace.operations[1].llm["canonical_request"] == downstream_before


def test_cache_scopes_are_deterministically_isolated(tmp_path):
    trace = build_golden("debate", tmp_path / "trace")
    one = golden_scenario(trace)
    two = golden_scenario(trace)
    assert one.resolved_cache_scope_salts == two.resolved_cache_scope_salts
    assert len(set(one.resolved_cache_scope_salts.values())) == len(one.resolved_cache_scope_salts)
    broken = copy.deepcopy(one)
    values = list(broken.resolved_cache_scope_salts)
    broken.resolved_cache_scope_salts[values[1]] = broken.resolved_cache_scope_salts[values[0]]
    broken.scenario_hash = broken.compute_hash()
    system = golden_system()
    report = RunValidator().validate(trace, broken, system)
    assert "CACHE_SCOPE_VIOLATION" in report.invalid_reasons


def test_token_lock_never_silently_degrades(tmp_path):
    trace = build_golden("spawn", tmp_path / "trace")
    scenario = golden_scenario(trace, mode="token_locked")
    system = golden_system(); system.capabilities = {"supports_length_lock": True, "supports_token_lock": False}
    report = RunValidator().validate(trace, scenario, system)
    assert "REPLAY_MODE_UNSUPPORTED" in report.invalid_reasons


def test_invalid_trace_reason_codes(tmp_path):
    trace = build_golden("spawn", tmp_path / "trace")
    trace.operations[1].data_dependencies.append("missing")
    report = SemanticValidator().validate(trace)
    assert not report.run_valid
    assert "MISSING_DEPENDENCY" in report.invalid_reasons


def test_task_metrics_keep_failed_tasks_in_denominator():
    run = {"run_valid": True, "measurement_duration_sec": 10, "tasks": [
        {"successful": True, "task_latency_sec": 1, "raw_wall_clock_latency_sec": 1},
        {"successful": False, "task_latency_sec": None, "raw_wall_clock_latency_sec": 2}]}
    result = task_metrics(run, slo_sec=1.5, accelerator_count=2)
    assert result["slo_attainment"] == .5
    assert result["task_goodput"] == .1
    assert result["resource_efficiency_tasks_per_sec_per_accelerator"] == .05
    assert sustainable_capacity([{"load": 1, "run_valid": True, "slo_attainment": .95}, {"load": 2, "run_valid": True, "slo_attainment": .8}], attainment_target=.9)["maximum_tested_load"] == 1


def test_coverage_metrics(tmp_path):
    corpus = {"schema_version": "masbench.coverage_corpus/1.0.0", "corpus_split_version": "split-1", "abstraction_version": "1", "systems": [
        {"unit_id": "a", "system": "A", "version": "1", "canonical_configuration": "default", "split": "held_out",
         "relevance": "causal_multi_agent", "inspectability_sources": ["code"], "frozen_reference": "abc", "zero_extension_covered": True,
         "external_impact_snapshot": {"captured_at": "now", "github_stars": 1, "citation_count": 1, "publication_or_release_date": "today", "organization_or_vendor": "A", "maintenance_status": "active", "downstream_adoption_evidence": []},
         "workflow_assessment": {"stages": [{"stage_id": "s", "template_type": "spawn", "standard_semantics": True, "requires_extension": False, "wrapped_as_atomic": False}], "standard_composition_expressible": True, "arbitrary_control_callback_required": False}},
        {"unit_id": "b", "system": "B", "version": "2", "canonical_configuration": "debate", "split": "held_out",
         "relevance": "causal_multi_agent", "inspectability_sources": ["paper"], "frozen_reference": "doi", "zero_extension_covered": False,
         "external_impact_snapshot": {"captured_at": "now", "github_stars": 1, "citation_count": 1, "publication_or_release_date": "today", "organization_or_vendor": "B", "maintenance_status": "active", "downstream_adoption_evidence": []},
         "workflow_assessment": {"stages": [{"stage_id": "s", "template_type": "extension", "standard_semantics": False, "requires_extension": True, "wrapped_as_atomic": False}], "standard_composition_expressible": True, "arbitrary_control_callback_required": False},
         "failure_reason": "shared state", "missing_semantics": ["transaction"], "residual_category": "state"}]}
    value = workflow_coverage(corpus)
    assert value["metric"] == "C_workflow" and value["value"] == .5
    real = [{"workload_id": f"r{i}", **{key: float(i + j) for j, key in enumerate(FEATURES)}} for i in range(4)]
    bench = [{"workload_id": "m", **{key: float(j) for j, key in enumerate(FEATURES)}}]
    cluster = cluster_coverage(real, bench, clusters=2, seed=3, pca_dimensions=2)
    assert cluster["metric"] == "C_cluster" and cluster["denominator"] == 2


def test_lowering_provenance_validator(tmp_path):
    trace = build_golden("spawn", tmp_path / "trace")
    manifest = {"converter_version": "chakra-1", "lowered_operation_ids": ["op:dispatch"],
        "semantic_to_chakra_nodes": {"op:dispatch": ["1", "2"]},
        "chakra_node_to_semantic": {"1": "op:dispatch", "2": "op:dispatch"}}
    assert LoweringValidator().validate(trace, manifest).run_valid
    fork = build_golden("fork_join", tmp_path / "fork")
    broken = {"converter_version": "chakra-1", "lowered_operation_ids": [x.operation_id for x in fork.operations],
        "semantic_to_chakra_nodes": {x.operation_id: [str(i)] for i, x in enumerate(fork.operations)},
        "chakra_node_to_semantic": {str(i): x.operation_id for i, x in enumerate(fork.operations)}, "chakra_edges": []}
    assert "MISSING_DEPENDENCY" in LoweringValidator().validate(fork, broken).invalid_reasons

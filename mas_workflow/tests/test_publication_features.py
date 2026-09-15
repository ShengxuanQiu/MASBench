"""Acceptance tests for adaptive structure, provenance and publication protocols."""
import json
from collections import Counter
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from app.benchmark import build_experiment
from app.execution_graph import ExecutionGraph
from app.llm_backends import MockLLM
from app.specs import (AtomicStage, DeploymentSpec, ExperimentConfig, MotifSpec,
                       StageSpec, StructureSpec, TaskBinding, WorkflowSpec)
from app.replay import replay_trace
from app.replay_compare import compare_replay
from app.publication import preset, generate
from app.capacity import capacity_summary, aggregate, run_capacity
from app.workload_analysis import analyze_events


def record(path, stages, *, task=None, motifs=None):
    exp = ExperimentConfig(StructureSpec(motifs or {}, WorkflowSpec(tuple(stages))),
                           task or TaskBinding('Compare alternatives'), DeploymentSpec())
    runner = build_experiment(exp, trace_dir=path)
    runner.config.export_trace_views = False
    assert runner.run()['status'] in {'accepted', 'completed'}
    ExecutionGraph.from_events(runner.trace.events, replay=True)
    return runner


def test_adaptive_subset_skip_join_and_adversarial_replay(tmp_path):
    select = {'from_stage': 'route', 'field': 'branch', 'equals': 'left'}
    stages = [StageSpec('join', atomic=AtomicStage('transform'),
                        inputs={'context': ['left.result', 'right.result']}, allow_skipped=True),
              StageSpec('left', 'p', condition=select,
                        participants={'from_stage': 'route', 'field': 'workers'}),
              StageSpec('right', 'p', condition={**select, 'equals': 'right'}),
              StageSpec('route', atomic=AtomicStage('router'))]
    task = TaskBinding('Choose', stage_bindings={'route': {'parameters': {
        'constant': {'branch': 'left', 'workers': [0, 2]}}}})
    r = record(tmp_path/'source', stages, task=task, motifs={'p': MotifSpec('ParallelAggregate', width=4)})
    skips = [e for e in r.trace.events if e.get('canonical_type') == 'stage_skip']
    assert [e['stage_id'] for e in skips] == ['right']
    assert sum(op.kind == 'llm' for op in r.trace.execution_graph.operations.values()) == 3
    assert any(e['decision'].get('kind') == 'participants' for e in r.trace.execution_graph.decisions)

    class Different(MockLLM):
        def invoke(self, *args):
            return replace(super().invoke(*args), content='{"branch":"right","workers":[1,3]}')
    _, trace = replay_trace(r.trace.trace_path, DeploymentSpec(), trace_dir=tmp_path/'replay', backend=Different(), export_views=False)
    assert compare_replay(r.trace.events, trace.events)['fixed_workload_invariants_pass']
    assert [e['stage_id'] for e in trace.events if e.get('canonical_type') == 'stage_skip'] == ['right']
    assert all('"workers":[1,3]' not in str(op.payload) for op in trace.execution_graph.operations.values())


@pytest.mark.parametrize('kind', ['llm', 'transform', 'router', 'tool'])
def test_atomic_stage_is_operation_not_core_motif(tmp_path, kind):
    task = TaskBinding('Atomic', stage_bindings={'one': {'parameters': {'constant': {'route': 1}}}})
    r = record(tmp_path, [StageSpec('one', atomic=AtomicStage(kind))], task=task)
    assert len(r.trace.execution_graph.operations) == 1
    op = next(iter(r.trace.execution_graph.operations.values()))
    assert op.kind == kind
    assert not op.identities.get('motif_instance_id')
    assert not any(e['event_type'] == 'motif_start' for e in r.trace.events)


@pytest.mark.parametrize('mode', ['full', 'selected', 'summarized', 'referenced', 'retrieved'])
def test_delivery_provenance_roundtrip_and_replay(tmp_path, mode):
    stages = [StageSpec('a', atomic=AtomicStage('transform')),
              StageSpec('b', atomic=AtomicStage('transform')),
              StageSpec('join', atomic=AtomicStage(), inputs={'context': ['a.result', 'b.result']}, delivery={'context': mode})]
    task = TaskBinding('Facts', stage_bindings={
        'a': {'parameters': {'transform': 'constant', 'value': 'alpha'}},
        'b': {'parameters': {'transform': 'constant', 'value': 'beta'}},
        'join': {'delivery_instruction': 'Summarize the supplied facts.', 'parameters': {'delivery': {'context': {'indices': [1]}}}}})
    r = record(tmp_path/'source', stages, task=task)
    graph = ExecutionGraph.from_events(r.trace.events, replay=True)
    deliveries = [e for e in graph.deliveries if e['delivery_mode'] == mode]
    assert deliveries and all(e['source_artifact_ids'] for e in deliveries)
    assert graph.to_dict() == r.trace.execution_graph.to_dict()
    if mode == 'summarized':
        assert sum(op.kind == 'llm' for op in graph.operations.values()) == 2
        assert len(deliveries[-1]['source_artifact_ids']) == 2
    if mode == 'selected':
        assert len(deliveries) == 1
        assert graph.artifacts[deliveries[0]['artifact_id']]['content'] == 'beta'
    _, trace = replay_trace(r.trace.trace_path, DeploymentSpec(), trace_dir=tmp_path/'replay', export_views=False)
    assert compare_replay(r.trace.events, trace.events)['fixed_workload_invariants_pass']


def test_peer_k_sparsity_change_data_edges(tmp_path):
    counts = []
    for index, options in enumerate([{'k': 0}, {'k': 1}, {'k': 3}, {'sparsity': 0}]):
        r = record(tmp_path/str(index), [StageSpec('one', 'p')],
            motifs={'p': MotifSpec('PeerDeliberation', width=4, rounds=1, connectivity='random_k', **options)})
        counts.append(sum(k == 'data' for a, b, k in r.trace.execution_graph.edges))
    assert counts[0] < counts[1] < counts[2] == counts[3]


@pytest.mark.parametrize('name', ['parallel_refine', 'dispatch_parallel_refine', 'fanout_fanin'])
def test_composed_presets(tmp_path, name):
    exp = preset(name, TaskBinding('Plan'), DeploymentSpec(), width=2, max_revisions=1)
    r = record(tmp_path, exp.structure.workflow.stages, task=exp.task, motifs=exp.structure.motifs)
    assert analyze_events(r.trace.events)['structure']['nodes'] > 3


def test_matched_work_preserves_request_payload_multiset(tmp_path):
    graphs = []
    for name in ['matched_chain', 'matched_parallel']:
        exp = preset(name, TaskBinding('Identical content'), DeploymentSpec(), requests=5)
        r = record(tmp_path/name, exp.structure.workflow.stages, task=exp.task)
        graphs.append(r.trace.execution_graph)
    payloads = [Counter(json.dumps(op.payload, sort_keys=True) for op in g.operations.values()) for g in graphs]
    assert payloads[0] == payloads[1]
    assert len(graphs[0].edges) == 4 and not graphs[1].edges


def row(rate, passed=True, **overrides):
    return dict(deployment='gpu', rate=rate, client_overflow=0, offered=100, failed_or_rejected=0,
                slo_success_fraction=1 if passed else .8, goodput_qps=rate, internal_qps=rate*3,
                completed_workflow_qps=rate, completed_e2e_sec={'mean': .4, 'p50': .3, 'p95': .8},
                queue_growth_client_per_sec=.2, **overrides)


def test_capacity_boundary_censoring_ci_and_invalid_replay():
    result = capacity_summary([row(1), row(1), row(2, False), row(2, False)])['gpu']
    assert result['bracket'] == [1, 2] and result['lambda_knee'] == 2
    assert result['points'][0]['rho'] == .5
    assert aggregate([1, 3])['mean_ci95'] is not None
    assert aggregate([1])['mean_ci95'] is None
    assert capacity_summary([row(1)])['gpu']['status'] == 'right_censored'
    assert capacity_summary([row(1, False)])['gpu']['status'] == 'left_censored'
    assert capacity_summary([row(1, False), row(2)])['gpu']['status'] == 'nonmonotonic'
    assert capacity_summary([row(1, replay_equivalence_failures=1)])['gpu']['lambda_knee'] is None


def test_capacity_dense_search_and_resume_fingerprint(tmp_path, monkeypatch):
    import app.capacity as module
    (tmp_path/'study.json').write_text(json.dumps({'rates': [1, 3], 'deployments': {'gpu': 'dep.json'}}))
    plan = tmp_path/'capacity.json'
    plan.write_text(json.dumps({'study': 'study.json', 'capacity': {'dense_points': 3}}))
    calls = []
    def study(path, output, **kwargs):
        cfg = json.loads(path.read_text()); calls.append(cfg['rates'])
        return [row(rate, rate < 2) for rate in cfg['rates']]
    monkeypatch.setattr(module, 'run_study', study)
    assert run_capacity(plan, tmp_path/'out')['gpu']['bracket'] == [1.5, 2.0]
    assert calls == [[1, 3], [1.5, 2.0, 2.5]]
    run_capacity(plan, tmp_path/'out', resume=True)
    plan.write_text(json.dumps({'study': 'study.json', 'capacity': {'dense_points': 2}}))
    with pytest.raises(ValueError, match='fingerprint'):
        run_capacity(plan, tmp_path/'out', resume=True)


def test_length_identity_and_cache_fail_closed(monkeypatch):
    import app.replay_protocol as protocol
    assert not protocol.length_agreement([{'recorded': 10, 'replayed': 12}], .1)['pass']
    assert protocol.length_agreement([{'recorded': 10, 'replayed': 11}], .1)['pass']
    assert not protocol.length_agreement([{'recorded': 10, 'replayed': None}])['pass']
    identity = {'model_revision': 'revision', 'tokenizer_sha256': 'a'*64, 'chat_template_sha256': 'b'*64}
    assert protocol.identity_agreement(identity, identity)['status'] == 'matched'
    assert protocol.identity_agreement({}, identity)['status'] == 'unavailable'
    assert protocol.identity_agreement({**identity, 'tokenizer_sha256': 'TODO'}, identity)['status'] != 'matched'
    deployment = DeploymentSpec(backend='openai_compatible')
    monkeypatch.setattr(protocol, 'fetch_prometheus_metrics', lambda _: {'vllm:cache_config_info{enable_prefix_caching="False"}': 1})
    assert protocol.verify_cache_protocol(deployment, 'cache_disabled')['verified']
    with pytest.raises(ValueError): protocol.verify_cache_protocol(deployment, 'warm_cache_enabled')
    monkeypatch.setattr(protocol, 'fetch_prometheus_metrics', lambda _: {})
    with pytest.raises(ValueError): protocol.verify_cache_protocol(deployment, 'cache_disabled')


def test_device_sampling_survives_missing_endpoint(tmp_path, monkeypatch):
    import app.backend_metrics as module
    def unavailable(*args, **kwargs): raise OSError('metrics down')
    monkeypatch.setattr(module, 'fetch_prometheus_metrics', unavailable)
    adapter = SimpleNamespace(serving_sample=lambda _: {}, cache_memory_sample=lambda _: {},
                              collect_device_metrics=lambda: {'power_w': 75})
    sampler = module.BackendMetricsSampler('none', tmp_path/'metrics.json', adapter=adapter)
    sampler.sample_once()
    assert sampler.samples[0]['status'] == 'error'
    assert sampler.samples[0]['device_metrics']['power_w'] == 75


def test_publication_generation_and_config_substitution(tmp_path):
    (tmp_path/'task.json').write_text(json.dumps({'task_input': 'Task'}))
    (tmp_path/'dep.json').write_text(json.dumps({'backend': 'mock'}))
    manifest = tmp_path/'matrix.json'
    manifest.write_text(json.dumps({'tasks': {'task': 'task.json'}, 'deployments': {'cpu': 'dep.json'},
        'workloads': [{'preset': 'peer', 'parameters': [{'width': 4, 'connectivity': 'random_k', 'k': 1}, {'width': 4, 'connectivity': 'random_k', 'k': 3}]}], 'study': {'rates': [1, 2]}}))
    cells = generate(manifest, tmp_path/'out')
    assert len(cells) == 2
    assert generate(manifest, tmp_path/'out', resume=True) == cells
    (tmp_path/'task.json').write_text(json.dumps({'task_input': 'Changed'}))
    with pytest.raises(ValueError, match='fingerprint'): generate(manifest, tmp_path/'out', resume=True)


def test_ascend_telemetry_fixture_missing_and_profiler(tmp_path, monkeypatch):
    import app.backend_adapters as adapters
    import time
    def command(args, **kwargs):
        assert args[-4:] == ['-i', '2', '-c', '0']
        return SimpleNamespace(stdout='Aicore Usage Rate(%) : 43\nMemory Usage Rate(%) : 25\nMemory Capacity(MB) : 10810\n' if 'usages' in args else 'Power(W) : 51.5\n')
    monkeypatch.setattr(adapters.subprocess, 'run', command)
    snapshot = tmp_path/'counters.json'
    snapshot.write_text(json.dumps({'device_id': '2', 'timestamp_unix': time.time(), 'metrics': {'cycles': 123}}))
    adapter = adapters.NPUTraceAdapter({'npu_id': 2, 'profiler_counters_path': str(snapshot)})
    sample = adapter.collect_device_metrics()
    assert sample['device_utilization_percent'] == 43 and sample['power_watts'] == 51.5
    assert sample['profiler_counters'] == {'cycles': 123}
    normalized = adapters.canonical_telemetry({'device_metrics': sample})
    assert normalized['device_metrics.memory_used_bytes']['source'] == 'estimated'
    assert normalized['device_metrics.power_watts']['source'] == 'observed'
    def missing(*args, **kwargs): raise FileNotFoundError('no npu-smi')
    monkeypatch.setattr(adapters.subprocess, 'run', missing)
    missing_sample = adapter.collect_device_metrics()
    assert missing_sample['power_watts'] is None and missing_sample['status'] == 'unavailable'


def test_study_wires_adapter_and_archives_source(tmp_path, monkeypatch):
    import app.study as study
    import app.backend_adapters as adapters
    import app.backend_metrics as metrics
    exp = preset('matched_chain', TaskBinding('Test'), DeploymentSpec(), requests=2)
    (tmp_path/'exp.json').write_text(json.dumps(asdict(exp)))
    dep = {'backend': 'openai_compatible', 'telemetry': {'adapter': 'ascend', 'npu_id': 3}}
    (tmp_path/'dep.json').write_text(json.dumps(dep))
    config = tmp_path/'study.json'
    config.write_text(json.dumps({'experiments': ['exp.json'], 'deployments': {'npu': 'dep.json'},
        'rates': [100], 'count': 2, 'collect_backend_metrics': True}))
    real_build = study.build_experiment
    def build(*args, **kwargs):
        r = real_build(*args, **kwargs); r.llm = MockLLM(); return r
    monkeypatch.setattr(study, 'build_experiment', build)
    calls = []
    adapter = adapters.NPUTraceAdapter({'npu_id': 3})
    monkeypatch.setattr(adapter, 'collect_device_metrics', lambda: {'power_watts': None})
    def factory(kind, config): calls.append((kind, config)); return adapter
    monkeypatch.setattr(adapters, 'build_backend_trace_adapter', factory)
    monkeypatch.setattr(metrics, 'fetch_prometheus_metrics', lambda *args, **kwargs: {})
    rows = study.run_study(config, tmp_path/'out')
    assert calls == [('ascend', dep['telemetry'])]
    assert not rows[0]['analysis_errors']
    assert (tmp_path/'out/source_snapshot/study.py').exists()
    path = next((tmp_path/'out').rglob('backend_observations.jsonl'))
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events[0]['attributes']['serving_observation']['device_metrics.power_watts']['source'] == 'unavailable'
    assert study.run_study(config, tmp_path/'out', resume=True) == rows
    assert len(calls) == 1  # completed case is not submitted again


def test_pressure_units_and_queue_growth(tmp_path):
    from app.pressure import queue_slope
    r = record(tmp_path, [StageSpec('one', 'p')], motifs={'p': MotifSpec('ParallelAggregate', width=3)})
    analysis = analyze_events(r.trace.events)
    pressure = analysis['pressure']
    assert pressure['work_amplification']['llm_requests_per_workflow'] == 4
    assert pressure['logical_state_residency']['source'] == 'estimated'
    assert pressure['logical_state_residency']['peak_artifact_bytes'] > 0
    assert analysis['runtime']['kv_live']['source'] == 'unavailable'
    assert queue_slope([{'time_sec': i/20, 'ready_waiting': i} for i in range(21)], 1) == pytest.approx(20)


def test_local_identity_hashes_change_with_template(tmp_path):
    from app.replay_protocol import local_identity
    (tmp_path/'tokenizer.json').write_text('{"vocab":{}}')
    (tmp_path/'tokenizer_config.json').write_text('{"chat_template":"first"}')
    first = local_identity(tmp_path, 'revision')
    (tmp_path/'tokenizer_config.json').write_text('{"chat_template":"second"}')
    second = local_identity(tmp_path, 'revision')
    assert first['identity']['chat_template_sha256'] != second['identity']['chat_template_sha256']
    assert first['identity']['tokenizer_sha256'] != second['identity']['tokenizer_sha256']


def test_missing_backend_counters_are_not_zero():
    from app.backend_metrics import summarize_backend_metrics
    summary = summarize_backend_metrics([{'status': 'success', 'relative_time_sec': i, 'metrics': {}} for i in [0, 1]])
    assert summary['prompt_tokens_total_delta'] is None
    assert summary['backend_generation_tokens_per_sec_window'] is None


def test_replay_study_archives_corpus_and_checks_identity(tmp_path):
    from app.study import run_study
    identity = {'model_revision': 'test-revision', 'tokenizer_sha256': 'a'*64, 'chat_template_sha256': 'b'*64}
    deployment = DeploymentSpec(identity=identity)
    exp = preset('parallel', TaskBinding('Test'), deployment, width=2)
    runner = build_experiment(exp, trace_dir=tmp_path/'record')
    runner.config.export_trace_views = False
    runner.run()
    (tmp_path/'dep.json').write_text(json.dumps(asdict(deployment)))
    path = tmp_path/'replay.json'
    path.write_text(json.dumps({'mode': 'replay', 'traces': [str(runner.trace.trace_path)],
        'deployments': {'mock': 'dep.json'}, 'rates': [100], 'count': 2, 'require_identity_match': True}))
    rows = run_study(path, tmp_path/'out')
    assert not rows[0]['analysis_errors']
    assert all(c['identity']['status'] == 'matched' for c in rows[0]['replay_checks'])
    archived = json.loads((tmp_path/'out/source_corpus.json').read_text())['runs']
    assert len(archived) == 1
    assert (tmp_path/'out'/archived[0]['trace']).read_bytes() == runner.trace.trace_path.read_bytes()

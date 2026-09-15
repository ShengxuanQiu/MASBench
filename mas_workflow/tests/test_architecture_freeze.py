import json
from dataclasses import asdict
from pathlib import Path

import jsonschema
import pytest

from app.benchmark import build_experiment
from app.coverage_audit import calculate,validate
from app.execution_graph import ExecutionGraph
from app.publication import generate
from app.quality import evaluate
from app.replay import replay_trace
from app.replay_compare import compare_replay
from app.specs import (DeliverySpec,DeploymentSpec,ExperimentConfig,MotifSpec,StageSpec,
                       StructureSpec,TaskBinding,WorkflowSpec)


def run(tmp_path,motifs,stages,task=None):
    exp=ExperimentConfig(StructureSpec(motifs,WorkflowSpec(tuple(stages))),task or TaskBinding('Test'),DeploymentSpec())
    runner=build_experiment(exp,trace_dir=tmp_path);runner.config.export_trace_views=False
    assert runner.run()['status'] in {'completed','accepted','max_revisions'}
    return runner


def test_typed_per_edge_delivery_and_structured_selector(tmp_path):
    stages=[StageSpec('a',atomic={'kind':'transform'}),StageSpec('b',atomic={'kind':'transform'}),
        StageSpec('join',atomic={'kind':'llm'},inputs={'context':[
            {'source':'a.result','delivery':{'mode':'referenced'}},
            {'source':'b.result','delivery':{'mode':'selected','selector':{'json_fields':['answer']}}}]})]
    task=TaskBinding('Base',stage_bindings={'a':{'parameters':{'transform':'constant','value':'alpha'}},
        'b':{'parameters':{'transform':'constant','value':json.dumps({'answer':'beta','extra':1})}}})
    runner=run(tmp_path,{},stages,task);graph=runner.trace.execution_graph
    assert isinstance(stages[2].inputs['context'][0]['delivery'],DeliverySpec)
    modes={d['delivery_mode'] for d in graph.deliveries}
    assert {'referenced','selected'} <= modes
    assert {'transform'} <= {op.kind for op in graph.operations.values()}
    assert all(d['source_artifact_ids'] and d['delivered_artifact_id'] and d['materialized_bytes']>=0 for d in graph.deliveries)


@pytest.mark.parametrize('family,relation,kwargs,reason',[
    ('DispatchExecute','coordinator_to_worker',{},'selected_work_returned'),
    ('ParallelAggregate','worker_to_reducer',{},'required_branches_completed'),
    ('EvaluateRefine','producer_to_reviewer',{'width':1,'max_revisions':0},'accepted'),
    ('PeerExchange','peer_to_peer',{'rounds':1,'connectivity':'ring'},'fixed_rounds_completed')])
def test_intra_motif_delivery_semantics_and_completion(tmp_path,family,relation,kwargs,reason):
    motif=MotifSpec(family,delivery={relation:DeliverySpec('referenced')},**{'width':2,**kwargs})
    runner=run(tmp_path,{'m':motif},[StageSpec('one','m')]);graph=runner.trace.execution_graph
    assert any(d['delivery_mode']=='referenced' for d in graph.deliveries)
    stage=next(iter(graph.stages.values()))
    assert stage['semantics']['delivery'][relation]['mode']=='referenced'
    assert stage['completion_reason']==reason
    assert stage['semantics']['canonical_family']==family


def test_peer_exchange_public_name_and_legacy_alias():
    assert MotifSpec('PeerExchange',width=2).family=='peer_exchange'
    assert MotifSpec('PeerDeliberation',width=2).family=='peer_exchange'
    assert MotifSpec('peer_deliberation',width=2).family=='peer_exchange'


def test_realized_hierarchy_export_and_replay(tmp_path):
    stages=[StageSpec('a',atomic={'kind':'router'}),
            StageSpec('b','m',condition={'from_stage':'a','field':'go','equals':True})]
    task=TaskBinding('go',stage_bindings={'a':{'parameters':{'constant':{'go':True}}}})
    source=run(tmp_path/'source',{'m':MotifSpec('ParallelAggregate',width=2)},stages,task)
    exported=source.trace.execution_graph.to_dict()
    assert set(exported)>= {'stages','decisions','deliveries','hierarchy'}
    assert set(exported['hierarchy']['operation_to_stage'].values())==set(exported['stages'])
    assert {s['logical_stage_id'] for s in exported['stages'].values()}=={'a','b'}
    _,trace=replay_trace(source.trace.trace_path,DeploymentSpec(),trace_dir=tmp_path/'replay',export_views=False)
    report=compare_replay(source.trace.events,trace.events)
    assert report['fixed_workload_invariants_pass'],report['errors']


def test_atomic_operation_kind_preserved(tmp_path):
    for kind in ('transform','router','tool'):
        task=TaskBinding('x',stage_bindings={'one':{'parameters':{'constant':{'x':1}}}})
        runner=run(tmp_path/kind,{},[StageSpec('one',atomic={'kind':kind})],task)
        assert next(iter(runner.trace.execution_graph.operations.values())).kind==kind


def test_coverage_calculator_exact_metrics_and_example_warning():
    path=Path(__file__).parents[2]/'evaluation'/'coverage_audit.example.json'
    result=calculate(validate(json.loads(path.read_text())))
    assert result['source_kind']=='example' and result['warning']
    assert result['C_sub']=={**result['C_sub'],'numerator':2,'denominator':3,'value':2/3}
    assert result['C_wf_0']['value']==.5
    assert result['F_dep_spec']['f1']==1 and result['F_info_spec']['macro_f1']==1


def test_quality_evaluators_are_out_of_band():
    assert evaluate('answer',{'method':'exact','expected':'answer'})['pass']
    assert evaluate('10.2',{'method':'numeric','expected':10,'tolerance':.3})['pass']
    row=evaluate('{"score":0.9}',{'method':'json_field','field':'score','expected':1,'tolerance':.11})
    assert row['pass'] and not row['included_in_serving_workload'] and row['tokens_in_benchmark']==0
    from app.quality import register_evaluator
    register_evaluator('fixture',lambda value,spec:{'score':.75,'pass':True,'observed':'fixture'})
    assert evaluate('x',{'method':'external:fixture'})['score']==.75


def test_study_saves_out_of_band_quality(tmp_path):
    from app.study import run_study
    stages=[StageSpec('one',atomic={'kind':'transform'})]
    task=TaskBinding('x',stage_bindings={'one':{'parameters':{'transform':'constant','value':'gold'}}})
    exp=ExperimentConfig(StructureSpec({},WorkflowSpec(tuple(stages))),task,DeploymentSpec())
    (tmp_path/'experiment.json').write_text(json.dumps(asdict(exp)))
    (tmp_path/'deployment.json').write_text(json.dumps(asdict(DeploymentSpec())))
    config={'mode':'run','experiments':['experiment.json'],'deployments':{'mock':'deployment.json'},
            'rates':[100],'count':2,'quality_evaluator':{'method':'exact','expected':'gold'}}
    (tmp_path/'study.json').write_text(json.dumps(config))
    rows=run_study(tmp_path/'study.json',tmp_path/'out')
    assert rows[0]['quality_pass_fraction']==1 and rows[0]['quality_score_mean']==1
    quality=json.loads(next((tmp_path/'out').rglob('quality.json')).read_text())
    assert len(quality['evaluations'])==2 and all(not q['included_in_serving_workload'] for q in quality['evaluations'])


def test_delivery_publication_sweep_and_generic_overrides(tmp_path):
    root=Path(__file__).parents[1]/'configs'/'publication'
    cells=generate(root/'information-delivery-sweep.json',tmp_path/'matrix')
    assert len(cells)==8
    experiments=[json.loads((tmp_path/'matrix'/c['id']/'experiment.json').read_text()) for c in cells]
    peer={e['structure']['motifs']['peer']['delivery']['peer_to_peer']['mode'] for e in experiments if 'peer' in e['structure']['motifs']}
    cross={e['structure']['workflow']['stages'][1]['inputs']['candidate'][0]['delivery']['mode'] for e in experiments if 'parallel' in e['structure']['motifs']}
    assert peer==cross=={'full','summarized','referenced','retrieved'}


def test_canonical_schema_accepts_new_trace(tmp_path):
    runner=run(tmp_path,{'m':MotifSpec('ParallelAggregate',width=2)},[StageSpec('one','m')])
    schema=json.loads((Path(__file__).parents[1]/'configs'/'canonical_trace.schema.json').read_text())
    for event in runner.trace.events:
        if event.get('canonical_schema'):jsonschema.validate(event,schema)

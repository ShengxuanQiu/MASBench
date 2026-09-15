"""Adaptive/atomic helpers used by the existing canonical FamilyWorkload runner."""
import json
import time
from dataclasses import replace
from uuid import uuid4
from .contracts import AgentInstance, Artifact, MotifResult, RoleBinding, RoleSlot
from ..tracing import estimate_tokens


def decision_value(runner, selector, completed):
    result=completed[selector['from_stage']]
    sid=runner._stage_context[result.motif_instance_id]['id']
    decisions=[e['decision'] for e in runner.trace.execution_graph.decisions
               if e.get('stage_instance_id')==sid and e['decision'].get('kind') not in {'stage_activation','participants'}]
    value=decisions[-1] if decisions else json.loads(result.outputs['result'].content)
    for key in selector.get('field','').split('.'):
        if key: value=value[key]
    return value


def record_choice(runner, selector, completed, decision):
    result=completed[selector['from_stage']]
    runner.trace.emit(event_type='control_decision',node_id=result.outputs['result'].producer_node,
        stage_instance_id=runner._stage_context[result.motif_instance_id]['id'],decision=decision)


def skip_stage(runner, name, reason, stage=None):
    mid='skipped_'+uuid4().hex;sid='stage_'+uuid4().hex
    runner._stage_context[mid]={'id':sid,'nodes':[]}
    from .families import stage_semantics
    semantics=stage_semantics(stage,[]) if stage else {}
    runner.trace.emit(event_type='stage_skip',stage_id=name,stage_instance_id=sid,reason=reason,
                      completion_reason=reason,activation='skipped',
                      stage_dependencies=list((stage or {}).get('_realized_dependencies',(stage or {}).get('depends_on',[]))),stage_semantics=semantics)
    return MotifResult('skipped',mid,'skipped',{})


def operation(runner,mid,inputs,fn,*,decision=False,delivery_mode=None,operation_kind='transform',
              selector=None,transform=None):
    ctx=runner._stage_context[mid]
    node='atomic_'+uuid4().hex
    identity={'stage_instance_id':ctx['id'],'motif_instance_id':'' if ctx.get('atomic') else mid,
              'atomic_instance_id':mid if ctx.get('atomic') else '',
              'role':'Coordinator' if decision else 'Worker'}
    parents=list(dict.fromkeys(ctx['parents']+[a.producer_node for a in inputs]))
    runner.trace.emit(event_type='operation_start',node_id=node,operation_kind=operation_kind,parents=parents,**identity)
    started=time.perf_counter()
    for artifact in inputs:
        runner.trace.emit(event_type='dependency',src=artifact.producer_node,dst=node,dependency_kind='data',**identity)
        runner.trace.emit(event_type='artifact_consume',node_id=node,artifact_id=artifact.artifact_id,
                          delivery_mode=delivery_mode or 'full',**identity)
    try:
        value=fn()
        content=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
    except Exception as exc:
        runner.trace.emit(event_type='operation_fail',node_id=node,error=str(exc),**identity)
        raise
    runner.trace.emit(event_type='operation_finish',node_id=node,tool_snapshot=value,
                      duration_sec=time.perf_counter()-started,**identity)
    result=Artifact(content,node,delivery_mode=delivery_mode or 'full',
                    source_artifact_ids=tuple(a.artifact_id for a in inputs),
                    delivery_selector=selector or {},delivery_transform=transform or {})
    runner._produce(result,identity)
    for artifact in inputs:
        runner.trace.emit(event_type='artifact_delivery',node_id=node,artifact_id=artifact.artifact_id,
                          producer_operation_id=artifact.producer_node,consumer_operation_id=node,
                          source_artifact_ids=list(artifact.source_artifact_ids or (artifact.artifact_id,)),
                          delivered_artifact_id=result.artifact_id,delivery_mode=delivery_mode or artifact.delivery_mode,
                          delivery_selector=selector or artifact.delivery_selector,
                          delivery_transform=transform or artifact.delivery_transform,
                          materialized_bytes=len(content.encode('utf-8')),
                          materialized_tokens_est=estimate_tokens(content),**identity)
    if decision:
        if not isinstance(value,dict): raise ValueError('Router must return an object')
        runner.trace.emit(event_type='control_decision',node_id=node,decision=value,**identity)
    ctx['nodes'].append(node)
    return result


def materialize(runner,stage,items,mid,spec):
    """Apply one edge delivery policy and return delivered artifacts."""
    from ..specs import delivery_spec
    spec=delivery_spec(spec)
    selector=dict(spec.selector);transform=dict(spec.transform)
    if spec.mode=='selected':
        indices=selector.get('artifact_indices',selector.get('indices'))
        if indices is not None:
            if any(i>=len(items) for i in indices): raise ValueError('Invalid delivery selection')
            items=[items[i] for i in indices]
        fields=selector.get('json_fields')
        if fields:
            def select(artifact):
                value=json.loads(artifact.content)
                chosen={field:value[field] for field in fields}
                return operation(runner,mid,[artifact],lambda:chosen,delivery_mode=spec.mode,
                                 operation_kind='transform',selector=selector,transform={'kind':'json_field_selection'})
            return [select(a) for a in items]
    if spec.mode=='summarized':
        instruction=transform.get('instruction') or stage['delivery_instruction']
        agent=AgentInstance(RoleSlot('Reducer'),RoleBinding(instruction),mid)
        result=runner._call(agent,stage['task'],items)
        return [replace(result,delivery_mode=spec.mode,source_artifact_ids=tuple(a.artifact_id for a in items),
                        delivery_selector=selector,delivery_transform={**transform,'kind':'llm_summary'})]
    if spec.mode in {'referenced','retrieved'}:
        converted=[]
        for artifact in items:
            result=operation(runner,mid,[artifact],
                lambda a=artifact: ('artifact://'+a.artifact_id if spec.mode=='referenced' else a.content),
                delivery_mode=spec.mode,operation_kind='retrieval' if spec.mode=='retrieved' else 'transform',
                selector=selector,transform={**transform,'kind':'artifact_reference' if spec.mode=='referenced' else 'snapshot_retrieval'})
            converted.append(result)
        return converted
    return [replace(a,delivery_mode=spec.mode,source_artifact_ids=(a.artifact_id,),
                    delivery_selector=selector,delivery_transform=transform) for a in items]


def deliver(runner,stage,inputs,mid):
    delivered={}
    for port,items in inputs.items():
        items=items if isinstance(items,list) else [items]
        legacy=stage.get('delivery',{}).get(port,{'mode':'full'})
        params=stage.get('parameters',{}).get('delivery',{}).get(port,{})
        if isinstance(legacy,str): legacy={'mode':legacy}
        if params.get('indices') and legacy.get('mode')=='selected' and not legacy.get('selector'):
            legacy={**legacy,'selector':{'artifact_indices':params['indices']}}
        # A port policy can select across all incoming edges. Otherwise each edge
        # carries its own typed policy, allowing different modes on one port.
        if items and all(a.delivery_scope=='port' for a in items):
            items=materialize(runner,stage,items,mid,legacy)
        elif legacy.get('mode')=='selected' and legacy.get('selector',{}).get('artifact_indices') is not None:
            items=materialize(runner,stage,items,mid,legacy)
        else:
            items=[out for artifact in items for out in materialize(
                runner,stage,[artifact],mid,artifact.delivery_spec or legacy)]
        if port=='candidate' and len(items)!=1: raise ValueError('candidate requires one delivered artifact')
        delivered[port]=items[0] if port=='candidate' else items
    return delivered


def deliver_relation(runner,stage,relation,items,mid):
    spec=stage.get('motif_delivery',{}).get(relation,{'mode':'full'})
    return materialize(runner,stage,list(items),mid,spec)


def atomic_stage(runner,stage,inputs,mid):
    ctx=runner._stage_context[mid];ctx['atomic']=True
    inputs=deliver(runner,stage,inputs,mid)
    artifacts=[a for v in inputs.values() for a in (v if isinstance(v,list) else [v])]
    kind=stage['atomic']['kind'];role=stage['atomic']['role']
    params=stage.get('parameters',{})
    if kind=='llm':
        agent=AgentInstance(RoleSlot(role),RoleBinding(**stage['roles'][role]),mid)
        result=runner._call(agent,stage['task'],artifacts)
    else:
        def transform():
            if kind=='tool':
                return runner.search(node_id='search_'+mid,node_name='Atomic search',query=stage['task'])
            if kind=='router':
                if 'constant' in params: return params['constant']
                text=artifacts[-1].content if artifacts else stage['task']
                return params.get('routes',{}).get(text,params.get('default',{}))
            transform=params.get('transform','identity')
            if transform=='constant': return params['value']
            if transform=='identity': return artifacts[-1].content if artifacts else stage['task']
            if transform=='concat': return '\n'.join(a.content for a in artifacts)
            if transform=='json_field': return json.loads(artifacts[-1].content)[params['field']]
            raise ValueError('Unknown deterministic transform')
        result=operation(runner,mid,artifacts,transform,decision=kind=='router',operation_kind=kind)
    reason='router_decision_emitted' if kind=='router' else 'operation_completed'
    runner.trace.emit(event_type='stage_finish',stage_instance_id=ctx['id'],atomic_instance_id=mid,status='completed',completion_reason=reason)
    return MotifResult('atomic',mid,'completed',{'result':result})

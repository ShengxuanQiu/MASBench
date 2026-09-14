"""Adaptive/atomic helpers used by the existing canonical FamilyWorkload runner."""
import json
import time
from dataclasses import replace
from uuid import uuid4
from .contracts import AgentInstance, Artifact, MotifResult, RoleBinding, RoleSlot


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


def skip_stage(runner, name, reason):
    mid='skipped_'+uuid4().hex;sid='stage_'+uuid4().hex
    runner._stage_context[mid]={'id':sid,'nodes':[]}
    runner.trace.emit(event_type='stage_skip',stage_id=name,stage_instance_id=sid,reason=reason)
    return MotifResult('skipped',mid,'skipped',{})


def operation(runner,mid,inputs,fn,*,decision=False,delivery_mode=None):
    ctx=runner._stage_context[mid]
    node='atomic_'+uuid4().hex
    identity={'stage_instance_id':ctx['id'],'motif_instance_id':'' if ctx.get('atomic') else mid,
              'atomic_instance_id':mid,'role':'Coordinator' if decision else 'Worker'}
    parents=list(dict.fromkeys(ctx['parents']+[a.producer_node for a in inputs]))
    runner.trace.emit(event_type='operation_start',node_id=node,operation_kind='tool',parents=parents,**identity)
    started=time.perf_counter()
    for artifact in inputs:
        runner.trace.emit(event_type='dependency',src=artifact.producer_node,dst=node,dependency_kind='data',**identity)
        runner.trace.emit(event_type='artifact_consume',node_id=node,artifact_id=artifact.artifact_id,
                          delivery_mode=delivery_mode or 'full',**identity)
        runner.trace.emit(event_type='artifact_delivery',node_id=node,artifact_id=artifact.artifact_id,
                          producer_operation_id=artifact.producer_node,
                          source_artifact_ids=list(artifact.source_artifact_ids or (artifact.artifact_id,)),
                          delivery_mode=delivery_mode or artifact.delivery_mode,**identity)
    try:
        value=fn()
        content=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False)
    except Exception as exc:
        runner.trace.emit(event_type='operation_fail',node_id=node,error=str(exc),**identity)
        raise
    runner.trace.emit(event_type='operation_finish',node_id=node,tool_snapshot=value,
                      duration_sec=time.perf_counter()-started,**identity)
    result=Artifact(content,node)
    runner._produce(result,identity)
    if decision:
        if not isinstance(value,dict): raise ValueError('Router must return an object')
        runner.trace.emit(event_type='control_decision',node_id=node,decision=value,**identity)
    ctx['nodes'].append(node)
    return result


def deliver(runner,stage,inputs,mid):
    delivered={}
    for port,items in inputs.items():
        items=items if isinstance(items,list) else [items]
        mode=stage.get('delivery',{}).get(port,'full')
        params=stage.get('parameters',{}).get('delivery',{}).get(port,{})
        if mode=='selected':
            indices=params.get('indices',[0])
            if not indices or len(set(indices))!=len(indices) or any(type(i) is not int or i<0 or i>=len(items) for i in indices):
                raise ValueError('Invalid delivery selection')
            items=[items[i] for i in indices]
        if mode=='summarized':
            agent=AgentInstance(RoleSlot('Reducer'),RoleBinding(stage['delivery_instruction']),mid)
            result=runner._call(agent,stage['task'],items)
            items=[replace(result,delivery_mode=mode,source_artifact_ids=tuple(a.artifact_id for a in items))]
        elif mode in {'referenced','retrieved'}:
            converted=[]
            for artifact in items:
                result=operation(runner,mid,[artifact],
                    lambda a=artifact: ('artifact://'+a.artifact_id if mode=='referenced' else a.content),delivery_mode=mode)
                converted.append(replace(result,delivery_mode=mode,source_artifact_ids=(artifact.artifact_id,)))
            items=converted
        else:
            items=[replace(a,delivery_mode=mode,source_artifact_ids=(a.artifact_id,)) for a in items]
        if port=='candidate' and len(items)!=1: raise ValueError('candidate requires one delivered artifact')
        delivered[port]=items[0] if port=='candidate' else items
    return delivered


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
        result=operation(runner,mid,artifacts,transform,decision=kind=='router')
    runner.trace.emit(event_type='stage_finish',stage_instance_id=ctx['id'],atomic_instance_id=mid,status='completed')
    return MotifResult('atomic',mid,'completed',{'result':result})

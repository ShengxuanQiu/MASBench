"""Logical pressure metrics; never report logical state as physical KV residency."""
from collections import Counter,defaultdict


def _distribution(values):
    values=sorted(values)
    if not values:return {'count':0,'min':None,'mean':None,'p50':None,'p95':None,'max':None}
    def percentile(p):
        x=(len(values)-1)*p;lo=int(x)
        return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(x-lo)
    return {'count':len(values),'min':values[0],'mean':sum(values)/len(values),
            'p50':percentile(.5),'p95':percentile(.95),'max':values[-1]}


def queue_slope(points,arrival_span):
    if arrival_span<=0:return None
    samples=[];index=0;value=0
    for i in range(21):
        at=arrival_span*i/20
        while index<len(points) and points[index]['time_sec']<=at:
            value=points[index].get('ready_waiting',0);index+=1
        samples.append((at,value))
    mx=sum(x for x,y in samples)/len(samples);my=sum(y for x,y in samples)/len(samples)
    return sum((x-mx)*(y-my) for x,y in samples)/sum((x-mx)**2 for x,y in samples)


def pressure_metrics(graph,events,runtime):
    llm={n for n,op in graph.operations.items() if op.kind=='llm'}
    roots=[]
    for n in llm:
        seen=set();pending=list(graph.operations[n].parents);has_llm=False
        while pending:
            p=pending.pop()
            if p in seen:continue
            seen.add(p)
            if p in llm:has_llm=True;break
            pending.extend(graph.operations[p].parents)
        if not has_llm:roots.append(n)
    tokens={e['operation_id']:e.get('backend_prompt_tokens') for e in events if e.get('canonical_type')=='request_finish'}
    complete=len(tokens)==len(llm) and all(v is not None for v in tokens.values())
    baseline=sum(tokens[n] for n in roots) if complete else None
    finishes={e['operation_id']:e['relative_time_sec'] for e in events if e.get('canonical_type') in {'operation_finish','request_finish'}}
    produced={e['artifact_id']:e['relative_time_sec'] for e in events if e.get('canonical_type')=='artifact_produce'}
    consumers=defaultdict(set)
    for a,c in graph.consumptions:consumers[a].add(c)
    run_end=max(e['relative_time_sec'] for e in events if e.get('canonical_type')=='run_finish')
    changes=Counter();byte_seconds=0;lifetimes=[]
    for aid,artifact in graph.artifacts.items():
        content=artifact['content'];size=len(str(content).encode('utf-8'))
        begin=produced[aid];end=max([finishes[c] for c in consumers[aid]],default=run_end)
        end=max(begin,end);changes[begin]+=size;changes[end]-=size;byte_seconds+=size*(end-begin);lifetimes.append(end-begin)
    current=peak=0;residency=[]
    for at,change in sorted(changes.items()):
        current+=change;peak=max(peak,current);residency.append({'time_sec':at,'logical_artifact_bytes':current})
    ready_area=0
    for a,b in zip(runtime['timeline'],runtime['timeline'][1:]):
        ready_area+=a['ready_waiting']*(b['time_sec']-a['time_sec'])
    return {'work_amplification':{'llm_requests_per_workflow':len(llm),'baseline':'one atomic LLM request; not quality-equivalent work'},
            'total_input_token_amplification':{'value':sum(tokens.values())/baseline if baseline else None,
                'baseline_root_request_tokens':baseline,'source':'backend-reported' if complete else 'unavailable'},
            'temporal_pressure':{'ready_wait_seconds':ready_area,'peak_ready_waiting':max((p['ready_waiting'] for p in runtime['timeline']),default=0),
                                 'peak_llm_inflight':max((p['llm_inflight'] for p in runtime['timeline']),default=0)},
            'synchronization_exposure_sec':sum(b['aggregate_branch_wait_sec'] for b in runtime['barriers']),
            'logical_state_residency':{'source':'estimated','peak_artifact_bytes':peak,'byte_seconds':byte_seconds,
                'artifact_lifetime_sec':_distribution(lifetimes),'timeline':residency,
                'definition':'Unique artifact payload retained from produce until last consumer finish; unconsumed outputs until run end. Excludes object overhead, allocator, KV and Mamba state.'}}

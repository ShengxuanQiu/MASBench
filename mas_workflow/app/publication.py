"""Generate and optionally execute fingerprinted publication experiment cells."""
import argparse
from dataclasses import asdict
import json
import itertools
from pathlib import Path
from .specs import (AtomicStage,DeploymentSpec,ExperimentConfig,MotifSpec,RoleTask,
                    StageSpec,StructureSpec,TaskBinding,WorkflowSpec,read_config,load_experiment,experiment_from_dict)
from .study import atomic_json,digest


def preset(name,task,deployment,*,width=4,rounds=2,max_revisions=2,connectivity='ring',k=None,sparsity=None,requests=8):
    pa=MotifSpec('ParallelAggregate',width=width)
    er=MotifSpec('EvaluateRefine',width=1,max_revisions=max_revisions)
    de=MotifSpec('DispatchExecute',width=width)
    if name in {'matched_parallel','matched_chain'}:
        stages=[StageSpec(f'op{i}',atomic=AtomicStage(),depends_on=(f'op{i-1}',) if i and name=='matched_chain' else ()) for i in range(requests)]
        motifs={}
    elif name=='parallel_refine':
        motifs={'p':pa,'e':er};stages=[StageSpec('parallel','p'),StageSpec('review','e',inputs={'candidate':'parallel.result'})]
    elif name=='dispatch_parallel_refine':
        motifs={'d':de,'p':pa,'e':er};stages=[StageSpec('dispatch','d'),StageSpec('parallel','p',inputs={'context':'dispatch.result'}),StageSpec('review','e',inputs={'candidate':'parallel.result'})]
    elif name=='fanout_fanin':
        motifs={'p':pa};stages=[StageSpec('root','p'),StageSpec('left','p',inputs={'context':'root.result'}),StageSpec('right','p',inputs={'context':'root.result'}),StageSpec('join','p',inputs={'context':['left.result','right.result']})]
    else:
        families={'dispatch':'DispatchExecute','parallel':'ParallelAggregate','refine':'EvaluateRefine','peer':'PeerExchange'}
        if name not in families:raise ValueError('Unknown preset')
        motif=MotifSpec(families[name],width=width,rounds=rounds,max_revisions=max_revisions,
                        **({'connectivity':connectivity,'k':k,'sparsity':sparsity} if name=='peer' else {}))
        motifs={'m':motif};stages=[StageSpec('one','m')]
    return ExperimentConfig(StructureSpec(motifs,WorkflowSpec(tuple(stages))),task,deployment)


def set_path(obj,path,value):
    parts=path.split('.')
    if not parts or any(not p for p in parts):raise ValueError('Invalid override path')
    target=obj
    for part in parts[:-1]:
        if isinstance(target,list):target=target[int(part)]
        else:
            if part not in target:raise ValueError('Unknown override path: '+path)
            target=target[part]
    key=parts[-1]
    if isinstance(target,list):target[int(key)]=value
    elif key not in target:raise ValueError('Unknown override path: '+path)
    else:target[key]=value


def combinations(workload):
    rows=list(workload.get('overrides',workload.get('parameters',[{}])))
    factors=workload.get('factors',{})
    if not factors:return rows
    expanded=[]
    for row in rows:
        for values in itertools.product(*(factors[k] for k in factors)):
            expanded.append({**row,**dict(zip(factors,values))})
    return expanded


def generate(path,output,*,execute=False,resume=False):
    path=Path(path).resolve();plan=read_config(path);base=path.parent
    if set(plan)-{'tasks','deployments','workloads','study','capacity'}:raise ValueError('Unknown publication manifest field')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    deployments={name:read_config(base/p) for name,p in plan['deployments'].items()}
    tasks={name:read_config(base/p) for name,p in plan['tasks'].items()}
    bases={str(w['base']):asdict(load_experiment(base/w['base'])) for w in plan['workloads'] if 'base' in w}
    fingerprint=digest({'plan':plan,'deployments':deployments,'tasks':tasks,'bases':bases})
    manifest=output/'matrix_manifest.json'
    if manifest.exists() and (not resume or json.loads(manifest.read_text())['fingerprint']!=fingerprint):
        raise ValueError('Publication manifest exists or fingerprint differs')
    dep_paths={}
    for name,dep in deployments.items():
        DeploymentSpec(**dep)
        p=output/('deployment_'+digest(name)[:12]+'.json');atomic_json(p,dep);dep_paths[name]=str(p)
    cells=[]
    for workload in plan['workloads']:
        if set(workload)-{'preset','base','parameters','overrides','factors','name'} or ('preset' in workload)==('base' in workload):
            raise ValueError('Workload requires exactly one preset or base')
        for parameters in combinations(workload):
            for task_name,task in tasks.items():
                binding=TaskBinding(**{**task,'roles':{k:RoleTask(**v) for k,v in task.get('roles',{}).items()}})
                if 'preset' in workload:
                    legacy={k:v for k,v in parameters.items() if '.' not in k}
                    exp=preset(workload['preset'],binding,DeploymentSpec(),**legacy)
                    raw=asdict(exp)
                    for path,value in parameters.items():
                        if '.' in path:set_path(raw,path,value)
                    exp=experiment_from_dict(raw)
                else:
                    raw=json.loads(json.dumps(bases[str(workload['base'])]));raw['task']=asdict(binding)
                    for path,value in parameters.items():set_path(raw,path,value)
                    exp=experiment_from_dict(raw)
                key=digest([workload,parameters,task_name])[:16]
                cell=output/key;cell.mkdir(exist_ok=True)
                atomic_json(cell/'experiment.json',asdict(exp))
                study={**plan['study'],'mode':'run','experiments':[str(cell/'experiment.json')],'deployments':dep_paths}
                atomic_json(cell/'study.json',study)
                atomic_json(cell/'capacity.json',{'study':'study.json','capacity':plan.get('capacity',{})})
                cells.append({'id':key,'preset':workload.get('preset'),'workload':workload.get('name',workload.get('preset',workload.get('base'))),
                              'parameters':parameters,'overrides':parameters,'task':task_name,'config':str(cell/'capacity.json')})
    atomic_json(manifest,{'fingerprint':fingerprint,'resolved_plan':plan,'cells':cells})
    if execute:
        from .capacity import run_capacity
        for cell in cells:run_capacity(cell['config'],output/cell['id']/'results',resume=resume)
    return cells


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    print(json.dumps({'cells':len(generate(a.manifest,a.output,execute=a.execute,resume=a.resume))}))


if __name__=='__main__':main()

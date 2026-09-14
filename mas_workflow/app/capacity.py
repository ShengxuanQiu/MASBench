"""SLO-defined capacity boundary search; no invented hardware saturation labels."""
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
from .study import run_study,atomic_json,digest,source_metadata
from .specs import read_config
from .workload_analysis import distribution


def aggregate(values,seed=42):
    stats=distribution(values)
    if len(values)<2:
        return {**stats,'mean_ci95':None,'ci_unit':'independent_repetitions'}
    rng=random.Random(seed)
    means=sorted(sum(rng.choice(values) for _ in values)/len(values) for _ in range(2000))
    return {**stats,'mean_ci95':[means[49],means[1949]],'ci_unit':'independent_repetitions'}


def capacity_summary(rows,*,min_slo_success=.95,max_failure_fraction=.01):
    groups=defaultdict(list)
    for row in rows: groups[(row['deployment'],row['rate'])].append(row)
    deployments=defaultdict(list)
    for (deployment,rate),runs in sorted(groups.items()):
        # Client overflow or launch failure does not identify backend capacity.
        valid=all(not r['client_overflow'] and not r.get('analysis_errors') and not r.get('replay_equivalence_failures') and r.get('internal_request_count_complete',True) for r in runs)
        success=[r['slo_success_fraction'] for r in runs]
        failure=[r['failed_or_rejected']/r['offered'] for r in runs]
        passed=valid and min(success)>=min_slo_success and max(failure)<=max_failure_fraction
        metrics={}
        for key in ('goodput_qps','internal_qps','completed_workflow_qps','slo_success_fraction'):
            metrics[key]=aggregate([r[key] for r in runs])
        for stat in ('mean','p50','p95'):
            metrics['e2e_'+stat+'_sec']=aggregate([r['completed_e2e_sec'][stat] for r in runs if r['completed_e2e_sec'].get(stat) is not None])
        metrics['failure_fraction']=aggregate(failure)
        metrics['queue_growth_client_per_sec']=aggregate([r['queue_growth_client_per_sec'] for r in runs if r.get('queue_growth_client_per_sec') is not None])
        deployments[deployment].append({'rate':rate,'valid_client':valid,'slo_pass':passed,'repetitions':len(runs),'metrics':metrics})
    output={}
    for name,points in deployments.items():
        failed=next((p for p in points if p['valid_client'] and not p['slo_pass']),None)
        prior=[p['rate'] for p in points if p['slo_pass'] and (failed is None or p['rate']<failed['rate'])]
        bracket=[max(prior),failed['rate']] if failed and prior else None
        invalid=any(not p['valid_client'] for p in points)
        nonmonotonic=bool(failed and any(p['slo_pass'] and p['rate']>failed['rate'] for p in points))
        knee=bracket[1] if bracket and not invalid and not nonmonotonic else None
        for p in points:p['rho']=p['rate']/knee if knee else None
        output[name]={'lambda_knee':knee,'bracket':bracket,'points':points,
            'status':'client_limited' if invalid else 'nonmonotonic' if nonmonotonic else 'bracketed' if bracket else 'right_censored' if failed is None else 'left_censored',
            'definition':'First failing tested QPS above a passing QPS under configured SLO/error criteria; a finite-cohort boundary, not a resource-profiler diagnosis.'}
    return output


def run_capacity(path,output,*,resume=False):
    path=Path(path).resolve();plan=read_config(path)
    if set(plan)-{'study','capacity'}:raise ValueError('Unknown capacity configuration')
    policy=plan.get('capacity',{})
    if set(policy)-{'dense_points','min_slo_success','max_failure_fraction'}:raise ValueError('Unknown capacity policy')
    dense=policy.get('dense_points',3)
    if type(dense) is not int or dense<1:raise ValueError('dense_points must be positive')
    for key,default in [('min_slo_success',.95),('max_failure_fraction',.01)]:
        if not 0<=policy.get(key,default)<=1:raise ValueError('Invalid capacity threshold')
    study_path=(path.parent/plan['study']).resolve()
    study=read_config(study_path)
    for key in ('experiments','traces'):
        if key in study:study[key]=[str((study_path.parent/p).resolve()) for p in study[key]]
    if study.get('trace_corpus'):study['trace_corpus']=str((study_path.parent/study['trace_corpus']).resolve())
    study['deployments']={k:str((study_path.parent/p).resolve()) for k,p in study['deployments'].items()}
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    fingerprint=digest({'plan':plan,'study':study,'source':source_metadata()['app_source_sha256']})
    manifest=output/'capacity_manifest.json'
    if manifest.exists():
        if not resume or json.loads(manifest.read_text())['fingerprint']!=fingerprint:raise ValueError('Capacity resume fingerprint mismatch')
    else:atomic_json(manifest,{'fingerprint':fingerprint,'plan':plan,'resolved_study':study})
    coarse_path=output/'coarse.json';atomic_json(coarse_path,study)
    rows=run_study(coarse_path,output/'coarse',resume=resume)
    criteria={k:v for k,v in policy.items() if k!='dense_points'}
    coarse=capacity_summary(rows,**criteria)
    for name,result in coarse.items():
        if result['status']!='bracketed':continue
        lo,hi=result['bracket']
        refined={**study,'deployments':{name:study['deployments'][name]},
                 'rates':[lo+(hi-lo)*i/(dense+1) for i in range(1,dense+1)]}
        key=digest(name)[:12];refined_path=output/(key+'.json');atomic_json(refined_path,refined)
        rows+=run_study(refined_path,output/('dense_'+key),resume=resume)
    result=capacity_summary(rows,**criteria)
    atomic_json(output/'capacity_results.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();print(json.dumps(run_capacity(a.config,a.output,resume=a.resume),indent=2))


if __name__=='__main__':main()

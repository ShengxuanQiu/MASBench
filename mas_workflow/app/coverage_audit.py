"""Validate Section 4.1 annotations and calculate specification coverage."""
import argparse
from collections import Counter
import json
from pathlib import Path

FAMILIES={"DispatchExecute","ParallelAggregate","EvaluateRefine","PeerExchange"}


def validate(data,schema_path=None):
    import jsonschema
    schema_path=Path(schema_path or Path(__file__).parents[2]/'evaluation'/'coverage_audit.schema.json')
    jsonschema.validate(data,json.loads(schema_path.read_text()))
    for paper in data['papers']:
        ids={s['stage_id'] for s in paper['stages']}
        if len(ids)!=len(paper['stages']):raise ValueError('Duplicate stage_id')
        for key in ('workflow_dependencies','benchmark_dependencies','information_relations','benchmark_information_relations'):
            for edge in paper[key]:
                if edge['producer'] not in ids or edge['consumer'] not in ids:
                    raise ValueError(f'{key} references an unknown stage')
    return data


def f1(gold,pred):
    gold,pred=set(gold),set(pred);tp=len(gold&pred);fp=len(pred-gold);fn=len(gold-pred)
    precision=tp/(tp+fp) if tp+fp else (1.0 if not gold else 0.0)
    recall=tp/(tp+fn) if tp+fn else 1.0
    value=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return {'tp':tp,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':value}


def calculate(data):
    validate(data)
    stage_total=stage_hit=workflow_hit=0
    dep_gold=[];dep_pred=[];info_gold=[];info_pred=[];dep_macro=[];info_macro=[]
    intersections=Counter()
    for paper in data['papers']:
        matched=True;families=set()
        for stage in paper['stages']:
            stage_total+=1
            covered=stage.get('canonical_family') in FAMILIES and stage.get('benchmark_family')==stage.get('canonical_family')
            stage_hit+=covered;matched &= covered
            if stage.get('canonical_family') in FAMILIES:families.add(stage['canonical_family'])
        if families:intersections[tuple(sorted(families))]+=1
        dg={(e['producer'],e['consumer']) for e in paper['workflow_dependencies']}
        dp={(e['producer'],e['consumer']) for e in paper['benchmark_dependencies']}
        def info(edge):return (edge['producer'],edge['consumer'],edge['artifact_semantics'],edge['delivery_mode'])
        ig={info(e) for e in paper['information_relations']};ip={info(e) for e in paper['benchmark_information_relations']}
        workflow_hit += matched and dg==dp
        prefix=paper['paper_id']
        dep_gold.extend((prefix,*x) for x in dg);dep_pred.extend((prefix,*x) for x in dp)
        info_gold.extend((prefix,*x) for x in ig);info_pred.extend((prefix,*x) for x in ip)
        dep_macro.append(f1(dg,dp)['f1']);info_macro.append(f1(ig,ip)['f1'])
    dep=f1(dep_gold,dep_pred);info_score=f1(info_gold,info_pred)
    dep['macro_f1']=sum(dep_macro)/len(dep_macro) if dep_macro else None
    info_score['macro_f1']=sum(info_macro)/len(info_macro) if info_macro else None
    return {'schema':'masbench_coverage_metrics_v1','source_kind':data.get('source_kind','unspecified'),
        'paper_count':len(data['papers']),
        'C_sub':{'numerator':stage_hit,'denominator':stage_total,'value':stage_hit/stage_total if stage_total else None,
                 'definition':'Local stages whose annotated canonical family is represented by the benchmark family.'},
        'C_wf_0':{'numerator':workflow_hit,'denominator':len(data['papers']),'value':workflow_hit/len(data['papers']) if data['papers'] else None,
                  'definition':'Papers with every local stage family represented and an exact directed stage-dependency set.'},
        'F_dep_spec':dep,'F_info_spec':info_score,
        'plot_data':{'family_intersections':[{'families':list(k),'papers':v} for k,v in sorted(intersections.items())],
                     'metrics':[{'label':'Local stages','numerator':stage_hit,'denominator':stage_total},
                                {'label':'Workflow exact','numerator':workflow_hit,'denominator':len(data['papers'])},
                                {'label':'Dependency micro F1','value':dep['f1']},
                                {'label':'Information micro F1','value':info_score['f1']}]},
        'warning':'Example input is schema demonstration only; it is not evidence for paper claims.' if data.get('source_kind')=='example' else None}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--schema');a=p.parse_args();data=json.loads(Path(a.input).read_text())
    result=calculate(validate(data,a.schema));Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({'papers':result['paper_count'],'source_kind':result['source_kind']}))


if __name__=='__main__':main()

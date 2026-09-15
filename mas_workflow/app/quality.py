"""Out-of-band deterministic quality guardrails.

These evaluators consume recorded final artifacts after the measured workload.
They never add operations, latency, or tokens to the serving trace.
"""
import json
import math
from typing import Protocol,Any


class QualityEvaluator(Protocol):
    def __call__(self,value: str,spec: dict[str,Any]) -> dict[str,Any]: ...


EXTERNAL_EVALUATORS: dict[str,QualityEvaluator]={}


def register_evaluator(name,evaluator):
    if not name or name in {'exact','numeric','json_field'} or not callable(evaluator):
        raise ValueError('Invalid external evaluator registration')
    EXTERNAL_EVALUATORS[name]=evaluator


def final_output(graph):
    children={s['logical_stage_id']:set() for s in graph.stages.values()}
    by_logical={s['logical_stage_id']:s for s in graph.stages.values()}
    for stage in graph.stages.values():
        for parent in stage['dependencies']:
            children[parent].add(stage['logical_stage_id'])
    sinks={name for name,value in children.items() if not value and by_logical[name]['activation']!='skipped'}
    outputs={}
    for name in sorted(sinks):
        sid=by_logical[name]['stage_instance_id']
        candidates=[a['content'] for a in graph.artifacts.values()
                    if graph.operations[a['producer']].identities.get('stage_instance_id')==sid]
        if candidates: outputs[name]=candidates[-1]
    if len(outputs)==1:return next(iter(outputs.values()))
    return json.dumps(outputs,sort_keys=True)


def _field(value,path):
    for key in path.split('.') if path else []:
        value=value[int(key)] if isinstance(value,list) else value[key]
    return value


def evaluate(value,spec):
    if not isinstance(spec,dict) or set(spec)-{'method','expected','field','tolerance'}:
        raise ValueError('Invalid quality evaluator configuration')
    method=spec.get('method','exact');expected=spec.get('expected')
    if method=='exact':
        observed=value;passed=observed==expected;score=1.0 if passed else 0.0
    elif method in {'numeric','json_field'}:
        parsed=json.loads(value) if method=='json_field' or isinstance(value,str) else value
        observed=_field(parsed,spec.get('field',''))
        tolerance=spec.get('tolerance',0)
        if not isinstance(tolerance,(int,float)) or not math.isfinite(tolerance) or tolerance<0:
            raise ValueError('Quality tolerance must be finite and nonnegative')
        if not isinstance(observed,(int,float)) or not isinstance(expected,(int,float)):
            raise ValueError('Numeric quality evaluator requires numeric values')
        error=abs(observed-expected);passed=error<=tolerance
        score=max(0.0,1-error/max(abs(expected),tolerance,1))
    elif method.startswith('external:'):
        name=method.split(':',1)[1]
        if name not in EXTERNAL_EVALUATORS:raise ValueError('External quality evaluator is not registered')
        result=EXTERNAL_EVALUATORS[name](value,spec)
        if not isinstance(result,dict) or type(result.get('pass')) is not bool or not isinstance(result.get('score'),(int,float)):
            raise ValueError('External quality evaluator must return numeric score and boolean pass')
        observed=result.get('observed');passed=result['pass'];score=float(result['score'])
        if not math.isfinite(score) or not 0<=score<=1:raise ValueError('External quality score must be in [0,1]')
    else:
        raise ValueError('Unknown quality evaluator')
    return {'method':method,'score':score,'pass':passed,'observed':observed,'expected':expected,
            'scope':'out_of_band','included_in_serving_workload':False,
            'latency_sec_in_benchmark':0,'tokens_in_benchmark':0}

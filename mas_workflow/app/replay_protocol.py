"""Explicit cache evidence and bounded output-length/identity agreement checks."""
import re
from .backend_metrics import fetch_prometheus_metrics,metrics_url_from_base_url


def verify_cache_protocol(deployment,protocol):
    if protocol in {'uncontrolled','warm_sequence'}:
        return {'protocol':protocol,'verified':False,'source':'unavailable'}
    if protocol not in {'cache_disabled','warm_cache_enabled'}:raise ValueError('Unknown cache protocol')
    desired=protocol=='warm_cache_enabled'
    if deployment.backend=='mock':
        return {'protocol':protocol,'verified':False,'source':'synthetic_validation','enabled':desired}
    url=deployment.telemetry.get('metrics_url') or metrics_url_from_base_url(deployment.endpoint)
    metrics=fetch_prometheus_metrics(url)
    states=set()
    for name in metrics:
        match=re.search(r'enable_prefix_caching="(true|false)"',name,re.I)
        if match:states.add(match[1].lower()=='true')
    metric=deployment.telemetry.get('metadata',{}).get('prefix_cache_enabled_metric')
    if metric in metrics and metrics[metric] in (0,1):states.add(bool(metrics[metric]))
    if states!={desired}:
        raise ValueError('Cache protocol cannot be verified from endpoint configuration metrics; expose enable_prefix_caching or configure prefix_cache_enabled_metric')
    return {'protocol':protocol,'verified':True,'source':'backend-reported','enabled':desired,'metrics_url':url}


def length_agreement(pairs,tolerance=0):
    if not 0<=tolerance<=1:raise ValueError('output length tolerance must be in [0,1]')
    unknown=0; mismatches=[]
    for row in pairs:
        a,b=row['recorded'],row['replayed']
        if a is None or b is None:unknown+=1;continue
        delta=abs(b-a)/max(1,a)
        if delta>tolerance:mismatches.append({**row,'relative_variation':delta})
    return {'pass':not unknown and not mismatches,'relative_tolerance':tolerance,'unavailable':unknown,
            'mismatches':mismatches,'strict_workload_equivalence':False,
            'reason':'Length agreement does not enforce output token sequence or tokenizer equivalence.'}


def identity_agreement(source,target):
    keys=('model_revision','tokenizer_sha256','chat_template_sha256')
    missing=[k for k in keys if not source.get(k) or not target.get(k)]
    invalid=[k for k in keys if k.endswith('_sha256') and any(v.get(k) and not re.fullmatch(r'[0-9a-f]{64}',v[k]) for v in (source,target))]
    mismatch=[k for k in keys if source.get(k) and target.get(k) and source[k]!=target[k]]
    return {'status':'mismatch' if mismatch else 'unavailable' if missing or invalid else 'matched',
            'missing':missing,'invalid':invalid,'mismatched':mismatch,'source':'recorded_deployment_identity_metadata'}


def local_identity(model_dir, revision):
    """Hash local tokenizer/template bytes; endpoint model identity needs operator attestation."""
    import hashlib
    import json
    from pathlib import Path
    root=Path(model_dir).resolve()
    names=('tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json',
           'merges.txt','tokenizer.model','added_tokens.json')
    files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names if (root/name).is_file()}
    if not files: raise ValueError('No tokenizer files found')
    cfg=json.loads((root/'tokenizer_config.json').read_text()) if (root/'tokenizer_config.json').exists() else {}
    templates={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('**/*.jinja')}
    if cfg.get('chat_template') is not None:templates['tokenizer_config.chat_template']=cfg['chat_template']
    if not templates: raise ValueError('No chat template found; do not invent identity')
    def hash_value(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'identity':{'model_revision':revision,'tokenizer_sha256':hash_value(files),'chat_template_sha256':hash_value(templates)},
            'evidence':{'model_dir':str(root),'tokenizer_files':files,'templates':templates,
                        'scope':'local files only; does not attest endpoint weights or runtime template overrides'}}


if __name__=='__main__':
    import argparse,json
    from pathlib import Path
    p=argparse.ArgumentParser(description='Compute local tokenizer and chat-template identity evidence')
    p.add_argument('--model-dir',required=True);p.add_argument('--revision',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();Path(a.output).write_text(json.dumps(local_identity(a.model_dir,a.revision),indent=2))

"""Verify fixed-workload invariants before comparing replay timing across endpoints."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from .execution_graph import ExecutionGraph
from .workload_analysis import analyze_events, read_events


def compare_replay(source_events, replay_events, *, output_length_tolerance=0):
    source=ExecutionGraph.from_events(source_events,replay=True)
    replay=ExecutionGraph.from_events(replay_events,replay=True)
    manifests=[e for e in replay_events if e.get("event_type")=="replay_manifest"]
    if len(manifests)!=1 or manifests[0]["source_run_id"]!=source.run_id:
        raise ValueError("Replay source manifest mismatch")
    manifest=manifests[0]
    ids=manifest["source_operation_ids"]
    if set(ids)!=set(source.operations) or set(ids.values())!=set(replay.operations):
        raise ValueError("Replay operation mapping mismatch")
    errors=[]
    if {(ids[a],ids[b],k) for a,b,k in source.edges} != replay.edges:
        errors.append("dependency_edges")
    tokens=[]
    for oid,op in source.operations.items():
        other=replay.operations[ids[oid]]
        if op.kind!=other.kind or {ids[p] for p in op.parents}!=other.parents:
            errors.append("operation_structure:"+oid)
        if op.kind=="llm":
            a,b=dict(op.payload),dict(other.payload)
            a.pop("model",None); b.pop("model",None)
            if a!=b:
                errors.append("request_payload:"+oid)
            tokens.append({"source_operation_id":oid,"recorded":op.output_tokens,"replayed":other.output_tokens,
                           "equal":op.output_tokens==other.output_tokens if op.output_tokens is not None and other.output_tokens is not None else None})
        elif op.snapshot!=other.snapshot:
            errors.append("tool_snapshot:"+oid)
    def artifacts(graph, convert):
        return Counter((convert[v["producer"]],v["hash"]) for v in graph.artifacts.values())
    if artifacts(source,ids)!=artifacts(replay,{n:n for n in replay.operations}):
        errors.append("artifact_snapshots")
    def uses(graph,convert):
        return Counter((convert[graph.artifacts[a]["producer"]],graph.artifacts[a]["hash"],convert[c]) for a,c in graph.consumptions)
    if uses(source,ids)!=uses(replay,{n:n for n in replay.operations}):
        errors.append("artifact_consumption")
    if Counter((ids[e["node_id"]],json.dumps(e["decision"],sort_keys=True)) for e in source.decisions)!=Counter((e["node_id"],json.dumps(e["decision"],sort_keys=True)) for e in replay.decisions):
        errors.append("control_decisions")
    def deliveries(graph,convert):
        return Counter((convert[e["operation_id"]],graph.artifacts[e["artifact_id"]]["hash"],e["delivery_mode"],
                        tuple(sorted(graph.artifacts[a]["hash"] for a in e.get("source_artifact_ids",[])))) for e in graph.deliveries)
    if deliveries(source,ids)!=deliveries(replay,{n:n for n in replay.operations}):errors.append("delivery_provenance")
    def skips(events):
        return Counter((e['stage_id'],e.get('reason')) for e in events if e.get('canonical_type')=='stage_skip')
    if skips(source_events)!=skips(replay_events):errors.append('skipped_stages')
    from .replay_protocol import length_agreement,identity_agreement
    lengths=length_agreement(tokens,output_length_tolerance)
    identities=identity_agreement(manifest.get("source_identity",{}),manifest["deployment"].get("identity",{}))
    analysis=analyze_events(replay_events)
    return {"source_run_id":source.run_id,"replay_run_id":replay.run_id,
            "fixed_workload_invariants_pass":not errors,"errors":errors,
            "capabilities":manifest["capabilities"],"deployment":manifest["deployment"],
            "output_length_agreement":lengths,"identity_agreement":identities,"output_length_comparison":tokens,"runtime":analysis["runtime"],
            "bottleneck_classification":None,
            "interpretation":"Client timings do not identify compute, bandwidth, or memory-capacity bottlenecks without backend/device evidence."}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True)
    p.add_argument("--replays",required=True,nargs="+")
    p.add_argument("--output",required=True)
    p.add_argument('--output-length-tolerance',type=float,default=0)
    args=p.parse_args()
    source=read_events(args.source)
    rows=[compare_replay(source,read_events(path),output_length_tolerance=args.output_length_tolerance) for path in args.replays]
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(rows,indent=2))
    return int(any(not r["fixed_workload_invariants_pass"] for r in rows))


if __name__=="__main__":
    raise SystemExit(main())

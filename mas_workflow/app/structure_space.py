"""Generate separated-factor configurations and controlled mock realized graphs."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from .benchmark import build_experiment
from .llm_backends import MockLLM
from .specs import (DeploymentSpec, ExperimentConfig, MotifSpec, StageSpec,
                    StructureSpec, TaskBinding, WorkflowSpec)
from .study import atomic_json
from .workload_analysis import analyze_events


class ControlledMock(MockLLM):
    """Synthetic control policy for structural coverage, never a task-quality result."""
    def __init__(self, revisions):
        super().__init__()
        self.revisions = revisions

    def invoke(self, system, user, metadata):
        result = super().invoke(system,user,metadata)
        if metadata.get("role_slot") == "evaluator":
            decision = "revise" if metadata.get("round_id",0) < self.revisions else "accept"
            return replace(result,content=json.dumps({"decision":decision,"feedback":"Synthetic structural control policy."}))
        return result


def generate_space(output, *, widths=(2,4,8), rounds=(0,1,2), revisions=(0,1,3), record=False):
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh generated corpus directory")
    output.mkdir(parents=True,exist_ok=True)
    specs=[]
    for width in widths:
        specs.append((f"dispatch_w{width}",MotifSpec("DispatchExecute",width=width)))
        specs.append((f"parallel_w{width}",MotifSpec("ParallelAggregate",width=width)))
        for r in rounds:
            for shape in ("ring","all_to_all","pairwise","random_k"):
                specs.append((f"peer_w{width}_r{r}_{shape}",MotifSpec("PeerDeliberation",width=width,rounds=r,connectivity=shape)))
    for rev in revisions:
        specs.append((f"refine_r{rev}",MotifSpec("EvaluateRefine",width=1,max_revisions=rev)))
    rows=[]
    for name,motif in specs:
        exp=ExperimentConfig(StructureSpec({"m":motif},WorkflowSpec((StageSpec("one","m"),))),
                             TaskBinding("Compare two plans and explain their tradeoffs."),DeploymentSpec())
        atomic_json(output/(name+".json"),asdict(exp))
        row={"id":name,"experiment":name+".json","structure":asdict(exp.structure),
             "provenance":"generated_space", "control_policy":"accept_after_revision_limit"}
        if record:
            runner=build_experiment(exp,trace_dir=output/"traces")
            runner.config.export_trace_views=False
            runner.llm=ControlledMock(motif.max_revisions)
            runner.trace.emit(event_type="structural_control_manifest",synthetic=True,
                              acceptance_policy="accept_after_revision_limit",revisions=motif.max_revisions)
            summary=runner.run()
            if summary["status"] not in {"accepted","completed"}:
                raise RuntimeError(f"Structural recording failed: {name}")
            result=analyze_events(runner.trace.events)
            atomic_json(output/"analysis"/(name+".json"),result)
            row.update(trace=str(runner.trace.trace_path),metrics=result["structure"])
        rows.append(row)
    atomic_json(output/"corpus_manifest.json",{"schema":"masbench_generated_space_v1","runs":rows,
                "interpretation":"Synthetic realized operation graphs under explicit mock controls; not surveyed systems, task fidelity, or GPU performance."})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    p.add_argument("--widths",type=int,nargs="+",default=[2,4,8])
    p.add_argument("--rounds",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--revisions",type=int,nargs="+",default=[0,1,3])
    p.add_argument("--record",action="store_true")
    args=p.parse_args()
    rows=generate_space(args.output,widths=args.widths,rounds=args.rounds,revisions=args.revisions,record=args.record)
    print(json.dumps({"generated":len(rows),"output":str(Path(args.output).resolve())}))


if __name__ == "__main__":
    main()

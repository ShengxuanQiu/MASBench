from dataclasses import replace
import itertools
import json
import random
import threading
import time

import pytest

from app.benchmark import build_experiment
from app.replay import replay_trace
from app.replay_compare import compare_replay
from app.specs import DeploymentSpec, ExperimentConfig, MotifSpec, StageSpec, StructureSpec, TaskBinding, WorkflowSpec
from app.structure_space import ControlledMock, generate_space
from app.study import arrival_offsets, run_open_loop, run_study
from app.workload_analysis import analyze_events, dag_metrics
from app.llm_backends import MockLLM


def exp(family="ParallelAggregate", **kwargs):
    return ExperimentConfig(StructureSpec({"m":MotifSpec(family,**kwargs)},WorkflowSpec((StageSpec("one","m"),))),TaskBinding("Compare plans"),DeploymentSpec())


def record(tmp_path,experiment):
    r=build_experiment(experiment,trace_dir=tmp_path)
    r.config.export_trace_views=False
    r.llm=ControlledMock(next(iter(experiment.structure.motifs.values())).max_revisions)
    assert r.run()["status"] in {"accepted","completed"}
    return r


def test_exact_width_differs_from_level_width():
    # a -> b, x -> y -> z. Antichain {b,z} not necessarily same level.
    p={"a":set(),"b":{"a"},"c":{"a"},"d":{"b"},"e":{"b"}}
    m,_,_=dag_metrics(p)
    assert m["width_max_antichain"]==3
    assert m["width_level_lower_bound"]==2
    assert m["depth_nodes"]==3
    assert m["dag_edge_density"]==.4
    assert dag_metrics(p,exact_width_limit=2)[0]["width_max_antichain"] is None


def test_matching_against_bruteforce():
    rng=random.Random(7)
    for _ in range(30):
        p={i:{j for j in range(i) if rng.random()<.3} for i in range(7)}
        ancestors={}
        for i in p:
            ancestors[i]=set(p[i])
            for j in p[i]:
                ancestors[i].update(ancestors[j])
        expected=max(len(s) for n in range(8) for s in itertools.combinations(p,n)
                     if all(a not in ancestors[b] and b not in ancestors[a] for a,b in itertools.combinations(s,2)))
        assert dag_metrics(p)[0]["width_max_antichain"]==expected


def test_parallel_refine_and_peer_metrics(tmp_path):
    parallel=analyze_events(record(tmp_path/"p",exp(width=4)).trace.events)
    refine=analyze_events(record(tmp_path/"r",exp("EvaluateRefine",width=1,max_revisions=2)).trace.events)
    ring=analyze_events(record(tmp_path/"ring",exp("PeerDeliberation",width=4,rounds=1,connectivity="ring")).trace.events)
    dense=analyze_events(record(tmp_path/"dense",exp("PeerDeliberation",width=4,rounds=1,connectivity="all_to_all")).trace.events)
    assert parallel["structure"]["width_max_antichain"]==4
    assert parallel["structure"]["depth_nodes"]==2
    assert refine["structure"]["depth_nodes"]==7  # six LLM calls + terminal artifact pack
    assert refine["structure"]["width_max_antichain"]==1
    assert ring["structure"]["edges"]==dense["structure"]["edges"]
    assert ring["structure"]["data_edges"]<dense["structure"]["data_edges"]
    assert ring["structure"]["artifact_consumed_bytes"]<dense["structure"]["artifact_consumed_bytes"]
    assert dense["runtime"]["kv_live"]["source"]=="unavailable"
    assert all(row["ready_waiting"]>=0 for row in dense["runtime"]["timeline"])


def test_replay_global_slots_ready_before_admission(tmp_path):
    source=record(tmp_path/"source",exp(width=3))
    lock=threading.Lock()
    active=peak=0
    class Slow(MockLLM):
        def invoke(self,*args):
            nonlocal active,peak
            with lock:
                active+=1;peak=max(peak,active)
            try:
                time.sleep(.025)
                return super().invoke(*args)
            finally:
                with lock: active-=1
    slots=threading.BoundedSemaphore(1)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(2) as pool:
        results=list(pool.map(lambda i:replay_trace(source.trace.trace_path,DeploymentSpec(concurrency=1),
            trace_dir=tmp_path/str(i),llm_slots=slots,backend=Slow(),export_views=False),range(2)))
    assert peak==1
    for _,trace in results:
        report=compare_replay(source.trace.events,trace.events)
        assert report["fixed_workload_invariants_pass"]
        assert report["runtime"]["client_admission_wait_sec"]["max"]>.015
        assert report["capabilities"]["output_token_sequence"]=="uncontrolled"


def test_open_loop_overflow_and_slo_no_omission():
    def work(i):
        time.sleep(.04)
        return {"status":"completed"}
    summary,rows=run_open_loop(work,count=6,rate=1000,arrival="constant",seed=1,max_inflight=1,slo_sec=.001)
    assert len(rows)==summary["offered"]==6
    assert summary["client_overflow"]>=1
    assert summary["slo_success_fraction"]==0
    assert summary["goodput_qps"]==0
    assert len({r["index"] for r in rows})==6
    assert [r["scheduled_sec"] for r in rows]==arrival_offsets(6,1000,"constant",1)


def test_failures_are_journaled():
    def fail(i): raise RuntimeError("broken backend")
    journal=[]
    summary,rows=run_open_loop(fail,count=2,rate=100,arrival="poisson",seed=2,max_inflight=2,slo_sec=1,on_record=journal.append)
    assert len(journal)==2
    assert summary["failed_or_rejected"]==2
    assert summary["completed_e2e_sec"]["p95"] is None


def test_study_run_replay_resume_and_input_fingerprint(tmp_path):
    from dataclasses import asdict
    (tmp_path/"exp.json").write_text(json.dumps(asdict(exp(width=2))))
    (tmp_path/"deployment.json").write_text(json.dumps(asdict(DeploymentSpec(concurrency=2))))
    config={"mode":"run","experiments":["exp.json"],"deployments":{"mock":"deployment.json"},
            "rates":[50],"count":3,"max_inflight_workflows":3,"slo_sec":10}
    path=tmp_path/"study.json"
    path.write_text(json.dumps(config))
    result=run_study(path,tmp_path/"run")
    assert result[0]["completed"]==3
    assert not result[0]["analysis_errors"]
    assert result[0]["internal_requests"]==9
    assert result[0]["max_llm_inflight"] >= 1
    from pathlib import Path
    dynamics=json.loads(Path(result[0]["case_trace_dynamics_path"]).read_text())
    assert dynamics["complete"]
    assert dynamics["timeline"][-1]["llm_inflight"]==0
    assert dynamics["timeline"][-1]["ready_waiting"]==0
    assert run_study(path,tmp_path/"run",resume=True)==result
    config["mode"]="replay"
    config["traces"]=[str(p) for p in (tmp_path/"run").rglob("*.jsonl") if p.name!="arrivals.jsonl"]
    config.pop("experiments")
    path.write_text(json.dumps(config))
    assert run_study(path,tmp_path/"replay")[0]["completed"]==3
    with pytest.raises(ValueError,match="fingerprint"):
        run_study(path,tmp_path/"run",resume=True)


def test_generate_explicit_revision_policy(tmp_path):
    rows=generate_space(tmp_path/"space",widths=[2],rounds=[0],revisions=[2],record=True)
    refine=next(r for r in rows if r["id"]=="refine_r2")
    assert refine["metrics"]["nodes"]==7
    assert refine["provenance"]=="generated_space"


@pytest.mark.parametrize("rate",[0,-1,float("nan"),float("inf")])
def test_invalid_rate(rate):
    with pytest.raises(ValueError):arrival_offsets(2,rate,"constant",1)


def test_trace_journal_preserves_partial_then_atomic_final(tmp_path):
    runner=build_experiment(exp(width=1),trace_dir=tmp_path)
    trace=runner.trace
    trace.journal_enabled=True
    trace.emit(event_type="pilot_manifest",label="interruption checkpoint")
    trace.close_journal()
    partial=trace.trace_path.with_suffix(".partial")
    assert json.loads(partial.read_text().splitlines()[0])["label"]=="interruption checkpoint"
    runner.config.export_trace_views=False
    assert runner.run()["status"]=="completed"
    assert not partial.exists()
    assert trace.trace_path.exists()
    assert not trace.trace_path.with_suffix(".jsonl.tmp").exists()


def test_tool_snapshot_does_not_occupy_llm_capacity(tmp_path):
    from app.specs import RoleTask
    base=exp(width=1)
    base=replace(base,task=TaskBinding("Search",{"Worker":RoleTask(tools=("search",))}),
                 structure=replace(base.structure,workflow=WorkflowSpec((StageSpec("a","m"),StageSpec("b","m")))))
    source=record(tmp_path/"source",base)
    events=source.trace.events
    tools=[e for e in events if e.get("canonical_type")=="operation_finish"]
    tools[0]["duration_sec"]=.12
    tools[1]["duration_sec"]=0
    path=tmp_path/"source.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events))
    _,trace=replay_trace(path,DeploymentSpec(concurrency=1),trace_dir=tmp_path/"replay",export_views=False)
    slow_end=next(e["relative_time_sec"] for e in trace.events if e.get("source_operation_id")==tools[0]["operation_id"] and e.get("canonical_type")=="operation_finish")
    assert any(e["relative_time_sec"]<slow_end for e in trace.events if e.get("canonical_type")=="request_finish")


def test_study_http_endpoint_measurements(tmp_path):
    from dataclasses import asdict
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            self.send_response(200);self.end_headers()
            self.wfile.write(b'test_running_requests{model="test"} 2\n')
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type","text/event-stream");self.end_headers()
            chunk={"id":"response","choices":[{"delta":{"content":"A proposal"},"finish_reason":"stop"}],
                   "usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}
            self.wfile.write(("data: "+json.dumps(chunk)+"\n\ndata: [DONE]\n\n").encode())
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        dep=DeploymentSpec(backend="openai_compatible",model="test",endpoint=f"http://127.0.0.1:{server.server_port}/v1")
        (tmp_path/"dep.json").write_text(json.dumps(asdict(dep)))
        (tmp_path/"exp.json").write_text(json.dumps(asdict(exp(width=1))))
        plan={"mode":"run","experiments":["exp.json"],"deployments":{"http":"dep.json"},"rates":[20],
              "count":2,"collect_backend_metrics":True,"backend_metrics_interval_sec":.05}
        (tmp_path/"study.json").write_text(json.dumps(plan))
        result=run_study(tmp_path/"study.json",tmp_path/"out")[0]
        assert result["completed"]==2 and not result["analysis_errors"]
        files=list((tmp_path/"out").rglob("backend_observations.jsonl"))
        rows=[json.loads(line) for line in files[0].read_text().splitlines()]
        import jsonschema
        from pathlib import Path
        schema=json.loads((Path(__file__).parents[1]/"configs/canonical_trace.schema.json").read_text())
        for row in rows:
            jsonschema.validate(row,schema)
        values=rows[0]["attributes"]["serving_observation"]
        assert values['test_running_requests{model="test"}']=={"value":2.0,"source":"backend-reported"}
    finally:
        server.shutdown();server.server_close();thread.join()

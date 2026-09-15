"""Reproducible open-loop run/replay sweeps for MASBench Sections 4.3--4.5."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import subprocess
import threading
import time
from uuid import uuid4

from .benchmark import build_experiment
from .execution_graph import ExecutionGraph
from .replay import replay_trace, prepare_replay
from .specs import DeploymentSpec, load_experiment, read_config
from .workload_analysis import analyze_events, distribution, read_events


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)


def arrival_offsets(count, rate, policy, seed):
    if type(count) is not int or count < 1 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("count and rate must be positive and finite")
    if policy not in {"constant", "poisson"}:
        raise ValueError("arrival must be constant or poisson")
    rng, offsets, t = random.Random(seed), [], 0.0
    for _ in range(count):
        offsets.append(t)
        t += 1/rate if policy == "constant" else rng.expovariate(rate)
    return offsets


def run_open_loop(execute, *, count, rate, arrival, seed, max_inflight, slo_sec,
                  on_record=lambda row: None):
    """Never block the arrival clock on completion; explicitly reject client overflow."""
    if type(max_inflight) is not int or max_inflight < 1 or not math.isfinite(slo_sec) or slo_sec <= 0:
        raise ValueError("max_inflight and SLO must be positive")
    offsets = arrival_offsets(count, rate, arrival, seed)
    cpu_started = time.process_time()
    sampled_peak_threads = threading.active_count()
    start = time.perf_counter()
    clock_origin_unix = time.time()
    records, active, lock = [], [], threading.Lock()
    def save(row):
        with lock:
            records.append(row)
            on_record(row)
    def job(index, offset):
        began = time.perf_counter()-start
        row = {"index":index, "scheduled_sec":offset, "started_sec":began,
               "launch_lag_sec":max(0,began-offset)}
        try:
            row.update(execute(index))
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        driver_ended = time.perf_counter()-start
        ended = row.pop("workload_finished_perf", start+driver_ended)-start
        if "trace_origin_perf" in row:
            row["trace_origin_sec"] = row.pop("trace_origin_perf")-start
        row.update(finished_sec=ended, e2e_from_arrival_sec=ended-offset,
                   execution_wall_sec=ended-began, driver_finished_sec=driver_ended,
                   finalization_sec=max(0,driver_ended-ended))
        save(row)
    with ThreadPoolExecutor(max_workers=max_inflight) as pool:
        for i, offset in enumerate(offsets):
            sampled_peak_threads = max(sampled_peak_threads,threading.active_count())
            remaining = start+offset-time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            # Resolve completed futures: infrastructure errors must not disappear.
            still_active = []
            for f in active:
                if f.done():
                    f.result()
                else:
                    still_active.append(f)
            active = still_active
            if len(active) >= max_inflight:
                save({"index":i,"scheduled_sec":offset,"status":"client_overflow",
                      "finished_sec":time.perf_counter()-start})
            else:
                active.append(pool.submit(job,i,offset))
        for f in active:
            f.result()
    elapsed = time.perf_counter()-start
    good = [r for r in records if r.get("status") in {"completed","accepted"}]
    eligible = [r for r in good if r["e2e_from_arrival_sec"] <= slo_sec]
    window = max(count/rate, elapsed)
    summary = {"client_cpu_sec":time.process_time()-cpu_started,
               "client_sampled_peak_threads":sampled_peak_threads,
               "finalization_sec":distribution([r["finalization_sec"] for r in records if "finalization_sec" in r]),
               "clock_origin_unix":clock_origin_unix, "offered":count, "completed":len(good), "failed_or_rejected":count-len(good),
               "client_overflow":sum(r["status"] == "client_overflow" for r in records),
               "target_user_qps":rate, "arrival_span_sec":offsets[-1], "drain_elapsed_sec":elapsed,
               "accounting_window_sec":window, "completed_workflow_qps":len(good)/window,
               "slo_sec":slo_sec,"goodput_qps":len(eligible)/window,"slo_success_fraction":len(eligible)/count,
               "completed_e2e_sec":distribution([r["e2e_from_arrival_sec"] for r in good]),
               "launch_lag_sec":distribution([r["launch_lag_sec"] for r in records if "launch_lag_sec" in r]),
               "status_counts":{s:sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})},
               "interpretation":"Finite arrival cohort plus drain; not a steady-state capacity estimate. Failed/rejected arrivals count against SLO success."}
    return summary, sorted(records,key=lambda r:r["index"])


def source_metadata():
    def git(*args):
        try:
            return subprocess.check_output(["git",*args], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            return "unavailable"
    files = sorted(Path(__file__).parent.rglob("*.py"))
    return {"git_head":git("rev-parse","HEAD"), "git_status":git("status","--short"),
            "python":platform.python_version(), "platform":platform.platform(),
            "app_source_sha256":digest({str(p.relative_to(Path(__file__).parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})}


def run_study(config_path, output, *, resume=False):
    base = Path(config_path).resolve().parent
    config = read_config(config_path)
    allowed = {"mode","experiments","traces","trace_corpus","deployments","rates","count","repetitions","seed",
               "arrival","max_inflight_workflows","slo_sec","warmup_count","cache_protocol",
               "max_replay_operations","strict_replay","output_length_tolerance","require_identity_match","collect_backend_metrics","backend_metrics_interval_sec",
               "quality_evaluator"}
    if set(config)-allowed:
        raise ValueError(f"Unknown study fields: {sorted(set(config)-allowed)}")
    mode = config.get("mode","run")
    if mode not in {"run","replay"}:
        raise ValueError("mode must be run or replay")
    if config.get("strict_replay",False):
        raise ValueError("Current adapters only support best-effort replay")
    if not 0 <= config.get('output_length_tolerance',0) <= 1:
        raise ValueError('output_length_tolerance must be in [0,1]')
    protocol = config.get("cache_protocol","uncontrolled")
    if protocol not in {"uncontrolled","warm_sequence","cache_disabled","warm_cache_enabled"}:
        raise ValueError("Cold/isolated cache requires backend reset integration; do not claim it from a label")
    warmup = config.get("warmup_count",0)
    if type(warmup) is not int or warmup < 0 or (protocol in {"warm_sequence","warm_cache_enabled"} and not warmup):
        raise ValueError("warm_sequence requires warmup_count > 0")
    for field, default in (("max_inflight_workflows",64),("max_replay_operations",4096)):
        if type(config.get(field,default)) is not int or config.get(field,default)<1:
            raise ValueError(field+" must be a positive integer")
    for field, default in (("slo_sec",60),("backend_metrics_interval_sec",.5)):
        if not math.isfinite(config.get(field,default)) or config.get(field,default)<=0:
            raise ValueError(field+" must be positive and finite")
    reps = config.get("repetitions",1)
    if type(reps) is not int or reps < 1:
        raise ValueError("repetitions must be positive")
    rates = config["rates"]
    if not rates or len(set(rates)) != len(rates):
        raise ValueError("rates must be a nonempty unique list")
    for rate in rates:
        arrival_offsets(config.get("count",20),rate,config.get("arrival","constant"),config.get("seed",42))
    experiments = [load_experiment(base/p) for p in config.get("experiments",[])]
    trace_paths = [(base/p).resolve() for p in config.get("traces",[])]
    if config.get("trace_corpus"):
        if trace_paths:
            raise ValueError("Use traces or trace_corpus, not both")
        corpus_path=(base/config["trace_corpus"]).resolve()
        corpus=json.loads(corpus_path.read_text())
        if not corpus.get("runs") or any("trace" not in row for row in corpus["runs"]):
            raise ValueError("trace_corpus requires recorded traces for every row")
        trace_paths=[(corpus_path.parent/row["trace"]).resolve() for row in corpus["runs"]]
    trace_hashes, prepared_traces = {}, {}
    for path in trace_paths:
        prepared_traces[path] = prepare_replay(path,max_operations=config.get("max_replay_operations",4096))
        trace_hashes[str(path)] = prepared_traces[path].sha256
    if (mode == "run" and not experiments) or (mode == "replay" and not trace_paths):
        raise ValueError("Provide experiments for run or traces for replay")
    deployments = [(name, DeploymentSpec(**read_config(base/p))) for name,p in config["deployments"].items()]
    if not deployments:
        raise ValueError("No deployments")
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    metadata = source_metadata()
    resolved = {"config":config,"experiments":[asdict(e) for e in experiments],
                "deployments":{n:asdict(d) for n,d in deployments},"trace_hashes":trace_hashes,
                "app_source_sha256":metadata["app_source_sha256"]}
    fingerprint = digest(resolved)
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        if not resume or json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Output exists or resume fingerprint differs; use a fresh output directory")
    else:
        atomic_json(manifest_path,{"schema":"masbench_study_v1","fingerprint":fingerprint,
                                  "resolved":resolved,"environment":metadata})
    # Self-contained inputs and exact Python implementation, including dirty edits.
    # Resume still verifies the original inputs and code fingerprint above.
    import shutil
    snapshot = output/'source_snapshot'
    if not snapshot.exists():
        shutil.copytree(Path(__file__).parent,snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    corpus_entries=[]
    for path in trace_paths:
        target=output/'source_corpus'/(trace_hashes[str(path)]+'.jsonl')
        target.parent.mkdir(exist_ok=True)
        if not target.exists(): shutil.copyfile(path,target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=trace_hashes[str(path)]:
            raise ValueError('Archived source trace hash mismatch')
        corpus_entries.append({'original':str(path),'trace':str(target.relative_to(output)), 'sha256':trace_hashes[str(path)]})
    atomic_json(output/'source_corpus.json',{'runs':corpus_entries})
    cases = [(name,dep,rate,rep) for name,dep in deployments for rate in rates for rep in range(reps)]
    random.Random(config.get("seed",42)).shuffle(cases)
    summaries=[]
    for case_index,(name,deployment,rate,rep) in enumerate(cases):
        key = digest([name,rate,rep])[:16]
        case_dir = output/key
        case_dir.mkdir(exist_ok=True)
        if resume and (case_dir/"summary.json").exists():
            summaries.append(json.loads((case_dir/"summary.json").read_text()))
            continue
        # Incomplete attempts get a new directory, preserving all prior partial traces.
        attempt = case_dir/("attempt_"+uuid4().hex[:12])
        attempt.mkdir()
        from .replay_protocol import verify_cache_protocol
        cache_evidence=verify_cache_protocol(deployment,protocol)
        semaphore = threading.BoundedSemaphore(deployment.concurrency)
        def execute(index, *, warm=False):
            root = attempt/("warmup" if warm else "traces")
            if mode == "run":
                exp = replace(experiments[index%len(experiments)],deployment=deployment,load=1)
                runner = build_experiment(exp,trace_dir=root)
                runner.config.export_trace_views = False
                runner._llm_slots = semaphore
                runner.trace.journal_enabled = True
                try:
                    summary = runner.run()
                finally:
                    runner.trace.close_journal()
                trace = runner.trace
            else:
                summary,trace = replay_trace(trace_paths[index%len(trace_paths)],deployment,trace_dir=root,
                                            llm_slots=semaphore,export_views=False,journal=True,
                                            max_operations=config.get("max_replay_operations",4096),
                                            prepared=prepared_traces[trace_paths[index%len(trace_paths)]])
            terminal = next(e for e in reversed(trace.events) if e.get("canonical_type") == "run_finish")
            return {"status":summary["status"],"trace_path":str(trace.trace_path),
                    "trace_origin_perf":trace.start_perf,
                    "workload_finished_perf":trace.start_perf+terminal["relative_time_sec"],
                    "source_index":index%(len(experiments) if mode == "run" else len(trace_paths))}
        for index in range(warmup):
            row = execute(index,warm=True)
            if row["status"] not in {"completed","accepted"}:
                raise RuntimeError("Warmup failed: "+str(row))
        journal = (attempt/"arrivals.jsonl").open("a",encoding="utf-8")
        # A single endpoint-wide sampler avoids one redundant polling loop per workflow.
        sampler = None
        if config.get("collect_backend_metrics") and deployment.backend != "mock":
            from .backend_metrics import BackendMetricsSampler, metrics_url_from_base_url
            from .backend_adapters import build_backend_trace_adapter
            sampler = BackendMetricsSampler(url=deployment.telemetry.get("metrics_url") or metrics_url_from_base_url(deployment.endpoint),
                output_path=attempt/"backend_metrics.json", interval_sec=config.get("backend_metrics_interval_sec",.5),
                adapter=build_backend_trace_adapter(deployment.telemetry.get("adapter","generic"),deployment.telemetry))
            sampler.start()
        try:
            schedule_seed=config.get("seed",42)+rep
            atomic_json(attempt/'arrival_schedule.json',{"policy":config.get("arrival","constant"),"rate":rate,
                "seed":schedule_seed,"offsets_sec":arrival_offsets(config.get("count",20),rate,config.get("arrival","constant"),schedule_seed)})
            def append(row):
                journal.write(json.dumps(row)+"\n")
                journal.flush()
            summary,records = run_open_loop(execute,count=config.get("count",20),rate=rate,
                arrival=config.get("arrival","constant"),seed=schedule_seed,
                max_inflight=config.get("max_inflight_workflows",64),slo_sec=config.get("slo_sec",60),on_record=append)
        finally:
            journal.close()
            if sampler:
                sampler.stop()
                # Canonical endpoint observations, separate from per-workflow DAGs.
                # Preserve metric labels; never infer units or device attribution.
                with (attempt/"backend_observations.jsonl").open("w") as stream:
                    for sample in sampler.samples:
                        event={"canonical_schema":"masbench_execution_v1", "canonical_type":"backend_measurement",
                               "event_type":"backend_measurement", "event_id":"measurement_"+uuid4().hex,
                               "run_id":"study_"+key,"scope":"endpoint", "timestamp_unix":sample["timestamp_unix"],
                               "timestamp":datetime.fromtimestamp(sample["timestamp_unix"],timezone.utc).isoformat(),
                               **{k:"" for k in ("operation_id","request_id","stage_instance_id","motif_instance_id","agent_instance_id","artifact_id")},
                               "relative_time_sec":sample["relative_time_sec"], "status":sample["status"],
                               "error":sample.get("error"), "attributes":{"operation":{},"information_flow":{},
                               "serving_observation":{k:{"value":v if math.isfinite(v) else None,
                                                          "source":"backend-reported" if math.isfinite(v) else "unavailable"}
                                   for k,v in sample.get("metrics",{}).items()}}}
                        from .backend_adapters import canonical_telemetry
                        event["attributes"]["serving_observation"].update(canonical_telemetry(sample))
                        event["telemetry_metadata"]=deployment.telemetry
                        stream.write(json.dumps(event,allow_nan=False)+"\n")
        # Analysis is deliberately outside the measured load window.
        analyses,invalid,submit_times = [],[],[]
        quality_evaluations=[]
        replay_checks=[]
        dynamics_delta=defaultdict(Counter)
        envelope_complete=True
        for row in records:
            if "trace_path" not in row:
                continue
            try:
                events = read_events(row["trace_path"])
                submit_times.extend(row["trace_origin_sec"]+e["relative_time_sec"] for e in events if e.get("canonical_type") == "request_submit")
                analysis = analyze_events(events)
                if config.get('quality_evaluator'):
                    from .quality import evaluate,final_output
                    graph=ExecutionGraph.from_events(events,replay=True)
                    quality={"run_id":analysis["run_id"],**evaluate(final_output(graph),config['quality_evaluator'])}
                    quality_evaluations.append(quality)
                if mode=="replay":
                    from .replay_compare import compare_replay
                    comparison=compare_replay(prepared_traces[trace_paths[row["source_index"]]].events,events,
                                              output_length_tolerance=config.get("output_length_tolerance",0))
                    check={"run_id":analysis["run_id"],"invariants":comparison["fixed_workload_invariants_pass"],
                           "length":comparison["output_length_agreement"],"identity":comparison["identity_agreement"]}
                    check["eligible"]=check["invariants"] and check["length"]["pass"] and (not config.get("require_identity_match") or check["identity"]["status"]=="matched")
                    replay_checks.append(check)
                atomic_json(attempt/"analysis"/(analysis["run_id"]+".json"),analysis)
                previous=Counter()
                for point in analysis["runtime"]["timeline"]:
                    at=row["trace_origin_sec"]+point["time_sec"]
                    for metric,value in point.items():
                        if metric=="time_sec":
                            continue
                        if value is None:
                            envelope_complete=False
                            continue
                        dynamics_delta[at][metric]+=value-previous[metric]
                        previous[metric]=value
                analyses.append({"arrival_index":row["index"],"run_id":analysis["run_id"],"source_index":row["source_index"],"structure":analysis["structure"]})
            except Exception as exc:
                invalid.append({"trace":row["trace_path"],"error":str(exc)})
        state,combined=Counter(),[]
        for at,delta in sorted(dynamics_delta.items()):
            state.update(delta)
            combined.append({"time_sec":at,**dict(state)})
        if not envelope_complete:
            for row in combined:
                row["inflight_token_envelope"]=None
        dynamic_path=attempt/"case_trace_dynamics.json"
        atomic_json(dynamic_path,{"clock_origin_unix":summary["clock_origin_unix"],
                                  "complete":not invalid and all("trace_path" in r or r["status"]=="client_overflow" for r in records),
                                  "scope":"sum of analyzable workflow traces; endpoint resident KV remains unavailable",
                                  "timeline":combined})
        from .pressure import queue_slope
        queue_growth=queue_slope(combined,summary["arrival_span_sec"])
        atomic_json(attempt/'quality.json',{'scope':'out_of_band','evaluations':quality_evaluations})
        summary.update(queue_growth_client_per_sec=queue_growth,cache_protocol_evidence=cache_evidence,
                       quality_evaluations=quality_evaluations,
                       quality_pass_fraction=(sum(q['pass'] for q in quality_evaluations)/len(quality_evaluations) if quality_evaluations else None),
                       quality_score_mean=(sum(q['score'] for q in quality_evaluations)/len(quality_evaluations) if quality_evaluations else None),
                       replay_checks=replay_checks,replay_equivalence_failures=sum(not c["eligible"] for c in replay_checks),
                       case_trace_dynamics_path=str(dynamic_path),
                       max_client_ready_waiting=max((r.get("ready_waiting",0) for r in combined),default=0),
                       max_llm_inflight=max((r.get("llm_inflight",0) for r in combined),default=0),
                       case_id=key,deployment=name,rate=rate,repetition=rep,mode=mode,
                       cache_protocol=protocol,cache_reset_performed=False,warmup_count=warmup,
                       replay_mode="best-effort" if mode=="replay" else None,
                       internal_request_count_complete=all("trace_path" in r or r["status"]=="client_overflow" for r in records),
                       internal_requests=len(submit_times),internal_qps=len(submit_times)/summary["accounting_window_sec"],
                       submit_times_sec=sorted(submit_times),analysis_errors=invalid,
                       attempt_path=str(attempt),mock=deployment.backend=="mock")
        atomic_json(attempt/"corpus.json",analyses)
        atomic_json(case_dir/"summary.json",summary)
        summaries.append(summary)
        atomic_json(output/"results.json",summaries)
        print(json.dumps({"case":case_index+1,"total":len(cases),"deployment":name,"rate":rate,
                          "completed":summary["completed"],"offered":summary["offered"]}),flush=True)
    atomic_json(output/"results.json",summaries)
    return summaries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    results=run_study(args.config,args.output,resume=args.resume)
    return int(any(r["analysis_errors"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())

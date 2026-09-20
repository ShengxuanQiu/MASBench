"""Deterministic heterogeneous workload multiplexing on one shared backend.

This is an experiment-layer driver.  It reuses the canonical run/replay paths and
only adds a saved class/arrival schedule plus per-class accounting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import threading
from uuid import uuid4

from .benchmark import build_experiment
from .backend_metrics import BackendMetricsSampler, metrics_url_from_base_url, summarize_backend_metrics
from .replay import prepare_replay, replay_trace
from .replay_compare import compare_replay
from .replay_protocol import verify_cache_protocol
from .specs import DeploymentSpec, load_experiment, read_config
from .study import arrival_offsets, atomic_json, digest, run_open_loop, source_metadata
from .workload_analysis import analyze_events, distribution, read_events


def weighted_class_schedule(count, weights, seed):
    """Return an exact-count, seeded schedule using largest-remainder allocation."""
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if not weights or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("mix weights must be finite, non-negative, and nonempty")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("at least one mix weight must be positive")
    exact = {name: count * value / total for name, value in weights.items()}
    counts = {name: int(math.floor(value)) for name, value in exact.items()}
    remainder = count - sum(counts.values())
    order = sorted(weights, key=lambda name: (-(exact[name] - counts[name]), name))
    for name in order[:remainder]:
        counts[name] += 1
    labels = [name for name in sorted(counts) for _ in range(counts[name])]
    random.Random(seed).shuffle(labels)
    return labels, counts


def _trace_totals(events, analysis):
    finishes = [e for e in events if e.get("canonical_type") == "request_finish"]
    prompt = [e.get("backend_prompt_tokens") for e in finishes]
    output = [e.get("backend_completion_tokens") for e in finishes]
    pressure = analysis["pressure"]
    info = analysis["information_flow"]
    return {
        "internal_requests": len(finishes),
        "prompt_tokens": sum(x for x in prompt if x is not None) if all(x is not None for x in prompt) else None,
        "output_tokens": sum(x for x in output if x is not None) if all(x is not None for x in output) else None,
        "ready_wait_seconds": pressure["temporal_pressure"]["ready_wait_seconds"],
        "peak_ready_waiting": pressure["temporal_pressure"]["peak_ready_waiting"],
        "barrier_wait_seconds": pressure["synchronization_exposure_sec"],
        "logical_state_peak_bytes": pressure["logical_state_residency"]["peak_artifact_bytes"],
        "logical_state_byte_seconds": pressure["logical_state_residency"]["byte_seconds"],
        "delivered_bytes": info["delivered_bytes"],
        "delivered_tokens_est": info["delivered_tokens_est"],
        "context_information_amplification_bytes": info["context_information_amplification_bytes"],
    }


def _sum_complete(rows, key):
    values = [row[key] for row in rows]
    return sum(values) if values and all(v is not None for v in values) else None


def _endpoint_observations(samples):
    values = defaultdict(list)
    sources = {"device_metrics": "observed", "serving_metrics": "backend-reported",
               "cache_memory_metrics": "backend-reported"}
    def visit(prefix, value):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(prefix + "." + key, child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values[prefix].append(value)
    for sample in samples:
        for scope in sources:
            visit(scope, sample.get(scope, {}))
    return {key: {"distribution": distribution(series), "source": sources[key.split(".", 1)[0]]}
            for key, series in sorted(values.items())}


def _class_summary(name, scheduled, records, accounting_window, default_slo):
    rows = [row for row in records if row.get("workload_class") == name]
    completed = [row for row in rows if row.get("status") in {"completed", "accepted"}]
    slo = scheduled.get("slo_sec", default_slo)
    e2e = [row["e2e_from_arrival_sec"] for row in completed]
    metrics = [row["trace_metrics"] for row in completed if "trace_metrics" in row]
    offered = len(rows)
    eligible = sum(value <= slo for value in e2e)
    internal = sum(m["internal_requests"] for m in metrics)
    return {
        "workload_class": name,
        "offered": offered,
        "completed": len(completed),
        "failed_or_rejected": offered - len(completed),
        "slo_sec": slo,
        "slo_success_fraction": eligible / offered if offered else None,
        "goodput_qps": eligible / accounting_window,
        "completed_workflow_qps": len(completed) / accounting_window,
        "completed_e2e_sec": distribution(e2e),
        "internal_requests": internal,
        "internal_qps": internal / accounting_window,
        "prompt_tokens": _sum_complete(metrics, "prompt_tokens"),
        "output_tokens": _sum_complete(metrics, "output_tokens"),
        "ready_wait_seconds": sum(m["ready_wait_seconds"] for m in metrics),
        "barrier_wait_seconds": sum(m["barrier_wait_seconds"] for m in metrics),
        "logical_state_byte_seconds": sum(m["logical_state_byte_seconds"] for m in metrics),
        "logical_state_peak_bytes": max((m["logical_state_peak_bytes"] for m in metrics), default=None),
        "delivered_bytes": sum(m["delivered_bytes"] for m in metrics),
        "delivered_tokens_est": sum(m["delivered_tokens_est"] for m in metrics),
        "context_information_amplification_bytes": distribution([
            m["context_information_amplification_bytes"] for m in metrics
            if m["context_information_amplification_bytes"] is not None]),
        "resource_attribution": "logical request/token/information/state demand by class; shared physical device samples remain endpoint-wide",
    }


def mixed_capacity_summary(rows, min_slo_success=.95, max_failure_fraction=.01):
    """SLO boundary by deployment and composition, with every class protected."""
    from .capacity import aggregate
    groups = defaultdict(list)
    for row in rows:
        if not row["mix"].startswith("isolated__"):
            groups[(row["deployment"], row["mix"], row["rate"])].append(row)
    by_mix = defaultdict(list)
    for (deployment, mix, rate), repetitions in sorted(groups.items()):
        valid = all(not row["client_overflow"] and not row["analysis_errors"] and
                    not row["replay_equivalence_failures"] for row in repetitions)
        failures = [row["failed_or_rejected"] / row["offered"] for row in repetitions]
        all_class_slo = [min(metrics["slo_success_fraction"] for metrics in row["per_class"].values())
                         for row in repetitions]
        passed = valid and min(all_class_slo) >= min_slo_success and max(failures) <= max_failure_fraction
        point = {"rate": rate, "valid": valid, "slo_pass": passed,
                 "repetitions": len(repetitions),
                 "metrics": {
                    "aggregate_slo_success_fraction": aggregate([row["slo_success_fraction"] for row in repetitions]),
                    "worst_class_slo_success_fraction": aggregate(all_class_slo),
                    "goodput_qps": aggregate([row["goodput_qps"] for row in repetitions]),
                    "internal_qps": aggregate([row["internal_qps"] for row in repetitions]),
                    "failure_fraction": aggregate(failures),
                    "p95_e2e_sec": aggregate([row["completed_e2e_sec"]["p95"] for row in repetitions
                                               if row["completed_e2e_sec"]["p95"] is not None])}}
        by_mix[(deployment, mix)].append(point)
    output = {}
    for (deployment, mix), points in by_mix.items():
        failed = next((point for point in points if point["valid"] and not point["slo_pass"]), None)
        prior = [point["rate"] for point in points if point["slo_pass"] and
                 (failed is None or point["rate"] < failed["rate"])]
        bracket = [max(prior), failed["rate"]] if failed and prior else None
        invalid = any(not point["valid"] for point in points)
        nonmonotonic = bool(failed and any(point["slo_pass"] and point["rate"] > failed["rate"] for point in points))
        knee = bracket[1] if bracket and not invalid and not nonmonotonic else None
        for point in points:
            point["rho"] = point["rate"] / knee if knee else None
        output.setdefault(deployment, {})[mix] = {"lambda_knee": knee, "bracket": bracket,
            "status": "invalid" if invalid else "nonmonotonic" if nonmonotonic else
                      "bracketed" if bracket else "right_censored" if failed is None else "left_censored",
            "points": points,
            "definition": "First tested total user-QPS point where aggregate error and every workflow class fail the configured SLO criterion."}
    return output


def _validate_config(config):
    allowed = {"schema", "mode", "classes", "mixes", "deployments", "rates", "count", "repetitions",
               "seed", "arrival", "max_inflight_workflows", "slo_sec", "warmup_count", "cache_protocol",
               "collect_backend_metrics", "backend_metrics_interval_sec", "max_replay_operations",
               "output_length_tolerance", "require_identity_match", "isolated_baselines"}
    allowed.update({"min_slo_success", "max_failure_fraction"})
    if set(config) - allowed:
        raise ValueError("Unknown mixed-study fields: " + str(sorted(set(config) - allowed)))
    mode = config.get("mode", "replay")
    if mode not in {"run", "replay"}:
        raise ValueError("mode must be run or replay")
    classes = config.get("classes", {})
    if not classes:
        raise ValueError("classes must be nonempty")
    for name, item in classes.items():
        if set(item) - {"experiment", "traces", "trace_corpus", "slo_sec", "description"}:
            raise ValueError(f"Unknown fields for class {name}")
        expected = "experiment" if mode == "run" else ("traces" if "traces" in item else "trace_corpus")
        if expected not in item:
            raise ValueError(f"Class {name} has no {mode} source")
    if not config.get("mixes"):
        raise ValueError("mixes must be nonempty")
    for mix, weights in config["mixes"].items():
        unknown = set(weights) - set(classes)
        if unknown:
            raise ValueError(f"Mix {mix} references unknown classes: {sorted(unknown)}")
        weighted_class_schedule(config.get("count", 20), weights, config.get("seed", 42))
    for rate in config.get("rates", []):
        arrival_offsets(config.get("count", 20), rate, config.get("arrival", "constant"), config.get("seed", 42))
    if not config.get("rates") or not config.get("deployments"):
        raise ValueError("rates and deployments must be nonempty")
    for field, default in (("min_slo_success", .95), ("max_failure_fraction", .01)):
        if not 0 <= config.get(field, default) <= 1:
            raise ValueError(field + " must be in [0,1]")


def _resolve_sources(base, config):
    mode = config.get("mode", "replay")
    sources = {}
    for name, item in config["classes"].items():
        if mode == "run":
            sources[name] = {"experiment": load_experiment((base / item["experiment"]).resolve())}
            continue
        paths = [(base / p).resolve() for p in item.get("traces", [])]
        if item.get("trace_corpus"):
            corpus_path = (base / item["trace_corpus"]).resolve()
            corpus = json.loads(corpus_path.read_text())
            paths = [(corpus_path.parent / row["trace"]).resolve() for row in corpus.get("runs", [])]
        if not paths:
            raise ValueError(f"Class {name} has an empty trace corpus")
        prepared = [prepare_replay(p, max_operations=config.get("max_replay_operations", 4096)) for p in paths]
        sources[name] = {"paths": paths, "prepared": prepared}
    return sources


def _add_baseline_mixes(config):
    mixes = dict(config["mixes"])
    baseline = config.get("isolated_baselines")
    if not baseline:
        return mixes, []
    names = list(config["classes"]) if baseline is True else list(baseline)
    for name in names:
        if name not in config["classes"]:
            raise ValueError("Unknown isolated baseline class: " + name)
        mixes["isolated__" + name] = {name: 1}
    return mixes, names


def run_mixed_study(config_path, output, *, resume=False):
    config_path = Path(config_path).resolve()
    base = config_path.parent
    config = read_config(config_path)
    _validate_config(config)
    mode = config.get("mode", "replay")
    sources = _resolve_sources(base, config)
    deployments = {name: DeploymentSpec(**read_config((base / path).resolve())) for name, path in config["deployments"].items()}
    mixes, baseline_classes = _add_baseline_mixes(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resolved_sources = {}
    for name, source in sources.items():
        if mode == "run":
            resolved_sources[name] = asdict(source["experiment"])
        else:
            resolved_sources[name] = [{"path": str(p), "sha256": prepared.sha256}
                                      for p, prepared in zip(source["paths"], source["prepared"])]
    fingerprint = digest({"config": config, "sources": resolved_sources,
                          "deployments": {k: asdict(v) for k, v in deployments.items()},
                          "source": source_metadata()["app_source_sha256"]})
    manifest = output / "manifest.json"
    if manifest.exists():
        if not resume or json.loads(manifest.read_text())["fingerprint"] != fingerprint:
            raise ValueError("Mixed-study output exists or resume fingerprint differs")
    else:
        atomic_json(manifest, {"schema": "masbench_mixed_study_v1", "fingerprint": fingerprint,
                               "config": config, "resolved_sources": resolved_sources,
                               "environment": source_metadata()})
    cases = [(d, dep, mix, weights, rate, rep)
             for d, dep in deployments.items() for mix, weights in mixes.items()
             for rate in config["rates"] for rep in range(config.get("repetitions", 1))]
    # Run low-load single-class references first; shuffle only the measured mixed cases.
    baseline_cases = [c for c in cases if c[2].startswith("isolated__")]
    measured_cases = [c for c in cases if not c[2].startswith("isolated__")]
    random.Random(config.get("seed", 42)).shuffle(measured_cases)
    cases = baseline_cases + measured_cases
    results = []
    for case_index, (deployment_name, deployment, mix_name, weights, rate, repetition) in enumerate(cases):
        key = digest([deployment_name, mix_name, rate, repetition])[:16]
        case_dir = output / key
        case_dir.mkdir(exist_ok=True)
        summary_path = case_dir / "summary.json"
        if resume and summary_path.exists():
            results.append(json.loads(summary_path.read_text()))
            continue
        attempt = case_dir / ("attempt_" + uuid4().hex[:12])
        attempt.mkdir()
        seed = config.get("seed", 42) + repetition
        labels, exact_counts = weighted_class_schedule(config.get("count", 20), weights, seed)
        offsets = arrival_offsets(len(labels), rate, config.get("arrival", "constant"), seed)
        ordinals = defaultdict(int)
        source_indices = []
        for label in labels:
            index = ordinals[label]
            ordinals[label] += 1
            size = 1 if mode == "run" else len(sources[label]["paths"])
            source_indices.append(index % size)
        schedule = [{"index": i, "offset_sec": offsets[i], "workload_class": label,
                     "source_index": source_indices[i]} for i, label in enumerate(labels)]
        atomic_json(attempt / "arrival_schedule.json", {"policy": config.get("arrival", "constant"),
                    "rate": rate, "seed": seed, "mix": mix_name, "exact_counts": exact_counts,
                    "schedule_sha256": digest(schedule), "arrivals": schedule})
        protocol = config.get("cache_protocol", "uncontrolled")
        cache_evidence = verify_cache_protocol(deployment, protocol)
        semaphore = threading.BoundedSemaphore(deployment.concurrency)
        sampler = None
        if config.get("collect_backend_metrics") and deployment.backend != "mock":
            from .backend_adapters import build_backend_trace_adapter
            sampler = BackendMetricsSampler(
                url=deployment.telemetry.get("metrics_url") or metrics_url_from_base_url(deployment.endpoint),
                output_path=attempt / "backend_metrics.json",
                interval_sec=config.get("backend_metrics_interval_sec", .5),
                adapter=build_backend_trace_adapter(deployment.telemetry.get("adapter", "generic"), deployment.telemetry))
            sampler.start()

        def execute(index):
            label = labels[index]
            source_index = source_indices[index]
            root = attempt / "traces" / label
            if mode == "run":
                experiment = replace(sources[label]["experiment"], deployment=deployment, load=1)
                runner = build_experiment(experiment, trace_dir=root)
                runner.config.export_trace_views = False
                runner._llm_slots = semaphore
                runner.trace.journal_enabled = True
                try:
                    result = runner.run()
                finally:
                    runner.trace.close_journal()
                trace = runner.trace
            else:
                src = sources[label]
                result, trace = replay_trace(src["paths"][source_index], deployment, trace_dir=root,
                    llm_slots=semaphore, export_views=False, journal=True,
                    max_operations=config.get("max_replay_operations", 4096),
                    prepared=src["prepared"][source_index])
            terminal = next(e for e in reversed(trace.events) if e.get("canonical_type") == "run_finish")
            return {"status": result["status"], "workload_class": label, "source_index": source_index,
                    "trace_path": str(trace.trace_path), "trace_origin_perf": trace.start_perf,
                    "workload_finished_perf": trace.start_perf + terminal["relative_time_sec"]}

        journal = (attempt / "arrivals.jsonl").open("a", encoding="utf-8")
        try:
            summary, records = run_open_loop(execute, count=len(labels), rate=rate,
                arrival=config.get("arrival", "constant"), seed=seed,
                max_inflight=config.get("max_inflight_workflows", 64),
                slo_sec=config.get("slo_sec", 60),
                on_record=lambda row: (journal.write(json.dumps(row) + "\n"), journal.flush()))
        finally:
            journal.close()
            if sampler:
                sampler.stop()
        errors, replay_checks = [], []
        for row in records:
            if "trace_path" not in row:
                continue
            try:
                events = read_events(row["trace_path"])
                analysis = analyze_events(events)
                atomic_json(attempt / "analysis" / (analysis["run_id"] + ".json"), analysis)
                row["trace_metrics"] = _trace_totals(events, analysis)
                if mode == "replay":
                    src = sources[row["workload_class"]]
                    comparison = compare_replay(src["prepared"][row["source_index"]].events, events,
                        output_length_tolerance=config.get("output_length_tolerance", 0))
                    check = {"run_id": analysis["run_id"], "workload_class": row["workload_class"],
                             "invariants": comparison["fixed_workload_invariants_pass"],
                             "length": comparison["output_length_agreement"],
                             "identity": comparison["identity_agreement"]}
                    check["eligible"] = check["invariants"] and check["length"]["pass"] and (
                        not config.get("require_identity_match") or check["identity"]["status"] == "matched")
                    replay_checks.append(check)
            except Exception as exc:
                errors.append({"trace": row["trace_path"], "error": f"{type(exc).__name__}: {exc}"})
        per_class = {name: _class_summary(name, config["classes"][name], records,
                                          summary["accounting_window_sec"], config.get("slo_sec", 60))
                     for name in exact_counts}
        total_internal = sum(row["internal_requests"] for row in per_class.values())
        for row in per_class.values():
            row["internal_request_share"] = row["internal_requests"] / total_internal if total_internal else None
        backend = summarize_backend_metrics(sampler.samples) if sampler else {
            "backend_metrics_sample_count": 0, "scope": "not collected"}
        summary.update({"schema": "masbench_mixed_case_v1", "case_id": key,
            "deployment": deployment_name, "mix": mix_name, "mix_weights": weights,
            "exact_class_counts": exact_counts, "rate": rate, "repetition": repetition,
            "mode": mode, "schedule_sha256": digest(schedule), "per_class": per_class,
            "analysis_errors": errors, "replay_checks": replay_checks,
            "replay_equivalence_failures": sum(not row["eligible"] for row in replay_checks),
            "cache_protocol": protocol, "cache_protocol_evidence": cache_evidence,
            "backend_metrics": backend, "attempt_path": str(attempt),
            "backend_metrics_path": str(attempt / "backend_metrics.json") if sampler else None,
            "endpoint_observations": _endpoint_observations(sampler.samples) if sampler else {},
            "physical_resource_scope": "endpoint-wide; no per-class physical attribution is inferred",
            "mock": deployment.backend == "mock"})
        summary["internal_requests"] = sum(row["internal_requests"] for row in per_class.values())
        summary["internal_qps"] = summary["internal_requests"] / summary["accounting_window_sec"]
        atomic_json(attempt / "records.json", records)
        atomic_json(summary_path, summary)
        results.append(summary)
        atomic_json(output / "results.json", results)
        print(json.dumps({"case": case_index + 1, "total": len(cases), "deployment": deployment_name,
                          "mix": mix_name, "rate": rate, "completed": summary["completed"]}), flush=True)
    # Derive class slowdown from the matching deployment/rate isolated references.
    baselines = defaultdict(lambda: defaultdict(list))
    for row in results:
        if row["mix"].startswith("isolated__"):
            name = row["mix"].split("isolated__", 1)[1]
            baselines[(row["deployment"], row["rate"])][name].append(row["per_class"][name])
    for row in results:
        refs = baselines.get((row["deployment"], row["rate"]), {})
        for name, metrics in row["per_class"].items():
            base_rows = refs.get(name, [])
            base_p50 = distribution([r["completed_e2e_sec"]["p50"] for r in base_rows
                                     if r["completed_e2e_sec"]["p50"] is not None])["mean"]
            base_p95 = distribution([r["completed_e2e_sec"]["p95"] for r in base_rows
                                     if r["completed_e2e_sec"]["p95"] is not None])["mean"]
            metrics["isolated_baseline_p50_sec"] = base_p50
            metrics["isolated_baseline_p95_sec"] = base_p95
            metrics["slowdown_p50"] = metrics["completed_e2e_sec"]["p50"] / base_p50 if base_p50 else None
            metrics["slowdown_p95"] = metrics["completed_e2e_sec"]["p95"] / base_p95 if base_p95 else None
        atomic_json(output / row["case_id"] / "summary.json", row)
    atomic_json(output / "results.json", results)
    atomic_json(output / "capacity_results.json", mixed_capacity_summary(results,
        min_slo_success=config.get("min_slo_success", .95),
        max_failure_fraction=config.get("max_failure_fraction", .01)))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        path = Path(args.config).resolve()
        config = read_config(path)
        _validate_config(config)
        _resolve_sources(path.parent, config)
        for deployment in config["deployments"].values():
            DeploymentSpec(**read_config((path.parent / deployment).resolve()))
        print(json.dumps({"valid": True, "mode": config.get("mode", "replay"),
                          "classes": sorted(config["classes"]), "mixes": sorted(config["mixes"])}))
        return 0
    results = run_mixed_study(args.config, args.output, resume=args.resume)
    return int(any(row["analysis_errors"] or row["replay_equivalence_failures"] for row in results))


if __name__ == "__main__":
    raise SystemExit(main())

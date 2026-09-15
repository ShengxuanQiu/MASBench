"""Hardware-independent graph metrics and explicitly sourced trace dynamics."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path

from .execution_graph import ExecutionGraph
from .tracing import estimate_tokens


def read_events(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "min": None, "mean": None, "p50": None, "p95": None, "max": None}
    def percentile(p):
        x = (len(values) - 1) * p
        lo = int(x)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (x - lo)
    return dict(count=len(values), min=values[0], mean=sum(values)/len(values),
                p50=percentile(.5), p95=percentile(.95), max=values[-1])


def dag_metrics(parents, *, exact_width_limit=2000):
    """Width = maximum antichain, NOT maximum level size. Density = E/[N(N-1)/2]."""
    parents = {n: set(p) for n, p in parents.items()}
    children = {n: set() for n in parents}
    for n, ps in parents.items():
        for p in ps:
            if p not in parents:
                raise ValueError(f"Unknown parent: {p}")
            children[p].add(n)
    indegree = {n: len(p) for n, p in parents.items()}
    queue = deque(n for n, d in indegree.items() if not d)
    order, levels = [], {}
    while queue:
        n = queue.popleft()
        order.append(n)
        levels[n] = 1 + max((levels[p] for p in parents[n]), default=0)
        for child in sorted(children[n]):
            indegree[child] -= 1
            if not indegree[child]:
                queue.append(child)
    if len(order) != len(parents):
        raise ValueError("Cyclic graph")
    size = len(order)
    depth = max(levels.values(), default=0)
    level_width = max(Counter(levels.values()).values(), default=0)
    width = None
    if size <= exact_width_limit:
        # Dilworth via bipartite matching on the transitive closure.
        reachable = {n: set() for n in parents}
        for n in reversed(order):
            for c in children[n]:
                reachable[n].add(c)
                reachable[n].update(reachable[c])
        left_match, right_match = {}, {}
        for root in order:
            search, previous, seen = deque([root]), {}, {root}
            free = None
            while search and free is None:
                u = search.popleft()
                for v in sorted(reachable[u]):
                    if v in previous:
                        continue
                    previous[v] = u
                    if v not in right_match:
                        free = v
                        break
                    nxt = right_match[v]
                    if nxt not in seen:
                        seen.add(nxt)
                        search.append(nxt)
            while free is not None:
                u = previous[free]
                old = left_match.get(u)
                left_match[u], right_match[free] = free, u
                free = old
        width = size - len(left_match)
    edges = sum(map(len, parents.values()))
    return {"nodes": size, "edges": edges, "depth_nodes": depth,
            "width_max_antichain": width, "width_source": "observed" if width is not None else "unavailable",
            "width_level_lower_bound": level_width,
            "dag_edge_density": edges / (size*(size-1)/2) if size > 1 else 0,
            "critical_path_node_ratio": depth/size if size else 0,
            "fan_in": distribution([len(p) for p in parents.values()]),
            "fan_out": distribution([len(c) for c in children.values()])}, order, children


def analyze_events(events):
    graph = ExecutionGraph.from_events(events)
    if any(op.status != "completed" for op in graph.operations.values()):
        raise ValueError("Incomplete/failed DAG: retain separately, do not mix with completed structural corpus")
    graph.validate(replay=True)  # Full payload/snapshot/hash integrity is required for this analysis contract.
    if not any(e.get("canonical_type") == "run_finish" for e in events):
        raise ValueError("Missing run_finish")
    structure, order, children = dag_metrics({n: op.parents for n, op in graph.operations.items()})
    data_edges = {(a,b) for a,b,k in graph.edges if k == "data"}
    control_edges = {(a,b) for a,b,k in graph.edges if k == "control"}
    consumers = defaultdict(set)
    for aid, oid in graph.consumptions:
        consumers[aid].add(oid)
    lengths = {a: len((v["content"] if isinstance(v["content"], str) else json.dumps(v["content"], ensure_ascii=False)).encode())
               for a, v in graph.artifacts.items()}
    unique_bytes = sum(lengths[a] for a in consumers)
    consumed_bytes = sum(lengths[a]*len(c) for a,c in consumers.items())
    llm_consumed_bytes = sum(lengths[a]*sum(graph.operations[c].kind == "llm" for c in cs) for a,cs in consumers.items())
    structure.update(llm_operations=sum(op.kind == "llm" for op in graph.operations.values()),
                     tool_operations=sum(op.kind != "llm" for op in graph.operations.values()),
                     llm_artifact_consumed_bytes=llm_consumed_bytes, data_edges=len(data_edges), control_edges=len(control_edges),
                     data_control_overlap=len(data_edges & control_edges),
                     artifact_reuse_multiplicity=distribution([len(consumers[a]) for a in graph.artifacts]),
                     artifact_consumed_bytes=consumed_bytes,
                     artifact_unique_consumed_bytes=unique_bytes,
                     artifact_byte_amplification=consumed_bytes/unique_bytes if unique_bytes else None,
                     artifacts=len(graph.artifacts))
    delivery_sources=Counter()
    delivered_bytes=delivered_tokens=source_delivery_bytes=source_delivery_tokens=0
    unique_sources=set()
    for event in graph.deliveries:
        delivered_bytes+=event.get('materialized_bytes',0)
        delivered_tokens+=event.get('materialized_tokens_est',0)
        sources=event.get('source_artifact_ids',[]) or [event['artifact_id']]
        for aid in sources:
            unique_sources.add(aid);delivery_sources[aid]+=1
            source_delivery_bytes+=lengths[aid]
            source_delivery_tokens+=estimate_tokens(graph.artifacts[aid]['content'])
    unique_source_bytes=sum(lengths[aid] for aid in unique_sources)
    unique_source_tokens=sum(estimate_tokens(graph.artifacts[aid]['content']) for aid in unique_sources)
    information_flow={"delivered_bytes":delivered_bytes,"delivered_tokens_est":delivered_tokens,
        "unique_source_artifact_bytes":unique_source_bytes,"unique_source_artifact_tokens_est":unique_source_tokens,
        "artifact_reuse_multiplicity":distribution(list(delivery_sources.values())),
        "delivery_compression_ratio_bytes":delivered_bytes/source_delivery_bytes if source_delivery_bytes else None,
        "delivery_compression_ratio_tokens":delivered_tokens/source_delivery_tokens if source_delivery_tokens else None,
        "context_information_amplification_bytes":delivered_bytes/unique_source_bytes if unique_source_bytes else None,
        "context_information_amplification_tokens":delivered_tokens/unique_source_tokens if unique_source_tokens else None,
        "definition":"Delivery-event materialized payload divided by its source payload (compression), or by unique source artifacts (reuse-inclusive amplification). Tokens use the benchmark estimator. Byte/token repetition is not semantic-information duplication.",
        "token_source":"estimated"}
    # Source stage IDs and declared dependencies are retained in the run manifest.
    declared = None
    run_config = {}
    for e in events:
        if e.get("canonical_type") == "run_start":
            run_config = e.get("extra", {}).get("config", {})
            exp = run_config.get("extra", {}).get("experiment")
            if exp:
                from .specs import stage_dependencies
                declared = dag_metrics(stage_dependencies(exp["structure"]["workflow"]["stages"]))[0]
    by_kind = defaultdict(dict)
    start_run = min(e["relative_time_sec"] for e in events if e.get("canonical_type") == "run_start")
    finish_run = max(e["relative_time_sec"] for e in events if e.get("canonical_type") == "run_finish")
    for e in events:
        oid = e.get("operation_id")
        if oid in graph.operations:
            by_kind[e.get("canonical_type")][oid] = e
    start, end, ready, inputs, outputs = {}, {}, {}, {}, {}
    for n in order:
        llm = graph.operations[n].kind == "llm"
        s = by_kind["request_submit" if llm else "operation_start"][n]
        f = by_kind["request_finish" if llm else "operation_finish"][n]
        start[n], end[n] = s["relative_time_sec"], f["relative_time_sec"]
        if end[n] < start[n]:
            raise ValueError("Non-monotonic operation timestamps")
        logical_ready = max((end[p] for p in graph.operations[n].parents), default=start_run)
        if start[n] + 1e-6 < logical_ready:
            raise ValueError(f"Operation submitted before its prerequisites: {n}")
        if llm:
            ready[n] = by_kind["request_ready"][n]["relative_time_sec"]
            if ready[n] + 1e-6 < logical_ready or ready[n] > start[n] + 1e-6:
                raise ValueError(f"Invalid ready/submit dependency timing: {n}")
            inputs[n] = f.get("backend_prompt_tokens")
            outputs[n] = f.get("backend_completion_tokens")
    # Recompute weights from observed wall time, never mock/simulated duration fields.
    weights = {n: end[n]-start[n] for n in order}
    earliest, tail = {}, {}
    for n in order:
        earliest[n] = weights[n] + max((earliest[p] for p in graph.operations[n].parents), default=0)
    for n in reversed(order):
        tail[n] = weights[n] + max((tail[c] for c in children[n]), default=0)
    cp = max(earliest.values(), default=0)
    critical = {n for n in order if abs(earliest[n]+tail[n]-weights[n]-cp) <= max(1e-9, cp*1e-8)}
    timeline = defaultdict(Counter)
    for n in order:
        if n in ready:
            timeline[ready[n]]["ready_waiting"] += 1
            timeline[start[n]]["ready_waiting"] -= 1
            timeline[start[n]]["llm_inflight"] += 1
            timeline[end[n]]["llm_inflight"] -= 1
            if inputs[n] is not None and outputs[n] is not None:
                # Reservation envelope, NOT allocated or resident KV cache.
                token_envelope = inputs[n] + outputs[n]
                timeline[start[n]]["inflight_token_envelope"] += token_envelope
                timeline[end[n]]["inflight_token_envelope"] -= token_envelope
        else:
            timeline[start[n]]["tools_inflight"] += 1
            timeline[end[n]]["tools_inflight"] -= 1
        if n in critical:
            r = max((end[p] for p in graph.operations[n].parents), default=start_run)
            timeline[r]["critical_frontier"] += 1
            timeline[end[n]]["critical_frontier"] -= 1
    series, state = [], Counter()
    for t, delta in sorted(timeline.items()):
        state.update(delta)
        series.append({"time_sec": t, **{k:state[k] for k in ("ready_waiting", "llm_inflight", "tools_inflight", "critical_frontier", "inflight_token_envelope")}})
    barriers = []
    for e in events:
        if e.get("canonical_type") == "barrier_sync":
            ps = e.get("waiting_for_nodes", [])
            if ps and all(p in end for p in ps):
                last = max(end[p] for p in ps)
                barriers.append({"event_id": e["event_id"], "arrival_spread_sec": last-min(end[p] for p in ps),
                                 "aggregate_branch_wait_sec": sum(last-end[p] for p in ps),
                                 "release_delay_sec": max(0,e["relative_time_sec"]-last)})
    measurements = [{"time_sec":e["relative_time_sec"], "attributes":e.get("attributes", {}).get("serving_observation", {})}
                    for e in events if e.get("canonical_type") == "backend_measurement"]
    llm_count = len(ready)
    actual_tokens = all(inputs[n] is not None and outputs[n] is not None for n in ready)
    stage_intervals = []
    stages = defaultdict(list)
    for n, op in graph.operations.items():
        stages[op.identities.get("stage_instance_id", "")].append(n)
    for sid, nodes in stages.items():
        stage_intervals.append({"stage_instance_id":sid,"first_operation_sec":min(start[n] for n in nodes),
                                "last_operation_sec":max(end[n] for n in nodes),"operations":len(nodes)})
    payload_bytes = [len(json.dumps(graph.operations[n].payload["messages"],ensure_ascii=False).encode()) for n in ready]
    task_bytes = len(str(run_config.get("query", "")).encode())
    runtime = {"e2e_sec": finish_run-start_run, "weighted_critical_path_sec": cp,
               "critical_path_work_ratio": cp/sum(weights.values()) if sum(weights.values()) else None,
               "weighted_cp_over_e2e": cp/(finish_run-start_run) if finish_run>start_run else None,
               "critical_operation_ids": sorted(critical), "llm_requests":llm_count,
               "client_admission_wait_sec": distribution([start[n]-ready[n] for n in ready]),
               "scheduler_release_delay_sec": distribution([ready[n]-max((end[p] for p in graph.operations[n].parents), default=start_run) for n in ready]),
               "request_service_wall_sec": distribution([weights[n] for n in ready]),
               "request_payload_message_bytes":distribution(payload_bytes),
               "total_input_bytes_per_task_byte":sum(payload_bytes)/task_bytes if task_bytes else None,
               "context_amplification_definition":"Serialized request message bytes / original task UTF-8 bytes; includes role prompts and JSON overhead, not tokenizer or prefix reuse.",
               "stage_operation_intervals":stage_intervals,
               "input_tokens": distribution([x for x in inputs.values() if x is not None]),
               "output_tokens": distribution([x for x in outputs.values() if x is not None]),
               "token_observation_complete": actual_tokens,
               "kv_live": {"value":None, "source":"unavailable", "reason":"Client trace cannot determine resident KV, eviction, sharing, or allocator blocks."},
               "token_envelope_source":"estimated" if actual_tokens else "unavailable",
               "barriers":barriers,
               "stages":[{"type":e["canonical_type"], "time_sec":e["relative_time_sec"], "stage_instance_id":e.get("stage_instance_id")}
                         for e in events if e.get("canonical_type") in {"stage_start","stage_finish"}],
               "timeline":series, "backend_measurements":measurements}
    if not actual_tokens:
        for row in series:
            row["inflight_token_envelope"] = None
    from .pressure import pressure_metrics
    pressure=pressure_metrics(graph,events,runtime)
    return {"pressure":pressure,"information_flow":information_flow,"schema":"masbench_analysis_v1", "run_id":graph.run_id,
            "provenance":{"backend":run_config.get("llm_mode"), "mock":run_config.get("llm_mode")=="mock",
                          "structural_control":[e for e in events if e.get("event_type")=="structural_control_manifest"],
                          "metric_scope":"Realized operation DAG including local artifact packing tools; no transitive reduction."},
            "structure":structure, "declared_stage_graph":declared, "runtime":runtime}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows, errors = [], []
    for path in args.traces:
        try:
            result = analyze_events(read_events(path))
            (output/(result["run_id"]+".json")).write_text(json.dumps(result, indent=2))
            rows.append({"trace":str(path), "run_id":result["run_id"], **result["structure"]})
        except Exception as exc:
            errors.append({"trace":str(path), "error":str(exc)})
    (output/"corpus.json").write_text(json.dumps({"runs":rows,"errors":errors}, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

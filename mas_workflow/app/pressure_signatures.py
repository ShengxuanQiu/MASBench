"""Pressure-signature analysis for realized MASBench execution graphs.

Graph-derived demand stays separate from backend observations. The KV-equivalent
envelope counts each request's full recorded input and output while that request
is active. It is neither instantaneous generated-token growth nor device KV
residency, and excludes retained prefix-cache blocks after request completion.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .execution_graph import ExecutionGraph
from .workload_analysis import analyze_events


@dataclass(frozen=True)
class KVModelContract:
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    bytes_per_element: int = 2

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.num_hidden_layers * self.num_key_value_heads * self.head_dim * self.bytes_per_element

    @classmethod
    def from_model_config(cls, config: dict[str, Any]) -> "KVModelContract":
        missing = [key for key in ("num_hidden_layers", "num_key_value_heads") if config.get(key) is None]
        if missing:
            raise ValueError("Model config lacks KV fields: " + ", ".join(missing))
        head_dim = config.get("head_dim")
        if head_dim is None:
            hidden, heads = config.get("hidden_size"), config.get("num_attention_heads")
            if not hidden or not heads or hidden % heads:
                raise ValueError("Cannot derive head_dim from model config")
            head_dim = hidden // heads
        dtype = str(config.get("torch_dtype", "float16")).lower()
        bytes_per_element = {"float32": 4, "fp32": 4, "float16": 2, "fp16": 2,
                             "bfloat16": 2, "bf16": 2, "int8": 1}.get(dtype)
        if bytes_per_element is None:
            raise ValueError("Unsupported KV dtype for byte estimate: " + dtype)
        return cls(int(config["num_hidden_layers"]), int(config["num_key_value_heads"]),
                   int(head_dim), bytes_per_element)


def causal_waves(parents: dict[str, set[str]]) -> dict[str, int]:
    """Return one-based longest-path levels for an acyclic dependency graph."""
    indegree = {node: len(deps) for node, deps in parents.items()}
    children = {node: [] for node in parents}
    for node, deps in parents.items():
        for parent in deps:
            if parent not in parents:
                raise ValueError("Unknown dependency: " + parent)
            children[parent].append(node)
    queue = deque(sorted(node for node, degree in indegree.items() if degree == 0))
    level: dict[str, int] = {}
    while queue:
        node = queue.popleft()
        level[node] = 1 + max((level[parent] for parent in parents[node]), default=0)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(level) != len(parents):
        raise ValueError("Cyclic dependency graph")
    return level


def _event_maps(events):
    starts, finishes = {}, {}
    for event in events:
        oid, kind = event.get("operation_id"), event.get("canonical_type")
        if not oid:
            continue
        if kind in {"request_submit", "operation_start"}:
            starts[oid] = event
        elif kind in {"request_finish", "operation_finish"}:
            finishes[oid] = event
    return starts, finishes


def _resource_value(sample, scope, key):
    value = sample.get(scope, {}).get(key)
    if isinstance(value, dict):
        value = value.get("value")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def backend_resource_series(samples):
    """Normalize endpoint samples while preserving measurement provenance."""
    output = []
    for sample in samples:
        ai = _resource_value(sample, "device_metrics", "ai_core_utilization_percent")
        bw = _resource_value(sample, "device_metrics", "memory_bandwidth_percent")
        memory = _resource_value(sample, "device_metrics", "memory_usage_percent")
        npu = _resource_value(sample, "device_metrics", "device_utilization_percent")
        kv = _resource_value(sample, "cache_memory_metrics", "kv_cache_usage_percent")
        if kv is None:
            kv = _resource_value(sample, "cache_memory_metrics", "kv_cache_usage_perc")
        ratio = ai / bw if ai is not None and bw is not None and bw > 0 else None
        output.append({
            "time_sec": sample.get("relative_time_sec"),
            "ai_core_utilization_percent": ai,
            "hbm_bandwidth_utilization_percent": bw,
            "hbm_memory_usage_percent": memory,
            "npu_utilization_percent": npu,
            "physical_kv_cache_usage_percent": kv,
            "compute_to_hbm_utilization_ratio": ratio,
            "sources": {
                "ai_core_utilization_percent": "observed" if ai is not None else "unavailable",
                "hbm_bandwidth_utilization_percent": "observed" if bw is not None else "unavailable",
                "physical_kv_cache_usage_percent": "backend-reported" if kv is not None else "unavailable",
                "compute_to_hbm_utilization_ratio": "derived-from-observed" if ratio is not None else "unavailable",
                "arithmetic_intensity_flops_per_dram_byte": "unavailable",
            },
        })
    return output


def build_pressure_signature(events, *, model_config, backend_samples=None):
    analysis = analyze_events(events)
    graph = ExecutionGraph.from_events(events)
    _starts, finishes = _event_maps(events)
    levels = causal_waves({oid: set(op.parents) for oid, op in graph.operations.items()})
    kv = KVModelContract.from_model_config(model_config)
    waves = defaultdict(lambda: {"operation_count": 0, "llm_operation_count": 0,
                                  "tool_operation_count": 0, "prompt_tokens": 0,
                                  "output_tokens": 0, "operation_ids": [],
                                  "stage_instance_ids": set()})
    for oid, op in graph.operations.items():
        row = waves[levels[oid]]
        row["operation_count"] += 1
        row["llm_operation_count" if op.kind == "llm" else "tool_operation_count"] += 1
        row["operation_ids"].append(oid)
        row["stage_instance_ids"].add(op.identities.get("stage_instance_id", ""))
        finish = finishes.get(oid, {})
        row["prompt_tokens"] += int(finish.get("backend_prompt_tokens") or 0)
        row["output_tokens"] += int(finish.get("backend_completion_tokens") or 0)
    wave_rows, cumulative_tokens = [], 0
    for level in sorted(waves):
        row = waves[level]
        active_tokens = row["prompt_tokens"] + row["output_tokens"]
        cumulative_tokens += active_tokens
        wave_rows.append({"wave": level,
            **{key: value for key, value in row.items() if key != "stage_instance_ids"},
            "stage_instance_ids": sorted(row["stage_instance_ids"]),
            "active_token_envelope": active_tokens,
            "logical_kv_equivalent_bytes": active_tokens * kv.bytes_per_token,
            "cumulative_processed_tokens": cumulative_tokens})
    timeline = []
    for point in analysis["runtime"]["timeline"]:
        token_envelope = point.get("inflight_token_envelope")
        timeline.append({**point, "logical_kv_equivalent_bytes":
            token_envelope * kv.bytes_per_token if token_envelope is not None else None})
    stage_rows = []
    for stage in analysis["runtime"]["stage_operation_intervals"]:
        sid, meta = stage["stage_instance_id"], graph.stages.get(stage["stage_instance_id"], {})
        operation_ids = [oid for oid, op in graph.operations.items()
                         if op.identities.get("stage_instance_id") == sid]
        prompt_tokens = sum(int(finishes.get(oid, {}).get("backend_prompt_tokens") or 0) for oid in operation_ids)
        output_tokens = sum(int(finishes.get(oid, {}).get("backend_completion_tokens") or 0) for oid in operation_ids)
        semantics = meta.get("semantics", {})
        stage_rows.append({**stage, "logical_stage_id": meta.get("logical_stage_id", ""),
            "template_type": semantics.get("public_template") or semantics.get("template_type")
                or semantics.get("family") or "atomic",
            "prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
            "processed_tokens": prompt_tokens + output_tokens,
            "completion_reason": meta.get("completion_reason", "")})
    return {
        "schema": "masbench_pressure_signature_v1", "run_id": graph.run_id,
        "structure": analysis["structure"], "information_flow": analysis["information_flow"],
        "causal_waves": wave_rows, "timeline": timeline,
        "stages": sorted(stage_rows, key=lambda row: row["first_operation_sec"]),
        "model_kv_contract": {"bytes_per_token": kv.bytes_per_token, "source": "estimated",
            "definition": "2 * layers * KV heads * head_dim * bytes_per_element; complete recorded tokens of each active request, not instantaneous or retained backend KV.",
            "not_physical_residency": True},
        "backend_resources": backend_resource_series(backend_samples or []),
        "metric_contract": {
            "logical_kv_equivalent_bytes": "estimated full-request token demand envelope; request completion removes it, regardless of backend KV or prefix-cache residency",
            "compute_to_hbm_utilization_ratio": "ratio of observed utilization percentages, not FLOPs/DRAM-byte arithmetic intensity",
            "arithmetic_intensity_flops_per_dram_byte": {"value": None, "source": "unavailable",
                "reason": "No synchronized FLOP and DRAM-byte profiler counters were supplied."}}}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--backend-metrics")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    events = [json.loads(line) for line in Path(args.trace).read_text(encoding="utf-8").splitlines() if line.strip()]
    model = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    samples = json.loads(Path(args.backend_metrics).read_text(encoding="utf-8")).get("samples", []) if args.backend_metrics else []
    result = build_pressure_signature(events, model_config=model, backend_samples=samples)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

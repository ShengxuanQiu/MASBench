#!/usr/bin/env python3
"""Prepare dense Part 5 pressure-signature experiments for one Ascend backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo, out = Path(args.repo).resolve(), Path(args.output).resolve()
    cfg = out / "configs"
    deployment = {
        "model": "masbench-qwen3-8b-pressure", "backend": "openai_compatible",
        "endpoint": "http://127.0.0.1:8001/v1",
        "generation": {"max_tokens": 48, "temperature": 0, "seed": 42,
                       "chat_template_kwargs": {"enable_thinking": False}},
        "concurrency": 32,
        "hardware": {"device": "Huawei Ascend 910, chip version V1", "npu_id": 0,
                     "accelerator_count": 1, "precision": "bfloat16",
                     "parallelism": {"tp": 1, "pp": 1, "dp": 1},
                     "prefix_cache_enabled": False},
        "telemetry": {"adapter": "ascend", "npu_id": 0, "command_timeout_sec": 8,
                      "command_retries": 3, "metrics_url": "http://127.0.0.1:8001/metrics",
                      "metadata": {"chip_id": 1}}, "identity": {}}
    task = {
        "task_input": ("Compare practical ways to reduce water use in a small office. "
                       "State assumptions, cost, maintenance, measurable outcomes, uncertainty, and trade-offs. "
                       "Use concise evidence-labelled bullet points."),
        "roles": {
            "Coordinator": {"instructions": "Create short non-overlapping work items."},
            "Worker": {"instructions": "Produce one concise concrete proposal without invented measurements."},
            "Reducer": {"instructions": "Synthesize every supplied artifact and preserve disagreements."},
            "Reviewer": {"instructions": "For this controlled structure profile, always return a revise decision; the runtime stops at the configured revision limit."}},
        "criteria": "The answer is feasible, specific, assumption-aware, and contains no invented measurements."}
    dump(cfg / "deployment.json", deployment)
    dump(cfg / "task.json", task)

    def experiment(motifs, stages, prompt=None):
        bound = json.loads(json.dumps(task))
        if prompt:
            bound["task_input"] = prompt
        return {"structure": {"motifs": motifs, "workflow": {"stages": stages}},
                "task": bound, "deployment": {"backend": "mock"}}

    canonical = {
        "spawn": experiment({"m": {"family": "Spawn", "width": 7}},
                            [{"id": "spawn", "motif": "m"}]),
        "fork_join": experiment({"m": {"family": "Fork--Join", "width": 7,
                                         "delivery": {"worker_to_reducer": {"mode": "full"}}}},
                                [{"id": "fork_join", "motif": "m"}]),
        "refinement_loop": experiment({"m": {"family": "Refinement Loop", "width": 1,
                                               "max_revisions": 3}},
                                      [{"id": "refinement", "motif": "m"}]),
        "debate": experiment({"m": {"family": "Debate", "width": 4, "rounds": 1,
                                      "connectivity": "all_to_all",
                                      "delivery": {"peer_to_peer": {"mode": "full"}}}},
                             [{"id": "debate", "motif": "m"}])}
    canonical_paths = {}
    for name, value in canonical.items():
        path = cfg / "canonical" / f"{name}.json"
        dump(path, value); canonical_paths[name] = str(path)

    workflows = {
        "spawn_fork_refine": experiment(
            {"s": {"family": "Spawn", "width": 4},
             "f": {"family": "Fork--Join", "width": 4},
             "r": {"family": "Refinement Loop", "max_revisions": 2}},
            [{"id": "spawn", "motif": "s"},
             {"id": "fork", "motif": "f", "inputs": {"context": "spawn.result"}},
             {"id": "refine", "motif": "r", "inputs": {"candidate": "fork.result"}}]),
        "fork_debate_refine": experiment(
            {"f": {"family": "Fork--Join", "width": 4},
             "d": {"family": "Debate", "width": 4, "rounds": 1, "connectivity": "ring"},
             "r": {"family": "Refinement Loop", "max_revisions": 2}},
            [{"id": "fork", "motif": "f"},
             {"id": "debate", "motif": "d", "inputs": {"context": "fork.result"}},
             {"id": "refine", "motif": "r", "inputs": {"candidate": "debate.result"}}]),
        "branched_hierarchy": experiment(
            {"s": {"family": "Spawn", "width": 3},
             "f": {"family": "Fork--Join", "width": 3},
             "d": {"family": "Debate", "width": 3, "rounds": 1, "connectivity": "all_to_all"},
             "j": {"family": "Fork--Join", "width": 3}},
            [{"id": "root", "motif": "s"},
             {"id": "left", "motif": "f", "inputs": {"context": "root.result"}},
             {"id": "right", "motif": "d", "inputs": {"context": "root.result"}},
             {"id": "join", "motif": "j", "inputs": {"context": ["left.result", "right.result"]}}])}
    workflow_paths = {}
    for name, value in workflows.items():
        path = cfg / "workflows" / f"{name}.json"
        dump(path, value); workflow_paths[name] = str(path)

    common = {"schema": "masbench_mixed_study_v1", "mode": "run",
              "deployments": {"ascend910": str(cfg / "deployment.json")},
              "rates": [1.0], "count": 1, "repetitions": 1, "seed": 5202,
              "arrival": "constant", "max_inflight_workflows": 32, "slo_sec": 120,
              "cache_protocol": "cache_disabled", "collect_backend_metrics": True,
              "backend_metrics_interval_sec": 0.1, "min_slo_success": 0,
              "max_failure_fraction": 1, "isolated_baselines": True}
    shape_classes = {name: {"experiment": path, "slo_sec": 120} for name, path in canonical_paths.items()}
    dump(cfg / "5_2_shapes.json", {**common, "classes": shape_classes,
                                   "mixes": {"pilot": {"spawn": 1}}})
    workflow_classes = {name: {"experiment": path, "slo_sec": 180} for name, path in workflow_paths.items()}
    dump(cfg / "5_3_workflows.json", {**common, "classes": workflow_classes, "seed": 5303,
                                      "slo_sec": 180,
                                      "mixes": {"pilot": {"spawn_fork_refine": 1}}})

    frozen = out / "5_2" / "frozen"
    replay_classes = {name: {"trace_corpus": str(frozen / f"{name}.json"), "slo_sec": 30}
                      for name in canonical}
    rates = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
    replay_common = {"schema": "masbench_mixed_study_v1", "mode": "replay",
        "classes": replay_classes, "deployments": {"ascend910": str(cfg / "deployment.json")},
        "repetitions": 1, "arrival": "poisson", "max_inflight_workflows": 192,
        "slo_sec": 30, "cache_protocol": "cache_disabled", "collect_backend_metrics": True,
        "backend_metrics_interval_sec": 0.1, "output_length_tolerance": 1.0,
        "require_identity_match": False, "min_slo_success": 0.9,
        "max_failure_fraction": 0.05}
    dump(cfg / "5_4_load.json", {**replay_common, "rates": rates, "count": 6,
        "seed": 5404, "isolated_baselines": True,
        "mixes": {"balanced": {name: 1 for name in canonical}}})
    compositions = {
        "balanced": {name: 1 for name in canonical},
        "burst_dominated": {"spawn": 2, "fork_join": 6, "refinement_loop": 1, "debate": 1},
        "state_dominated": {"spawn": 1, "fork_join": 1, "refinement_loop": 2, "debate": 6},
        "dependency_dominated": {"spawn": 1, "fork_join": 1, "refinement_loop": 7, "debate": 1}}
    dump(cfg / "5_5_mix.json", {**replay_common,
        "rates": [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0],
        "count": 8, "seed": 5505, "isolated_baselines": False, "mixes": compositions})
    dump(out / "experiment_plan.json", {
        "schema": "masbench_part5_pressure_plan_v1", "model_config": "/model/Qwen3-8B/config.json",
        "figures": ["canonical causal-wave/temporal signatures", "hierarchical stage trajectories",
                    "dense multiplexing latency/goodput/resource curves", "workload-mix capacity curves"],
        "metric_policy": {"logical_kv": "estimated from exact backend token counts and model config",
                          "physical_kv": "backend-reported", "device": "observed via npu-smi",
                          "arithmetic_intensity": "unavailable without FLOP and DRAM-byte counters"}})
    print(json.dumps({"output": str(out), "canonical": list(canonical),
                      "workflows": list(workflows), "rates": rates}, indent=2))


if __name__ == "__main__":
    main()

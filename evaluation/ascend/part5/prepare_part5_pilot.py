#!/usr/bin/env python3
"""Create the reduced, reproducible Ascend pilot matrix for paper Part 5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    out = Path(args.output).resolve()
    cfg = out / "configs"

    deployment = {
        "model": "masbench-qwen3-8b-part5",
        "backend": "openai_compatible",
        "endpoint": "http://127.0.0.1:8001/v1",
        "generation": {
            "max_tokens": 96,
            "temperature": 0,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "concurrency": 16,
        "hardware": {
            "device": "Huawei Ascend 910, chip version V1",
            "npu_id": 0,
            "accelerator_count": 1,
            "precision": "bfloat16",
            "parallelism": {"tp": 1, "pp": 1, "dp": 1},
            "prefix_cache_enabled": False,
        },
        "telemetry": {
            "adapter": "ascend",
            "npu_id": 0,
            "command_timeout_sec": 8,
            "command_retries": 3,
            "metrics_url": "http://127.0.0.1:8001/metrics",
            "metadata": {"chip_id": 1},
        },
        "identity": {},
    }
    task = {
        "task_input": "Compare two practical ways to reduce water use in a small office. State assumptions, cost, maintenance, measurable outcomes, uncertainty, and trade-offs. Be concise.",
        "roles": {
            "Worker": {"instructions": "Produce one concise, concrete proposal. State assumptions and do not invent measurements."},
            "Coordinator": {"instructions": "Decompose the task into short, non-overlapping work items."},
            "Reducer": {"instructions": "Synthesize every supplied artifact and preserve material disagreement."},
            "Reviewer": {"instructions": "Check feasibility, specificity, assumptions, and unsupported claims."},
        },
        "criteria": "The answer is feasible and specific, states assumptions, compares trade-offs, and invents no measurements.",
    }
    dump(cfg / "deployment-ascend.json", deployment)
    dump(cfg / "task-controlled.json", task)

    def experiment(structure: dict, prompt: str) -> dict:
        bound = json.loads(json.dumps(task))
        bound["task_input"] = prompt
        return {"structure": structure, "task": bound, "deployment": {"backend": "mock"}}

    burst = experiment(
        {"motifs": {"burst": {"family": "ParallelAggregate", "width": 8, "aggregation": "concat_summary",
          "delivery": {"worker_to_reducer": {"mode": "full"}}}},
         "workflow": {"stages": [{"id": "burst", "motif": "burst"}]}},
        "Generate independent mitigation options for a small-office water-reduction plan, then synthesize them.",
    )
    context = experiment(
        {"motifs": {"exchange": {"family": "PeerExchange", "width": 4, "rounds": 2,
          "connectivity": "all_to_all", "delivery": {"peer_to_peer": {"mode": "full"}}}},
         "workflow": {"stages": [{"id": "exchange", "motif": "exchange"}]}},
        "Compare water-reduction proposals while preserving conflicting assumptions across peer exchanges.",
    )
    stages = []
    for index in range(8):
        stage = {"id": f"step{index}", "atomic": {"kind": "llm", "role": "Worker"}}
        if index:
            stage["depends_on"] = [f"step{index - 1}"]
        stages.append(stage)
    dependency = experiment(
        {"motifs": {}, "workflow": {"stages": stages}},
        "Refine a water-reduction plan through eight ordered constraint checks.",
    )
    dump(cfg / "burst.json", burst)
    dump(cfg / "context.json", context)
    dump(cfg / "dependency.json", dependency)

    common_study = {
        "count": 1,
        "repetitions": 1,
        "seed": 5202,
        "arrival": "constant",
        "max_inflight_workflows": 16,
        "slo_sec": 20,
        "cache_protocol": "cache_disabled",
        "collect_backend_metrics": True,
        "backend_metrics_interval_sec": 0.2,
    }
    p52 = {
        "tasks": {"controlled": str(cfg / "task-controlled.json")},
        "deployments": {"ascend910": str(cfg / "deployment-ascend.json")},
        "workloads": [
            {"name": "fork-join-width", "preset": "parallel", "factors": {"width": [1, 4, 8]}},
            {"name": "refinement-depth", "preset": "refine", "factors": {"max_revisions": [0, 2, 4]}},
            {"name": "debate-rounds", "preset": "peer", "parameters": [
                {"width": 4, "rounds": 1, "connectivity": "ring"},
                {"width": 4, "rounds": 2, "connectivity": "ring"},
                {"width": 4, "rounds": 3, "connectivity": "ring"},
            ]},
            {"name": "debate-connectivity", "preset": "peer", "parameters": [
                {"width": 4, "rounds": 2, "connectivity": "pairwise"},
                {"width": 4, "rounds": 2, "connectivity": "ring"},
                {"width": 4, "rounds": 2, "connectivity": "all_to_all"},
            ]},
            {"name": "debate-delivery", "base": str(repo / "mas_workflow/configs/publication/base-peer-exchange.json"),
             "factors": {"structure.motifs.peer.delivery.peer_to_peer": [
                 {"mode": "full"},
                 {"mode": "selected", "selector": {"artifact_indices": [0]}},
                 {"mode": "summarized"},
                 {"mode": "referenced"},
                 {"mode": "retrieved"},
             ]}},
        ],
        "study": {**common_study, "rates": [0.05]},
        "capacity": {"dense_points": 1, "min_slo_success": 0.9, "max_failure_fraction": 0.05},
    }
    dump(cfg / "5_2.json", p52)

    p53 = {
        "tasks": {"controlled": str(cfg / "task-controlled.json")},
        "deployments": {"ascend910": str(cfg / "deployment-ascend.json")},
        "workloads": [
            {"name": "matched-chain", "preset": "matched_chain", "factors": {"requests": [4, 8]}},
            {"name": "matched-parallel", "preset": "matched_parallel", "factors": {"requests": [4, 8]}},
            {"name": "fork-join", "preset": "parallel", "factors": {"width": [8]}},
            {"name": "refinement", "preset": "refine", "factors": {"max_revisions": [2]}},
            {"name": "debate", "preset": "peer", "parameters": [
                {"width": 4, "rounds": 2, "connectivity": "ring"},
                {"width": 4, "rounds": 2, "connectivity": "all_to_all"},
            ]},
        ],
        "study": {**common_study, "rates": [0.15, 0.8, 1.6], "count": 4, "arrival": "poisson", "seed": 5303,
                  "max_inflight_workflows": 64},
        "capacity": {"dense_points": 1, "min_slo_success": 0.9, "max_failure_fraction": 0.05},
    }
    dump(cfg / "5_3.json", p53)

    classes = {
        "burst_heavy": {"experiment": str(cfg / "burst.json"), "slo_sec": 20,
                         "description": "width-8 fork-join burst"},
        "context_heavy": {"experiment": str(cfg / "context.json"), "slo_sec": 20,
                           "description": "dense full-delivery debate"},
        "dependency_heavy": {"experiment": str(cfg / "dependency.json"), "slo_sec": 20,
                              "description": "eight-request sequential chain"},
    }
    mixes = {
        "balanced": {"burst_heavy": 1, "context_heavy": 1, "dependency_heavy": 1},
        "burst_context": {"burst_heavy": 1, "context_heavy": 1},
        "burst_dependency": {"burst_heavy": 1, "dependency_heavy": 1},
        "context_dependency": {"context_heavy": 1, "dependency_heavy": 1},
    }
    p54 = {
        "schema": "masbench_mixed_study_v1", "mode": "run", "classes": classes, "mixes": mixes,
        "isolated_baselines": True, "deployments": {"ascend910": str(cfg / "deployment-ascend.json")},
        "rates": [0.15, 0.8], "count": 6, "repetitions": 1, "seed": 5404, "arrival": "poisson",
        "max_inflight_workflows": 96, "slo_sec": 20, "cache_protocol": "cache_disabled",
        "collect_backend_metrics": True, "backend_metrics_interval_sec": 0.2,
        "min_slo_success": 0.9, "max_failure_fraction": 0.05,
    }
    dump(cfg / "5_4.json", p54)

    frozen = out / "5_4/frozen"
    # Pilot SLOs are 1.75x the isolated low-load p95 measured in 5.4,
    # rounded upward. The policy and resolved thresholds are report inputs.
    replay_slos = {"burst_heavy": 6.2, "context_heavy": 10.0, "dependency_heavy": 21.8}
    replay_classes = {
        name: {"trace_corpus": str(frozen / f"{name}.json"), "slo_sec": replay_slos[name]}
        for name in classes
    }
    composition = {
        "balanced": {"burst_heavy": 1, "context_heavy": 1, "dependency_heavy": 1},
        "burst_dominated": {"burst_heavy": 7, "context_heavy": 1.5, "dependency_heavy": 1.5},
        "context_dominated": {"burst_heavy": 1.5, "context_heavy": 7, "dependency_heavy": 1.5},
        "dependency_dominated": {"burst_heavy": 1.5, "context_heavy": 1.5, "dependency_heavy": 7},
    }
    replay_common = {
        "schema": "masbench_mixed_study_v1", "mode": "replay", "classes": replay_classes,
        "deployments": {"ascend910": str(cfg / "deployment-ascend.json")}, "repetitions": 1,
        "arrival": "poisson", "max_inflight_workflows": 128, "slo_sec": 21.8,
        "cache_protocol": "cache_disabled", "collect_backend_metrics": True,
        "backend_metrics_interval_sec": 0.2, "output_length_tolerance": 0.25,
        "require_identity_match": False, "min_slo_success": 0.9, "max_failure_fraction": 0.05,
    }
    p55 = {**replay_common, "mixes": composition, "isolated_baselines": False,
           "rates": [0.5, 2.0, 5.0], "count": 9, "seed": 5505}
    dump(cfg / "5_5.json", p55)
    p56 = {**replay_common, "mixes": {"balanced": composition["balanced"]}, "isolated_baselines": False,
           "rates": [2.0], "count": 9, "seed": 5606}
    dump(cfg / "5_6.json", p56)
    print(json.dumps({"output": str(out), "configs": sorted(p.name for p in cfg.glob("*.json"))}, indent=2))


if __name__ == "__main__":
    main()

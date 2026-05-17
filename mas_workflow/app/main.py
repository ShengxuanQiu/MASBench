"""MASBench-Arch week-1 topology CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .topologies import TopologyConfig, build_workflow
from .tracing import now_ts


def load_env_files() -> None:
    for path in (Path.cwd() / ".env", Path.cwd().parent / ".env"):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text or text.startswith("#") or "=" not in text:
                continue
            key, value = text.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def str_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MASBench-Arch week-1 base topology prototype")
    parser.add_argument("--topology", choices=["single", "independent", "centralized", "decentralized", "hybrid"], default="single")
    parser.add_argument("--task-source", choices=["manual", "swebench_lite"], default="manual")
    parser.add_argument("--swebench-split", default="test")
    parser.add_argument("--swebench-start-index", type=int, default=0)
    parser.add_argument("--swebench-num-instances", type=int, default=1)
    parser.add_argument("--swebench-instance-ids", default="")
    parser.add_argument("--query", default="")
    parser.add_argument("--repo", default="")
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--debate-rounds", type=int, default=2)
    parser.add_argument("--peer-rounds", type=int, default=1)
    parser.add_argument("--manager-policy", choices=["rule_based", "llm"], default="rule_based")
    parser.add_argument("--aggregation-policy", choices=["concat_summary", "vote", "judge"], default="concat_summary")
    parser.add_argument("--communication-topology", choices=["all_to_all", "ring", "random_k", "pairwise"], default="all_to_all")
    parser.add_argument("--tool-mode", choices=["live", "replay", "synthetic"], default="synthetic")
    parser.add_argument("--search-provider", choices=["tavily", "synthetic", "recorded", "auto", "local_repo"], default="auto")
    parser.add_argument("--record-tool-results", default="true")
    parser.add_argument("--replay-snapshot-dir", default="")
    parser.add_argument("--latency-profile", choices=["none", "fast", "medium", "slow", "heavy_tail"], default="none")
    parser.add_argument("--latency-scale", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--llm-mode", choices=["mock", "openai_compatible"], default="mock")
    parser.add_argument("--backend-base-url", "--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="local-mas-model")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-concurrent-llm-calls", type=int, default=2)
    parser.add_argument("--dispatch-policy", choices=["fcfs", "criticality"], default="fcfs")
    parser.add_argument("--trace-dir", default="traces")
    parser.add_argument("--force-live-search-test", default="false")
    parser.add_argument("--allow-synthetic-tools", default="false")
    parser.add_argument("--agent-execution", choices=["fixed", "react"], default="fixed")
    parser.add_argument("--react-max-steps", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--stop-condition", default="default")
    parser.add_argument("--force-centralized-rounds", type=int, default=0)
    parser.add_argument("--allow-parallel-workers", default="true")
    parser.add_argument("--trace-level", choices=["basic", "arch", "detailed"], default="arch")
    parser.add_argument("--export-trace-views", default="true")
    return parser.parse_args()


def manual_instance_id(query: str) -> str:
    digest = hashlib.sha256(query.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"manual_{digest}"


def load_swebench_lite(args: argparse.Namespace) -> list[dict[str, Any]]:
    requested = [item.strip() for item in args.swebench_instance_ids.split(",") if item.strip()]
    instances: list[dict[str, Any]] = []
    try:
        from datasets import load_dataset  # type: ignore

        ds = load_dataset("princeton-nlp/SWE-bench_Lite", split=args.swebench_split)
        rows = list(ds)
        if requested:
            rows = [row for row in rows if row.get("instance_id") in requested]
        else:
            rows = rows[args.swebench_start_index : args.swebench_start_index + args.swebench_num_instances]
        for row in rows:
            instances.append(
                {
                    "instance_id": row.get("instance_id"),
                    "repo": row.get("repo"),
                    "problem_statement": row.get("problem_statement", ""),
                    "base_commit": row.get("base_commit", ""),
                }
            )
    except Exception as exc:
        print(f"WARNING: failed to load SWE-bench Lite via datasets; trying Hugging Face rows API: {exc}", file=sys.stderr)
    try:
        if requested:
            rows = _load_swebench_lite_rows_via_hf_api(0, max(100, args.swebench_num_instances))
            rows = [row for row in rows if row.get("instance_id") in requested]
        else:
            rows = _load_swebench_lite_rows_via_hf_api(args.swebench_start_index, args.swebench_num_instances)
        if rows:
            return rows
    except Exception as exc:
        print(f"WARNING: failed to load SWE-bench Lite via Hugging Face rows API; using synthetic task metadata: {exc}", file=sys.stderr)
    if instances:
        return instances
    fallback_ids = requested or [
        "astropy__astropy-12907",
        "django__django-11099",
        "sympy__sympy-13480",
        "matplotlib__matplotlib-23913",
        "pylint-dev__pylint-7080",
    ][args.swebench_start_index : args.swebench_start_index + args.swebench_num_instances]
    return [
        {
            "instance_id": instance_id,
            "repo": instance_id.split("-")[0].replace("__", "/"),
            "problem_statement": f"SWE-bench Lite problem statement placeholder for {instance_id}. Analyze likely files and evidence needs.",
            "base_commit": "unknown",
            "setup_warning": "synthetic_swebench_metadata_fallback",
        }
        for instance_id in fallback_ids
    ]


def _load_swebench_lite_rows_via_hf_api(offset: int, length: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "dataset": "princeton-nlp/SWE-bench_Lite",
            "config": "default",
            "split": "test",
            "offset": int(offset),
            "length": int(length),
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - public dataset metadata endpoint
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    instances = []
    for item in data.get("rows") or []:
        row = item.get("row") or {}
        if not row.get("instance_id"):
            continue
        instances.append(
            {
                "instance_id": row.get("instance_id"),
                "repo": row.get("repo"),
                "problem_statement": row.get("problem_statement", ""),
                "base_commit": row.get("base_commit", ""),
                "environment_setup_commit": row.get("environment_setup_commit", ""),
                "source": "hf_rows_api",
            }
        )
    return instances


def repo_for_instance(args: argparse.Namespace, instance: dict[str, Any]) -> Path | None:
    if args.repo:
        path = Path(args.repo).expanduser().resolve()
        return path if path.exists() else None
    candidate = Path.cwd().parent / "swebench_repos" / str(instance.get("instance_id"))
    return candidate.resolve() if candidate.exists() else None


def config_for(args: argparse.Namespace, *, query: str, instance_id: str, task_source: str, repo_path: Path | None, run_id: str) -> TopologyConfig:
    provider = args.search_provider
    if provider == "auto":
        provider = "tavily" if args.tool_mode == "live" else "synthetic"
    if provider == "recorded":
        provider = "recorded"
    return TopologyConfig(
        topology_name=args.topology,
        run_id=run_id,
        task_id=instance_id,
        instance_id=instance_id,
        query=query,
        max_rounds=args.max_rounds,
        num_agents=args.num_agents,
        max_retries=args.max_retries,
        stop_condition=args.stop_condition,
        llm_mode=args.llm_mode,
        tool_mode=args.tool_mode,
        latency_profile=args.latency_profile,
        latency_scale=args.latency_scale,
        random_seed=args.random_seed,
        max_concurrent_llm_calls=args.max_concurrent_llm_calls,
        dispatch_policy=args.dispatch_policy,
        backend_base_url=args.backend_base_url,
        model=args.model,
        max_output_tokens=args.max_output_tokens,
        task_source=task_source,
        trace_dir=Path(args.trace_dir),
        repo_path=repo_path,
        replay_snapshot_dir=Path(args.replay_snapshot_dir).expanduser().resolve() if args.replay_snapshot_dir else None,
        record_tool_results=str_bool(args.record_tool_results),
        search_provider=provider,
        manager_policy=args.manager_policy,
        aggregation_policy=args.aggregation_policy,
        communication_topology=args.communication_topology,
        debate_rounds=args.debate_rounds,
        peer_rounds=args.peer_rounds,
        force_centralized_rounds=args.force_centralized_rounds,
        allow_parallel_workers=str_bool(args.allow_parallel_workers),
        force_live_search_test=str_bool(args.force_live_search_test),
        allow_synthetic_tools=str_bool(args.allow_synthetic_tools),
        agent_execution=args.agent_execution,
        react_max_steps=args.react_max_steps,
        trace_level=args.trace_level,
        export_trace_views=str_bool(args.export_trace_views),
    )


def run_one(config: TopologyConfig) -> dict[str, Any]:
    workflow = build_workflow(config)
    return workflow.run()


def main() -> int:
    load_env_files()
    args = parse_args()
    run_id = now_ts()
    summaries = []
    if args.task_source == "manual":
        query = args.query or "分析 MAS benchmark 的研究意义"
        instance_id = manual_instance_id(query)
        repo_path = Path(args.repo).expanduser().resolve() if args.repo else None
        if repo_path and not repo_path.exists():
            print(f"WARNING: repo path does not exist; continuing without repo: {repo_path}", file=sys.stderr)
            repo_path = None
        config = config_for(args, query=query, instance_id=instance_id, task_source="manual", repo_path=repo_path, run_id=run_id)
        summaries.append(run_one(config))
    else:
        instances = load_swebench_lite(args)
        for instance in instances:
            repo_path = repo_for_instance(args, instance)
            config = config_for(
                args,
                query=str(instance.get("problem_statement") or ""),
                instance_id=str(instance.get("instance_id")),
                task_source="swebench_lite",
                repo_path=repo_path,
                run_id=run_id,
            )
            config.extra["swebench"] = instance
            if repo_path is None:
                config.extra["setup_error"] = "repo clone/checkout unavailable; using no_repo synthetic tool path"
            summaries.append(run_one(config))
    for summary in summaries:
        print(f"{summary['topology']} {summary['instance_id']} trace={summary['trace_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

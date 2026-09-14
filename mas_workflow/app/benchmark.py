"""Run separated benchmark factors or replay a recorded execution."""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .specs import DeploymentSpec, load_experiment, read_config
from .runtime import WorkloadConfig
from .motifs.registry import build_workflow


def build_experiment(experiment, *, trace_dir="traces", run_id=None):
    deployment = experiment.deployment
    config = WorkloadConfig(topology_name="composition", run_id=run_id or "run_" + uuid4().hex,
                            task_id="task", instance_id="input", query=experiment.task.task_input,
                            trace_dir=Path(trace_dir), model=deployment.model, llm_mode=deployment.backend,
                            backend_base_url=deployment.endpoint, max_output_tokens=deployment.generation.get("max_tokens", 256),
                            max_concurrent_llm_calls=deployment.concurrency,
                            search_provider=experiment.task.tool_provider, tool_mode=experiment.task.tool_mode,
                            extra={"workload_spec": experiment.compile(), "generation": deployment.generation,
                                   "experiment": asdict(experiment)})
    return build_workflow(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--experiment", required=True)
    run.add_argument("--trace-dir", default="traces/benchmark")
    replay = commands.add_parser("replay")
    replay.add_argument("--trace", required=True)
    replay.add_argument("--deployment", required=True)
    replay.add_argument("--trace-dir", default="traces/replay")
    replay.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.command == "replay":
        from .replay import replay_trace
        summary, _ = replay_trace(args.trace, DeploymentSpec(**read_config(args.deployment)),
                                  trace_dir=args.trace_dir, strict=args.strict)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    experiment = load_experiment(args.experiment)
    semaphore = threading.BoundedSemaphore(experiment.deployment.concurrency)
    def execute():
        runner = build_experiment(experiment, trace_dir=args.trace_dir)
        runner._llm_slots = semaphore
        return runner.run()
    with ThreadPoolExecutor(max_workers=experiment.load) as pool:
        futures = []
        for index in range(experiment.load):
            if index:
                time.sleep(experiment.arrival_interval_sec)
            futures.append(pool.submit(execute))
        results = [future.result() for future in futures]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return int(any(r["status"] not in {"completed", "accepted"} for r in results))


if __name__ == "__main__":
    raise SystemExit(main())

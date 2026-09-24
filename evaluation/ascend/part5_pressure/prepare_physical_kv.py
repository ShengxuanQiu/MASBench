#!/usr/bin/env python3
"""Resolve high-frequency vLLM KV-block telemetry profiles from Part 5 inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part5-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("cache_disabled", "warm_cache_enabled"), required=True)
    args = parser.parse_args()
    root, output = Path(args.part5_root).resolve(), Path(args.output).resolve()
    shape = json.loads((root / "configs/5_2_shapes.json").read_text(encoding="utf-8"))
    deployment = json.loads((root / "configs/deployment.json").read_text(encoding="utf-8"))

    enabled = args.mode == "warm_cache_enabled"
    deployment["hardware"]["prefix_cache_enabled"] = enabled
    # NPU-SMI calls take roughly 1.3 s on this host. Sampling vLLM directly
    # gives a dense KV trace without claiming device-counter observations.
    deployment["telemetry"]["adapter"] = "kv_only"
    deployment_path = output / "deployment.json"
    write(deployment_path, deployment)

    common = {**shape, "deployments": {"ascend910": str(deployment_path)},
              "cache_protocol": args.mode, "backend_metrics_interval_sec": 0.05}
    write(output / "shapes.json", common)
    repeat = {**common, "classes": {"spawn": shape["classes"]["spawn"]},
              "mixes": {"repeat_spawn": {"spawn": 1}},
              "isolated_baselines": False, "count": 2, "rates": [0.2],
              "max_inflight_workflows": 1, "arrival": "constant"}
    write(output / "repeat_spawn.json", repeat)
    write(output / "measurement_contract.json", {
        "schema": "masbench_physical_kv_profile_v1",
        "source": "vLLM Prometheus /metrics: vllm:kv_cache_usage_perc",
        "unit": "percent of vLLM KV block pool occupied",
        "provenance": "backend-reported",
        "prefix_cache_enabled": enabled,
        "metric_scope": "entire isolated backend endpoint",
        "sampling_interval_sec_requested": 0.05,
        "note": "Shape profiles are sequential native executions. The repeated Spawn pair is an exploratory prefix-retention check, not a controlled cross-hardware cache comparison."
    })
    print(json.dumps({"output": str(output), "mode": args.mode, "prefix_cache_enabled": enabled}))


if __name__ == "__main__":
    main()

"""Task-first benchmark metrics and capacity selection."""
from __future__ import annotations

import math
from typing import Any


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    rank = (len(values) - 1) * q
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - rank) + values[hi] * (rank - lo)


def task_metrics(run: dict[str, Any], *, slo_sec: float, accelerator_count: int,
                 physical_power_watts: float | None = None) -> dict[str, Any]:
    tasks = run.get("tasks", [])
    attempted = len(tasks)
    latencies = [x["task_latency_sec"] for x in tasks if x.get("successful") and x.get("task_latency_sec") is not None]
    good = sum(bool(x.get("successful")) and x.get("task_latency_sec") is not None and x["task_latency_sec"] <= slo_sec for x in tasks)
    duration = float(run.get("measurement_duration_sec") or 0.0)
    goodput = good / duration if duration > 0 else 0.0
    result = {
        "schema_version": "masbench.metrics/1.0.0", "run_valid": bool(run.get("run_valid")),
        "task_latency_sec": {"p50": percentile(latencies, .50), "p90": percentile(latencies, .90), "p99": percentile(latencies, .99)},
        "raw_wall_clock_task_latency_sec": {"p50": percentile([x["raw_wall_clock_latency_sec"] for x in tasks if x.get("raw_wall_clock_latency_sec") is not None], .50)},
        "task_goodput": goodput, "slo_attainment": good / attempted if attempted else 0.0,
        "attempted_tasks": attempted, "successful_tasks": sum(bool(x.get("successful")) for x in tasks),
        "slo_compliant_tasks": good, "slo_sec": slo_sec,
        "resource_efficiency_tasks_per_sec_per_accelerator": goodput / accelerator_count,
        "sustainable_capacity": None,
        "sustainable_capacity_note": "Requires a configured multi-load sweep; use sustainable_capacity().",
    }
    if physical_power_watts is not None and physical_power_watts > 0 and good > 0:
        result["energy_per_good_task_joules"] = physical_power_watts * duration / good
        result["task_goodput_per_watt"] = goodput / physical_power_watts
    return result


def sustainable_capacity(cells: list[dict[str, Any]], *, attainment_target: float) -> dict[str, Any]:
    eligible = [x for x in cells if x.get("run_valid") and x.get("slo_attainment", 0.0) >= attainment_target]
    return {"attainment_target": attainment_target,
            "maximum_tested_load": max((x["load"] for x in eligible), default=None),
            "tested_loads": sorted(x["load"] for x in cells)}

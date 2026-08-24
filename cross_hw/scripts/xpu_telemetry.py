#!/usr/bin/env python3
"""Low-overhead xPU and vLLM telemetry sampler.

The sampler intentionally records vendor counters as reported, without turning
utilization percentages into bandwidth or FLOP/s estimates.  This keeps the
result usable across CUDA/CANN collectors while preserving provenance.
"""

from __future__ import annotations

import csv
import argparse
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


NPU_FIELDS = {
    "HBM Usage Rate(%)": "hbm_memory_used_pct",
    "Aicore Usage Rate(%)": "ai_core_util_pct",
    "Aivector Usage Rate(%)": "ai_vector_util_pct",
    "Aicpu Usage Rate(%)": "ai_cpu_util_pct",
    "HBM Bandwidth Usage Rate(%)": "hbm_bandwidth_util_pct",
    "NPU Utilization(%)": "npu_util_pct",
    "Aicube Usage Rate(%)": "ai_cube_util_pct",
}

VLLM_FIELDS = {
    "vllm:num_requests_running": "vllm_requests_running",
    "vllm:num_requests_waiting": "vllm_requests_waiting",
    "vllm:kv_cache_usage_perc": "vllm_kv_cache_used_frac",
    "vllm:prompt_tokens_total": "vllm_prompt_tokens_total",
    "vllm:generation_tokens_total": "vllm_generation_tokens_total",
    "vllm:estimated_flops_per_gpu_total": "vllm_estimated_flops_total",
    "vllm:estimated_read_bytes_per_gpu_total": "vllm_estimated_read_bytes_total",
    "vllm:estimated_write_bytes_per_gpu_total": "vllm_estimated_write_bytes_total",
}

FIELDNAMES = [
    "timestamp_unix",
    "sample_duration_sec",
    *NPU_FIELDS.values(),
    "power_w",
    *VLLM_FIELDS.values(),
    "collector_error",
]


def _number(value: str) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    return float(match.group(0)) if match else None


def read_npu() -> dict[str, float]:
    completed = subprocess.run(
        ["npu-smi", "info", "-t", "usages", "-i", "0", "-c", "0"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    values: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        if ":" not in line:
            continue
        key, raw = (part.strip() for part in line.split(":", 1))
        if key in NPU_FIELDS and (value := _number(raw)) is not None:
            values[NPU_FIELDS[key]] = value
    power = subprocess.run(
        ["npu-smi", "info", "-t", "power", "-i", "0", "-c", "0"],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if (value := _number(power.stdout)) is not None:
        values["power_w"] = value
    return values


def read_vllm(metrics_url: str) -> dict[str, float]:
    with urllib.request.urlopen(metrics_url, timeout=3) as response:
        body = response.read().decode("utf-8")
    values: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        metric = line.split("{", 1)[0].split(" ", 1)[0]
        if metric not in VLLM_FIELDS:
            continue
        try:
            values[VLLM_FIELDS[metric]] = float(line.rsplit(" ", 1)[-1])
        except ValueError:
            continue
    return values


class TelemetrySampler:
    def __init__(self, output: Path, *, interval: float = 0.25, metrics_url: str = "http://127.0.0.1:8000/metrics") -> None:
        self.output = output
        self.interval = interval
        self.metrics_url = metrics_url
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._run, name="xpu-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self) -> None:
        with self.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            while not self._stop.is_set():
                started = time.time()
                row: dict[str, object] = {"timestamp_unix": started, "collector_error": ""}
                errors: list[str] = []
                try:
                    row.update(read_npu())
                except Exception as exc:  # preserve partial vLLM data
                    errors.append(f"npu:{type(exc).__name__}")
                try:
                    row.update(read_vllm(self.metrics_url))
                except Exception as exc:
                    errors.append(f"vllm:{type(exc).__name__}")
                row["sample_duration_sec"] = time.time() - started
                row["collector_error"] = ";".join(errors)
                writer.writerow(row)
                handle.flush()
                self._stop.wait(max(0.0, self.interval - (time.time() - started)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    args = parser.parse_args()
    sampler = TelemetrySampler(args.output, interval=args.interval, metrics_url=args.metrics_url)
    sampler.start()
    if args.ready_file:
        args.ready_file.touch()
    try:
        while not args.stop_file.exists():
            time.sleep(0.1)
    finally:
        sampler.stop()


if __name__ == "__main__":
    main()

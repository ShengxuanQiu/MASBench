"""Cross-platform backend trace adapters.

The MAS tracer owns graph-level events. Backend adapters normalize serving and
device telemetry into a stable xPU-facing schema so GPU/TPU/NPU backends can be
compared without forcing every runtime to expose the same low-level counters.
"""

from __future__ import annotations

import json
import os
import re
import math
import time
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _latest_matching(metrics: dict[str, float], needles: tuple[str, ...]) -> float | None:
    matches = [float(value) for name, value in metrics.items() if any(needle in name for needle in needles)]
    return max(matches) if matches else None


def _sum_matching(metrics: dict[str, float], needles: tuple[str, ...]) -> float:
    values=[float(value) for name,value in metrics.items() if any(needle in name for needle in needles)]
    return sum(values) if values else None


def _label_value(metrics: dict[str, float], label: str) -> float | None:
    marker = f'{label}="'
    for name in metrics:
        if marker not in name:
            continue
        value = name.split(marker, 1)[1].split('"', 1)[0]
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _nvidia_smi_sample(device_id=None) -> dict[str, Any]:
    fields = [
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "power.draw",
        "temperature.gpu",
    ]
    try:
        proc = subprocess.run(
            ["nvidia-smi", *(["-i",str(device_id)] if device_id is not None else []), f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must be best-effort
        return {"status": "unavailable", "error": str(exc)}
    line = next((item for item in proc.stdout.splitlines() if item.strip()), "")
    if not line:
        return {"status": "unavailable", "error": "nvidia-smi returned no GPU rows"}
    values = [part.strip() for part in line.split(",")]
    out: dict[str, Any] = {"status": "success", "source": "nvidia-smi"}
    names = [
        "device_utilization_percent",
        "memory_utilization_percent",
        "memory_used_mib",
        "memory_total_mib",
        "power_watts",
        "temperature_c",
    ]
    for name, value in zip(names, values):
        try:
            out[name] = float(value)
        except ValueError:
            out[name] = None
    if out.get("memory_used_mib") is not None:
        out["memory_used_bytes"] = int(float(out["memory_used_mib"]) * 1024 * 1024)
    if out.get("memory_total_mib") is not None:
        out["memory_total_bytes"] = int(float(out["memory_total_mib"]) * 1024 * 1024)
    return out


def _load_launch_config() -> dict[str, Any]:
    raw = os.environ.get("MAS_VLLM_LAUNCH_CONFIG_JSON", "")
    if raw:
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {"launch_config_error": "invalid MAS_VLLM_LAUNCH_CONFIG_JSON"}
    path = os.environ.get("MAS_VLLM_LAUNCH_CONFIG_PATH", "")
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001 - metadata is best-effort
        return {"launch_config_error": str(exc), "launch_config_path": path}


@dataclass
class BackendTraceAdapter:
    """Base adapter for xPU serving telemetry."""

    device_type: str
    vendor: str
    runtime: str
    device_model: str = "unknown"
    precision: str = "unknown"
    parallelism: dict[str, int] | None = None

    def backend_device_metadata(self) -> dict[str, Any]:
        return {
            "device_type": self.device_type,
            "vendor": self.vendor,
            "device_model": self.device_model,
            "backend_runtime": self.runtime,
            "precision": self.precision,
            "parallelism": self.parallelism or {"tp": 1, "pp": 1, "dp": 1},
            "adapter_status": "active" if self.device_type == "gpu" else "placeholder",
        }

    def collect_device_metrics(self) -> dict[str, Any]:
        return {"status": "unavailable", "reason": f"{self.device_type} device sampler is not implemented"}

    def serving_sample(self, metrics: dict[str, float]) -> dict[str, Any]:
        return {
            "num_requests_running": _latest_matching(metrics, ("num_requests_running", "num_running_requests")),
            "num_requests_waiting": _latest_matching(metrics, ("num_requests_waiting", "num_waiting_requests")),
            "request_success_total": _sum_matching(metrics, ("request_success_total", "requests_total")),
            "prompt_tokens_total": _sum_matching(metrics, ("prompt_tokens_total", "prompt_tokens_sum")),
            "generation_tokens_total": _sum_matching(metrics, ("generation_tokens_total", "generation_tokens_sum")),
            "ttft_seconds_sum": _sum_matching(metrics, ("time_to_first_token_seconds_sum",)),
            "tpot_seconds_sum": _sum_matching(metrics, ("time_per_output_token_seconds_sum",)),
            "itl_seconds_sum": _sum_matching(metrics, ("inter_token_latency_seconds_sum",)),
            "e2e_latency_seconds_sum": _sum_matching(metrics, ("e2e_request_latency_seconds_sum",)),
            "metric_scope": "aggregate_backend_observed",
        }

    def cache_memory_sample(self, metrics: dict[str, float]) -> dict[str, Any]:
        usage = _latest_matching(metrics, ("kv_cache_usage_perc", "gpu_cache_usage_perc", "gpu_cache_usage"))
        blocks = _label_value(metrics, "num_gpu_blocks")
        block_size = _label_value(metrics, "block_size")
        return {
            "cache_unit_type": "unknown",
            "cache_used_ratio": usage,
            "cache_total_units": blocks,
            "cache_unit_token_capacity": block_size,
            "cache_used_units": (usage * blocks) if usage is not None and blocks is not None else None,
            "cache_used_token_capacity": (usage * blocks * block_size) if usage is not None and blocks is not None and block_size is not None else None,
            "prefix_cache_hit_units": _sum_matching(metrics, ("prefix_cache_hits_total", "gpu_prefix_cache_hits_total")),
            "prefix_cache_query_units": _sum_matching(metrics, ("prefix_cache_queries_total", "gpu_prefix_cache_queries_total")),
            "evicted_units": "unavailable",
            "recomputed_tokens": "unavailable",
            "metric_scope": "aggregate_backend_observed",
        }


class VLLMGPUTraceAdapter(BackendTraceAdapter):
    def __init__(self, config=None) -> None:
        self.config=config or {}
        self.launch_config = _load_launch_config()
        parallelism = {
            "tp": int(self.launch_config.get("tensor_parallel_size") or self.launch_config.get("tp") or 1),
            "pp": int(self.launch_config.get("pipeline_parallel_size") or self.launch_config.get("pp") or 1),
            "dp": int(self.launch_config.get("data_parallel_size") or self.launch_config.get("dp") or 1),
        }
        precision = str(self.launch_config.get("dtype") or self.launch_config.get("precision") or "bf16")
        super().__init__(device_type="gpu", vendor="nvidia", runtime="vllm", precision=precision, parallelism=parallelism)

    def backend_device_metadata(self) -> dict[str, Any]:
        out = super().backend_device_metadata()
        for key in (
            "gpu_count",
            "dtype",
            "max_model_len",
            "kv_block_size",
            "gpu_memory_utilization",
            "model_path",
            "served_model_name",
            "vllm_mas_trace_path",
            "launch_config_path",
            "launch_config_error",
        ):
            if key in self.launch_config:
                out[key] = self.launch_config[key]
        return out

    def collect_device_metrics(self) -> dict[str, Any]:
        if self.config and self.config.get("device_id") is None:
            return {"status":"unavailable","reason":"Explicit telemetry device_id is required"}
        sample = _nvidia_smi_sample(self.config.get("device_id"))
        sample.setdefault("metric_scope", "device_level_observed")
        return sample

    def cache_memory_sample(self, metrics: dict[str, float]) -> dict[str, Any]:
        out = super().cache_memory_sample(metrics)
        out["cache_unit_type"] = "kv_block"
        return out


class TPUTraceAdapter(BackendTraceAdapter):
    def __init__(self) -> None:
        super().__init__(device_type="tpu", vendor="google", runtime="placeholder")


class NPUTraceAdapter(BackendTraceAdapter):
    def __init__(self,config=None):
        super().__init__(device_type="npu",vendor="huawei",runtime="ascend")
        self.config=config or {}
    def backend_device_metadata(self):
        return {**super().backend_device_metadata(),"adapter_status":"active","collector":"npu-smi"}
    def collect_device_metrics(self):
        device=str(self.config.get("npu_id",self.config.get("device_id",0)))
        chip=str(self.config.get("metadata",{}).get("chip_id",0))
        values={"device_utilization_percent":None,"memory_usage_percent":None,
                "memory_capacity_mb":None,"memory_bandwidth_percent":None,"power_watts":None}
        raw,errors={},[]
        for kind in ("usages","power"):
            try:
                proc=subprocess.run(["npu-smi","info","-t",kind,"-i",device,"-c",chip],
                                    check=True,capture_output=True,text=True,timeout=1.5)
                raw[kind]=proc.stdout
                labels={"Aicore Usage Rate(%)":"device_utilization_percent","Memory Usage Rate(%)":"memory_usage_percent",
                        "Memory Capacity(MB)":"memory_capacity_mb","Memory Bandwidth Usage Rate(%)":"memory_bandwidth_percent",
                        "Power(W)":"power_watts","Power":"power_watts"}
                for line in proc.stdout.splitlines():
                    if ":" not in line: continue
                    key,value=line.split(":",1)
                    if key.strip() not in labels: continue
                    match=re.match(r"\s*([0-9]+(?:\.[0-9]+)?)",value)
                    if match: values[labels[key.strip()]]=float(match[1])
            except Exception as exc: errors.append(str(exc))
        result={**values,"status":"success" if any(v is not None for v in values.values()) else "unavailable",
                "source":"npu-smi","device_id":device,"chip_id":chip,"raw":raw,"errors":errors}
        path=self.config.get("profiler_counters_path")
        if path:
            try:
                snapshot=json.loads(Path(path).read_text())
                if str(snapshot["device_id"])!=device or not 0<=time.time()-snapshot["timestamp_unix"]<=5:
                    raise ValueError("Stale or mismatched profiler snapshot")
                result["profiler_counters"]={k:v for k,v in snapshot["metrics"].items() if type(v) in (int,float) and math.isfinite(v)}
            except Exception as exc: result["profiler_error"]=str(exc)
        return result


def build_backend_trace_adapter(kind: str, config=None) -> BackendTraceAdapter:
    normalized=(kind or "generic").lower()
    if normalized in {"gpu","vllm","vllm_gpu","nvidia_gpu"}: return VLLMGPUTraceAdapter(config)
    if normalized in {"tpu","google_tpu"}: return TPUTraceAdapter()
    if normalized in {"npu","ascend","ascend_npu"}: return NPUTraceAdapter(config)
    return BackendTraceAdapter(device_type=normalized,vendor="unknown",runtime="unknown")


def canonical_telemetry(sample):
    """Canonical source vocabulary: device-observed maps to observed with device scope."""
    result={}
    for group in ("serving_metrics","cache_memory_metrics","device_metrics"):
        values=sample.get(group,{})
        for key,value in values.items():
            if type(value) in (int,float) or value is None or value=="unavailable":
                available=type(value) in (int,float) and math.isfinite(value)
                source="observed" if group=="device_metrics" else "backend-reported"
                if key in {"cache_used_units","cache_used_token_capacity"}: source="estimated"
                result[group+"."+key]={"value":value if available else None,"source":source if available else "unavailable"}
        if group=="device_metrics":
            for key in ("device_utilization_percent","memory_used_bytes","memory_usage_percent","power_watts"):
                result.setdefault(group+"."+key,{"value":None,"source":"unavailable"})
            for key,value in values.get("profiler_counters",{}).items():
                result["profiler."+key]={"value":value,"source":"observed"}
    return result

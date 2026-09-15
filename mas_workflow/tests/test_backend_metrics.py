from __future__ import annotations

import subprocess
from types import SimpleNamespace

import app.backend_adapters as backend_adapters
from app.backend_adapters import NPUTraceAdapter, VLLMGPUTraceAdapter, build_backend_trace_adapter
from app.backend_metrics import metrics_url_from_base_url, parse_prometheus_metrics, summarize_backend_metrics
from app.specs import DeploymentSpec


def test_metrics_url_from_openai_base_url() -> None:
    assert metrics_url_from_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/metrics"


def test_parse_prometheus_metrics_with_labels() -> None:
    parsed = parse_prometheus_metrics(
        """
# HELP vllm:num_requests_running Number of running requests.
vllm:num_requests_running{model_name="local"} 2
vllm:generation_tokens_total{model_name="local"} 42
"""
    )
    assert parsed['vllm:num_requests_running{model_name="local"}'] == 2
    assert parsed['vllm:generation_tokens_total{model_name="local"}'] == 42


def test_summarize_backend_metrics_deltas() -> None:
    samples = [
        {
            "status": "success",
            "relative_time_sec": 1.0,
            "metrics": {
                "vllm:request_success_total{model_name=\"local\"}": 10,
                "vllm:prompt_tokens_total{model_name=\"local\"}": 100,
                "vllm:generation_tokens_total{model_name=\"local\"}": 200,
                "vllm:num_requests_running{model_name=\"local\"}": 1,
            },
        },
        {
            "status": "success",
            "relative_time_sec": 3.0,
            "metrics": {
                "vllm:request_success_total{model_name=\"local\"}": 13,
                "vllm:prompt_tokens_total{model_name=\"local\"}": 160,
                "vllm:generation_tokens_total{model_name=\"local\"}": 260,
                "vllm:num_requests_running{model_name=\"local\"}": 2,
            },
        },
    ]
    summary = summarize_backend_metrics(samples)
    assert summary["request_success_total_delta"] == 3
    assert summary["prompt_tokens_total_delta"] == 60
    assert summary["generation_tokens_total_delta"] == 60
    assert summary["max_num_requests_running"] == 2
    assert summary["backend_generation_tokens_per_sec_window"] == 30


def test_vllm_gpu_adapter_normalizes_cache_and_serving_metrics() -> None:
    metrics = parse_prometheus_metrics(
        """
vllm:num_requests_running{model_name="local"} 2
vllm:num_requests_waiting{model_name="local"} 1
vllm:request_success_total{model_name="local"} 7
vllm:prompt_tokens_total{model_name="local"} 100
vllm:generation_tokens_total{model_name="local"} 50
vllm:gpu_cache_usage_perc{model_name="local"} 0.25
vllm:cache_config_info{model_name="local",num_gpu_blocks="80",block_size="16"} 1
vllm:gpu_prefix_cache_hits_total{model_name="local"} 4
vllm:gpu_prefix_cache_queries_total{model_name="local"} 10
"""
    )
    adapter = VLLMGPUTraceAdapter()
    serving = adapter.serving_sample(metrics)
    cache = adapter.cache_memory_sample(metrics)

    assert serving["num_requests_running"] == 2
    assert serving["num_requests_waiting"] == 1
    assert serving["request_success_total"] == 7
    assert cache["cache_unit_type"] == "kv_block"
    assert cache["cache_total_units"] == 80
    assert cache["cache_unit_token_capacity"] == 16
    assert cache["cache_used_units"] == 20
    assert cache["cache_used_token_capacity"] == 320
    assert cache["prefix_cache_hit_units"] == 4
    assert cache["prefix_cache_query_units"] == 10


def test_placeholder_adapters_define_xpu_metadata() -> None:
    tpu = build_backend_trace_adapter("tpu")
    npu = build_backend_trace_adapter("npu")

    assert tpu.backend_device_metadata()["device_type"] == "tpu"
    assert tpu.backend_device_metadata()["adapter_status"] == "placeholder"
    assert npu.backend_device_metadata()["device_type"] == "npu"
    assert npu.backend_device_metadata()["adapter_status"] == "active"


def test_ascend_adapter_parses_current_hbm_and_power_fields() -> None:
    values = NPUTraceAdapter._parse_npu_smi(
        """
        HBM Capacity(MB)               : 65536
        HBM Usage Rate(%)              : 86
        Aicore Usage Rate(%)           : 57
        NPU Utilization(%)             : 54
        HBM Bandwidth Usage Rate(%)    : 31
        NPU Real-time Power(W)         : 161.2
        """
    )

    assert values["device_utilization_percent"] == 54
    assert values["ai_core_utilization_percent"] == 57
    assert values["memory_usage_percent"] == 86
    assert values["memory_bandwidth_percent"] == 31
    assert values["power_watts"] == 161.2
    assert values["memory_total_bytes"] == 65536 * 1024 * 1024
    assert values["memory_used_bytes"] == int(65536 * 0.86 * 1024 * 1024)


def test_deployment_accepts_vllm_chat_template_kwargs() -> None:
    deployment = DeploymentSpec(
        model="local-mas-model",
        backend="openai_compatible",
        generation={
            "max_tokens": 128,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    assert deployment.generation["chat_template_kwargs"] == {"enable_thinking": False}


def test_ascend_adapter_retries_transient_command_timeout(monkeypatch) -> None:
    attempts = {"usages": 0, "power": 0}

    def command(args, **kwargs):
        kind = args[args.index("-t") + 1]
        attempts[kind] += 1
        assert kwargs["timeout"] == 8
        if kind == "usages" and attempts[kind] == 1:
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        output = "NPU Utilization(%) : 12\n" if kind == "usages" else "NPU Real-time Power(W) : 160.5\n"
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(backend_adapters.subprocess, "run", command)
    sample = NPUTraceAdapter({"command_timeout_sec": 8, "command_retries": 3}).collect_device_metrics()

    assert sample["status"] == "success"
    assert sample["device_utilization_percent"] == 12
    assert sample["power_watts"] == 160.5
    assert attempts == {"usages": 2, "power": 1}


def test_vllm_gpu_adapter_reads_launch_config(monkeypatch) -> None:
    monkeypatch.setenv(
        "MAS_VLLM_LAUNCH_CONFIG_JSON",
        '{"tensor_parallel_size": 2, "pipeline_parallel_size": 1, "data_parallel_size": 3, '
        '"gpu_count": 6, "dtype": "bfloat16", "max_model_len": 8192, '
        '"kv_block_size": 16, "gpu_memory_utilization": 0.85}',
    )
    adapter = VLLMGPUTraceAdapter()
    metadata = adapter.backend_device_metadata()

    assert metadata["parallelism"] == {"tp": 2, "pp": 1, "dp": 3}
    assert metadata["gpu_count"] == 6
    assert metadata["dtype"] == "bfloat16"
    assert metadata["max_model_len"] == 8192
    assert metadata["kv_block_size"] == 16
    assert metadata["gpu_memory_utilization"] == 0.85

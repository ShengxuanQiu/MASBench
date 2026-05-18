from __future__ import annotations

from app.backend_metrics import metrics_url_from_base_url, parse_prometheus_metrics, summarize_backend_metrics


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

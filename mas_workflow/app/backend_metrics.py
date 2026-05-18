"""Lightweight vLLM/Prometheus metrics sampling for MAS traces."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


METRIC_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+([-+eE0-9.]+)$")


def metrics_url_from_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return "http://127.0.0.1:8000/metrics"
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/metrics", "", "", ""))


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_RE.match(line)
        if not match:
            continue
        name, labels, value = match.groups()
        try:
            numeric = float(value)
        except ValueError:
            continue
        if labels:
            normalized = name + "{" + labels + "}"
        else:
            normalized = name
        metrics[normalized] = numeric
    return metrics


def fetch_prometheus_metrics(url: str, *, timeout: float = 2.0) -> dict[str, float]:
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:  # noqa: S310 - local metrics endpoint
        return parse_prometheus_metrics(response.read().decode("utf-8", errors="replace"))


def _sum_matching(metrics: dict[str, float], needles: tuple[str, ...]) -> float:
    total = 0.0
    for name, value in metrics.items():
        if any(needle in name for needle in needles):
            total += float(value)
    return total


def _latest_matching(metrics: dict[str, float], needles: tuple[str, ...]) -> float | None:
    matches = [float(value) for name, value in metrics.items() if any(needle in name for needle in needles)]
    if not matches:
        return None
    return max(matches)


def summarize_backend_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [s for s in samples if s.get("status") == "success" and isinstance(s.get("metrics"), dict)]
    if not valid:
        errors = [s.get("error") for s in samples if s.get("status") == "error"]
        return {
            "backend_metrics_sample_count": 0,
            "backend_metrics_error_count": len(errors),
            "backend_metrics_errors": [e for e in errors if e][:3],
        }
    first = valid[0]["metrics"]
    last = valid[-1]["metrics"]
    interval_sec = float(valid[-1]["relative_time_sec"]) - float(valid[0]["relative_time_sec"])

    counters = {
        "request_success_total_delta": ("request_success_total", "requests_total"),
        "prompt_tokens_total_delta": ("prompt_tokens_total", "prompt_tokens_sum"),
        "generation_tokens_total_delta": ("generation_tokens_total", "generation_tokens_sum"),
        "time_to_first_token_seconds_sum_delta": ("time_to_first_token_seconds_sum",),
        "time_per_output_token_seconds_sum_delta": ("time_per_output_token_seconds_sum",),
        "e2e_request_latency_seconds_sum_delta": ("e2e_request_latency_seconds_sum",),
    }
    summary: dict[str, Any] = {
        "backend_metrics_sample_count": len(valid),
        "backend_metrics_error_count": len(samples) - len(valid),
        "backend_metrics_interval_sec": round(max(0.0, interval_sec), 6),
    }
    for key, needles in counters.items():
        delta = _sum_matching(last, needles) - _sum_matching(first, needles)
        summary[key] = round(delta, 6)

    running = [_latest_matching(s["metrics"], ("num_requests_running", "num_running_requests")) for s in valid]
    waiting = [_latest_matching(s["metrics"], ("num_requests_waiting", "num_waiting_requests")) for s in valid]
    cache = [_latest_matching(s["metrics"], ("kv_cache_usage_perc", "gpu_cache_usage_perc", "gpu_cache_usage")) for s in valid]
    for key, series in {
        "max_num_requests_running": running,
        "max_num_requests_waiting": waiting,
        "max_gpu_cache_usage_perc": cache,
    }.items():
        vals = [v for v in series if v is not None]
        summary[key] = round(max(vals), 6) if vals else None

    gen_tokens = float(summary.get("generation_tokens_total_delta") or 0.0)
    prompt_tokens = float(summary.get("prompt_tokens_total_delta") or 0.0)
    ttft_sum = float(summary.get("time_to_first_token_seconds_sum_delta") or 0.0)
    tpot_sum = float(summary.get("time_per_output_token_seconds_sum_delta") or 0.0)
    reqs = float(summary.get("request_success_total_delta") or 0.0)
    summary["backend_generation_tokens_per_sec_window"] = round(gen_tokens / interval_sec, 6) if interval_sec > 0 else 0.0
    summary["backend_prompt_tokens_per_sec_window"] = round(prompt_tokens / interval_sec, 6) if interval_sec > 0 else 0.0
    summary["backend_avg_ttft_sec_from_metrics"] = round(ttft_sum / reqs, 6) if reqs > 0 else None
    summary["backend_avg_tpot_sec_from_metrics"] = round(tpot_sum / gen_tokens, 6) if gen_tokens > 0 else None
    return summary


@dataclass
class BackendMetricsSampler:
    url: str
    output_path: Path
    interval_sec: float = 0.5
    timeout_sec: float = 2.0
    start_perf: float = field(default_factory=time.perf_counter)
    samples: list[dict[str, Any]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="backend-metrics-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.timeout_sec + self.interval_sec))
        self.sample_once()
        self.save()

    def sample_once(self) -> None:
        ts = time.time()
        rel = round(time.perf_counter() - self.start_perf, 6)
        try:
            metrics = fetch_prometheus_metrics(self.url, timeout=self.timeout_sec)
            self.samples.append({"timestamp_unix": ts, "relative_time_sec": rel, "status": "success", "metrics": metrics})
        except Exception as exc:
            self.samples.append({"timestamp_unix": ts, "relative_time_sec": rel, "status": "error", "error": str(exc)})

    def _run(self) -> None:
        self.sample_once()
        while not self._stop.wait(max(0.05, self.interval_sec)):
            self.sample_once()

    def save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics_url": self.url,
            "interval_sec": self.interval_sec,
            "sample_count": len(self.samples),
            "summary": summarize_backend_metrics(self.samples),
            "samples": self.samples,
        }
        self.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

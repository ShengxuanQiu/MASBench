"""Search providers with live/replay/synthetic separation."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tracing import TraceContext, estimate_tokens, latency_delay, stable_hash, token_count_source


@dataclass
class SearchResult:
    provider_name: str
    query: str
    results: list[dict[str, Any]]
    answer: str
    result_hash: str
    measured_duration_sec: float
    injected_delay_sec: float
    effective_duration_sec: float
    snapshot_path: str | None = None
    warning: str | None = None


class BaseSearchProvider:
    name = "base"
    external_dependency = "none"
    network_dependent = False
    deterministic = True

    def __init__(self, *, latency_profile: str = "none", latency_scale: float = 1.0, rng: random.Random | None = None) -> None:
        self.latency_profile = latency_profile
        self.latency_scale = latency_scale
        self.rng = rng or random.Random(42)

    def search(self, query: str, **kwargs: Any) -> SearchResult:
        raise NotImplementedError

    def _delay(self) -> float:
        delay = latency_delay(self.latency_profile, self.rng, self.latency_scale)
        if delay > 0:
            time.sleep(delay)
        return delay


class SyntheticSearchProvider(BaseSearchProvider):
    name = "synthetic"

    def search(self, query: str, **kwargs: Any) -> SearchResult:
        start = time.perf_counter()
        results = [
            {
                "title": f"Synthetic evidence {i}",
                "url": f"synthetic://result/{stable_hash(query)[:8]}/{i}",
                "content": f"Stable synthetic result {i} for query: {query}",
            }
            for i in range(1, 4)
        ]
        measured = time.perf_counter() - start
        injected = self._delay()
        digest = stable_hash({"query": query, "results": results})
        return SearchResult(self.name, query, results, "Synthetic answer", digest, measured, injected, measured + injected)


class TavilySearchProvider(BaseSearchProvider):
    name = "tavily"
    external_dependency = "web"
    network_dependent = True
    deterministic = False

    def __init__(self, *, api_key: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.api_key = api_key

    def search(self, query: str, **kwargs: Any) -> SearchResult:
        start = time.perf_counter()
        attempts = int(kwargs.get("attempts") or 3)
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": int(kwargs.get("max_results") or 5),
            "include_answer": True,
            "include_raw_content": False,
            "search_depth": "basic",
        }
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_exc: Exception | None = None
        data: dict[str, Any] = {}
        for attempt in range(1, max(1, attempts) + 1):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - explicit live provider
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                break
            except (urllib.error.URLError, TimeoutError, ConnectionResetError) as exc:
                last_exc = exc
                if attempt >= attempts:
                    raise
                time.sleep(min(2.0, 0.5 * attempt))
        measured = time.perf_counter() - start
        injected = self._delay()
        results = data.get("results") or []
        digest = stable_hash({"query": query, "results": results, "answer": data.get("answer", "")})
        warning = f"retried_after_{type(last_exc).__name__}" if last_exc is not None else None
        return SearchResult(self.name, query, results, data.get("answer", ""), digest, measured, injected, measured + injected, warning=warning)


class RecordedSearchProvider(BaseSearchProvider):
    name = "recorded"

    def __init__(self, *, snapshot_dir: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.snapshot_dir = snapshot_dir

    def search(self, query: str, **kwargs: Any) -> SearchResult:
        input_hash = stable_hash(query)
        candidates = sorted(self.snapshot_dir.glob(f"*search*{input_hash[:12]}*.json"))
        if not candidates:
            candidates = sorted(self.snapshot_dir.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No recorded snapshot found in {self.snapshot_dir}")
        snap = json.loads(candidates[0].read_text(encoding="utf-8"))
        measured = float(snap.get("measured_duration_sec") or 0.0)
        injected = 0.0 if self.latency_profile == "none" else self._delay()
        output = snap.get("output") or {}
        if isinstance(output, dict):
            results = output.get("results") or []
            answer = output.get("answer", "")
        elif isinstance(output, list):
            results = output
            answer = ""
        else:
            results = []
            answer = ""
        digest = snap.get("output_hash") or stable_hash(output)
        return SearchResult(self.name, query, results, answer, digest, measured, injected, measured + injected, str(candidates[0]))


class LocalRepoSearchProvider(BaseSearchProvider):
    name = "local_repo"
    external_dependency = "local_repo"

    def __init__(self, *, repo_path: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.repo_path = repo_path.expanduser().resolve()

    def search(self, query: str, **kwargs: Any) -> SearchResult:
        start = time.perf_counter()
        if not self.repo_path.exists() or not self.repo_path.is_dir():
            raise ValueError(f"repo_path not found: {self.repo_path}")
        cmd = ["rg", "-n", "-i", "--max-count", "5", "--", query[:80], str(self.repo_path)]
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10, check=False)
            lines = (proc.stdout or "").splitlines()[:10]
        except (OSError, subprocess.TimeoutExpired):
            lines = []
        results = [{"title": "repo match", "url": f"file://{line.split(':', 1)[0]}", "content": line[:1000]} for line in lines]
        if not results:
            results = [{"title": "repo fallback", "url": f"file://{self.repo_path}", "content": "No exact rg match; repo accessible."}]
        measured = time.perf_counter() - start
        injected = self._delay()
        digest = stable_hash({"query": query, "results": results})
        return SearchResult(self.name, query, results, "Local repository search", digest, measured, injected, measured + injected)


def build_search_provider(
    *,
    provider_name: str,
    tool_mode: str,
    replay_snapshot_dir: Path | None,
    latency_profile: str,
    latency_scale: float,
    random_seed: int,
    repo_path: Path | None = None,
    force_live_search_test: bool = False,
    allow_synthetic_fallback: bool = False,
) -> BaseSearchProvider:
    rng = random.Random(random_seed)
    if tool_mode == "replay" or provider_name == "recorded":
        if not replay_snapshot_dir:
            raise ValueError("replay mode requires --replay-snapshot-dir")
        return RecordedSearchProvider(snapshot_dir=replay_snapshot_dir, latency_profile=latency_profile, latency_scale=latency_scale, rng=rng)
    if provider_name == "local_repo":
        if not repo_path:
            raise ValueError("local_repo provider requires repo_path")
        return LocalRepoSearchProvider(repo_path=repo_path, latency_profile=latency_profile, latency_scale=latency_scale, rng=rng)
    if provider_name in {"tavily", "auto"} and tool_mode == "live":
        api_key = os.environ.get("TAVILY_API_KEY")
        if api_key:
            return TavilySearchProvider(api_key=api_key, latency_profile=latency_profile, latency_scale=latency_scale, rng=rng)
        if force_live_search_test or not allow_synthetic_fallback:
            raise RuntimeError("TAVILY_API_KEY is required for live Tavily search")
        print("WARNING: TAVILY_API_KEY missing; falling back to synthetic search.")
    return SyntheticSearchProvider(latency_profile=latency_profile, latency_scale=latency_scale, rng=rng)


def record_search_event(
    trace: TraceContext,
    *,
    node_id: str,
    node_name: str,
    tool_mode: str,
    result: SearchResult,
    snapshot_dir: Path | None,
    replay_policy: str,
    latency_profile: str,
    trace_fields: dict[str, Any] | None = None,
) -> None:
    snapshot_path = result.snapshot_path
    if snapshot_dir and tool_mode == "live":
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        input_hash = stable_hash(result.query)
        path = snapshot_dir / f"{node_id}_search_{input_hash[:12]}.json"
        snapshot = {
            "run_id": trace.run_id,
            "topology": trace.topology,
            "instance_id": trace.instance_id,
            "node_id": node_id,
            "tool_name": "search",
            "tool_query": result.query,
            "input_hash": input_hash,
            "output": {"answer": result.answer, "results": result.results},
            "output_hash": result.result_hash,
            "measured_duration_sec": result.measured_duration_sec,
            "timestamp": trace.events[-1]["timestamp"] if trace.events else "",
            "environment_id": trace.environment_id,
            "provider_name": result.provider_name,
            "status": "success",
        }
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        snapshot_path = str(path)
    external_dependency = "web" if result.provider_name == "tavily" else ("local_repo" if result.provider_name == "local_repo" else "none")
    trace.emit(
        event_type="tool_search",
        node_id=node_id,
        node_name=node_name,
        node_type="tool",
        status="success",
        duration_sec=round(result.effective_duration_sec, 6),
        duration_source="recorded_replay" if tool_mode == "replay" else ("synthetic_latency" if result.provider_name == "synthetic" else "measured_live"),
        replay_policy=replay_policy,
        tool_name="search",
        tool_mode=tool_mode if result.provider_name != "synthetic" else "synthetic",
        tool_query=result.query,
        command_summary=f"{result.provider_name}.search",
        result_snapshot_id=snapshot_path,
        tool_result_hash=result.result_hash,
        output_size_chars=len(json.dumps(result.results, ensure_ascii=False)),
        output_size_tokens_est=estimate_tokens(json.dumps(result.results, ensure_ascii=False)),
        token_count_source=token_count_source(),
        measured_duration_sec=result.measured_duration_sec,
        injected_delay_sec=result.injected_delay_sec,
        effective_duration_sec=result.effective_duration_sec,
        latency_profile=latency_profile,
        external_dependency=external_dependency,
        network_dependent=result.provider_name == "tavily",
        deterministic=result.provider_name != "tavily",
        result_count=len(result.results),
        extra={"provider_name": result.provider_name, "warning": result.warning},
        **(trace_fields or {}),
    )

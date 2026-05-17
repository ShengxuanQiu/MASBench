from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from app.search_providers import RecordedSearchProvider, SyntheticSearchProvider, TavilySearchProvider, build_search_provider
from app.tracing import stable_hash


def test_synthetic_provider_returns_non_empty_results() -> None:
    provider = SyntheticSearchProvider(latency_profile="none")
    result = provider.search("mas benchmark")
    assert result.results
    assert result.result_hash


def test_recorded_provider_replay_hash_consistent(tmp_path: Path) -> None:
    output = {"answer": "recorded", "results": [{"title": "A", "content": "B"}]}
    digest = stable_hash(output)
    snap = tmp_path / "node_search_abc.json"
    snap.write_text(
        json.dumps({"output": output, "output_hash": digest, "measured_duration_sec": 0.01}),
        encoding="utf-8",
    )
    provider = RecordedSearchProvider(snapshot_dir=tmp_path, latency_profile="none")
    result = provider.search("same query")
    assert result.result_hash == digest
    assert result.answer == "recorded"
    assert result.results == [{"title": "A", "content": "B"}]


def test_tavily_or_fallback_behavior() -> None:
    if os.environ.get("TAVILY_API_KEY"):
        provider = TavilySearchProvider(api_key=os.environ["TAVILY_API_KEY"], latency_profile="none")
        result = provider.search("OpenAI")
        assert result.results
    else:
        provider = build_search_provider(
            provider_name="auto",
            tool_mode="live",
            replay_snapshot_dir=None,
            latency_profile="none",
            latency_scale=1.0,
            random_seed=42,
            allow_synthetic_fallback=True,
        )
        assert isinstance(provider, SyntheticSearchProvider)


def test_force_live_search_without_key_fails() -> None:
    if os.environ.get("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY is present")
    with pytest.raises(RuntimeError):
        build_search_provider(
            provider_name="tavily",
            tool_mode="live",
            replay_snapshot_dir=None,
            latency_profile="none",
            latency_scale=1.0,
            random_seed=42,
            force_live_search_test=True,
        )

import json
from pathlib import Path

import pytest

from case_study2.memory import TaskMemoryStore, pairwise_distinctiveness, terms
from case_study2.trace import TraceWriter


def test_store_append_retrieve_and_freeze(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "trace.jsonl", run_id="r", workload="w", mode="producer")
    store = TaskMemoryStore(tmp_path / "memory", "task", trace)
    store.append(
        {"source_agent": "agent_A", "artifact_id": "a_1"},
        [{"type": "claim", "content": "Graph context is repeated", "keywords": ["graph"], "evidence_refs": ["a_1"], "token_count": 5}],
    )
    rows = store.retrieve(query="graph", source_artifact_ids=["a_1"], top_k=1)
    assert rows[0]["source_artifact_id"] == "a_1"
    store.freeze()
    assert json.loads((tmp_path / "memory" / "task.json").read_text())["state"] == "frozen"
    with pytest.raises(RuntimeError):
        store.append({"source_agent": "x", "artifact_id": "a_2"}, [])


def test_terms_and_distinctiveness() -> None:
    assert {"graph-aware", "prefill"} <= terms("Graph-aware prefill reuse")
    score = pairwise_distinctiveness([{"prefill", "batching"}, {"provenance", "coverage"}])
    assert score == 1.0


def test_store_rejects_invalid_record(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "trace.jsonl", run_id="r", workload="w", mode="producer")
    store = TaskMemoryStore(tmp_path / "memory", "task", trace)
    with pytest.raises(ValueError):
        store.append(
            {"source_agent": "agent_A", "artifact_id": "a_1"},
            [{"type": "unknown", "content": "", "token_count": 0}],
        )

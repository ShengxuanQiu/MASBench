import json
from pathlib import Path

import pytest

from case_study2.memory import TaskMemoryStore, _parse_json_object
from case_study2.trace import TraceWriter


def test_store_append_retrieve_and_freeze(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "trace.jsonl", run_id="r", workload="w", mode="structured")
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


def test_parser_accepts_think_and_fence() -> None:
    parsed = _parse_json_object('<think>ignored</think>```json\n{"records": [{"type": "claim"}]}\n```')
    assert parsed["records"][0]["type"] == "claim"


def test_placeholder_record_is_not_valid_memory(tmp_path: Path) -> None:
    from case_study2.memory import AsyncMemoryAgent

    agent = object.__new__(AsyncMemoryAgent)
    agent.token_count = lambda text: len(text.split())
    with pytest.raises(ValueError):
        agent._validated_records(
            '{"records":[{"type":"claim","content":"...","keywords":["..."],"evidence_refs":["a_1"]}]}',
            "a_1",
        )

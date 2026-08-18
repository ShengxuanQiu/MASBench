from __future__ import annotations

import argparse
import json
from pathlib import Path


ALLOWED_RECORD_TYPES = {"claim", "evidence", "constraint", "keyword_summary"}


def validate(root: Path, expected_runs: int | None = None) -> list[str]:
    summaries = sorted(root.glob("*/*/*/summary.json"))
    if expected_runs is not None and len(summaries) != expected_runs:
        raise AssertionError(f"expected {expected_runs} runs, found {len(summaries)}")
    checked: list[str] = []
    for summary_path in summaries:
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        tool_events = [event for event in events if event["event_type"] == "tool_return"]
        if len(tool_events) != 1 or tool_events[0].get("provider") != "tavily":
            raise AssertionError(f"{run_dir}: not a single live Tavily return")
        requests = [event for event in events if event["event_type"] == "llm_request_end"]
        if not requests or not all(
            event.get("timing_source") == "openai_compatible_streaming_sse"
            and event.get("first_token_ts")
            and event.get("completion_ts")
            for event in requests
        ):
            raise AssertionError(f"{run_dir}: incomplete SSE timing")
        if not (run_dir / "tavily_snapshot.json").exists():
            raise AssertionError(f"{run_dir}: missing Tavily snapshot")
        if summary["mode"] == "producer":
            memories = list((run_dir / "memory").glob("*.json"))
            if len(memories) != 1:
                raise AssertionError(f"{run_dir}: expected one task memory file")
            memory = json.loads(memories[0].read_text(encoding="utf-8"))
            eligible = int(summary.get("graph_eligible_artifact_count") or 0)
            if memory.get("state") != "frozen":
                raise AssertionError(f"{run_dir}: memory is not frozen")
            if eligible == 0:
                if memory.get("records"):
                    raise AssertionError(f"{run_dir}: negative control unexpectedly materialized memory")
                if summary["execution"].get("memory_materialization") != "not_used":
                    raise AssertionError(f"{run_dir}: negative control did not bypass materialization")
                checked.append(f"{summary['workload']}/{summary['mode']}/{summary['run_id']}")
                continue
            if not memory.get("records"):
                raise AssertionError(f"{run_dir}: memory is empty")
            if not all(record.get("type") in ALLOWED_RECORD_TYPES for record in memory["records"]):
                raise AssertionError(f"{run_dir}: invalid record type")
            if float(summary["evidence_source_coverage"]) != 1.0:
                raise AssertionError(f"{run_dir}: source coverage is incomplete")
            if float(summary["memory_focus_coverage"]) != 1.0:
                raise AssertionError(f"{run_dir}: graph-assigned focus coverage is incomplete")
            if float(summary["memory_grounding_ratio"]) < 0.5:
                raise AssertionError(f"{run_dir}: memory is weakly grounded in producer answers")
            if summary["workload"] != "retry_debug_loop" and float(summary["memory_distinctiveness"]) < 0.35:
                raise AssertionError(f"{run_dir}: source memories are insufficiently distinct")
            if summary["execution"].get("memory_materialization") != "producer_same_generation":
                raise AssertionError(f"{run_dir}: memory was not produced in the source request")
            if any(event["event_type"].startswith("memory_llm") for event in events):
                raise AssertionError(f"{run_dir}: found forbidden second-pass memory LLM request")
        checked.append(f"{summary['workload']}/{summary['mode']}/{summary['run_id']}")
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).with_name("results") / "runs",
    )
    parser.add_argument("--expected-runs", type=int, default=32)
    args = parser.parse_args()
    checked = validate(args.root, args.expected_runs)
    print(f"Validated {len(checked)} real runs")
    for name in checked:
        print(name)


if __name__ == "__main__":
    main()

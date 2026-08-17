from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from transformers import AutoTokenizer

from mas_workflow.app.llm_client import LocalLLMClient
from mas_workflow.app.search_providers import TavilySearchProvider

from .memory import AsyncMemoryAgent, TaskMemoryStore
from .trace import TraceWriter, read_jsonl


WORKLOADS = {
    "debate_allgather_pressure_meso",
    "shared_memory_fanin_meso",
    "hierarchical_synthesis_pressure_meso",
}


def task_consistency_score(text: str) -> float:
    lowered = text.lower()
    required_concepts = [
        ("graph",),
        ("context",),
        ("evidence",),
        ("redundan", "duplicat", "repeat"),
        ("prefill", "token"),
    ]
    matched = sum(any(term in lowered for term in alternatives) for alternatives in required_concepts)
    return matched / len(required_concepts)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def overlap_duration(interval: tuple[float, float], others: list[tuple[float, float]]) -> float:
    start, end = interval
    clipped = sorted((max(start, a), min(end, b)) for a, b in others if min(end, b) > max(start, a))
    merged: list[list[float]] = []
    for left, right in clipped:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return sum(right - left for left, right in merged)


def union_duration(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return sum(right - left for left, right in merged)


class GraphMemoryWorkflow:
    def __init__(self, config: dict[str, Any], *, workload: str, mode: str, run_dir: Path, run_id: str) -> None:
        if workload not in WORKLOADS or mode not in {"raw", "structured"}:
            raise ValueError(f"unsupported workload/mode: {workload}/{mode}")
        self.config = config
        self.workload = workload
        self.mode = mode
        self.run_dir = run_dir
        self.run_id = run_id
        self.task_id = f"{workload}_{run_id}"
        self.trace = TraceWriter(run_dir / "trace.jsonl", run_id=run_id, workload=workload, mode=mode)
        self.tokenizer = AutoTokenizer.from_pretrained(config["main_model"]["path"], trust_remote_code=True)
        self.main = LocalLLMClient(
            model=config["main_model"]["served_name"],
            base_url=config["main_model"]["base_url"],
            temperature=0.0,
            max_tokens=int(config["main_model"]["max_tokens"]),
        )
        self.memory_store: TaskMemoryStore | None = None
        self.memory_agent: AsyncMemoryAgent | None = None
        if mode == "structured":
            self.memory_store = TaskMemoryStore(run_dir / "memory", self.task_id, self.trace)
            memory_client = LocalLLMClient(
                model=config["memory_model"]["served_name"],
                base_url=config["memory_model"]["base_url"],
                temperature=0.0,
                max_tokens=int(config["memory_model"]["max_tokens"]),
            )
            self.memory_agent = AsyncMemoryAgent(
                client=memory_client,
                store=self.memory_store,
                trace=self.trace,
                token_count=self.tokens,
                max_workers=int(config["memory_model"]["max_workers"]),
                max_tokens=int(config["memory_model"]["max_tokens"]),
                seed=int(config["seed"]),
            )
        self.artifacts: dict[str, dict[str, Any]] = {}
        self._artifact_index = 0
        self._artifact_lock = threading.Lock()
        self.start_ts = 0.0

    def tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _activity(self, kind: str, node_id: str, fn: Callable[[], Any]) -> Any:
        start = time.time()
        self.trace.emit("graph_activity_start", activity_kind=kind, node_id=node_id, start_ts=start)
        try:
            return fn()
        finally:
            end = time.time()
            self.trace.emit("graph_activity_end", activity_kind=kind, node_id=node_id, start_ts=start, end_ts=end)

    def artifact(
        self,
        source_agent: str,
        content: str,
        *,
        artifact_type: str,
        memory_eligible: bool = True,
    ) -> str:
        with self._artifact_lock:
            self._artifact_index += 1
            artifact_id = f"a_{self._artifact_index:04d}"
            row = {
                "artifact_id": artifact_id,
                "source_agent": source_agent,
                "artifact_type": artifact_type,
                "content": content,
                "token_count": self.tokens(content),
                "created_ts": time.time(),
                "memory_eligible": memory_eligible,
            }
            self.artifacts[artifact_id] = row
        self.trace.emit("artifact_created", **{key: value for key, value in row.items() if key != "content"})
        if self.memory_agent is not None and memory_eligible:
            self.memory_agent.schedule(row, objective=self.config["objective"])
        return artifact_id

    def search(self) -> str:
        provider = TavilySearchProvider(api_key=os.environ["TAVILY_API_KEY"])
        query = self.config["search_queries"][self.workload]
        start = time.time()

        def call() -> Any:
            return provider.search(
                query,
                max_results=int(self.config["tavily"]["max_results"]),
                include_answer=True,
                include_raw_content=False,
                search_depth=self.config["tavily"]["search_depth"],
            )

        result = self._activity("live_tavily", "tavily_grounding", call)
        snapshot = {
            "provider": result.provider_name,
            "query": result.query,
            "answer": result.answer,
            "results": result.results,
            "result_hash": result.result_hash,
            "measured_duration_sec": result.measured_duration_sec,
        }
        (self.run_dir / "tavily_snapshot.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        text = json.dumps({"answer": result.answer, "results": result.results}, ensure_ascii=False)
        self.trace.emit(
            "tool_return",
            node_id="tavily_grounding",
            provider="tavily",
            start_ts=start,
            end_ts=time.time(),
            latency_sec=result.measured_duration_sec,
            result_tokens=self.tokens(text),
            result_hash=result.result_hash,
        )
        return self.artifact("tavily_grounding", text, artifact_type="tool_evidence")

    def context(self, consumer: str, role: str, artifact_ids: list[str], *, current: str = "") -> str:
        if self.mode == "raw":
            sections = []
            for artifact_id in artifact_ids:
                artifact = self.artifacts[artifact_id]
                sections.append(f"[{artifact_id} | {artifact['source_agent']}]\n{artifact['content']}")
                self.trace.emit(
                    "context_propagation",
                    consumer=consumer,
                    source_artifact_id=artifact_id,
                    source_agent=artifact["source_agent"],
                    propagated_tokens=artifact["token_count"],
                    transfer="raw_artifact",
                )
            return "\n\n".join(sections)

        assert self.memory_agent is not None and self.memory_store is not None
        eligible_ids = [
            artifact_id for artifact_id in artifact_ids if self.artifacts[artifact_id]["memory_eligible"]
        ]
        direct_ids = [
            artifact_id for artifact_id in artifact_ids if not self.artifacts[artifact_id]["memory_eligible"]
        ]
        wait_start = time.time()
        waited = self.memory_agent.wait_for(eligible_ids) if eligible_ids else 0.0
        self.trace.emit(
            "memory_consumer_wait",
            consumer=consumer,
            source_artifact_ids=eligible_ids,
            start_ts=wait_start,
            end_ts=time.time(),
            wait_sec=waited,
        )
        query = f"role={role}; objective={self.config['objective']}; current context={current}"
        rows = self.memory_store.retrieve(
            query=query,
            source_artifact_ids=eligible_ids,
            top_k=int(self.config["retrieval"]["top_k"]),
        ) if eligible_ids else []
        structured_view = "\n".join(
            f"- {row['type']} [{row['source_artifact_id']}]: {row['content']}" for row in rows
        )
        expected_raw = sum(self.artifacts[artifact_id]["token_count"] for artifact_id in eligible_ids)
        self.trace.emit(
            "memory_retrieval",
            consumer=consumer,
            role=role,
            query=query,
            source_artifact_ids=eligible_ids,
            retrieved_source_artifact_ids=sorted({row["source_artifact_id"] for row in rows}),
            memory_ids=[row["memory_id"] for row in rows],
            retrieval_tokens=self.tokens(structured_view),
            raw_equivalent_tokens=expected_raw,
            top_k=int(self.config["retrieval"]["top_k"]),
        )
        direct_sections = []
        for artifact_id in direct_ids:
            artifact = self.artifacts[artifact_id]
            direct_sections.append(f"[{artifact_id} | {artifact['source_agent']}]\n{artifact['content']}")
            self.trace.emit(
                "context_propagation",
                consumer=consumer,
                source_artifact_id=artifact_id,
                source_agent=artifact["source_agent"],
                propagated_tokens=artifact["token_count"],
                transfer="direct_nonshared_artifact",
            )
        return "\n\n".join(part for part in [structured_view, *direct_sections] if part)

    def llm(
        self,
        node_id: str,
        role: str,
        instruction: str,
        context: str,
        *,
        parents: list[str],
        context_consumer: bool,
        max_tokens: int | None = None,
    ) -> str:
        user = f"Task objective:\n{self.config['objective']}\n\nAvailable context:\n{context}\n\nInstruction:\n{instruction}"
        messages = [
            {"role": "system", "content": f"You are the {role} agent. Be evidence-grounded, satisfy the requested detail, and do not invent sources. /no_think"},
            {"role": "user", "content": user},
        ]
        request_id = f"{self.run_id}__main__{node_id}"
        ready = time.time()
        self.trace.emit(
            "llm_request_ready",
            request_id=request_id,
            node_id=node_id,
            agent_role=role,
            parents=parents,
            context_consumer=context_consumer,
            request_ready_ts=ready,
            prompt_tokens_local=self.tokens(user),
        )

        def call() -> tuple[str, dict[str, Any]]:
            submit = time.time()
            self.trace.emit("llm_request_submit", request_id=request_id, node_id=node_id, submit_ts=submit)
            return self.main.invoke_messages_streaming_with_metadata(
                messages,
                metadata={"request_id_for_backend": request_id, "agent_id": node_id, "priority": "normal"},
                max_tokens=max_tokens or int(self.config["main_model"]["max_tokens"]),
                seed=int(self.config["seed"]),
            )

        content, metadata = self._activity("main_llm", node_id, call)
        usage = metadata.get("usage") or {}
        self.trace.emit(
            "llm_request_end",
            request_id=request_id,
            node_id=node_id,
            agent_role=role,
            parents=parents,
            context_consumer=context_consumer,
            request_ready_ts=ready,
            first_token_ts=metadata.get("first_token_ts"),
            completion_ts=metadata.get("completion_ts"),
            input_tokens=int(usage.get("prompt_tokens") or self.tokens(user)),
            output_tokens=int(usage.get("completion_tokens") or self.tokens(content)),
            ttft_sec=metadata.get("ttft_sec"),
            request_latency_sec=(metadata.get("stream_e2e_ms") or 0) / 1000.0,
            tpot_sec=metadata.get("tpot_sec"),
            stream_inter_token_ms=metadata.get("stream_inter_token_ms"),
            timing_source=metadata.get("timing_source"),
        )
        return content

    def parallel(self, items: list[Any], fn: Callable[[Any], tuple[str, str]]) -> dict[str, str]:
        with ThreadPoolExecutor(max_workers=len(items)) as executor:
            return dict(executor.map(fn, items))

    def consume_and_run(
        self,
        node: str,
        role: str,
        source_ids: list[str],
        instruction: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        context = self.context(node, role, source_ids, current=instruction)
        output = self.llm(
            node,
            role,
            instruction,
            context,
            parents=source_ids,
            context_consumer=True,
            max_tokens=max_tokens,
        )
        return self.artifact(node, output, artifact_type="agent_output")

    def debate(self) -> str:
        n = int(self.config["workloads"][self.workload]["num_agents"])
        rounds = int(self.config["workloads"][self.workload]["rounds"])

        def opinion(i: int) -> tuple[str, str]:
            node = f"opinion_{i}"
            out = self.llm(
                node,
                "peer",
                f"Develop distinct position {i} as exactly {6 + 2 * i} numbered, self-contained evidence statements.",
                "Grounding evidence will be integrated downstream.",
                parents=[],
                context_consumer=False,
                max_tokens=160 + 64 * i,
            )
            return node, self.artifact(node, out, artifact_type="opinion")

        with ThreadPoolExecutor(max_workers=2) as executor:
            web_future = executor.submit(self.search)
            opinion_future = executor.submit(self.parallel, list(range(1, n + 1)), opinion)
            web = web_future.result()
            previous = opinion_future.result()
        for round_id in range(1, rounds + 1):
            sources = [web, *previous.values()]

            def update(i: int) -> tuple[str, str]:
                node = f"agent_{i}_round_{round_id}"
                artifact_id = self.consume_and_run(
                    node,
                    "debater",
                    sources,
                    f"Update position {i} after all-gather round {round_id}; return exactly {6 + 2 * i} numbered synthesis statements.",
                    max_tokens=160 + 48 * i,
                )
                return node, artifact_id

            previous = self.parallel(list(range(1, n + 1)), update)
        consensus_context = self.context("consensus_reviewer", "reviewer", [web, *previous.values()], current="Form consensus")
        consensus_output = self.llm("consensus_reviewer", "reviewer", "Form a consensus preserving disagreements and evidence.", consensus_context, parents=[web, *previous.values()], context_consumer=True, max_tokens=192)
        consensus = self.artifact("consensus_reviewer", consensus_output, artifact_type="agent_output", memory_eligible=False)
        final_context = self.context("finalizer", "finalizer", [consensus], current="Produce final")
        final_output = self.llm("finalizer", "finalizer", "Produce the final concise answer.", final_context, parents=[consensus], context_consumer=True, max_tokens=128)
        return self.artifact("finalizer", final_output, artifact_type="agent_output", memory_eligible=False)

    def shared_fanin(self) -> str:
        settings = self.config["workloads"][self.workload]

        def draft(i: int) -> tuple[str, str]:
            node = f"writer_{i}_private_draft"
            out = self.llm(
                node,
                "evidence writer",
                f"Prepare exactly {6 + 4 * i} numbered findings from perspective {i}; each finding must be a complete sentence.",
                "External grounding will be integrated in the writer synthesis stage.",
                parents=[],
                context_consumer=False,
                max_tokens=160 + 96 * i,
            )
            return node, self.artifact(node, out, artifact_type="private_draft", memory_eligible=False)

        with ThreadPoolExecutor(max_workers=2) as executor:
            web_future = executor.submit(self.search)
            draft_future = executor.submit(
                self.parallel,
                list(range(1, int(settings["writers"]) + 1)),
                draft,
            )
            web = web_future.result()
            drafts = draft_future.result()

        def writer(i: int) -> tuple[str, str]:
            node = f"writer_{i}"
            aid = self.consume_and_run(
                node,
                "evidence writer",
                [web, drafts[f"writer_{i}_private_draft"]],
                f"Integrate grounding and draft into exactly {8 + 4 * i} numbered evidence statements.",
                max_tokens=224 + 96 * i,
            )
            return node, aid

        writers = self.parallel(list(range(1, int(settings["writers"]) + 1)), writer)
        sources = list(writers.values())

        def reader(i: int) -> tuple[str, str]:
            node = f"reader_{i}"
            aid = self.consume_and_run(
                node,
                "memory reader",
                sources,
                f"Derive exactly {4 + 2 * i} numbered implications from shared evidence.",
                max_tokens=112 + 32 * i,
            )
            return node, aid

        readers = self.parallel(list(range(1, int(settings["readers"]) + 1)), reader)
        review_context = self.context("memory_reviewer", "reviewer", list(readers.values()), current="Review implications")
        review_output = self.llm("memory_reviewer", "reviewer", "Review implications, resolve contradictions, retain evidence.", review_context, parents=list(readers.values()), context_consumer=True, max_tokens=160)
        review = self.artifact("memory_reviewer", review_output, artifact_type="agent_output", memory_eligible=False)
        final_context = self.context("finalizer", "finalizer", [review], current="Produce final")
        final_output = self.llm("finalizer", "finalizer", "Produce the final concise answer.", final_context, parents=[review], context_consumer=True, max_tokens=128)
        return self.artifact("finalizer", final_output, artifact_type="agent_output", memory_eligible=False)

    def hierarchical(self) -> str:
        settings = self.config["workloads"][self.workload]
        groups = int(settings["groups"])
        per_group = int(settings["agents_per_group"])

        def researcher(item: tuple[int, int]) -> tuple[str, str]:
            g, i = item
            node = f"group_{g}_agent_{i}"
            out = self.llm(
                node,
                "researcher",
                f"Develop group {g} perspective {i} as exactly {6 + 2 * (i + g)} numbered evidence statements.",
                "Grounding evidence will be integrated downstream.",
                parents=[],
                context_consumer=False,
                max_tokens=176 + 64 * (i + g - 1),
            )
            return node, self.artifact(node, out, artifact_type="group_evidence")

        items = [(g, i) for g in range(1, groups + 1) for i in range(1, per_group + 1)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            web_future = executor.submit(self.search)
            agents_future = executor.submit(self.parallel, items, researcher)
            web = web_future.result()
            all_agents = agents_future.result()

        def group(g: int) -> tuple[str, str]:
            agents = {
                node: artifact_id
                for node, artifact_id in all_agents.items()
                if node.startswith(f"group_{g}_")
            }
            node = f"group_{g}_synth"
            aid = self.consume_and_run(node, "group synthesizer", [web, *agents.values()], f"Synthesize group {g} evidence.", max_tokens=192)
            return node, aid

        group_outputs = self.parallel(list(range(1, groups + 1)), group)
        review_context = self.context("cross_group_reviewer", "cross-group reviewer", list(group_outputs.values()), current="Compare groups")
        review_output = self.llm("cross_group_reviewer", "cross-group reviewer", "Compare group summaries and select robust evidence.", review_context, parents=list(group_outputs.values()), context_consumer=True, max_tokens=160)
        review = self.artifact("cross_group_reviewer", review_output, artifact_type="agent_output", memory_eligible=False)
        final_context = self.context("global_synthesizer", "finalizer", [review], current="Produce final")
        final_output = self.llm("global_synthesizer", "finalizer", "Produce the final concise synthesis.", final_context, parents=[review], context_consumer=True, max_tokens=128)
        return self.artifact("global_synthesizer", final_output, artifact_type="agent_output", memory_eligible=False)

    def run(self) -> dict[str, Any]:
        self.start_ts = time.time()
        self.trace.emit("workflow_start", start_ts=self.start_ts, task_id=self.task_id, graph_definition=self.workload)
        if self.workload == "debate_allgather_pressure_meso":
            final_id = self.debate()
        elif self.workload == "shared_memory_fanin_meso":
            final_id = self.shared_fanin()
        else:
            final_id = self.hierarchical()
        result_ts = time.time()
        self.trace.emit("workflow_result_ready", result_ts=result_ts, result_artifact_id=final_id)
        if self.memory_agent is not None:
            self.memory_agent.wait_all()
            assert self.memory_store is not None
            self.memory_store.freeze()
            self.memory_agent.close()
        end_ts = time.time()
        self.trace.emit("workflow_end", result_ts=result_ts, end_ts=end_ts, result_latency_sec=result_ts - self.start_ts, drain_latency_sec=end_ts - self.start_ts)
        summary = summarize(self.run_dir / "trace.jsonl")
        summary["final_answer"] = self.artifacts[final_id]["content"]
        summary["task_consistency"] = task_consistency_score(summary["final_answer"])
        summary["task_consistency_definition"] = "lexical_required_concept_coverage"
        (self.run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary


def summarize(trace_path: Path) -> dict[str, Any]:
    events = read_jsonl(trace_path)
    llms = [e for e in events if e["event_type"] == "llm_request_end"]
    consumers = [e for e in llms if e.get("context_consumer")]
    created = {e["artifact_id"]: e for e in events if e["event_type"] == "artifact_created"}
    raw_edges = [e for e in events if e["event_type"] == "context_propagation"]
    retrievals = [e for e in events if e["event_type"] == "memory_retrieval"]
    direct_tokens = sum(int(e["propagated_tokens"]) for e in raw_edges)
    mode = events[0]["mode"]
    raw_actual = direct_tokens if mode == "raw" else 0
    retrieval_tokens = sum(int(e["retrieval_tokens"]) for e in retrievals)
    raw_equivalent = direct_tokens + sum(int(e["raw_equivalent_tokens"]) for e in retrievals)
    consumed_ids = {e["source_artifact_id"] for e in raw_edges}
    consumed_ids.update(source for e in retrievals for source in e["source_artifact_ids"])
    unique_tokens = sum(int(created[source]["token_count"]) for source in consumed_ids)
    actual_tokens = direct_tokens + retrieval_tokens
    memory_ends = [e for e in events if e["event_type"] == "memory_construction_end"]
    activities = [(float(e["start_ts"]), float(e["end_ts"])) for e in events if e["event_type"] == "graph_activity_end"]
    wait_intervals = [
        (float(e["start_ts"]), float(e["end_ts"]))
        for e in events
        if e["event_type"] == "memory_consumer_wait" and float(e["wait_sec"]) > 0
    ]
    memory_duration = sum(float(e["duration_sec"]) for e in memory_ends)
    memory_overlap = sum(overlap_duration((float(e["start_ts"]), float(e["end_ts"])), activities) for e in memory_ends)
    expected = {(e["consumer"], source) for e in retrievals for source in e["source_artifact_ids"]}
    covered = {(e["consumer"], source) for e in retrievals for source in e["retrieved_source_artifact_ids"]}
    if raw_edges:
        evidence_coverage = 1.0
    else:
        evidence_coverage = len(covered) / len(expected) if expected else 1.0
    workflow_end = next(e for e in reversed(events) if e["event_type"] == "workflow_end")
    per_artifact_consumers: dict[str, set[str]] = {}
    for e in raw_edges:
        per_artifact_consumers.setdefault(e["source_artifact_id"], set()).add(e["consumer"])
    for e in retrievals:
        for source in e["source_artifact_ids"]:
            per_artifact_consumers.setdefault(source, set()).add(e["consumer"])
    return {
        "schema_version": "masbench_case_study2.summary.v1",
        "run_id": events[0]["run_id"],
        "workload": events[0]["workload"],
        "mode": events[0]["mode"],
        "execution": {"main_llm": "real_vllm_sse", "tool": "live_tavily", "memory_llm": "real_vllm_sse" if memory_ends else "not_used"},
        "result_latency_sec": float(workflow_end["result_latency_sec"]),
        "drain_latency_sec": float(workflow_end["drain_latency_sec"]),
        "consumer_input_tokens": sum(int(e["input_tokens"]) for e in consumers),
        "consumer_ttft_median_sec": statistics.median([float(e["ttft_sec"]) for e in consumers if e.get("ttft_sec") is not None]),
        "consumer_ttft_p95_sec": percentile([float(e["ttft_sec"]) for e in consumers if e.get("ttft_sec") is not None], 0.95),
        "consumer_latency_median_sec": statistics.median([float(e["request_latency_sec"]) for e in consumers]),
        "consumer_latency_p95_sec": percentile([float(e["request_latency_sec"]) for e in consumers], 0.95),
        "raw_propagated_tokens": raw_actual,
        "direct_passthrough_tokens": direct_tokens if mode == "structured" else 0,
        "raw_equivalent_tokens": raw_equivalent,
        "unique_artifact_tokens": unique_tokens,
        "repeated_context_tokens": max(0, raw_equivalent - unique_tokens),
        "actual_context_tokens": actual_tokens,
        "duplication_ratio": raw_equivalent / unique_tokens if unique_tokens else 0.0,
        "actual_amplification_ratio": actual_tokens / unique_tokens if unique_tokens else 0.0,
        "structured_memory_tokens": sum(int(e["structured_tokens"]) for e in memory_ends),
        "retrieval_tokens": retrieval_tokens,
        "compression_ratio": (sum(int(e["source_tokens"]) for e in memory_ends) / sum(int(e["structured_tokens"]) for e in memory_ends)) if memory_ends and sum(int(e["structured_tokens"]) for e in memory_ends) else None,
        "memory_wait_sec": sum(float(e["wait_sec"]) for e in events if e["event_type"] == "memory_consumer_wait"),
        "memory_wait_wall_sec": union_duration(wait_intervals),
        "memory_construction_overlap_ratio": memory_overlap / memory_duration if memory_duration else None,
        "evidence_source_coverage": evidence_coverage,
        "artifact_consumer_count": {source: len(consumers_) for source, consumers_ in sorted(per_artifact_consumers.items())},
        "tool_latency_sec": sum(float(e["latency_sec"]) for e in events if e["event_type"] == "tool_return"),
        "main_request_count": len(llms),
        "memory_request_count": len([e for e in events if e["event_type"] == "memory_llm_end"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--mode", choices=["raw", "structured"], required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).with_name("results") / "runs")
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY is required; no synthetic fallback is permitted")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = args.output_root / args.workload / args.mode / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = {**config, "selected_workload": args.workload, "selected_mode": args.mode, "run_id": run_id}
    (run_dir / "config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    workflow = GraphMemoryWorkflow(config, workload=args.workload, mode=args.mode, run_dir=run_dir, run_id=run_id)
    print(json.dumps(workflow.run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

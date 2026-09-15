from __future__ import annotations

import argparse
import json
import os
import re
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
from mas_workflow.app.search_providers import SearchResult, TavilySearchProvider

from case_study2.memory import TaskMemoryStore, pairwise_distinctiveness, terms
from case_study2.trace import TraceWriter, read_jsonl


WORKLOADS = {
    "single_agent_control",
    "independent_fanin",
    "centralized_manager_worker",
    "debate_allgather_pressure_meso",
    "shared_memory_fanin_meso",
    "retry_debug_loop",
    "hierarchical_synthesis_pressure_meso",
    "issue_to_patch_workflow",
}

QUALITY_DIMENSIONS = {
    "single_agent_control": [
        ("graph",), ("context",), ("prefill", "token"), ("evidence",),
    ],
    "independent_fanin": [
        ("fan-in", "aggregation"), ("context",), ("prefill", "token"), ("evidence",),
    ],
    "centralized_manager_worker": [
        ("manager",), ("worker",), ("context",), ("dependency", "coordination"),
    ],
    "debate_allgather_pressure_meso": [
        ("serving", "prefill", "batching", "decode"),
        ("dependency", "fan-out", "critical", "topology"),
        ("structured", "retrieval", "cache", "reuse"),
        ("faithfulness", "coverage", "provenance"),
    ],
    "shared_memory_fanin_meso": [
        ("latency", "ttft", "prefill"),
        ("token", "amplification", "fan-out", "reuse"),
        ("provenance", "coverage", "evidence"),
        ("overhead", "crossover", "tradeoff"),
    ],
    "retry_debug_loop": [
        ("retry", "round"), ("debug",), ("context", "history"), ("prefill", "token"),
    ],
    "hierarchical_synthesis_pressure_meso": [
        ("latency", "prefill"),
        ("provenance", "evidence"),
        ("context", "reuse"),
        ("compression", "faithfulness"),
    ],
    "issue_to_patch_workflow": [
        ("separability", "matrix"), ("compound", "nested"), ("patch", "fix"), ("test", "regression"),
    ],
}


def task_consistency_score(workload: str, text: str) -> float:
    if workload == "issue_to_patch_workflow":
        return workload_quality_score(workload, text)
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


def workload_quality_score(workload: str, text: str) -> float:
    lowered = text.lower()
    dimensions = QUALITY_DIMENSIONS[workload]
    return sum(any(term in lowered for term in alternatives) for alternatives in dimensions) / len(dimensions)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


class GraphMemoryWorkflow:
    def __init__(self, config: dict[str, Any], *, workload: str, mode: str, run_dir: Path, run_id: str) -> None:
        if workload not in WORKLOADS or mode not in {"raw", "producer"}:
            raise ValueError(f"unsupported workload/mode: {workload}/{mode}")
        self.config = config
        self.workload = workload
        self.mode = mode
        self.run_dir = run_dir
        self.run_id = run_id
        self.task_id = f"{workload}_{run_id}"
        self.objective = str(config.get("workload_objectives", {}).get(workload) or config["objective"])
        self.trace = TraceWriter(run_dir / "trace.jsonl", run_id=run_id, workload=workload, mode=mode)
        self.tokenizer = AutoTokenizer.from_pretrained(config["main_model"]["path"], trust_remote_code=True)
        self.main = LocalLLMClient(
            model=config["main_model"]["served_name"],
            base_url=config["main_model"]["base_url"],
            temperature=0.0,
            max_tokens=int(config["main_model"]["max_tokens"]),
        )
        self.memory_store: TaskMemoryStore | None = None
        self._producer_records: dict[str, dict[str, Any]] = {}
        self._producer_record_lock = threading.Lock()
        if mode == "producer":
            self.memory_store = TaskMemoryStore(run_dir / "memory", self.task_id, self.trace)
        self.artifacts: dict[str, dict[str, Any]] = {}
        self._artifact_index = 0
        self._artifact_lock = threading.Lock()
        self.start_ts = 0.0

    def tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def policy(
        self,
        *,
        reason: str,
        focus: str,
        focus_terms: list[str],
        expected_consumers: int = 1,
        fanin_width: int = 1,
        multi_round_reuse: bool = False,
    ) -> dict[str, Any]:
        threshold = int(self.config["producer_memory"]["graph_degree_threshold"])
        enabled = (
            expected_consumers >= threshold
            or fanin_width >= threshold
            or multi_round_reuse
        )
        return {
            "enabled": enabled,
            "reason": reason if enabled else "below_graph_benefit_threshold",
            "focus": focus,
            "focus_terms": focus_terms,
            "expected_consumers": expected_consumers,
            "fanin_width": fanin_width,
            "multi_round_reuse": multi_round_reuse,
            "threshold": threshold,
        }

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
        policy: dict[str, Any] | None = None,
    ) -> str:
        decision = policy or self.policy(
            reason="direct_handoff",
            focus=source_agent,
            focus_terms=[source_agent],
        )
        memory_eligible = bool(decision["enabled"])
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
                "graph_reason": decision["reason"],
                "expected_consumers": decision["expected_consumers"],
                "fanin_width": decision["fanin_width"],
                "multi_round_reuse": decision["multi_round_reuse"],
            }
            self.artifacts[artifact_id] = row
        self.trace.emit("artifact_created", **{key: value for key, value in row.items() if key != "content"})
        self.trace.emit(
            "memory_policy_decision",
            source_agent=source_agent,
            source_artifact_id=artifact_id,
            eligible=memory_eligible,
            materialized=self.mode == "producer" and memory_eligible,
            policy="graph_aware_producer_materialization",
            **decision,
        )
        if self.mode == "producer" and memory_eligible:
            assert self.memory_store is not None
            with self._producer_record_lock:
                materialized = self._producer_records.pop(source_agent, None)
            if materialized is None:
                raise RuntimeError(f"producer-side memory missing for eligible artifact from {source_agent}")
            record = dict(materialized["record"])
            refs = [str(ref) for ref in record.get("evidence_refs", []) if str(ref)]
            if artifact_id not in refs:
                refs.insert(0, artifact_id)
            record["evidence_refs"] = refs[:8]
            record["token_count"] = self.tokens(record["content"])
            appended = self.memory_store.append(row, [record])
            self.trace.emit(
                "producer_memory_materialized",
                source_artifact_id=artifact_id,
                source_agent=source_agent,
                source_tokens=row["token_count"],
                structured_tokens=sum(item["token_count"] for item in appended),
                producer_completion_tokens=materialized["completion_tokens"],
                answer_tokens=materialized["answer_tokens"],
                answer_words=materialized["answer_words"],
                extra_output_tokens=max(0, materialized["completion_tokens"] - materialized["answer_tokens"]),
                materialization="same_generation",
                graph_reason=decision["reason"],
                expected_consumers=decision["expected_consumers"],
                fanin_width=decision["fanin_width"],
                grounding_ratio=materialized["grounding_ratio"],
                focus_match=materialized["focus_match"],
                memory_terms=materialized["memory_terms"],
            )
        return artifact_id

    def search(self) -> str:
        query = self.config["search_queries"][self.workload]
        start = time.time()
        replay_root = self.config["tavily"].get("replay_snapshot_root")
        if replay_root:
            candidates = sorted((Path(replay_root) / self.workload / "raw").glob("*/tavily_snapshot.json"))
            if not candidates:
                raise FileNotFoundError(f"no frozen Tavily snapshot for {self.workload} under {replay_root}")
            snapshot_source = candidates[0]

            def call() -> Any:
                frozen = json.loads(snapshot_source.read_text(encoding="utf-8"))
                results = frozen.get("results") or []
                answer = str(frozen.get("answer") or "")
                return SearchResult(
                    "recorded_tavily",
                    query,
                    results,
                    answer,
                    str(frozen.get("result_hash") or ""),
                    0.0,
                    0.0,
                    0.0,
                    str(snapshot_source),
                )

            activity_kind = "recorded_tavily"
        else:
            provider = TavilySearchProvider(api_key=os.environ["TAVILY_API_KEY"])

            def call() -> Any:
                return provider.search(
                    query,
                    max_results=int(self.config["tavily"]["max_results"]),
                    include_answer=True,
                    include_raw_content=False,
                    search_depth=self.config["tavily"]["search_depth"],
                )

            activity_kind = "live_tavily"

        result = self._activity(activity_kind, "tavily_grounding", call)
        snapshot = {
            "provider": result.provider_name,
            "snapshot_source": result.snapshot_path,
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
            provider=result.provider_name,
            start_ts=start,
            end_ts=time.time(),
            latency_sec=result.measured_duration_sec,
            result_tokens=self.tokens(text),
            result_hash=result.result_hash,
        )
        tool_policy = self.policy(
            reason="shared_tool_fanout",
            focus="external evidence and named systems",
            focus_terms=["evidence", "system", "agent", "memory", "context"],
            expected_consumers={
                "single_agent_control": 1,
                "independent_fanin": 4,
                "centralized_manager_worker": 3,
                "debate_allgather_pressure_meso": 9,
                "shared_memory_fanin_meso": 2,
                "retry_debug_loop": 6,
                "hierarchical_synthesis_pressure_meso": 2,
                "issue_to_patch_workflow": 7,
            }[self.workload],
            multi_round_reuse=self.workload in {
                "debate_allgather_pressure_meso", "retry_debug_loop", "issue_to_patch_workflow"
            },
        )
        if self.mode == "producer" and tool_policy["enabled"]:
            tool_view = (result.answer or "").strip()
            if not tool_view and result.results:
                tool_view = str(result.results[0].get("content") or result.results[0].get("title") or "")
            with self._producer_record_lock:
                self._producer_records["tavily_grounding"] = {
                    "record": {
                        "type": "evidence",
                        "content": tool_view,
                        "keywords": sorted(self.config["search_queries"][self.workload].lower().split())[:8],
                        "evidence_refs": [str(item.get("url")) for item in result.results[:4] if item.get("url")],
                    },
                    "completion_tokens": 0,
                    "answer_tokens": 0,
                    "answer_words": 0,
                    "grounding_ratio": 1.0,
                    "focus_match": True,
                    "memory_terms": sorted(terms(tool_view)),
                }
        return self.artifact("tavily_grounding", text, artifact_type="tool_evidence", policy=tool_policy)

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

        assert self.memory_store is not None
        eligible_ids = [
            artifact_id for artifact_id in artifact_ids if self.artifacts[artifact_id]["memory_eligible"]
        ]
        direct_ids = [
            artifact_id for artifact_id in artifact_ids if not self.artifacts[artifact_id]["memory_eligible"]
        ]
        query = f"role={role}; objective={self.objective}; current context={current}"
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
        memory_policy: dict[str, Any] | None = None,
    ) -> str:
        user = f"Task objective:\n{self.objective}\n\nAvailable context:\n{context}\n\nInstruction:\n{instruction}"
        producer_side = self.mode == "producer" and bool(memory_policy and memory_policy["enabled"])
        requested_max_tokens = max_tokens or int(self.config["main_model"]["max_tokens"])
        min_answer_words = max(24, round(requested_max_tokens * 0.40))
        max_answer_words = max(min_answer_words + 8, round(requested_max_tokens * 0.75))
        memory_instruction = (
            f" Produce the complete requested answer first, preserving the statement count and using "
            f"{min_answer_words}-{max_answer_words} words. Then append exactly one final line in this compact "
            "format: the literal opening tag <MEMORY>, followed by a single-line JSON object with exactly the "
            "keys t, c, and k, followed by the literal closing tag </MEMORY>. Allowed t values are claim, "
            "evidence, constraint, keyword_summary. Write c as a new, answer-specific factual summary of 8-18 "
            f"words focused on {memory_policy['focus']!r}; c must include at least one of these anchor terms "
            f"verbatim: {', '.join(memory_policy['focus_terms'])}. Write k as 1-4 answer-specific keywords. "
            "Capture a concrete mechanism, measurement, named evidence, limitation, or decision—not merely the "
            "task objective. Never copy wording from these format instructions. "
            "Do not add text after </MEMORY> and do not shorten the answer merely because the sidecar is requested."
            if producer_side else ""
        )
        messages = [
            {"role": "system", "content": f"You are the {role} agent. Be evidence-grounded, satisfy the requested detail, and do not invent sources.{memory_instruction} /no_think"},
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
            metadata: dict[str, Any] = {"request_id_for_backend": request_id, "agent_id": node_id, "priority": "normal"}
            return self.main.invoke_messages_streaming_with_metadata(
                messages,
                metadata=metadata,
                max_tokens=requested_max_tokens
                + (int(self.config.get("producer_memory", {}).get("max_extra_tokens", 80)) if producer_side else 0),
                seed=int(self.config["seed"]),
            )

        content, metadata = self._activity("main_llm", node_id, call)
        usage = metadata.get("usage") or {}
        returned_content = content
        if producer_side:
            match = re.search(r"<MEMORY>\s*(\{.*?\})\s*</MEMORY>\s*$", content, flags=re.DOTALL)
            if match is None:
                raise ValueError(f"producer-side memory sidecar missing from {node_id}")
            parsed = json.loads(match.group(1))
            answer = content[: match.start()].strip()
            record = {
                "type": parsed.get("t"),
                "content": parsed.get("c"),
                "keywords": parsed.get("k", []),
                "evidence_refs": [],
            }
            if not answer or not str(record.get("content", "")).strip():
                raise ValueError(f"invalid producer-side dual view from {node_id}")
            record_type = str(record.get("type", "")).strip().lower()
            record_type = {
                "plan": "constraint",
                "decision": "constraint",
                "recommendation": "claim",
                "finding": "evidence",
                "fact": "evidence",
                "summary": "keyword_summary",
            }.get(record_type, record_type)
            if record_type not in {"claim", "evidence", "constraint", "keyword_summary"}:
                record_type = "claim"
            record["type"] = record_type
            memory_words = str(record["content"]).split()
            forbidden = {"at most 18 words", "compact factual summary", "answer-specific factual summary"}
            if not 5 <= len(memory_words) <= 22 or str(record["content"]).strip().lower() in forbidden:
                raise ValueError(f"invalid producer-side memory content from {node_id}: {record['content']!r}")
            answer_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", answer.lower()))
            memory_terms = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(record["content"]).lower()))
            grounded = answer_terms & memory_terms
            if len(grounded) < 2:
                raise ValueError(f"producer-side memory from {node_id} is not grounded in its answer")
            focus_terms = {item.lower() for item in memory_policy["focus_terms"]}
            focus_match = any(
                memory_term == focus_term
                or (len(memory_term) >= 4 and len(focus_term) >= 4 and (
                    memory_term.startswith(focus_term) or focus_term.startswith(memory_term)
                ))
                for memory_term in memory_terms
                for focus_term in focus_terms
            )
            if not focus_match:
                raise ValueError(f"producer-side memory from {node_id} misses its graph-assigned focus")
            objective_only = memory_terms <= terms(self.objective)
            if objective_only:
                raise ValueError(f"producer-side memory from {node_id} only restates the objective")
            answer_words = len(answer.split())
            if answer_words < max(20, min_answer_words - 30):
                raise ValueError(
                    f"producer-side answer from {node_id} is too short: "
                    f"{answer_words} words < {max(20, min_answer_words - 30)}"
                )
            record["content"] = str(record["content"]).strip()
            record["keywords"] = [str(item).strip() for item in record.get("keywords", []) if str(item).strip()][:8]
            record["evidence_refs"] = [str(item).strip() for item in record.get("evidence_refs", []) if str(item).strip()][:8]
            completion_tokens = int(usage.get("completion_tokens") or self.tokens(content))
            with self._producer_record_lock:
                self._producer_records[node_id] = {
                    "record": record,
                    "completion_tokens": completion_tokens,
                    "answer_tokens": self.tokens(answer),
                    "answer_words": answer_words,
                    "grounding_ratio": len(grounded) / len(memory_terms),
                    "focus_match": focus_match,
                    "memory_terms": sorted(memory_terms),
                }
            returned_content = answer
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
            producer_sidecar=producer_side,
            answer_tokens=self.tokens(returned_content),
        )
        return returned_content

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
        output_policy: dict[str, Any] | None = None,
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
            memory_policy=output_policy,
        )
        return self.artifact(node, output, artifact_type="agent_output", policy=output_policy)

    def single_control(self) -> str:
        """Linear negative control: every artifact has exactly one consumer."""
        web = self.search()
        analysis = self.consume_and_run(
            "linear_analyst",
            "analyst",
            [web],
            "Explain graph context, evidence, and prefill cost in exactly eight concise numbered statements.",
            max_tokens=224,
        )
        final = self.consume_and_run(
            "linear_finalizer",
            "finalizer",
            [analysis],
            "Produce four one-sentence bullets covering graph, context, prefill tokens, and evidence.",
            max_tokens=160,
        )
        return final

    def independent_fanin(self) -> str:
        """Independent specialists followed by a four-way synthesis barrier."""
        count = int(self.config["workloads"][self.workload]["num_agents"])
        focuses = {
            1: ("fan-in aggregation", ["fan-in", "aggregation"]),
            2: ("context token growth", ["context", "tokens"]),
            3: ("downstream prefill", ["prefill", "latency"]),
            4: ("evidence preservation", ["evidence", "coverage"]),
        }
        web = self.search()

        def specialist(i: int) -> tuple[str, str]:
            node = f"independent_specialist_{i}"
            focus, anchors = focuses[i]
            policy = self.policy(
                reason="independent_terminal_fanin",
                focus=focus,
                focus_terms=anchors,
                fanin_width=count,
            )
            artifact_id = self.consume_and_run(
                node,
                "independent specialist",
                [web],
                f"Write exactly {5 + i} numbered findings about {focus}; retain {', '.join(anchors)}.",
                max_tokens=144 + 24 * i,
                output_policy=policy,
            )
            return node, artifact_id

        specialists = self.parallel(list(range(1, count + 1)), specialist)
        synthesis = self.consume_and_run(
            "fanin_synthesizer",
            "synthesizer",
            list(specialists.values()),
            "Synthesize the independent branches while preserving fan-in, context, prefill, and evidence findings.",
            max_tokens=224,
        )
        return self.consume_and_run(
            "fanin_finalizer",
            "finalizer",
            [synthesis],
            "Produce four one-sentence bullets covering fan-in aggregation, context growth, prefill tokens, and evidence.",
            max_tokens=160,
        )

    def manager_worker(self) -> str:
        """Centralized manager broadcasts one plan and collects specialized workers."""
        count = int(self.config["workloads"][self.workload]["num_workers"])
        plan_policy = self.policy(
            reason="manager_plan_broadcast",
            focus="manager dependency plan",
            focus_terms=["manager", "dependency", "plan"],
            expected_consumers=count,
        )
        plan_text = self.llm(
            "manager_plan",
            "manager",
            "Create eight numbered manager decisions about worker coordination, dependency, and context sharing.",
            "External evidence will be collected concurrently.",
            parents=[],
            context_consumer=False,
            max_tokens=224,
            memory_policy=plan_policy,
        )
        plan = self.artifact("manager_plan", plan_text, artifact_type="manager_plan", policy=plan_policy)
        web = self.search()
        focuses = {
            1: ("worker context reuse", ["worker", "context", "reuse"]),
            2: ("manager dependency control", ["manager", "dependency", "control"]),
            3: ("coordination overhead", ["coordination", "overhead", "worker"]),
        }

        def worker(i: int) -> tuple[str, str]:
            node = f"manager_worker_{i}"
            focus, anchors = focuses[i]
            policy = self.policy(
                reason="manager_collection_fanin",
                focus=focus,
                focus_terms=anchors,
                fanin_width=count,
            )
            aid = self.consume_and_run(
                node,
                "managed worker",
                [plan, web],
                f"Return exactly {6 + i} numbered observations about {focus}; include {', '.join(anchors)}.",
                max_tokens=176 + 24 * i,
                output_policy=policy,
            )
            return node, aid

        workers = self.parallel(list(range(1, count + 1)), worker)
        collected = self.consume_and_run(
            "manager_collect",
            "manager",
            [plan, *workers.values()],
            "Collect the worker results and preserve manager, worker, context, dependency, and coordination evidence.",
            max_tokens=224,
        )
        return self.consume_and_run(
            "manager_finalizer",
            "finalizer",
            [collected],
            "Produce four one-sentence bullets covering manager control, worker execution, context reuse, and dependency coordination.",
            max_tokens=160,
        )

    def retry_loop(self) -> str:
        """Temporal negative/positive mix with a stable artifact reread each round."""
        rounds = int(self.config["workloads"][self.workload]["rounds"])
        web = self.search()
        initial_policy = self.policy(
            reason="loop_carried_state",
            focus="retry history and stable context",
            focus_terms=["retry", "history", "context"],
            expected_consumers=2 * rounds,
            multi_round_reuse=True,
        )
        initial_text = self.consume_and_run(
            "initial_attempt",
            "generator",
            [web],
            "Create an initial eight-step analysis using retry, debug, context, history, prefill, and token concepts.",
            max_tokens=224,
            output_policy=initial_policy,
        )
        current = initial_text
        for round_id in range(1, rounds + 1):
            review = self.consume_and_run(
                f"review_round_{round_id}",
                "reviewer",
                [web, current],
                f"Review retry round {round_id}; identify debug evidence and repeated context carried from history.",
                max_tokens=176,
            )
            next_policy = self.policy(
                reason="loop_carried_state",
                focus="debug revision and repeated prefill",
                focus_terms=["debug", "revision", "prefill"],
                expected_consumers=2 if round_id < rounds else 1,
                multi_round_reuse=round_id < rounds,
            )
            current = self.consume_and_run(
                f"debug_round_{round_id}",
                "debugger",
                [web, current, review],
                f"Produce retry round {round_id} revision with explicit debug, history, context, and prefill implications.",
                max_tokens=208,
                output_policy=next_policy,
            )
        return self.consume_and_run(
            "retry_finalizer",
            "finalizer",
            [web, current],
            "Produce four one-sentence bullets covering retry rounds, debugging, repeated context/history, and prefill tokens.",
            max_tokens=176,
        )

    def issue_to_patch(self) -> str:
        """Full multi-stage issue-to-patch graph with live external evidence."""
        settings = self.config["workloads"][self.workload]
        evidence_count = int(settings["evidence_agents"])
        diagnosis_count = int(settings["diagnosis_agents"])
        patch_count = int(settings["patch_agents"])
        web = self.search()
        plan_policy = self.policy(
            reason="full_workflow_plan_broadcast",
            focus="nested CompoundModel investigation plan",
            focus_terms=["nested", "CompoundModel", "plan"],
            expected_consumers=evidence_count,
        )
        plan_text = self.llm(
            "issue_manager_plan",
            "software manager",
            "Plan diagnosis of nested CompoundModel separability_matrix behavior, likely code locations, patch criteria, and regression tests.",
            "Use the fixed Astropy issue objective.",
            parents=[],
            context_consumer=False,
            max_tokens=224,
            memory_policy=plan_policy,
        )
        plan = self.artifact("issue_manager_plan", plan_text, artifact_type="plan", policy=plan_policy)
        evidence_focuses = {
            1: ("nested CompoundModel behavior", ["nested", "CompoundModel"]),
            2: ("separability matrix implementation", ["separability", "matrix"]),
            3: ("regression test design", ["regression", "test"]),
        }

        def evidence_agent(i: int) -> tuple[str, str]:
            node = f"issue_evidence_{i}"
            focus, anchors = evidence_focuses[i]
            policy = self.policy(
                reason="full_workflow_evidence_fanin",
                focus=focus,
                focus_terms=anchors,
                fanin_width=evidence_count,
            )
            aid = self.consume_and_run(
                node,
                "software evidence analyst",
                [plan, web],
                f"Extract eight concrete findings about {focus}; retain {', '.join(anchors)} and avoid invented test results.",
                max_tokens=224,
                output_policy=policy,
            )
            return node, aid

        evidence = self.parallel(list(range(1, evidence_count + 1)), evidence_agent)
        synthesis_policy = self.policy(
            reason="full_workflow_cross_stage_reuse",
            focus="evidence synthesis for nested separability",
            focus_terms=["evidence", "nested", "separability"],
            expected_consumers=diagnosis_count + 2,
            multi_round_reuse=True,
        )
        synthesis = self.consume_and_run(
            "issue_evidence_synthesis",
            "evidence synthesizer",
            list(evidence.values()),
            "Synthesize evidence about nested CompoundModel separability, matrix construction, likely root cause, and regression coverage.",
            max_tokens=256,
            output_policy=synthesis_policy,
        )
        diagnosis_focuses = {
            1: ("root cause", ["root", "cause", "separability"]),
            2: ("minimal patch", ["minimal", "patch", "matrix"]),
            3: ("regression risk", ["regression", "risk", "nested"]),
        }

        def diagnose(i: int) -> tuple[str, str]:
            node = f"issue_diagnosis_{i}"
            focus, anchors = diagnosis_focuses[i]
            policy = self.policy(
                reason="full_workflow_diagnosis_fanin",
                focus=focus,
                focus_terms=anchors,
                fanin_width=diagnosis_count,
            )
            aid = self.consume_and_run(
                node,
                "software diagnosis agent",
                [synthesis],
                f"Argue the {focus} hypothesis in eight numbered statements; include {', '.join(anchors)}.",
                max_tokens=208,
                output_policy=policy,
            )
            return node, aid

        diagnoses = self.parallel(list(range(1, diagnosis_count + 1)), diagnose)
        consensus_policy = self.policy(
            reason="full_workflow_patch_fanout",
            focus="consensus patch constraints",
            focus_terms=["consensus", "patch", "constraints"],
            expected_consumers=patch_count,
            fanin_width=diagnosis_count,
        )
        consensus = self.consume_and_run(
            "issue_diagnosis_consensus",
            "diagnosis reviewer",
            list(diagnoses.values()),
            "Form a consensus root cause and explicit patch constraints for nested separability_matrix behavior.",
            max_tokens=224,
            output_policy=consensus_policy,
        )

        def coder(i: int) -> tuple[str, str]:
            node = f"issue_patch_candidate_{i}"
            policy = self.policy(
                reason="full_workflow_patch_selection_fanin",
                focus="candidate patch and regression test",
                focus_terms=["patch", "regression", "test"],
                fanin_width=patch_count,
            )
            aid = self.consume_and_run(
                node,
                "patch author",
                [synthesis, consensus],
                f"Propose candidate patch {i} with pseudocode and regression tests for nested CompoundModels; do not claim tests ran.",
                max_tokens=240,
                output_policy=policy,
            )
            return node, aid

        candidates = self.parallel(list(range(1, patch_count + 1)), coder)
        selected_policy = self.policy(
            reason="full_workflow_selected_patch_reuse",
            focus="selected patch and verification plan",
            focus_terms=["selected", "patch", "verification"],
            expected_consumers=3,
            multi_round_reuse=True,
        )
        selected = self.consume_and_run(
            "issue_patch_selector",
            "patch selector",
            list(candidates.values()),
            "Select one patch approach and provide a verification plan; distinguish proposed tests from executed tests.",
            max_tokens=240,
            output_policy=selected_policy,
        )
        review = self.consume_and_run(
            "issue_test_reviewer",
            "test reviewer",
            [synthesis, selected],
            "Review the selected patch against nested CompoundModel examples and specify regression assertions; do not fabricate execution.",
            max_tokens=208,
        )
        debug = self.consume_and_run(
            "issue_debugger",
            "debugger",
            [synthesis, selected, review],
            "Revise the patch design using evidence and review feedback; preserve separability matrix shape and nested behavior.",
            max_tokens=224,
        )
        return self.consume_and_run(
            "issue_final_report",
            "software finalizer",
            [synthesis, selected, debug],
            "Produce exactly four one-sentence bullets covering separability matrix root cause, nested CompoundModel behavior, proposed patch, and regression tests; state that tests are proposed, not executed.",
            max_tokens=192,
        )

    def debate(self) -> str:
        n = int(self.config["workloads"][self.workload]["num_agents"])
        rounds = int(self.config["workloads"][self.workload]["rounds"])
        focuses = {
            1: ("serving interference and batching", ["prefill", "batching", "decode"]),
            2: ("workflow topology and dependency", ["dependency", "fan-out", "critical"]),
            3: ("structured representation and reuse", ["structured", "retrieval", "cache"]),
            4: ("faithfulness and provenance", ["faithfulness", "coverage", "provenance"]),
        }

        def opinion(i: int) -> tuple[str, str]:
            node = f"opinion_{i}"
            focus, anchors = focuses[i]
            policy = self.policy(
                reason="all_gather_fanout",
                focus=focus,
                focus_terms=anchors,
                expected_consumers=n,
                multi_round_reuse=True,
            )
            out = self.llm(
                node,
                "peer",
                f"Develop exactly {6 + 2 * i} numbered, self-contained statements about {focus}. "
                f"Use the anchor concepts {', '.join(anchors)} and include concrete mechanisms or limitations.",
                "Grounding evidence will be integrated downstream.",
                parents=[],
                context_consumer=False,
                max_tokens=160 + 64 * i,
                memory_policy=policy,
            )
            return node, self.artifact(node, out, artifact_type="opinion", policy=policy)

        with ThreadPoolExecutor(max_workers=2) as executor:
            web_future = executor.submit(self.search)
            opinion_future = executor.submit(self.parallel, list(range(1, n + 1)), opinion)
            web = web_future.result()
            previous = opinion_future.result()
        for round_id in range(1, rounds + 1):
            sources = [web, *previous.values()]

            def update(i: int) -> tuple[str, str]:
                node = f"agent_{i}_round_{round_id}"
                focus, anchors = focuses[i]
                policy = self.policy(
                    reason="multi_round_all_gather" if round_id < rounds else "all_gather_terminal_fanin",
                    focus=focus,
                    focus_terms=anchors,
                    expected_consumers=n if round_id < rounds else 1,
                    fanin_width=1 if round_id < rounds else n + 1,
                    multi_round_reuse=round_id < rounds,
                )
                artifact_id = self.consume_and_run(
                    node,
                    "debater",
                    sources,
                    f"Update the {focus} position after all-gather round {round_id}; return exactly "
                    f"{6 + 2 * i} numbered statements, retain the anchors {', '.join(anchors)}, and cite "
                    "specific agreements or conflicts from peer evidence.",
                    max_tokens=160 + 48 * i,
                    output_policy=policy,
                )
                return node, artifact_id

            previous = self.parallel(list(range(1, n + 1)), update)
        consensus_context = self.context("consensus_reviewer", "reviewer", [web, *previous.values()], current="Form consensus")
        consensus_output = self.llm("consensus_reviewer", "reviewer", "Form a consensus that explicitly preserves serving, topology, representation, and faithfulness findings.", consensus_context, parents=[web, *previous.values()], context_consumer=True, max_tokens=224)
        consensus = self.artifact("consensus_reviewer", consensus_output, artifact_type="agent_output")
        final_context = self.context("finalizer", "finalizer", [consensus], current="Produce final")
        final_output = self.llm("finalizer", "finalizer", "Produce exactly four one-sentence bullets covering serving, graph topology, structured reuse, and faithfulness; use at most 140 words total and no preamble.", final_context, parents=[consensus], context_consumer=True, max_tokens=224)
        return self.artifact("finalizer", final_output, artifact_type="agent_output")

    def shared_fanin(self) -> str:
        settings = self.config["workloads"][self.workload]
        writer_focuses = {
            1: ("measured token amplification", ["tokens", "amplification", "prefill"]),
            2: ("evidence preservation tradeoffs", ["evidence", "faithfulness", "coverage"]),
        }
        reader_focuses = {
            1: "latency and TTFT",
            2: "graph fan-out and reuse",
            3: "source provenance and coverage",
            4: "deployment overhead and crossover",
        }

        def draft(i: int) -> tuple[str, str]:
            node = f"writer_{i}_private_draft"
            focus, anchors = writer_focuses[i]
            policy = self.policy(
                reason="writer_fanin_compaction",
                focus=focus,
                focus_terms=anchors,
                fanin_width=2,
            )
            out = self.llm(
                node,
                "evidence writer",
                f"Prepare exactly {6 + 4 * i} numbered findings about {focus}; use the anchors "
                f"{', '.join(anchors)} and make every finding a complete sentence.",
                "External grounding will be integrated in the writer synthesis stage.",
                parents=[],
                context_consumer=False,
                max_tokens=160 + 96 * i,
                memory_policy=policy,
            )
            return node, self.artifact(node, out, artifact_type="private_draft", policy=policy)

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
            focus, anchors = writer_focuses[i]
            policy = self.policy(
                reason="shared_memory_fanout",
                focus=focus,
                focus_terms=anchors,
                expected_consumers=int(settings["readers"]),
            )
            aid = self.consume_and_run(
                node,
                "evidence writer",
                [web, drafts[f"writer_{i}_private_draft"]],
                f"Integrate grounding and draft into exactly {8 + 4 * i} numbered statements about {focus}; "
                f"retain the anchors {', '.join(anchors)} and concrete evidence.",
                max_tokens=224 + 96 * i,
                output_policy=policy,
            )
            return node, aid

        writers = self.parallel(list(range(1, int(settings["writers"]) + 1)), writer)
        sources = list(writers.values())

        def reader(i: int) -> tuple[str, str]:
            node = f"reader_{i}"
            focus = reader_focuses[i]
            anchors = {
                1: ["latency", "TTFT"],
                2: ["fan-out", "reuse"],
                3: ["provenance", "coverage"],
                4: ["overhead", "crossover"],
            }[i]
            policy = self.policy(
                reason="reader_fanin_compaction",
                focus=focus,
                focus_terms=anchors,
                fanin_width=int(settings["readers"]),
            )
            aid = self.consume_and_run(
                node,
                "memory reader",
                sources,
                f"Derive exactly {4 + 2 * i} numbered implications about {focus}; include "
                f"{', '.join(anchors)} and trace claims to shared evidence.",
                max_tokens=112 + 32 * i,
                output_policy=policy,
            )
            return node, aid

        readers = self.parallel(list(range(1, int(settings["readers"]) + 1)), reader)
        review_context = self.context("memory_reviewer", "reviewer", list(readers.values()), current="Review implications")
        review_output = self.llm("memory_reviewer", "reviewer", "Review implications and retain distinct latency, reuse, provenance, and crossover evidence.", review_context, parents=list(readers.values()), context_consumer=True, max_tokens=224)
        review = self.artifact("memory_reviewer", review_output, artifact_type="agent_output")
        final_context = self.context("finalizer", "finalizer", [review], current="Produce final")
        final_output = self.llm("finalizer", "finalizer", "Produce exactly four one-sentence bullets covering latency, token amplification, provenance, and deployment crossover; use at most 140 words total and no preamble.", final_context, parents=[review], context_consumer=True, max_tokens=224)
        return self.artifact("finalizer", final_output, artifact_type="agent_output")

    def hierarchical(self) -> str:
        settings = self.config["workloads"][self.workload]
        groups = int(settings["groups"])
        per_group = int(settings["agents_per_group"])
        focuses = {
            (1, 1): ("local prefill cost", ["prefill", "latency"]),
            (1, 2): ("local evidence provenance", ["evidence", "provenance"]),
            (2, 1): ("cross-group context reuse", ["context", "reuse"]),
            (2, 2): ("semantic compression risk", ["compression", "faithfulness"]),
        }

        def researcher(item: tuple[int, int]) -> tuple[str, str]:
            g, i = item
            node = f"group_{g}_agent_{i}"
            focus, anchors = focuses[(g, i)]
            policy = self.policy(
                reason="hierarchical_local_fanin",
                focus=focus,
                focus_terms=anchors,
                fanin_width=per_group + 1,
            )
            out = self.llm(
                node,
                "researcher",
                f"Develop exactly {6 + 2 * (i + g)} numbered statements about {focus}; include the anchors "
                f"{', '.join(anchors)} and concrete mechanisms or risks.",
                "Grounding evidence will be integrated downstream.",
                parents=[],
                context_consumer=False,
                max_tokens=176 + 64 * (i + g - 1),
                memory_policy=policy,
            )
            return node, self.artifact(node, out, artifact_type="group_evidence", policy=policy)

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
            focus = "runtime and provenance synthesis" if g == 1 else "reuse and faithfulness synthesis"
            anchors = ["latency", "provenance"] if g == 1 else ["reuse", "faithfulness"]
            policy = self.policy(
                reason="hierarchical_global_fanin",
                focus=focus,
                focus_terms=anchors,
                fanin_width=groups,
            )
            aid = self.consume_and_run(
                node,
                "group synthesizer",
                [web, *agents.values()],
                f"Synthesize group {g} evidence about {focus}; explicitly retain {', '.join(anchors)}.",
                max_tokens=224,
                output_policy=policy,
            )
            return node, aid

        group_outputs = self.parallel(list(range(1, groups + 1)), group)
        review_context = self.context("cross_group_reviewer", "cross-group reviewer", list(group_outputs.values()), current="Compare groups")
        review_output = self.llm("cross_group_reviewer", "cross-group reviewer", "Compare groups and preserve latency, provenance, reuse, and faithfulness evidence.", review_context, parents=list(group_outputs.values()), context_consumer=True, max_tokens=224)
        review = self.artifact("cross_group_reviewer", review_output, artifact_type="agent_output")
        final_context = self.context("global_synthesizer", "finalizer", [review], current="Produce final")
        final_output = self.llm("global_synthesizer", "finalizer", "Produce exactly four one-sentence bullets covering latency, provenance, reuse, and faithfulness; use at most 140 words total and no preamble.", final_context, parents=[review], context_consumer=True, max_tokens=224)
        return self.artifact("global_synthesizer", final_output, artifact_type="agent_output")

    def run(self) -> dict[str, Any]:
        self.start_ts = time.time()
        self.trace.emit("workflow_start", start_ts=self.start_ts, task_id=self.task_id, graph_definition=self.workload)
        if self.workload == "single_agent_control":
            final_id = self.single_control()
        elif self.workload == "independent_fanin":
            final_id = self.independent_fanin()
        elif self.workload == "centralized_manager_worker":
            final_id = self.manager_worker()
        elif self.workload == "debate_allgather_pressure_meso":
            final_id = self.debate()
        elif self.workload == "shared_memory_fanin_meso":
            final_id = self.shared_fanin()
        elif self.workload == "retry_debug_loop":
            final_id = self.retry_loop()
        elif self.workload == "hierarchical_synthesis_pressure_meso":
            final_id = self.hierarchical()
        else:
            final_id = self.issue_to_patch()
        result_ts = time.time()
        self.trace.emit("workflow_result_ready", result_ts=result_ts, result_artifact_id=final_id)
        if self.memory_store is not None:
            self.memory_store.freeze()
        end_ts = time.time()
        self.trace.emit("workflow_end", result_ts=result_ts, end_ts=end_ts, result_latency_sec=result_ts - self.start_ts, drain_latency_sec=end_ts - self.start_ts)
        summary = summarize(self.run_dir / "trace.jsonl")
        summary["final_answer"] = self.artifacts[final_id]["content"]
        summary["task_consistency"] = task_consistency_score(self.workload, summary["final_answer"])
        summary["task_consistency_definition"] = "lexical_required_concept_coverage"
        summary["workload_quality_score"] = workload_quality_score(self.workload, summary["final_answer"])
        summary["workload_quality_definition"] = "workload_specific_dimension_coverage"
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
    producer_materializations = [e for e in events if e["event_type"] == "producer_memory_materialized"]
    llm_materializations = [e for e in producer_materializations if int(e["producer_completion_tokens"]) > 0]
    policy_decisions = [e for e in events if e["event_type"] == "memory_policy_decision"]
    expected = {(e["consumer"], source) for e in retrievals for source in e["source_artifact_ids"]}
    covered = {(e["consumer"], source) for e in retrievals for source in e["retrieved_source_artifact_ids"]}
    if raw_edges:
        evidence_coverage = 1.0
    else:
        evidence_coverage = len(covered) / len(expected) if expected else 1.0
    workflow_end = next(e for e in reversed(events) if e["event_type"] == "workflow_end")
    tool_events = [e for e in events if e["event_type"] == "tool_return"]
    per_artifact_consumers: dict[str, set[str]] = {}
    for e in raw_edges:
        per_artifact_consumers.setdefault(e["source_artifact_id"], set()).add(e["consumer"])
    for e in retrievals:
        for source in e["source_artifact_ids"]:
            per_artifact_consumers.setdefault(source, set()).add(e["consumer"])
    return {
        "schema_version": "masbench_case_study2.summary.v2",
        "run_id": events[0]["run_id"],
        "workload": events[0]["workload"],
        "mode": events[0]["mode"],
        "execution": {
            "main_llm": "real_vllm_sse",
            "tool": str(tool_events[0].get("provider") or "unknown") if tool_events else "not_used",
            "memory_materialization": "producer_same_generation" if producer_materializations else "not_used",
        },
        "result_latency_sec": float(workflow_end["result_latency_sec"]),
        "drain_latency_sec": float(workflow_end["drain_latency_sec"]),
        "consumer_input_tokens": sum(int(e["input_tokens"]) for e in consumers),
        "consumer_ttft_median_sec": statistics.median([float(e["ttft_sec"]) for e in consumers if e.get("ttft_sec") is not None]),
        "consumer_ttft_p95_sec": percentile([float(e["ttft_sec"]) for e in consumers if e.get("ttft_sec") is not None], 0.95),
        "consumer_latency_median_sec": statistics.median([float(e["request_latency_sec"]) for e in consumers]),
        "consumer_latency_p95_sec": percentile([float(e["request_latency_sec"]) for e in consumers], 0.95),
        "raw_propagated_tokens": raw_actual,
        "direct_passthrough_tokens": direct_tokens if mode == "producer" else 0,
        "raw_equivalent_tokens": raw_equivalent,
        "unique_artifact_tokens": unique_tokens,
        "repeated_context_tokens": max(0, raw_equivalent - unique_tokens),
        "actual_context_tokens": actual_tokens,
        "duplication_ratio": raw_equivalent / unique_tokens if unique_tokens else 0.0,
        "actual_amplification_ratio": actual_tokens / unique_tokens if unique_tokens else 0.0,
        "structured_memory_tokens": sum(int(e["structured_tokens"]) for e in producer_materializations),
        "retrieval_tokens": retrieval_tokens,
        "compression_ratio": (sum(int(e["source_tokens"]) for e in producer_materializations) / sum(int(e["structured_tokens"]) for e in producer_materializations)) if producer_materializations and sum(int(e["structured_tokens"]) for e in producer_materializations) else None,
        "avoided_propagation_tokens": max(0, raw_equivalent - actual_tokens),
        "evidence_source_coverage": evidence_coverage,
        "artifact_consumer_count": {source: len(consumers_) for source, consumers_ in sorted(per_artifact_consumers.items())},
        "tool_latency_sec": sum(float(e["latency_sec"]) for e in events if e["event_type"] == "tool_return"),
        "main_request_count": len(llms),
        "producer_sidecar_main_request_count": len([e for e in llms if e.get("producer_sidecar")]),
        "producer_memory_extra_output_tokens": sum(int(e["extra_output_tokens"]) for e in producer_materializations),
        "answer_output_tokens": sum(int(e.get("answer_tokens") or e["output_tokens"]) for e in llms),
        "total_completion_tokens": sum(int(e["output_tokens"]) for e in llms),
        "memory_grounding_ratio": statistics.median(float(e["grounding_ratio"]) for e in llm_materializations) if llm_materializations else None,
        "memory_focus_coverage": sum(bool(e["focus_match"]) for e in llm_materializations) / len(llm_materializations) if llm_materializations else None,
        "memory_distinctiveness": pairwise_distinctiveness([set(e["memory_terms"]) for e in llm_materializations]),
        "graph_eligible_artifact_count": sum(bool(e["eligible"]) for e in policy_decisions),
        "graph_bypassed_artifact_count": sum(not bool(e["eligible"]) for e in policy_decisions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("configs") / "ascend_qwen3_8b.json")
    parser.add_argument("--workload", choices=sorted(WORKLOADS), required=True)
    parser.add_argument("--mode", choices=["raw", "producer"], required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).with_name("raw_runs"))
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not config.get("tavily", {}).get("replay_snapshot_root") and not os.environ.get("TAVILY_API_KEY"):
        raise RuntimeError("TAVILY_API_KEY is required unless replay_snapshot_root is configured")
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = args.output_root / args.workload / args.mode / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = {**config, "selected_workload": args.workload, "selected_mode": args.mode, "run_id": run_id}
    (run_dir / "config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    workflow = GraphMemoryWorkflow(config, workload=args.workload, mode=args.mode, run_dir=run_dir, run_id=run_id)
    print(json.dumps(workflow.run(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from mas_workflow.app.llm_client import LocalLLMClient
from mas_workflow.app.search_providers import TavilySearchProvider

from .admission import CriticalFrontierAdmission
from .trace import TraceWriter


ROOT = Path(__file__).resolve().parent


def digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


class CaseStudyWorkflow:
    def __init__(self, config: dict[str, Any], *, policy: str, out_dir: Path) -> None:
        self.config = config
        self.policy = policy
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_id = f"cs1_{policy}_{stamp}"
        self.out_dir = out_dir / self.run_id
        self.out_dir.mkdir(parents=True, exist_ok=False)
        (self.out_dir / "experiment_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.trace = TraceWriter(self.out_dir / "client_trace.jsonl", run_id=self.run_id, policy=policy)
        model = config["model"]
        workload = config["workflow"]
        self.client = LocalLLMClient(
            model=model["served_name"],
            base_url=model["base_url"],
            temperature=float(workload["temperature"]),
            max_tokens=max(
                int(workload["critical_reviewer_max_tokens"]),
                int(workload["critical_finalizer_max_tokens"]),
            ),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model["path"], local_files_only=True)
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY is required; no fallback is allowed")
        self.search = TavilySearchProvider(api_key=api_key)
        admission = config["admission"]
        self.controller = CriticalFrontierAdmission(
            policy=policy,
            prefill_budget_tokens=int(admission["critical_prefill_budget_tokens"]),
            max_defer_sec=float(admission["max_defer_sec"]),
            serialize_deferred_drain=bool(admission["serialize_deferred_drain"]),
            emit=self.trace.emit,
        )
        self.coder_done = threading.Event()
        self.task = (
            "Design a reliable patch for an event-driven Python service where tool-return "
            "callbacks can race with a critical review/finalization chain. Explain invariants, "
            "edge cases, tests, and why the patch preserves top-level result latency. "
            f"Experiment nonce: {self.run_id}."
        )

    def prompt_tokens(self, messages: list[dict[str, Any]]) -> int:
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if hasattr(ids, "shape") and len(ids.shape) > 1:
            return int(ids.shape[-1])
        return len(ids)

    def llm(
        self,
        *,
        node_id: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        phase: str,
        critical: bool,
        tool_resumed: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        request_id = f"{self.run_id}__{node_id}"
        ready_ts = time.time()
        ready_prompt_tokens = self.prompt_tokens(messages)
        self.trace.emit(
            "llm_request_ready",
            request_id=request_id,
            node_id=node_id,
            phase=phase,
            critical_path_candidate=critical,
            tool_resumed=tool_resumed,
            ready_ts=ready_ts,
            ready_prompt_tokens=ready_prompt_tokens,
            message_count=len(messages),
            messages_hash=digest(messages),
        )
        decision = self.controller.admit(
            request_id=request_id,
            node_id=node_id,
            phase=phase,
            prompt_tokens=ready_prompt_tokens,
            critical=critical,
            tool_resumed=tool_resumed,
            ready_ts=ready_ts,
        )
        self.trace.emit(
            "llm_request_submit",
            request_id=request_id,
            node_id=node_id,
            submit_ts=decision.submit_ts,
            phase=phase,
            critical_path_candidate=critical,
        )

        def first_token(ts: float) -> None:
            self.trace.emit(
                "llm_first_token",
                request_id=request_id,
                node_id=node_id,
                first_token_ts=ts,
                phase=phase,
                critical_path_candidate=critical,
            )

        try:
            content, metadata = self.client.invoke_messages_streaming_with_metadata(
                messages,
                metadata={
                    "request_id_for_backend": request_id,
                    "agent_id": node_id,
                    "priority": "critical" if critical else "background",
                    "_on_first_token": first_token,
                },
                max_tokens=max_tokens,
                seed=int(self.config["workflow"]["seed"]),
            )
            usage = metadata.get("usage") or {}
            self.trace.emit(
                "llm_request_end",
                request_id=request_id,
                node_id=node_id,
                phase=phase,
                critical_path_candidate=critical,
                tool_resumed=tool_resumed,
                completion_ts=metadata.get("completion_ts"),
                first_token_ts=metadata.get("first_token_ts"),
                ttft_sec=metadata.get("ttft_sec"),
                tpot_sec=metadata.get("tpot_sec"),
                tpot_p95_sec=metadata.get("tpot_p95_sec"),
                stream_chunk_timestamps=metadata.get("stream_chunk_timestamps", []),
                stream_inter_token_ms=metadata.get("stream_inter_token_ms", []),
                input_tokens=int(usage.get("prompt_tokens") or ready_prompt_tokens),
                output_tokens=int(usage.get("completion_tokens") or 0),
                response_hash=digest(content),
                response_chars=len(content),
                backend_response_id=metadata.get("response_id"),
                timing_source="openai_sse_and_vllm_usage",
            )
            return content, metadata
        finally:
            self.controller.on_request_complete(request_id)

    def pre_tool_branch(self, index: int) -> dict[str, Any]:
        workload = self.config["workflow"]
        initial = [
            {
                "role": "system",
                "content": (
                    "You are a non-critical evidence agent. Formulate one focused web query, "
                    "then later resume from the returned evidence. Preserve the conversation."
                ),
            },
            {
                "role": "user",
                "content": f"Task:\n{self.task}\nEvidence branch: {index}. Prepare a search query.",
            },
        ]
        pre, _ = self.llm(
            node_id=f"tool_branch_{index}_pre",
            messages=initial,
            max_tokens=int(workload["pre_tool_max_tokens"]),
            phase="cold_prefill",
            critical=False,
        )
        self.coder_done.wait()
        query = (
            "multi-agent LLM serving tool resume prefill decode interference workflow "
            f"critical path evidence branch {index}"
        )
        tool_start = time.time()
        self.trace.emit(
            "tool_call",
            node_id=f"tool_branch_{index}_tavily",
            branch_id=index,
            tool="tavily",
            query_hash=digest(query),
            tool_start_ts=tool_start,
            dependency=[f"tool_branch_{index}_pre", "critical_coder"],
        )
        tavily_cfg = self.config["tavily"]
        result = self.search.search(
            query,
            max_results=int(tavily_cfg["max_results"]),
            include_answer=bool(tavily_cfg["include_answer"]),
            include_raw_content=bool(tavily_cfg["include_raw_content"]),
            search_depth=str(tavily_cfg["search_depth"]),
        )
        tool_end = time.time()
        tool_payload = {"answer": result.answer, "results": result.results}
        tool_path = self.out_dir / f"tool_branch_{index}_tavily.json"
        tool_path.write_text(
            json.dumps(tool_payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tool_text = json.dumps(tool_payload, ensure_ascii=False, sort_keys=True)
        self.trace.emit(
            "tool_return",
            node_id=f"tool_branch_{index}_tavily",
            branch_id=index,
            tool="tavily",
            provider="tavily",
            tool_start_ts=tool_start,
            tool_return_ts=tool_end,
            tool_latency_sec=tool_end - tool_start,
            result_count=len(result.results),
            result_chars=len(tool_text),
            result_hash=result.result_hash,
            raw_content_requested=True,
            result_truncated=False,
            artifact_path=str(tool_path),
        )
        resume_messages = initial + [
            {"role": "assistant", "content": pre},
            {
                "role": "user",
                "content": "TOOL_RETURN tavily (complete, untruncated JSON):\n" + tool_text,
            },
        ]
        resumed, meta = self.llm(
            node_id=f"tool_branch_{index}_resume",
            messages=resume_messages,
            max_tokens=int(workload["resume_max_tokens"]),
            phase="resume_prefill",
            critical=False,
            tool_resumed=True,
        )
        return {
            "branch": index,
            "resume_hash": digest(resumed),
            "prompt_tokens": int((meta.get("usage") or {}).get("prompt_tokens") or 0),
        }

    def run(self) -> dict[str, Any]:
        ok, detail = self.client.is_available(timeout=5)
        if not ok:
            raise RuntimeError(f"vLLM endpoint unavailable: {detail}")
        self.trace.emit(
            "workflow_start",
            model=self.config["model"]["served_name"],
            task_hash=digest(self.task),
            graph={
                "critical_chain": ["critical_coder", "critical_reviewer", "critical_finalizer"],
                "background_pattern": ["pre_tool_llm", "tavily", "resume_prefill"],
                "result_terminal": "critical_finalizer",
            },
        )
        workload = self.config["workflow"]
        coder_messages = [
            {"role": "system", "content": "You are the critical-path coder. Produce a concrete design and patch sketch."},
            {"role": "user", "content": self.task},
        ]
        with ThreadPoolExecutor(max_workers=int(workload["tool_branches"]) + 2) as pool:
            tool_futures = [
                pool.submit(self.pre_tool_branch, i)
                for i in range(1, int(workload["tool_branches"]) + 1)
            ]
            code, _ = self.llm(
                node_id="critical_coder",
                messages=coder_messages,
                max_tokens=int(workload["critical_coder_max_tokens"]),
                phase="cold_prefill",
                critical=True,
            )
            self.controller.begin_frontier("critical_reviewer_to_result")
            self.coder_done.set()
            review_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are the latency-critical reviewer. Analyze correctness, races, "
                        "failure recovery, tests, and serving implications in detail."
                    ),
                },
                {"role": "user", "content": f"Task:\n{self.task}\nCritical candidate:\n{code}"},
            ]
            review, _ = self.llm(
                node_id="critical_reviewer",
                messages=review_messages,
                max_tokens=int(workload["critical_reviewer_max_tokens"]),
                phase="cold_prefill",
                critical=True,
            )
            self.controller.transition(
                completed_node="critical_reviewer",
                ready_successor="critical_finalizer",
            )
            final_messages = [
                {"role": "system", "content": "Return the final top-level answer from the critical chain only."},
                {"role": "user", "content": f"Candidate:\n{code}\nReview:\n{review}"},
            ]
            final, _ = self.llm(
                node_id="critical_finalizer",
                messages=final_messages,
                max_tokens=int(workload["critical_finalizer_max_tokens"]),
                phase="cold_prefill",
                critical=True,
            )
            result_ready_ts = time.time()
            self.trace.emit(
                "workflow_result_ready",
                node_id="critical_finalizer",
                result_ready_ts=result_ready_ts,
                result_hash=digest(final),
            )
            self.controller.end_frontier("critical_reviewer_to_result")
            background = [future.result() for future in tool_futures]
        workflow_end_ts = time.time()
        self.trace.emit(
            "workflow_end",
            workflow_end_ts=workflow_end_ts,
            result_ready_ts=result_ready_ts,
            background_branch_count=len(background),
            all_background_completed=True,
        )
        summary = {
            "run_id": self.run_id,
            "policy": self.policy,
            "client_trace": str(self.trace.path),
            "result_ready_sec": result_ready_ts - self.trace.started_wall,
            "drain_complete_sec": workflow_end_ts - self.trace.started_wall,
            "background": background,
            "model": self.config["model"]["served_name"],
            "experiment_config": str(self.out_dir / "experiment_config.json"),
            "real_vllm": True,
            "real_tavily": True,
        }
        (self.out_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["default_vllm", "critical_frontier"], required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts" / "runs")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    CaseStudyWorkflow(config, policy=args.policy, out_dir=args.out_dir).run()


if __name__ == "__main__":
    main()

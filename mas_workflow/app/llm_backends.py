"""LLM backends for week-1 topology workloads."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .llm_client import LocalLLMClient
from .tracing import stable_hash


@dataclass
class LLMResult:
    content: str
    request_id_for_backend: str
    queue_wait_sec: float
    dispatch_start_ts: float
    dispatch_end_ts: float
    generation_start_ts: float
    generation_end_ts: float
    duration_sec: float
    request_metadata: dict[str, Any]


class MockLLM:
    mode = "mock"

    def __init__(self, model: str = "mock-mas-model") -> None:
        self.model = model

    def invoke(self, system_prompt: str, user_prompt: str, metadata: dict[str, Any]) -> LLMResult:
        request_id = str(uuid.uuid4())
        start = time.time()
        role = str(metadata.get("agent_role") or metadata.get("node_type") or "agent")
        agent_id = str(metadata.get("agent_id") or "agent")
        payload: dict[str, Any]
        if role == "manager":
            round_id = int(metadata.get("manager_round_id") or metadata.get("round_id") or 0)
            max_rounds = int(metadata.get("max_rounds") or 2)
            force_rounds = int(metadata.get("force_continue_rounds") or 0)
            decision = "continue" if round_id < min(max_rounds - 1, force_rounds) else "finish"
            if "需要多轮" in user_prompt and round_id < max_rounds - 1:
                decision = "continue"
            payload = {
                "decision": decision,
                "selected_workers": metadata.get("worker_ids") or ["worker_1"],
                "reason": f"mock manager round {round_id}",
                "next_instruction": "collect concise evidence and confidence",
                "confidence": 0.74,
            }
        elif role in {"worker", "independent_worker"}:
            payload = {
                "agent_id": agent_id,
                "answer": f"{agent_id} gives a bounded analysis for: {user_prompt[:80]}",
                "evidence": [f"synthetic evidence from {agent_id}"],
                "confidence": 0.72,
            }
        elif role == "debate_agent":
            payload = {
                "agent_id": agent_id,
                "argument": f"{agent_id} revises position using peer context.",
                "critique": "peer messages considered",
                "confidence": 0.69,
            }
        elif role == "aggregator":
            payload = {
                "final_summary": "Aggregated mock conclusion across agents.",
                "vote": "consensus",
                "confidence": 0.76,
            }
        elif role == "finalizer":
            payload = {
                "final_answer": "Week-1 topology run completed with mock LLM output.",
                "status": "complete",
            }
        else:
            payload = {"answer": f"Mock response for {agent_id}", "confidence": 0.7}
        content = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        end = time.time()
        return LLMResult(
            content=content,
            request_id_for_backend=request_id,
            queue_wait_sec=0.0,
            dispatch_start_ts=start,
            dispatch_end_ts=start,
            generation_start_ts=start,
            generation_end_ts=end,
            duration_sec=max(0.0, end - start),
            request_metadata={"mock": True, "metadata_hash": stable_hash(metadata)[:16]},
        )


class OpenAICompatibleLLM:
    mode = "openai_compatible"

    def __init__(self, *, model: str, base_url: str, max_tokens: int = 4096) -> None:
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.client = LocalLLMClient(model=model, base_url=base_url, max_tokens=max_tokens)

    def invoke(self, system_prompt: str, user_prompt: str, metadata: dict[str, Any]) -> LLMResult:
        request_id = str(uuid.uuid4())
        dispatch_start = time.time()
        metadata = dict(metadata)
        metadata["request_id_for_backend"] = request_id
        headers_metadata = dict(metadata)
        headers_metadata["X-Request-Id"] = request_id
        generation_start = time.time()
        request_max_tokens = int(metadata.get("request_max_output_tokens") or self.max_tokens)
        content, response_metadata = self.client.invoke_with_metadata(system_prompt, user_prompt, metadata=headers_metadata, max_tokens=request_max_tokens)
        end = time.time()
        usage = response_metadata.get("usage") or {}
        return LLMResult(
            content=content,
            request_id_for_backend=request_id,
            queue_wait_sec=0.0,
            dispatch_start_ts=dispatch_start,
            dispatch_end_ts=generation_start,
            generation_start_ts=generation_start,
            generation_end_ts=end,
            duration_sec=end - generation_start,
            request_metadata={
                "x_request_id": request_id,
                "backend_response_id": response_metadata.get("response_id"),
                "backend_finish_reason": response_metadata.get("finish_reason"),
                "backend_prompt_tokens": usage.get("prompt_tokens"),
                "backend_completion_tokens": usage.get("completion_tokens"),
                "backend_total_tokens": usage.get("total_tokens"),
                "backend_system_fingerprint": response_metadata.get("system_fingerprint"),
                "max_output_tokens": request_max_tokens,
            },
        )


def build_llm_backend(llm_mode: str, *, model: str, backend_base_url: str, max_output_tokens: int = 4096) -> MockLLM | OpenAICompatibleLLM:
    if llm_mode == "mock":
        return MockLLM(model=model)
    if llm_mode == "openai_compatible":
        return OpenAICompatibleLLM(model=model, base_url=backend_base_url, max_tokens=max_output_tokens)
    raise ValueError(f"Unsupported llm_mode: {llm_mode}")

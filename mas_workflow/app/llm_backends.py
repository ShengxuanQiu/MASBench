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
        request_id = str(metadata.get("request_id_for_backend") or uuid.uuid4())
        start = time.time()
        role = str(metadata.get("agent_role") or metadata.get("node_type") or "agent")
        agent_id = str(metadata.get("agent_id") or "agent")
        payload: dict[str, Any]
        if metadata.get("motif_family") and role == "evaluator":
            payload = {"decision": "accept", "feedback": "Synthetic acceptance for runtime smoke tests only."}
        elif metadata.get("motif_family") and role == "dispatcher" and '"worker_index"' in system_prompt:
            payload = {"worker_index": 0, "instruction": "Address the task."}
        elif metadata.get("motif_family") and role == "collector" and '"selected_index"' in system_prompt:
            payload = {"selected_index": 0, "reason": "Synthetic selection for runtime smoke tests only."}
        elif role == "manager":
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

    def __init__(self, *, model: str, base_url: str, max_tokens: int = 4096,
                 generation: dict[str, Any] | None = None) -> None:
        generation = dict(generation or {})
        self.model = model
        self.base_url = base_url
        self.max_tokens = int(generation.get("max_tokens", max_tokens))
        self.generation = generation
        self.default_request_payload = {key: value for key, value in generation.items()
                                        if key != "max_tokens"}
        self.client = LocalLLMClient(
            model=model,
            base_url=base_url,
            max_tokens=self.max_tokens,
            temperature=float(generation.get("temperature", 0.2)),
        )

    def invoke(self, system_prompt: str, user_prompt: str, metadata: dict[str, Any]) -> LLMResult:
        request_id = str(metadata.get("request_id_for_backend") or uuid.uuid4())
        dispatch_start = time.time()
        metadata = dict(metadata)
        metadata["request_id_for_backend"] = request_id
        # Fixed replay payloads are authoritative. Native runs use the fully
        # resolved DeploymentSpec generation contract (seed, template kwargs,
        # sampling parameters, etc.) rather than recording parameters that the
        # backend never received.
        if not metadata.get("_fixed_payload"):
            metadata["_request_payload"] = {
                **self.default_request_payload,
                **dict(metadata.get("_request_payload") or {}),
            }
        headers_metadata = dict(metadata)
        headers_metadata["X-Request-Id"] = request_id
        request_max_tokens = int(metadata.get("request_max_output_tokens") or self.max_tokens)
        # Canonical runs cannot silently fall back to clients that drop recorded parameters.
        invoke = self.client.invoke_streaming_with_metadata if metadata.get("_fixed_payload") else self.client.invoke_with_metadata
        content, response_metadata = invoke(system_prompt, user_prompt, metadata=headers_metadata, max_tokens=request_max_tokens)
        end = float(response_metadata.get("completion_ts") or time.time())
        generation_start = float(response_metadata.get("first_token_ts") or end)
        usage = response_metadata.get("usage") or {}
        return LLMResult(
            content=content,
            request_id_for_backend=request_id,
            queue_wait_sec=0.0,
            dispatch_start_ts=dispatch_start,
            dispatch_end_ts=generation_start,
            generation_start_ts=generation_start,
            generation_end_ts=end,
            duration_sec=end - dispatch_start,
            request_metadata={
                "x_request_id": request_id,
                "backend_response_id": response_metadata.get("response_id"),
                "backend_finish_reason": response_metadata.get("finish_reason"),
                "backend_prompt_tokens": usage.get("prompt_tokens"),
                "backend_completion_tokens": usage.get("completion_tokens"),
                "backend_total_tokens": usage.get("total_tokens"),
                "backend_system_fingerprint": response_metadata.get("system_fingerprint"),
                "max_output_tokens": request_max_tokens,
                "first_token_ts": response_metadata.get("first_token_ts"),
                "ttft_sec": response_metadata.get("ttft_sec"),
                "tpot_sec": response_metadata.get("tpot_sec"),
                "tpot_p95_sec": response_metadata.get("tpot_p95_sec"),
                "stream_chunk_timestamps": response_metadata.get("stream_chunk_timestamps", []),
                "stream_chunk_token_counts": response_metadata.get("stream_chunk_token_counts", []),
                "streaming_timing_granularity": response_metadata.get("streaming_timing_granularity", "unavailable"),
            },
        )


def build_llm_backend(llm_mode: str, *, model: str, backend_base_url: str,
                      max_output_tokens: int = 4096,
                      generation: dict[str, Any] | None = None) -> MockLLM | OpenAICompatibleLLM:
    if llm_mode == "mock":
        return MockLLM(model=model)
    if llm_mode == "openai_compatible":
        return OpenAICompatibleLLM(model=model, base_url=backend_base_url,
                                   max_tokens=max_output_tokens, generation=generation)
    raise ValueError(f"Unsupported llm_mode: {llm_mode}")

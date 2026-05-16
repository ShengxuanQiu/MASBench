"""调度器占位实现。

第一版不真正改 vLLM scheduler，只保留 criticality/cache-aware 调度研究所需接口。
"""

from __future__ import annotations

from typing import Any

from .state import MASState


def compute_agent_priority(state: MASState, agent_name: str, metadata: dict[str, Any] | None = None) -> float:
    """计算 agent 优先级，支持 criticality-aware 调度。"""
    metadata = metadata or {}
    is_critical = float(bool(metadata.get("is_critical", _default_is_critical(agent_name))))
    expected_downstream_unblock = float(metadata.get("expected_downstream_unblock", _downstream_unblock(state, agent_name)))
    estimated_remaining_steps = float(metadata.get("estimated_remaining_steps", _remaining_steps(agent_name)))
    cache_residency_hint = float(metadata.get("cache_residency_hint", estimate_cache_residency(state, agent_name)))
    return (
        3.0 * is_critical
        + 1.0 * expected_downstream_unblock
        - 0.5 * estimated_remaining_steps
        + 0.5 * cache_residency_hint
    )


def _default_is_critical(agent_name: str) -> bool:
    if agent_name in {"terminal_agent", "repo_search_agent"}:
        return True
    return any(
        key in agent_name
        for key in ("reviewer", "terminal_verify", "finalizer", "revise", "planner_replan")
    )


def _downstream_unblock(state: MASState, agent_name: str) -> int:
    return len(
        [
            name
            for name, parents in state.get("agent_dependencies", {}).items()
            if agent_name in parents
        ]
    )


def _remaining_steps(agent_name: str) -> int:
    if agent_name in {"finalizer"}:
        return 0
    if "terminal_verify" in agent_name or "reviewer_round2" in agent_name:
        return 1
    if "revise" in agent_name or agent_name == "reviewer":
        return 2
    if agent_name.startswith("coder_"):
        return 3
    if agent_name in {"merge_context", "merge_context_round2"}:
        return 4
    if agent_name == "planner":
        return 6
    return 5


def estimate_cache_residency(state: MASState, agent_name: str) -> float:
    """估计该 agent 上下文留在 KV cache 中的收益。"""
    kv_tokens = state.get("estimated_kv_tokens", {}).get(agent_name, 0)
    return min(1.0, kv_tokens / 32768.0)


def estimate_remaining_work(state: MASState, agent_name: str) -> float:
    """估计剩余工作量，第一版用输出 token 和状态粗略估计。"""
    status = state.get("agent_status", {}).get(agent_name, "pending")
    if status == "finished":
        return 0.0
    return 1.0 + state.get("estimated_output_tokens", {}).get(agent_name, 0) / 2048.0


def select_next_agent(ready_agents: list[str], state: MASState) -> str:
    """从 ready agents 中选择下一个执行者，第一版近似 FCFS。"""
    if not ready_agents:
        raise ValueError("ready_agents 不能为空")
    return max(ready_agents, key=lambda name: compute_agent_priority(state, name))


def build_request_metadata(state: MASState, agent_name: str) -> dict[str, object]:
    """构造传给 vLLM 的请求元数据。"""
    metadata = {
        "workflow_id": state.get("workflow_id", ""),
        "agent_id": agent_name,
        "agent_name": agent_name,
        "node_id": agent_name,
        "round_id": 2 if "round2" in agent_name or "revise" in agent_name or "more_evidence" in agent_name else 1,
        "shared_context_hash": state.get("shared_context_hash", ""),
        "is_critical": _default_is_critical(agent_name),
        "expected_downstream_unblock": _downstream_unblock(state, agent_name),
        "estimated_remaining_steps": _remaining_steps(agent_name),
        "cache_residency_hint": estimate_cache_residency(state, agent_name),
        "retry_count": state.get("retry_count", 0),
    }
    metadata["criticality"] = 1.0 if metadata["is_critical"] else 0.0
    metadata["priority"] = compute_agent_priority(state, agent_name, metadata)
    return {
        **metadata,
        "cache_residency": estimate_cache_residency(state, agent_name),
        "remaining_work": estimate_remaining_work(state, agent_name),
    }

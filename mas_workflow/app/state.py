"""LangGraph 全局状态定义。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict


def merge_dict(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """合并并行节点返回的字典字段。"""
    merged: dict[str, Any] = {}
    if left:
        merged.update(left)
    if right:
        merged.update(right)
    return merged


AgentStatus = Literal["pending", "running", "blocked", "finished", "failed"]


class MASState(TypedDict, total=False):
    """复合型 MAS workflow 的共享状态。"""

    user_query: str
    repo_path: str
    test_command: str
    plan: str
    file_context: str
    web_context: str
    doc_context: str
    repo_search_context: str
    test_log: str
    verify_log: str
    candidate_patch_a: str
    candidate_patch_b: str
    candidate_patch_c: str
    candidate_patch_a_round2: str
    candidate_patch_b_round2: str
    candidate_patch_c_round2: str
    review_result: str
    reviewer_critique: str
    review_result_round2: str
    final_answer: str
    retry_count: int
    max_retries: int
    enable_round2: bool
    workflow_id: str
    shared_context_hash: str
    dispatcher: Any
    tool_delay_profile: str
    burst_tool_return: bool
    committee_width: int

    trace_events: Annotated[list[dict[str, Any]], operator.add]
    agent_outputs: Annotated[dict[str, Any], merge_dict]
    agent_status: Annotated[dict[str, AgentStatus], merge_dict]
    agent_dependencies: Annotated[dict[str, list[str]], merge_dict]
    estimated_prompt_tokens: Annotated[dict[str, int], merge_dict]
    estimated_output_tokens: Annotated[dict[str, int], merge_dict]
    estimated_kv_tokens: Annotated[dict[str, int], merge_dict]
    tool_wait_time: Annotated[dict[str, float], merge_dict]
    llm_time: Annotated[dict[str, float], merge_dict]
    request_metadata: Annotated[dict[str, dict[str, Any]], merge_dict]
    parallel_groups: Annotated[dict[str, dict[str, Any]], merge_dict]
    dispatcher_records: Annotated[dict[str, dict[str, Any]], merge_dict]
    all_gather_stats: Annotated[dict[str, dict[str, Any]], merge_dict]


def initial_agent_dependencies() -> dict[str, list[str]]:
    """返回第一版 DAG 的依赖关系，便于 trace 和调度研究。"""
    return {
        "planner": [],
        "file_agent": ["planner"],
        "terminal_agent": ["planner"],
        "web_agent": ["planner"],
        "doc_agent": ["planner"],
        "repo_search_agent": ["planner"],
        "merge_context": ["file_agent", "terminal_agent", "web_agent", "doc_agent", "repo_search_agent"],
        "coder_a": ["merge_context"],
        "coder_b": ["merge_context"],
        "coder_c": ["merge_context"],
        "reviewer": ["coder_a", "coder_b", "coder_c"],
        "doc_agent_more_evidence": ["reviewer"],
        "web_agent_more_evidence": ["reviewer"],
        "repo_search_agent_more_evidence": ["reviewer"],
        "merge_context_round2": ["doc_agent_more_evidence", "web_agent_more_evidence", "repo_search_agent_more_evidence"],
        "broadcast_critique": ["reviewer", "merge_context_round2", "planner_replan"],
        "coder_a_revise": ["broadcast_critique"],
        "coder_b_revise": ["broadcast_critique"],
        "coder_c_revise": ["broadcast_critique"],
        "reviewer_round2": ["coder_a_revise", "coder_b_revise", "coder_c_revise"],
        "terminal_verify": ["reviewer", "reviewer_round2"],
        "planner_replan": ["terminal_verify"],
        "finalizer": ["terminal_verify"],
    }

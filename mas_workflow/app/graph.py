"""LangGraph workflow 编排。"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .agents import (
    broadcast_critique_node,
    make_coder_node,
    make_coder_revision_node,
    make_doc_agent_node,
    make_file_agent_node,
    make_finalizer_node,
    make_merge_context_node,
    make_planner_node,
    make_repo_search_agent_node,
    make_reviewer_node,
    make_reviewer_round2_node,
    make_terminal_agent_node,
    make_web_agent_node,
)
from .llm_client import LocalLLMClient
from .state import MASState


def build_workflow(llm: LocalLLMClient):
    """构建动态争用 MAS DAG。

    第一版仍使用 LangGraph 的并行 fan-out/fan-in；真正的 LLM 争用由
    LLMDispatcher 在每个 agent 的 LLM 调用处统一制造和记录。
    """
    graph = StateGraph(MASState)

    graph.add_node("planner", make_planner_node(llm))
    graph.add_node("file_agent", make_file_agent_node(llm))
    graph.add_node("terminal_agent", make_terminal_agent_node(llm))
    graph.add_node("web_agent", make_web_agent_node(llm))
    graph.add_node("doc_agent", make_doc_agent_node(llm))
    graph.add_node("repo_search_agent", make_repo_search_agent_node(llm))
    graph.add_node("merge_context", make_merge_context_node(name="merge_context"))

    graph.add_node("coder_a", make_coder_node(llm, name="coder_a", prompt_name="coder_a", output_key="candidate_patch_a"))
    graph.add_node("coder_b", make_coder_node(llm, name="coder_b", prompt_name="coder_b", output_key="candidate_patch_b"))
    graph.add_node("coder_c", make_coder_node(llm, name="coder_c", prompt_name="coder_c", output_key="candidate_patch_c"))
    graph.add_node("reviewer", make_reviewer_node(llm))

    graph.add_node("broadcast_critique", broadcast_critique_node)
    graph.add_node(
        "coder_a_revise",
        make_coder_revision_node(
            llm,
            name="coder_a_revise",
            base_candidate_key="candidate_patch_a",
            output_key="candidate_patch_a_round2",
        ),
    )
    graph.add_node(
        "coder_b_revise",
        make_coder_revision_node(
            llm,
            name="coder_b_revise",
            base_candidate_key="candidate_patch_b",
            output_key="candidate_patch_b_round2",
        ),
    )
    graph.add_node(
        "coder_c_revise",
        make_coder_revision_node(
            llm,
            name="coder_c_revise",
            base_candidate_key="candidate_patch_c",
            output_key="candidate_patch_c_round2",
        ),
    )
    graph.add_node("reviewer_round2", make_reviewer_round2_node(llm))

    graph.add_node("doc_agent_more_evidence", make_doc_agent_node(llm, name="doc_agent_more_evidence", output_key="doc_context"))
    graph.add_node("web_agent_more_evidence", make_web_agent_node(llm, name="web_agent_more_evidence", output_key="web_context"))
    graph.add_node(
        "repo_search_agent_more_evidence",
        make_repo_search_agent_node(llm, name="repo_search_agent_more_evidence", output_key="repo_search_context"),
    )
    graph.add_node("merge_context_round2", make_merge_context_node(name="merge_context_round2"))

    graph.add_node("terminal_verify", make_terminal_agent_node(llm, name="terminal_verify", output_key="verify_log"))
    graph.add_node("planner_replan", make_planner_node(llm))
    graph.add_node("finalizer", make_finalizer_node(llm))

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "file_agent")
    graph.add_edge("planner", "terminal_agent")
    graph.add_edge("planner", "web_agent")
    graph.add_edge("planner", "doc_agent")
    graph.add_edge("planner", "repo_search_agent")
    graph.add_edge(["file_agent", "terminal_agent", "web_agent", "doc_agent", "repo_search_agent"], "merge_context")

    graph.add_edge("merge_context", "coder_a")
    graph.add_edge("merge_context", "coder_b")
    graph.add_edge("merge_context", "coder_c")
    graph.add_edge(["coder_a", "coder_b", "coder_c"], "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {
            "pass": "terminal_verify",
            "revise": "broadcast_critique",
            "more_evidence": "doc_agent_more_evidence",
        },
    )
    graph.add_edge("doc_agent_more_evidence", "merge_context_round2")
    graph.add_edge("web_agent_more_evidence", "merge_context_round2")
    graph.add_edge("repo_search_agent_more_evidence", "merge_context_round2")
    graph.add_edge("doc_agent_more_evidence", "web_agent_more_evidence")
    graph.add_edge("web_agent_more_evidence", "repo_search_agent_more_evidence")
    graph.add_edge("merge_context_round2", "broadcast_critique")

    graph.add_edge("broadcast_critique", "coder_a_revise")
    graph.add_edge("broadcast_critique", "coder_b_revise")
    graph.add_edge("broadcast_critique", "coder_c_revise")
    graph.add_edge(["coder_a_revise", "coder_b_revise", "coder_c_revise"], "reviewer_round2")
    graph.add_edge("reviewer_round2", "terminal_verify")

    graph.add_conditional_edges(
        "terminal_verify",
        _route_after_verify,
        {"replan": "planner_replan", "final": "finalizer"},
    )
    graph.add_edge("planner_replan", "broadcast_critique")
    graph.add_edge("finalizer", END)

    return graph.compile()


def _route_after_review(state: MASState) -> str:
    if not state.get("enable_round2", True):
        return "pass"
    if state.get("retry_count", 0) >= state.get("max_retries", 1):
        return "pass"
    text = state.get("review_result", "").lower()
    if "need_more_evidence" in text or "更多证据" in text or "补充证据" in text:
        return "more_evidence"
    return "revise"


def _route_after_verify(state: MASState) -> str:
    text = state.get("verify_log", "").lower()
    failed = any(marker in text for marker in ["failed", "error", "失败", "returncode", "traceback"])
    if failed and state.get("retry_count", 0) < state.get("max_retries", 1):
        return "replan"
    return "final"

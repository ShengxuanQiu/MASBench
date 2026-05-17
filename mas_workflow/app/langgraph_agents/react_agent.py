"""LangGraph ReAct-style agent runner with MASBench-Arch tracing."""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from ..search_providers import BaseSearchProvider
from ..tracing import TraceContext, estimate_tokens, stable_hash, token_count_source
from .traced_tools import TracedToolRuntime, make_search_tools


def _no_proxy_for_local(base_url: str) -> None:
    if "127.0.0.1" not in base_url and "localhost" not in base_url:
        return
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = {part.strip() for part in existing.split(",") if part.strip()}
    parts.update({"127.0.0.1", "localhost"})
    value = ",".join(sorted(parts))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def run_react_agent(
    *,
    config: Any,
    trace: TraceContext,
    search_provider: BaseSearchProvider,
    motif_tags: list[str],
    node_id: str,
    node_name: str,
    node_type: str,
    agent_role: str,
    prompt_template: str,
    system_prompt: str,
    user_prompt: str,
    round_id: int | None,
    manager_round_id: int | None,
    peer_round_id: int | None,
    parents: list[str],
    parallel_group: str | None,
    criticality: str,
    extra_metadata: dict[str, Any],
) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    _no_proxy_for_local(config.backend_base_url)
    runtime = TracedToolRuntime(
        config=config,
        trace=trace,
        search_provider=search_provider,
        agent_id=node_id,
        agent_role=agent_role,
        round_id=round_id,
        manager_round_id=manager_round_id,
        peer_round_id=peer_round_id,
        parallel_group=parallel_group,
        motif_tags=motif_tags,
    )
    tools = make_search_tools(runtime)
    request_id = str(uuid.uuid4())
    model = ChatOpenAI(
        model=config.model,
        base_url=config.backend_base_url,
        api_key="EMPTY",
        temperature=0.2,
        max_tokens=config.max_output_tokens,
        default_headers={
            "X-Request-Id": request_id,
            "X-MAS-Agent-ID": node_id,
        },
    ).bind_tools(tools)

    prompt = (
        f"{system_prompt}\n\n"
        "You may call tools when evidence is needed. Use the search tool for web, documentation, "
        "repository, issue, API, or factual evidence. If the answer is already clear, do not call a tool.\n\n"
        f"{user_prompt}"
    )
    prompt_hash = stable_hash(prompt)

    def call_model(state: MessagesState) -> dict[str, Any]:
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        input_text = "\n".join(_message_text(m) for m in messages)
        trace.emit(
            event_type="llm_request_start",
            node_id=node_id,
            node_name=node_name,
            node_type="llm",
            round_id=round_id,
            manager_round_id=manager_round_id,
            peer_round_id=peer_round_id,
            parents=parents,
            motif_tags=motif_tags,
            parallel_group=parallel_group,
            criticality=criticality,
            status="start",
            agent_id=node_id,
            agent_role=agent_role,
            prompt_template=prompt_template,
            llm_mode=config.llm_mode,
            backend_base_url=config.backend_base_url,
            model=config.model,
            max_output_tokens=config.max_output_tokens,
            llm_request_id=request_id,
            request_id_for_backend=request_id,
            input_chars=len(input_text),
            input_tokens_est=estimate_tokens(input_text),
            token_count_source=token_count_source(),
            prompt_hash=prompt_hash,
            priority=1.0 if criticality == "critical" else 0.0,
            dispatch_policy=config.dispatch_policy,
            request_metadata={**extra_metadata, "react_agent": True},
        )
        start = time.time()
        response = model.invoke(messages)
        end = time.time()
        output = _message_text(response)
        usage = getattr(response, "usage_metadata", None) or {}
        response_metadata = getattr(response, "response_metadata", None) or {}
        tool_calls = getattr(response, "tool_calls", None) or []
        trace.emit(
            event_type="llm_request_end",
            node_id=node_id,
            node_name=node_name,
            node_type="llm",
            round_id=round_id,
            manager_round_id=manager_round_id,
            peer_round_id=peer_round_id,
            parents=parents,
            motif_tags=motif_tags,
            parallel_group=parallel_group,
            criticality=criticality,
            status="success",
            duration_sec=round(end - start, 6),
            duration_source="llm_backend_measured" if config.llm_mode == "openai_compatible" else "mock_measured",
            replay_policy="live",
            agent_id=node_id,
            agent_role=agent_role,
            prompt_template=prompt_template,
            llm_mode=config.llm_mode,
            backend_base_url=config.backend_base_url,
            model=config.model,
            max_output_tokens=config.max_output_tokens,
            llm_request_id=request_id,
            request_id_for_backend=request_id,
            input_chars=len(input_text),
            output_chars=len(output),
            input_tokens_est=estimate_tokens(input_text),
            output_tokens_est=estimate_tokens(output),
            total_tokens_est=estimate_tokens(input_text) + estimate_tokens(output),
            token_count_source=token_count_source(),
            queue_wait_sec=0.0,
            generation_start_ts=start,
            generation_end_ts=end,
            prompt_hash=prompt_hash,
            output_hash=stable_hash(output),
            backend_prompt_tokens=usage.get("input_tokens"),
            backend_completion_tokens=usage.get("output_tokens"),
            backend_total_tokens=usage.get("total_tokens"),
            backend_finish_reason=response_metadata.get("finish_reason"),
            backend_response_id=response_metadata.get("id"),
            tool_call_count=len(tool_calls),
            tool_call_names=[call.get("name") for call in tool_calls],
            dispatch_policy=config.dispatch_policy,
            request_metadata={**extra_metadata, "react_agent": True, "tool_calls": tool_calls},
        )
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    app = graph.compile()
    result = app.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config={"recursion_limit": max(2, int(config.react_max_steps) * 2 + 2)},
    )
    messages = result.get("messages") or []
    final = _message_text(messages[-1]) if messages else ""
    trace.emit(
        event_type="agent_react_end",
        node_id=f"{node_id}_react",
        node_name=f"{node_name} ReactLoop",
        node_type=node_type,
        round_id=round_id,
        manager_round_id=manager_round_id,
        peer_round_id=peer_round_id,
        parents=parents,
        motif_tags=motif_tags,
        parallel_group=parallel_group,
        criticality=criticality,
        status="success",
        agent_id=node_id,
        agent_role=agent_role,
        output_hash=stable_hash(final),
        output_chars=len(final),
        output_tokens_est=estimate_tokens(final),
        token_count_source=token_count_source(),
        extra={"message_count": len(messages)},
    )
    return final


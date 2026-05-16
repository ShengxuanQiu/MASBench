"""LangGraph 节点实现。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

from .dispatcher import LLMDispatcher
from .llm_client import LocalLLMClient, to_json_prompt
from .prompts import load_prompt
from .scheduler import build_request_metadata
from .state import MASState
from .tools import run_pytest, safe_list_files, safe_read_files, simulated_tool_delay
from .tracing import Timer, estimate_tokens, make_event


NodeFn = Callable[[MASState], dict[str, Any]]


AGENT_MAX_TOKENS = {
    "planner": 1536,
    "file_agent": 2048,
    "terminal_agent": 1536,
    "terminal_verify": 1536,
    "web_agent": 1024,
    "web_agent_more_evidence": 1024,
    "doc_agent": 2048,
    "doc_agent_more_evidence": 2048,
    "repo_search_agent": 2048,
    "repo_search_agent_more_evidence": 2048,
    "coder_a": 3072,
    "coder_b": 3072,
    "coder_c": 3072,
    "coder_a_revise": 3072,
    "coder_b_revise": 3072,
    "coder_c_revise": 3072,
    "reviewer": 4096,
    "reviewer_round2": 4096,
    "finalizer": 3072,
}


def _deps(state: MASState, agent_name: str) -> list[str]:
    return state.get("agent_dependencies", {}).get(agent_name, [])


def _children(state: MASState, agent_name: str) -> list[str]:
    return [
        name
        for name, parents in state.get("agent_dependencies", {}).items()
        if agent_name in parents
    ]


def _shared_context(state: MASState) -> str:
    return to_json_prompt(
        {
            "plan": _clip(state.get("plan", ""), 1200),
            "file_context": _clip(state.get("file_context", ""), 5000),
            "web_context": _clip(state.get("web_context", ""), 2000),
            "doc_context": _clip(state.get("doc_context", ""), 3500),
            "repo_search_context": _clip(state.get("repo_search_context", ""), 3500),
            "test_log": _clip(state.get("test_log", ""), 4000),
            "verify_log": _clip(state.get("verify_log", ""), 2500),
        }
    )


def _clip(text: str | None, max_chars: int) -> str:
    """限制聚合节点输入长度，避免第一版 demo 超过本地模型上下文。"""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[已截断，原始长度 {len(text)} 字符]"


def _query_terms(text: str) -> list[str]:
    """从任务描述中抽取用于文件优先级排序的轻量关键词。"""
    raw_terms = {
        token.strip("`'\"()[]{}:,. ")
        for token in text.replace("/", " ").replace("_", " ").split()
    }
    terms = {term.lower() for term in raw_terms if len(term) >= 4}
    for seed in ("separable", "separability", "modeling", "compound", "test"):
        if seed in text.lower():
            terms.add(seed)
    return sorted(terms)


def _rank_files(files: list[str], query: str, suffixes: tuple[str, ...], limit: int) -> list[str]:
    """按 query 关键词把更可能相关的文件排到前面。"""
    terms = _query_terms(query)
    file_set = set(files)
    explicit_paths = []
    for token in query.replace(",", " ").replace("]", " ").replace("[", " ").split():
        candidate = token.strip("`'\"(){}:;.")
        if candidate in file_set and candidate.lower().endswith(suffixes):
            explicit_paths.append(candidate)

    def score(path: str) -> tuple[int, str]:
        lower = path.lower()
        value = 0
        for term in terms:
            if term in lower:
                value += 3
        if "/tests/" in lower or lower.startswith("tests/"):
            value += 1
        return (-value, path)

    candidates = [path for path in files if path.lower().endswith(suffixes)]
    ranked = explicit_paths + [path for path in sorted(candidates, key=score) if path not in explicit_paths]
    return ranked[:limit]


def _common_metrics(agent_name: str, input_text: str, output_text: str) -> dict[str, Any]:
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(output_text)
    return {
        "estimated_prompt_tokens": {agent_name: input_tokens},
        "estimated_output_tokens": {agent_name: output_tokens},
        "estimated_kv_tokens": {agent_name: input_tokens + output_tokens},
        "agent_outputs": {agent_name: output_text},
        "agent_status": {agent_name: "finished"},
    }


def _llm_call(
    llm: LocalLLMClient,
    *,
    agent_name: str,
    prompt_name: str,
    payload: dict[str, Any],
    state: MASState,
) -> tuple[str, list[dict[str, Any]], float]:
    system_prompt = load_prompt(prompt_name)
    user_prompt = to_json_prompt(payload)
    request_metadata = build_request_metadata(state, agent_name)
    events = [
        make_event(agent_name=agent_name, event_type="llm_submit_prepare", input_text=user_prompt, extra={"prompt_name": prompt_name, "request_metadata": request_metadata})
    ]
    with Timer() as timer:
        dispatcher = state.get("dispatcher")
        if isinstance(dispatcher, LLMDispatcher):
            response = dispatcher.submit(
                agent_name=agent_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata=request_metadata,
                priority=float(request_metadata.get("priority", 0.0)),
                max_tokens=AGENT_MAX_TOKENS.get(agent_name),
            )
            output = response.content
            events.extend(response.events)
        else:
            output = llm.invoke(
                system_prompt,
                user_prompt,
                metadata=request_metadata,
                max_tokens=AGENT_MAX_TOKENS.get(agent_name),
            )
            events.append(
                make_event(
                    agent_name=agent_name,
                    event_type="llm_end",
                    input_text=user_prompt,
                    output_text=output,
                    duration_sec=timer.duration if hasattr(timer, "duration") else 0.0,
                    extra={"prompt_name": prompt_name, "request_metadata": request_metadata},
                )
            )
    duration = timer.duration
    return output, events, duration


def _tool_delay(state: MASState, agent_name: str, input_text: str) -> tuple[list[dict[str, Any]], float]:
    events = [make_event(agent_name=agent_name, event_type="tool_start", input_text=input_text, extra={"tool": "simulated_delay"})]
    with Timer() as timer:
        delay = simulated_tool_delay(
            state.get("tool_delay_profile", "none"),
            burst=bool(state.get("burst_tool_return", False)),
        )
    events.append(
        make_event(
            agent_name=agent_name,
            event_type="tool_end",
            input_text=input_text,
            output_text=f"simulated_delay={delay:.3f}",
            duration_sec=timer.duration,
            extra={"tool": "simulated_delay", "delay_profile": state.get("tool_delay_profile", "none"), "burst_tool_return": state.get("burst_tool_return", False)},
        )
    )
    return events, timer.duration


def _parallel_group(agent_name: str) -> str:
    if agent_name in {
        "file_agent",
        "terminal_agent",
        "web_agent",
        "doc_agent",
        "repo_search_agent",
        "doc_agent_more_evidence",
        "web_agent_more_evidence",
        "repo_search_agent_more_evidence",
    }:
        return "evidence_collection"
    if agent_name in {"coder_a", "coder_b", "coder_c"}:
        return "candidate_generation"
    if agent_name in {"coder_a_revise", "coder_b_revise", "coder_c_revise"}:
        return "candidate_revision"
    return ""


def _node_start(agent_name: str, state: MASState, input_text: str) -> dict[str, Any]:
    return make_event(
        agent_name=agent_name,
        event_type="node_start",
        parents=_deps(state, agent_name),
        children=_children(state, agent_name),
        input_text=input_text,
        extra={"parallel_group": _parallel_group(agent_name)},
    )


def _node_end(agent_name: str, state: MASState, input_text: str, output_text: str, duration: float) -> dict[str, Any]:
    return make_event(
        agent_name=agent_name,
        event_type="node_end",
        parents=_deps(state, agent_name),
        children=_children(state, agent_name),
        input_text=input_text,
        output_text=output_text,
        duration_sec=duration,
        extra={"parallel_group": _parallel_group(agent_name)},
    )


def make_planner_node(llm: LocalLLMClient) -> NodeFn:
    def planner(state: MASState) -> dict[str, Any]:
        agent_name = "planner"
        payload = {
            "user_query": state.get("user_query", ""),
            "repo_path": state.get("repo_path", ""),
            "test_command": state.get("test_command", ""),
            "retry_count": state.get("retry_count", 0),
            "previous_review_result": state.get("review_result", ""),
        }
        input_text = to_json_prompt(payload)
        events = [_node_start(agent_name, state, input_text)]
        with Timer() as node_timer:
            output, llm_events, llm_duration = _llm_call(
                llm, agent_name=agent_name, prompt_name="planner", payload=payload, state=state
            )
            events.extend(llm_events)
        events.append(_node_end(agent_name, state, input_text, output, node_timer.duration))
        result = {
            "plan": output,
            "trace_events": events,
            "llm_time": {agent_name: llm_duration},
            "request_metadata": {agent_name: build_request_metadata(state, agent_name)},
        }
        result.update(_common_metrics(agent_name, input_text, output))
        return result

    return planner


def make_file_agent_node(llm: LocalLLMClient) -> NodeFn:
    def file_agent(state: MASState) -> dict[str, Any]:
        agent_name = "file_agent"
        base_payload = {
            "user_query": state.get("user_query", ""),
            "repo_path": state.get("repo_path", ""),
            "plan": state.get("plan", ""),
        }
        input_text = to_json_prompt(base_payload)
        events = [_node_start(agent_name, state, input_text)]
        with Timer() as node_timer:
            events.append(make_event(agent_name=agent_name, event_type="tool_start", input_text=input_text))
            with Timer() as tool_timer:
                simulated_tool_delay(
                    state.get("tool_delay_profile", "none"),
                    burst=bool(state.get("burst_tool_return", False)),
                )
                files = safe_list_files(state["repo_path"], max_files=3000)
                interesting = _rank_files(
                    files,
                    state.get("user_query", ""),
                    (".py", ".toml", ".cfg", ".ini", ".md", ".txt", ".yaml", ".yml"),
                    8,
                )
                contents = safe_read_files(state["repo_path"], interesting, max_chars_per_file=3000)
            tool_output = to_json_prompt({"ranked_files": interesting, "read_files": list(contents)})
            events.append(
                make_event(
                    agent_name=agent_name,
                    event_type="tool_end",
                    input_text=input_text,
                    output_text=tool_output,
                    duration_sec=tool_timer.duration,
                    extra={"file_count": len(files), "read_count": len(contents)},
                )
            )
            payload = base_payload | {
                "repo_file_count": len(files),
                "ranked_repo_files": interesting,
                "file_contents": contents,
            }
            output, llm_events, llm_duration = _llm_call(
                llm, agent_name=agent_name, prompt_name="file_agent", payload=payload, state=state
            )
            events.extend(llm_events)
        events.append(_node_end(agent_name, state, input_text, output, node_timer.duration))
        result = {
            "file_context": output,
            "trace_events": events,
            "tool_wait_time": {agent_name: tool_timer.duration},
            "llm_time": {agent_name: llm_duration},
            "request_metadata": {agent_name: build_request_metadata(state, agent_name)},
        }
        result.update(_common_metrics(agent_name, input_text + tool_output, output))
        return result

    return file_agent


def make_terminal_agent_node(
    llm: LocalLLMClient,
    *,
    name: str = "terminal_agent",
    output_key: str = "test_log",
) -> NodeFn:
    def terminal_agent(state: MASState) -> dict[str, Any]:
        agent_name = name
        payload = {
            "user_query": state.get("user_query", ""),
            "repo_path": state.get("repo_path", ""),
            "test_command": state.get("test_command", "pytest -q"),
            "plan": state.get("plan", ""),
        }
        input_text = to_json_prompt(payload)
        events = [_node_start(agent_name, state, input_text)]
        with Timer() as node_timer:
            events.append(make_event(agent_name=agent_name, event_type="tool_start", input_text=input_text))
            with Timer() as tool_timer:
                simulated_tool_delay(
                    state.get("tool_delay_profile", "none"),
                    burst=bool(state.get("burst_tool_return", False)),
                )
                pytest_result = run_pytest(
                    state["repo_path"],
                    state.get("test_command", "pytest -q"),
                    timeout=120,
                )
            tool_output = to_json_prompt(pytest_result)
            events.append(
                make_event(
                    agent_name=agent_name,
                    event_type="tool_end",
                    input_text=input_text,
                    output_text=tool_output,
                    duration_sec=tool_timer.duration,
                    status="ok" if pytest_result.get("returncode") == 0 else "failed",
                    extra={"returncode": pytest_result.get("returncode")},
                )
            )
            output, llm_events, llm_duration = _llm_call(
                llm,
                agent_name=agent_name,
                prompt_name="terminal_agent",
                payload=payload | {"pytest_result": pytest_result},
                state=state,
            )
            events.extend(llm_events)
        events.append(_node_end(agent_name, state, input_text, output, node_timer.duration))
        result = {
            output_key: output,
            "trace_events": events,
            "tool_wait_time": {agent_name: tool_timer.duration},
            "llm_time": {agent_name: llm_duration},
            "request_metadata": {agent_name: build_request_metadata(state, agent_name)},
        }
        result.update(_common_metrics(agent_name, input_text + tool_output, output))
        return result

    return terminal_agent


def make_web_agent_node(
    llm: LocalLLMClient,
    *,
    name: str = "web_agent",
    output_key: str = "web_context",
) -> NodeFn:
    def web_agent(state: MASState) -> dict[str, Any]:
        agent_name = name
        payload = {
            "user_query": state.get("user_query", ""),
            "repo_path": state.get("repo_path", ""),
            "plan": _clip(state.get("plan", ""), 1800),
            "review_result": _clip(state.get("review_result", ""), 1800),
            "network_policy": "第一版不主动联网；只根据用户 query 和已知任务类型给出外部资料需求清单。",
        }
        input_text = to_json_prompt(payload)
        events = [_node_start(agent_name, state, input_text)]
        with Timer() as node_timer:
            delay_events, delay_duration = _tool_delay(state, agent_name, input_text)
            events.extend(delay_events)
            output, llm_events, llm_duration = _llm_call(
                llm, agent_name=agent_name, prompt_name="web_agent", payload=payload, state=state
            )
            events.extend(llm_events)
        events.append(_node_end(agent_name, state, input_text, output, node_timer.duration))
        result = {
            output_key: output,
            "trace_events": events,
            "tool_wait_time": {agent_name: delay_duration},
            "llm_time": {agent_name: llm_duration},
            "request_metadata": {agent_name: build_request_metadata(state, agent_name)},
        }
        result.update(_common_metrics(agent_name, input_text, output))
        return result

    return web_agent


def make_doc_agent_node(
    llm: LocalLLMClient,
    *,
    name: str = "doc_agent",
    output_key: str = "doc_context",
) -> NodeFn:
    def doc_agent(state: MASState) -> dict[str, Any]:
        agent_name = name
        base_payload = {
            "user_query": state.get("user_query", ""),
            "repo_path": state.get("repo_path", ""),
            "plan": _clip(state.get("plan", ""), 1800),
        }
        input_text = to_json_prompt(base_payload)
        events = [_node_start(agent_name, state, input_text)]
        with Timer() as node_timer:
            events.append(make_event(agent_name=agent_name, event_type="tool_start", input_text=input_text))
            with Timer() as tool_timer:
                simulated_tool_delay(
                    state.get("tool_delay_profile", "none"),
                    burst=bool(state.get("burst_tool_return", False)),
                )
                files = safe_list_files(state["repo_path"], max_files=180)
                doc_files = [
                    path
                    for path in files
                    if path.lower().endswith((".md", ".rst", ".txt"))
                    or path.lower() in {"pyproject.toml", "setup.cfg", "tox.ini"}
                ][:8]
                contents = safe_read_files(state["repo_path"], doc_files, max_chars_per_file=2500)
            tool_output = to_json_prompt({"doc_files": doc_files, "read_files": list(contents)})
            events.append(
                make_event(
                    agent_name=agent_name,
                    event_type="tool_end",
                    input_text=input_text,
                    output_text=tool_output,
                    duration_sec=tool_timer.duration,
                    extra={"doc_count": len(doc_files), "read_count": len(contents)},
                )
            )
            payload = base_payload | {"doc_files": doc_files, "doc_contents": contents}
            output, llm_events, llm_duration = _llm_call(
                llm, agent_name=agent_name, prompt_name="doc_agent", payload=payload, state=state
            )
            events.extend(llm_events)
        events.append(_node_end(agent_name, state, input_text, output, node_timer.duration))
        result = {
            output_key: output,
            "trace_events": events,
            "tool_wait_time": {agent_name: tool_timer.duration},
            "llm_time": {agent_name: llm_duration},
            "request_metadata": {agent_name: build_request_metadata(state, agent_name)},
        }
        result.update(_common_metrics(agent_name, input_text + tool_output, output))
        return result

    return doc_agent


def make_repo_search_agent_node(
    llm: LocalLLMClient,
    *,
    name: str = "repo_search_agent",
    output_key: str = "repo_search_context",
) -> NodeFn:
    def repo_search_agent(state: MASState) -> dict[str, Any]:
        agent_name = name
        base_payload = {
            "user_query": state.get("user_query", ""),
            "repo_path": state.get("repo_path", ""),
            "plan": _clip(state.get("plan", ""), 1800),
            "review_result": _clip(state.get("review_result", ""), 1800),
        }
        input_text = to_json_prompt(base_payload)
        events = [_node_start(agent_name, state, input_text)]
        with Timer() as node_timer:
            events.append(make_event(agent_name=agent_name, event_type="tool_start", input_text=input_text))
            with Timer() as tool_timer:
                simulated_tool_delay(
                    state.get("tool_delay_profile", "none"),
                    burst=bool(state.get("burst_tool_return", False)),
                )
                files = safe_list_files(state["repo_path"], max_files=4000)
                ranked = _rank_files(
                    files,
                    state.get("user_query", "") + "\n" + state.get("review_result", ""),
                    (".py", ".pyi", ".pxd", ".pyx", ".rst", ".md", ".toml", ".cfg"),
                    12,
                )
                contents = safe_read_files(state["repo_path"], ranked, max_chars_per_file=2200)
            tool_output = to_json_prompt({"ranked_files": ranked, "read_files": list(contents)})
            events.append(
                make_event(
                    agent_name=agent_name,
                    event_type="tool_end",
                    input_text=input_text,
                    output_text=tool_output,
                    duration_sec=tool_timer.duration,
                    extra={"file_count": len(files), "read_count": len(contents)},
                )
            )
            payload = base_payload | {"ranked_files": ranked, "file_contents": contents}
            output, llm_events, llm_duration = _llm_call(
                llm, agent_name=agent_name, prompt_name="repo_search_agent", payload=payload, state=state
            )
            events.extend(llm_events)
        events.append(_node_end(agent_name, state, input_text, output, node_timer.duration))
        result = {
            output_key: output,
            "trace_events": events,
            "tool_wait_time": {agent_name: tool_timer.duration},
            "llm_time": {agent_name: llm_duration},
            "request_metadata": {agent_name: build_request_metadata(state, agent_name)},
        }
        result.update(_common_metrics(agent_name, input_text + tool_output, output))
        return result

    return repo_search_agent


def make_merge_context_node(*, name: str = "merge_context") -> NodeFn:
    def merge_context(state: MASState) -> dict[str, Any]:
        agent_name = name
        shared = _shared_context(state)
        digest = hashlib.sha256(shared.encode("utf-8")).hexdigest()
        events = [
            make_event(
                agent_name=agent_name,
                event_type="barrier_start",
                parents=_deps(state, agent_name),
                children=_children(state, agent_name),
                input_text=shared,
                extra={"parallel_group": "evidence_collection", "barrier_for": _deps(state, agent_name)},
            ),
            _node_start(agent_name, state, shared),
            _node_end(agent_name, state, shared, digest, 0.0),
            make_event(
                agent_name=agent_name,
                event_type="barrier_end",
                parents=_deps(state, agent_name),
                children=_children(state, agent_name),
                input_text=shared,
                output_text=digest,
                extra={"parallel_group": "evidence_collection", "shared_context_hash": digest},
            ),
        ]
        return {
            "shared_context_hash": digest,
            "trace_events": events,
            "agent_outputs": {agent_name: digest},
            "agent_status": {agent_name: "finished"},
            "estimated_prompt_tokens": {agent_name: estimate_tokens(shared)},
            "estimated_output_tokens": {agent_name: estimate_tokens(digest)},
            "estimated_kv_tokens": {agent_name: estimate_tokens(shared)},
            "parallel_groups": {
                "evidence_collection": {
                    "parents": _deps(state, agent_name),
                    "barrier": agent_name,
                    "shared_context_hash": digest,
                }
            },
        }

    return merge_context


def merge_context_node(state: MASState) -> dict[str, Any]:
    return make_merge_context_node(name="merge_context")(state)


def old_merge_context_node(state: MASState) -> dict[str, Any]:
    agent_name = "merge_context"
    shared = _shared_context(state)
    digest = hashlib.sha256(shared.encode("utf-8")).hexdigest()
    events = [
        make_event(
            agent_name=agent_name,
            event_type="barrier_start",
            parents=_deps(state, agent_name),
            children=_children(state, agent_name),
            input_text=shared,
            extra={"parallel_group": "evidence_collection", "barrier_for": _deps(state, agent_name)},
        ),
        _node_start(agent_name, state, shared),
        _node_end(agent_name, state, shared, digest, 0.0),
        make_event(
            agent_name=agent_name,
            event_type="barrier_end",
            parents=_deps(state, agent_name),
            children=_children(state, agent_name),
            input_text=shared,
            output_text=digest,
            extra={"parallel_group": "evidence_collection", "shared_context_hash": digest},
        ),
    ]
    return {
        "shared_context_hash": digest,
        "trace_events": events,
        "agent_outputs": {agent_name: digest},
        "agent_status": {agent_name: "finished"},
        "estimated_prompt_tokens": {agent_name: estimate_tokens(shared)},
        "estimated_output_tokens": {agent_name: estimate_tokens(digest)},
        "estimated_kv_tokens": {agent_name: estimate_tokens(shared)},
        "parallel_groups": {
            "evidence_collection": {
                "parents": _deps(state, agent_name),
                "barrier": agent_name,
                "shared_context_hash": digest,
            }
        },
    }


def make_coder_node(llm: LocalLLMClient, *, name: str, prompt_name: str, output_key: str) -> NodeFn:
    def coder(state: MASState) -> dict[str, Any]:
        shared = _shared_context(state)
        payload = {
            "user_query": state.get("user_query", ""),
            "repo_path": state.get("repo_path", ""),
            "shared_context_hash": state.get("shared_context_hash", ""),
            "shared_context": json.loads(shared),
            "web_context": _clip(state.get("web_context", ""), 1600),
            "doc_context": _clip(state.get("doc_context", ""), 2200),
        }
        input_text = to_json_prompt(payload)
        events = [_node_start(name, state, input_text)]
        with Timer() as node_timer:
            output, llm_events, llm_duration = _llm_call(
                llm, agent_name=name, prompt_name=prompt_name, payload=payload, state=state
            )
            events.extend(llm_events)
        events.append(_node_end(name, state, input_text, output, node_timer.duration))
        result = {
            output_key: output,
            "trace_events": events,
            "llm_time": {name: llm_duration},
            "request_metadata": {name: build_request_metadata(state, name)},
        }
        result.update(_common_metrics(name, input_text, output))
        return result

    return coder


def make_reviewer_node(llm: LocalLLMClient) -> NodeFn:
    def reviewer(state: MASState) -> dict[str, Any]:
        agent_name = "reviewer"
        payload = {
            "user_query": state.get("user_query", ""),
            "plan": _clip(state.get("plan", ""), 2200),
            "file_context": _clip(state.get("file_context", ""), 3800),
            "web_context": _clip(state.get("web_context", ""), 1600),
            "doc_context": _clip(state.get("doc_context", ""), 2400),
            "test_log": _clip(state.get("test_log", ""), 3000),
            "candidate_patch_a": _clip(state.get("candidate_patch_a", ""), 2800),
            "candidate_patch_b": _clip(state.get("candidate_patch_b", ""), 2800),
            "candidate_patch_c": _clip(state.get("candidate_patch_c", ""), 2800),
            "retry_count": state.get("retry_count", 0),
            "max_retries": state.get("max_retries", 1),
            "aggregation_note": "Reviewer 输入已做长度上限控制；完整输出仍保存在 agent_outputs 和 trace 统计中。",
        }
        input_text = to_json_prompt(payload)
        events = [
            make_event(
                agent_name=agent_name,
                event_type="barrier_start",
                parents=_deps(state, agent_name),
                children=_children(state, agent_name),
                input_text=input_text,
                extra={"parallel_group": "candidate_generation", "barrier_for": _deps(state, agent_name)},
            ),
            _node_start(agent_name, state, input_text),
        ]
        with Timer() as node_timer:
            output, llm_events, llm_duration = _llm_call(
                llm, agent_name=agent_name, prompt_name="reviewer", payload=payload, state=state
            )
            events.extend(llm_events)
        events.append(_node_end(agent_name, state, input_text, output, node_timer.duration))
        events.append(
            make_event(
                agent_name=agent_name,
                event_type="barrier_end",
                parents=_deps(state, agent_name),
                children=_children(state, agent_name),
                input_text=input_text,
                output_text=output,
                duration_sec=0.0,
                extra={"parallel_group": "candidate_generation"},
            )
        )
        result = {
            "review_result": output,
            "trace_events": events,
            "llm_time": {agent_name: llm_duration},
            "request_metadata": {agent_name: build_request_metadata(state, agent_name)},
            "parallel_groups": {
                "candidate_generation": {
                    "parents": _deps(state, agent_name),
                    "barrier": agent_name,
                    "shared_context_hash": state.get("shared_context_hash", ""),
                }
            },
        }
        result.update(_common_metrics(agent_name, input_text, output))
        return result

    return reviewer


def make_reviewer_round2_node(llm: LocalLLMClient) -> NodeFn:
    def reviewer_round2(state: MASState) -> dict[str, Any]:
        agent_name = "reviewer_round2"
        payload = {
            "user_query": state.get("user_query", ""),
            "shared_context_hash": state.get("shared_context_hash", ""),
            "reviewer_critique": _clip(state.get("reviewer_critique", ""), 3600),
            "candidate_patch_a_round1": _clip(state.get("candidate_patch_a", ""), 2200),
            "candidate_patch_b_round1": _clip(state.get("candidate_patch_b", ""), 2200),
            "candidate_patch_c_round1": _clip(state.get("candidate_patch_c", ""), 2200),
            "candidate_patch_a_round2": _clip(state.get("candidate_patch_a_round2", ""), 3000),
            "candidate_patch_b_round2": _clip(state.get("candidate_patch_b_round2", ""), 3000),
            "candidate_patch_c_round2": _clip(state.get("candidate_patch_c_round2", ""), 3000),
            "test_log": _clip(state.get("test_log", ""), 2800),
        }
        input_text = to_json_prompt(payload)
        events = [
            make_event(
                agent_name=agent_name,
                event_type="barrier_start",
                parents=_deps(state, agent_name),
                children=_children(state, agent_name),
                input_text=input_text,
                extra={"parallel_group": "candidate_revision", "barrier_for": _deps(state, agent_name)},
            ),
            _node_start(agent_name, state, input_text),
        ]
        with Timer() as node_timer:
            output, llm_events, llm_duration = _llm_call(
                llm, agent_name=agent_name, prompt_name="reviewer", payload=payload, state=state
            )
            events.extend(llm_events)
        events.append(_node_end(agent_name, state, input_text, output, node_timer.duration))
        events.append(
            make_event(
                agent_name=agent_name,
                event_type="barrier_end",
                parents=_deps(state, agent_name),
                children=_children(state, agent_name),
                input_text=input_text,
                output_text=output,
                duration_sec=0.0,
                extra={"parallel_group": "candidate_revision"},
            )
        )
        result = {
            "review_result_round2": output,
            "review_result": output,
            "trace_events": events,
            "llm_time": {agent_name: llm_duration},
            "request_metadata": {agent_name: build_request_metadata(state, agent_name)},
        }
        result.update(_common_metrics(agent_name, input_text, output))
        return result

    return reviewer_round2


def broadcast_critique_node(state: MASState) -> dict[str, Any]:
    """把 reviewer critique 广播给第二轮候选 agent，形成 all-gather-like 输入。"""
    agent_name = "broadcast_critique"
    round_id = "round2"
    critique = state.get("review_result", "")
    previous_candidates = "\n\n".join(
        [
            _clip(state.get("candidate_patch_a", ""), 1800),
            _clip(state.get("candidate_patch_b", ""), 1800),
            _clip(state.get("candidate_patch_c", ""), 1800),
        ]
    )
    shared_feedback = to_json_prompt(
        {
            "reviewer_critique": critique,
            "candidate_summaries_round1": previous_candidates,
            "instruction": "第二轮 coder 必须读取原始 shared_context、全部第一轮候选、reviewer 共享批评和自己的上一轮候选。",
        }
    )
    tokens = estimate_tokens(shared_feedback)
    receiver_count = int(state.get("committee_width", 3) or 3)
    redundant_feedback = tokens * max(0, receiver_count - 1)
    events = [
        _node_start(agent_name, state, shared_feedback),
        make_event(
            agent_name=agent_name,
            event_type="feedback_broadcast",
            parents=_deps(state, agent_name),
            children=_children(state, agent_name),
            input_text=shared_feedback,
            output_text=f"redundant_feedback_tokens={redundant_feedback}",
            extra={
                "round_id": round_id,
                "shared_feedback_tokens": tokens,
                "num_agents_receiving_feedback": receiver_count,
                "redundant_feedback_tokens": redundant_feedback,
            },
        ),
        _node_end(agent_name, state, shared_feedback, "broadcast_done", 0.0),
    ]
    retry_count = state.get("retry_count", 0) + 1
    return {
        "retry_count": retry_count,
        "reviewer_critique": shared_feedback,
        "trace_events": events,
        "agent_outputs": {agent_name: shared_feedback},
        "agent_status": {agent_name: "finished"},
        "all_gather_stats": {
            round_id: {
                "round_id": round_id,
                "shared_feedback_tokens": tokens,
                "num_agents_receiving_feedback": receiver_count,
                "redundant_feedback_tokens": redundant_feedback,
                "redundant_prefill_tokens_estimated": redundant_feedback,
                "reviewer_input_tokens": estimate_tokens(critique),
            }
        },
    }


def make_coder_revision_node(
    llm: LocalLLMClient,
    *,
    name: str,
    base_candidate_key: str,
    output_key: str,
    prompt_name: str = "coder_revise",
) -> NodeFn:
    def coder_revise(state: MASState) -> dict[str, Any]:
        shared = _shared_context(state)
        payload = {
            "user_query": state.get("user_query", ""),
            "repo_path": state.get("repo_path", ""),
            "shared_context_hash": state.get("shared_context_hash", ""),
            "original_shared_context": json.loads(shared),
            "candidate_patch_a_round1": _clip(state.get("candidate_patch_a", ""), 2400),
            "candidate_patch_b_round1": _clip(state.get("candidate_patch_b", ""), 2400),
            "candidate_patch_c_round1": _clip(state.get("candidate_patch_c", ""), 2400),
            "reviewer_shared_critique": _clip(state.get("reviewer_critique", ""), 4200),
            "own_previous_candidate": _clip(state.get(base_candidate_key, ""), 3000),
        }
        input_text = to_json_prompt(payload)
        events = [_node_start(name, state, input_text)]
        with Timer() as node_timer:
            output, llm_events, llm_duration = _llm_call(
                llm, agent_name=name, prompt_name=prompt_name, payload=payload, state=state
            )
            events.extend(llm_events)
        events.append(_node_end(name, state, input_text, output, node_timer.duration))
        result = {
            output_key: output,
            "trace_events": events,
            "llm_time": {name: llm_duration},
            "request_metadata": {name: build_request_metadata(state, name)},
        }
        result.update(_common_metrics(name, input_text, output))
        return result

    return coder_revise


def retry_gate_node(state: MASState) -> dict[str, Any]:
    agent_name = "retry_gate"
    input_text = state.get("review_result", "")
    retry_count = state.get("retry_count", 0) + 1
    output_text = f"retry_count={retry_count}"
    events = [
        _node_start(agent_name, state, input_text),
        _node_end(agent_name, state, input_text, output_text, 0.0),
    ]
    return {
        "retry_count": retry_count,
        "trace_events": events,
        "agent_outputs": {agent_name: output_text},
        "agent_status": {agent_name: "finished"},
    }


def make_finalizer_node(llm: LocalLLMClient) -> NodeFn:
    def finalizer(state: MASState) -> dict[str, Any]:
        agent_name = "finalizer"
        payload = {
            "user_query": state.get("user_query", ""),
            "plan": _clip(state.get("plan", ""), 1800),
            "file_context": _clip(state.get("file_context", ""), 3200),
            "web_context": _clip(state.get("web_context", ""), 1200),
            "doc_context": _clip(state.get("doc_context", ""), 1800),
            "test_log": _clip(state.get("test_log", ""), 2400),
            "verify_log": _clip(state.get("verify_log", ""), 2400),
            "candidate_patch_a": _clip(state.get("candidate_patch_a", ""), 2200),
            "candidate_patch_b": _clip(state.get("candidate_patch_b", ""), 2200),
            "candidate_patch_c": _clip(state.get("candidate_patch_c", ""), 2200),
            "candidate_patch_a_round2": _clip(state.get("candidate_patch_a_round2", ""), 2200),
            "candidate_patch_b_round2": _clip(state.get("candidate_patch_b_round2", ""), 2200),
            "candidate_patch_c_round2": _clip(state.get("candidate_patch_c_round2", ""), 2200),
            "review_result": _clip(state.get("review_result", ""), 3000),
            "review_result_round2": _clip(state.get("review_result_round2", ""), 3000),
            "all_gather_stats": state.get("all_gather_stats", {}),
            "shared_context_hash": state.get("shared_context_hash", ""),
            "retry_count": state.get("retry_count", 0),
            "trace_event_count": len(state.get("trace_events", [])),
            "aggregation_note": "Finalizer 输入保留较大聚合窗口；若出现截断，会在字段文本中显式标注。",
        }
        input_text = to_json_prompt(payload)
        events = [_node_start(agent_name, state, input_text)]
        with Timer() as node_timer:
            output, llm_events, llm_duration = _llm_call(
                llm, agent_name=agent_name, prompt_name="finalizer", payload=payload, state=state
            )
            events.extend(llm_events)
        events.append(_node_end(agent_name, state, input_text, output, node_timer.duration))
        result = {
            "final_answer": output,
            "trace_events": events,
            "llm_time": {agent_name: llm_duration},
            "request_metadata": {agent_name: build_request_metadata(state, agent_name)},
        }
        result.update(_common_metrics(agent_name, input_text, output))
        return result

    return finalizer


def review_needs_retry(state: MASState) -> bool:
    """从 reviewer 文本中保守判断是否需要重试。"""
    text = state.get("review_result", "").lower()
    positive_retry = [
        '"whether_need_retry": true',
        '"need_retry": true',
        "whether_need_retry: true",
        "需要重试",
        "retry: true",
    ]
    negative_retry = [
        '"whether_need_retry": false',
        '"need_retry": false',
        "无需重试",
        "不需要重试",
        "retry: false",
    ]
    if any(marker in text for marker in negative_retry):
        return False
    return any(marker in text for marker in positive_retry)

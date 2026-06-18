from __future__ import annotations

import time
import uuid
from typing import Any

from ..software_tools import apply_patch, revert_patch, run_tests, search_code
from ..topologies.base import BaseTopology, TopologyConfig
from ..tracing import estimate_tokens, stable_hash, token_count_source
from ..motifs.base import CompositeMotif, MotifSpec


class FullWorkflowRuntime(CompositeMotif):
    """Runtime reuse layer for full workflows.

    This inherits CompositeMotif only to reuse the shared trace, LLM, tool,
    edge, barrier, and token helpers. Publicly, workflows in this package are
    full workflows, not motif registry entries.
    """


class IssueToVerifiedPatchWorkflow(FullWorkflowRuntime):
    """End-to-end software workflow assembled from existing motif semantics."""

    spec = MotifSpec(
        "issue_to_verified_patch",
        ["centralized", "independent", "decentralized"],
        ["full_workflow", "swebench", "software_engineering", "motif_interaction"],
        "SWE-bench issue -> evidence -> multi-coder candidates -> tests -> review/debug -> final report.",
    )
    composed_from_motifs = [
        "planner_executor",
        "evidence_collection",
        "researcher_synthesizer",
        "debate_reviewer",
        "multi_coder_branch",
        "all_gather_round",
        "retry_debug_loop",
    ]

    def __init__(self, config: TopologyConfig, **deps: Any) -> None:
        super().__init__(config, **deps)
        self.config.mode = "full"
        self.config.motif_name = ""
        self.config.workload_name = self.spec.name
        self.trace.mode = "full"
        self.trace.motif_name = ""
        self.trace.motif_instance_id = ""
        self.trace.composed_from_topologies = list(self.spec.composed_from)
        self.motif_tags = ["full_workflow", self.spec.name, *self.spec.tags]

    def workflow_start(self) -> None:
        BaseTopology.workflow_start(self)
        self.trace.emit(
            event_type="full_workflow_start",
            node_id=f"{self.spec.name}_START",
            node_name=f"{self.spec.name} START",
            node_type="workflow",
            mode="full",
            motif_tags=self.motif_tags,
            topology_role="full_workflow",
            workflow_type="full_software_workflow",
            composed_from_motifs=list(self.composed_from_motifs),
            composed_from_topologies=self.spec.composed_from,
            status="start",
            extra={"description": self.spec.description},
        )

    def workflow_end(self, final_answer: str) -> dict[str, Any]:
        self.trace.emit(
            event_type="full_workflow_end",
            node_id=f"{self.spec.name}_END",
            node_name=f"{self.spec.name} END",
            node_type="workflow",
            mode="full",
            motif_tags=self.motif_tags,
            topology_role="full_workflow",
            workflow_type="full_software_workflow",
            composed_from_motifs=list(self.composed_from_motifs),
            status="success",
            output_hash=stable_hash(final_answer),
            output_tokens_est=estimate_tokens(final_answer),
            token_count_source=token_count_source(),
        )
        return BaseTopology.workflow_end(self, final_answer)

    def topology_summary(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        llm = [event for event in events if event.get("event_type") == "llm_request_end"]
        stages = [event for event in events if event.get("event_type") == "stage_summary"]
        return {
            "full_workflow_name": self.spec.name,
            "workflow_type": "full_software_workflow",
            "motif_name": "",
            "composed_from_motifs": list(self.composed_from_motifs),
            "stage_count": len(stages),
            "full_workflow_input_tokens_est": sum(int(event.get("input_tokens_est") or 0) for event in llm),
            "full_workflow_output_tokens_est": sum(int(event.get("output_tokens_est") or 0) for event in llm),
            "candidate_tokens_total": sum(int(event.get("candidate_tokens_total") or 0) for event in events),
            "selected_candidate_tokens": sum(int(event.get("selected_candidate_tokens") or 0) for event in events),
            "discarded_candidate_tokens": sum(int(event.get("discarded_candidate_tokens") or 0) for event in events),
            "selector_fan_in_tokens": sum(int(event.get("selector_fan_in_tokens") or 0) for event in events),
            "test_tool_wait_ms": sum(int(event.get("tool_wait_ms") or 0) for event in events if event.get("stage") == "test_execution"),
            "potential_reusable_prefix_tokens": sum(int(event.get("potential_reusable_prefix_tokens") or 0) for event in events),
        }

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        task = self._task()
        self.trace.emit(
            event_type="full_workflow_plan",
            node_id="issue_to_verified_patch_plan",
            node_name="Issue-to-Verified-Patch Workflow Plan",
            node_type="workflow",
            mode="full",
            motif_tags=self.motif_tags,
            workflow_type="full_software_workflow",
            composed_from_motifs=list(self.composed_from_motifs),
            swebench_repo=task.get("repo", ""),
            swebench_base_commit=task.get("base_commit", ""),
            repo_dir=str(self.config.repo_path or ""),
            test_command=self._test_command(task),
            workflow_stages=self.stage_plan(),
            bottleneck_fields=self.bottleneck_fields(),
        )

        plan = self._stage_manager_planning(task)
        evidence = self._stage_evidence_collection(task, plan)
        synthesis = self._stage_evidence_synthesis(task, plan, evidence)
        diagnosis = self._stage_diagnosis(task, plan, synthesis)
        candidates = self._stage_parallel_patch_generation(task, plan, synthesis, diagnosis)
        selected = self._stage_patch_selection(task, synthesis, candidates)
        test_result = self._stage_test_execution(task, selected)
        reviewed = self._stage_review_debug_loop(task, synthesis, selected, test_result)
        final = self._stage_final_report(task, synthesis, selected, reviewed, test_result)
        self.emit_edge(src_node="final_report", dst_node="END", artifact_type="final_report", content=final, transfer_type="aggregation")
        return self.workflow_end(final)

    @classmethod
    def stage_plan(cls) -> list[dict[str, str]]:
        return [
            {"stage": "manager_planning", "motif": "planner_executor"},
            {"stage": "codebase_evidence_collection", "motif": "evidence_collection"},
            {"stage": "evidence_synthesis", "motif": "researcher_synthesizer"},
            {"stage": "diagnosis_fix_strategy", "motif": "debate_reviewer"},
            {"stage": "parallel_patch_generation", "motif": "multi_coder_branch"},
            {"stage": "patch_selection", "motif": "all_gather_round"},
            {"stage": "test_execution", "motif": "tool_heavy_branch/tool_stall"},
            {"stage": "review_debug_loop", "motif": "retry_debug_loop"},
            {"stage": "final_report", "motif": "final_synthesizer/fan_in"},
        ]

    @staticmethod
    def bottleneck_fields() -> list[str]:
        return [
            "candidate_tokens_total",
            "selected_candidate_tokens",
            "discarded_candidate_tokens",
            "fan_in_sources",
            "fan_in_tokens",
            "synthesizer_input_tokens",
            "tool_wait_ms",
            "resume_agents",
            "post_tool_resume_burst_size",
            "stable_prefix_tokens",
            "dynamic_suffix_tokens",
            "potential_reusable_prefix_tokens",
        ]

    def _task(self) -> dict[str, Any]:
        swebench = dict(self.config.extra.get("swebench") or {})
        return {
            "instance_id": swebench.get("instance_id") or self.config.instance_id,
            "repo": swebench.get("repo") or "",
            "base_commit": swebench.get("base_commit") or "",
            "problem_statement": swebench.get("problem_statement") or self.config.query,
            "test_patch": swebench.get("test_patch") or "",
            "patch": swebench.get("patch") or "",
            "FAIL_TO_PASS": swebench.get("FAIL_TO_PASS") or [],
            "PASS_TO_PASS": swebench.get("PASS_TO_PASS") or [],
            "extra": swebench.get("extra_fields") or {},
        }

    def _test_command(self, task: dict[str, Any]) -> str:
        explicit = str(self.config.extra.get("test_command") or "").strip()
        if explicit:
            return explicit
        fail_to_pass = task.get("FAIL_TO_PASS") or []
        if fail_to_pass:
            return "python -m pytest " + " ".join(str(item) for item in fail_to_pass)
        return "python -m pytest -q"

    def _meta(
        self,
        *,
        stage: str,
        motif_id: str,
        agent_id: str,
        parents: list[str] | None = None,
        round_id: int = 0,
        critical: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        parent_ids = parents or []
        return {
            "stage": stage,
            "motif_id": motif_id,
            "composed_from_motifs": list(self.composed_from_motifs),
            "agent_id": agent_id,
            "round": round_id,
            "request_id": f"{stage}:{agent_id}:{uuid.uuid4()}",
            "depends_on": list(parent_ids),
            "parent_node_ids": list(parent_ids),
            "dependency_edges": [{"src": parent, "dst": agent_id} for parent in parent_ids],
            "is_critical_path_candidate": critical,
            "critical_path_candidate": critical,
            "criticality": "critical" if critical else "background",
            **extra,
        }

    def _stage_summary(
        self,
        *,
        stage: str,
        motif_id: str,
        start_idx: int,
        start_perf: float,
        fan_in_tokens: int = 0,
        discarded_candidate_tokens: int = 0,
        tool_wait_ms: int = 0,
        stable_prefix_tokens: int = 0,
        dynamic_suffix_tokens: int = 0,
        potential_reusable_prefix_tokens: int = 0,
        **extra: Any,
    ) -> None:
        events = self.trace.events[start_idx:]
        llm = [event for event in events if event.get("event_type") == "llm_request_end"]
        tools = [event for event in events if str(event.get("event_type", "")).startswith("tool_") and str(event.get("event_type", "")).endswith("_end")]
        self.trace.emit(
            event_type="stage_summary",
            node_id=f"{stage}_summary",
            node_name=f"{stage} summary",
            node_type="stage_summary",
            mode="full",
            stage=stage,
            motif_id=motif_id,
            llm_calls=len(llm),
            tool_calls=len(tools),
            input_tokens=sum(int(event.get("input_tokens") or event.get("input_tokens_est") or 0) for event in llm),
            output_tokens=sum(int(event.get("output_tokens") or event.get("output_tokens_est") or 0) for event in llm),
            wall_time_ms=int((time.perf_counter() - start_perf) * 1000),
            start_ts=time.time() - max(0.0, time.perf_counter() - start_perf),
            end_ts=time.time(),
            critical_path_ms=int(sum(float(event.get("duration_sec") or 0.0) * 1000 for event in llm if event.get("critical_path_candidate") or event.get("is_critical_path_candidate"))),
            fan_in_tokens=fan_in_tokens,
            discarded_candidate_tokens=discarded_candidate_tokens,
            tool_wait_ms=tool_wait_ms,
            stable_prefix_tokens=stable_prefix_tokens,
            dynamic_suffix_tokens=dynamic_suffix_tokens,
            potential_reusable_prefix_tokens=potential_reusable_prefix_tokens,
            token_count_source=token_count_source(),
            **extra,
        )

    def _emit_tool(
        self,
        *,
        node_id: str,
        node_name: str,
        stage: str,
        motif_id: str,
        tool_name: str,
        fn: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        parents: list[str] | None = None,
        critical: bool = False,
        tool_stall: bool = False,
    ) -> dict[str, Any]:
        kwargs = kwargs or {}
        start = time.time()
        meta = self._meta(stage=stage, motif_id=motif_id, agent_id=node_id, parents=parents, critical=critical)
        self.trace.emit(event_type="tool_call_start", node_id=node_id, node_name=node_name, node_type="tool", status="start", tool_name=tool_name, tool_start_ts=start, start_ts=start, tool_stalled=tool_stall, **meta)
        result = fn(*args, **kwargs)
        end = time.time()
        text = str(result)
        event = self.trace.emit(
            event_type="tool_call_end",
            node_id=node_id,
            node_name=node_name,
            node_type="tool",
            status=result.get("status", "success") if isinstance(result, dict) else "success",
            duration_sec=round(end - start, 6),
            duration_source="tool_wrapper_measured",
            effective_duration_sec=round(end - start, 6),
            tool_name=tool_name,
            tool_start_ts=start,
            tool_end_ts=end,
            start_ts=start,
            end_ts=end,
            tool_return_ts=end,
            tool_latency_sec=round(end - start, 6),
            tool_wait_ms=int((end - start) * 1000),
            tool_stalled=tool_stall,
            resume_after_tool=False,
            input_size_chars=sum(len(str(arg)) for arg in args) + sum(len(str(value)) for value in kwargs.values()),
            output_size_chars=len(text),
            input_size_tokens=estimate_tokens(" ".join(str(arg) for arg in args)),
            output_size_tokens=estimate_tokens(text),
            output_hash=stable_hash(text),
            tool_result_summary=text[:1000],
            **meta,
        )
        result = dict(result) if isinstance(result, dict) else {"status": "success", "result": result}
        result["_trace_event_id"] = event.get("event_id")
        result["_tool_wait_ms"] = int((end - start) * 1000)
        return result

    def _stage_manager_planning(self, task: dict[str, Any]) -> str:
        stage, motif_id = "manager_planning", "planner_executor"
        start_idx, start = len(self.trace.events), time.perf_counter()
        prompt = f"Instance: {task['instance_id']}\nRepo: {task['repo']}\nBase commit: {task['base_commit']}\nIssue:\n{task['problem_statement']}"
        plan = self.llm_agent(node_id="manager_planner", node_name="Manager Planner", agent_role="planner", system_prompt="Plan an issue-to-verified-patch workflow. Keep dependencies explicit.", user_prompt=prompt, parents=["START"], criticality="critical", extra_metadata=self._meta(stage=stage, motif_id=motif_id, agent_id="manager_planner", parents=["START"], critical=True))
        self.emit_edge(src_node="manager_planner", dst_node="evidence_searchers", artifact_type="plan", content=plan, transfer_type="fan_out")
        self._stage_summary(stage=stage, motif_id=motif_id, start_idx=start_idx, start_perf=start)
        return plan

    def _stage_evidence_collection(self, task: dict[str, Any], plan: str) -> dict[str, str]:
        stage, motif_id = "codebase_evidence_collection", "evidence_collection"
        start_idx, start = len(self.trace.events), time.perf_counter()
        queries = [
            ("repo_searcher", f"{task['problem_statement']} relevant files"),
            ("test_searcher", f"{task['problem_statement']} tests FAIL_TO_PASS {task.get('FAIL_TO_PASS')}"),
            ("history_searcher", f"{task['repo']} {task['base_commit']} regression clues"),
            ("api_doc_searcher", f"{task['problem_statement']} public API behavior"),
        ][: max(2, min(4, self.config.num_agents or 3))]

        def collect(item: tuple[str, str]) -> tuple[str, str]:
            agent, query = item
            self.emit_edge(src_node="manager_planner", dst_node=agent, artifact_type="evidence_query", content=query, transfer_type="broadcast", parallel_group="codebase_evidence", fanout_count=len(queries), recipient_count=len(queries))
            web_results = self.search(
                node_id=f"{agent}_web_search",
                node_name=f"{agent} web_search",
                query=query,
                trace_fields=self._meta(stage=stage, motif_id=motif_id, agent_id=f"{agent}_web_search", parents=["manager_planner"]),
            )
            tool = self._emit_tool(node_id=f"{agent}_search_code", node_name=f"{agent} search_code", stage=stage, motif_id=motif_id, tool_name="search_code", fn=search_code, args=(query, self.config.repo_path), kwargs={"max_files": 30}, parents=["manager_planner"])
            out = self.llm_agent(node_id=agent, node_name=agent, agent_role="researcher", system_prompt="Extract codebase evidence: file paths, symptoms, likely tests, and uncertainty.", user_prompt=f"Plan:\n{plan}\nWeb search result:\n{web_results}\nCode search result:\n{tool}", parents=[f"{agent}_web_search", f"{agent}_search_code"], parallel_group="codebase_evidence", criticality="background", extra_metadata=self._meta(stage=stage, motif_id=motif_id, agent_id=agent, parents=[f"{agent}_web_search", f"{agent}_search_code"], fan_in_sources=[]))
            self.emit_edge(src_node=agent, dst_node="evidence_synthesizer", artifact_type="evidence", content=out, transfer_type="aggregation", parallel_group="codebase_evidence")
            return agent, out

        evidence = dict(self.run_parallel(queries, collect))
        self.barrier(barrier_id="codebase_evidence_fanin", waiting_for_nodes=list(evidence))
        fan_in_tokens = estimate_tokens("\n".join(evidence.values()))
        self._stage_summary(stage=stage, motif_id=motif_id, start_idx=start_idx, start_perf=start, fan_in_tokens=fan_in_tokens, fan_in_sources=list(evidence), synthesizer_input_tokens=fan_in_tokens)
        return evidence

    def _stage_evidence_synthesis(self, task: dict[str, Any], plan: str, evidence: dict[str, str]) -> str:
        stage, motif_id = "evidence_synthesis", "researcher_synthesizer"
        start_idx, start = len(self.trace.events), time.perf_counter()
        synth_input = f"Issue:\n{task['problem_statement']}\nPlan:\n{plan}\nEvidence:\n" + "\n".join(f"{k}: {v}" for k, v in evidence.items())
        synth = self.llm_agent(node_id="evidence_synthesizer", node_name="Evidence Synthesizer", agent_role="synthesizer", system_prompt="Synthesize evidence into concise localization and test hypotheses.", user_prompt=synth_input, parents=list(evidence), criticality="critical", extra_metadata=self._meta(stage=stage, motif_id=motif_id, agent_id="evidence_synthesizer", parents=list(evidence), critical=True, fan_in_sources=list(evidence), fan_in_tokens=estimate_tokens(synth_input), synthesizer_input_tokens=estimate_tokens(synth_input)))
        self.emit_edge(src_node="evidence_synthesizer", dst_node="diagnosis_debate", artifact_type="synthesis", content=synth, transfer_type="handoff")
        self._stage_summary(stage=stage, motif_id=motif_id, start_idx=start_idx, start_perf=start, fan_in_tokens=estimate_tokens(synth_input), synthesizer_input_tokens=estimate_tokens(synth_input))
        return synth

    def _stage_diagnosis(self, task: dict[str, Any], plan: str, synthesis: str) -> str:
        stage, motif_id = "diagnosis_fix_strategy", "debate_reviewer"
        start_idx, start = len(self.trace.events), time.perf_counter()
        roles = ["root_cause", "minimal_patch", "regression_risk"]

        def argue(role: str) -> tuple[str, str]:
            node = f"diagnosis_{role}"
            out = self.llm_agent(node_id=node, node_name=node, agent_role="reviewer", system_prompt=f"Argue the {role} hypothesis for this bug fix.", user_prompt=f"Issue:\n{task['problem_statement']}\nPlan:\n{plan}\nEvidence synthesis:\n{synthesis}", parents=["evidence_synthesizer"], parallel_group="diagnosis_debate", criticality="background", extra_metadata=self._meta(stage=stage, motif_id=motif_id, agent_id=node, parents=["evidence_synthesizer"]))
            self.emit_edge(src_node=node, dst_node="diagnosis_consensus", artifact_type="diagnosis_hypothesis", content=out, transfer_type="aggregation", parallel_group="diagnosis_debate")
            return node, out

        hypotheses = dict(self.run_parallel(roles, argue))
        self.barrier(barrier_id="diagnosis_debate_barrier", waiting_for_nodes=list(hypotheses))
        consensus_input = "\n".join(f"{k}: {v}" for k, v in hypotheses.items())
        consensus = self.llm_agent(node_id="diagnosis_consensus", node_name="Diagnosis Consensus", agent_role="aggregator", system_prompt="Resolve conflicting hypotheses into one fix strategy.", user_prompt=consensus_input, parents=list(hypotheses), criticality="critical", extra_metadata=self._meta(stage=stage, motif_id=motif_id, agent_id="diagnosis_consensus", parents=list(hypotheses), critical=True, fan_in_tokens=estimate_tokens(consensus_input)))
        self._stage_summary(stage=stage, motif_id=motif_id, start_idx=start_idx, start_perf=start, fan_in_tokens=estimate_tokens(consensus_input))
        return consensus

    def _stage_parallel_patch_generation(self, task: dict[str, Any], plan: str, synthesis: str, diagnosis: str) -> dict[str, str]:
        stage, motif_id = "parallel_patch_generation", "multi_coder_branch"
        start_idx, start = len(self.trace.events), time.perf_counter()
        n = max(2, min(6, self.config.num_agents or 3))

        def code(i: int) -> tuple[str, str]:
            node = f"patch_coder_{i}"
            prompt = f"Issue:\n{task['problem_statement']}\nPlan:\n{plan}\nEvidence:\n{synthesis}\nFix strategy:\n{diagnosis}\nProduce candidate patch {i}."
            out = self.llm_agent(node_id=node, node_name=f"Patch Coder {i}", agent_role="coder", system_prompt="Produce a concrete candidate patch or patch sketch. Keep it inspectable.", user_prompt=prompt, parents=["diagnosis_consensus"], parallel_group="patch_candidates", criticality="background", extra_metadata=self._meta(stage=stage, motif_id=motif_id, agent_id=node, parents=["diagnosis_consensus"], candidate_id=node))
            self.emit_edge(src_node=node, dst_node="patch_selector", artifact_type="candidate_patch", content=out, transfer_type="aggregation", parallel_group="patch_candidates")
            return node, out

        candidates = dict(self.run_parallel(list(range(1, n + 1)), code))
        self.barrier(barrier_id="patch_candidate_barrier", waiting_for_nodes=list(candidates))
        candidate_tokens = {node: estimate_tokens(text) for node, text in candidates.items()}
        total = sum(candidate_tokens.values())
        self.trace.emit(event_type="candidate_set_summary", node_id="candidate_set_summary", node_name="Candidate Patch Set Summary", node_type="artifact", stage=stage, motif_id=motif_id, candidate_count=len(candidates), candidate_tokens_by_id=candidate_tokens, candidate_tokens_total=total, selected_candidate_tokens=0, discarded_candidate_tokens=total)
        self._stage_summary(stage=stage, motif_id=motif_id, start_idx=start_idx, start_perf=start, fan_in_tokens=total, discarded_candidate_tokens=total, candidate_tokens_total=total)
        return candidates

    def _stage_patch_selection(self, task: dict[str, Any], synthesis: str, candidates: dict[str, str]) -> dict[str, Any]:
        stage, motif_id = "patch_selection", "selector/all_gather/fan_in"
        start_idx, start = len(self.trace.events), time.perf_counter()
        selector_input = f"Issue:\n{task['problem_statement']}\nEvidence:\n{synthesis}\nCandidates:\n" + "\n".join(f"{k}:\n{v}" for k, v in candidates.items())
        selected_text = self.llm_agent(node_id="patch_selector", node_name="Patch Selector", agent_role="reviewer", system_prompt="Read all candidate patches, select one, and explain discarded alternatives.", user_prompt=selector_input, parents=list(candidates), criticality="critical", extra_metadata=self._meta(stage=stage, motif_id=motif_id, agent_id="patch_selector", parents=list(candidates), critical=True, candidate_count=len(candidates), candidate_tokens_total=estimate_tokens("\n".join(candidates.values())), selector_input_tokens=estimate_tokens(selector_input)))
        selected_id = next(iter(candidates)) if candidates else ""
        selected_tokens = estimate_tokens(candidates.get(selected_id, ""))
        discarded = max(0, estimate_tokens("\n".join(candidates.values())) - selected_tokens)
        self.trace.emit(event_type="patch_selection_summary", node_id="patch_selection_summary", node_name="Patch Selection Summary", node_type="artifact", stage=stage, motif_id=motif_id, selected_candidate_id=selected_id, candidate_tokens_total=selected_tokens + discarded, selected_candidate_tokens=selected_tokens, discarded_candidate_tokens=discarded, selector_fan_in_tokens=estimate_tokens(selector_input))
        self.emit_edge(src_node="patch_selector", dst_node="test_runner", artifact_type="selected_patch", content=selected_text, transfer_type="handoff")
        self._stage_summary(stage=stage, motif_id=motif_id, start_idx=start_idx, start_perf=start, fan_in_tokens=estimate_tokens(selector_input), discarded_candidate_tokens=discarded, selected_candidate_tokens=selected_tokens, candidate_tokens_total=selected_tokens + discarded)
        return {"selected_candidate_id": selected_id, "selected_patch": selected_text, "selected_candidate_tokens": selected_tokens, "discarded_candidate_tokens": discarded}

    def _stage_test_execution(self, task: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
        stage, motif_id = "test_execution", "tool_heavy_branch/tool_stall"
        start_idx, start = len(self.trace.events), time.perf_counter()
        patch_text = str(selected.get("selected_patch") or "")
        apply_result = self._emit_tool(node_id="apply_patch_tool", node_name="Apply Patch", stage=stage, motif_id=motif_id, tool_name="apply_patch", fn=apply_patch, args=(patch_text, self.config.repo_path), kwargs={"dry_run": bool(self.config.extra.get("dry_run_patch", True))}, parents=["patch_selector"])
        test_result = self._emit_tool(node_id="run_tests_tool", node_name="Run Tests", stage=stage, motif_id=motif_id, tool_name="run_tests", fn=run_tests, args=(self._test_command(task), self.config.repo_path), kwargs={"dry_run": bool(self.config.extra.get("dry_run_tests", True))}, parents=["apply_patch_tool"], critical=True, tool_stall=True)
        resume_agents = ["test_reviewer", "debugger_r0", "final_report"]
        self.trace.emit(event_type="tool_resume_burst_summary", node_id="test_tool_resume_burst", node_name="Test Tool Resume Burst", node_type="artifact", stage=stage, motif_id=motif_id, tool_wait_ms=test_result.get("_tool_wait_ms", 0), resume_agents=resume_agents, post_tool_resume_burst_size=len(resume_agents), tool_stall=True, resume_after_tool=True)
        self._stage_summary(stage=stage, motif_id=motif_id, start_idx=start_idx, start_perf=start, tool_wait_ms=int(test_result.get("_tool_wait_ms", 0)), resume_agents=resume_agents, post_tool_resume_burst_size=len(resume_agents))
        return {"apply_patch": apply_result, "run_tests": test_result}

    def _stage_review_debug_loop(self, task: dict[str, Any], synthesis: str, selected: dict[str, Any], test_result: dict[str, Any]) -> dict[str, Any]:
        stage, motif_id = "review_debug_loop", "retry_debug_loop"
        start_idx, start = len(self.trace.events), time.perf_counter()
        stable_prefix = f"Issue:\n{task['problem_statement']}\nEvidence:\n{synthesis}\nSelected patch:\n{selected.get('selected_patch')}\nTest command:\n{self._test_command(task)}"
        dynamic = str(test_result.get("run_tests", {}))
        stable_tokens = estimate_tokens(stable_prefix)
        result: dict[str, Any] = {"rounds": []}
        max_rounds = max(1, int(self.config.max_retries or 1))
        current_patch = str(selected.get("selected_patch") or "")
        for round_id in range(max_rounds):
            prompt = f"{stable_prefix}\nFailure log / current dynamic context:\n{dynamic}"
            review = self.llm_agent(node_id=f"test_reviewer_r{round_id}", node_name=f"Test Reviewer R{round_id}", agent_role="reviewer", system_prompt="Review selected patch and test failure. Decide whether to debug.", user_prompt=prompt, parents=["run_tests_tool" if round_id == 0 else f"debugger_r{round_id-1}"], retry_count=round_id, criticality="critical", extra_metadata=self._meta(stage=stage, motif_id=motif_id, agent_id=f"test_reviewer_r{round_id}", parents=["run_tests_tool" if round_id == 0 else f"debugger_r{round_id-1}"], round_id=round_id, critical=True, stable_prefix_tokens=stable_tokens, dynamic_suffix_tokens=estimate_tokens(dynamic), potential_reusable_prefix_tokens=stable_tokens, reread_sources=["selected_patch", "test_log", "evidence_synthesis"]))
            debug = self.llm_agent(node_id=f"debugger_r{round_id}", node_name=f"Debugger R{round_id}", agent_role="debugger", system_prompt="Produce a revised patch sketch from review and failure log.", user_prompt=f"{prompt}\nReview:\n{review}", parents=[f"test_reviewer_r{round_id}"], retry_count=round_id, criticality="critical", extra_metadata=self._meta(stage=stage, motif_id=motif_id, agent_id=f"debugger_r{round_id}", parents=[f"test_reviewer_r{round_id}"], round_id=round_id, critical=True, stable_prefix_tokens=stable_tokens, dynamic_suffix_tokens=estimate_tokens(dynamic + review), potential_reusable_prefix_tokens=stable_tokens, reread_sources=["selected_patch", "failure_log", "review_feedback", "evidence_synthesis"]))
            current_patch = debug
            result["rounds"].append({"round": round_id, "review": review, "debug": debug, "input_tokens": estimate_tokens(prompt), "stable_prefix_tokens": stable_tokens, "dynamic_suffix_tokens": estimate_tokens(dynamic), "potential_reusable_prefix_tokens": stable_tokens})
            dynamic = f"{dynamic}\nReview:\n{review}\nRevision:\n{debug}"
        self.trace.emit(event_type="debug_loop_prefix_summary", node_id="debug_loop_prefix_summary", node_name="Debug Loop Prefix Summary", node_type="artifact", stage=stage, motif_id=motif_id, loop_rounds=len(result["rounds"]), stable_prefix_tokens=stable_tokens, dynamic_suffix_tokens=estimate_tokens(dynamic), potential_reusable_prefix_tokens=stable_tokens * len(result["rounds"]), reread_sources=["issue", "evidence", "selected_patch", "test_command", "failure_log"])
        self._stage_summary(stage=stage, motif_id=motif_id, start_idx=start_idx, start_perf=start, stable_prefix_tokens=stable_tokens, dynamic_suffix_tokens=estimate_tokens(dynamic), potential_reusable_prefix_tokens=stable_tokens * len(result["rounds"]))
        result["final_patch"] = current_patch
        return result

    def _stage_final_report(self, task: dict[str, Any], synthesis: str, selected: dict[str, Any], reviewed: dict[str, Any], test_result: dict[str, Any]) -> str:
        stage, motif_id = "final_report", "final_synthesizer/fan_in"
        start_idx, start = len(self.trace.events), time.perf_counter()
        final_input = f"Issue:\n{task['problem_statement']}\nEvidence:\n{synthesis}\nSelected patch:\n{selected}\nTest result:\n{test_result}\nReview/debug:\n{reviewed}"
        final = self.llm_agent(node_id="final_report", node_name="Final Report", agent_role="finalizer", system_prompt="Write final report with candidate patch, test result, remaining risks, and trace-relevant bottleneck notes.", user_prompt=final_input, parents=["patch_selector", "run_tests_tool", "debug_loop_prefix_summary"], criticality="critical", extra_metadata=self._meta(stage=stage, motif_id=motif_id, agent_id="final_report", parents=["patch_selector", "run_tests_tool", "debug_loop_prefix_summary"], critical=True, fan_in_tokens=estimate_tokens(final_input)))
        self._emit_tool(node_id="revert_patch_tool", node_name="Revert Patch", stage=stage, motif_id=motif_id, tool_name="revert_patch", fn=revert_patch, args=(self.config.repo_path,), kwargs={"dry_run": bool(self.config.extra.get("dry_run_patch", True))}, parents=["final_report"])
        self._stage_summary(stage=stage, motif_id=motif_id, start_idx=start_idx, start_perf=start, fan_in_tokens=estimate_tokens(final_input))
        return final


def build_workflow(config: TopologyConfig, **deps) -> IssueToVerifiedPatchWorkflow:
    return IssueToVerifiedPatchWorkflow(config, **deps)

"""Composite MAS motifs built from the base topology runtime.

The classes in this module keep one TraceContext and reuse BaseTopology's
LLM/tool/edge/barrier helpers. Each motif also instantiates the named base
topology motifs with shared dependencies so later full workflows can call or
inspect the same component graph without creating another trace system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..topologies.base import BaseTopology, TopologyConfig, parse_json_maybe
from ..topologies.centralized import build_motif as centralized_motif
from ..topologies.decentralized_debate import build_motif as decentralized_motif
from ..topologies.independent import build_motif as independent_motif
from ..topologies.single_agent import build_motif as single_motif
from ..topologies.hybrid import build_motif as hybrid_motif
from ..tracing import estimate_tokens, stable_hash, token_count_source


BASE_MOTIF_BUILDERS = {
    "single": single_motif,
    "independent": independent_motif,
    "centralized": centralized_motif,
    "decentralized": decentralized_motif,
    "hybrid": hybrid_motif,
}


@dataclass(frozen=True)
class MotifSpec:
    name: str
    composed_from: list[str]
    tags: list[str]
    description: str
    roles: list[str] = field(default_factory=list)


class CompositeMotif(BaseTopology):
    spec = MotifSpec("composite", [], ["composite"], "generic composite motif")

    def __init__(self, config: TopologyConfig, **deps: Any) -> None:
        super().__init__(config, **deps)
        self.config.mode = "motif"
        self.config.motif_name = self.spec.name
        self.config.composed_from_topologies = list(self.spec.composed_from)
        self.trace.mode = "motif"
        self.trace.motif_name = self.spec.name
        self.trace.motif_instance_id = self.config.instance_id
        self.trace.parent_motif_id = self.config.parent_motif_id
        self.trace.composed_from_topologies = list(self.spec.composed_from)
        self.motif_tags = list(dict.fromkeys(["composite", self.spec.name, *self.spec.tags]))
        self.component_motifs = {
            name: BASE_MOTIF_BUILDERS[name](config, **deps)
            for name in self.spec.composed_from
            if name in BASE_MOTIF_BUILDERS
        }

    def workflow_start(self) -> None:
        super().workflow_start()
        self.trace.emit(
            event_type="motif_start",
            node_id=f"{self.spec.name}_START",
            node_name=f"{self.spec.name} START",
            node_type="workflow",
            motif_tags=self.motif_tags,
            topology_role="workflow",
            composed_from_topologies=self.spec.composed_from,
            status="start",
            extra={"description": self.spec.description, "component_motifs": sorted(self.component_motifs)},
        )

    def workflow_end(self, final_answer: str) -> dict[str, Any]:
        self.trace.emit(
            event_type="motif_end",
            node_id=f"{self.spec.name}_END",
            node_name=f"{self.spec.name} END",
            node_type="workflow",
            motif_tags=self.motif_tags,
            topology_role="workflow",
            composed_from_topologies=self.spec.composed_from,
            status="success",
            output_hash=stable_hash(final_answer),
            output_tokens_est=estimate_tokens(final_answer),
            token_count_source=token_count_source(),
        )
        return super().workflow_end(final_answer)

    def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
        parent = self.config.parent_motif_id
        if payload.get("parent_motif_id"):
            self.config.parent_motif_id = str(payload["parent_motif_id"])
            self.trace.parent_motif_id = self.config.parent_motif_id
        try:
            return super().invoke(payload)
        finally:
            self.config.parent_motif_id = parent
            self.trace.parent_motif_id = parent

    def emit_control(
        self,
        *,
        node_id: str,
        node_name: str,
        event_type: str = "motif_control",
        round_id: int | None = None,
        manager_round_id: int | None = None,
        peer_round_id: int | None = None,
        **fields: Any,
    ) -> None:
        self.trace.emit(
            event_type=event_type,
            node_id=node_id,
            node_name=node_name,
            node_type=fields.pop("node_type", "manager"),
            round_id=round_id,
            manager_round_id=manager_round_id,
            peer_round_id=peer_round_id,
            motif_tags=self.motif_tags,
            composed_from_topologies=fields.pop("composed_from_topologies", self.spec.composed_from),
            **fields,
        )

    def finalizer(self, content: str, *, parents: list[str] | None = None) -> str:
        self.emit_edge(src_node=(parents or ["motif"])[0], dst_node="finalizer", artifact_type="summary", content=content, transfer_type="aggregation")
        final = self.call_llm(
            node_id="finalizer",
            node_name="Finalizer",
            node_type="agent",
            agent_role="finalizer",
            prompt_template="finalizer.md",
            system_prompt="Produce the final answer for this composite MAS motif.",
            user_prompt=content,
            parents=parents or [],
            criticality="critical",
            extra_metadata={"motif_name": self.spec.name},
        )
        self.emit_edge(src_node="finalizer", dst_node="END", artifact_type="final", content=final, transfer_type="aggregation")
        return final

    def maybe_tool_evidence(self, *, node_id: str, node_name: str, query: str, force: bool = False) -> str:
        if force or not self.use_react_agents():
            results = self.search(node_id=f"{node_id}_search", node_name=f"{node_name} Search", query=query)
            return self._clip_prompt_text(str(results), max_chars=2000)
        return "ReAct agent may call tools if needed."

    def _clip_prompt_text(self, text: str, *, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        digest = stable_hash(text)[:16]
        return f"{text[:max_chars]}\n[truncated_for_prompt hash={digest} original_chars={len(text)}]"

    def llm_agent(
        self,
        *,
        node_id: str,
        node_name: str,
        agent_role: str,
        system_prompt: str,
        user_prompt: str,
        parents: list[str] | None = None,
        round_id: int | None = None,
        manager_round_id: int | None = None,
        peer_round_id: int | None = None,
        retry_count: int = 0,
        parallel_group: str | None = None,
        criticality: str = "near_critical",
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.call_agent(
            node_id=node_id,
            node_name=node_name,
            node_type="agent",
            agent_role=agent_role,
            prompt_template=f"{agent_role}.md",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parents=parents or [],
            round_id=round_id,
            manager_round_id=manager_round_id,
            peer_round_id=peer_round_id,
            retry_count=retry_count,
            parallel_group=parallel_group,
            criticality=criticality,
            extra_metadata=extra_metadata or {},
        )

    def topology_summary(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        motif_events = [e for e in events if e.get("mode") == "motif"]
        tools = [e for e in events if str(e.get("event_type", "")).startswith("tool_")]
        llm = [e for e in events if e.get("event_type") == "llm_request_end"]
        edges = [e for e in events if e.get("node_type") == "edge"]
        return {
            "motif_name": self.spec.name,
            "motif_count": len([e for e in events if e.get("event_type") == "motif_start"]),
            "motif_duration_sec": max([float(e.get("relative_time_sec") or 0) for e in motif_events] or [0.0]),
            "motif_input_tokens_est": sum(int(e.get("input_tokens_est") or 0) for e in llm),
            "motif_output_tokens_est": sum(int(e.get("output_tokens_est") or 0) for e in llm),
            "motif_artifact_tokens_est": sum(int(e.get("artifact_tokens_est") or 0) for e in edges),
            "motif_tool_time_sec": round(sum(float(e.get("effective_duration_sec") or e.get("duration_sec") or 0) for e in tools), 6),
            "motif_llm_time_sec": round(sum(float(e.get("duration_sec") or 0) for e in llm), 6),
            "motif_barrier_wait_sec": round(sum(float(e.get("barrier_wait_sec") or 0) for e in events), 6),
            "motif_retry_count": sum(int(e.get("retry_count") or 0) for e in events),
            "composed_from_topologies": self.spec.composed_from,
            "shared_evidence_read_tokens_est": sum(int(e.get("shared_evidence_read_tokens_est") or 0) for e in events),
            "shared_evidence_write_tokens_est": sum(int(e.get("shared_evidence_write_tokens_est") or 0) for e in events),
            "duplicated_context_tokens_est": sum(int(e.get("duplicated_context_tokens_est") or 0) for e in events),
            "retry_loop_count": len([e for e in events if int(e.get("debug_loop_count") or 0) > 0]),
            "handoff_count": sum(int(e.get("handoff_count") or 0) for e in events),
        }


class PlannerExecutorMotif(CompositeMotif):
    spec = MotifSpec("planner_executor", ["centralized"], ["manager_worker", "plan_execution"], "Planner -> Executor -> Finalizer")

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        plan = self.llm_agent(node_id="planner", node_name="Planner", agent_role="planner", system_prompt="Create a concise executable plan.", user_prompt=self.config.query, parents=["START"], manager_round_id=0, criticality="critical")
        self.emit_edge(src_node="planner", dst_node="executor", artifact_type="plan", content=plan, transfer_type="plan_to_execution", manager_round_id=0)
        execution_input = f"Task:\n{self.config.query}\nPlan artifact:\n{plan}"
        self.emit_control(node_id="plan_artifact", node_name="Plan Artifact", artifact_type="plan", plan_tokens_est=estimate_tokens(plan), executor_input_tokens_est=estimate_tokens(execution_input))
        result = self.llm_agent(node_id="executor", node_name="Executor", agent_role="executor", system_prompt="Execute the plan and report concrete results.", user_prompt=execution_input, parents=["planner"], manager_round_id=0, criticality="critical")
        final = self.finalizer(result, parents=["executor"])
        return self.workflow_end(final)


class EvidenceCollectionMotif(CompositeMotif):
    spec = MotifSpec("evidence_collection", ["independent"], ["parallel_evidence", "aggregation"], "Planner -> specialist evidence agents -> merge")

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        plan = self.llm_agent(node_id="planner", node_name="Planner", agent_role="planner", system_prompt="Plan evidence collection across search, repo, docs, and tools.", user_prompt=self.config.query, parents=["START"], criticality="critical")
        roles = [("searcher", "Searcher"), ("repo_searcher", "RepoSearcher"), ("doc_searcher", "DocSearcher"), ("tool_agent", "ToolAgent")]
        for role, label in roles:
            self.emit_edge(src_node="planner", dst_node=role, artifact_type="evidence_group", content=plan, transfer_type="broadcast", parallel_group="evidence_collectors", fanout_count=len(roles), recipient_count=len(roles))

        def run_role(item: tuple[str, str]) -> tuple[str, str]:
            role, label = item
            evidence = self.maybe_tool_evidence(node_id=role, node_name=label, query=f"{self.config.query}\n{label}", force=role in {"searcher", "tool_agent"})
            out = self.llm_agent(node_id=role, node_name=label, agent_role=role, system_prompt=f"You are {label}; collect evidence only.", user_prompt=f"Plan:\n{plan}\nEvidence:\n{evidence}", parents=["planner"], parallel_group="evidence_collectors", extra_metadata={"evidence_group": role})
            self.emit_edge(src_node=role, dst_node="evidence_merge", artifact_type="evidence", content=out, transfer_type="aggregation", parallel_group="evidence_collectors")
            return role, out

        outputs = dict(self.run_parallel(roles, run_role))
        self.barrier(barrier_id="evidence_merge_barrier", waiting_for_nodes=list(outputs), round_id=0)
        merge_input = "\n".join(f"{k}: {v}" for k, v in outputs.items())
        self.emit_control(node_id="evidence_metrics", node_name="Evidence Metrics", evidence_tokens=estimate_tokens(merge_input), merge_input_tokens=estimate_tokens(merge_input), aggregation_tokens_est=estimate_tokens(merge_input))
        merged = self.llm_agent(node_id="evidence_merge", node_name="EvidenceMerge", agent_role="aggregator", system_prompt="Merge evidence, deduplicate, and preserve source uncertainty.", user_prompt=merge_input, parents=list(outputs), criticality="critical")
        final = self.finalizer(merged, parents=["evidence_merge"])
        return self.workflow_end(final)


class ResearcherSynthesizerMotif(CompositeMotif):
    spec = MotifSpec("researcher_synthesizer", ["independent"], ["research", "synthesis"], "Parallel researchers -> synthesizer")

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        n = max(1, self.config.num_agents)
        for i in range(1, n + 1):
            self.emit_edge(src_node="START", dst_node=f"researcher_{i}", artifact_type="task", content=self.config.query, transfer_type="broadcast", parallel_group="researchers", fanout_count=n, recipient_count=n)

        def run_researcher(i: int) -> tuple[str, str]:
            node = f"researcher_{i}"
            evidence = self.maybe_tool_evidence(node_id=node, node_name=f"Researcher-{i}", query=f"{self.config.query}\nresearch angle {i}")
            depth = ["very concise", "concise", "moderately detailed", "detailed", "exhaustive"][min(i - 1, 4)]
            budget = [192, 256, 384, 512, 768][min(i - 1, 4)]
            out = self.llm_agent(node_id=node, node_name=f"Researcher-{i}", agent_role="researcher", system_prompt=f"Research angle {i} independently. Produce a {depth} analysis; vary depth materially from other researchers.", user_prompt=f"Task:\n{self.config.query}\nEvidence:\n{evidence}", parents=["START"], parallel_group="researchers", extra_metadata={"request_max_output_tokens": budget, "research_depth": depth})
            self.emit_edge(src_node=node, dst_node="synthesizer", artifact_type="research_artifact", content=out, transfer_type="aggregation", parallel_group="researchers")
            return node, out

        outputs = dict(self.run_parallel(list(range(1, n + 1)), run_researcher))
        self.barrier(barrier_id="synthesis_barrier", waiting_for_nodes=list(outputs), round_id=0)
        synth_input = "\n".join(outputs.values())
        self.emit_control(node_id="research_tokens", node_name="Research Tokens", redundant_input_tokens=estimate_tokens(self.config.query) * max(0, n - 1), research_artifact_tokens=estimate_tokens(synth_input), synthesis_tokens=estimate_tokens(synth_input))
        synth = self.llm_agent(node_id="synthesizer", node_name="Synthesizer", agent_role="synthesizer", system_prompt="Synthesize independent research into a coherent answer.", user_prompt=synth_input, parents=list(outputs), criticality="critical")
        final = self.finalizer(synth, parents=["synthesizer"])
        return self.workflow_end(final)


class GeneratorVerifierMotif(CompositeMotif):
    spec = MotifSpec("generator_verifier", ["centralized"], ["two_stage_control", "verification"], "Generator -> Verifier with optional revision")

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        draft = self.llm_agent(node_id="generator", node_name="Generator", agent_role="generator", system_prompt="Generate a candidate solution.", user_prompt=self.config.query, parents=["START"], manager_round_id=0, criticality="critical")
        self.emit_edge(src_node="generator", dst_node="verifier", artifact_type="candidate", content=draft, transfer_type="handoff", manager_round_id=0)
        verdict = self.llm_agent(node_id="verifier", node_name="Verifier", agent_role="verifier", system_prompt="Verify the candidate. Return APPROVE or REQUEST_CHANGES with reasons.", user_prompt=f"Task:\n{self.config.query}\nCandidate:\n{draft}", parents=["generator"], manager_round_id=0, criticality="critical")
        decision = self._verifier_decision(verdict)
        retry_count = 0
        self.emit_control(node_id="verifier_decision", node_name="Verifier Decision", event_type="verifier_decision", verifier_decision=decision, retry_count=retry_count, critical_path_length=2)
        result = draft
        if decision == "request_changes" and self.config.max_retries > 0:
            retry_count = 1
            self.emit_edge(src_node="verifier", dst_node="generator_revise", artifact_type="review", content=verdict, transfer_type="revision", retry_count=retry_count)
            result = self.llm_agent(node_id="generator_revise", node_name="Generator-Revise", agent_role="generator", system_prompt="Revise the candidate according to verifier feedback.", user_prompt=f"Original:\n{draft}\nFeedback:\n{verdict}", parents=["verifier"], retry_count=retry_count, manager_round_id=1, criticality="critical")
            verdict = self.llm_agent(node_id="verifier_round2", node_name="Verifier-Round2", agent_role="verifier", system_prompt="Verify revised candidate.", user_prompt=result, parents=["generator_revise"], retry_count=retry_count, manager_round_id=1, criticality="critical")
            self.emit_control(node_id="verifier_decision_r2", node_name="Verifier Decision R2", event_type="verifier_decision", verifier_decision=self._verifier_decision(verdict), retry_count=retry_count, critical_path_length=4)
        final = self.finalizer(f"Candidate:\n{result}\nVerification:\n{verdict}", parents=["verifier_round2" if retry_count else "verifier"])
        return self.workflow_end(final)

    def _verifier_decision(self, text: str) -> str:
        lowered = text.lower()
        return "request_changes" if "request_changes" in lowered or "revise" in lowered or "change" in lowered else "approve"


class CoderReviewerMotif(GeneratorVerifierMotif):
    spec = MotifSpec("coder_reviewer", ["centralized"], ["generator_verifier_motif", "code_review", "revision_loop"], "Coder -> Reviewer with optional revise/review loop")

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        code = self.llm_agent(node_id="coder", node_name="Coder", agent_role="coder", system_prompt="Write a concise implementation plan or patch sketch.", user_prompt=self.config.query, parents=["START"], manager_round_id=0, criticality="critical")
        review = self.llm_agent(node_id="reviewer", node_name="Reviewer", agent_role="reviewer", system_prompt="Review the code. APPROVE or REQUEST_CHANGES.", user_prompt=code, parents=["coder"], manager_round_id=0, criticality="critical")
        decision = self._verifier_decision(review)
        loop_count = 0
        self.emit_control(node_id="review_decision", node_name="Review Decision", event_type="review_decision", verifier_decision=decision, review_loop_count=loop_count, reviewer_input_tokens=estimate_tokens(code))
        result = code
        if decision == "request_changes" and self.config.max_retries > 0:
            loop_count = 1
            self.emit_edge(src_node="reviewer", dst_node="coder_revise", artifact_type="review", content=review, transfer_type="revision", retry_count=1)
            result = self.llm_agent(node_id="coder_revise", node_name="Coder-Revise", agent_role="coder", system_prompt="Revise the code from review feedback.", user_prompt=f"Code:\n{code}\nReview:\n{review}", parents=["reviewer"], retry_count=1, manager_round_id=1, criticality="critical")
            review = self.llm_agent(node_id="reviewer_round2", node_name="Reviewer-Round2", agent_role="reviewer", system_prompt="Review revised code.", user_prompt=result, parents=["coder_revise"], retry_count=1, manager_round_id=1, criticality="critical")
        self.emit_control(node_id="review_loop_metrics", node_name="Review Loop Metrics", event_type="review_loop_metrics", review_loop_count=loop_count, retry_amplification=round((estimate_tokens(result) + estimate_tokens(review)) / max(1, estimate_tokens(code)), 4), revision_tokens=estimate_tokens(result), retry_count=loop_count)
        final = self.finalizer(f"Code:\n{result}\nReview:\n{review}", parents=["reviewer_round2" if loop_count else "reviewer"])
        return self.workflow_end(final)


class MultiCoderBranchMotif(CompositeMotif):
    spec = MotifSpec("multi_coder_branch", ["independent", "centralized"], ["parallel_candidates", "reviewer_selector"], "Planner -> multiple coders -> reviewer/selector")

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        plan = self.llm_agent(node_id="planner", node_name="Planner", agent_role="planner", system_prompt="Split this coding task into candidate implementation strategies.", user_prompt=self.config.query, parents=["START"], criticality="critical")
        n = max(2, self.config.num_agents)

        def run_coder(i: int) -> tuple[str, str]:
            node = f"coder_{chr(64 + i)}"
            self.emit_edge(src_node="planner", dst_node=node, artifact_type="plan", content=plan, transfer_type="broadcast", parallel_group="coder_candidates", fanout_count=n, recipient_count=n)
            out = self.llm_agent(node_id=node, node_name=f"Coder-{chr(64 + i)}", agent_role="coder", system_prompt=f"Produce candidate implementation {i}.", user_prompt=f"Task:\n{self.config.query}\nPlan:\n{plan}", parents=["planner"], parallel_group="coder_candidates")
            self.emit_edge(src_node=node, dst_node="selector", artifact_type="candidate", content=out, transfer_type="aggregation", parallel_group="coder_candidates")
            return node, out

        outputs = dict(self.run_parallel(list(range(1, n + 1)), run_coder))
        self.barrier(barrier_id="candidate_selection_barrier", waiting_for_nodes=list(outputs), round_id=0)
        candidates = "\n".join(f"{k}: {v}" for k, v in outputs.items())
        self.emit_control(node_id="candidate_metrics", node_name="Candidate Metrics", candidate_agent_count=n, candidate_count=n, candidate_tokens=estimate_tokens(candidates), reviewer_aggregation_tokens=estimate_tokens(candidates), aggregation_tokens_est=estimate_tokens(candidates))
        selected = self.llm_agent(node_id="selector", node_name="Reviewer / Selector", agent_role="reviewer", system_prompt="Review candidates and select the best one with rationale.", user_prompt=candidates, parents=list(outputs), criticality="critical")
        final = self.finalizer(selected, parents=["selector"])
        return self.workflow_end(final)


class DebateReviewerMotif(CompositeMotif):
    spec = MotifSpec("debate_reviewer", ["decentralized"], ["debate", "review_consensus"], "Candidate/evidence -> reviewer debate -> consensus")

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        n = max(2, self.config.num_agents)
        initial = self.llm_agent(node_id="candidate", node_name="Candidate", agent_role="generator", system_prompt="Summarize the candidate to be reviewed.", user_prompt=self.config.query, parents=["START"], criticality="critical")
        messages: dict[str, str] = {}

        def initial_review(i: int) -> tuple[str, str]:
            node = f"reviewer_{i}"
            self.emit_edge(src_node="candidate", dst_node=node, artifact_type="candidate", content=initial, transfer_type="broadcast", parallel_group="reviewers", fanout_count=n, recipient_count=n)
            out = self.llm_agent(node_id=node, node_name=f"Reviewer-{i}", agent_role="reviewer", system_prompt=f"You are reviewer {i}; critique independently.", user_prompt=initial, parents=["candidate"], peer_round_id=0, parallel_group="reviewers")
            return node, out

        messages = dict(self.run_parallel(list(range(1, n + 1)), initial_review))
        for r in range(1, self.config.debate_rounds + 1):
            def debate_step(i: int) -> tuple[str, str]:
                node = f"reviewer_{i}"
                peers = {k: v for k, v in messages.items() if k != node}
                peer_text = "\n".join(f"{k}: {v}" for k, v in peers.items())
                for peer, content in peers.items():
                    self.emit_edge(src_node=peer, dst_node=f"{node}_r{r}", artifact_type="peer_message", content=content, transfer_type="broadcast" if self.config.communication_topology == "all_to_all" else "message_passing", peer_round_id=r, round_id=r, parallel_group=f"review_debate_r{r}", recipient_count=1)
                return node, self.llm_agent(node_id=f"{node}_r{r}", node_name=f"Reviewer-{i}-Round-{r}", agent_role="reviewer", system_prompt="Update your review after peer critiques.", user_prompt=f"Candidate:\n{initial}\nPeer reviews:\n{peer_text}", parents=list(peers), peer_round_id=r, round_id=r, parallel_group=f"review_debate_r{r}", extra_metadata={"peer_messages": peer_text})
            messages = dict(self.run_parallel(list(range(1, n + 1)), debate_step))
            self.barrier(barrier_id=f"review_debate_barrier_r{r}", waiting_for_nodes=[f"reviewer_{i}_r{r}" for i in range(1, n + 1)], peer_round_id=r)
        consensus_input = "\n".join(messages.values())
        self.emit_control(node_id="debate_metrics", node_name="Debate Metrics", debate_rounds=self.config.debate_rounds, peer_message_edges=len([e for e in self.trace.events if e.get("artifact_type") == "peer_message"]), peer_message_tokens_est=estimate_tokens(consensus_input) * max(1, n - 1), consensus_input_tokens=estimate_tokens(consensus_input))
        consensus = self.llm_agent(node_id="consensus", node_name="Consensus", agent_role="aggregator", system_prompt="Produce review consensus.", user_prompt=consensus_input, parents=list(messages), criticality="critical")
        final = self.finalizer(consensus, parents=["consensus"])
        return self.workflow_end(final)


class ToolSpecialistTeamMotif(CompositeMotif):
    spec = MotifSpec("tool_specialist_team", ["centralized"], ["dynamic_selection", "tool_heavy"], "Manager -> tool specialists -> merge")

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        specialists = [("tool_search", "ToolAgent-Search"), ("tool_repo", "ToolAgent-Repo"), ("tool_doc", "ToolAgent-Doc"), ("tool_test", "ToolAgent-Test")]
        selected = specialists[: max(1, min(len(specialists), self.config.max_selected_agents or len(specialists)))]
        candidate_routes = [name for name, _ in specialists]
        self.emit_control(node_id="manager", node_name="Manager", event_type="manager_decision", selected_agent_count=len(selected), candidate_agent_count=len(specialists), selected_workers=[s[0] for s in selected], candidate_routes=candidate_routes)

        def run_tool(item: tuple[str, str]) -> tuple[str, str]:
            node, label = item
            evidence = self.maybe_tool_evidence(node_id=node, node_name=label, query=f"{self.config.query}\n{label}", force=True)
            self.emit_control(node_id=f"{node}_tool_metrics", node_name=f"{label} Tool Metrics", node_type="tool", tool_mode=self.config.tool_mode, tool_name="search", tool_result_hash=stable_hash(evidence), measured_tool_time=0.0, tool_stall_events=1)
            out = self.llm_agent(node_id=node, node_name=label, agent_role="tool_agent", system_prompt=f"Use tool evidence for {label}.", user_prompt=f"Task:\n{self.config.query}\nTool evidence:\n{evidence}", parents=["manager"], parallel_group="tool_specialists")
            self.emit_edge(src_node=node, dst_node="tool_result_merge", artifact_type="tool_result", content=out, transfer_type="aggregation", parallel_group="tool_specialists")
            return node, out

        outputs = dict(self.run_parallel(selected, run_tool))
        self.barrier(barrier_id="tool_merge_barrier", waiting_for_nodes=list(outputs))
        merged = self.llm_agent(node_id="tool_result_merge", node_name="ToolResultMerge", agent_role="aggregator", system_prompt="Merge tool specialist results.", user_prompt="\n".join(outputs.values()), parents=list(outputs), criticality="critical")
        final = self.finalizer(merged, parents=["tool_result_merge"])
        return self.workflow_end(final)


class AllGatherRoundMotif(CompositeMotif):
    spec = MotifSpec("all_gather_round", ["decentralized"], ["all_gather", "broadcast"], "Agents produce messages -> broadcast all -> aggregate")

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        n = max(2, self.config.num_agents)

        def produce(i: int) -> tuple[str, str]:
            node = f"agent_{i}"
            return node, self.llm_agent(node_id=node, node_name=f"Agent-{i}", agent_role="peer_agent", system_prompt="Produce your local message.", user_prompt=self.config.query, parents=["START"], parallel_group="all_gather_initial")

        messages = dict(self.run_parallel(list(range(1, n + 1)), produce))
        self.barrier(barrier_id="all_gather_initial_barrier", waiting_for_nodes=list(messages))
        broadcast_tokens = 0

        def receive(i: int) -> tuple[str, str]:
            nonlocal broadcast_tokens
            node = f"agent_{i}"
            context = []
            for src, msg in messages.items():
                self.emit_edge(src_node=src, dst_node=f"{node}_gathered", artifact_type="peer_message", content=msg, transfer_type="broadcast", round_id=1, peer_round_id=1, parallel_group="all_gather_broadcast", fanout_count=n, recipient_count=n)
                broadcast_tokens += estimate_tokens(msg)
                context.append(f"{src}: {msg}")
            peer_text = "\n".join(context)
            return node, self.llm_agent(node_id=f"{node}_gathered", node_name=f"Agent-{i}-Gathered", agent_role="peer_agent", system_prompt="Update using all gathered peer messages.", user_prompt=peer_text, parents=list(messages), round_id=1, peer_round_id=1, parallel_group="all_gather_broadcast", extra_metadata={"peer_messages": peer_text})

        gathered = dict(self.run_parallel(list(range(1, n + 1)), receive))
        self.barrier(barrier_id="all_gather_broadcast_barrier", waiting_for_nodes=[f"agent_{i}_gathered" for i in range(1, n + 1)], peer_round_id=1)
        aggregate_input = "\n".join(gathered.values())
        self.emit_control(node_id="all_gather_metrics", node_name="All Gather Metrics", broadcast_tokens_est=broadcast_tokens, recipient_count=n, all_gather_tokens=estimate_tokens(aggregate_input), duplicated_context_tokens_est=max(0, broadcast_tokens - estimate_tokens("\n".join(messages.values()))))
        aggregate = self.llm_agent(node_id="aggregator", node_name="Aggregator", agent_role="aggregator", system_prompt="Aggregate all-gather updates.", user_prompt=aggregate_input, parents=list(gathered), criticality="critical")
        final = self.finalizer(aggregate, parents=["aggregator"])
        return self.workflow_end(final)


class SharedEvidenceStoreMotif(CompositeMotif):
    spec = MotifSpec("shared_evidence_store", ["independent", "centralized"], ["shared_memory", "dataflow"], "Writers -> shared evidence store -> readers")

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        writer_count = max(2, min(self.config.writer_count or self.config.num_agents, 8))

        def write(i: int) -> tuple[str, str]:
            node = f"writer_{i}"
            evidence = self.maybe_tool_evidence(node_id=node, node_name=f"Writer-{i}", query=f"{self.config.query}\nwriter {i}")
            out = self.llm_agent(node_id=node, node_name=f"Writer-{i}", agent_role="writer", system_prompt="Write evidence into the shared store.", user_prompt=f"Task:\n{self.config.query}\nEvidence:\n{evidence}", parents=["START"], parallel_group="store_writers")
            self.emit_edge(src_node=node, dst_node="shared_evidence_store", artifact_type="memory_write", content=out, transfer_type="memory_write", parallel_group="store_writers")
            self.trace.emit(event_type="memory_write", node_id=f"{node}_write", node_name=f"{node} write", node_type="memory", motif_tags=self.motif_tags, artifact_type="memory_write", artifact_version=i, artifact_hash=stable_hash(out), write_set=[f"evidence_{i}"], shared_evidence_write_tokens_est=estimate_tokens(out))
            return f"evidence_{i}", out

        store = dict(self.run_parallel(list(range(1, writer_count + 1)), write))
        self.barrier(barrier_id="shared_store_write_barrier", waiting_for_nodes=[f"writer_{i}" for i in range(1, writer_count + 1)])
        store_text = "\n".join(f"{k}: {v}" for k, v in store.items())

        def read(i: int) -> tuple[str, str]:
            node = f"reader_{i}"
            self.trace.emit(event_type="memory_read", node_id=f"{node}_read", node_name=f"{node} read", node_type="memory", motif_tags=self.motif_tags, artifact_type="memory_read", artifact_version=len(store), artifact_hash=stable_hash(store), read_set=list(store), stale_read=False, shared_evidence_read_tokens_est=estimate_tokens(store_text))
            out = self.llm_agent(node_id=node, node_name=f"Reader-{i}", agent_role="reader", system_prompt="Read shared evidence and extract implications.", user_prompt=store_text, parents=["shared_evidence_store"], parallel_group="store_readers")
            self.emit_edge(src_node=node, dst_node="finalizer", artifact_type="memory_read", content=out, transfer_type="aggregation", parallel_group="store_readers")
            return node, out

        reader_count = max(2, min(self.config.reader_count, 16))
        readers = dict(self.run_parallel(list(range(1, reader_count + 1)), read))
        final = self.finalizer("\n".join(readers.values()), parents=list(readers))
        return self.workflow_end(final)


class RetryDebugLoopMotif(GeneratorVerifierMotif):
    spec = MotifSpec("retry_debug_loop", ["centralized"], ["generator_verifier_motif", "debug_loop", "retry"], "Executor/Coder -> Tester/Verifier -> Debugger -> revise")

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        result = self.llm_agent(node_id="executor", node_name="Executor / Coder", agent_role="executor", system_prompt="Execute the task or draft code.", user_prompt=self.config.query, parents=["START"], manager_round_id=0, criticality="critical")
        max_loops = max(1, self.config.max_retries or 1)
        passed = False
        test_result = ""
        loop = 0
        for loop in range(max_loops + 1):
            test_result = self.llm_agent(node_id=f"tester_r{loop}", node_name=f"Tester / Verifier R{loop}", agent_role="tester", system_prompt="Test or verify the result. Say PASS if acceptable, otherwise FAIL with reason.", user_prompt=result, parents=["executor" if loop == 0 else f"executor_revise_r{loop}"], retry_count=loop, manager_round_id=loop, criticality="critical")
            passed = "pass" in test_result.lower() and "fail" not in test_result.lower()
            self.trace.emit(event_type="test_result", node_id=f"test_result_r{loop}", node_name=f"Test Result R{loop}", node_type="artifact", motif_tags=self.motif_tags, artifact_type="test_result", test_result="pass" if passed else "fail", failure_reason="" if passed else test_result[:500], retry_count=loop, debug_loop_count=loop, wasted_work_estimate=estimate_tokens(result))
            if passed or loop >= max_loops:
                break
            debug = self.llm_agent(node_id=f"debugger_r{loop}", node_name=f"Debugger R{loop}", agent_role="debugger", system_prompt="Diagnose failure and propose a minimal fix.", user_prompt=f"Result:\n{result}\nFailure:\n{test_result}", parents=[f"tester_r{loop}"], retry_count=loop + 1, manager_round_id=loop, criticality="critical")
            self.emit_edge(src_node=f"debugger_r{loop}", dst_node=f"executor_revise_r{loop+1}", artifact_type="debug_feedback", content=debug, transfer_type="revision", retry_count=loop + 1)
            result = self.llm_agent(node_id=f"executor_revise_r{loop+1}", node_name=f"Executor-Revise R{loop+1}", agent_role="executor", system_prompt="Revise from debug feedback.", user_prompt=f"Previous:\n{result}\nDebug:\n{debug}", parents=[f"debugger_r{loop}"], retry_count=loop + 1, manager_round_id=loop + 1, criticality="critical")
        final = self.finalizer(f"Result:\n{result}\nTest:\n{test_result}", parents=[f"tester_r{loop}"])
        return self.workflow_end(final)


class ToolResumeContentionMesoMotif(CompositeMotif):
    spec = MotifSpec(
        "tool_resume_contention_meso",
        ["independent", "centralized"],
        ["meso_workload", "tool_resume", "critical_path", "compound_contention_opportunity"],
        "Planner plus critical coder/reviewer/finalizer path with delayed tool resume branches.",
    )
    composed_from_motifs = ["evidence_collection", "tool_specialist_team", "multi_coder_branch", "coder_reviewer"]

    def meso_meta(
        self,
        *,
        role: str,
        criticality: str,
        stage: str = "none",
        branch_id: str = "",
        tool_stalled: bool = False,
        resume_after_tool: bool = False,
        resume_group_id: str = "",
        expected_target: str | None = None,
        parents: list[str] | None = None,
        dependency_edges: list[dict[str, str]] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        target = expected_target
        if target is None:
            target = self.config.critical_stage_marker if self.config.resume_phase_policy in {"overlap_reviewer", "overlap_finalizer"} else "none"
        base = {
            "meso_workload_name": self.spec.name,
            "composed_from_motifs": list(self.composed_from_motifs),
            "composed_from_topologies": list(self.spec.composed_from),
            "workload_role": role,
            "criticality": criticality,
            "critical_path_candidate": criticality == "critical",
            "critical_stage": stage,
            "background_branch_id": branch_id,
            "tool_stalled": tool_stalled,
            "resume_after_tool": resume_after_tool,
            "resume_group_id": resume_group_id,
            "resume_phase_policy": self.config.resume_phase_policy,
            "controlled_tool_delay_sec": self.config.controlled_tool_delay_sec,
            "expected_overlap_target": target,
            "expected_overlap_window_sec": self.expected_overlap_window_sec(),
            "parent_node_ids": parents or [],
            "dependency_edges": dependency_edges or [],
        }
        base.update(extra)
        return base

    def expected_overlap_window_sec(self) -> float:
        if self.config.resume_phase_policy in {"overlap_reviewer", "overlap_finalizer"}:
            return max(0.5, min(float(self.config.controlled_tool_delay_sec or 0.0), 5.0))
        return 0.0

    def critical_llm(
        self,
        *,
        node_id: str,
        node_name: str,
        agent_role: str,
        stage: str,
        system_prompt: str,
        user_prompt: str,
        parents: list[str],
    ) -> str:
        meta = self.meso_meta(
            role="merge_or_finalizer" if stage == "finalizer" else "critical_path",
            criticality="critical",
            stage=stage,
            parents=parents,
            dependency_edges=[{"src": parent, "dst": node_id} for parent in parents],
            critical_request_marker=stage if stage in {"reviewer", "finalizer"} else "",
            nearby_background_resume_expected=self.config.contention_labeling
            and self.config.background_resume_enabled
            and self.config.resume_phase_policy in {"overlap_reviewer", "overlap_finalizer"},
            overlap_analysis_status="planned" if self.config.contention_labeling else "unavailable",
        )
        return self.llm_agent(
            node_id=node_id,
            node_name=node_name,
            agent_role=agent_role,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            parents=parents,
            criticality="critical",
            extra_metadata=meta,
        )

    def background_resume_llm(
        self,
        *,
        node_id: str,
        node_name: str,
        branch_id: str,
        tool_event_id: str,
        user_prompt: str,
        parents: list[str],
    ) -> str:
        target = {
            "overlap_reviewer": "reviewer",
            "overlap_finalizer": "finalizer",
            "before_critical": "reviewer",
            "after_critical": "none",
        }.get(self.config.resume_phase_policy, "none")
        meta = self.meso_meta(
            role="background_tool_branch",
            criticality="background",
            branch_id=branch_id,
            tool_stalled=True,
            resume_after_tool=True,
            resume_group_id="resume_group_tool_evidence",
            parents=parents,
            dependency_edges=[{"src": parent, "dst": node_id} for parent in parents],
            background_resume_request=True,
            resumed_from_tool_event_id=tool_event_id,
            intended_to_overlap_with=target,
            expected_resume_to_critical_delta_sec=0.0
            if self.config.resume_phase_policy in {"overlap_reviewer", "overlap_finalizer"}
            else None,
        )
        return self.llm_agent(
            node_id=node_id,
            node_name=node_name,
            agent_role="evidence_processor",
            system_prompt="Resume after tool return and convert delayed evidence into reviewer-ready notes.",
            user_prompt=user_prompt,
            parents=parents,
            parallel_group="background_tool_resume",
            criticality="background",
            extra_metadata=meta,
        )

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        planner_meta = self.meso_meta(role="critical_path", criticality="critical", stage="planner", parents=["START"], dependency_edges=[{"src": "START", "dst": "planner"}])
        plan = self.llm_agent(
            node_id="planner",
            node_name="Manager / Planner",
            agent_role="planner",
            system_prompt="Plan a realistic software workflow with critical coding/review and delayed evidence branches.",
            user_prompt=self.config.query,
            parents=["START"],
            criticality="critical",
            extra_metadata=planner_meta,
        )

        def run_critical_path(_: int) -> tuple[str, str, str]:
            self.emit_edge(src_node="planner", dst_node="critical_coder", artifact_type="plan", content=plan, transfer_type="handoff")
            code = self.critical_llm(
                node_id="critical_coder",
                node_name="CriticalWorker / Coder",
                agent_role="coder",
                stage="coder",
                system_prompt="Produce the main candidate patch or implementation plan without waiting for late tools.",
                user_prompt=f"Task:\n{self.config.query}\nPlan:\n{plan}",
                parents=["planner"],
            )
            self.emit_edge(src_node="critical_coder", dst_node="critical_reviewer", artifact_type="candidate", content=code, transfer_type="handoff")
            review = self.critical_llm(
                node_id="critical_reviewer",
                node_name="Reviewer",
                agent_role="reviewer",
                stage="reviewer",
                system_prompt="Review the critical-path candidate. Mark risks that late evidence may affect.",
                user_prompt=f"Plan:\n{plan}\nCandidate:\n{code}",
                parents=["critical_coder"],
            )
            return "critical_reviewer", code, review

        def run_background_tool_branch(index: int) -> tuple[str, str]:
            node = f"tool_branch_{index}"
            label = ["Search", "Repo", "Doc", "Test"][index - 1]
            branch_id = f"background_tool_{index}"
            self.emit_edge(src_node="planner", dst_node=node, artifact_type="evidence_plan", content=plan, transfer_type="broadcast", parallel_group="background_tool_branches", fanout_count=self.config.tool_branch_width, recipient_count=self.config.tool_branch_width)
            tool_meta = self.meso_meta(
                role="background_tool_branch",
                criticality="background",
                branch_id=branch_id,
                tool_stalled=True,
                parents=["planner"],
                dependency_edges=[{"src": "planner", "dst": node}, {"src": node, "dst": f"resume_evidence_processor_{index}"}],
            )
            tool_event = self.controlled_delay_tool(
                node_id=f"{node}_controlled_delay",
                node_name=f"ToolAgent-{label} Controlled Delay",
                configured_delay_sec=self.config.controlled_tool_delay_sec,
                trace_fields=tool_meta,
            )
            if not self.config.background_resume_enabled:
                return node, ""
            evidence = f"Delayed {label} evidence after controlled tool return for: {self.config.query}"
            resumed = self.background_resume_llm(
                node_id=f"resume_evidence_processor_{index}",
                node_name=f"ResumeEvidenceProcessor-{index}",
                branch_id=branch_id,
                tool_event_id=str(tool_event.get("event_id")),
                user_prompt=f"Plan:\n{plan}\nTool evidence:\n{evidence}",
                parents=[f"{node}_controlled_delay"],
            )
            self.emit_edge(src_node=f"resume_evidence_processor_{index}", dst_node="evidence_merge", artifact_type="resumed_evidence", content=resumed, transfer_type="aggregation", parallel_group="background_tool_resume")
            return f"resume_evidence_processor_{index}", resumed

        def run_parallel_branch(_: int) -> tuple[str, str]:
            branch_meta = self.meso_meta(
                role="background_parallel_branch",
                criticality="background",
                branch_id="parallel_coder_A",
                parents=["planner"],
                dependency_edges=[{"src": "planner", "dst": "background_coder_A"}, {"src": "background_coder_A", "dst": "optional_selector"}],
            )
            self.emit_edge(src_node="planner", dst_node="background_coder_A", artifact_type="plan", content=plan, transfer_type="broadcast", parallel_group="background_parallel_branch")
            out = self.llm_agent(
                node_id="background_coder_A",
                node_name="Background Coder A",
                agent_role="coder",
                system_prompt="Produce an alternate candidate in the background.",
                user_prompt=f"Task:\n{self.config.query}\nPlan:\n{plan}",
                parents=["planner"],
                parallel_group="background_parallel_branch",
                criticality="background",
                extra_metadata=branch_meta,
            )
            self.emit_edge(src_node="background_coder_A", dst_node="optional_selector", artifact_type="candidate", content=out, transfer_type="aggregation", parallel_group="background_parallel_branch")
            return "background_coder_A", out

        work_items: list[tuple[str, int]] = [("critical", 0), ("parallel", 0)]
        for i in range(1, max(1, min(4, self.config.tool_branch_width)) + 1):
            work_items.append(("tool", i))

        def dispatch(item: tuple[str, int]) -> tuple[str, Any]:
            kind, index = item
            if kind == "critical":
                return kind, run_critical_path(index)
            if kind == "parallel":
                return kind, run_parallel_branch(index)
            return f"tool_{index}", run_background_tool_branch(index)

        results = dict(self.run_parallel(work_items, dispatch))
        critical_node, code, review = results["critical"]
        background_outputs = {
            str(value[0]): str(value[1])
            for key, value in results.items()
            if key.startswith("tool_") and value and value[1]
        }
        optional_parallel = results.get("parallel")
        if optional_parallel:
            background_outputs[str(optional_parallel[0])] = str(optional_parallel[1])
        waiting_nodes = [critical_node, *background_outputs.keys()]
        self.barrier(barrier_id="evidence_resume_fanin_barrier", waiting_for_nodes=waiting_nodes, round_id=0)
        merge_input = "\n".join(f"{node}: {text}" for node, text in background_outputs.items())
        merge_meta = self.meso_meta(
            role="merge_or_finalizer",
            criticality="merge",
            stage="none",
            parents=list(background_outputs),
            dependency_edges=[{"src": node, "dst": "evidence_merge"} for node in background_outputs],
            downstream_fanin_node_present=True,
        )
        self.emit_control(
            node_id="tool_resume_contention_plan",
            node_name="Tool Resume Contention Plan",
            event_type="meso_trace_plan",
            node_type="workflow",
            **self.meso_meta(
                role="merge_or_finalizer",
                criticality="merge",
                parents=waiting_nodes,
                dependency_edges=[{"src": node, "dst": "finalizer"} for node in waiting_nodes],
                expected_overlap_window_sec=self.expected_overlap_window_sec(),
                critical_stage_marker=self.config.critical_stage_marker,
            ),
        )
        merged = self.llm_agent(
            node_id="evidence_merge",
            node_name="EvidenceMerge",
            agent_role="aggregator",
            system_prompt="Merge delayed resumed evidence with optional background candidates.",
            user_prompt=merge_input or "No background resume output.",
            parents=list(background_outputs),
            criticality="merge",
            extra_metadata=merge_meta,
        )
        final_input = f"Critical candidate:\n{code}\nReview:\n{review}\nDelayed evidence merge:\n{merged}"
        final = self.critical_llm(
            node_id="finalizer",
            node_name="Finalizer",
            agent_role="finalizer",
            stage="finalizer",
            system_prompt="Finalize the workflow using critical-path review and any late resumed evidence.",
            user_prompt=final_input,
            parents=[critical_node, "evidence_merge"],
        )
        self.emit_edge(src_node="finalizer", dst_node="END", artifact_type="final", content=final, transfer_type="aggregation")
        return self.workflow_end(final)


class Week3MesoMotif(CompositeMotif):
    composed_from_motifs: list[str] = []

    def week3_meta(
        self,
        *,
        role: str,
        criticality: str = "unknown",
        stage: str = "none",
        parents: list[str] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        meta = {
            "workflow_name": self.spec.name,
            "meso_workload_name": self.spec.name,
            "composed_from_motifs": list(self.composed_from_motifs),
            "composed_from_topologies": list(self.spec.composed_from),
            "workload_role": role,
            "criticality": criticality,
            "critical_path_candidate": criticality == "critical",
            "critical_stage": stage,
            "parent_node_ids": parents or [],
            "dependency_edges": [{"src": p, "dst": extra.get("dst_node", "")} for p in (parents or [])],
        }
        meta.update(extra)
        return meta

    def block_meta(
        self,
        *,
        private_text: str,
        shared_blocks: dict[str, str],
        agent_id: str,
        round_id: int,
        group_id: str,
    ) -> dict[str, Any]:
        block_ids = list(shared_blocks)
        block_tokens = {block_id: estimate_tokens(text) for block_id, text in shared_blocks.items()}
        return {
            "agent_id": agent_id,
            "round_id": round_id,
            "private_history_tokens": estimate_tokens(private_text),
            "shared_block_ids": block_ids,
            "shared_block_tokens": block_tokens,
            "shared_block_hashes": {block_id: stable_hash(text)[:16] for block_id, text in shared_blocks.items()},
            "block_position_in_prompt": {block_id: idx for idx, block_id in enumerate(block_ids)},
            "all_gather_group_id": group_id,
            "round_shared_context_tokens": sum(block_tokens.values()),
            "duplicated_shared_context_tokens": max(0, (len(block_ids) - 1) * sum(block_tokens.values())),
            "pairwise_shared_block_similarity_proxy": 1.0 if len(block_ids) > 1 else 0.0,
            "estimated_kv_tokens": estimate_tokens(private_text) + sum(block_tokens.values()),
        }


class CriticalPathToolResumeContentionMesoMotif(Week3MesoMotif):
    spec = MotifSpec("tool_resume_contention_meso", ["independent", "centralized"], ["meso_workload", "tool_resume", "critical_path"], "Replay/live tool branches resume near reviewer/finalizer without controlled delay.")
    composed_from_motifs = ["evidence_collection", "tool_specialist_team", "multi_coder_branch", "coder_reviewer"]

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        plan = self.llm_agent(node_id="planner", node_name="Planner", agent_role="planner", system_prompt="Plan critical and non-critical tool-stalled branches.", user_prompt=self.config.query, parents=["START"], criticality="critical", extra_metadata=self.week3_meta(role="critical_path", criticality="critical", stage="planner", parents=["START"], dst_node="planner", contention_role="critical_agent"))

        def critical(_: int) -> tuple[str, str, str, str]:
            code = self.llm_agent(node_id="critical_coder", node_name="Critical Coder", agent_role="coder", system_prompt="Produce critical-path candidate.", user_prompt=f"Task:\n{self.config.query}\nPlan:\n{plan}", parents=["planner"], criticality="critical", extra_metadata=self.week3_meta(role="critical_path", criticality="critical", stage="coder", parents=["planner"], dst_node="critical_coder", expected_overlap_target=self.config.critical_stage_marker, contention_role="critical_agent"))
            review = self.llm_agent(node_id="critical_reviewer", node_name="Critical Reviewer", agent_role="reviewer", system_prompt="Review the critical-path candidate.", user_prompt=code, parents=["critical_coder"], criticality="critical", extra_metadata=self.week3_meta(role="critical_path", criticality="critical", stage="reviewer", parents=["critical_coder"], dst_node="critical_reviewer", critical_request_marker="reviewer", nearby_background_resume_expected=True, overlap_analysis_status="observed_or_unavailable", expected_overlap_target="reviewer", contention_role="critical_agent"))
            final = self.llm_agent(node_id="finalizer", node_name="Finalizer", agent_role="finalizer", system_prompt="Finalize from the critical review without waiting for non-critical tool branches.", user_prompt=f"Code:\n{code}\nReview:\n{review}", parents=["critical_reviewer"], criticality="critical", extra_metadata=self.week3_meta(role="merge_or_finalizer", criticality="critical", stage="finalizer", parents=["critical_reviewer"], dst_node="finalizer", critical_request_marker="finalizer", contention_role="critical_agent"))
            self.emit_edge(src_node="finalizer", dst_node="END", artifact_type="final", content=final, transfer_type="critical_path")
            return "finalizer", code, review, final

        def background_tool(i: int) -> tuple[str, str]:
            branch = f"tool_branch_{i}"
            pre_id = f"{branch}_tool_agent"
            pre = self.llm_agent(
                node_id=pre_id,
                node_name=f"Tool Agent {i} LLM-1",
                agent_role="tool_agent",
                system_prompt="Prepare a non-critical tool query before function call.",
                user_prompt=f"Task:\n{self.config.query}\nPlan:\n{plan}\nPrepare evidence query for branch {i}.",
                parents=["planner"],
                parallel_group="background_tool_prepare",
                criticality="non_critical",
                extra_metadata=self.week3_meta(
                    role="background_tool_branch",
                    criticality="non_critical",
                    parents=["planner"],
                    dst_node=pre_id,
                    background_branch_id=branch,
                    contention_role="non_critical_agent",
                    function_call_lifecycle_stage="llm_1_pre_tool",
                    tool_stalled=False,
                    resume_after_tool=False,
                    expected_overlap_target=self.config.critical_stage_marker,
                ),
            )
            tool_id = f"{branch}_search"
            meta = self.week3_meta(role="background_tool_branch", criticality="non_critical", parents=[pre_id], dst_node=tool_id, background_branch_id=branch, tool_stalled=True, contention_role="non_critical_agent", function_call_lifecycle_stage="tool_call_wait", resume_phase_policy=self.config.resume_phase_policy, expected_overlap_target=self.config.critical_stage_marker, tool_trace_source=self.config.tool_mode)
            evidence = self.search(node_id=tool_id, node_name=f"Tool Branch {i}", query=f"MAS workflow tracing simulator graph-aware serving evidence branch {i}", trace_fields=meta)
            if not self.config.background_resume_enabled:
                return tool_id, ""
            resume_id = f"resume_evidence_processor_{i}"
            out = self.llm_agent(node_id=resume_id, node_name=f"Resume Evidence Processor {i}", agent_role="evidence_processor", system_prompt="Resume after tool return and summarize evidence.", user_prompt=str(evidence)[:4000], parents=[tool_id], parallel_group="background_tool_resume", criticality="non_critical", extra_metadata=self.week3_meta(role="background_tool_branch", criticality="non_critical", parents=[tool_id], dst_node=resume_id, contention_role="non_critical_agent", function_call_lifecycle_stage="llm_2_resume", tool_stalled=True, background_resume_request=True, resume_after_tool=True, resume_group_id="week3_tool_resume", resume_phase_policy=self.config.resume_phase_policy, expected_overlap_target=self.config.critical_stage_marker, intended_to_overlap_with=self.config.critical_stage_marker))
            self.emit_edge(src_node=resume_id, dst_node="evidence_merge", artifact_type="resumed_evidence", content=out, transfer_type="aggregation", parallel_group="background_tool_resume")
            return resume_id, out

        def background_parallel(_: int) -> tuple[str, str]:
            out = self.llm_agent(node_id="background_coder", node_name="Background Coder", agent_role="coder", system_prompt="Produce an alternate background candidate.", user_prompt=f"Task:\n{self.config.query}\nPlan:\n{plan}", parents=["planner"], parallel_group="background_parallel", criticality="non_critical", extra_metadata=self.week3_meta(role="background_parallel_branch", criticality="non_critical", parents=["planner"], dst_node="background_coder", contention_role="non_critical_agent"))
            self.emit_edge(src_node="background_coder", dst_node="evidence_merge", artifact_type="background_candidate", content=out, transfer_type="aggregation")
            return "background_coder", out

        items = [("critical", 0), ("parallel", 0)] + [("tool", i) for i in range(1, max(1, min(8, self.config.tool_branch_width)) + 1)]
        def dispatch(item: tuple[str, int]) -> tuple[str, Any]:
            kind, idx = item
            if kind == "critical":
                return kind, critical(idx)
            if kind == "parallel":
                return kind, background_parallel(idx)
            return f"tool_{idx}", background_tool(idx)
        results = dict(self.run_parallel(items, dispatch))
        critical_node, code, review, final = results["critical"]
        bg = {str(v[0]): str(v[1]) for k, v in results.items() if k != "critical" and v and v[1]}
        self.barrier(barrier_id="late_evidence_merge_barrier", waiting_for_nodes=list(bg))
        merged = self.llm_agent(node_id="evidence_merge", node_name="Evidence Merge", agent_role="aggregator", system_prompt="Merge late evidence and background candidates.", user_prompt="\n".join(bg.values()), parents=list(bg), criticality="merge", extra_metadata=self.week3_meta(role="merge_or_finalizer", criticality="merge", parents=list(bg), dst_node="evidence_merge", fan_in_count=len(bg)))
        self.emit_edge(src_node="evidence_merge", dst_node="finalizer", artifact_type="late_evidence", content=merged, transfer_type="late_non_blocking_context")
        return self.workflow_end(final)


class HierarchicalSynthesisPressureMesoMotif(Week3MesoMotif):
    spec = MotifSpec("hierarchical_synthesis_pressure_meso", ["independent", "centralized"], ["meso_workload", "hierarchical_fanin"], "Grouped researchers/coders synthesize locally then globally.")
    composed_from_motifs = ["researcher_synthesizer", "multi_coder_branch", "generator_verifier", "debate_reviewer"]

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        groups = max(1, self.config.group_count)
        agents = max(1, self.config.agents_per_group)
        web_context = self.search(node_id="hier_web_context_search", node_name="Hierarchical Web Context Search", query=f"{self.config.query} multi-agent workflow tracing graph-aware simulator", trace_fields=self.week3_meta(role="background_tool_branch", criticality="background", parents=["START"], dst_node="hier_web_context_search", tool_stalled=True, tool_trace_source=self.config.tool_mode))
        plan = self.llm_agent(node_id="hier_planner", node_name="Hierarchical Planner", agent_role="planner", system_prompt="Split work into groups.", user_prompt=f"Task:\n{self.config.query}\nWeb context:\n{web_context}", parents=["hier_web_context_search"], criticality="critical", extra_metadata=self.week3_meta(role="critical_path", criticality="critical", stage="planner", parents=["hier_web_context_search"], hierarchy_depth=1, group_count=groups, agents_per_group=agents, retrieved_tool_context_tokens_est=estimate_tokens(str(web_context))))

        def group_run(g: int) -> tuple[str, str]:
            def agent_run(i: int) -> tuple[str, str]:
                node = f"group_{g}_agent_{i}"
                private = f"Group {g} task:\n{plan}"
                out = self.llm_agent(node_id=node, node_name=node, agent_role="researcher", system_prompt="Produce local group evidence.", user_prompt=private, parents=["hier_planner"], parallel_group=f"group_{g}_agents", criticality="background", extra_metadata=self.week3_meta(role="background_parallel_branch", criticality="background", parents=["hier_planner"], group_id=g, hierarchy_depth=2, **self.block_meta(private_text=private, shared_blocks={}, agent_id=node, round_id=0, group_id=f"group_{g}")))
                self.emit_edge(src_node=node, dst_node=f"group_{g}_synth", artifact_type="group_artifact", content=out, transfer_type="aggregation", parallel_group=f"group_{g}_agents")
                return node, out
            outs = dict(self.run_parallel(list(range(1, agents + 1)), agent_run))
            self.barrier(barrier_id=f"group_{g}_barrier", waiting_for_nodes=list(outs))
            shared_blocks = {node: text for node, text in outs.items()}
            synth_input = "\n".join(outs.values())
            synth = self.llm_agent(node_id=f"group_{g}_synth", node_name=f"Group {g} Synthesizer", agent_role="synthesizer", system_prompt="Synthesize group outputs.", user_prompt=synth_input, parents=list(outs), criticality="merge", extra_metadata=self.week3_meta(role="merge_or_finalizer", criticality="merge", parents=list(outs), fan_in_count=len(outs), hierarchy_depth=3, group_id=g, downstream_input_tokens=estimate_tokens(synth_input), **self.block_meta(private_text="", shared_blocks=shared_blocks, agent_id=f"group_{g}_synth", round_id=1, group_id=f"group_{g}")))
            self.emit_edge(src_node=f"group_{g}_synth", dst_node="cross_group_reviewer", artifact_type="group_summary", content=synth, transfer_type="aggregation")
            return f"group_{g}_synth", synth
        group_outputs = dict(self.run_parallel(list(range(1, groups + 1)), group_run))
        self.barrier(barrier_id="cross_group_barrier", waiting_for_nodes=list(group_outputs))
        group_blocks = {node: text for node, text in group_outputs.items()}
        reviewer_input = "\n".join(group_outputs.values())
        reviewer = self.llm_agent(node_id="cross_group_reviewer", node_name="Cross Group Reviewer", agent_role="reviewer", system_prompt="Review group summaries and select key evidence.", user_prompt=reviewer_input, parents=list(group_outputs), criticality="critical", extra_metadata=self.week3_meta(role="critical_path", criticality="critical", stage="reviewer", parents=list(group_outputs), fan_in_count=len(group_outputs), hierarchy_depth=4, **self.block_meta(private_text="", shared_blocks=group_blocks, agent_id="cross_group_reviewer", round_id=2, group_id="cross_group")))
        final = self.llm_agent(node_id="global_synthesizer", node_name="Global Synthesizer", agent_role="finalizer", system_prompt="Produce global synthesis.", user_prompt=reviewer, parents=["cross_group_reviewer"], criticality="critical", extra_metadata=self.week3_meta(role="merge_or_finalizer", criticality="critical", stage="finalizer", parents=["cross_group_reviewer"], hierarchy_depth=5))
        return self.workflow_end(final)


class DebateAllGatherPressureMesoMotif(Week3MesoMotif):
    spec = MotifSpec("debate_allgather_pressure_meso", ["decentralized", "centralized"], ["meso_workload", "all_gather"], "Local opinions, all-gather/debate rounds, consensus/finalizer.")
    composed_from_motifs = ["debate_reviewer", "all_gather_round"]

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        n = max(2, self.config.num_agents)
        web_context = self.search(node_id="debate_web_context_search", node_name="Debate Web Context Search", query=f"{self.config.query} debate all-gather multi-agent context redundancy", trace_fields=self.week3_meta(role="background_tool_branch", criticality="background", parents=["START"], dst_node="debate_web_context_search", tool_stalled=True, tool_trace_source=self.config.tool_mode))
        def produce(i: int) -> tuple[str, str]:
            node = f"opinion_{i}"
            private_prompt = f"Task:\n{self.config.query}\nWeb context:\n{web_context}"
            out = self.llm_agent(node_id=node, node_name=node, agent_role="peer_agent", system_prompt="Produce local opinion.", user_prompt=private_prompt, parents=["debate_web_context_search"], parallel_group="local_opinions", criticality="background", extra_metadata=self.week3_meta(role="background_parallel_branch", criticality="background", parents=["debate_web_context_search"], agent_count=n, retrieved_tool_context_tokens_est=estimate_tokens(str(web_context)), **self.block_meta(private_text=private_prompt, shared_blocks={}, agent_id=node, round_id=0, group_id="all_gather")))
            return node, out
        messages = dict(self.run_parallel(list(range(1, n + 1)), produce))
        for r in range(1, max(1, self.config.debate_rounds) + 1):
            def gather(i: int) -> tuple[str, str]:
                node = f"agent_{i}_round_{r}"
                private_text = messages.get(f"opinion_{i}") or messages.get(f"agent_{i}_round_{r-1}") or ""
                shared_blocks = {k: v for k, v in messages.items() if k != node}
                ordered_blocks = dict(list(shared_blocks.items())[i - 1:] + list(shared_blocks.items())[:i - 1])
                peer_text = "\n".join(f"{k}: {v}" for k, v in ordered_blocks.items())
                for src, msg in messages.items():
                    self.emit_edge(src_node=src, dst_node=node, artifact_type="peer_message", content=msg, transfer_type="broadcast", round_id=r, peer_round_id=r, parallel_group=f"all_gather_r{r}", fanout_count=n, recipient_count=n)
                meta = self.block_meta(private_text=private_text, shared_blocks=ordered_blocks, agent_id=node, round_id=r, group_id=f"all_gather_r{r}")
                out = self.llm_agent(node_id=node, node_name=node, agent_role="peer_agent", system_prompt="Update opinion after all gathered peers.", user_prompt=f"PRIVATE:\n{private_text}\nSHARED:\n{peer_text}", parents=list(messages), round_id=r, peer_round_id=r, parallel_group=f"all_gather_r{r}", criticality="background", extra_metadata=self.week3_meta(role="background_parallel_branch", criticality="background", parents=list(messages), peer_messages=peer_text, agent_count=n, debate_rounds=self.config.debate_rounds, broadcast_tokens_est=estimate_tokens(peer_text) * n, duplicated_context_tokens_est=max(0, meta["round_shared_context_tokens"] * (n - 1)), **meta))
                return node, out
            messages = dict(self.run_parallel(list(range(1, n + 1)), gather))
            self.barrier(barrier_id=f"all_gather_barrier_r{r}", waiting_for_nodes=list(messages), peer_round_id=r)
        consensus = self.llm_agent(node_id="consensus_reviewer", node_name="Consensus Reviewer", agent_role="reviewer", system_prompt="Review all debate outputs and form consensus.", user_prompt="\n".join(messages.values()), parents=list(messages), criticality="critical", extra_metadata=self.week3_meta(role="critical_path", criticality="critical", stage="reviewer", parents=list(messages), fan_in_count=len(messages)))
        final = self.finalizer(consensus, parents=["consensus_reviewer"])
        return self.workflow_end(final)


class RetryDebugPressureMesoMotif(Week3MesoMotif):
    spec = MotifSpec("retry_debug_pressure_meso", ["centralized"], ["meso_workload", "retry_debug"], "Generator/coder, verifier/tester, debugger/reviser loop, finalizer.")
    composed_from_motifs = ["coder_reviewer", "generator_verifier", "retry_debug_loop"]

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        web_context = self.search(node_id="retry_web_context_search", node_name="Retry Web Context Search", query=f"{self.config.query} review loop debugging prefix reuse", trace_fields=self.week3_meta(role="background_tool_branch", criticality="background", parents=["START"], dst_node="retry_web_context_search", tool_stalled=True, tool_trace_source=self.config.tool_mode))
        result = self.llm_agent(node_id="generator", node_name="Generator/Coder", agent_role="generator", system_prompt="Generate candidate solution.", user_prompt=f"Task:\n{self.config.query}\nWeb context:\n{web_context}", parents=["retry_web_context_search"], criticality="critical", extra_metadata=self.week3_meta(role="critical_path", criticality="critical", stage="coder", parents=["retry_web_context_search"], retrieved_tool_context_tokens_est=estimate_tokens(str(web_context))))
        max_depth = max(1, self.config.max_retries or 1)
        loop = 0
        verifier = ""
        for loop in range(max_depth):
            verifier = self.llm_agent(node_id=f"verifier_r{loop}", node_name=f"Verifier R{loop}", agent_role="verifier", system_prompt="Verify candidate; provide failures to debug.", user_prompt=result, parents=["generator" if loop == 0 else f"reviser_r{loop}"], retry_count=loop, criticality="critical", extra_metadata=self.week3_meta(role="critical_path", criticality="critical", stage="reviewer", retry_count=loop, debug_loop_count=loop))
            debug = self.llm_agent(node_id=f"debugger_r{loop}", node_name=f"Debugger R{loop}", agent_role="debugger", system_prompt="Diagnose verifier feedback and propose fix.", user_prompt=verifier, parents=[f"verifier_r{loop}"], retry_count=loop + 1, criticality="critical", extra_metadata=self.week3_meta(role="critical_path", criticality="critical", retry_count=loop + 1, debug_loop_count=loop + 1))
            result = self.llm_agent(node_id=f"reviser_r{loop+1}", node_name=f"Reviser R{loop+1}", agent_role="generator", system_prompt="Revise candidate from debug feedback.", user_prompt=f"Previous:\n{result}\nDebug:\n{debug}", parents=[f"debugger_r{loop}"], retry_count=loop + 1, criticality="critical", extra_metadata=self.week3_meta(role="critical_path", criticality="critical", stage="coder", retry_count=loop + 1, debug_loop_count=loop + 1))
        final = self.finalizer(f"Result:\n{result}\nVerifier:\n{verifier}", parents=[f"reviser_r{loop+1}"])
        return self.workflow_end(final)


class SharedMemoryFaninMesoMotif(Week3MesoMotif):
    spec = MotifSpec("shared_memory_fanin_meso", ["independent", "centralized"], ["meso_workload", "shared_memory"], "Writers/readers over shared evidence, reviewer/finalizer fan-in.")
    composed_from_motifs = ["shared_evidence_store", "researcher_synthesizer", "generator_verifier"]

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        writers = max(1, self.config.writer_count)
        readers = max(1, self.config.reader_count)
        def write(i: int) -> tuple[str, str]:
            node = f"writer_{i}"
            evidence = self.search(node_id=f"{node}_search", node_name=f"Writer {i} Search", query=f"{self.config.query} writer {i}", trace_fields=self.week3_meta(role="background_tool_branch", criticality="background", parents=["START"], dst_node=f"{node}_search"))
            private = str(evidence)
            out = self.llm_agent(node_id=node, node_name=node, agent_role="writer", system_prompt="Write evidence artifact to shared store.", user_prompt=private, parents=[f"{node}_search"], parallel_group="memory_writers", criticality="background", extra_metadata=self.week3_meta(role="background_parallel_branch", criticality="background", parents=[f"{node}_search"], memory_write=True, writer_count=writers, **self.block_meta(private_text=private, shared_blocks={}, agent_id=node, round_id=0, group_id="shared_memory_write")))
            self.emit_edge(src_node=node, dst_node="shared_evidence_store", artifact_type="memory_write", content=out, transfer_type="memory_write", parallel_group="memory_writers")
            self.trace.emit(event_type="memory_write", node_id=f"{node}_write", node_name=f"{node} write", node_type="memory", artifact_type="memory_write", motif_tags=self.motif_tags, shared_evidence_write_tokens_est=estimate_tokens(out), workflow_name=self.spec.name, workload_role="background_parallel_branch", criticality="background")
            return f"evidence_{i}", out
        store = dict(self.run_parallel(list(range(1, writers + 1)), write))
        self.barrier(barrier_id="memory_write_barrier", waiting_for_nodes=[f"writer_{i}" for i in range(1, writers + 1)])
        store_text = "\n".join(store.values())
        def read(i: int) -> tuple[str, str]:
            node = f"reader_{i}"
            shared_blocks = {block_id: text for block_id, text in store.items()}
            block_meta = self.block_meta(private_text=f"reader_{i}_private_state", shared_blocks=shared_blocks, agent_id=node, round_id=1, group_id="shared_memory_read")
            self.trace.emit(event_type="memory_read", node_id=f"{node}_read", node_name=f"{node} read", node_type="memory", artifact_type="memory_read", motif_tags=self.motif_tags, shared_evidence_read_tokens_est=estimate_tokens(store_text), workflow_name=self.spec.name, workload_role="background_parallel_branch", criticality="background", **block_meta)
            out = self.llm_agent(node_id=node, node_name=node, agent_role="reader", system_prompt="Read shared evidence and extract implications.", user_prompt=store_text, parents=["shared_evidence_store"], parallel_group="memory_readers", criticality="background", extra_metadata=self.week3_meta(role="background_parallel_branch", criticality="background", parents=["shared_evidence_store"], memory_read=True, reader_count=readers, fan_in_count=writers, shared_evidence_read_tokens_est=estimate_tokens(store_text), **block_meta))
            self.emit_edge(src_node=node, dst_node="memory_reviewer", artifact_type="memory_read", content=out, transfer_type="aggregation", parallel_group="memory_readers")
            return node, out
        read_outputs = dict(self.run_parallel(list(range(1, readers + 1)), read))
        reviewer = self.llm_agent(node_id="memory_reviewer", node_name="Memory Reviewer", agent_role="reviewer", system_prompt="Review shared-memory reader outputs.", user_prompt="\n".join(read_outputs.values()), parents=list(read_outputs), criticality="critical", extra_metadata=self.week3_meta(role="critical_path", criticality="critical", stage="reviewer", parents=list(read_outputs), fan_in_count=len(read_outputs)))
        final = self.finalizer(reviewer, parents=["memory_reviewer"])
        return self.workflow_end(final)


class RouterHandoffMotif(CompositeMotif):
    spec = MotifSpec("router_handoff", ["centralized", "single"], ["router", "handoff"], "Router -> selected specialist -> optional handoff")

    def run(self) -> dict[str, Any]:
        self.workflow_start()
        routes = ["planner_executor", "evidence_collection", "coder_reviewer", "tool_specialist"]
        router_out = self.llm_agent(node_id="router", node_name="Router", agent_role="router", system_prompt=f"Choose one route from {routes}. Return JSON if possible.", user_prompt=self.config.query, parents=["START"], manager_round_id=0, criticality="critical")
        parsed = parse_json_maybe(router_out)
        selected = str(parsed.get("selected_route") or parsed.get("route") or self._route_by_query(routes))
        if selected not in routes:
            selected = self._route_by_query(routes)
        not_selected = [r for r in routes if r != selected]
        self.emit_control(node_id="route_decision", node_name="Route Decision", event_type="route_decision", selected_route=selected, candidate_routes=routes, not_selected_routes=not_selected, selected_agent_count=1, candidate_agent_count=len(routes), handoff_count=0)
        self.emit_edge(src_node="router", dst_node=selected, artifact_type="route", content=router_out, transfer_type="handoff")
        specialist = self.llm_agent(node_id=selected, node_name=f"Selected Specialist: {selected}", agent_role="specialist", system_prompt=f"You are the selected route {selected}. Solve the task.", user_prompt=self.config.query, parents=["router"], manager_round_id=0, criticality="critical")
        handoff_count = 0
        if "handoff" in specialist.lower() and self.config.max_retries > 0:
            handoff_count = 1
            alternate = not_selected[0]
            self.emit_edge(src_node=selected, dst_node=alternate, artifact_type="handoff", content=specialist, transfer_type="handoff", retry_count=1)
            specialist = self.llm_agent(node_id=alternate, node_name=f"Handoff Specialist: {alternate}", agent_role="specialist", system_prompt=f"Handle handoff from {selected}.", user_prompt=specialist, parents=[selected], retry_count=1, manager_round_id=1, criticality="critical")
        self.emit_control(node_id="handoff_metrics", node_name="Handoff Metrics", event_type="handoff_metrics", selected_route=selected, candidate_routes=routes, not_selected_routes=not_selected, handoff_count=handoff_count)
        final = self.finalizer(specialist, parents=[selected])
        return self.workflow_end(final)

    def _route_by_query(self, routes: list[str]) -> str:
        text = self.config.query.lower()
        if "tool" in text or "search" in text or "evidence" in text:
            return "evidence_collection"
        if "code" in text or "patch" in text:
            return "coder_reviewer"
        if "plan" in text:
            return "planner_executor"
        return routes[0]

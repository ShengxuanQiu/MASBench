from __future__ import annotations

from .base import BaseTopology, TopologyConfig, parse_json_maybe


class CentralizedTopology(BaseTopology):
    motif_tags = ["centralized", "manager_worker", "dynamic_rounds", "manager_control", "fan_out", "fan_in"]

    def run(self) -> dict:
        self.workflow_start()
        history: list[str] = []
        final_summary = ""
        pool = self.agent_pool(prefix="central_agent")
        previous_selected: list[str] = []
        for round_id in range(self.config.max_rounds):
            manager_prompt = (
                f"Task:\n{self.config.query}\n\n"
                f"Available specialist pool:\n{self._pool_prompt(pool)}\n\n"
                f"Previous selected agents: {previous_selected}\n"
                "History:\n" + "\n".join(history[-6:])
            )
            manager_out = self.call_llm(
                node_id=f"manager_r{round_id}",
                node_name=f"Manager-Round-{round_id}",
                node_type="manager",
                agent_role="manager",
                prompt_template="central_manager.md",
                system_prompt="Return manager decision JSON.",
                user_prompt=manager_prompt,
                round_id=round_id,
                manager_round_id=round_id,
                parents=["START"] if round_id == 0 else [f"{agent_id}_r{round_id-1}" for agent_id in previous_selected],
                criticality="critical",
            )
            decision = self._manager_decision(round_id, manager_out, history, pool)
            proposed_selected_agents = decision["selected_workers"]
            selected_agents = [] if decision["decision"] == "finish" and round_id > 0 else proposed_selected_agents
            not_selected = [agent["id"] for agent in pool if agent["id"] not in selected_agents]
            self.trace.emit(
                event_type="manager_decision",
                node_id=f"manager_decision_r{round_id}",
                node_name=f"ManagerDecision-Round-{round_id}",
                node_type="manager",
                round_id=round_id,
                manager_round_id=round_id,
                motif_tags=self.motif_tags,
                decision=decision["decision"],
                available_agent_count=len(pool),
                selected_agent_count=len(selected_agents),
                selected_workers=selected_agents,
                proposed_selected_workers=proposed_selected_agents,
                selected_agent_roles=[self._agent_by_id(pool, agent_id)["role"] for agent_id in selected_agents],
                not_selected_agents=not_selected,
                dynamic_fanout_count=len(selected_agents),
                selection_reason=decision["selection_reason"],
                agent_pool=pool,
                stop_reason=decision["stop_reason"],
                confidence_est=decision["confidence_est"],
                decision_tokens_est=decision["decision_tokens_est"],
            )
            if decision["decision"] == "finish" and round_id > 0:
                final_summary = manager_out
                break
            for worker_id in selected_agents:
                self.emit_edge(src_node=f"manager_r{round_id}", dst_node=f"{worker_id}_r{round_id}", artifact_type="plan", content=self._instruction_for(decision, worker_id), transfer_type="broadcast", manager_round_id=round_id, fanout_count=len(selected_agents), recipient_count=len(selected_agents))

            def run_worker(worker_id: str) -> str:
                agent = self._agent_by_id(pool, worker_id)
                instruction = self._instruction_for(decision, worker_id)
                if self.use_react_agents():
                    user_prompt = f"Instruction: {instruction}\nTask: {self.config.query}\nSpecialty: {agent['capability']}\nDecide whether tool evidence is needed."
                else:
                    evidence = self.search(
                        node_id=f"{worker_id}_search_r{round_id}",
                        node_name=f"{worker_id}-Search-Round-{round_id}",
                        query=f"{self.config.query}\n{instruction}",
                    )
                    user_prompt = f"Instruction: {instruction}\nTask: {self.config.query}\nSpecialty: {agent['capability']}\nSearch evidence:\n{evidence}"
                out = self.call_agent(
                    node_id=f"{worker_id}_r{round_id}",
                    node_name=f"{agent['name']}-Round-{round_id}",
                    node_type="worker",
                    agent_role=agent["role"],
                    prompt_template="worker.md",
                    system_prompt=agent["system_prompt"],
                    user_prompt=user_prompt,
                    round_id=round_id,
                    manager_round_id=round_id,
                    parents=[f"manager_r{round_id}"],
                    parallel_group=f"centralized_workers_r{round_id}",
                    criticality="near_critical",
                    extra_metadata={"manager_instruction": instruction, "agent_capability": agent["capability"]},
                )
                self.emit_edge(src_node=f"{worker_id}_r{round_id}", dst_node=f"manager_r{round_id+1}", artifact_type="evidence", content=out, transfer_type="aggregation", manager_round_id=round_id)
                return out

            worker_outputs = self.run_parallel(list(selected_agents), run_worker)
            self.barrier(barrier_id=f"manager_collect_r{round_id}", waiting_for_nodes=[f"{w}_r{round_id}" for w in selected_agents], manager_round_id=round_id)
            history.extend(worker_outputs)
            final_summary = "\n".join(history)
            previous_selected = selected_agents
        self.emit_edge(src_node="manager", dst_node="finalizer", artifact_type="summary", content=final_summary, transfer_type="aggregation")
        final = self.call_llm(
            node_id="finalizer",
            node_name="Finalizer",
            node_type="agent",
            agent_role="finalizer",
            prompt_template="finalizer.md",
            system_prompt="Finalize manager-worker result.",
            user_prompt=final_summary,
            parents=["manager"],
            criticality="critical",
        )
        self.emit_edge(src_node="finalizer", dst_node="END", artifact_type="final", content=final, transfer_type="aggregation")
        return self.workflow_end(final)

    def _manager_decision(self, round_id: int, manager_out: str, history: list[str], pool: list[dict[str, str]]) -> dict:
        selected = self._rule_based_selection(round_id, history, pool)
        per_agent_instructions = {agent_id: f"round {round_id}: use your specialty to gather concise evidence and confidence" for agent_id in selected}
        if self.config.manager_policy == "llm":
            parsed = parse_json_maybe(manager_out)
            selected = self.clamp_selected_agents(parsed.get("selected_workers") or parsed.get("selected_agents") or selected, pool)
            decision = parsed.get("decision") or "continue"
            reason = parsed.get("reason") or "llm decision"
            instruction = parsed.get("next_instruction") or "collect evidence"
            per_agent_instructions = parsed.get("instructions") or per_agent_instructions
            confidence = float(parsed.get("confidence") or 0.7)
            selection_reason = reason
        else:
            must_continue = round_id < self.config.force_centralized_rounds
            at_limit = round_id >= self.config.max_rounds - 1
            confidence = min(0.9, 0.5 + 0.12 * len(history))
            enough = bool(history) and confidence >= self.config.orchestrator_stop_confidence
            decision = "continue" if (must_continue or (not enough and not at_limit)) else "finish"
            reason = "max_rounds" if at_limit and not must_continue else ("force_continue_rounds" if must_continue else ("sufficient_evidence" if enough else "needs_more_evidence"))
            instruction = f"round {round_id}: collect evidence using selected specialist roles"
            selection_reason = f"rule_based_dynamic_selection_round_{round_id}"
        selected = self.clamp_selected_agents(selected, pool)
        return {
            "decision": decision,
            "selected_workers": selected,
            "stop_reason": reason,
            "next_instruction": instruction,
            "instructions": per_agent_instructions,
            "selection_reason": selection_reason,
            "confidence_est": confidence,
            "decision_tokens_est": len(manager_out) // 4,
        }

    def _rule_based_selection(self, round_id: int, history: list[str], pool: list[dict[str, str]]) -> list[str]:
        by_role = {agent["role"]: agent["id"] for agent in pool}
        stages = [
            ["issue_triage", "repo_search"],
            ["code_localization", "api_doc", "test_reasoning"],
            ["patch_planning", "regression_risk", "performance"],
            ["judge", "maintainer"],
        ]
        roles = stages[min(round_id, len(stages) - 1)]
        if history and round_id % 2 == 1:
            roles = roles + ["judge"]
        return [by_role[role] for role in roles if role in by_role]

    def _pool_prompt(self, pool: list[dict[str, str]]) -> str:
        return "\n".join(f"- {agent['id']}: {agent['name']} ({agent['capability']})" for agent in pool)

    def _agent_by_id(self, pool: list[dict[str, str]], agent_id: str) -> dict[str, str]:
        return next(agent for agent in pool if agent["id"] == agent_id)

    def _instruction_for(self, decision: dict, agent_id: str) -> str:
        instructions = decision.get("instructions") or {}
        return str(instructions.get(agent_id) or decision.get("next_instruction") or "collect evidence")

    def topology_summary(self, events: list[dict]) -> dict:
        decisions = [e for e in events if e.get("event_type") == "manager_decision"]
        fanout = sum(len(e.get("selected_workers") or []) for e in decisions)
        selected_counts = [int(e.get("selected_agent_count") or 0) for e in decisions]
        return {
            "manager_decision_count": len(decisions),
            "worker_fanout_count": fanout,
            "available_agent_count": max([int(e.get("available_agent_count") or 0) for e in decisions] or [0]),
            "dynamic_fanout_by_round": selected_counts,
            "avg_dynamic_fanout": round(sum(selected_counts) / max(1, len(selected_counts)), 4),
        }


def build_workflow(config: TopologyConfig, **deps) -> CentralizedTopology:
    return CentralizedTopology(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> CentralizedTopology:
    return CentralizedTopology(config, **deps)

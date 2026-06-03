from __future__ import annotations

from .base import BaseTopology, TopologyConfig, parse_json_maybe


class HybridTopology(BaseTopology):
    motif_tags = ["hybrid", "manager_worker", "peer_communication", "nested_rounds", "all_gather", "fan_in"]

    def run(self) -> dict:
        self.workflow_start()
        history: list[str] = []
        final_summary = ""
        pool = self.agent_pool(prefix="hybrid_agent")
        previous_selected: list[str] = []
        for manager_round in range(self.config.max_rounds):
            manager_prompt = (
                f"Task:\n{self.config.query}\n\n"
                f"Available specialist pool:\n{self._pool_prompt(pool)}\n\n"
                f"Previous selected agents: {previous_selected}\n"
                f"History:\n" + "\n".join(history[-8:])
            )
            manager_out = self.call_llm(
                node_id=f"hybrid_manager_r{manager_round}",
                node_name=f"HybridManager-Round-{manager_round}",
                node_type="manager",
                agent_role="manager",
                prompt_template="hybrid_manager.md",
                system_prompt="Assign peer tasks, select a subset from the specialist pool, and decide whether to continue.",
                user_prompt=manager_prompt,
                round_id=manager_round,
                manager_round_id=manager_round,
                parents=["START"] if manager_round == 0 else [f"{agent_id}_mr{manager_round-1}" for agent_id in previous_selected],
                criticality="critical",
            )
            decision = self._manager_decision(manager_round, manager_out, history, pool)
            selected_agents = decision["selected_workers"]
            not_selected = [agent["id"] for agent in pool if agent["id"] not in selected_agents]
            instruction = decision["next_instruction"]
            self.trace.emit(
                event_type="manager_instruction",
                node_id=f"hybrid_instruction_r{manager_round}",
                node_name=f"HybridInstruction-Round-{manager_round}",
                node_type="manager",
                round_id=manager_round,
                manager_round_id=manager_round,
                motif_tags=self.motif_tags,
                manager_instruction=instruction,
                available_agent_count=len(pool),
                selected_agent_count=len(selected_agents),
                selected_workers=selected_agents,
                selected_agent_roles=[self._agent_by_id(pool, agent_id)["role"] for agent_id in selected_agents],
                not_selected_agents=not_selected,
                dynamic_fanout_count=len(selected_agents),
                selection_reason=decision["selection_reason"],
                agent_pool=pool,
            )
            messages = {}
            for agent_id in selected_agents:
                node = f"{agent_id}_mr{manager_round}"
                self.emit_edge(src_node=f"hybrid_manager_r{manager_round}", dst_node=node, artifact_type="plan", content=self._instruction_for(decision, agent_id), transfer_type="broadcast", manager_round_id=manager_round, fanout_count=len(selected_agents), recipient_count=len(selected_agents))

            def run_worker(agent_id: str) -> tuple[str, str]:
                agent = self._agent_by_id(pool, agent_id)
                node = f"{agent_id}_mr{manager_round}"
                agent_instruction = self._instruction_for(decision, agent_id)
                if self.use_react_agents():
                    user_prompt = f"Manager instruction: {agent_instruction}\nTask: {self.config.query}\nSpecialty: {agent['capability']}\nDecide whether tool evidence is needed."
                else:
                    evidence = self.search(
                        node_id=f"{node}_search",
                        node_name=f"{agent['name']}-Search-MR{manager_round}",
                        query=f"MASBench graph-aware simulator tracing {agent['capability']} round {manager_round}",
                    )
                    user_prompt = f"Manager instruction: {agent_instruction}\nTask: {self.config.query}\nSpecialty: {agent['capability']}\nSearch evidence:\n{evidence}"
                message = self.call_agent(
                    node_id=node,
                    node_name=f"{agent['name']}-MR{manager_round}",
                    node_type="worker",
                    agent_role=agent["role"],
                    prompt_template="hybrid_worker.md",
                    system_prompt=agent["system_prompt"],
                    user_prompt=user_prompt,
                    round_id=manager_round,
                    manager_round_id=manager_round,
                    peer_round_id=0,
                    parents=[f"hybrid_manager_r{manager_round}"],
                    parallel_group=f"hybrid_workers_mr{manager_round}",
                    criticality="near_critical",
                    extra_metadata={"manager_instruction": agent_instruction, "agent_capability": agent["capability"]},
                )
                return node, message

            messages = dict(self.run_parallel(list(selected_agents), run_worker))
            peer_round_cap = max(0, min(self.config.peer_rounds, int(decision.get("peer_rounds", self.config.peer_rounds))))
            for peer_round in range(1, peer_round_cap + 1):
                nodes = list(messages)
                def peer_step(item: tuple[int, str]) -> tuple[str, str]:
                    idx, node = item
                    peers = self._peer_nodes(nodes, idx)
                    peer_context = []
                    for peer in peers:
                        self.emit_edge(src_node=peer, dst_node=node, artifact_type="peer_message", content=messages[peer], transfer_type="message_passing", manager_round_id=manager_round, peer_round_id=peer_round, parallel_group=f"hybrid_peer_mr{manager_round}_pr{peer_round}")
                        peer_context.append(messages[peer])
                    message = self.call_agent(
                        node_id=f"{node}_pr{peer_round}",
                        node_name=f"{node}-PeerRound-{peer_round}",
                        node_type="worker",
                        agent_role="debate_agent",
                        prompt_template="debate_agent.md",
                        system_prompt="Revise with peer evidence.",
                        user_prompt=f"Task: {self.config.query}\nPeers:\n" + "\n".join(peer_context),
                        round_id=manager_round,
                        manager_round_id=manager_round,
                        peer_round_id=peer_round,
                        parents=peers,
                        parallel_group=f"hybrid_peer_mr{manager_round}_pr{peer_round}",
                        criticality="near_critical",
                        extra_metadata={"peer_messages": peer_context, "manager_instruction": instruction},
                    )
                    return node, message

                next_messages = dict(self.run_parallel(list(enumerate(nodes)), peer_step))
                self.barrier(
                    barrier_id=f"hybrid_peer_barrier_mr{manager_round}_pr{peer_round}",
                    waiting_for_nodes=[f"{node}_pr{peer_round}" for node in next_messages],
                    manager_round_id=manager_round,
                    peer_round_id=peer_round,
                )
                messages = next_messages
            for node, content in messages.items():
                self.emit_edge(src_node=node, dst_node=f"hybrid_manager_collect_r{manager_round}", artifact_type="evidence", content=content, transfer_type="aggregation", manager_round_id=manager_round, peer_round_id=peer_round_cap)
            self.barrier(barrier_id=f"hybrid_manager_collect_r{manager_round}", waiting_for_nodes=list(messages), manager_round_id=manager_round, peer_round_id=peer_round_cap)
            history.extend(messages.values())
            self.trace.emit(
                event_type="manager_decision",
                node_id=f"hybrid_decision_r{manager_round}",
                node_name=f"HybridDecision-Round-{manager_round}",
                node_type="manager",
                round_id=manager_round,
                manager_round_id=manager_round,
                motif_tags=self.motif_tags,
                decision=decision["decision"],
                available_agent_count=len(pool),
                selected_agent_count=len(selected_agents),
                selected_workers=selected_agents,
                selected_agent_roles=[self._agent_by_id(pool, agent_id)["role"] for agent_id in selected_agents],
                not_selected_agents=not_selected,
                dynamic_fanout_count=len(selected_agents),
                selection_reason=decision["selection_reason"],
                stop_reason=decision["stop_reason"],
                confidence_est=decision["confidence_est"],
                decision_tokens_est=decision["decision_tokens_est"],
            )
            final_summary = "\n".join(history)
            previous_selected = selected_agents
            if decision["decision"] == "finish":
                break
        final = self.call_llm(
            node_id="finalizer",
            node_name="Finalizer",
            node_type="agent",
            agent_role="finalizer",
            prompt_template="finalizer.md",
            system_prompt="Finalize hybrid manager + peer result.",
            user_prompt=final_summary,
            parents=["hybrid_manager_collect"],
            criticality="critical",
        )
        self.emit_edge(src_node="finalizer", dst_node="END", artifact_type="final", content=final, transfer_type="aggregation")
        return self.workflow_end(final)

    def _peer_nodes(self, nodes: list[str], idx: int) -> list[str]:
        if self.config.communication_topology in {"all_to_all", "peer_all_to_all"}:
            return [n for n in nodes if n != nodes[idx]]
        if self.config.communication_topology == "pairwise":
            return [nodes[idx ^ 1]] if idx ^ 1 < len(nodes) else [nodes[idx - 1]]
        return [nodes[(idx - 1) % len(nodes)]]

    def _manager_decision(self, round_id: int, manager_out: str, history: list[str], pool: list[dict[str, str]]) -> dict:
        selected = self._rule_based_selection(round_id, history, pool)
        per_agent_instructions = {agent_id: f"manager round {round_id}: collect evidence and then discuss with selected peers" for agent_id in selected}
        peer_rounds = self.config.peer_rounds
        if self.config.manager_policy == "llm":
            parsed = parse_json_maybe(manager_out)
            selected = self.clamp_selected_agents(parsed.get("selected_workers") or parsed.get("selected_agents") or selected, pool)
            decision = parsed.get("decision") or "continue"
            instruction = parsed.get("next_instruction") or "peer agents exchange evidence"
            per_agent_instructions = parsed.get("instructions") or per_agent_instructions
            confidence = float(parsed.get("confidence") or 0.7)
            reason = parsed.get("reason") or "llm decision"
            peer_rounds = int(parsed.get("peer_rounds") or peer_rounds)
            selection_reason = reason
        else:
            at_limit = round_id >= self.config.max_rounds - 1
            confidence = min(0.9, 0.48 + 0.1 * len(history))
            enough = bool(history) and confidence >= self.config.orchestrator_stop_confidence
            decision = "finish" if (enough or at_limit) else "continue"
            reason = "max_rounds" if at_limit and not enough else ("sufficient_evidence" if enough else "needs_peer_evidence")
            instruction = f"manager round {round_id}: selected agents gather evidence, then exchange only relevant peer messages"
            selection_reason = f"rule_based_dynamic_hybrid_selection_round_{round_id}"
        selected = self.clamp_selected_agents(selected, pool)
        return {
            "decision": decision,
            "selected_workers": selected,
            "next_instruction": instruction,
            "instructions": per_agent_instructions,
            "peer_rounds": max(0, peer_rounds),
            "stop_reason": reason,
            "confidence_est": confidence,
            "selection_reason": selection_reason,
            "decision_tokens_est": len(manager_out) // 4,
        }

    def _rule_based_selection(self, round_id: int, history: list[str], pool: list[dict[str, str]]) -> list[str]:
        by_role = {agent["role"]: agent["id"] for agent in pool}
        stages = [
            ["issue_triage", "repo_search", "code_localization"],
            ["test_reasoning", "patch_planning"],
            ["regression_risk", "performance", "judge"],
            ["maintainer", "judge"],
        ]
        roles = stages[min(round_id, len(stages) - 1)]
        if not history and "tool_heavy" in by_role:
            roles.append("tool_heavy")
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
        selected_counts = [int(e.get("selected_agent_count") or 0) for e in decisions]
        return {
            "nested_round_count": len({(e.get("manager_round_id"), e.get("peer_round_id")) for e in events if e.get("peer_round_id") is not None}),
            "manager_collection_barrier": len([e for e in events if "manager_collect" in str(e.get("node_id")) and e.get("node_type") == "barrier"]),
            "available_agent_count": max([int(e.get("available_agent_count") or 0) for e in decisions] or [0]),
            "dynamic_fanout_by_round": selected_counts,
            "avg_dynamic_fanout": round(sum(selected_counts) / max(1, len(selected_counts)), 4),
        }


def build_workflow(config: TopologyConfig, **deps) -> HybridTopology:
    return HybridTopology(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> HybridTopology:
    return HybridTopology(config, **deps)

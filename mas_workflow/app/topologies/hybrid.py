from __future__ import annotations

from .base import BaseTopology, TopologyConfig, parse_json_maybe


class HybridTopology(BaseTopology):
    motif_tags = ["hybrid", "manager_worker", "peer_communication", "nested_rounds", "all_gather", "fan_in"]

    def run(self) -> dict:
        self.workflow_start()
        history: list[str] = []
        final_summary = ""
        for manager_round in range(self.config.max_rounds):
            manager_out = self.call_llm(
                node_id=f"hybrid_manager_r{manager_round}",
                node_name=f"HybridManager-Round-{manager_round}",
                node_type="manager",
                agent_role="manager",
                prompt_template="hybrid_manager.md",
                system_prompt="Assign peer tasks and decide whether to continue.",
                user_prompt=f"Task:\n{self.config.query}\nHistory:\n" + "\n".join(history),
                round_id=manager_round,
                manager_round_id=manager_round,
                parents=["START"] if manager_round == 0 else [f"hybrid_agent_{i}_mr{manager_round-1}" for i in range(1, self.config.num_agents + 1)],
                criticality="critical",
            )
            instruction = parse_json_maybe(manager_out).get("next_instruction", "peer agents exchange evidence")
            self.trace.emit(
                event_type="manager_instruction",
                node_id=f"hybrid_instruction_r{manager_round}",
                node_name=f"HybridInstruction-Round-{manager_round}",
                node_type="manager",
                round_id=manager_round,
                manager_round_id=manager_round,
                motif_tags=self.motif_tags,
                manager_instruction=instruction,
            )
            messages = {}
            for i in range(1, self.config.num_agents + 1):
                node = f"hybrid_agent_{i}_mr{manager_round}"
                self.emit_edge(src_node=f"hybrid_manager_r{manager_round}", dst_node=node, artifact_type="plan", content=instruction, transfer_type="broadcast", manager_round_id=manager_round, fanout_count=self.config.num_agents, recipient_count=self.config.num_agents)

            def run_worker(i: int) -> tuple[str, str]:
                node = f"hybrid_agent_{i}_mr{manager_round}"
                if self.use_react_agents():
                    user_prompt = f"Manager instruction: {instruction}\nTask: {self.config.query}\nDecide whether tool evidence is needed."
                else:
                    evidence = self.search(
                        node_id=f"{node}_search",
                        node_name=f"HybridAgent-{i}-Search-MR{manager_round}",
                        query=f"{self.config.query}\n{instruction}",
                    )
                    user_prompt = f"Manager instruction: {instruction}\nTask: {self.config.query}\nSearch evidence:\n{evidence}"
                message = self.call_agent(
                    node_id=node,
                    node_name=f"HybridAgent-{i}-MR{manager_round}",
                    node_type="worker",
                    agent_role="worker",
                    prompt_template="hybrid_worker.md",
                    system_prompt=f"You are hybrid worker {i}.",
                    user_prompt=user_prompt,
                    round_id=manager_round,
                    manager_round_id=manager_round,
                    peer_round_id=0,
                    parents=[f"hybrid_manager_r{manager_round}"],
                    parallel_group=f"hybrid_workers_mr{manager_round}",
                    criticality="near_critical",
                    extra_metadata={"manager_instruction": instruction},
                )
                return node, message

            messages = dict(self.run_parallel(list(range(1, self.config.num_agents + 1)), run_worker))
            for peer_round in range(1, self.config.peer_rounds + 1):
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
                self.emit_edge(src_node=node, dst_node=f"hybrid_manager_collect_r{manager_round}", artifact_type="evidence", content=content, transfer_type="aggregation", manager_round_id=manager_round, peer_round_id=self.config.peer_rounds)
            self.barrier(barrier_id=f"hybrid_manager_collect_r{manager_round}", waiting_for_nodes=list(messages), manager_round_id=manager_round, peer_round_id=self.config.peer_rounds)
            history.extend(messages.values())
            decision = "continue" if manager_round < self.config.max_rounds - 1 else "finish"
            self.trace.emit(
                event_type="manager_decision",
                node_id=f"hybrid_decision_r{manager_round}",
                node_name=f"HybridDecision-Round-{manager_round}",
                node_type="manager",
                round_id=manager_round,
                manager_round_id=manager_round,
                motif_tags=self.motif_tags,
                decision=decision,
                selected_workers=[f"hybrid_agent_{i}" for i in range(1, self.config.num_agents + 1)],
                stop_reason="max_rounds" if decision == "finish" else "continue",
                confidence_est=0.74,
                decision_tokens_est=16,
            )
            final_summary = "\n".join(history)
            if decision == "finish":
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

    def topology_summary(self, events: list[dict]) -> dict:
        return {
            "nested_round_count": len({(e.get("manager_round_id"), e.get("peer_round_id")) for e in events if e.get("peer_round_id") is not None}),
            "manager_collection_barrier": len([e for e in events if "manager_collect" in str(e.get("node_id")) and e.get("node_type") == "barrier"]),
        }


def build_workflow(config: TopologyConfig, **deps) -> HybridTopology:
    return HybridTopology(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> HybridTopology:
    return HybridTopology(config, **deps)

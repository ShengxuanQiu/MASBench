from __future__ import annotations

import random

from .base import BaseTopology, TopologyConfig


class DecentralizedDebateTopology(BaseTopology):
    motif_tags = ["decentralized", "debate", "peer_to_peer", "all_gather", "multi_round"]

    def run(self) -> dict:
        self.workflow_start()
        rng = random.Random(self.config.random_seed)
        messages: dict[str, str] = {}
        for i in range(1, self.config.num_agents + 1):
            node = f"debate_agent_{i}"
            self.emit_edge(src_node="START", dst_node=node, artifact_type="task", content=self.config.query, transfer_type="broadcast", round_id=0)
            evidence = self.search(node_id=f"{node}_search", node_name=f"DebateAgent-{i} Search", query=self.config.query)
            messages[node] = self.call_llm(
                node_id=node,
                node_name=f"DebateAgent-{i}",
                node_type="agent",
                agent_role="debate_agent",
                prompt_template="debate_agent.md",
                system_prompt=f"You are debate peer {i}.",
                user_prompt=f"Initial answer for task:\n{self.config.query}\nSearch evidence:\n{evidence}",
                round_id=0,
                peer_round_id=0,
                parents=["START"],
                parallel_group="debate_initial",
                criticality="near_critical",
            )
        for peer_round in range(1, self.config.debate_rounds + 1):
            next_messages: dict[str, str] = {}
            for i in range(1, self.config.num_agents + 1):
                node = f"debate_agent_{i}"
                peers = self._peers(node, list(messages), rng)
                peer_texts = []
                for peer in peers:
                    content = messages[peer]
                    self.emit_edge(src_node=peer, dst_node=node, artifact_type="peer_message", content=content, transfer_type="message_passing" if self.config.communication_topology != "all_to_all" else "broadcast", round_id=peer_round, peer_round_id=peer_round, parallel_group=f"debate_round_{peer_round}", fanout_count=len(peers), recipient_count=1)
                    peer_texts.append(f"{peer}: {content}")
                peer_context = "\n".join(peer_texts)
                next_messages[node] = self.call_llm(
                    node_id=f"{node}_r{peer_round}",
                    node_name=f"DebateAgent-{i}-Round-{peer_round}",
                    node_type="agent",
                    agent_role="debate_agent",
                    prompt_template="debate_agent.md",
                    system_prompt=f"You are debate peer {i}; revise from peer messages.",
                    user_prompt=f"Task:\n{self.config.query}\nPeer messages:\n{peer_context}",
                    round_id=peer_round,
                    peer_round_id=peer_round,
                    parents=peers,
                    parallel_group=f"debate_round_{peer_round}",
                    criticality="near_critical",
                    extra_metadata={"peer_messages": peer_context},
                )
            self.barrier(barrier_id=f"debate_barrier_r{peer_round}", waiting_for_nodes=list(next_messages), round_id=peer_round, peer_round_id=peer_round)
            messages = next_messages
        consensus_input = "\n".join(messages.values())
        for node, content in messages.items():
            self.emit_edge(src_node=node, dst_node="consensus", artifact_type="vote", content=content, transfer_type="aggregation", peer_round_id=self.config.debate_rounds)
        consensus = self.call_llm(
            node_id="consensus",
            node_name="Consensus",
            node_type="aggregator",
            agent_role="aggregator",
            prompt_template="consensus.md",
            system_prompt=f"Apply consensus policy {self.config.extra.get('consensus_policy', 'final_summarizer')}.",
            user_prompt=consensus_input,
            parents=list(messages),
            criticality="critical",
        )
        final = self.call_llm(
            node_id="finalizer",
            node_name="Finalizer",
            node_type="agent",
            agent_role="finalizer",
            prompt_template="finalizer.md",
            system_prompt="Finalize debate consensus.",
            user_prompt=consensus,
            parents=["consensus"],
            criticality="critical",
        )
        self.emit_edge(src_node="finalizer", dst_node="END", artifact_type="final", content=final, transfer_type="aggregation")
        return self.workflow_end(final)

    def _peers(self, node: str, nodes: list[str], rng: random.Random) -> list[str]:
        others = [n for n in nodes if n != node]
        if self.config.communication_topology == "all_to_all":
            return others
        idx = nodes.index(node)
        if self.config.communication_topology == "ring":
            return [nodes[(idx - 1) % len(nodes)]]
        k = int(self.config.extra.get("random_k", 1))
        return rng.sample(others, k=min(k, len(others)))

    def topology_summary(self, events: list[dict]) -> dict:
        peer_edges = [e for e in events if e.get("artifact_type") == "peer_message"]
        return {
            "peer_message_count": len(peer_edges),
            "all_to_all_tokens": sum(int(e.get("artifact_tokens_est") or 0) for e in peer_edges),
        }


def build_workflow(config: TopologyConfig, **deps) -> DecentralizedDebateTopology:
    return DecentralizedDebateTopology(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> DecentralizedDebateTopology:
    return DecentralizedDebateTopology(config, **deps)

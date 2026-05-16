from __future__ import annotations

from .base import BaseTopology, TopologyConfig


class IndependentTopology(BaseTopology):
    motif_tags = ["independent", "parallel_agents", "final_aggregation", "fan_out", "fan_in"]

    def run(self) -> dict:
        self.workflow_start()
        outputs: list[str] = []
        artifact_base = None
        for i in range(1, self.config.num_agents + 1):
            node = f"agent_{i}"
            artifact_base = self.emit_edge(
                src_node="START",
                dst_node=node,
                artifact_type="task",
                content=self.config.query,
                transfer_type="broadcast",
                parallel_group="independent_workers",
                fanout_count=self.config.num_agents,
                recipient_count=self.config.num_agents,
                duplicated_from_artifact_id=artifact_base,
            )
            perspective = self.config.extra.get("per_agent_perspective", {}).get(node, f"perspective_{i}")
            evidence = self.search(node_id=f"{node}_search", node_name=f"Agent-{i} Search", query=f"{self.config.query}\n{perspective}")
            out = self.call_llm(
                node_id=node,
                node_name=f"Agent-{i}",
                node_type="agent",
                agent_role="independent_worker",
                prompt_template="independent_worker.md",
                system_prompt=f"You are independent worker {i}.",
                user_prompt=f"Perspective: {perspective}\nTask:\n{self.config.query}\nSearch evidence:\n{evidence}",
                round_id=0,
                parents=["START"],
                parallel_group="independent_workers",
                criticality="near_critical",
            )
            outputs.append(out)
            self.emit_edge(src_node=node, dst_node="aggregator", artifact_type="answer", content=out, transfer_type="aggregation", parallel_group="independent_workers")
        self.barrier(barrier_id="aggregator_barrier", waiting_for_nodes=[f"agent_{i}" for i in range(1, self.config.num_agents + 1)], round_id=0)
        aggregate_input = "\n".join(outputs)
        aggregate = self.call_llm(
            node_id="aggregator",
            node_name="Aggregator",
            node_type="aggregator",
            agent_role="aggregator",
            prompt_template="aggregator.md",
            system_prompt=f"Aggregate using policy {self.config.aggregation_policy}.",
            user_prompt=aggregate_input,
            round_id=0,
            parents=[f"agent_{i}" for i in range(1, self.config.num_agents + 1)],
            criticality="critical",
        )
        self.emit_edge(src_node="aggregator", dst_node="finalizer", artifact_type="summary", content=aggregate, transfer_type="aggregation")
        final = self.call_llm(
            node_id="finalizer",
            node_name="Finalizer",
            node_type="agent",
            agent_role="finalizer",
            prompt_template="finalizer.md",
            system_prompt="Produce final answer.",
            user_prompt=aggregate,
            round_id=0,
            parents=["aggregator"],
            criticality="critical",
        )
        self.emit_edge(src_node="finalizer", dst_node="END", artifact_type="final", content=final, transfer_type="aggregation")
        return self.workflow_end(final)

    def topology_summary(self, events: list[dict]) -> dict:
        worker_inputs = [int(e.get("input_tokens_est") or 0) for e in events if str(e.get("node_id", "")).startswith("agent_") and e.get("event_type") == "llm_request_end"]
        aggregation_input = sum(int(e.get("input_tokens_est") or 0) for e in events if e.get("node_id") == "aggregator" and e.get("event_type") == "llm_request_end")
        return {
            "independent_aggregation_input_tokens": aggregation_input,
            "redundant_worker_input_tokens": max(0, sum(worker_inputs) - (max(worker_inputs) if worker_inputs else 0)),
        }


def build_workflow(config: TopologyConfig, **deps) -> IndependentTopology:
    return IndependentTopology(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> IndependentTopology:
    return IndependentTopology(config, **deps)

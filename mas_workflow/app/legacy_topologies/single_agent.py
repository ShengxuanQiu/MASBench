from __future__ import annotations

from .base import BaseTopology, TopologyConfig


class SingleAgentTopology(BaseTopology):
    motif_tags = ["single_agent", "baseline"]

    def run(self) -> dict:
        self.workflow_start()
        self.emit_edge(src_node="START", dst_node="single_agent", artifact_type="task", content=self.config.query, transfer_type="prompt_inclusion")
        if self.use_react_agents():
            user_prompt = f"Task:\n{self.config.query}\nDecide whether search evidence is needed before answering."
            parents = ["START"]
        else:
            evidence = self.search(node_id="single_search", node_name="Single Search", query=self.config.query)
            user_prompt = f"Task:\n{self.config.query}\nEvidence:\n{evidence}"
            parents = ["START", "single_search"]
        answer = self.call_agent(
            node_id="single_agent",
            node_name="SingleAgent",
            node_type="agent",
            agent_role="worker",
            prompt_template="single_agent.md",
            system_prompt="You are a single baseline MASBench-Arch agent.",
            user_prompt=user_prompt,
            round_id=0,
            parents=parents,
            criticality="critical",
        )
        self.emit_edge(src_node="single_agent", dst_node="finalizer", artifact_type="answer", content=answer, transfer_type="message_passing")
        final = self.call_llm(
            node_id="finalizer",
            node_name="Finalizer",
            node_type="agent",
            agent_role="finalizer",
            prompt_template="finalizer.md",
            system_prompt="Produce the final answer.",
            user_prompt=answer,
            round_id=0,
            parents=["single_agent"],
            criticality="critical",
        )
        self.emit_edge(src_node="finalizer", dst_node="END", artifact_type="final", content=final, transfer_type="aggregation")
        return self.workflow_end(final)


def build_workflow(config: TopologyConfig, **deps) -> SingleAgentTopology:
    return SingleAgentTopology(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> SingleAgentTopology:
    return SingleAgentTopology(config, **deps)

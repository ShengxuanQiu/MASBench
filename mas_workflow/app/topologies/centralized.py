from __future__ import annotations

from .base import BaseTopology, TopologyConfig, parse_json_maybe


class CentralizedTopology(BaseTopology):
    motif_tags = ["centralized", "manager_worker", "dynamic_rounds", "manager_control", "fan_out", "fan_in"]

    def run(self) -> dict:
        self.workflow_start()
        history: list[str] = []
        final_summary = ""
        for round_id in range(self.config.max_rounds):
            manager_prompt = f"Task:\n{self.config.query}\nHistory:\n" + "\n".join(history)
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
                parents=["START"] if round_id == 0 else [f"worker_{i}_r{round_id-1}" for i in range(1, self.config.num_agents + 1)],
                criticality="critical",
            )
            decision = self._manager_decision(round_id, manager_out, history)
            self.trace.emit(
                event_type="manager_decision",
                node_id=f"manager_decision_r{round_id}",
                node_name=f"ManagerDecision-Round-{round_id}",
                node_type="manager",
                round_id=round_id,
                manager_round_id=round_id,
                motif_tags=self.motif_tags,
                decision=decision["decision"],
                selected_workers=decision["selected_workers"],
                stop_reason=decision["stop_reason"],
                confidence_est=decision["confidence_est"],
                decision_tokens_est=decision["decision_tokens_est"],
            )
            if decision["decision"] == "finish" and round_id > 0:
                final_summary = manager_out
                break
            worker_outputs = []
            for worker_id in decision["selected_workers"]:
                self.emit_edge(src_node=f"manager_r{round_id}", dst_node=f"{worker_id}_r{round_id}", artifact_type="plan", content=decision["next_instruction"], transfer_type="broadcast", manager_round_id=round_id, fanout_count=len(decision["selected_workers"]), recipient_count=len(decision["selected_workers"]))
                evidence = self.search(
                    node_id=f"{worker_id}_search_r{round_id}",
                    node_name=f"{worker_id}-Search-Round-{round_id}",
                    query=f"{self.config.query}\n{decision['next_instruction']}",
                )
                out = self.call_llm(
                    node_id=f"{worker_id}_r{round_id}",
                    node_name=f"{worker_id}-Round-{round_id}",
                    node_type="worker",
                    agent_role="worker",
                    prompt_template="worker.md",
                    system_prompt=f"You are {worker_id}.",
                    user_prompt=f"Instruction: {decision['next_instruction']}\nTask: {self.config.query}\nSearch evidence:\n{evidence}",
                    round_id=round_id,
                    manager_round_id=round_id,
                    parents=[f"manager_r{round_id}"],
                    parallel_group=f"centralized_workers_r{round_id}",
                    criticality="near_critical",
                    extra_metadata={"manager_instruction": decision["next_instruction"]},
                )
                worker_outputs.append(out)
                self.emit_edge(src_node=f"{worker_id}_r{round_id}", dst_node=f"manager_r{round_id+1}", artifact_type="evidence", content=out, transfer_type="aggregation", manager_round_id=round_id)
            self.barrier(barrier_id=f"manager_collect_r{round_id}", waiting_for_nodes=[f"{w}_r{round_id}" for w in decision["selected_workers"]], manager_round_id=round_id)
            history.extend(worker_outputs)
            final_summary = "\n".join(history)
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

    def _manager_decision(self, round_id: int, manager_out: str, history: list[str]) -> dict:
        selected = [f"worker_{i}" for i in range(1, self.config.num_agents + 1)]
        if self.config.manager_policy == "llm":
            parsed = parse_json_maybe(manager_out)
            selected = parsed.get("selected_workers") or selected
            decision = parsed.get("decision") or "continue"
            reason = parsed.get("reason") or "llm decision"
            instruction = parsed.get("next_instruction") or "collect evidence"
            confidence = float(parsed.get("confidence") or 0.7)
        else:
            must_continue = round_id < self.config.force_centralized_rounds
            at_limit = round_id >= self.config.max_rounds - 1
            insufficient = not history or "low_confidence" in "\n".join(history)
            decision = "continue" if (must_continue or (insufficient and not at_limit)) else "finish"
            reason = "max_rounds" if at_limit and not must_continue else ("force_continue_rounds" if must_continue else "sufficient_evidence")
            instruction = f"round {round_id}: collect concise evidence and confidence"
            confidence = 0.72 if history else 0.55
        return {
            "decision": decision,
            "selected_workers": selected,
            "stop_reason": reason,
            "next_instruction": instruction,
            "confidence_est": confidence,
            "decision_tokens_est": len(manager_out) // 4,
        }

    def topology_summary(self, events: list[dict]) -> dict:
        decisions = [e for e in events if e.get("event_type") == "manager_decision"]
        fanout = sum(len(e.get("selected_workers") or []) for e in decisions)
        return {"manager_decision_count": len(decisions), "worker_fanout_count": fanout}


def build_workflow(config: TopologyConfig, **deps) -> CentralizedTopology:
    return CentralizedTopology(config, **deps)


def build_motif(config: TopologyConfig, **deps) -> CentralizedTopology:
    return CentralizedTopology(config, **deps)

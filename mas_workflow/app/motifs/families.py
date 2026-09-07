"""Four executable collaboration templates with explicit role and artifact binding.

Connectivity constrains message delivery, coordination specifies who decides,
and a task binding supplies domain semantics. Neither is a base topology object.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import asdict, replace
from typing import Any
from uuid import uuid4

from ..llm_backends import build_llm_backend
from ..runtime import WorkloadRuntime
from ..tracing import stable_hash
from .contracts import AgentInstance, Artifact, MotifResult, RoleSlot, role_bindings


FAMILY_SLOTS = {
    "dispatch_execute": (
        RoleSlot("dispatcher", "Produce concrete instructions for the executor to address the task."),
        RoleSlot("executor", "Execute the supplied instructions using the task and inputs."),
    ),
    "parallel_aggregate": (
        RoleSlot("worker", "Independently address the task from your assigned perspective."),
        RoleSlot("collector", "Combine the worker results into an answer to the task."),
    ),
    "evaluate_refine": (
        RoleSlot("producer", "Produce or revise the candidate using the supplied evaluation feedback."),
        RoleSlot("evaluator", "Evaluate the candidate against the task criteria."),
    ),
    "peer_deliberation": (
        RoleSlot("peer", "Address the task, then update your result using the delivered peer messages."),
    ),
}
MOTIF_NAMES = list(FAMILY_SLOTS)


def peer_sources(index: int, count: int, connectivity: str, seed: int) -> list[int]:
    """Return actual incoming peer edges. Self state is supplied separately."""
    others = [j for j in range(count) if j != index]
    if connectivity == "all_to_all":
        return others
    if connectivity == "ring":
        return [(index - 1) % count]
    if connectivity == "pairwise":
        partner = index ^ 1
        return [partner] if partner < count else []
    if connectivity == "random_k":
        return random.Random(seed + index).sample(others, min(2, len(others)))
    raise ValueError(f"Unsupported peer connectivity: {connectivity}")


class FamilyWorkload(WorkloadRuntime):
    """A run may execute one template or a sequential composition on one trace.

    Instances persist within a motif (e.g. a peer across rounds), but each call
    has a fresh graph node. Cross-motif reuse is limited to explicit artifacts;
    reusing a role binding never reuses its mutable history or agent identity.
    """

    def __init__(self, config, *, spec: dict[str, Any], **deps):
        super().__init__(config, **deps)
        self.spec = spec
        self.motif_tags = ["motif_families_v1"]
        self.config.mode = "motif"
        self.trace.mode = "motif"
        self.trace.motif_instance_id = ""
        self.trace.composed_from_topologies = []
        self.config.composed_from_topologies = []
        self._validate_spec(spec)

    def _validate_spec(self, spec):
        if not isinstance(spec, dict):
            raise ValueError("Workload specification must be an object")
        if "stages" in spec and set(spec) != {"stages"}:
            raise ValueError("Composition root accepts only stages; configure each stage explicitly")
        stages = spec.get("stages", [spec])
        if not isinstance(stages, list) or not stages:
            raise ValueError("Composition requires at least one stage")
        seen = set()
        for i, stage in enumerate(stages):
            allowed = {"id", "family", "roles", "inputs", "task", "width", "rounds", "max_revisions",
                       "connectivity", "aggregation", "dispatch", "criteria"}
            if not isinstance(stage, dict) or set(stage) - allowed:
                raise ValueError("Invalid stage fields")
            family = stage.get("family")
            if family not in FAMILY_SLOTS:
                raise ValueError(f"Unknown family: {family}")
            role_bindings(FAMILY_SLOTS[family], stage.get("roles", {}))
            stage_id = stage.get("id", f"stage_{i}")
            if not isinstance(stage_id, str) or not stage_id or "." in stage_id:
                raise ValueError("Stage id must be a nonempty string without dots")
            if stage_id in seen:
                raise ValueError(f"Duplicate stage id: {stage_id}")
            for port, reference in stage.get("inputs", {}).items():
                if port not in {"context", "candidate"}:
                    raise ValueError(f"Unsupported input port: {port}")
                source, output = reference.rsplit(".", 1)
                if source not in seen or output != "result":
                    raise ValueError(f"Input must reference an earlier stage's result: {reference}")
                if port == "candidate" and family != "evaluate_refine":
                    raise ValueError("Only evaluate_refine accepts a candidate input")
            seen.add(stage_id)
            width = stage.get("width", self.config.num_agents)
            if not isinstance(width, int) or width < 1:
                raise ValueError("width must be a positive integer")
            rounds = stage.get("rounds", self.config.peer_rounds)
            revisions = stage.get("max_revisions", self.config.max_retries)
            if not isinstance(rounds, int) or rounds < 0 or not isinstance(revisions, int) or revisions < 0:
                raise ValueError("rounds and max_revisions must be nonnegative integers")
            if family == "peer_deliberation":
                if width < 2:
                    raise ValueError("peer_deliberation requires at least two peers")
                peer_sources(0, width, stage.get("connectivity", self.config.communication_topology), self.config.random_seed)
            elif "connectivity" in stage:
                expected = {"dispatch_execute": "star", "parallel_aggregate": "fan_out_fan_in", "evaluate_refine": "feedback_pair"}[family]
                if stage["connectivity"] != expected:
                    raise ValueError(f"{family} uses {expected}; unsupported shape override")
            if stage.get("aggregation", self.config.aggregation_policy) not in {"concat_summary", "judge", "vote"}:
                raise ValueError("aggregation must be concat_summary, judge or vote")
            if stage.get("dispatch", "plan") not in {"plan", "route"}:
                raise ValueError("dispatch must be plan or route")

    def _call(self, instance: AgentInstance, task: str, inputs: list[Artifact], *, instruction="", round_id=0, group=None) -> Artifact:
        node = "call_" + uuid4().hex
        binding = instance.binding
        parents = list(dict.fromkeys(a.producer_node for a in inputs))
        for artifact in inputs:
            self.emit_edge(src_node=artifact.producer_node, dst_node=node, artifact_type=artifact.kind,
                           content=artifact.content, transfer_type="artifact_binding", source_artifact_id=artifact.artifact_id,
                           trace_fields={"motif_instance_id": instance.motif_instance_id, "role_slot": instance.slot.name})
        payload = {"task": task, "role_index": instance.index,
                   "inputs": [{"kind": a.kind, "content": a.content} for a in inputs]}
        # Search is an explicit bound action, never enabled merely by the role name.
        if "search" in binding.tools:
            tool_node = node + "_search"
            evidence = self.search(node_id=tool_node, node_name="Bound search", query=task,
                                   trace_fields={"motif_instance_id": instance.motif_instance_id,
                                                 "agent_instance_id": instance.instance_id, "role_slot": instance.slot.name})
            payload["tool_evidence"] = evidence
            parents.append(tool_node)
            self.emit_edge(src_node=tool_node, dst_node=node, artifact_type="evidence",
                           content=str(evidence), transfer_type="tool_result")
        system = binding.instructions + "\n" + instruction
        if binding.output_format == "json":
            system += "\nReturn valid JSON without markdown fences."
        backend = None
        if binding.model or binding.backend_base_url:
            backend = build_llm_backend(self.config.llm_mode, model=binding.model or self.config.model,
                                        backend_base_url=binding.backend_base_url or self.config.backend_base_url,
                                        max_output_tokens=self.config.max_output_tokens)
        text = self.call_llm(
            node_id=node, node_name=instance.slot.name, node_type="agent", agent_role=instance.slot.name,
            prompt_template="task_binding", system_prompt=system,
            user_prompt=json.dumps(payload, ensure_ascii=False), parents=parents, round_id=round_id,
            parallel_group=group, llm_override=backend,
            extra_metadata={"role_slot": instance.slot.name, "role_index": instance.index,
                            "agent_instance_id": instance.instance_id,
                            "motif_instance_id": instance.motif_instance_id,
                            "parent_motif_id": self.trace.workflow_id,
                            "motif_family": self._family_for(instance),
                            "motif_name": self._family_for(instance),
                            "input_artifact_ids": [a.artifact_id for a in inputs],
                            "output_contract": binding.output_format},
        )
        binding.validate(text)
        instance.history.append({"node_id": node, "output": text})
        result = Artifact(text, node)
        self.trace.emit(event_type="artifact_created", node_id=node, node_type="artifact",
                        motif_instance_id=instance.motif_instance_id, role_slot=instance.slot.name,
                        agent_instance_id=instance.instance_id, artifact_id=result.artifact_id,
                        output_hash=stable_hash(text))
        return result

    @staticmethod
    def _family_for(instance):
        # Slots are shared only where semantics match; peer and worker are distinct.
        return next(name for name, slots in FAMILY_SLOTS.items() if instance.slot in slots)

    def execute(self, stage: dict[str, Any], task: str, inputs: dict[str, Artifact]) -> MotifResult:
        family = stage["family"]
        mid = "motif_" + uuid4().hex
        bindings = role_bindings(FAMILY_SLOTS[family], stage.get("roles", {}))
        instances = {}

        def agent(slot, index=0):
            key = (slot, index)
            if key not in instances:
                role = next(s for s in FAMILY_SLOTS[family] if s.name == slot)
                instances[key] = AgentInstance(role, bindings[slot], mid, index)
                self.trace.emit(event_type="agent_instance_created", node_id=instances[key].instance_id,
                                role_slot=slot, role_index=index, motif_instance_id=mid,
                                agent_instance_id=instances[key].instance_id, node_type="agent",
                                binding_hash=stable_hash(asdict(bindings[slot])))
            return instances[key]

        width = stage.get("width", self.config.num_agents)
        context = [inputs["context"]] if "context" in inputs else []
        task = stage.get("task", task)
        shapes = {"dispatch_execute": "star", "parallel_aggregate": "fan_out_fan_in",
                  "evaluate_refine": "feedback_pair", "peer_deliberation": self.config.communication_topology}
        shape = stage.get("connectivity", shapes[family])
        regimes = {"dispatch_execute": "centralized", "parallel_aggregate": "independent",
                   "evaluate_refine": "centralized", "peer_deliberation": "decentralized"}
        self.trace.emit(event_type="motif_start", node_id=mid, motif_instance_id=mid, motif_name=family,
                        parent_motif_id=self.trace.workflow_id, status="start",
                        connectivity=shape, coordination=regimes[family],
                        extra={"stage_id": stage.get("id", ""), "spec": stage,
                               "aggregation_rule": stage.get("aggregation", self.config.aggregation_policy) if family == "parallel_aggregate" else None})
        status, iterations = "completed", 0
        outputs = {}
        try:
            if family == "dispatch_execute":
                route = stage.get("dispatch", "plan") == "route"
                instruction = (f'Return JSON with "worker_index" (integer from 0 to {width - 1}) and "instruction" (string).'
                               if route else "Return an execution plan.")
                plan = self._call(agent("dispatcher"), task, context, instruction=instruction)
                selected = list(range(width))
                if route:
                    decision = json.loads(plan.content)
                    index = decision["worker_index"]
                    if type(index) is not int or index not in selected or not isinstance(decision.get("instruction"), str):
                        raise ValueError("Invalid dispatcher route")
                    selected = [index]
                values = self.run_parallel(selected, lambda i: self._call(agent("executor", i), task, context + [plan], group=mid))
                result = values[0] if len(values) == 1 else self._pack(values, mid)
            elif family == "parallel_aggregate":
                values = self.run_parallel(list(range(width)), lambda i: self._call(agent("worker", i), task, context, group=mid))
                self.barrier(barrier_id=mid + "_join", waiting_for_nodes=[v.producer_node for v in values], trace_fields={"motif_instance_id": mid})
                policy = stage.get("aggregation", self.config.aggregation_policy)
                if policy == "vote":
                    winner = Counter(a.content for a in values).most_common(1)[0][0]
                    result = self._pack(values, mid, content=winner)
                else:
                    instruction = ('Return JSON with "selected_index" (zero-based integer) and "reason". Select one supplied candidate.'
                                   if policy == "judge" else "Synthesize the supplied results into one answer.")
                    combined = self._call(agent("collector"), task, values, instruction=instruction)
                    if policy == "judge":
                        index = json.loads(combined.content)["selected_index"]
                        if type(index) is not int or not 0 <= index < len(values):
                            raise ValueError("Invalid collector selection")
                        result = self._pack([combined, values[index]], mid, content=values[index].content)
                    else:
                        result = combined
            elif family == "evaluate_refine":
                candidate = inputs.get("candidate")
                if candidate is None:
                    candidate = self._call(agent("producer"), task, context)
                limit = stage.get("max_revisions", self.config.max_retries)
                for revision in range(limit + 1):
                    feedback = self._call(agent("evaluator"), task, context + [candidate], round_id=revision,
                                          instruction='Return JSON with "decision": "accept" or "revise", and "feedback": a string. Criteria: ' + stage.get("criteria", "Satisfy the task and its constraints."))
                    decision = json.loads(feedback.content)
                    if decision.get("decision") not in {"accept", "revise"} or not isinstance(decision.get("feedback"), str):
                        raise ValueError("Invalid evaluator decision")
                    if decision["decision"] == "accept":
                        status = "accepted"
                        break
                    if revision == limit:
                        status = "max_revisions"
                        break
                    candidate = self._call(agent("producer"), task, context + [candidate, feedback], round_id=revision + 1)
                    iterations += 1
                # Include the acceptance/termination dependency without another LLM call.
                result = self._pack([candidate, feedback], mid, content=candidate.content)
            else:
                peers = [agent("peer", i) for i in range(width)]
                values = self.run_parallel(peers, lambda p: self._call(p, task, context, group=mid + "_initial"))
                for round_id in range(1, stage.get("rounds", self.config.peer_rounds) + 1):
                    previous = values
                    self.barrier(barrier_id=mid + f"_barrier_{round_id}", waiting_for_nodes=[v.producer_node for v in previous], peer_round_id=round_id, trace_fields={"motif_instance_id": mid})
                    def update(p):
                        sources = peer_sources(p.index, width, shape, self.config.random_seed)
                        return self._call(p, task, context + [replace(previous[p.index], kind="own_state")] + [replace(previous[j], kind="peer_message") for j in sources],
                                          instruction="The first peer result is your prior state; the remaining results are incoming peer messages.",
                                          round_id=round_id, group=mid + f"_round_{round_id}")
                    values = self.run_parallel(peers, update)
                    iterations = round_id
                # Fixed rounds do not imply consensus. Return all final peer results.
                result = self._pack(values, mid)
            outputs["result"] = result
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = ""
        self.trace.emit(event_type="motif_end", node_id=mid, motif_instance_id=mid, motif_name=family,
                        parent_motif_id=self.trace.workflow_id, status=status,
                        extra={"iterations": iterations, "error": error,
                               "output_artifact_ids": {k: v.artifact_id for k, v in outputs.items()}})
        return MotifResult(family, mid, status, outputs, iterations, error)

    def _pack(self, values, mid, *, content=None):
        node = "combine_" + uuid4().hex
        for value in values:
            self.emit_edge(src_node=value.producer_node, dst_node=node, artifact_type=value.kind,
                           content=value.content, transfer_type="aggregation", source_artifact_id=value.artifact_id,
                           trace_fields={"motif_instance_id": mid})
        text = content if content is not None else json.dumps([v.content for v in values], ensure_ascii=False)
        result = Artifact(text, node)
        self.trace.emit(event_type="artifact_created", node_id=node, node_type="artifact",
                        parents=[v.producer_node for v in values], motif_instance_id=mid,
                        artifact_id=result.artifact_id, output_hash=stable_hash(text))
        return result

    def run(self):
        self.workflow_start()
        available, records = {}, []
        final, status = "", "completed"
        for i, stage in enumerate(self.spec.get("stages", [self.spec])):
            inputs = {port: available[reference] for port, reference in stage.get("inputs", {}).items()}
            result = self.execute(stage, self.config.query, inputs)
            records.append(asdict(result))
            status = result.status
            if "result" in result.outputs:
                final = result.outputs["result"].content
            if status not in {"completed", "accepted"}:
                break
            name = stage.get("id", f"stage_{i}")
            available.update({name + "." + key: artifact for key, artifact in result.outputs.items()})
        self._results = records
        return self.workflow_end(final, status=status)

    def topology_summary(self, events):
        results = getattr(self, "_results", [])
        return {"motif_results": results, "workload_schema": "motif_families_v1",
                "motif_instance_id": results[0]["motif_instance_id"] if len(results) == 1 else "",
                "composed_from_topologies": []}

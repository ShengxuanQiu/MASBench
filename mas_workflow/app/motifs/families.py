"""Four executable collaboration templates with explicit role and artifact binding.

Connectivity constrains message delivery, coordination specifies who decides,
and a task binding supplies domain semantics. Neither is a base topology object.
"""
from __future__ import annotations

import json
from copy import deepcopy
import random
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict, replace
from typing import Any
from uuid import uuid4

from ..llm_backends import build_llm_backend
from ..runtime import WorkloadRuntime
from ..tracing import stable_hash, estimate_tokens
from .contracts import AgentInstance, Artifact, MotifResult, role_bindings
from .task_defaults import FAMILY_SLOTS
from ..specs import ROLE_ALIASES, stage_dependencies

MOTIF_NAMES = list(FAMILY_SLOTS)
PUBLIC_FAMILY = {"dispatch_execute":"Spawn","parallel_aggregate":"Fork--Join",
                 "evaluate_refine":"Refinement Loop","peer_exchange":"Debate","atomic":"Atomic"}
# Names accepted by the compatibility parser; exported semantics use Part 3 names.
LEGACY_PUBLIC_FAMILY = {"DispatchExecute":"Spawn", "ParallelAggregate":"Fork--Join",
                        "EvaluateRefine":"Refinement Loop", "PeerExchange":"Debate", "PeerDeliberation":"Debate"}


def stage_semantics(stage, participants):
    family=stage['family']
    connectivity={"dispatch_execute":"star","parallel_aggregate":"fan_out_fan_in",
                  "evaluate_refine":"feedback_pair","peer_exchange":stage.get('connectivity','all_to_all'),
                  "atomic":"single_operation"}[family]
    coordination={"dispatch_execute":"centralized_dispatch","parallel_aggregate":"independent_then_reduce",
                  "evaluate_refine":"review_feedback_control","peer_exchange":"decentralized_exchange",
                  "atomic":"local"}[family]
    completion={"dispatch_execute":"selected_work_returned","parallel_aggregate":"required_branches_completed",
                "evaluate_refine":"accepted_or_revision_limit","peer_exchange":"fixed_rounds",
                "atomic":"operation_completed"}[family]
    if family=='dispatch_execute': realized=[{'role':'Coordinator','index':0}]+[{'role':'Worker','index':i} for i in participants]
    elif family=='parallel_aggregate': realized=[{'role':'Worker','index':i} for i in participants]+[{'role':'Reducer','index':0}]
    elif family=='evaluate_refine': realized=[{'role':'Worker','index':0},{'role':'Reviewer','index':0}]
    elif family=='peer_exchange': realized=[{'role':'Worker','index':i} for i in participants]
    else: realized=[{'role':stage['atomic']['role'],'index':0}]
    return {'canonical_family':PUBLIC_FAMILY[family],'participants':realized,'connectivity':connectivity,
            'coordination':coordination,'delivery':stage.get('motif_delivery',{}),'completion':completion,'context_construction':{'delivery':stage.get('motif_delivery',{}),'reuse_scope_id':stage.get('reuse_scope_id')}}


def peer_sources(index: int, count: int, connectivity: str, seed: int, k=None, sparsity=None) -> list[int]:
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
        return random.Random(seed + index).sample(others, min(k if k is not None else (round((1-sparsity)*len(others)) if sparsity is not None else 2), len(others)))
    raise ValueError(f"Unsupported peer connectivity: {connectivity}")


class FamilyWorkload(WorkloadRuntime):
    """A run expands a dependency-aware workflow into motif instances on one trace.

    Instances persist within a motif (e.g. a peer across rounds), but each call
    has a fresh graph node. Cross-motif reuse is limited to explicit artifacts;
    reusing a role binding never reuses its mutable history or agent identity.
    """

    def __init__(self, config, *, spec: dict[str, Any], **deps):
        super().__init__(config, **deps)
        spec=deepcopy(spec)
        for stage in spec.get('stages',[spec]):
            if stage.get('family')=='peer_deliberation':stage['family']='peer_exchange'
        self.spec = spec
        self.motif_tags = ["motif_families_v1"]
        self.config.mode = "motif"
        self.trace.mode = "motif"
        self.trace.motif_instance_id = ""
        self.trace.composed_from_topologies = []
        self.config.composed_from_topologies = []
        from ..execution_graph import ExecutionGraph
        self.trace.canonical_enabled = True
        self.trace.execution_graph = ExecutionGraph()
        self._stage_context = {}
        self._context_lock = threading.Lock()
        self._llm_slots = threading.BoundedSemaphore(config.max_concurrent_llm_calls)
        self._validate_spec(spec)

    def _validate_spec(self, spec):
        if not isinstance(spec, dict):
            raise ValueError("Workload specification must be an object")
        if "stages" in spec and set(spec) != {"stages"}:
            raise ValueError("Composition root accepts only stages; configure each stage explicitly")
        stages = spec.get("stages", [spec])
        if not isinstance(stages, list) or not stages:
            raise ValueError("Composition requires at least one stage")
        self._dependencies = stage_dependencies(stages, legacy_sequential=not any("depends_on" in s for s in stages))
        seen = set()
        for i, stage in enumerate(stages):
            allowed = {"id", "family", "roles", "inputs", "task", "width", "rounds", "max_revisions",
                       "connectivity", "aggregation", "dispatch", "criteria", "depends_on", "seed",
                       "routing_semantics", "evaluation_semantics", "atomic", "condition", "participants", "delivery", "motif_delivery", "allow_skipped", "parameters", "delivery_instruction", "reuse_scope_id", "k", "sparsity"}
            if not isinstance(stage, dict) or set(stage) - allowed:
                raise ValueError("Invalid stage fields")
            family = stage.get("family")
            if family == "atomic":
                from ..specs import AtomicStage
                AtomicStage(**stage["atomic"])
                continue
            if family not in FAMILY_SLOTS:
                raise ValueError(f"Unknown family: {family}")
            role_bindings(FAMILY_SLOTS[family], stage.get("roles", {}))
            from ..specs import delivery_spec
            relations={"dispatch_execute":{"coordinator_to_worker"},"parallel_aggregate":{"worker_to_reducer"},
                       "evaluate_refine":{"producer_to_reviewer","reviewer_to_producer"},"peer_exchange":{"peer_to_peer"}}[family]
            if set(stage.get('motif_delivery',{}))-relations:raise ValueError('Invalid motif delivery relation')
            for value in stage.get('motif_delivery',{}).values():delivery_spec(value)
            for value in stage.get('delivery',{}).values():delivery_spec(value)
            stage_id = stage.get("id", f"stage_{i}")
            if not isinstance(stage_id, str) or not stage_id or "." in stage_id:
                raise ValueError("Stage id must be a nonempty string without dots")
            if stage_id in seen:
                raise ValueError(f"Duplicate stage id: {stage_id}")
            if "candidate" in stage.get("inputs", {}) and family != "evaluate_refine":
                raise ValueError("Only evaluate_refine accepts a candidate input")
            seen.add(stage_id)
            width = stage.get("width", self.config.num_agents)
            if not isinstance(width, int) or width < 1:
                raise ValueError("width must be a positive integer")
            rounds = stage.get("rounds", self.config.peer_rounds)
            revisions = stage.get("max_revisions", self.config.max_retries)
            if not isinstance(rounds, int) or rounds < 0 or not isinstance(revisions, int) or revisions < 0:
                raise ValueError("rounds and max_revisions must be nonnegative integers")
            if family == "peer_exchange":
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
        ctx = self._stage_context[instance.motif_instance_id]
        control = list(ctx["parents"]) + list(ctx["round_parents"].get(round_id, []))
        parents = list(dict.fromkeys([a.producer_node for a in inputs] + control))
        identity = {"stage_instance_id": ctx["id"], "role": ROLE_ALIASES[instance.slot.name],
                    "motif_instance_id": "" if ctx.get("atomic") else instance.motif_instance_id,
                    "atomic_instance_id":instance.motif_instance_id if ctx.get('atomic') else '',
                    "agent_instance_id": instance.instance_id}
        payload = {"task": task, "role_index": instance.index,
                   "inputs": [{"kind": a.kind, "content": a.content} for a in inputs]}
        # Search is an explicit bound action, never enabled merely by the role name.
        if "search" in binding.tools:
            tool_node = node + "_search"
            self.trace.emit(event_type="operation_start", node_id=tool_node, operation_kind="tool", parents=parents, **identity)
            started = time.perf_counter()
            try:
                evidence = self.search(node_id=tool_node, node_name="Bound search", query=task, trace_fields=identity)
            except Exception:
                self.trace.emit(event_type="operation_fail", node_id=tool_node, **identity)
                raise
            self.trace.emit(event_type="operation_finish", node_id=tool_node, tool_snapshot=evidence,
                            duration_sec=time.perf_counter() - started, **identity)
            tool_artifact = Artifact(json.dumps(evidence, ensure_ascii=False), tool_node, "evidence")
            self._produce(tool_artifact, identity)
            with self._context_lock:
                ctx["nodes"].append(tool_node)
            payload["tool_evidence"] = evidence
            parents.append(tool_node)
            self.emit_edge(src_node=tool_node, dst_node=node, artifact_type="evidence",
                           content=str(evidence), transfer_type="tool_result")
        binding.validate_input(payload)
        system = binding.instructions + "\n" + instruction
        if binding.output_format == "json":
            system += "\nReturn valid JSON without markdown fences."
        backend = None
        if binding.model or binding.backend_base_url:
            backend = build_llm_backend(self.config.llm_mode, model=binding.model or self.config.model,
                                        backend_base_url=binding.backend_base_url or self.config.backend_base_url,
                                        max_output_tokens=self.config.max_output_tokens)
        for parent in control:
            self.trace.emit(event_type="dependency", src=parent, dst=node, dependency_kind="control", **identity)
        for artifact in inputs:
            self.trace.emit(event_type="dependency", src=artifact.producer_node, dst=node, dependency_kind="data", **identity)
            self.trace.emit(event_type="artifact_consume", node_id=node, artifact_id=artifact.artifact_id, **identity)
            self.trace.emit(event_type="artifact_delivery",node_id=node,artifact_id=artifact.artifact_id,
                            source_artifact_ids=list(artifact.source_artifact_ids or (artifact.artifact_id,)),
                            delivered_artifact_id=artifact.artifact_id,
                            producer_operation_id=artifact.producer_node,consumer_operation_id=node,
                            delivery_mode=artifact.delivery_mode,delivery_selector=artifact.delivery_selector,
                            delivery_transform=artifact.delivery_transform,
                            materialized_bytes=len(artifact.content.encode('utf-8')),
                            materialized_tokens_est=estimate_tokens(artifact.content),**identity)
            self.emit_edge(src_node=artifact.producer_node, dst_node=node, artifact_type=artifact.kind,
                           content=artifact.content, transfer_type="artifact_binding", source_artifact_id=artifact.artifact_id,
                           trace_fields={"motif_instance_id": instance.motif_instance_id, "role_slot": instance.slot.name})
        if "search" in binding.tools:
            self.trace.emit(event_type="artifact_consume", node_id=node, artifact_id=tool_artifact.artifact_id, **identity)
            self.trace.emit(event_type="dependency", src=tool_node, dst=node, dependency_kind="data", **identity)
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
                            "output_contract": binding.output_format,
                            "reuse_scope_id": ctx["stage"].get("reuse_scope_id") or ("session:" + instance.instance_id), **identity},
        )
        with self._context_lock:
            ctx["nodes"].append(node)
        binding.validate(text)
        instance.history.append({"node_id": node, "output": text})
        result = Artifact(text, node)
        self._produce(result, identity)
        return result

    def _produce(self, artifact, identity):
        self.trace.emit(event_type="artifact_created", node_id=artifact.producer_node, node_type="artifact",
                        artifact_id=artifact.artifact_id, content=artifact.content,
                        output_hash=stable_hash(artifact.content), **identity)

    @staticmethod
    def _family_for(instance):
        # Slots are shared only where semantics match; peer and worker are distinct.
        return next((name for name, slots in FAMILY_SLOTS.items() if instance.slot in slots), "atomic")

    def execute(self, stage: dict[str, Any], task: str, inputs: dict[str, Artifact], *, control_parents=()) -> MotifResult:
        family = stage["family"]
        mid = ("atomic_" if family == "atomic" else "motif_") + uuid4().hex
        sid = "stage_" + uuid4().hex
        self._stage_context[mid] = {"id": sid, "parents": list(control_parents), "nodes": [], "round_parents": {}, "stage": stage}
        width = stage.get("width", self.config.num_agents)
        participant_ids=stage.get("selected_participants",list(range(width)))
        semantics=stage_semantics(stage,participant_ids)
        realized_participants=semantics['participants']
        self.trace.emit(event_type="stage_start", stage_instance_id=sid, motif_instance_id=mid if family != 'atomic' else '',
                        atomic_instance_id=mid if family == 'atomic' else '',
                        stage_id=stage.get("id", ""), parents=list(control_parents),
                        stage_dependencies=list(stage.get('_realized_dependencies',stage.get('depends_on',[]))),stage_semantics=semantics,
                        realized_participants=semantics['participants'],activation='active')
        from .adaptive import atomic_stage, deliver, deliver_relation
        if family == "atomic":
            try: return atomic_stage(self,stage,inputs,mid)
            except Exception as exc:
                self.trace.emit(event_type="stage_finish",stage_instance_id=sid,status="failed",error=str(exc),completion_reason='failed')
                return MotifResult("atomic",mid,"failed",{},error=str(exc))
        try:
            inputs = deliver(self,stage,inputs,mid)
        except Exception as exc:
            self.trace.emit(event_type='stage_finish',stage_instance_id=sid,status='failed',error=str(exc),completion_reason='failed')
            return MotifResult(family,mid,'failed',{},error=str(exc))
        bindings = role_bindings(FAMILY_SLOTS[family], stage.get("roles", {}))
        instances = {}

        def agent(slot, index=0):
            key = (slot, index)
            if key not in instances:
                role = next(s for s in FAMILY_SLOTS[family] if s.name == slot)
                instances[key] = AgentInstance(role, bindings[slot], mid, index)
                self.trace.emit(event_type="agent_instance_created", node_id=instances[key].instance_id,
                                role_slot=slot, role=ROLE_ALIASES[slot], stage_instance_id=sid, role_index=index, motif_instance_id=mid,
                                agent_instance_id=instances[key].instance_id, node_type="agent",
                                binding_hash=stable_hash(asdict(bindings[slot])))
            return instances[key]

        context = inputs.get("context", [])
        context = context if isinstance(context, list) else [context]
        task = stage.get("task", task)
        shapes = {"dispatch_execute": "star", "parallel_aggregate": "fan_out_fan_in",
                  "evaluate_refine": "feedback_pair", "peer_exchange": self.config.communication_topology}
        shape = stage.get("connectivity", shapes[family])
        regimes = {"dispatch_execute": "centralized", "parallel_aggregate": "independent",
                   "evaluate_refine": "centralized", "peer_exchange": "decentralized"}
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
                instruction = (f'Return JSON with "worker_index" (one of {participant_ids}) and "instruction" (string).'
                               if route else "Return an execution plan.")
                instruction += "\n" + stage.get("routing_semantics", "")
                plan = self._call(agent("dispatcher"), task, context, instruction=instruction)
                selected = list(participant_ids)
                if route:
                    decision = json.loads(plan.content)
                    index = decision["worker_index"]
                    if type(index) is not int or index not in selected or not isinstance(decision.get("instruction"), str):
                        raise ValueError("Invalid dispatcher route")
                    selected = [index]
                    realized_participants=[{'role':'Coordinator','index':0},{'role':'Worker','index':index}]
                    self.trace.emit(event_type="control_decision", node_id=plan.producer_node, decision=decision, motif_instance_id=mid, stage_instance_id=sid)
                delivered_plan=deliver_relation(self,stage,'coordinator_to_worker',[plan],mid)
                values = self.run_parallel(selected, lambda i: self._call(agent("executor", i), task, context + delivered_plan, group=mid))
                result = values[0] if len(values) == 1 else self._pack(values, mid)
            elif family == "parallel_aggregate":
                values = self.run_parallel(participant_ids, lambda i: self._call(agent("worker", i), task, context, group=mid))
                self.barrier(barrier_id=mid + "_join", waiting_for_nodes=[v.producer_node for v in values], trace_fields={"motif_instance_id": mid})
                reduced_values=deliver_relation(self,stage,'worker_to_reducer',values,mid)
                policy = stage.get("aggregation", self.config.aggregation_policy)
                if policy == "vote":
                    winner = Counter(a.content for a in reduced_values).most_common(1)[0][0]
                    result = self._pack(reduced_values, mid, content=winner)
                    self.trace.emit(event_type="control_decision", node_id=result.producer_node,
                                    decision={"policy": "vote", "winner": winner}, motif_instance_id=mid, stage_instance_id=sid)
                else:
                    instruction = ('Return JSON with "selected_index" (zero-based integer) and "reason". Select one supplied candidate.'
                                   if policy == "judge" else "Synthesize the supplied results into one answer.")
                    combined = self._call(agent("collector"), task, reduced_values, instruction=instruction)
                    if policy == "judge":
                        index = json.loads(combined.content)["selected_index"]
                        if type(index) is not int or not 0 <= index < len(reduced_values):
                            raise ValueError("Invalid collector selection")
                        self.trace.emit(event_type="control_decision", node_id=combined.producer_node, decision={"selected_index": index}, motif_instance_id=mid, stage_instance_id=sid)
                        result = self._pack([combined, reduced_values[index]], mid, content=reduced_values[index].content)
                    else:
                        result = combined
            elif family == "evaluate_refine":
                candidate = inputs.get("candidate")
                if candidate is None:
                    candidate = self._call(agent("producer"), task, context)
                limit = stage.get("max_revisions", self.config.max_retries)
                for revision in range(limit + 1):
                    review_input=deliver_relation(self,stage,'producer_to_reviewer',[candidate],mid)
                    feedback = self._call(agent("evaluator"), task, context + review_input, round_id=revision,
                                          instruction='Return JSON with "decision": "accept" or "revise", and "feedback": a string. Criteria: ' + stage.get("criteria", "Satisfy the task and its constraints.") + "\n" + stage.get("evaluation_semantics", ""))
                    decision = json.loads(feedback.content)
                    if decision.get("decision") not in {"accept", "revise"} or not isinstance(decision.get("feedback"), str):
                        raise ValueError("Invalid evaluator decision")
                    self.trace.emit(event_type="control_decision", node_id=feedback.producer_node, decision=decision, motif_instance_id=mid, stage_instance_id=sid)
                    if decision["decision"] == "accept":
                        status = "accepted"
                        break
                    if revision == limit:
                        status = "max_revisions"
                        break
                    delivered_feedback=deliver_relation(self,stage,'reviewer_to_producer',[feedback],mid)
                    candidate = self._call(agent("producer"), task, context + [candidate]+delivered_feedback, round_id=revision + 1)
                    iterations += 1
                # Include the acceptance/termination dependency without another LLM call.
                result = self._pack([candidate, feedback], mid, content=candidate.content)
            else:
                peers = [agent("peer", i) for i in participant_ids]
                values = self.run_parallel(peers, lambda p: self._call(p, task, context, group=mid + "_initial"))
                for round_id in range(1, stage.get("rounds", self.config.peer_rounds) + 1):
                    previous = values
                    self.barrier(barrier_id=mid + f"_barrier_{round_id}", waiting_for_nodes=[v.producer_node for v in previous], peer_round_id=round_id, trace_fields={"motif_instance_id": mid})
                    self._stage_context[mid]["round_parents"][round_id] = [v.producer_node for v in previous]
                    def update(p):
                        sources = peer_sources(participant_ids.index(p.index), len(peers), shape, stage.get("seed", self.config.random_seed),stage.get("k"),stage.get("sparsity"))
                        messages=deliver_relation(self,stage,'peer_to_peer',[replace(previous[j], kind="peer_message") for j in sources],mid)
                        return self._call(p, task, context + [replace(previous[participant_ids.index(p.index)], kind="own_state")] + messages,
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
        reason = ('failed' if status=='failed' else 'accepted' if family=='evaluate_refine' and status=='accepted' else
                  'revision_limit' if family=='evaluate_refine' else 'selected_work_returned' if family=='dispatch_execute' else
                  'required_branches_completed' if family=='parallel_aggregate' else 'fixed_rounds_completed')
        self.trace.emit(event_type="stage_finish", stage_instance_id=sid, motif_instance_id=mid, status=status,
                        completion_reason=reason,realized_participants=realized_participants,
                        stage_semantics={**semantics,'participants':realized_participants})
        return MotifResult(family, mid, status, outputs, iterations, error)

    def _pack(self, values, mid, *, content=None):
        node = "combine_" + uuid4().hex
        ctx = self._stage_context[mid]
        identity = {"motif_instance_id": mid, "stage_instance_id": ctx["id"], "agent_instance_id": "", "role": "Reducer"}
        self.trace.emit(event_type="operation_start", node_id=node, operation_kind="tool",
                        parents=[v.producer_node for v in values], **identity)
        started = time.perf_counter()
        for value in values:
            self.trace.emit(event_type="dependency", src=value.producer_node, dst=node, dependency_kind="data", **identity)
            self.trace.emit(event_type="artifact_consume", node_id=node, artifact_id=value.artifact_id, **identity)
            self.trace.emit(event_type="artifact_delivery",node_id=node,artifact_id=value.artifact_id,
                            delivered_artifact_id=value.artifact_id,consumer_operation_id=node,
                            source_artifact_ids=list(value.source_artifact_ids or (value.artifact_id,)),
                            producer_operation_id=value.producer_node,delivery_mode=value.delivery_mode,
                            delivery_selector=value.delivery_selector,delivery_transform=value.delivery_transform,
                            materialized_bytes=len(value.content.encode('utf-8')),
                            materialized_tokens_est=estimate_tokens(value.content),**identity)
            self.emit_edge(src_node=value.producer_node, dst_node=node, artifact_type=value.kind,
                           content=value.content, transfer_type="aggregation", source_artifact_id=value.artifact_id,
                           trace_fields={"motif_instance_id": mid})
        text = content if content is not None else json.dumps([v.content for v in values], ensure_ascii=False)
        result = Artifact(text, node)
        self.trace.emit(event_type="operation_finish", node_id=node, tool_snapshot=text,
                        duration_sec=time.perf_counter() - started, **identity)
        self._produce(result, identity)
        with self._context_lock:
            ctx["nodes"].append(node)
        return result

    def run(self):
        self.workflow_start()
        stages = self.spec.get("stages", [self.spec])
        named = {s.get("id", f"stage_{i}"): s for i, s in enumerate(stages)}
        available, completed, pending, active = {}, {}, set(named), {}
        status = "completed"
        with ThreadPoolExecutor(max_workers=max(1, len(stages))) as pool:
            while pending or active:
                if status in {"completed", "accepted"}:
                    for name in named:
                        if name not in pending or not self._dependencies[name] <= completed.keys():
                            continue
                        from .adaptive import decision_value, record_choice, skip_stage
                        stage = dict(named[name])
                        stage['_realized_dependencies']=sorted(self._dependencies[name])
                        if any(completed[p].status == "skipped" for p in self._dependencies[name]) and not stage.get("allow_skipped"):
                            completed[name]=skip_stage(self,name,"prerequisite_skipped",stage);pending.remove(name);continue
                        if stage.get("condition"):
                            selector=stage["condition"]
                            value=decision_value(self,selector,completed)
                            activated=value in selector["in"] if "in" in selector else value==selector.get("equals",True)
                            record_choice(self,selector,completed,{"kind":"stage_activation","stage_id":name,"active":activated})
                            if not activated:
                                completed[name]=skip_stage(self,name,"condition_false",stage);pending.remove(name);continue
                        if stage.get("participants"):
                            ids=decision_value(self,stage["participants"],completed)
                            width=stage.get("width",self.config.num_agents)
                            if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)) or any(type(i) is not int or i<0 or i>=width for i in ids):
                                raise ValueError("Invalid participant subset")
                            if stage.get("family") in {"peer_exchange","peer_deliberation"} and len(ids)<2: raise ValueError("Peers require two participants")
                            stage["selected_participants"]=ids
                            record_choice(self,stage["participants"],completed,{"kind":"participants","stage_id":name,"indices":ids})
                        inputs={}
                        for port,refs in stage.get("inputs",{}).items():
                            refs=refs if isinstance(refs,list) else [refs]
                            edges=[r if isinstance(r,dict) else {'source':r,'delivery':{'mode':'full'}} for r in refs]
                            values=[replace(available[e['source']],delivery_spec=e.get('delivery',{'mode':'full'}),
                                            delivery_scope=e.get('delivery_scope','edge'))
                                    for e in edges if e['source'] in available]
                            if len(values)!=len(edges) and not stage.get("allow_skipped"): raise ValueError("Missing prerequisite artifact")
                            if values: inputs[port]=values[0] if port=="candidate" else values
                        parents = [oid for dep in sorted(self._dependencies[name])
                                   for oid in self._stage_context[completed[dep].motif_instance_id]["nodes"]]
                        active[pool.submit(self.execute, stage, self.config.query, inputs, control_parents=parents)] = name
                        pending.remove(name)
                if not active:
                    if pending and status in {"completed","accepted"}: continue
                    break
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    name = active.pop(future)
                    result = future.result()
                    completed[name] = result
                    if result.status not in {"completed", "accepted"}:
                        status = result.status
                    available.update({name + "." + key: value for key, value in result.outputs.items()})
        self._results = [asdict(completed[name]) for name in named if name in completed]
        if self.trace.execution_graph.operations:
            self.trace.execution_graph.validate()
        graph_path = self.trace.trace_path.with_name(self.config.run_id + "_execution_graph.json")
        self._graph_path = str(graph_path)
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(json.dumps(self.trace.execution_graph.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        sinks = [name for name in completed if not any(name in ds for ds in self._dependencies.values())]
        finals = {name: completed[name].outputs["result"].content for name in sinks if "result" in completed[name].outputs}
        final = next(iter(finals.values())) if len(finals) == 1 else json.dumps(finals, ensure_ascii=False)
        if status == "completed" and len(completed) == 1:
            status = next(iter(completed.values())).status
        return self.workflow_end(final, status=status)

    def topology_summary(self, events):
        results = getattr(self, "_results", [])
        return {"motif_results": results, "workload_schema": "motif_families_v1",
                "canonical_trace_schema": "masbench_execution_v1", "execution_graph_path": getattr(self, "_graph_path", ""),
                "motif_instance_id": results[0]["motif_instance_id"] if len(results) == 1 else "",
                "composed_from_topologies": []}

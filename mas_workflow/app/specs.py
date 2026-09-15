"""Public benchmark factors. Compilation is the only structure/task join point."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROLES = {"Coordinator", "Worker", "Reducer", "Reviewer"}
ROLE_ALIASES = {"dispatcher": "Coordinator", "executor": "Worker", "worker": "Worker",
                "Coordinator":"Coordinator", "Worker":"Worker", "Reducer":"Reducer", "Reviewer":"Reviewer", "collector": "Reducer", "producer": "Worker", "evaluator": "Reviewer", "peer": "Worker"}
FAMILIES = {"DispatchExecute": "dispatch_execute", "ParallelAggregate": "parallel_aggregate",
            "EvaluateRefine": "evaluate_refine", "PeerExchange": "peer_exchange",
            "PeerDeliberation": "peer_exchange", "peer_deliberation": "peer_exchange"}
DELIVERY_MODES = {"full", "selected", "summarized", "referenced", "retrieved"}


@dataclass(frozen=True)
class DeliverySpec:
    """Information semantics attached to a producer-consumer edge."""
    mode: str = "full"
    selector: dict[str, Any] = field(default_factory=dict)
    transform: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in DELIVERY_MODES:
            raise ValueError("Unknown delivery mode")
        if set(self.selector) - {"artifact_indices", "json_fields", "indices"}:
            raise ValueError("Unknown delivery selector")
        indices = self.selector.get("artifact_indices", self.selector.get("indices"))
        if indices is not None and (not isinstance(indices, list) or not indices or
                                    len(set(indices)) != len(indices) or
                                    any(type(i) is not int or i < 0 for i in indices)):
            raise ValueError("artifact_indices must be unique nonnegative integers")
        fields = self.selector.get("json_fields")
        if fields is not None and (not isinstance(fields, list) or not fields or
                                   any(not isinstance(v, str) or not v for v in fields)):
            raise ValueError("json_fields must be nonempty strings")
        if self.mode != "selected" and self.selector:
            raise ValueError("Selectors require selected delivery")
        if set(self.transform) - {"instruction", "retriever", "format"}:
            raise ValueError("Unknown delivery transform metadata")


def delivery_spec(value=None):
    if value is None:
        return DeliverySpec()
    if isinstance(value, DeliverySpec):
        return value
    if isinstance(value, str):
        return DeliverySpec(value)
    if isinstance(value, dict):
        # Legacy task parameters used {indices:[...]}; keep them deterministic.
        value = dict(value)
        if "indices" in value and "mode" not in value:
            return DeliverySpec("selected", {"artifact_indices": value["indices"]})
        return DeliverySpec(**value)
    raise ValueError("Delivery must be a mode string, object, or DeliverySpec")


def input_source(value):
    return value if isinstance(value, str) else value.get("source") if isinstance(value, dict) else None


def positive(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class MotifSpec:
    family: str
    width: int = 3
    rounds: int = 1
    max_revisions: int = 0
    connectivity: str | None = None
    coordination: str | None = None
    dispatch: str = "plan"
    aggregation: str = "concat_summary"
    seed: int = 42
    k: int | None = None
    sparsity: float | None = None
    delivery: dict[str, DeliverySpec] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "family", FAMILIES.get(self.family, self.family))
        if self.family not in FAMILIES.values():
            raise ValueError("Unknown motif family")
        positive(self.width, "width")
        if self.k is not None:
            positive(self.k, "k", 0)
            if self.k >= self.width: raise ValueError("k must be smaller than width")
        if self.sparsity is not None and not 0 <= self.sparsity <= 1:
            raise ValueError("sparsity must lie in [0,1]")
        if self.k is not None and self.sparsity is not None:
            raise ValueError("Use k or sparsity")
        if (self.k is not None or self.sparsity is not None) and (self.family != 'peer_exchange' or self.connectivity != 'random_k'):
            raise ValueError('k/sparsity require PeerDeliberation random_k connectivity')
        positive(self.rounds, "rounds", 0)
        positive(self.max_revisions, "max_revisions", 0)
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        shapes = {"dispatch_execute": {"star"}, "parallel_aggregate": {"fan_out_fan_in"},
                  "evaluate_refine": {"feedback_pair"}, "peer_exchange": {"all_to_all", "ring", "pairwise", "random_k"}}
        if self.connectivity is not None and self.connectivity not in shapes[self.family]:
            raise ValueError("Unsupported motif connectivity")
        if self.family == "peer_exchange" and self.width < 2:
            raise ValueError("PeerDeliberation requires width >= 2")
        regimes = {"dispatch_execute": "centralized", "parallel_aggregate": "independent",
                   "evaluate_refine": "centralized", "peer_exchange": "decentralized"}
        if self.coordination not in {None, regimes[self.family]}:
            raise ValueError("Unsupported motif coordination")
        if self.dispatch not in {"plan", "route"} or self.aggregation not in {"concat_summary", "judge", "vote"}:
            raise ValueError("Unsupported dispatch/aggregation structure")
        relations = {"dispatch_execute":{"coordinator_to_worker"},
                     "parallel_aggregate":{"worker_to_reducer"},
                     "evaluate_refine":{"producer_to_reviewer","reviewer_to_producer"},
                     "peer_exchange":{"peer_to_peer"}}[self.family]
        if set(self.delivery) - relations:
            raise ValueError("Delivery relation is not part of this canonical motif")
        object.__setattr__(self, "delivery", {k:delivery_spec(v) for k,v in self.delivery.items()})


@dataclass(frozen=True)
class AtomicStage:
    kind: str = "llm"
    role: str = "Worker"
    def __post_init__(self):
        if self.kind not in {"llm", "transform", "router", "tool"} or self.role not in ROLES:
            raise ValueError("Invalid AtomicStage kind/role")


@dataclass(frozen=True)
class StageSpec:
    id: str
    motif: str | None = None
    depends_on: tuple[str, ...] = ()
    inputs: dict[str, str | list[str]] = field(default_factory=dict)
    atomic: AtomicStage | None = None
    condition: dict[str, Any] | None = None
    participants: dict[str, Any] | None = None
    delivery: dict[str, DeliverySpec] = field(default_factory=dict)
    allow_skipped: bool = False

    def __post_init__(self):
        if (self.motif is None) == (self.atomic is None):
            raise ValueError("Stage references exactly one motif or AtomicStage")
        if isinstance(self.atomic, dict): object.__setattr__(self, "atomic", AtomicStage(**self.atomic))
        if set(self.delivery) - {"context","candidate"}:
            raise ValueError("Invalid delivery mode/port")
        object.__setattr__(self, "delivery", {k:delivery_spec(v) for k,v in self.delivery.items()})
        normalized_inputs={}
        for port, raw in self.inputs.items():
            refs = raw if isinstance(raw,list) else [raw]
            normalized=[]
            for ref in refs:
                source=input_source(ref)
                if not source:
                    raise ValueError("Input edge requires source")
                if isinstance(ref,dict):
                    if set(ref)-{"source","delivery"}: raise ValueError("Unknown input edge field")
                    normalized.append({"source":source,"delivery":delivery_spec(ref.get("delivery",self.delivery.get(port)))})
                else: normalized.append(ref)
            normalized_inputs[port]=normalized if isinstance(raw,list) else normalized[0]
        object.__setattr__(self,"inputs",normalized_inputs)
        for control in (self.condition, self.participants):
            if control is not None and (not isinstance(control,dict) or not control.get("from_stage") or set(control)-{"from_stage","field","equals","in"}):
                raise ValueError("Invalid stage decision selector")
        if self.condition and ('in' in self.condition and (not isinstance(self.condition['in'],list) or 'equals' in self.condition)):
            raise ValueError('Use equals or a list-valued in condition')
        if self.participants and set(self.participants)-{'from_stage','field'}:
            raise ValueError('Participant selector only accepts from_stage/field')


def stage_dependencies(stages, *, legacy_sequential=False):
    names = [s.get("id", f"stage_{i}") for i, s in enumerate(stages)]
    if len(set(names)) != len(names) or any(not isinstance(n, str) or not n or "." in n for n in names):
        raise ValueError("Stage IDs must be unique nonempty strings without dots")
    deps = {}
    for i, (name, stage) in enumerate(zip(names, stages)):
        declared = stage.get("depends_on", [])
        if not isinstance(declared, (list, tuple)) or any(not isinstance(d, str) for d in declared):
            raise ValueError("depends_on must be a list of stage IDs")
        parents = set(declared)
        for key in ("condition","participants"):
            if stage.get(key): parents.add(stage[key]["from_stage"])
        if legacy_sequential and i:
            parents.add(names[i - 1])
        for port, refs in stage.get("inputs", {}).items():
            if port not in {"context", "candidate"}:
                raise ValueError(f"Unsupported input port: {port}")
            refs = refs if isinstance(refs, list) else [refs]
            if port == "candidate" and len(refs) != 1:
                raise ValueError("candidate requires exactly one artifact")
            for ref in refs:
                ref=input_source(ref)
                if not isinstance(ref, str) or ref.count(".") != 1 or ref.split(".")[1] != "result":
                    raise ValueError("Artifact references must be stage_id.result")
                parents.add(ref.split(".")[0])
        if name in parents or parents - set(names):
            raise ValueError(f"Invalid prerequisites for {name}: {sorted(parents)}")
        deps[name] = parents
    done = set()
    while len(done) < len(names):
        ready = {n for n in names if n not in done and deps[n] <= done}
        if not ready:
            raise ValueError("Workflow dependency cycle")
        done.update(ready)
    return deps


@dataclass(frozen=True)
class WorkflowSpec:
    stages: tuple[StageSpec, ...]

    def __post_init__(self):
        if not self.stages:
            raise ValueError("Workflow requires stages")
        stage_dependencies([asdict(s) for s in self.stages])


@dataclass(frozen=True)
class StructureSpec:
    motifs: dict[str, MotifSpec]
    workflow: WorkflowSpec

    def __post_init__(self):
        for stage in self.workflow.stages:
            if stage.atomic is None and stage.motif not in self.motifs:
                raise ValueError(f"Unknown motif reference: {stage.motif}")
            if stage.participants and (stage.atomic or self.motifs[stage.motif].family == 'evaluate_refine'):
                raise ValueError('Participant selection requires a worker/peer motif')


@dataclass(frozen=True)
class RoleTask:
    instructions: str = "Address the task using the supplied inputs."
    tools: tuple[str, ...] = ()
    output_format: str = "text"
    io_schema: dict[str, Any] = field(default_factory=dict)
    input_schema: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        from .motifs.contracts import RoleBinding
        if not isinstance(self.instructions, str):
            raise ValueError("Role instructions must be text")
        RoleBinding(self.instructions, self.tools, self.output_format, io_schema=self.io_schema, input_schema=self.input_schema)


@dataclass(frozen=True)
class TaskBinding:
    task_input: str
    roles: dict[str, RoleTask] = field(default_factory=dict)
    criteria: str = "Satisfy the task and its constraints."
    routing_semantics: str = "Select the worker best suited to the task."
    evaluation_semantics: str = "Assess the supplied candidate against the criteria."
    tool_provider: str = "synthetic"
    tool_mode: str = "synthetic"
    stage_bindings: dict[str, dict[str, Any]] = field(default_factory=dict)
    delivery_instruction: str = 'Summarize the supplied information accurately and concisely.'

    def __post_init__(self):
        for binding in self.stage_bindings.values():
            if set(binding)-{"task_input","roles","criteria","routing_semantics","evaluation_semantics","parameters","delivery_instruction"}:
                raise ValueError("Task stage binding cannot define topology/deployment")
            for role, value in binding.get("roles",{}).items():
                if role not in ROLES: raise ValueError("Unknown role")
                RoleTask(**value)
        if set(self.roles) - ROLES:
            raise ValueError("Task roles must use Coordinator/Worker/Reducer/Reviewer")
        if not isinstance(self.task_input, str):
            raise ValueError("task_input must be text")


@dataclass(frozen=True)
class DeploymentSpec:
    model: str = "mock-mas-model"
    backend: str = "mock"
    endpoint: str = "http://127.0.0.1:8000/v1"
    generation: dict[str, Any] = field(default_factory=lambda: {"max_tokens": 256, "temperature": 0.2})
    hardware: dict[str, Any] = field(default_factory=dict)
    concurrency: int = 32
    telemetry: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        positive(self.concurrency, "concurrency")
        if set(self.telemetry)-{"adapter","device_id","metrics_url","metadata","npu_id","profiler_counters_path"}:
            raise ValueError("Unknown telemetry configuration")
        if self.backend not in {"mock", "openai_compatible"}:
            raise ValueError("Unsupported backend")
        if set(self.generation) - {"max_tokens", "temperature", "top_p", "seed", "stop", "frequency_penalty", "presence_penalty", "chat_template_kwargs"}:
            raise ValueError("Unsupported generation field (payload/topology overrides forbidden)")
        if "chat_template_kwargs" in self.generation and not isinstance(self.generation["chat_template_kwargs"], dict):
            raise ValueError("chat_template_kwargs must be an object")
        positive(self.generation.get("max_tokens", 256), "max_tokens")


@dataclass(frozen=True)
class ExperimentConfig:
    structure: StructureSpec
    task: TaskBinding
    deployment: DeploymentSpec
    load: int = 1
    arrival_interval_sec: float = 0.0

    def __post_init__(self):
        positive(self.load, "load")
        if set(self.task.stage_bindings) - {s.id for s in self.structure.workflow.stages}:
            raise ValueError('Task binding references an unknown stage')
        if self.arrival_interval_sec < 0:
            raise ValueError("arrival_interval_sec must be nonnegative")

    def compile(self):
        from .motifs.task_defaults import FAMILY_SLOTS
        stages = []
        for stage in self.structure.workflow.stages:
            binding = self.task.stage_bindings.get(stage.id,{})
            roles = {k:asdict(v) for k,v in self.task.roles.items()}
            roles.update(binding.get("roles",{}))
            if stage.atomic:
                values={"family":"atomic", "atomic":asdict(stage.atomic), "roles":{stage.atomic.role:roles.get(stage.atomic.role,asdict(RoleTask()))}}
            else:
                motif = self.structure.motifs[stage.motif]
                values = {k: v for k, v in asdict(motif).items() if v is not None and k not in {"coordination","delivery"}}
                values["motif_delivery"] = {k:asdict(v) for k,v in motif.delivery.items()}
                values["roles"] = {slot.name:roles[ROLE_ALIASES[slot.name]] for slot in FAMILY_SLOTS[motif.family] if ROLE_ALIASES[slot.name] in roles}
            compiled_inputs={}
            for port,raw in stage.inputs.items():
                refs=raw if isinstance(raw,list) else [raw]
                compiled=[]
                for ref in refs:
                    if isinstance(ref,dict):
                        edge=delivery_spec(ref.get("delivery",stage.delivery.get(port)))
                        compiled.append({"source":ref["source"],"delivery":asdict(edge),"delivery_scope":"edge"})
                    else:
                        compiled.append({"source":ref,"delivery":asdict(stage.delivery.get(port,DeliverySpec())),
                                         "delivery_scope":"port" if port in stage.delivery else "edge"})
                compiled_inputs[port]=compiled
            values.update(id=stage.id, depends_on=list(stage.depends_on), inputs=compiled_inputs,
                          task=binding.get("task_input",self.task.task_input), criteria=binding.get("criteria",self.task.criteria),
                          routing_semantics=binding.get("routing_semantics",self.task.routing_semantics),
                          evaluation_semantics=binding.get("evaluation_semantics",self.task.evaluation_semantics),
                          condition=stage.condition,participants=stage.participants,
                          delivery={k:asdict(v) for k,v in stage.delivery.items()},
                          allow_skipped=stage.allow_skipped,parameters=binding.get("parameters",{}),
                          delivery_instruction=binding.get("delivery_instruction",self.task.delivery_instruction))
            stages.append(values)
        return {"stages": stages}


def read_config(path):
    path = Path(path)
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment(path):
    path = Path(path)
    obj = read_config(path)
    def resolve(value):
        return read_config(path.parent / value) if isinstance(value, str) else value
    structure = dict(resolve(obj["structure"]))
    motifs = {k: MotifSpec(**resolve(v)) for k, v in structure.pop("motifs").items()}
    workflow = resolve(structure.pop("workflow"))
    if structure or set(workflow) != {"stages"}:
        raise ValueError("Unknown structure/workflow fields")
    task = dict(resolve(obj["task"]))
    task["roles"] = {k: RoleTask(**v) for k, v in task.get("roles", {}).items()}
    return ExperimentConfig(structure=StructureSpec(motifs, WorkflowSpec(tuple(StageSpec(**s) for s in workflow["stages"]))),
                            task=TaskBinding(**task), deployment=DeploymentSpec(**resolve(obj["deployment"])),
                            **{k: v for k, v in obj.items() if k not in {"structure", "task", "deployment"}})


def experiment_from_dict(obj):
    """Parse a fully embedded experiment, used after declarative overrides."""
    structure=dict(obj['structure'])
    motifs={k:MotifSpec(**v) for k,v in structure.pop('motifs').items()}
    workflow=structure.pop('workflow')
    if structure or set(workflow)!={'stages'}:raise ValueError('Unknown structure/workflow fields')
    task=dict(obj['task']);task['roles']={k:RoleTask(**v) for k,v in task.get('roles',{}).items()}
    return ExperimentConfig(StructureSpec(motifs,WorkflowSpec(tuple(StageSpec(**s) for s in workflow['stages']))),
        TaskBinding(**task),DeploymentSpec(**obj['deployment']),
        **{k:v for k,v in obj.items() if k not in {'structure','task','deployment'}})

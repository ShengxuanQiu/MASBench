"""Public benchmark factors. Compilation is the only structure/task join point."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROLES = {"Coordinator", "Worker", "Reducer", "Reviewer"}
ROLE_ALIASES = {"dispatcher": "Coordinator", "executor": "Worker", "worker": "Worker",
                "collector": "Reducer", "producer": "Worker", "evaluator": "Reviewer", "peer": "Worker"}
FAMILIES = {"DispatchExecute": "dispatch_execute", "ParallelAggregate": "parallel_aggregate",
            "EvaluateRefine": "evaluate_refine", "PeerDeliberation": "peer_deliberation"}


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

    def __post_init__(self):
        object.__setattr__(self, "family", FAMILIES.get(self.family, self.family))
        if self.family not in FAMILIES.values():
            raise ValueError("Unknown motif family")
        positive(self.width, "width")
        positive(self.rounds, "rounds", 0)
        positive(self.max_revisions, "max_revisions", 0)
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        shapes = {"dispatch_execute": {"star"}, "parallel_aggregate": {"fan_out_fan_in"},
                  "evaluate_refine": {"feedback_pair"}, "peer_deliberation": {"all_to_all", "ring", "pairwise", "random_k"}}
        if self.connectivity is not None and self.connectivity not in shapes[self.family]:
            raise ValueError("Unsupported motif connectivity")
        if self.family == "peer_deliberation" and self.width < 2:
            raise ValueError("PeerDeliberation requires width >= 2")
        regimes = {"dispatch_execute": "centralized", "parallel_aggregate": "independent",
                   "evaluate_refine": "centralized", "peer_deliberation": "decentralized"}
        if self.coordination not in {None, regimes[self.family]}:
            raise ValueError("Unsupported motif coordination")
        if self.dispatch not in {"plan", "route"} or self.aggregation not in {"concat_summary", "judge", "vote"}:
            raise ValueError("Unsupported dispatch/aggregation structure")


@dataclass(frozen=True)
class StageSpec:
    id: str
    motif: str
    depends_on: tuple[str, ...] = ()
    inputs: dict[str, str | list[str]] = field(default_factory=dict)


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
        if legacy_sequential and i:
            parents.add(names[i - 1])
        for port, refs in stage.get("inputs", {}).items():
            if port not in {"context", "candidate"}:
                raise ValueError(f"Unsupported input port: {port}")
            refs = refs if isinstance(refs, list) else [refs]
            if port == "candidate" and len(refs) != 1:
                raise ValueError("candidate requires exactly one artifact")
            for ref in refs:
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
            if stage.motif not in self.motifs:
                raise ValueError(f"Unknown motif reference: {stage.motif}")


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

    def __post_init__(self):
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

    def __post_init__(self):
        positive(self.concurrency, "concurrency")
        if self.backend not in {"mock", "openai_compatible"}:
            raise ValueError("Unsupported backend")
        if set(self.generation) - {"max_tokens", "temperature", "top_p", "seed", "stop", "frequency_penalty", "presence_penalty"}:
            raise ValueError("Unsupported generation field (payload/topology overrides forbidden)")
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
        if self.arrival_interval_sec < 0:
            raise ValueError("arrival_interval_sec must be nonnegative")

    def compile(self):
        from .motifs.task_defaults import FAMILY_SLOTS
        stages = []
        for stage in self.structure.workflow.stages:
            motif = self.structure.motifs[stage.motif]
            values = {k: v for k, v in asdict(motif).items() if v is not None and k != "coordination"}
            values.update(id=stage.id, depends_on=list(stage.depends_on), inputs=stage.inputs,
                          task=self.task.task_input, criteria=self.task.criteria,
                          routing_semantics=self.task.routing_semantics,
                          evaluation_semantics=self.task.evaluation_semantics)
            values["roles"] = {slot.name: asdict(self.task.roles[ROLE_ALIASES[slot.name]])
                               for slot in FAMILY_SLOTS[motif.family]
                               if ROLE_ALIASES[slot.name] in self.task.roles}
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

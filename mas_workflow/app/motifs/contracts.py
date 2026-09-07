"""Template slots, task bindings and run-local identities are separate objects."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class RoleSlot:
    name: str
    instructions: str


@dataclass(frozen=True)
class RoleBinding:
    instructions: str
    tools: tuple[str, ...] = ()
    output_format: str = "text"
    model: str = ""
    backend_base_url: str = ""

    def __post_init__(self):
        if self.output_format not in {"text", "json"}:
            raise ValueError("output_format must be text or json")
        if set(self.tools) - {"search"}:
            raise ValueError("Only the explicit search tool is currently supported by family bindings")

    def validate(self, value: str) -> None:
        if not value.strip():
            raise ValueError("Role returned an empty result")
        if self.output_format == "json":
            json.loads(value)


@dataclass
class AgentInstance:
    slot: RoleSlot
    binding: RoleBinding
    motif_instance_id: str
    index: int = 0
    instance_id: str = field(default_factory=lambda: "agent_" + uuid4().hex)
    # A slot/binding carries no conversation. State belongs to this instance.
    history: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Artifact:
    content: str
    producer_node: str
    kind: str = "result"
    artifact_id: str = field(default_factory=lambda: "artifact_" + uuid4().hex)


@dataclass
class MotifResult:
    family: str
    motif_instance_id: str
    status: str
    outputs: dict[str, Artifact]
    iterations: int = 0
    error: str = ""


def role_bindings(slots: tuple[RoleSlot, ...], values: dict[str, Any]) -> dict[str, RoleBinding]:
    unknown = set(values) - {slot.name for slot in slots}
    if unknown:
        raise ValueError(f"Unknown role slots: {sorted(unknown)}")
    return {
        slot.name: RoleBinding(**{"instructions": slot.instructions, **values.get(slot.name, {})})
        for slot in slots
    }

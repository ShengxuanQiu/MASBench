"""Resolved benchmark workload manifest and separate SUT description."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any

from .model import sha256_json

SCENARIO_VERSION = "masbench.scenario/1.0.0"


@dataclass
class ScenarioManifest:
    scenario_id: str
    benchmark_version: str
    trace_set_ids: list[str]
    trace_hashes: dict[str, str]
    workload_version: str
    model_input_contract: dict[str, Any]
    replay: dict[str, Any]
    root_arrival: dict[str, Any]
    measurement: dict[str, Any]
    validity: dict[str, Any]
    scenario_version: str = SCENARIO_VERSION
    resolved_cache_scope_salts: dict[str, str] = field(default_factory=dict)
    resolved_request_contracts: dict[str, dict[str, Any]] = field(default_factory=dict)
    scenario_hash: str = ""

    def resolve(self, reuse_scope_ids: list[str], trace: Any | None = None, encode_request: Any | None = None) -> "ScenarioManifest":
        seed = int(self.root_arrival.get("random_seed", 0))
        self.resolved_cache_scope_salts = {
            scope: sha256_json({"seed": seed, "reuse_scope_id": scope})[:24]
            for scope in sorted(set(reuse_scope_ids)) if scope
        }
        if trace is not None:
            if encode_request is None:
                raise ValueError("Resolving exact target inputs requires the declared tokenizer")
            self.resolved_request_contracts = {}
            for operation in trace.operations:
                if not operation.llm:
                    continue
                request = self.materialize_request(operation)
                token_ids = list(encode_request(request))
                self.resolved_request_contracts[operation.operation_id] = {
                    "request_hash": sha256_json(request), "input_token_ids": token_ids,
                    "input_token_count": len(token_ids), "reuse_scope_id": operation.llm.get("reuse_scope_id")}
        self.scenario_hash = self.compute_hash()
        return self

    def materialize_request(self, operation: Any) -> dict[str, Any]:
        request = copy.deepcopy(operation.llm["canonical_request"])
        scope = operation.llm.get("reuse_scope_id")
        if scope:
            salt = self.resolved_cache_scope_salts[scope]
            marker = f"[MASBENCH_CACHE_SCOPE:{salt}]"
            messages = request.setdefault("messages", [])
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = marker + "\n" + str(messages[0].get("content", ""))
            else:
                messages.insert(0, {"role": "system", "content": marker})
        return request

    def identity_projection(self) -> dict[str, Any]:
        value = asdict(self)
        value["scenario_hash"] = ""
        return value

    def compute_hash(self) -> str:
        return sha256_json(self.identity_projection())


@dataclass
class SystemConfig:
    backend_name: str
    backend_version: str
    hardware_model: str
    accelerator_count: int
    model_identity: str = ""
    tokenizer_hash: str = ""
    chat_template_hash: str = ""
    batching_policy: dict[str, Any] = field(default_factory=dict)
    cache_implementation: dict[str, Any] = field(default_factory=dict)
    scheduling_policy: dict[str, Any] = field(default_factory=dict)
    placement: dict[str, Any] = field(default_factory=dict)
    parallelism: dict[str, Any] = field(default_factory=dict)
    capabilities: dict[str, bool] = field(default_factory=dict)

    def system_hash(self) -> str:
        return sha256_json(asdict(self))

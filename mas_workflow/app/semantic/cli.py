"""CLI for the MASBench Semantic Trace benchmark pipeline."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .canonicalize import ByteTokenizer, TransformersTokenizer, canonicalize_native_trace
from .coverage import cluster_coverage, extract_feature_vector, read_feature_rows, workflow_coverage
from .golden import TEMPLATES, build_golden, golden_scenario, golden_system
from .lowering import lower_with_official_converter
from .metrics import sustainable_capacity, task_metrics
from .replay import MockExactBackend, ReplayExecutor, VLLMOpenAIBackend
from .scenario import ScenarioManifest, SystemConfig
from .store import TraceBundle
from .validators import LoweringValidator, SemanticValidator


def load_scenario(path: str) -> ScenarioManifest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    value.pop("scenario_hash", None)
    return ScenarioManifest(**value).resolve(value.get("resolved_cache_scope_salts", {}).keys())


def load_system(path: str) -> SystemConfig:
    return SystemConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def write(path: str | Path, value) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect", help="Canonicalize a native/source JSONL trace")
    p.add_argument("--source", required=True); p.add_argument("--output", required=True)
    p.add_argument("--workload-id", required=True); p.add_argument("--workload-version", default="1")
    p.add_argument("--tokenizer"); p.add_argument("--source-framework-version", default="unknown")
    p = sub.add_parser("validate"); p.add_argument("--trace", required=True); p.add_argument("--output")
    p = sub.add_parser("resolve-scenario"); p.add_argument("--template", required=True); p.add_argument("--trace", required=True)
    p.add_argument("--tokenizer", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("golden"); p.add_argument("--output", required=True); p.add_argument("--template", choices=TEMPLATES + ("all",), default="all")
    p = sub.add_parser("replay"); p.add_argument("--trace", required=True); p.add_argument("--scenario", required=True)
    p.add_argument("--system", required=True); p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=("synthetic", "vllm-openai"), required=True); p.add_argument("--base-url"); p.add_argument("--api-key", default="EMPTY")
    p = sub.add_parser("measure"); p.add_argument("--run", required=True); p.add_argument("--output", required=True)
    p.add_argument("--slo-sec", type=float, required=True); p.add_argument("--accelerators", type=int, required=True); p.add_argument("--power-watts", type=float)
    p = sub.add_parser("capacity"); p.add_argument("--cells", required=True); p.add_argument("--attainment-target", required=True, type=float); p.add_argument("--output", required=True)
    p = sub.add_parser("coverage-workflow"); p.add_argument("--corpus", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("features"); p.add_argument("--trace", required=True); p.add_argument("--workload-id"); p.add_argument("--output", required=True)
    p = sub.add_parser("coverage-cluster"); p.add_argument("--real", required=True); p.add_argument("--benchmark", required=True)
    p.add_argument("--clusters", required=True, type=int); p.add_argument("--seed", default=42, type=int); p.add_argument("--output", required=True)
    p = sub.add_parser("lower-chakra"); p.add_argument("--trace", required=True); p.add_argument("--instrumented-execution", required=True)
    p.add_argument("--chakra-output", required=True); p.add_argument("--manifest-output", required=True)
    p.add_argument("--converter-version", required=True); p.add_argument("--model-cost-version", required=True)
    p.add_argument("converter", nargs=argparse.REMAINDER, help="Official converter command using {input} and {output}")
    p = sub.add_parser("e2e"); p.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.command == "collect":
        tokenizer = TransformersTokenizer(args.tokenizer) if args.tokenizer else ByteTokenizer()
        if not args.tokenizer:
            raise SystemExit("collect requires --tokenizer for real traces; ByteTokenizer is reserved for golden traces")
        trace = canonicalize_native_trace(args.source, args.output, tokenizer=tokenizer, workload_id=args.workload_id,
            workload_version=args.workload_version, source_framework_version=args.source_framework_version)
        print(json.dumps({"trace": args.output, "workload_hash": trace.metadata.workload_hash}))
    elif args.command == "validate":
        trace = TraceBundle(args.trace).read(); result = SemanticValidator().validate(trace).to_dict()
        if args.output: write(args.output, result)
        print(json.dumps(result)); raise SystemExit(0 if result["run_valid"] else 2)
    elif args.command == "resolve-scenario":
        value = json.loads(Path(args.template).read_text(encoding="utf-8")); value.pop("scenario_hash", None)
        value.pop("resolved_cache_scope_salts", None); value.pop("resolved_request_contracts", None)
        trace = TraceBundle(args.trace).read(); tokenizer = TransformersTokenizer(args.tokenizer)
        value["trace_set_ids"] = [trace.metadata.trace_id]
        value["trace_hashes"] = {trace.metadata.trace_id: trace.metadata.workload_hash}
        value["workload_version"] = trace.metadata.workload_version
        value["root_arrival"]["task_mix"] = {trace.metadata.trace_id: 1.0}
        value["model_input_contract"]["tokenizer_identity"] = tokenizer.identity
        value["model_input_contract"]["tokenizer_hash"] = tokenizer.identity["hash"]
        value["model_input_contract"]["chat_template_identity"] = tokenizer.chat_template
        value["model_input_contract"]["chat_template_hash"] = tokenizer.chat_template["hash"]
        scopes = [x.reuse_scope_id for x in trace.sessions if x.reuse_scope_id]
        scenario = ScenarioManifest(**value).resolve(scopes, trace, tokenizer.encode_request)
        write(args.output, asdict(scenario)); print(json.dumps({"scenario_hash": scenario.scenario_hash, "resolved_requests": len(scenario.resolved_request_contracts)}))
    elif args.command == "golden":
        templates = TEMPLATES if args.template == "all" else (args.template,)
        for name in templates: build_golden(name, Path(args.output) / name)
        print(json.dumps({"golden_traces": list(templates)}))
    elif args.command == "replay":
        bundle = TraceBundle(args.trace); trace = bundle.read(); scenario = load_scenario(args.scenario); system = load_system(args.system)
        backend = MockExactBackend() if args.backend == "synthetic" else VLLMOpenAIBackend(args.base_url, args.api_key)
        result = ReplayExecutor(trace, scenario, system, backend, bundle.artifacts).run(); write(args.output, result)
        print(json.dumps({"run_valid": result["run_valid"], "invalid_reasons": result["invalid_reasons"]})); raise SystemExit(0 if result["run_valid"] else 2)
    elif args.command == "measure":
        result = task_metrics(json.loads(Path(args.run).read_text()), slo_sec=args.slo_sec, accelerator_count=args.accelerators, physical_power_watts=args.power_watts)
        write(args.output, result); print(json.dumps(result))
    elif args.command == "capacity":
        result = sustainable_capacity(json.loads(Path(args.cells).read_text()), attainment_target=args.attainment_target)
        write(args.output, result); print(json.dumps(result))
    elif args.command == "coverage-workflow":
        result = workflow_coverage(json.loads(Path(args.corpus).read_text())); write(args.output, result); print(json.dumps(result))
    elif args.command == "features":
        result = extract_feature_vector(TraceBundle(args.trace).read(), args.workload_id); write(args.output, result); print(json.dumps(result))
    elif args.command == "coverage-cluster":
        result = cluster_coverage(read_feature_rows(args.real), read_feature_rows(args.benchmark), clusters=args.clusters, seed=args.seed)
        write(args.output, result); print(json.dumps(result))
    elif args.command == "lower-chakra":
        trace = TraceBundle(args.trace).read(); result = lower_with_official_converter(trace, args.instrumented_execution, args.chakra_output,
            converter_command=args.converter, converter_version=args.converter_version, model_or_cost_version=args.model_cost_version)
        result["validation"] = LoweringValidator().validate(trace, result).to_dict(); write(args.manifest_output, result); print(json.dumps(result["validation"]))
    elif args.command == "e2e":
        output = Path(args.output); reports = {}
        for name in TEMPLATES:
            trace = build_golden(name, output / "traces" / name); scenario = golden_scenario(trace)
            write(output / "scenarios" / f"{name}.json", asdict(scenario))
            system = golden_system()
            result = ReplayExecutor(trace, scenario, system, MockExactBackend(), TraceBundle(output / "traces" / name).artifacts).run()
            reports[name] = result; write(output / "runs" / f"{name}.json", result)
            write(output / "metrics" / f"{name}.json", task_metrics(result, slo_sec=10.0, accelerator_count=1))
        summary = {name: {"run_valid": value["run_valid"], "invalid_reasons": value["invalid_reasons"]} for name, value in reports.items()}
        write(output / "summary.json", summary); print(json.dumps(summary)); raise SystemExit(0 if all(x["run_valid"] for x in reports.values()) else 2)


if __name__ == "__main__":
    main()

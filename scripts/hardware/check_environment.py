#!/usr/bin/env python3
"""Inventory a prepared host without running a workload or allocating device tensors."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def command(args):
    if not shutil.which(args[0]):
        return {"available": False}
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {"available": True, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except subprocess.TimeoutExpired:
        return {"available": True, "timeout": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--serving-import", action="store_true", help="Import torch/vLLM; no model load or device tensors")
    args = parser.parse_args()
    packages = {}
    for name in ["torch", "torch-npu", "vllm", "vllm-ascend", "transformers", "flashinfer-python", "pytest", "jsonschema", "PyYAML", "numpy", "nvidia-ml-py"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {"hostname": platform.node(), "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(), "packages": packages}
    report["gpu_inventory"] = command(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap", "--format=csv"])
    report["gpu_occupancy_snapshot"] = command(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv"])
    report["npu_inventory"] = command(["npu-smi", "info"])
    report["profilers"] = {name: command([name, "--version"]) for name in ["nsys", "ncu", "msprof"]}
    report["git_revision"] = command(["git", "rev-parse", "HEAD"])
    report["perf_event_paranoid"] = Path("/proc/sys/kernel/perf_event_paranoid").read_text().strip() if Path("/proc/sys/kernel/perf_event_paranoid").exists() else None
    if args.serving_import:
        import torch
        import vllm
        report["serving_import"] = {"torch": torch.__version__, "vllm": vllm.__version__, "torch_cuda": torch.version.cuda, "compiled_cuda_architectures": torch.cuda.get_arch_list()}
    if args.model:
        model = args.model.resolve()
        missing = []
        index = model / "model.safetensors.index.json"
        if index.is_file():
            manifest = json.loads(index.read_text())
            shards = sorted(set(manifest["weight_map"].values()))
            missing = [name for name in shards if not (model / name).is_file()]
        else:
            shards = [p.name for p in model.glob("*.safetensors")]
            if not shards:
                missing = ["model.safetensors or model.safetensors.index.json"]
        for required in ["config.json", "tokenizer_config.json"]:
            if not (model / required).is_file():
                missing.append(required)
        report["model_files"] = {"path": str(model), "shards": shards, "missing": missing, "weights_read_or_loaded": False}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    if args.model and report["model_files"]["missing"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

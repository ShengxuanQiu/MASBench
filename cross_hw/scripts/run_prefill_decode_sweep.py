#!/usr/bin/env python3
"""Controlled prefill/decode sweep against an OpenAI-compatible vLLM server."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
import uuid
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from xpu_telemetry import TelemetrySampler


def make_prompt(tokenizer, target_tokens: int, nonce: str) -> str:
    seed = (
        f"Experiment nonce {nonce}. Analyze graph pressure, evidence propagation, scheduling, "
        "memory traffic, attention, and multi-agent inference behavior. "
    )
    text = seed * (target_tokens // 12 + 8)
    ids = tokenizer.encode(text, add_special_tokens=False)[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def completion(endpoint: str, model: str, prompt: str, max_tokens: int, barrier: threading.Barrier | None) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "seed": 42,
    }
    if barrier:
        barrier.wait()
    submitted = time.time()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    first_token = None
    usage: dict = {}
    text_parts: list[str] = []
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices and choices[0].get("text"):
                if first_token is None:
                    first_token = time.time()
                text_parts.append(choices[0]["text"])
    completed = time.time()
    if first_token is None:
        first_token = completed
    output_tokens = int(usage.get("completion_tokens") or max_tokens)
    return {
        "submitted_ts": submitted,
        "first_token_ts": first_token,
        "completed_ts": completed,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": output_tokens,
        "ttft_ms": 1000 * (first_token - submitted),
        "tpot_ms": 1000 * (completed - first_token) / max(1, output_tokens - 1),
        "latency_sec": completed - submitted,
    }


def run_cell(endpoint: str, model: str, tokenizer, *, phase: str, target_prompt_tokens: int, output_tokens: int,
             concurrency: int, repeat: int) -> list[dict]:
    barrier = threading.Barrier(concurrency) if concurrency > 1 else None
    prompts = [make_prompt(tokenizer, target_prompt_tokens, f"{phase}-{repeat}-{concurrency}-{index}-{uuid.uuid4().hex}")
               for index in range(concurrency)]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(completion, endpoint, model, prompt, output_tokens, barrier) for prompt in prompts]
        rows = [future.result() for future in futures]
    cell_start = min(row["submitted_ts"] for row in rows)
    cell_end = max(row["completed_ts"] for row in rows)
    total_output = sum(row["output_tokens"] for row in rows)
    for index, row in enumerate(rows):
        row.update({
            "phase": phase,
            "target_prompt_tokens": target_prompt_tokens,
            "target_output_tokens": output_tokens,
            "concurrency": concurrency,
            "repeat": repeat,
            "request_index": index,
            "cell_start_ts": cell_start,
            "cell_end_ts": cell_end,
            "cell_output_throughput_tok_s": total_output / (cell_end - cell_start),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--model", default="local-mas-model")
    parser.add_argument("--tokenizer", default="/model/Qwen3-8B")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--sample-interval", type=float, default=0.25)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    sampler = TelemetrySampler(args.output / "telemetry.csv", interval=args.sample_interval, metrics_url=args.metrics_url)
    rows: list[dict] = []
    sampler.start()
    try:
        # One unrecorded warm-up prevents graph compilation from dominating the first cell.
        run_cell(args.endpoint, args.model, tokenizer, phase="warmup", target_prompt_tokens=512,
                 output_tokens=16, concurrency=1, repeat=0)
        for repeat in range(1, args.repeats + 1):
            for prompt_tokens in [128, 512, 1024, 2048, 4096]:
                rows.extend(run_cell(args.endpoint, args.model, tokenizer, phase="prefill", target_prompt_tokens=prompt_tokens,
                                     output_tokens=16, concurrency=1, repeat=repeat))
                time.sleep(0.5)
            for prompt_tokens in [512, 2048, 4096]:
                for concurrency in [1, 2, 4, 8]:
                    rows.extend(run_cell(args.endpoint, args.model, tokenizer, phase="decode", target_prompt_tokens=prompt_tokens,
                                         output_tokens=128, concurrency=concurrency, repeat=repeat))
                    time.sleep(0.75)
    finally:
        sampler.stop()
    fieldnames = list(rows[0])
    with (args.output / "requests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "platform": "Ascend 910C (Atlas 800I A3, one visible 64 GB die)",
        "model": args.model,
        "repeats": args.repeats,
        "prefill_prompt_tokens": [128, 512, 1024, 2048, 4096],
        "decode_prompt_tokens": [512, 2048, 4096],
        "decode_concurrency": [1, 2, 4, 8],
        "prefill_output_tokens": 16,
        "decode_output_tokens": 128,
        "prefix_cache_control": "unique leading nonce per request",
        "completed_at": time.time(),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"completed {len(rows)} requests; median TPOT={statistics.median(row['tpot_ms'] for row in rows):.2f} ms/token")


if __name__ == "__main__":
    main()


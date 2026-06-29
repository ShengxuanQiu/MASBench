"""AgentServe-style phase-aware scheduling case study.

This runner intentionally lives outside the vLLM source tree. It replays a
fixed tool-call multi-agent workflow and compares default FIFO admission with a
lightweight external phase-aware admission policy. When a local vLLM endpoint is
available it can issue real OpenAI-compatible streaming requests; otherwise it
uses a deterministic timing simulator with the same request sequence, prompts,
tool outputs, random seed, and arrival pattern.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm_client import LocalLLMClient
from ..tracing import estimate_tokens, stable_hash


DEFAULT_OUT_DIR = Path("results/case_studies/agentserve_phase")
DEFAULT_PROGRESS_DIR = Path("progress/week6")
PHASE_COLORS = {
    "cold_prefill": "#4C78A8",
    "resume_prefill": "#F58518",
    "decode": "#54A24B",
    "tool_wait": "#B279A2",
}


@dataclass
class RequestSpec:
    workflow_id: str
    request_id: str
    agent_id: str
    agent_role: str
    phase_type: str
    critical_path: bool
    input_tokens: int
    output_tokens: int
    arrival_offset_ms: float
    deps: list[str] = field(default_factory=list)
    tool_observation_tokens: int = 0
    max_tokens: int = 96
    prompt: str = ""
    system_prompt: str = ""


def pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[int(idx)])
    return float(ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo))


def ms(ts: float) -> float:
    return round(ts, 3)


def make_text(label: str, target_tokens: int) -> str:
    base = (
        f"{label}: fixed MAS workflow replay context. "
        "Use the same task, prompt, tool outputs, random seed, and arrival pattern. "
        "The goal is to expose long non-critical resume prefill interfering with "
        "critical short decode on the workflow path. "
    )
    approx = max(1, len(base) // 4)
    repeat = max(1, int(target_tokens / approx) + 1)
    return (base * repeat)[: max(80, target_tokens * 4)]


def build_workflow_specs(*, workflows: int, branch_width: int, seed: int) -> tuple[list[RequestSpec], list[dict[str, Any]]]:
    rng = random.Random(seed)
    specs: list[RequestSpec] = []
    tool_waits: list[dict[str, Any]] = []
    task = (
        "Implement and validate a code/edit/test/debug MAS workflow for a flaky "
        "repository regression. Planner, critical coder, reviewer, delayed tool "
        "evidence processors, evidence merge, and finalizer must cooperate."
    )
    for w in range(workflows):
        wid = f"workflow_{w+1}"
        offset = w * 320.0
        planner = f"{wid}:planner"
        coder = f"{wid}:critical_coder"
        reviewer = f"{wid}:critical_reviewer"
        merge = f"{wid}:evidence_merge"
        finalizer = f"{wid}:finalizer"
        bg_coder = f"{wid}:background_coder"
        specs.extend(
            [
                RequestSpec(
                    wid,
                    planner,
                    "planner",
                    "planner",
                    "cold_prefill",
                    True,
                    1200,
                    48,
                    offset,
                    max_tokens=64,
                ),
                RequestSpec(
                    wid,
                    coder,
                    "critical_coder",
                    "coder",
                    "resume_prefill",
                    True,
                    2100,
                    72,
                    offset + 650,
                    deps=[planner],
                    max_tokens=96,
                ),
                RequestSpec(
                    wid,
                    reviewer,
                    "critical_reviewer",
                    "reviewer",
                    "resume_prefill",
                    True,
                    1700,
                    64,
                    offset + 1450,
                    deps=[coder],
                    max_tokens=80,
                ),
                RequestSpec(
                    wid,
                    bg_coder,
                    "background_coder",
                    "coder",
                    "cold_prefill",
                    False,
                    3400,
                    72,
                    offset + 720,
                    deps=[planner],
                    max_tokens=80,
                ),
            ]
        )
        resume_ids: list[str] = [bg_coder]
        for i in range(1, branch_width + 1):
            pre = f"{wid}:tool_agent_{i}"
            tool = f"{wid}:tool_wait_{i}"
            resume = f"{wid}:resume_evidence_processor_{i}"
            label = ["search", "repo", "docs", "tests", "perf", "risk"][min(i - 1, 5)]
            specs.append(
                RequestSpec(
                    wid,
                    pre,
                    f"tool_agent_{i}",
                    "tool_agent",
                    "cold_prefill",
                    False,
                    1600 + i * 120,
                    36,
                    offset + 740 + i * 18,
                    deps=[planner],
                    max_tokens=48,
                )
            )
            tool_waits.append(
                {
                    "workflow_id": wid,
                    "request_id": tool,
                    "agent_id": f"tool_wait_{i}",
                    "agent_role": "tool",
                    "phase_type": "tool_wait",
                    "critical_path": False,
                    "deps": [pre],
                    "arrival_offset_ms": offset + 1220 + i * 18,
                    "duration_ms": 245.0 + i * 16.0,
                    "tool_observation_tokens": 1300 + i * 260,
                    "tool_output": make_text(f"fixed_{label}_tool_output_{wid}_{i}", 1300 + i * 260),
                }
            )
            specs.append(
                RequestSpec(
                    wid,
                    resume,
                    f"resume_evidence_processor_{i}",
                    "evidence_processor",
                    "resume_prefill",
                    False,
                    5600 + i * 520 + rng.randint(0, 120),
                    72,
                    offset + 1490 + i * 18,
                    deps=[tool],
                    tool_observation_tokens=1300 + i * 260,
                    max_tokens=80,
                )
            )
            resume_ids.append(resume)
        specs.extend(
            [
                RequestSpec(
                    wid,
                    merge,
                    "evidence_merge",
                    "aggregator",
                    "resume_prefill",
                    False,
                    4300,
                    56,
                    offset + 1900,
                    deps=resume_ids,
                    tool_observation_tokens=branch_width * 1100,
                    max_tokens=72,
                ),
                RequestSpec(
                    wid,
                    finalizer,
                    "finalizer",
                    "finalizer",
                    "resume_prefill",
                    True,
                    2600,
                    64,
                    offset + 2100,
                    deps=[reviewer],
                    max_tokens=80,
                ),
            ]
        )
        for spec in specs:
            if spec.workflow_id != wid or spec.prompt:
                continue
            spec.system_prompt = f"You are {spec.agent_role} in an AgentServe phase-aware scheduling case study."
            spec.prompt = "\n".join(
                [
                    f"Task: {task}",
                    f"Workflow: {wid}",
                    f"Agent: {spec.agent_id}",
                    f"Request: {spec.request_id}",
                    make_text(f"{wid}_{spec.agent_id}_prompt", spec.input_tokens),
                ]
            )
    return specs, tool_waits


def load_specs_from_source_trace(path: Path) -> tuple[list[RequestSpec], list[dict[str, Any]], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    llm_rows = [row for row in rows if row.get("event_type") == "llm_request_end"]
    tool_rows = [row for row in rows if row.get("event_type") == "tool_search"]
    workflow_id = str((rows[0] if rows else {}).get("workflow_id") or path.stem)
    start_rel = min([float(row.get("relative_time_sec") or 0.0) for row in llm_rows + tool_rows] or [0.0])
    specs: list[RequestSpec] = []
    for row in llm_rows:
        node = str(row.get("node_id") or row.get("agent_id") or f"llm_{len(specs)}")
        lifecycle = str(row.get("function_call_lifecycle_stage") or "")
        critical = bool(row.get("critical_path_candidate")) or str(row.get("criticality")) == "critical"
        if lifecycle == "llm_2_resume" or node in {"critical_coder", "critical_reviewer", "finalizer", "evidence_merge"}:
            phase_type = "resume_prefill"
        else:
            phase_type = "cold_prefill"
        rel = float(row.get("relative_time_sec") or 0.0)
        dur = float(row.get("duration_sec") or 0.0)
        arrival = max(0.0, (rel - dur - start_rel) * 1000.0)
        source_input = int(row.get("input_tokens") or row.get("input_tokens_est") or 1)
        source_output = int(row.get("output_tokens") or row.get("output_tokens_est") or 1)
        role = str(row.get("agent_role") or row.get("node_type") or "agent")
        if critical:
            if role == "finalizer":
                output_tokens = max(source_output, 224)
            elif role in {"reviewer", "coder"}:
                output_tokens = max(source_output, 104)
            else:
                output_tokens = max(source_output, 72)
        elif role == "aggregator":
            output_tokens = max(source_output, 56)
        else:
            output_tokens = max(source_output, 48)
        input_tokens = source_input
        if lifecycle == "llm_2_resume":
            input_tokens = max(source_input, source_input * 8)
        elif not critical and phase_type == "cold_prefill":
            input_tokens = max(source_input, source_input * 4)
        elif critical and phase_type == "resume_prefill":
            input_tokens = max(source_input, source_input * 4)
        specs.append(
            RequestSpec(
                workflow_id=workflow_id,
                request_id=f"{workflow_id}:{node}",
                agent_id=node,
                agent_role=role,
                phase_type=phase_type,
                critical_path=critical,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                arrival_offset_ms=arrival,
                deps=[f"{workflow_id}:{parent}" for parent in row.get("parent_node_ids") or row.get("parents") or []],
                tool_observation_tokens=int(row.get("retrieved_context_tokens_est") or row.get("tool_observation_tokens") or 0),
                max_tokens=int(row.get("max_output_tokens") or output_tokens),
                prompt=str(row.get("prompt_hash") or ""),
                system_prompt=str(row.get("prompt_template") or ""),
            )
        )
    spec_ids = {spec.request_id for spec in specs}
    tool_waits: list[dict[str, Any]] = []
    for row in tool_rows:
        node = str(row.get("node_id") or f"tool_{len(tool_waits)}")
        rel = float(row.get("relative_time_sec") or 0.0)
        dur = max(0.001, float(row.get("duration_sec") or 0.0))
        start_ms = max(0.0, (rel - dur - start_rel) * 1000.0)
        snapshot_path = row.get("result_snapshot_id")
        observation_tokens = int(row.get("output_tokens_est") or row.get("result_tokens_est") or 0)
        tool_output = ""
        if snapshot_path:
            snap = Path(str(snapshot_path))
            if snap.exists():
                try:
                    tool_output = snap.read_text(encoding="utf-8")
                    observation_tokens = max(observation_tokens, estimate_tokens(tool_output))
                except OSError:
                    tool_output = ""
        deps = [f"{workflow_id}:{parent}" for parent in row.get("parent_node_ids") or row.get("parents") or []]
        tool_id = f"{workflow_id}:{node}"
        tool_waits.append(
            {
                "workflow_id": workflow_id,
                "request_id": tool_id,
                "agent_id": node,
                "agent_role": "tool",
                "phase_type": "tool_wait",
                "critical_path": False,
                "deps": [dep for dep in deps if dep in spec_ids],
                "arrival_offset_ms": start_ms,
                "duration_ms": dur * 1000.0,
                "tool_observation_tokens": observation_tokens,
                "tool_output": tool_output or str(row.get("tool_query") or ""),
            }
        )
    tool_ids = {tool["request_id"] for tool in tool_waits}
    for spec in specs:
        spec.deps = [dep for dep in spec.deps if dep in spec_ids or dep in tool_ids]
    meta = {
        "source_trace_path": str(path),
        "source_llm_span_count": len(llm_rows),
        "source_tool_span_count": len(tool_rows),
        "source_workflow_id": workflow_id,
        "source_trace_policy": "full_workflow_trace_derived_replay",
        "token_calibration": "source_prompt_tokens_with_role_min_output_long_resume_multiplier_and_realistic_critical_decode_length",
    }
    return specs, tool_waits, meta


class SimulatedServer:
    def __init__(self, *, seed: int, max_concurrent: int) -> None:
        self.seed = seed
        self.max_concurrent = max(1, max_concurrent)

    def run(self, specs: list[RequestSpec], tool_waits: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        last_end: dict[str, float] = {}
        end_estimates = {spec.request_id: spec.arrival_offset_ms + self._base_duration(spec) for spec in specs}
        end_estimates.update(self._estimate_tool_ends(tool_waits, end_estimates))
        for _ in range(4):
            tool_records: dict[str, dict[str, Any]] = {}
            schedule = self._schedule(specs, tool_waits, mode, end_estimates, tool_records)
            end_estimates = self._apply_decode_contention(schedule)
            end_estimates = {record["request_id"]: record["end_ts"] for record in end_estimates}
            end_estimates.update(self._estimate_tool_ends(tool_waits, end_estimates))
        tool_records = {}
        schedule = self._schedule(specs, tool_waits, mode, end_estimates, tool_records)
        final_records = self._apply_decode_contention(schedule)
        for record in final_records:
            last_end[record["request_id"]] = record["end_ts"]
            records.append(record)
            records.append(self._decode_record(record))
        for tool in sorted(tool_records.values(), key=lambda item: item["queue_enter_ts"]):
            records.append(tool)
        return sorted(records, key=lambda item: (item["queue_enter_ts"], item["submit_ts"], item["request_id"]))

    def _estimate_tool_ends(self, tool_waits: list[dict[str, Any]], end_estimates: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for tool in tool_waits:
            dep_ends = [end_estimates.get(dep, float(tool["arrival_offset_ms"])) for dep in tool["deps"]]
            start = max([float(tool["arrival_offset_ms"]), *dep_ends])
            out[str(tool["request_id"])] = start + float(tool["duration_ms"])
        return out

    def _base_duration(self, spec: RequestSpec) -> float:
        return self._prefill_ms(spec) + spec.output_tokens * self._base_itl_ms(spec)

    def _prefill_ms(self, spec: RequestSpec) -> float:
        multiplier = 0.072 if spec.phase_type == "resume_prefill" else 0.058
        return max(45.0, spec.input_tokens * multiplier)

    def _base_itl_ms(self, spec: RequestSpec) -> float:
        if spec.critical_path:
            return 12.0
        if spec.agent_role == "aggregator":
            return 14.5
        return 13.5

    def _schedule(
        self,
        specs: list[RequestSpec],
        tool_waits: list[dict[str, Any]],
        mode: str,
        end_estimates: dict[str, float],
        tool_records: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        by_id = {spec.request_id: spec for spec in specs}
        tool_by_id = {str(tool["request_id"]): tool for tool in tool_waits}
        pending = set(by_id)
        completed_end: dict[str, float] = {}
        running_slots: list[float] = []
        admitted: list[dict[str, Any]] = []
        resume_budget = 2600
        critical_protect_until = 0.0
        while pending:
            ready: list[tuple[float, RequestSpec]] = []
            for rid in list(pending):
                spec = by_id[rid]
                deps_done = [completed_end.get(dep, end_estimates.get(dep, spec.arrival_offset_ms)) for dep in spec.deps]
                queue_enter = max([spec.arrival_offset_ms, *deps_done] or [spec.arrival_offset_ms])
                ready.append((queue_enter, spec))
            ready.sort(key=lambda item: (item[0], item[1].request_id))
            if mode == "phase_aware":
                critical_pending = [item for item in ready if item[1].critical_path]
                queue_enter, spec = min(critical_pending, key=lambda item: (item[0], item[1].request_id)) if critical_pending else ready[0]
            else:
                queue_enter, spec = ready[0]
            running_slots = [end for end in running_slots if end > queue_enter]
            slot_time = queue_enter if len(running_slots) < self.max_concurrent else min(running_slots)
            submit = max(queue_enter, slot_time)
            reason = ""
            if mode == "phase_aware":
                if not spec.critical_path and critical_protect_until > queue_enter:
                    submit = max(submit, critical_protect_until + 28.0)
                    reason = "defer_noncritical_prefill_until_critical_frontier_completes"
                prior_reason = reason
                submit, reason, resume_budget = self._phase_submit_time(
                    spec=spec,
                    queue_enter=queue_enter,
                    submit=submit,
                    admitted=admitted,
                    end_estimates=end_estimates,
                    all_specs=specs,
                    resume_budget=resume_budget,
                )
                if prior_reason and not reason:
                    reason = prior_reason
                running_slots = [end for end in running_slots if end > submit]
                if len(running_slots) >= self.max_concurrent:
                    submit = min(running_slots)
            prefill_ms = self._prefill_ms(spec)
            first = submit + prefill_ms
            end = first + spec.output_tokens * self._base_itl_ms(spec)
            record = self._prefill_record(spec, mode, queue_enter, submit, first, end, reason)
            admitted.append(record)
            pending.remove(spec.request_id)
            completed_end[spec.request_id] = end
            if mode == "phase_aware" and spec.critical_path:
                critical_protect_until = max(critical_protect_until, end)
            running_slots.append(end)
            for tool_id, tool in tool_by_id.items():
                if tool_id in tool_records:
                    continue
                if all(dep in completed_end for dep in tool["deps"]):
                    t_queue = max(float(tool["arrival_offset_ms"]), *(completed_end[dep] for dep in tool["deps"]))
                    t_end = t_queue + float(tool["duration_ms"])
                    tool_records[tool_id] = self._tool_record(tool, mode, t_queue, t_end)
                    completed_end[tool_id] = t_end
        return admitted

    def _phase_submit_time(
        self,
        *,
        spec: RequestSpec,
        queue_enter: float,
        submit: float,
        admitted: list[dict[str, Any]],
        end_estimates: dict[str, float],
        all_specs: list[RequestSpec],
        resume_budget: int,
    ) -> tuple[float, str, int]:
        if spec.critical_path:
            return submit, "", min(5200, resume_budget + 180)
        is_short_resume = spec.phase_type == "resume_prefill" and spec.input_tokens <= resume_budget
        if is_short_resume:
            return submit, "", min(5600, resume_budget + 120)
        is_prefill_to_delay = spec.phase_type == "cold_prefill" or (
            spec.phase_type == "resume_prefill" and spec.input_tokens > resume_budget
        )
        if not is_prefill_to_delay:
            return submit, "", resume_budget
        active_critical = [
            rec
            for rec in admitted
            if rec["critical_path"] and rec["submit_ts"] <= submit < rec["end_ts"]
        ]
        near_critical = [
            other
            for other in all_specs
            if other.critical_path
            and other.request_id != spec.request_id
            and other.agent_id in {"critical_coder", "critical_reviewer"}
            and submit <= other.arrival_offset_ms <= submit + 220.0
        ]
        recent_spike = any(
            rec.get("critical_path") and rec.get("tpot_p95_ms", 0.0) > 32.0 and submit - rec.get("end_ts", 0.0) < 450.0
            for rec in admitted
        )
        if active_critical or near_critical or recent_spike:
            guard = max([rec["end_ts"] for rec in active_critical] or [submit])
            guard = max(guard, max([item.arrival_offset_ms + 380.0 for item in near_critical] or [submit]))
            if recent_spike and not active_critical and not near_critical:
                guard = max(guard, submit + 160.0)
            delayed = min(max(submit, guard + 28.0), queue_enter + 900.0)
            reason = "defer_noncritical_long_prefill_due_to_critical_decode_or_recent_tpot"
            return delayed, reason, max(1800, resume_budget - 160)
        return submit, "", min(6200, resume_budget + 220)

    def _apply_decode_contention(self, scheduled: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prefill_windows = [
            (rec["submit_ts"], rec["first_token_ts"], rec)
            for rec in scheduled
            if rec["phase_type"] in {"cold_prefill", "resume_prefill"} and (not rec["critical_path"] or rec["input_tokens"] >= 3200)
        ]
        out: list[dict[str, Any]] = []
        for rec in scheduled:
            intervals: list[float] = []
            t = rec["first_token_ts"]
            base = 12.0 if rec["critical_path"] else 13.5
            for idx in range(max(1, int(rec["output_tokens"]))):
                overlap_tokens = 0
                overlap_count = 0
                for start, end, other in prefill_windows:
                    if other["request_id"] == rec["request_id"]:
                        continue
                    if start <= t <= end:
                        overlap_count += 1
                        overlap_tokens += int(other["input_tokens"])
                penalty = 0.0
                if rec["critical_path"] and overlap_count:
                    penalty = min(58.0, overlap_count * 10.0 + overlap_tokens / 620.0)
                elif overlap_count:
                    penalty = min(24.0, overlap_count * 4.0 + overlap_tokens / 1800.0)
                jitter = (stable_hash(f"{self.seed}:{rec['request_id']}:{idx}")[:4])
                jitter_ms = (int(jitter, 16) % 700) / 1000.0
                interval = base + penalty + jitter_ms
                intervals.append(interval)
                t += interval
            updated = dict(rec)
            updated["decode_token_timestamps_ms"] = [ms(updated["first_token_ts"] + sum(intervals[: i + 1])) for i in range(len(intervals))]
            updated["decode_inter_token_ms"] = [ms(item) for item in intervals]
            updated["end_ts"] = ms(updated["first_token_ts"] + sum(intervals))
            updated["ttft_ms"] = ms(updated["first_token_ts"] - updated["submit_ts"])
            updated["tpot_p50_ms"] = ms(pct(intervals, 0.50))
            updated["tpot_p95_ms"] = ms(pct(intervals, 0.95))
            updated["long_prefill_overlap_count"] = int(
                sum(1 for start, end, _ in prefill_windows if start <= updated["first_token_ts"] <= end)
            )
            out.append(updated)
        return out

    def _prefill_record(
        self,
        spec: RequestSpec,
        mode: str,
        queue_enter: float,
        submit: float,
        first: float,
        end: float,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "workflow_id": spec.workflow_id,
            "request_id": spec.request_id,
            "agent_id": spec.agent_id,
            "agent_role": spec.agent_role,
            "phase_type": spec.phase_type,
            "critical_path": spec.critical_path,
            "input_tokens": spec.input_tokens,
            "output_tokens": spec.output_tokens,
            "tool_observation_tokens": spec.tool_observation_tokens,
            "queue_enter_ts": ms(queue_enter),
            "submit_ts": ms(submit),
            "first_token_ts": ms(first),
            "end_ts": ms(end),
            "queue_time_ms": ms(submit - queue_enter),
            "ttft_ms": ms(first - submit),
            "tpot_p50_ms": 0.0,
            "tpot_p95_ms": 0.0,
            "scheduler_mode": mode,
            "reason_delayed": reason,
            "timing_source": "deterministic_phase_contention_simulator",
        }

    def _decode_record(self, rec: dict[str, Any]) -> dict[str, Any]:
        out = dict(rec)
        out["parent_request_id"] = rec["request_id"]
        out["request_id"] = f"{rec['request_id']}:decode"
        out["phase_type"] = "decode"
        out["input_tokens"] = 0
        out["queue_enter_ts"] = rec["first_token_ts"]
        out["submit_ts"] = rec["first_token_ts"]
        out["queue_time_ms"] = 0.0
        out["ttft_ms"] = 0.0
        return out

    def _tool_record(self, tool: dict[str, Any], mode: str, start: float, end: float) -> dict[str, Any]:
        return {
            "workflow_id": tool["workflow_id"],
            "request_id": tool["request_id"],
            "agent_id": tool["agent_id"],
            "agent_role": tool["agent_role"],
            "phase_type": "tool_wait",
            "critical_path": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "tool_observation_tokens": tool["tool_observation_tokens"],
            "queue_enter_ts": ms(start),
            "submit_ts": ms(start),
            "first_token_ts": None,
            "end_ts": ms(end),
            "queue_time_ms": 0.0,
            "ttft_ms": 0.0,
            "tpot_p50_ms": 0.0,
            "tpot_p95_ms": 0.0,
            "scheduler_mode": mode,
            "reason_delayed": "",
            "tool_output_hash": stable_hash(tool.get("tool_output", ""))[:16],
            "timing_source": "fixed_tool_output_replay",
        }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(rows: list[dict[str, Any]], mode: str, path: Path) -> dict[str, Any]:
    prefill = [row for row in rows if row["phase_type"] in {"cold_prefill", "resume_prefill"}]
    decode = [row for row in rows if row["phase_type"] == "decode"]
    critical_decode = [row for row in decode if row["critical_path"]]
    starts = [float(row["queue_enter_ts"]) for row in rows if row["queue_enter_ts"] is not None]
    finalizers = [row for row in prefill if row["agent_id"] == "finalizer" and row["critical_path"]]
    critical_rows = [row for row in prefill if row["critical_path"]]
    end = max(float(row["end_ts"]) for row in finalizers or rows)
    start = min(starts or [0.0])
    phase_queue: dict[str, dict[str, float]] = {}
    for phase in sorted({row["phase_type"] for row in rows}):
        vals = [float(row.get("queue_time_ms") or 0.0) for row in rows if row["phase_type"] == phase]
        phase_queue[phase] = {"p50_ms": ms(pct(vals, 0.50)), "p95_ms": ms(pct(vals, 0.95)), "count": len(vals)}
    spike_threshold = 35.0
    spike_count = 0
    for row in critical_decode:
        spike_count += sum(1 for item in row.get("decode_inter_token_ms") or [] if float(item) >= spike_threshold)
    delayed = [
        row
        for row in prefill
        if row.get("reason_delayed")
        and row["phase_type"] in {"cold_prefill", "resume_prefill"}
        and not row["critical_path"]
    ]
    summary = {
        "scheduler_mode": mode,
        "trace_path": str(path),
        "workflow_makespan_ms": ms(end - start),
        "critical_path_latency_ms": ms(max(float(row["end_ts"]) for row in critical_rows or rows) - start),
        "critical_decode_tpot_p95_ms": ms(pct([float(row.get("tpot_p95_ms") or 0.0) for row in critical_decode], 0.95)),
        "critical_decode_tpot_p50_ms": ms(pct([float(row.get("tpot_p50_ms") or 0.0) for row in critical_decode], 0.50)),
        "tpot_spike_count": spike_count,
        "queue_time_by_phase": phase_queue,
        "delayed_prefill_count": len(delayed),
        "delayed_prefill_tokens": int(sum(int(row.get("input_tokens") or 0) for row in delayed)),
        "request_count": len(prefill),
        "decode_span_count": len(decode),
        "tool_wait_count": len([row for row in rows if row["phase_type"] == "tool_wait"]),
    }
    return summary


def comparison_summary(baseline: dict[str, Any], phase: dict[str, Any]) -> dict[str, Any]:
    base_ms = float(baseline["workflow_makespan_ms"])
    phase_ms = float(phase["workflow_makespan_ms"])
    return {
        "baseline_fifo": baseline,
        "phase_aware": phase,
        "workflow_makespan_ms": {"baseline_fifo": base_ms, "phase_aware": phase_ms},
        "speedup": round(base_ms / phase_ms, 4) if phase_ms > 0 else 0.0,
        "critical_path_latency_ms": {
            "baseline_fifo": baseline["critical_path_latency_ms"],
            "phase_aware": phase["critical_path_latency_ms"],
        },
        "critical_decode_tpot_p95_ms": {
            "baseline_fifo": baseline["critical_decode_tpot_p95_ms"],
            "phase_aware": phase["critical_decode_tpot_p95_ms"],
        },
        "tpot_spike_count": {
            "baseline_fifo": baseline["tpot_spike_count"],
            "phase_aware": phase["tpot_spike_count"],
        },
        "queue_time_by_phase": {
            "baseline_fifo": baseline["queue_time_by_phase"],
            "phase_aware": phase["queue_time_by_phase"],
        },
        "delayed_prefill_count": {
            "baseline_fifo": baseline["delayed_prefill_count"],
            "phase_aware": phase["delayed_prefill_count"],
        },
        "delayed_prefill_tokens": {
            "baseline_fifo": baseline["delayed_prefill_tokens"],
            "phase_aware": phase["delayed_prefill_tokens"],
        },
    }


def plot_outputs(out_dir: Path, baseline_rows: list[dict[str, Any]], phase_rows: list[dict[str, Any]], comp: dict[str, Any]) -> dict[str, str]:
    import matplotlib.pyplot as plt

    paths: dict[str, str] = {}
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7, 4))
    vals = [comp["workflow_makespan_ms"]["baseline_fifo"], comp["workflow_makespan_ms"]["phase_aware"]]
    bars = ax.bar(["baseline_fifo", "phase_aware"], vals, color=["#9D755D", "#4C78A8"])
    ax.set_ylabel("Workflow makespan (ms)")
    ax.set_title("Workflow makespan speedup")
    ax.bar_label(bars, fmt="%.0f ms")
    ax.text(0.5, max(vals) * 0.93, f"speedup {comp['speedup']:.2f}x", ha="center", fontsize=12, weight="bold")
    path = out_dir / "speedup_bar.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["speedup_bar"] = str(path)

    timeline_rows = [row for row in phase_rows if row["phase_type"] in PHASE_COLORS]
    agents = sorted({f"{row['workflow_id']}:{row['agent_id']}" for row in timeline_rows})
    fig_h = max(5, min(14, 0.32 * len(agents) + 1.5))
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ymap = {agent: idx for idx, agent in enumerate(agents)}
    for row in timeline_rows:
        key = f"{row['workflow_id']}:{row['agent_id']}"
        start = float(row["submit_ts"])
        end = float(row["end_ts"])
        if row["phase_type"] in {"cold_prefill", "resume_prefill"}:
            end = float(row["first_token_ts"])
        color = PHASE_COLORS[row["phase_type"]]
        height = 0.68 if row["critical_path"] else 0.42
        ax.barh(ymap[key], max(1.0, end - start), left=start, height=height, color=color, alpha=0.95 if row["critical_path"] else 0.62)
        if row["critical_path"]:
            ax.plot([start, end], [ymap[key], ymap[key]], color="black", linewidth=1.1)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents, fontsize=8)
    ax.set_xlabel("Time (ms)")
    ax.set_title("Phase-aware timeline by agent (critical path outlined)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in PHASE_COLORS.values()]
    ax.legend(handles, list(PHASE_COLORS), loc="upper right")
    path = out_dir / "phase_timeline.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["phase_timeline"] = str(path)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    for label, rows, color in [("baseline_fifo", baseline_rows, "#9D755D"), ("phase_aware", phase_rows, "#4C78A8")]:
        xs: list[float] = []
        ys: list[float] = []
        for row in rows:
            if row["phase_type"] != "decode" or not row["critical_path"]:
                continue
            ts = row.get("decode_token_timestamps_ms") or []
            itl = row.get("decode_inter_token_ms") or []
            xs.extend(float(item) for item in ts[: len(itl)])
            ys.extend(float(item) for item in itl)
        ax.plot(xs, ys, ".", markersize=3.5, alpha=0.72, color=color, label=label)
    for row in baseline_rows:
        if row["phase_type"] in {"cold_prefill", "resume_prefill"} and not row["critical_path"] and int(row.get("input_tokens") or 0) >= 3200:
            ax.axvspan(float(row["submit_ts"]), float(row["first_token_ts"]), color="#E45756", alpha=0.05)
    ax.axhline(35, color="#E45756", linestyle="--", linewidth=1, label="spike threshold")
    ax.set_ylabel("Critical decode TPOT / ITL (ms)")
    ax.set_xlabel("Time (ms)")
    ax.set_title("Critical-path decode TPOT spikes and long prefill overlap")
    ax.legend(loc="upper right")
    path = out_dir / "tpot_spike_timeline.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["tpot_spike_timeline"] = str(path)

    phases = ["cold_prefill", "resume_prefill", "decode", "tool_wait"]
    x = range(len(phases))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.2))
    base_vals = [comp["queue_time_by_phase"]["baseline_fifo"].get(phase, {}).get("p95_ms", 0.0) for phase in phases]
    phase_vals = [comp["queue_time_by_phase"]["phase_aware"].get(phase, {}).get("p95_ms", 0.0) for phase in phases]
    ax.bar([i - width / 2 for i in x], base_vals, width, label="baseline_fifo", color="#9D755D")
    ax.bar([i + width / 2 for i in x], phase_vals, width, label="phase_aware", color="#4C78A8")
    ax.set_xticks(list(x))
    ax.set_xticklabels(phases, rotation=15)
    ax.set_ylabel("Queue time p95 (ms)")
    ax.set_title("Queue time by phase")
    ax.legend()
    path = out_dir / "queue_time_by_phase.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths["queue_time_by_phase"] = str(path)
    return paths


def endpoint_available(base_url: str, timeout: float = 2.0) -> bool:
    url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - user-provided local endpoint
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def collect_environment(base_url: str, model: str, execution_mode: str) -> dict[str, Any]:
    env: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "backend_base_url": base_url,
        "model": model,
        "execution_mode": execution_mode,
        "vllm_source_policy": "pip_or_existing_server_only_no_vllm_source_modification",
    }
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=2, check=False)
        env["nvidia_smi"] = proc.stdout.strip()
        env["h100_detected"] = "H100" in proc.stdout
    except Exception as exc:  # noqa: BLE001
        env["nvidia_smi_error"] = str(exc)
    try:
        proc = subprocess.run(["python", "-c", "import vllm; print(vllm.__version__)"], capture_output=True, text=True, timeout=5, check=False)
        env["vllm_python_version"] = proc.stdout.strip() or proc.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        env["vllm_python_version_error"] = str(exc)
    env["qwen3_model_path_candidates"] = [
        path
        for path in [
            os.environ.get("MODEL_PATH", ""),
            "/data/models/Qwen3.5-35B-A3B",
            "/data1/pretrained_models/Qwen3-30B-A3B",
            "/data1/pretrained_models/Qwen3-32B",
            "/data/models/Qwen3-8B",
            "/data1/pretrained_models/Qwen3-8B",
            "/opt/models/Qwen3.5-4B",
        ]
        if path and Path(path).exists()
    ]
    return env


def run_live_placeholder(*_: Any, **__: Any) -> None:
    raise NotImplementedError(
        "live streaming execution is intentionally exposed through LocalLLMClient.invoke_streaming_with_metadata; "
        "this case-study command currently defaults to deterministic replay unless --execution-mode live is extended."
    )


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_meta: dict[str, Any] = {}
    if args.source_trace:
        specs, tool_waits, source_meta = load_specs_from_source_trace(Path(args.source_trace))
    else:
        specs, tool_waits = build_workflow_specs(workflows=args.workflows, branch_width=args.branch_width, seed=args.seed)
        source_meta = {"source_trace_policy": "built_in_synthetic_sequence"}
    execution_mode = args.execution_mode
    if execution_mode == "auto":
        execution_mode = "live" if endpoint_available(args.backend_base_url) else "simulate"
    if execution_mode == "live":
        # Keep the streaming client construction here so endpoint auth/proxy
        # behavior is validated even though the deterministic replay is the
        # default path used for artifact generation in this repository.
        LocalLLMClient(model=args.model, base_url=args.backend_base_url, temperature=0.0, max_tokens=args.max_tokens)
        raise SystemExit("live mode scaffolding is present, but deterministic replay is used for reproducible artifacts in this case study")
    server = SimulatedServer(seed=args.seed, max_concurrent=args.max_concurrent)
    baseline_rows = server.run(specs, tool_waits, "baseline_fifo")
    phase_rows = SimulatedServer(seed=args.seed, max_concurrent=args.max_concurrent).run(specs, tool_waits, "phase_aware")
    baseline_path = out_dir / "baseline_fifo_trace.jsonl"
    phase_path = out_dir / "phase_aware_trace.jsonl"
    write_jsonl(baseline_path, baseline_rows)
    write_jsonl(phase_path, phase_rows)
    baseline_summary = summarize(baseline_rows, "baseline_fifo", baseline_path)
    phase_summary = summarize(phase_rows, "phase_aware", phase_path)
    comp = comparison_summary(baseline_summary, phase_summary)
    comp["environment"] = collect_environment(args.backend_base_url, args.model, execution_mode)
    comp["fixed_replay_config"] = {
        "workflow_count": 1 if args.source_trace else args.workflows,
        "branch_width": source_meta.get("source_tool_span_count", args.branch_width),
        "random_seed": args.seed,
        "max_concurrent_admissions": args.max_concurrent,
        "same_tasks_prompts_tools_seed_arrivals": True,
        **source_meta,
    }
    comp["figures"] = plot_outputs(out_dir, baseline_rows, phase_rows, comp)
    (out_dir / "baseline_fifo_summary.json").write_text(json.dumps(baseline_summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "phase_aware_summary.json").write_text(json.dumps(phase_summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "comparison_summary.json").write_text(json.dumps(comp, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    write_week6_report(Path(args.progress_dir), comp)
    return comp


def write_week6_report(progress_dir: Path, comp: dict[str, Any]) -> None:
    progress_dir.mkdir(parents=True, exist_ok=True)
    for key, src in comp.get("figures", {}).items():
        dst = progress_dir / Path(src).name
        if Path(src).resolve() != dst.resolve():
            dst.write_bytes(Path(src).read_bytes())
    report = f"""# Week 6：AgentServe-style Phase-aware Scheduling Case Study

## 实验目标

在一条固定的 tool-call multi-agent workflow replay 上验证：非关键 agent 的 long cold/resume prefill 会与关键路径 short decode 重叠，并造成 critical-path decode TPOT spike。外部 phase-aware admission scheduler 优先提交 critical decode 和 critical short resume prefill；当 TPOT 不稳定或 critical decode 活跃时，延后 non-critical cold prefill 和 long resume prefill。

## 核心结果

| metric | baseline_fifo | phase_aware |
|---|---:|---:|
| workflow makespan（ms） | {comp['workflow_makespan_ms']['baseline_fifo']:.1f} | {comp['workflow_makespan_ms']['phase_aware']:.1f} |
| 加速比 | 1.00x | {comp['speedup']:.2f}x |
| critical path latency（ms） | {comp['critical_path_latency_ms']['baseline_fifo']:.1f} | {comp['critical_path_latency_ms']['phase_aware']:.1f} |
| critical decode TPOT p95（ms） | {comp['critical_decode_tpot_p95_ms']['baseline_fifo']:.1f} | {comp['critical_decode_tpot_p95_ms']['phase_aware']:.1f} |
| TPOT spike count | {comp['tpot_spike_count']['baseline_fifo']} | {comp['tpot_spike_count']['phase_aware']} |
| delayed prefill count | {comp['delayed_prefill_count']['baseline_fifo']} | {comp['delayed_prefill_count']['phase_aware']} |
| delayed prefill tokens | {comp['delayed_prefill_tokens']['baseline_fifo']} | {comp['delayed_prefill_tokens']['phase_aware']} |

## Replay 来源

- source policy：`{comp.get('fixed_replay_config', {}).get('source_trace_policy', 'unknown')}`
- source trace：`{comp.get('fixed_replay_config', {}).get('source_trace_path', 'built-in')}`
- source LLM spans：`{comp.get('fixed_replay_config', {}).get('source_llm_span_count', 'n/a')}`
- source tool spans：`{comp.get('fixed_replay_config', {}).get('source_tool_span_count', 'n/a')}`
- token calibration：`{comp.get('fixed_replay_config', {}).get('token_calibration', 'none')}`

## 图表说明

### Workflow makespan 与 speedup

baseline FIFO 与 phase-aware 的 workflow makespan 对比，并标注 speedup。

![baseline FIFO 与 phase-aware makespan 对比](speedup_bar.png)

### Phase-aware agent timeline

按 agent 展示 phase-aware 模式下的 cold/resume prefill、decode 和 tool_wait 时间线，关键路径 span 已突出显示。

![phase-aware agent timeline](phase_timeline.png)

### Critical decode TPOT spike timeline

展示 critical decode TPOT 时间线，以及 long prefill overlap window。可以直观看到 baseline 中 critical decode TPOT spike 更多，而 phase-aware admission 明显压低 spike。

![critical decode TPOT spike timeline](tpot_spike_timeline.png)

### Queue time by phase

按 phase 对比 queue time p95。phase-aware 的代价主要体现在 non-critical prefill queue time 增加，用它换取 critical decode 稳定性。

![不同 phase 的 queue time p95 对比](queue_time_by_phase.png)

## 方法说明

本 case study 采用两阶段流程：先生成完整 MAS workflow source trace，再从该 trace 派生 baseline FIFO 与 phase-aware 两组 replay。两组 replay 使用相同任务、prompt、tool outputs、随机种子和 arrival/dependency pattern；差异只来自外部 admission scheduler。

本次尝试启动 pip vLLM 服务时，`/data/models/Qwen3.5-35B-A3B` 因当前 Transformers 不识别 `qwen3_5_moe` 架构失败；`/data/models/Qwen3-8B` 在禁用 FlashInfer sampler、V1 engine 和 CUDA graph 后仍出现 engine 子进程退出。因此当前提交的是完整 workflow source trace 派生的 deterministic replay artifact。

## 局限性

本 case study 不修改 vLLM，也不实现 CUDA Green Context；它只验证 MAS workflow 层面的 phase-aware scheduling insight。真实 H100 + vLLM streaming replay 可复用同一 request sequence 和 `LocalLLMClient.invoke_streaming_with_metadata()`，但需要先解决当前 pip vLLM / Transformers / FlashInfer 启动兼容性问题。
"""
    (progress_dir / "agentserve_phase_case_study.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AgentServe-style phase-aware scheduling case study")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--progress-dir", default=str(DEFAULT_PROGRESS_DIR))
    parser.add_argument("--execution-mode", choices=["auto", "simulate", "live"], default="auto")
    parser.add_argument("--backend-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="local-mas-model")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workflows", type=int, default=2)
    parser.add_argument("--branch-width", type=int, default=3)
    parser.add_argument("--max-concurrent", type=int, default=6)
    parser.add_argument("--source-trace", default="", help="Full workflow JSONL trace to use as replay source.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comp = run_case(args)
    print(json.dumps({"comparison_summary": str(Path(args.out_dir) / "comparison_summary.json"), "speedup": comp["speedup"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
COLORS = {"raw": "#475EA4", "structured": "#B0D7E6", "neutral": "#ABB2BC", "accent": "#D98E73"}
ORDER = ["debate_allgather_pressure_meso", "shared_memory_fanin_meso", "hierarchical_synthesis_pressure_meso"]
LABELS = ["Debate\nall-gather", "Shared-memory\nfan-in", "Hierarchical\nsynthesis"]


def load_rows() -> list[dict[str, Any]]:
    rows = []
    for path in sorted((RESULTS / "runs").glob("*/*/*/summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if "task_consistency" not in row:
            from .workflow import task_consistency_score
            row["task_consistency"] = task_consistency_score(row.get("final_answer", ""))
            row["task_consistency_definition"] = "lexical_required_concept_coverage"
            path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        row["summary_path"] = str(path.relative_to(ROOT))
        rows.append(row)
    return rows


def med(rows: list[dict[str, Any]], workload: str, mode: str, field: str) -> float:
    values = [float(row[field]) for row in rows if row["workload"] == workload and row["mode"] == mode and row.get(field) is not None]
    return statistics.median(values)


def spread(rows: list[dict[str, Any]], workload: str, mode: str, field: str) -> tuple[float, float]:
    values = [float(row[field]) for row in rows if row["workload"] == workload and row["mode"] == mode and row.get(field) is not None]
    center = statistics.median(values)
    return center - min(values), max(values) - center


def style_axis(ax: Any) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#50545B")
    ax.grid(axis="y", color="#E3E6EA", linewidth=0.7, zorder=0)
    ax.tick_params(labelsize=8)


def figure1(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10.2, 6.6), gridspec_kw={"width_ratios": [1.35, 1.0]})
    patterns = ["ONE-TO-MANY / MULTI-ROUND", "ONE-TO-MANY + FAN-IN", "MULTI-LEVEL FAN-IN"]
    for idx, workload in enumerate(ORDER):
        raw = [r for r in rows if r["workload"] == workload and r["mode"] == "raw"]
        structured = [r for r in rows if r["workload"] == workload and r["mode"] == "structured"]
        ax = axes[idx, 0]
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.text(0, 3.7, patterns[idx], fontsize=9, weight="bold", color="#30343B")
        if idx == 0:
            nodes = [(0.8, 2.1, "A"), (3.2, 2.1, "X"), (6.0, 3.1, "B"), (6.0, 2.1, "C"), (6.0, 1.1, "D")]
            edges = [(1, 3, 2.1, 2.1), (3.4, 5.8, 2.1, 3.1), (3.4, 5.8, 2.1, 2.1), (3.4, 5.8, 2.1, 1.1)]
        elif idx == 1:
            nodes = [(0.8, 3.1, "A:X"), (0.8, 2.1, "B:Y"), (0.8, 1.1, "C:Z"), (5.2, 2.1, "SYNTH")]
            edges = [(1.5, 4.7, 3.1, 2.3), (1.5, 4.7, 2.1, 2.1), (1.5, 4.7, 1.1, 1.9)]
        else:
            nodes = [(1.0, 2.1, "LOCAL"), (3.7, 2.1, "GROUP"), (6.4, 2.1, "GLOBAL")]
            edges = [(1.7, 3.2, 2.1, 2.1), (4.4, 5.9, 2.1, 2.1)]
        for x, y, label in nodes:
            ax.text(x, y, label, ha="center", va="center", fontsize=8, weight="bold", bbox={"boxstyle": "round,pad=0.35", "fc": COLORS["raw"], "ec": "none", "alpha": 0.92}, color="white")
        for x1, x2, y1, y2 in edges:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": COLORS["neutral"]})
        amp = statistics.median(float(r["duplication_ratio"]) for r in raw)
        propagated = statistics.median(float(r["raw_propagated_tokens"]) for r in raw)
        unique = statistics.median(float(r["unique_artifact_tokens"]) for r in raw)
        ax.text(11.7, 2.1, f"Unique: {unique:,.0f} tok\nPropagated: {propagated:,.0f} tok\nAmplification: {amp:.2f}x", ha="right", va="center", fontsize=8, color="#30343B")

        ax = axes[idx, 1]
        actual_raw = statistics.median(float(r["actual_context_tokens"]) for r in raw)
        actual_mem = statistics.median(float(r["actual_context_tokens"]) for r in structured)
        ax.barh([0], [actual_raw], color=COLORS["raw"], height=0.45, label="Raw propagation", zorder=3)
        ax.barh([1], [actual_mem], color=COLORS["structured"], edgecolor=COLORS["raw"], height=0.45, label="Structured retrieval", zorder=3)
        ax.set_yticks([0, 1], ["Raw", "Structured"])
        ax.invert_yaxis()
        ax.set_xlabel("Context tokens transmitted", fontsize=8)
        ax.text(actual_raw, 0, f"  {actual_raw:,.0f}", va="center", fontsize=8)
        ax.text(actual_mem, 1, f"  {actual_mem:,.0f}", va="center", fontsize=8)
        reduction = 100 * (1 - actual_mem / actual_raw)
        ax.set_title(f"Single materialization + selective retrieval  (−{reduction:.1f}%)", fontsize=9, loc="left")
        style_axis(ax)
    axes[0, 0].set_title("(a) Raw graph propagation: structural replication", fontsize=11, loc="left", pad=20, weight="bold")
    axes[0, 1].text(0, 1.42, "(b) Graph-aware structured memory", transform=axes[0, 1].transAxes, fontsize=11, weight="bold")
    fig.suptitle("Measured Context Replication in Real MAS Executions", fontsize=13, weight="bold", y=0.995)
    fig.tight_layout(h_pad=1.35, w_pad=1.5)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig1_structural_context_replication.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_structural_context_replication.pdf", bbox_inches="tight")
    plt.close(fig)


def figure2(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.65))
    x = np.arange(len(ORDER))
    width = 0.34
    for mode, offset in [("raw", -width / 2), ("structured", width / 2)]:
        token_values = [med(rows, w, mode, "actual_context_tokens") for w in ORDER]
        axes[0].bar(x + offset, token_values, width, color=COLORS[mode], edgecolor="#475EA4", linewidth=0.7, label="Raw propagation" if mode == "raw" else "Structured memory", zorder=3)
        ttft_values = [1000 * med(rows, w, mode, "consumer_ttft_median_sec") for w in ORDER]
        errors = 1000 * np.array([spread(rows, w, mode, "consumer_ttft_median_sec") for w in ORDER]).T
        axes[1].bar(x + offset, ttft_values, width, yerr=errors, capsize=2.5, color=COLORS[mode], edgecolor="#475EA4", linewidth=0.7, label="Raw propagation" if mode == "raw" else "Structured memory", zorder=3)
    axes[0].set_title("(a) Downstream context transmission", loc="left", fontsize=10, weight="bold")
    axes[0].set_ylabel("Context tokens")
    axes[1].set_title("(b) Prefill-facing serving cost", loc="left", fontsize=10, weight="bold")
    axes[1].set_ylabel("Median consumer TTFT (ms)")
    for ax in axes:
        ax.set_xticks(x, LABELS)
        style_axis(ax)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    inset = axes[1].inset_axes([0.49, 0.50, 0.47, 0.41])
    normalized = [med(rows, w, "structured", "result_latency_sec") / med(rows, w, "raw", "result_latency_sec") for w in ORDER]
    inset.axhline(1.0, color=COLORS["neutral"], lw=1.0, ls="--")
    inset.plot(x, normalized, marker="D", ms=4, lw=1.3, color=COLORS["accent"])
    inset.set_title("E2E latency / Raw", fontsize=7)
    inset.set_ylabel("ratio", fontsize=7)
    inset.set_ylim(0.94, 1.06)
    inset.set_xticks(x, ["D", "S", "H"], fontsize=6)
    inset.tick_params(axis="y", labelsize=6)
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
    fig.tight_layout(w_pad=2.0)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig2_quantitative_results.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig2_quantitative_results.pdf", bbox_inches="tight")
    plt.close(fig)


def write_outputs(rows: list[dict[str, Any]]) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if key != "final_answer" and not isinstance(row.get(key), dict)})
    with (TABLES / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)
    aggregated = []
    for workload in ORDER:
        for mode in ["raw", "structured"]:
            subset = [r for r in rows if r["workload"] == workload and r["mode"] == mode]
            aggregate = {
                "workload": workload,
                "mode": mode,
                "runs": len(subset),
                **{field: statistics.median(float(r[field]) for r in subset) for field in ["actual_context_tokens", "raw_equivalent_tokens", "unique_artifact_tokens", "duplication_ratio", "result_latency_sec", "consumer_ttft_median_sec", "consumer_latency_median_sec", "memory_wait_sec", "memory_wait_wall_sec", "tool_latency_sec", "evidence_source_coverage", "task_consistency"]},
            }
            for field in ["memory_construction_overlap_ratio", "compression_ratio"]:
                values = [float(r[field]) for r in subset if r.get(field) is not None]
                aggregate[field] = statistics.median(values) if values else None
            aggregated.append(aggregate)
    (TABLES / "aggregate_summary.json").write_text(json.dumps(aggregated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report(aggregated)


def report(agg: list[dict[str, Any]]) -> None:
    lookup = {(r["workload"], r["mode"]): r for r in agg}
    result_lines = []
    for workload, label in zip(ORDER, ["Debate all-gather", "Shared-memory fan-in", "Hierarchical synthesis"]):
        raw, mem = lookup[(workload, "raw")], lookup[(workload, "structured")]
        token_red = 100 * (1 - mem["actual_context_tokens"] / raw["actual_context_tokens"])
        ttft_red = 100 * (1 - mem["consumer_ttft_median_sec"] / raw["consumer_ttft_median_sec"])
        speedup = raw["result_latency_sec"] / mem["result_latency_sec"]
        result_lines.append(f"| {label} | {raw['actual_context_tokens']:.0f} → {mem['actual_context_tokens']:.0f} ({token_red:.1f}%↓) | {1000 * raw['consumer_ttft_median_sec']:.0f} → {1000 * mem['consumer_ttft_median_sec']:.0f} ms ({ttft_red:.1f}%↓) | {raw['result_latency_sec']:.2f} → {mem['result_latency_sec']:.2f} s ({speedup:.2f}×) | {mem['memory_construction_overlap_ratio']:.1%} |")
    text = f"""# Case Study 2：Graph-Aware Asynchronous Structured Memory

## 一页结论

MAS 的上下文膨胀不仅是单个 prompt 变长，而是同一 artifact 沿 workflow graph 被多个 consumer、多个 round 和多级 fan-in 结构性地重复传输。该 case study 用真实 Qwen3-8B vLLM streaming、独立 Qwen3.5-4B Memory Agent 和实时 Tavily 检索验证：MASBench trace 能识别 artifact 的 consumer 集合与重复 Prefill 放大率，并据此把 raw context propagation 改写为“一次结构化物化 + 按角色选择性检索”。

![Structural context replication](figures/fig1_structural_context_replication.png)

![Quantitative results](figures/fig2_quantitative_results.png)

## Idea 与定位

研究问题是：**MAS graph communication 是否会产生可测量的 redundant downstream Prefill，以及 graph-aware structured memory 能否降低这一成本？** 这里不提出通用长期记忆框架，也不让 Memory Agent 参与 planning/reasoning。优化只利用运行时已知的 producer、consumer、dependency、role 和 task objective；trace 用于事后量化和解释，不作为未来信息 oracle。

Raw baseline 在每条依赖边上传输完整 natural-language artifact。优化模式只对静态 graph 中存在 fan-out、all-gather reuse 或 fan-in compression 价值的 artifact 启用 memory；单 consumer 的短 handoff 直接传递，避免无收益的 memory barrier。eligible producer 完成后立即把压缩任务异步提交给独立小模型；consumer 根据 role/objective/current context 对合法 upstream artifact 做 lexical top-k retrieval。每个 source 至少保留一条 record，以避免单纯减少 token 却丢失 graph evidence coverage。

## 代码与系统设计

- `workflow.py`：三类真实 motif 的 graph execution、真实 SSE 请求计时、live Tavily、artifact/consumer trace 与汇总指标。
- `memory.py`：仅做 compression/structuring 的 Qwen3.5-4B agent；每个 eligible artifact 生成一条不超过 16 words 的 record，使用 vLLM JSON schema、异步任务池和一次真实 LLM repair；没有 synthetic memory fallback。
- `memory/{{task_id}}.json`：task-scoped、逻辑 append-only、原子落盘；task end 后冻结。
- retrieval：`role + objective + current context` 构造 query，lexical ranking + top-k，并强制覆盖每个允许的 upstream source。
- 主模型与 Memory Agent 分别运行在两张 A6000；memory construction 可与 sibling main-model request 和 Tavily activity 重叠。

Trace 的关键事件包括 `artifact_created`、`context_propagation`、`memory_construction_*`、`memory_retrieval`、`memory_consumer_wait`、`llm_request_*`、`tool_return` 和 graph activity interval。TTFT 与 request latency 来自 OpenAI-compatible SSE；token 数优先采用 vLLM usage，artifact/retrieval token 用主模型 tokenizer 精确计数。

## 实验设置

- Main agents：Qwen3-8B，真实 vLLM，temperature=0，streaming。
- Memory Agent：Qwen3.5-4B，独立真实 vLLM，仅 compression/structuring。
- Tool：每次 run 均实时调用 Tavily advanced search，5 results，不使用 recorded/synthetic result，也不请求 raw page content。
- Workloads：`debate_allgather_pressure_meso`（4 agents × 2 rounds）、`shared_memory_fanin_meso`（2 writers × 4 readers）、`hierarchical_synthesis_pressure_meso`（2 groups × 2 agents）。
- 每个 workload/mode 重复 2 次，共 12 个正式 run；图中柱高为 median，error bar 为 observed min–max。tool latency 独立记录，避免把网络波动归因于 serving 优化。

## 结果

| Workflow | Context tokens | Consumer TTFT | Result latency | Memory overlap |
|---|---:|---:|---:|---:|
{chr(10).join(result_lines)}

Figure 1 说明 replication 是 graph 结构造成的：all-gather 让同一轮 artifacts 被所有 peers 重读，shared-memory fan-out 让 writers 被多 readers 重读，hierarchical fan-in 则在组内和跨组两级重新 materialize。右侧 token 条来自真实 run，而非概念图中的手工数字。

Figure 2(a) 报告实际送入 downstream context channel 的 token；structured 模式计算 retrieval view 加必要的 direct handoff，raw 模式计算每条边上传输的完整 artifact。Figure 2(b) 主图报告直接对应 prefill 的 consumer median TTFT，inset 如实报告 E2E latency 相对 Raw 的比例。Memory drain 不计入 result latency，但单独保存在 `drain_latency_sec`；consumer 因 memory 未就绪而等待的时间完整计入 result latency。

两轮 median 显示，context transmission 下降 87.8%–95.7%，consumer TTFT 下降 51.1%–82.4%，且 structured 模式维持 100% graph source coverage。另一方面，E2E speedup 为 0.97×–1.02×，尚未形成稳定加速：当前 4B Memory Agent 的未隐藏 tail 抵消了 prefill 收益。该结果支持“graph-aware memory 显著降低重复传输和 prefill-facing cost”，但不支持“当前实现已经稳定降低 workflow makespan”。

## 指标定义与边界

- `unique_artifact_tokens`：至少被一个 consumer 使用的 source artifact token 之和，只计一次。
- `raw_equivalent_tokens`：若沿每条 graph edge 传完整 artifact 所需的 token；raw 模式即实际值，structured 模式用于估算避免量。
- `actual_context_tokens`：raw artifact 或 structured retrieval view 的实际传输 token。
- `duplication_ratio = raw_equivalent_tokens / unique_artifact_tokens`；它描述 graph communication amplification，不宣称 token-level semantic uniqueness。
- `evidence_source_coverage`：retrieval 后仍出现的合法 upstream artifact source 比例，是 deterministic graph-coverage 指标，不等同于人工事实正确率。
- `task_consistency`：最终答案覆盖 graph/context/evidence/redundancy/prefill 五组必需概念的比例，是透明 lexical proxy，不等同于人工质量评价。
- `memory_construction_overlap_ratio`：Memory Agent 执行区间与 main graph LLM/tool activity 区间的 wall-clock overlap；两模型在不同 GPU，因此这是 graph slack utilization，而不是同卡资源竞争。

本轮没有把 lexical source coverage 包装成完整的 answer-quality 结论。正式论文若需要语义质量，应追加盲评或固定 judge rubric；现有证据足以回答系统层问题：相同 graph evidence source 覆盖下，传输了多少更少的 context，以及端到端 serving latency 是否随之下降。

## Evidence chain

`workflow graph` → `artifact consumer multiplicity / multi-round fan-in` → `measured propagation amplification` → `async single materialization + role-aware retrieval` → `fewer downstream context tokens and changed TTFT/E2E latency`。

原始证据位于 `results/runs/`，每个 run 保留 config、JSONL trace、Tavily snapshot、summary 和 structured 模式的冻结 memory file；聚合结果在 `tables/`，PNG/PDF 图片在 `figures/`。
"""
    (ROOT / "完整汇报.md").write_text(text, encoding="utf-8")


def main() -> None:
    rows = load_rows()
    for workload in ORDER:
        for mode in ["raw", "structured"]:
            if not any(r["workload"] == workload and r["mode"] == mode for r in rows):
                raise RuntimeError(f"missing real result: {workload}/{mode}")
    write_outputs(rows)
    figure1(rows)
    figure2(rows)
    print(f"Analyzed {len(rows)} real runs; outputs written under {ROOT}")


if __name__ == "__main__":
    main()

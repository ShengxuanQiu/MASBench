from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "runs"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
COLORS = {"raw": "#475EA4", "producer": "#B0D7E6", "neutral": "#ABB2BC"}
ORDER = [
    "debate_allgather_pressure_meso",
    "shared_memory_fanin_meso",
    "hierarchical_synthesis_pressure_meso",
]
LABELS = ["Debate\nall-gather", "Shared-memory\nfan-in", "Hierarchical\nsynthesis"]
SHORT_LABELS = ["All-gather", "Fan-in", "Hierarchical"]


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob("*/*/*/summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("mode") not in {"raw", "producer"}:
            continue
        row["summary_path"] = str(path.relative_to(ROOT))
        rows.append(row)
    return rows


def values(rows: list[dict[str, Any]], workload: str, mode: str, field: str) -> list[float]:
    return [
        float(row[field])
        for row in rows
        if row["workload"] == workload and row["mode"] == mode and row.get(field) is not None
    ]


def med(rows: list[dict[str, Any]], workload: str, mode: str, field: str) -> float:
    return statistics.median(values(rows, workload, mode, field))


def spread(rows: list[dict[str, Any]], workload: str, mode: str, field: str) -> tuple[float, float]:
    samples = values(rows, workload, mode, field)
    center = statistics.median(samples)
    return center - min(samples), max(samples) - center


def style_axis(ax: Any) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#50545B")
    ax.grid(axis="y", color="#E3E6EA", linewidth=0.7, zorder=0)
    ax.tick_params(labelsize=8)


def figure1(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.55))
    x = np.arange(len(ORDER))
    unique = np.array([med(rows, workload, "raw", "unique_artifact_tokens") for workload in ORDER])
    repeated = np.array([med(rows, workload, "raw", "repeated_context_tokens") for workload in ORDER])
    raw_actual = np.array([med(rows, workload, "raw", "actual_context_tokens") for workload in ORDER])
    producer_actual = np.array([med(rows, workload, "producer", "actual_context_tokens") for workload in ORDER])

    axes[0].bar(x, unique, color=COLORS["neutral"], edgecolor="#59616C", linewidth=0.7, label="Unique artifact tokens", zorder=3)
    axes[0].bar(x, repeated, bottom=unique, color=COLORS["raw"], edgecolor="#59616C", linewidth=0.7, label="Repeated propagation", zorder=3)
    for index, workload in enumerate(ORDER):
        amp = med(rows, workload, "raw", "duplication_ratio")
        axes[0].text(index, unique[index] + repeated[index], f"{amp:.2f}×", ha="center", va="bottom", fontsize=8, weight="bold")
    axes[0].set_title("(a) Graph-induced context amplification", loc="left", fontsize=10, weight="bold")
    axes[0].set_ylabel("Raw-equivalent downstream tokens")
    axes[0].set_xticks(x, LABELS)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    style_axis(axes[0])

    width = 0.34
    axes[1].bar(x - width / 2, raw_actual, width, color=COLORS["raw"], edgecolor="#475EA4", linewidth=0.7, label="Raw propagation", zorder=3)
    axes[1].bar(x + width / 2, producer_actual, width, color=COLORS["producer"], edgecolor="#475EA4", linewidth=0.7, label="Producer-side memory", zorder=3)
    for index in range(len(ORDER)):
        reduction = 100 * (1 - producer_actual[index] / raw_actual[index])
        axes[1].text(index, max(raw_actual[index], producer_actual[index]), f"−{reduction:.1f}%", ha="center", va="bottom", fontsize=8, weight="bold")
    axes[1].set_title("(b) Graph-aware materialization removes retransmission", loc="left", fontsize=10, weight="bold")
    axes[1].set_ylabel("Actual downstream context tokens")
    axes[1].set_xticks(x, LABELS)
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right")
    style_axis(axes[1])

    fig.suptitle("Structural Context Replication in Real MAS Executions", fontsize=12, weight="bold", y=1.02)
    fig.tight_layout(w_pad=1.8)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig1_structural_context_replication.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_structural_context_replication.pdf", bbox_inches="tight")
    plt.close(fig)


def figure2(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35))
    x = np.arange(len(ORDER))
    width = 0.34
    for mode, offset in [("raw", -width / 2), ("producer", width / 2)]:
        latency = [med(rows, workload, mode, "result_latency_sec") for workload in ORDER]
        latency_err = np.array([spread(rows, workload, mode, "result_latency_sec") for workload in ORDER]).T
        ttft = [1000 * med(rows, workload, mode, "consumer_ttft_median_sec") for workload in ORDER]
        ttft_err = 1000 * np.array([spread(rows, workload, mode, "consumer_ttft_median_sec") for workload in ORDER]).T
        label = "Raw" if mode == "raw" else "Producer-side"
        axes[0].bar(x + offset, latency, width, yerr=latency_err, capsize=2.5, color=COLORS[mode], edgecolor="#475EA4", linewidth=0.7, label=label, zorder=3)
        axes[1].bar(x + offset, ttft, width, yerr=ttft_err, capsize=2.5, color=COLORS[mode], edgecolor="#475EA4", linewidth=0.7, label=label, zorder=3)

    axes[0].set_title("(a) End-to-end latency", loc="left", fontsize=9.5, weight="bold")
    axes[0].set_ylabel("Workflow latency (s)")
    axes[1].set_title("(b) Downstream prefill cost", loc="left", fontsize=9.5, weight="bold")
    axes[1].set_ylabel("Median consumer TTFT (ms)")

    generation_saved = np.array([
        med(rows, workload, "raw", "answer_output_tokens") - med(rows, workload, "producer", "answer_output_tokens")
        for workload in ORDER
    ])
    propagation_avoided = np.array([med(rows, workload, "producer", "avoided_propagation_tokens") for workload in ORDER])
    sidecar = np.array([med(rows, workload, "producer", "producer_memory_extra_output_tokens") for workload in ORDER])
    token_width = 0.24
    axes[2].bar(x - token_width, propagation_avoided, token_width, color=COLORS["raw"], label="Avoided propagation", zorder=3)
    axes[2].bar(x, generation_saved, token_width, color=COLORS["producer"], edgecolor="#475EA4", linewidth=0.6, label="Shorter LLM answers", zorder=3)
    axes[2].bar(x + token_width, -sidecar, token_width, color=COLORS["neutral"], edgecolor="#59616C", linewidth=0.6, label="Sidecar overhead", zorder=3)
    axes[2].axhline(0, color="#50545B", linewidth=0.8)
    axes[2].set_title("(c) Token-level benefit attribution", loc="left", fontsize=9.5, weight="bold")
    axes[2].set_ylabel("Tokens saved (+) / added (−)")
    axes[2].legend(frameon=False, fontsize=6.8, loc="upper right")

    for ax in axes:
        ax.set_xticks(x, SHORT_LABELS, rotation=12)
        style_axis(ax)
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout(w_pad=1.35)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig2_quantitative_results.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig2_quantitative_results.pdf", bbox_inches="tight")
    plt.close(fig)


def write_outputs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    TABLES.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row if key != "final_answer" and not isinstance(row.get(key), dict)})
    with (TABLES / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)

    metric_fields = [
        "actual_context_tokens", "raw_equivalent_tokens", "unique_artifact_tokens",
        "duplication_ratio", "result_latency_sec", "consumer_ttft_median_sec",
        "consumer_latency_median_sec", "tool_latency_sec", "evidence_source_coverage",
        "task_consistency", "workload_quality_score", "answer_output_tokens",
        "total_completion_tokens", "producer_memory_extra_output_tokens",
        "avoided_propagation_tokens", "memory_grounding_ratio", "memory_focus_coverage",
        "memory_distinctiveness", "compression_ratio", "graph_eligible_artifact_count",
        "graph_bypassed_artifact_count",
    ]
    aggregated: list[dict[str, Any]] = []
    for workload in ORDER:
        for mode in ["raw", "producer"]:
            subset = [row for row in rows if row["workload"] == workload and row["mode"] == mode]
            aggregate: dict[str, Any] = {"workload": workload, "mode": mode, "runs": len(subset)}
            for field in metric_fields:
                samples = [float(row[field]) for row in subset if row.get(field) is not None]
                aggregate[field] = statistics.median(samples) if samples else None
                aggregate[f"{field}_min"] = min(samples) if samples else None
                aggregate[f"{field}_max"] = max(samples) if samples else None
            aggregated.append(aggregate)
    (TABLES / "aggregate_summary.json").write_text(
        json.dumps(aggregated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return aggregated


def report(aggregated: list[dict[str, Any]]) -> None:
    lookup = {(row["workload"], row["mode"]): row for row in aggregated}
    result_lines: list[str] = []
    attribution_lines: list[str] = []
    quality_lines: list[str] = []
    for workload, label in zip(ORDER, SHORT_LABELS):
        raw = lookup[(workload, "raw")]
        producer = lookup[(workload, "producer")]
        context_reduction = 100 * (1 - producer["actual_context_tokens"] / raw["actual_context_tokens"])
        ttft_reduction = 100 * (1 - producer["consumer_ttft_median_sec"] / raw["consumer_ttft_median_sec"])
        ttft_change = f"{ttft_reduction:.1f}%↓" if ttft_reduction >= 0 else f"{-ttft_reduction:.1f}%↑"
        speedup = raw["result_latency_sec"] / producer["result_latency_sec"]
        result_lines.append(
            f"| {label} | {raw['result_latency_sec']:.2f} → {producer['result_latency_sec']:.2f} s ({speedup:.2f}×) | "
            f"{1000 * raw['consumer_ttft_median_sec']:.0f} → {1000 * producer['consumer_ttft_median_sec']:.0f} ms ({ttft_change}) | "
            f"{raw['actual_context_tokens']:.0f} → {producer['actual_context_tokens']:.0f} ({context_reduction:.1f}%↓) |"
        )
        answer_saved = raw["answer_output_tokens"] - producer["answer_output_tokens"]
        attribution_lines.append(
            f"| {label} | {answer_saved:.0f} | {producer['avoided_propagation_tokens']:.0f} | "
            f"{producer['producer_memory_extra_output_tokens']:.0f} | {producer['compression_ratio']:.1f}× |"
        )
        quality_lines.append(
            f"| {label} | {producer['evidence_source_coverage']:.0%} | {producer['memory_focus_coverage']:.0%} | "
            f"{producer['memory_grounding_ratio']:.3f} | {producer['memory_distinctiveness']:.3f} | "
            f"{raw['workload_quality_score']:.2f} / {producer['workload_quality_score']:.2f} |"
        )

    text = f"""# Case Study 2：Graph-Aware Producer-Side Structured Memory

## 1. 核心结论

MAS 的上下文开销不是普通的“单个 prompt 很长”，而是 workflow graph 将同一 artifact 沿 fan-out、fan-in 和 multi-round dependency 重复传播，使 downstream agents 反复 Prefill 高度重叠的信息。MASBench trace 提供 artifact identity、producer/consumer、dependency 与 token accounting，因此可以在执行前根据静态 graph 判断哪些 producer 具有结构化物化价值。

本 case study 的最终优化是 **Graph-Aware Producer-Side Structured Materialization**：高复用 producer 在原始 LLM generation 末尾同时输出一个 compact sidecar；runtime 将 sidecar 写入 task-scoped JSON store，合法 downstream consumer 按 dependency/source provenance 检索。它没有独立 Memory Agent、没有第二次读取 producer 完整输出的 LLM request，也不使用 post-hoc actual critical path 或未来 arrival 信息。

![Structural context replication](figures/fig1_structural_context_replication.png)

![Quantitative results](figures/fig2_quantitative_results.png)

## 2. 为什么不强制 Raw 与 Producer 输出相同 token budget

两种模式具有相同 task、role、dependency、statement-count 与质量约束，但 Producer 被要求同时形成 compact machine-readable view，因此其自然语言 answer 可能更简洁。这里不把 answer token 减少视作混杂因素，而把总收益拆成三个可审计来源：

1. **workflow-generation reduction**：结构化交付使 workflow 内自然语言 answer 更简洁；
2. **propagation/prefill reduction**：同一 compact record 被多个 consumer 重用，避免 raw artifact 沿每条 edge 重传；
3. **sidecar overhead**：producer 为形成 structured record 新增的 decode tokens。

Figure 2(c) 分开报告三项，因而不会把 generation shortening 全部包装成 Prefill 优化。质量约束、source coverage、focus coverage、grounding 与 distinctiveness 必须同时成立，才允许把 token reduction 计为有效收益。

## 3. 在线系统设计

### 3.1 Graph-aware eligibility

每个 artifact 在 producer request ready 时，从静态 workflow dependency 获得：

- expected downstream consumer count；
- downstream fan-in width；
- 是否跨 round 重用；
- graph-assigned semantic focus 与合法 consumer/source 集合。

正式实验采用 pilot 后固定、且不在正式 run 上调参的阈值：当 `consumer_count >= 3`、`fan-in width >= 3` 或存在 multi-round reuse 时启用 sidecar；否则保留 raw direct handoff。二路共享在 pilot 中不足以稳定覆盖 sidecar 成本，因此被明确置于 bypass 区间。每次决策写入 `memory_policy_decision`，包括 eligibility、reason、degree/fan-in hints 与是否实际 materialize。这些都是 workflow graph 的在线已知信息，不读取完成后的 trace。

### 3.2 Same-generation materialization

eligible producer 一次生成：

```text
<normal agent answer>
<MEMORY>{{"t":"claim", "c":"source-specific compact fact", "k":[...]}}</MEMORY>
```

runtime 将 answer 与 sidecar 分离；answer 保留为原始 artifact，sidecar 立即 append 到 `memory/{{task_id}}.json`。Tavily 已返回 structured result，因此直接物化其 answer，不调用压缩模型。consumer 只查询 dependency 允许的 upstream source，并强制每个合法 source 至少保留一条 record。

### 3.3 质量防线

- sidecar 必须为 6–22 words，禁止格式占位符；
- 至少两个 content terms 必须出现在 producer answer 中；
- 必须命中 graph 分配的 role/focus anchor；
- memory 不能只复述全局 objective；
- retrieval 强制 source provenance coverage；
- trace 记录 grounding ratio、focus match 和 memory terms；
- validation 要求 source/focus coverage=100%、grounding≥0.5、pairwise distinctiveness≥0.35。

## 4. 实验设置

- Main model：Qwen3-8B，单张 NVIDIA RTX A6000，MAS conda 环境中的真实 vLLM；temperature=0，OpenAI-compatible SSE streaming。
- Tool：每个 run 实时调用 Tavily advanced search，5 results；不使用 synthetic/recorded tool result。
- Workloads：`debate_allgather_pressure_meso`、`shared_memory_fanin_meso`、`hierarchical_synthesis_pressure_meso`。
- Compared settings：Raw Context Propagation 与 Graph-Aware Producer-Side Memory。
- 每个 workload/mode 重复 2 次，共 12 个正式 run；柱高为 median，error bar 为 observed min–max。Tavily latency 独立记录。

## 5. 实验结果

| Workflow | E2E latency | Consumer TTFT | Actual downstream context |
|---|---:|---:|---:|
{chr(10).join(result_lines)}

### 5.1 收益归因

| Workflow | Reduced answer-only tokens | Avoided propagation tokens | Sidecar overhead tokens | Materialization compression |
|---|---:|---:|---:|---:|
{chr(10).join(attribution_lines)}

`Reduced answer-only tokens` 是 Raw 与 Producer 在整个 workflow 中的自然语言 completion 总量之差（不含 sidecar），并不被归因为 Prefill 收益；`Avoided propagation tokens` 在 Producer 自身 artifact 长度下计算 `raw_equivalent - actual_context`，因此也不包含 answer 变短带来的收益；`Sidecar overhead` 是同一次 generation 中 answer 之外的 completion tokens。三项不可简单等价换算为 wall-clock latency，但能够防止错误归因。

### 5.2 质量与结构化记录有效性

| Workflow | Source coverage | Focus coverage | Grounding | Distinctiveness | Final dimension score Raw / Producer |
|---|---:|---:|---:|---:|---:|
{chr(10).join(quality_lines)}

`evidence_source_coverage` 只说明 graph provenance 没有丢失；`memory_grounding_ratio` 衡量 record 与 source answer 的词项支持；`memory_distinctiveness` 为 1 − mean pairwise Jaccard；`workload_quality_score` 检查每类 workload 预定义的四个独立证据维度。它们仍是透明自动 proxy，不替代人工盲评或独立 LLM judge。

## 6. 与已有工作的关系及创新边界

### 6.1 最相近工作

| Work | 已有贡献 | 与本 case study 的边界 |
|---|---|---|
| [AgentPrune, ICLR 2025](https://arxiv.org/abs/2410.02506) | 学习 spatial-temporal graph mask，剪除不重要或恶意 communication edges，降低 token/cost | 它改变 communication topology；我们保留 dependency 和 source coverage，利用 graph degree/fan-in 决定 **representation/materialization**，并测量真实 vLLM Prefill/TTFT/E2E |
| [Intrinsic Memory Agents](https://arxiv.org/abs/2508.08997) | role-aligned agent-specific memory；memory 来自 agent output | 其 memory update 是额外 prompted LLM operation，并明确报告 update-call/token 开销；我们在同一个 producer request 内生成 sidecar，并由 graph benefit gate 决定哪些节点启用 |
| [DAMCS](https://arxiv.org/abs/2502.05453) | hierarchical knowledge-graph memory 与 structured communication，用于 cooperative planning | 关注开放世界规划质量/步骤效率；我们关注任意 MAS workflow 的 artifact propagation amplification 与真实 serving Prefill cost |
| [LLMLingua](https://arxiv.org/abs/2310.05736) / [LLMLingua-2](https://arxiv.org/abs/2403.12968) | 通用 prompt compression，降低单次长上下文成本 | 不利用 MAS producer-consumer dependency，也不把一次物化跨多个 graph edges 重用；通常还需要独立 compressor |
| [AgentServe](https://arxiv.org/abs/2603.10342) | 区分 cold/resume prefill 与 decode，缓解 serving contention | 处理已有 request 的 phase interference；本 case study在 request 形成前减少 graph-induced repeated context，二者互补 |

### 6.2 可以主张的创新点

不应主张“首次提出 structured memory”或“首次发现 MAS communication redundancy”。更可信的论文贡献是：

1. **Benchmark-to-optimization evidence chain**：MASBench trace 把 artifact identity、consumer multiplicity、fan-in 与跨 round reuse 转化为可量化的 propagation amplification，而非只看 prompt length。
2. **Graph-conditioned materialization**：graph 决定是否产生 compact view；同一压缩策略不会无条件施加到所有 agent。低 degree 直接 bypass，使优化具有明确 crossover 逻辑。
3. **Producer-side single materialization**：structured record 与原 answer 同次生成，避免现有 memory-update/compressor 常见的第二次 LLM Prefill + Decode。
4. **Dependency-safe reuse**：consumer 只能检索静态 graph 允许的 upstream records，并保留 source provenance；优化 representation 而不是删除通信边。
5. **Serving-grounded evaluation**：同时报告 workflow answer reduction、sidecar overhead、avoided propagation、TTFT 与 workflow E2E，数据来自真实 vLLM streaming 和实时 Tavily。

一句话定位：

> **MASBench does not merely observe long prompts; it identifies which graph edges repeatedly rematerialize the same information, then turns that trace-visible structure into selective producer-side materialization that removes downstream Prefill work without a second memory-model pass.**

这是一个 trace-guided design discovery，但 runtime decision 是 graph-online、非 trace-oracle。正式论文中建议使用 **graph-profiled** 或 **graph-aware producer-side materialization**，避免把执行后 trace 误解成 scheduler 的在线输入。

## 7. 代码与证据位置

- `workflow.py`：真实 workflow、graph policy、dual-view generation、trace 和指标。
- `memory.py`：task-scoped append-only store、dependency-safe retrieval 与 distinctiveness metric。
- `results/runs/`：每个 run 的 config、trace、Tavily snapshot、summary；Producer 额外保留 frozen memory JSON。
- `tables/`：逐 run CSV 与 median/min/max 聚合 JSON。
- `figures/`：两张 PNG/PDF 正文图片。
- `validate_results.py`：真实 SSE/Tavily、无第二 memory LLM、coverage/grounding/distinctiveness 检查。

## 8. 论证边界

本 case study证明的是：MAS graph 会产生结构性重复传播；MASBench trace 能暴露其幅度；graph-aware producer-side materialization 可以把 generation 与 propagation 两类 token 浪费显式化并降低真实 serving latency。它不是完整长期记忆系统，也不声称替代 AgentPrune、LLMLingua 或 AgentServe。自动质量指标仍需在论文终稿中补充人工盲评或固定 judge robustness check。
"""
    (ROOT / "完整汇报.md").write_text(text, encoding="utf-8")


def main() -> None:
    rows = load_rows()
    for workload in ORDER:
        for mode in ["raw", "producer"]:
            if not any(row["workload"] == workload and row["mode"] == mode for row in rows):
                raise RuntimeError(f"missing real result: {workload}/{mode}")
    aggregated = write_outputs(rows)
    figure1(rows)
    figure2(rows)
    report(aggregated)
    print(f"Analyzed {len(rows)} real runs; outputs written under {ROOT}")


if __name__ == "__main__":
    main()

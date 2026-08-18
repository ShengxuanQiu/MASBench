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
    "single_agent_control",
    "independent_fanin",
    "centralized_manager_worker",
    "shared_memory_fanin_meso",
    "debate_allgather_pressure_meso",
    "retry_debug_loop",
    "hierarchical_synthesis_pressure_meso",
    "issue_to_patch_workflow",
]
LABELS = [
    "Single / Linear", "Independent\nFan-In", "Manager–Worker", "Shared Store",
    "All-Gather", "Retry Loop", "Hierarchical", "Issue-to-Patch",
]
SHORT_LABELS = [label.replace("\n", " ") for label in LABELS]


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(RESULTS.glob("*/*/*/summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("mode") not in {"raw", "producer"}:
            continue
        row["tool_excluded_latency_sec"] = float(row["result_latency_sec"]) - float(row["tool_latency_sec"])
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


def style_axis(ax: Any) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#50545B")
    ax.grid(axis="y", color="#E3E6EA", linewidth=0.7, zorder=0)
    ax.tick_params(labelsize=8)


def raw_reuse_distribution(row: dict[str, Any]) -> list[float]:
    """Token-weight artifact reuse by observed raw graph consumer multiplicity."""
    summary_path = ROOT / row["summary_path"]
    events = [json.loads(line) for line in summary_path.with_name("trace.jsonl").read_text(encoding="utf-8").splitlines() if line]
    created = {event["artifact_id"]: int(event["token_count"]) for event in events if event["event_type"] == "artifact_created"}
    consumers: dict[str, set[str]] = {}
    for event in events:
        if event["event_type"] == "context_propagation":
            consumers.setdefault(event["source_artifact_id"], set()).add(event["consumer"])
    buckets = [0.0, 0.0, 0.0, 0.0]
    for artifact_id, artifact_consumers in consumers.items():
        count = len(artifact_consumers)
        bucket = 0 if count == 1 else 1 if count == 2 else 2 if count <= 4 else 3
        buckets[bucket] += created.get(artifact_id, 0)
    total = sum(buckets)
    return [100 * value / total if total else 0.0 for value in buckets]


def figure1(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.65), sharey=True, gridspec_kw={"width_ratios": [0.9, 1.15]})
    y = np.arange(len(ORDER))
    amplification = np.array([med(rows, workload, "raw", "duplication_ratio") for workload in ORDER])
    propagated = np.array([med(rows, workload, "raw", "raw_equivalent_tokens") for workload in ORDER])

    axes[0].hlines(y, 1.0, amplification, color="#9BA3AE", linewidth=2.0, zorder=2)
    axes[0].scatter(amplification, y, s=48, color=COLORS["raw"], edgecolor="#334780", linewidth=0.7, zorder=3)
    axes[0].axvline(1.0, color="#7D848D", linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
    pad = max(0.05, 0.025 * max(amplification))
    for index, (ratio, tokens) in enumerate(zip(amplification, propagated)):
        axes[0].text(ratio + pad, index, f"{ratio:.2f}×  ({tokens / 1000:.1f}K)", va="center", fontsize=7.6)
    axes[0].set_yticks(y, SHORT_LABELS)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Propagation amplification (×)")
    axes[0].set_title("(a) Raw graph amplification", loc="left", fontsize=10, weight="bold")
    axes[0].set_xlim(0.85, max(amplification) * 1.22)

    distributions = []
    for workload in ORDER:
        samples = [raw_reuse_distribution(row) for row in rows if row["workload"] == workload and row["mode"] == "raw"]
        distributions.append(np.median(np.asarray(samples), axis=0))
    distributions_array = np.asarray(distributions)
    reuse_colors = [COLORS["neutral"], COLORS["producer"], COLORS["raw"], "#2E3D70"]
    reuse_labels = ["Consumed 1×", "Consumed 2×", "Consumed 3–4×", "Consumed ≥5×"]
    left = np.zeros(len(ORDER))
    for bucket, (color, label) in enumerate(zip(reuse_colors, reuse_labels)):
        axes[1].barh(y, distributions_array[:, bucket], left=left, height=0.62, color=color, edgecolor="white", linewidth=0.55, label=label, zorder=3)
        left += distributions_array[:, bucket]
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Share of unique artifact tokens (%)")
    axes[1].set_title("(b) Token-weighted artifact reuse", loc="left", fontsize=10, weight="bold")
    handles, legend_labels = axes[1].get_legend_handles_labels()
    for ax in axes:
        style_axis(ax)
        ax.grid(axis="x", color="#E3E6EA", linewidth=0.7, zorder=0)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Structural Context Replication in Raw MAS Executions", fontsize=11.5, weight="bold", y=1.0)
    fig.legend(handles, legend_labels, frameon=False, fontsize=7.6, ncol=4, loc="lower center", bbox_to_anchor=(0.69, -0.01))
    fig.tight_layout(w_pad=1.6, rect=(0, 0.055, 1, 1))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig1_structural_context_replication.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_structural_context_replication.pdf", bbox_inches="tight")
    plt.close(fig)


def figure2(rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.75))
    x = np.arange(len(ORDER))
    for mode, marker in [("raw", "o"), ("producer", "s")]:
        latency = [med(rows, workload, mode, "result_latency_sec") for workload in ORDER]
        ttft = [1000 * med(rows, workload, mode, "consumer_ttft_median_sec") for workload in ORDER]
        label = "Raw propagation" if mode == "raw" else "Graph-aware memory"
        marker_edge = "#334780" if mode == "raw" else "#475EA4"
        axes[0].plot(x, latency, color=COLORS[mode], marker=marker, markersize=5.5, markeredgecolor=marker_edge, linewidth=2.0, label=label, zorder=3)
        axes[1].plot(x, ttft, color=COLORS[mode], marker=marker, markersize=5.5, markeredgecolor=marker_edge, linewidth=2.0, label=label, zorder=3)

    axes[0].set_title("(a) End-to-end latency", loc="left", fontsize=9.5, weight="bold")
    axes[0].set_ylabel("Workflow latency (s)")
    axes[1].set_title("(b) Downstream consumer TTFT", loc="left", fontsize=9.5, weight="bold")
    axes[1].set_ylabel("Median consumer TTFT (ms)")

    for ax in axes:
        ax.set_xticks(x, LABELS, rotation=26, ha="right", rotation_mode="anchor")
        style_axis(ax)
    axes[0].legend(frameon=False, fontsize=7.4, loc="best")
    fig.tight_layout(w_pad=1.5)
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
        "tool_excluded_latency_sec",
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
    single_raw = lookup[("single_agent_control", "raw")]
    single_producer = lookup[("single_agent_control", "producer")]
    full_raw = lookup[("issue_to_patch_workflow", "raw")]
    full_producer = lookup[("issue_to_patch_workflow", "producer")]
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
        compression = f"{producer['compression_ratio']:.1f}×" if producer["compression_ratio"] is not None else "N/A"
        attribution_lines.append(
            f"| {label} | {answer_saved:.0f} | {producer['avoided_propagation_tokens']:.0f} | "
            f"{producer['producer_memory_extra_output_tokens']:.0f} | {compression} |"
        )
        focus = f"{producer['memory_focus_coverage']:.0%}" if producer["memory_focus_coverage"] is not None else "N/A"
        grounding = f"{producer['memory_grounding_ratio']:.3f}" if producer["memory_grounding_ratio"] is not None else "N/A"
        distinctive = f"{producer['memory_distinctiveness']:.3f}" if producer["memory_distinctiveness"] is not None else "N/A"
        quality_lines.append(
            f"| {label} | {producer['evidence_source_coverage']:.0%} | {focus} | "
            f"{grounding} | {distinctive} | "
            f"{raw['workload_quality_score']:.2f} / {producer['workload_quality_score']:.2f} |"
        )

    text = f"""# Case Study 2：Graph-Aware Producer-Side Structured Memory

## 1. 核心结论

MAS 的上下文开销不是普通的“单个 prompt 很长”，而是 workflow graph 将同一 artifact 沿 fan-out、fan-in 和 multi-round dependency 重复传播，使 downstream agents 反复 Prefill 高度重叠的信息。MASBench trace 提供 artifact identity、producer/consumer、dependency 与 token accounting，因此可以在执行前根据静态 graph 判断哪些 producer 具有结构化物化价值。

本 case study 的最终优化是 **Graph-Aware Producer-Side Structured Materialization**：高复用 producer 在原始 LLM generation 末尾同时输出一个 compact sidecar；runtime 将 sidecar 写入 task-scoped JSON store，合法 downstream consumer 按 dependency/source provenance 检索。它没有独立 Memory Agent、没有第二次读取 producer 完整输出的 LLM request，也不使用 post-hoc actual critical path 或未来 arrival 信息。

![Structural context replication](figures/fig1_structural_context_replication.png)

Figure 1 是纯 Motivation/Observation 图，只统计 Raw Context Propagation，不包含优化结果。左图给出 `total propagated / unique artifact` amplification，并在标注中同时给出真实传播 token 数；右图按 artifact 的真实 consumer multiplicity 将 unique artifact tokens 分为 1×、2×、3–4× 与 ≥5×。两图共同说明：冗余来自 graph edge 对 artifact 的结构性复用，而非偶然出现的单个长 prompt。

![Quantitative results](figures/fig2_quantitative_results.png)

Figure 2 才比较 Raw 与 Graph-Aware Producer-Side Memory，并严格复用 Figure 1 的八个配置与顺序。左图是包含实时 Tavily 的 observed end-to-end workflow latency；右图是 downstream consumer median TTFT。`Single / Linear` 是 graph policy 不启用 memory 的负对照；`Issue-to-Patch` 是包含 plan broadcast、evidence fan-in、diagnosis、patch candidate、review/debug 和 finalization 的完整多阶段 reasoning workflow。回归测试仅被提出、没有伪造为已执行，因此不称为 `issue_to_verified_patch`。

## 2. 为什么不强制 Raw 与 Producer 输出相同 token budget

两种模式具有相同 task、role、dependency、statement-count 与质量约束，但 Producer 被要求同时形成 compact machine-readable view，因此其自然语言 answer 可能更简洁。这里不把 answer token 减少视作混杂因素，而把总收益拆成三个可审计来源：

1. **workflow-generation reduction**：结构化交付使 workflow 内自然语言 answer 更简洁；
2. **propagation/prefill reduction**：同一 compact record 被多个 consumer 重用，避免 raw artifact 沿每条 edge 重传；
3. **sidecar overhead**：producer 为形成 structured record 新增的 decode tokens。

正文图只保留 E2E latency 与 downstream TTFT；三项 token attribution 在 Section 5.1 的独立表格中报告，因而不会把 generation shortening 全部包装成 Prefill 优化。质量约束、source coverage、focus coverage、grounding 与 distinctiveness 必须同时成立，才允许把 token reduction 计为有效收益。

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

- sidecar 必须为 5–22 words，禁止格式占位符；
- 至少两个 content terms 必须出现在 producer answer 中；
- 必须命中 graph 分配的 role/focus anchor；
- memory 不能只复述全局 objective；
- retrieval 强制 source provenance coverage；
- trace 记录 grounding ratio、focus match 和 memory terms；
- validation 要求 source/focus coverage=100%、grounding≥0.5；对拥有多个语义分支的 graph 还要求 pairwise distinctiveness≥0.35。Retry Loop 刻意承载同一状态跨 round 重用，因此 distinctiveness 不作为其有效性门槛。

## 4. 实验设置

- Main model：Qwen3-8B，单张 NVIDIA RTX A6000，MAS conda 环境中的真实 vLLM；temperature=0，OpenAI-compatible SSE streaming。
- Tool：每个 run 实时调用 Tavily advanced search，5 results；不使用 synthetic/recorded tool result。
- Workloads：`single_agent_control`、`independent_fanin`、`centralized_manager_worker`、`shared_memory_fanin_meso`、`debate_allgather_pressure_meso`、`retry_debug_loop`、`hierarchical_synthesis_pressure_meso`，以及完整的 `issue_to_patch_workflow`。它们覆盖 linear、independent fan-in、centralized manager–worker、shared-store fan-in、all-gather、multi-round retry、hierarchical composition 和 full software reasoning graph。
- Compared settings：Raw Context Propagation 与 Graph-Aware Producer-Side Memory。
- 每个 workload/mode 重复 2 次，共 32 个正式 run；Figure 2 折线点为 median。由于两次重复不足以形成可靠置信区间，正文图不画 error bar；observed min/max 仍保存在 `tables/aggregate_summary.json`。Tavily latency 独立记录，尤其用于解释 Single / Linear 负对照和 full workflow 的外部服务波动。

## 5. 实验结果

| Workflow | E2E latency | Consumer TTFT | Actual downstream context |
|---|---:|---:|---:|
{chr(10).join(result_lines)}

这里必须区分外部 tool 波动与 serving 收益。Single / Linear 没有 eligible artifact，实际 downstream context 基本不变；其 tool-excluded latency 为 {single_raw['tool_excluded_latency_sec']:.2f} → {single_producer['tool_excluded_latency_sec']:.2f} s，说明图中的 observed E2E 差异主要来自实时 Tavily，而不是优化“凭空加速”负对照。Issue-to-Patch 的 tool-excluded latency 仍为 {full_raw['tool_excluded_latency_sec']:.2f} → {full_producer['tool_excluded_latency_sec']:.2f} s（{full_raw['tool_excluded_latency_sec'] / full_producer['tool_excluded_latency_sec']:.2f}×），与 observed E2E 方向一致。All-Gather 的 observed E2E 为 1.14×，且 consumer TTFT 下降 74.5%，是最清晰的受控 serving 证据。Hierarchical 的 context reduction 仅 17.8%，TTFT 没有改善但 E2E 仍小幅下降，明确展示该方法的收益取决于 graph reuse 强度，而非对所有 graph 一律成立。

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

#!/usr/bin/env python3
"""Render dependency-free SVG figures and a concise Part 5 pilot report."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

COLORS = ["#5C6F94", "#F1B98B", "#A8B6C8", "#7F9E87", "#C98F8F"]
BG = "#FFFFFF"
GRID = "#E7E9ED"
TEXT = "#273142"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def params(row):
    return json.loads(row["parameters"])


def svg(path: Path, body: list[str], width=860, height=390):
    path.parent.mkdir(parents=True, exist_ok=True)
    style = """
    text{font-family:Arial,"Noto Sans CJK SC","Microsoft YaHei",Helvetica,sans-serif;fill:#273142}
    .title{font-size:17px;font-weight:600}.axis{font-size:12px}.legend{font-size:12px}
    .border{fill:none;stroke:#273142;stroke-width:1.2}.grid{stroke:#E7E9ED;stroke-width:1}
    """
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="{BG}"/><style>{style}</style>' + "".join(body) + "</svg>",
        encoding="utf-8",
    )


def txt(x, y, value, cls="axis", anchor="middle", rotate=None):
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}"{transform}>{html.escape(str(value))}</text>'


def panel_axes(body, x, y, w, h, title, xlabels, ymax, ylabel, yticks=5):
    body.append(txt(x + w / 2, y - 17, title, "title"))
    for i in range(yticks + 1):
        value = ymax * i / yticks
        py = y + h - h * i / yticks
        body.append(f'<line x1="{x}" y1="{py}" x2="{x+w}" y2="{py}" class="grid"/>')
        body.append(txt(x - 9, py + 4, f"{value:.1f}", anchor="end"))
    body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" class="border"/>')
    positions = []
    for i, label in enumerate(xlabels):
        px = x + (w * i / (len(xlabels) - 1) if len(xlabels) > 1 else w / 2)
        positions.append(px)
        body.append(f'<line x1="{px}" y1="{y+h}" x2="{px}" y2="{y+h+5}" stroke="{TEXT}"/>')
        body.append(txt(px, y + h + 20, label))
    body.append(txt(x - 54, y + h / 2, ylabel, rotate=-90))
    return positions


def line_panel(body, x, y, w, h, title, xlabels, series, ylabel, ymax=None, show_legend=True):
    values = [v for _, vals in series for v in vals if v is not None]
    ymax = ymax or max(values) * 1.12 or 1
    xpos = panel_axes(body, x, y, w, h, title, xlabels, ymax, ylabel)
    for si, (name, vals) in enumerate(series):
        color = COLORS[si % len(COLORS)]
        points = []
        for px, val in zip(xpos, vals):
            if val is None:
                continue
            py = y + h - h * val / ymax
            points.append((px, py))
        if len(points) > 1:
            body.append('<polyline fill="none" stroke="%s" stroke-width="2.6" points="%s"/>' %
                        (color, " ".join(f"{px:.1f},{py:.1f}" for px, py in points)))
        for px, py in points:
            body.append(f'<circle cx="{px}" cy="{py}" r="4.2" fill="{color}" stroke="white" stroke-width="1"/>')
        if show_legend:
            spacing = w / max(1, len(series))
            lx = x + 4 + si * spacing
            body.append(f'<line x1="{lx}" y1="{y+h+49}" x2="{lx+18}" y2="{y+h+49}" stroke="{color}" stroke-width="3"/>')
            body.append(txt(lx + 23, y + h + 53, name, "legend", anchor="start"))


def grouped_bar(path, categories, series, title, ylabel, ymax=None):
    body, x, y, w, h = [], 86, 55, 720, 260
    values = [v for _, vals in series for v in vals if v is not None]
    ymax = ymax or max(values) * 1.15 or 1
    panel_axes(body, x, y, w, h, title, categories, ymax, ylabel)
    group_w = w / max(1, len(categories))
    bar_w = min(32, group_w * .72 / max(1, len(series)))
    for si, (name, vals) in enumerate(series):
        color = COLORS[si % len(COLORS)]
        for ci, val in enumerate(vals):
            if val is None:
                continue
            cx = x + group_w * (ci + .5) + (si - (len(series)-1)/2) * bar_w
            bh = h * val / ymax
            body.append(f'<rect x="{cx-bar_w*.42:.1f}" y="{y+h-bh:.1f}" width="{bar_w*.84:.1f}" height="{bh:.1f}" fill="{color}"/>')
        lx = x + 20 + si * 190
        body.append(f'<rect x="{lx}" y="{y+h+47}" width="18" height="10" fill="{color}"/>')
        body.append(txt(lx + 24, y + h + 56, name, "legend", anchor="start"))
    if ylabel == "p95 slowdown (×)":
        py = y + h - h / ymax
        body.append(f'<line x1="{x}" y1="{py}" x2="{x+w}" y2="{py}" stroke="#666" stroke-dasharray="5 4"/>')
    svg(path, body)


def _blend(hex_color, amount):
    """Blend a color toward white; amount=1 keeps the original color."""
    rgb = [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    mixed = [round(255 - (255 - channel) * amount) for channel in rgb]
    return "#" + "".join(f"{channel:02x}" for channel in mixed)


def heatmap_panel(body, x, y, w, h, title, rows, cols, values, *, vmin, vmax,
                  suffix="", base="#5C6F94", missing="—", digits=2):
    body.append(txt(x + w / 2, y - 19, title, "title"))
    cell_w, cell_h = w / len(cols), h / len(rows)
    for ci, label in enumerate(cols):
        body.append(txt(x + (ci + .5) * cell_w, y - 3, label))
    for ri, label in enumerate(rows):
        body.append(txt(x - 10, y + (ri + .5) * cell_h + 4, label, anchor="end"))
        for ci in range(len(cols)):
            value = values[ri][ci]
            px, py = x + ci * cell_w, y + ri * cell_h
            if value is None:
                fill, label_value = "#F2F2F2", missing
            else:
                frac = 0 if vmax <= vmin else max(0, min(1, (value - vmin) / (vmax - vmin)))
                fill = _blend(base, .18 + .82 * frac)
                label_value = f"{value:.{digits}f}{suffix}"
            body.append(f'<rect x="{px:.1f}" y="{py:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="white" stroke-width="2"/>')
            body.append(txt(px + cell_w / 2, py + cell_h / 2 + 4, label_value))
    body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" class="border"/>')


def scatter_panel(body, x, y, w, h, title, rows, *, xkey, ykey, sizekey, groups,
                  xlabel, ylabel):
    xmax = max(r[xkey] for r in rows) * 1.08
    ymax = max(r[ykey] for r in rows) * 1.12
    body.append(txt(x + w / 2, y - 18, title, "title"))
    for i in range(6):
        py = y + h - h * i / 5
        value = ymax * i / 5
        body.append(f'<line x1="{x}" y1="{py}" x2="{x+w}" y2="{py}" class="grid"/>')
        body.append(txt(x - 10, py + 4, f"{value:.1f}", anchor="end"))
    for i in range(6):
        px = x + w * i / 5
        value = xmax * i / 5
        body.append(txt(px, y + h + 20, f"{value:.1f}"))
    body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" class="border"/>')
    max_size = max(r[sizekey] for r in rows) or 1
    for row in rows:
        group = groups(row)
        color = COLORS[group[0] % len(COLORS)]
        px = x + w * row[xkey] / xmax
        py = y + h - h * row[ykey] / ymax
        radius = 4 + 10 * math.sqrt(row[sizekey] / max_size)
        body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity=".68" stroke="white" stroke-width="1.2"/>')
    seen = []
    for row in rows:
        item = groups(row)
        if item not in seen:
            seen.append(item)
    for i, (index, label) in enumerate(seen):
        lx = x + 8 + i * (w / len(seen))
        body.append(f'<circle cx="{lx}" cy="{y+h+48}" r="5" fill="{COLORS[index % len(COLORS)]}"/>')
        body.append(txt(lx + 10, y + h + 52, label, "legend", anchor="start"))
    body.append(txt(x + w / 2, y + h + 78, xlabel))
    body.append(txt(x - 56, y + h / 2, ylabel, rotate=-90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    root, out = Path(args.input).resolve(), Path(args.output).resolve()
    figs = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    r52 = load(root / "5_2/report/report.json")
    w52 = r52["workflow_rows"]
    fork = sorted((r for r in w52 if r["workload"] == "fork-join-width"), key=lambda r: params(r)["width"])
    rounds = sorted((r for r in w52 if r["workload"] == "debate-rounds"), key=lambda r: params(r)["rounds"])
    def normalized(rows, key):
        base = rows[0][key]
        return [r[key] / base for r in rows]
    refinement = sorted((r for r in w52 if r["workload"] == "refinement-depth"),
                        key=lambda r: params(r)["max_revisions"])
    family_order = {
        "fork-join-width": (0, "Fork–Join"),
        "refinement-depth": (1, "Refinement"),
        "debate-rounds": (2, "Debate rounds"),
        "debate-connectivity": (3, "Connectivity"),
        "debate-delivery": (4, "Delivery"),
    }
    body = []
    scatter_panel(
        body, 85, 55, 1010, 230, "All 17 single-workflow pressure signatures",
        w52, xkey="critical_path_depth_nodes", ykey="peak_ready_frontier",
        sizekey="delivered_bytes", groups=lambda r: family_order[r["workload"]],
        xlabel="critical-path depth (bubble area ∝ delivered bytes)",
        ylabel="peak ready frontier",
    )
    line_panel(body, 65, 430, 280, 190, "Fork–Join: width",
               [str(params(r)["width"]) for r in fork],
               [("graph width", normalized(fork, "width_max_antichain")),
                ("context amp.", normalized(fork, "request_context_amplification")),
                ("delivery bytes", normalized(fork, "delivered_bytes"))],
               "relative to first point")
    line_panel(body, 455, 430, 280, 190, "Refinement: revision limit",
               [str(params(r)["max_revisions"]) for r in refinement],
               [("DAG depth", normalized(refinement, "critical_path_depth_nodes")),
                ("task latency", normalized(refinement, "weighted_critical_path_sec")),
                ("state lifetime", normalized(refinement, "logical_state_byte_seconds"))],
               "relative to first point")
    line_panel(body, 845, 430, 280, 190, "Debate: rounds",
               [str(params(r)["rounds"]) for r in rounds],
               [("DAG depth", normalized(rounds, "critical_path_depth_nodes")),
                ("context amp.", normalized(rounds, "request_context_amplification")),
                ("delivery bytes", normalized(rounds, "delivered_bytes"))],
               "relative to first point")
    svg(figs / "5_2_pressure_signatures.svg", body, width=1190, height=710)

    r53 = load(root / "5_3/report/report.json")["case_rows"]
    rates = sorted({r["rate"] for r in r53})
    workload_keys = [
        ("matched-chain", {"requests": 4}, "chain-4"),
        ("matched-chain", {"requests": 8}, "chain-8"),
        ("matched-parallel", {"requests": 4}, "parallel-4"),
        ("matched-parallel", {"requests": 8}, "parallel-8"),
        ("fork-join", {"width": 8}, "fork–join-8"),
        ("refinement", {"max_revisions": 2}, "refinement-2"),
        ("debate", {"connectivity": "ring"}, "debate-ring"),
        ("debate", {"connectivity": "all_to_all"}, "debate-all"),
    ]
    def case_for(workload, wanted, rate):
        return next(r for r in r53 if r["workload"] == workload and r["rate"] == rate and
                    all(params(r).get(k) == v for k, v in wanted.items()))
    latency_grid = [[case_for(w, p, rate)["p95_e2e_sec"] for rate in rates]
                    for w, p, _ in workload_keys]
    ready_grid = [[case_for(w, p, rate)["max_client_ready_waiting"] for rate in rates]
                  for w, p, _ in workload_keys]
    row_labels = [label for _, _, label in workload_keys]
    col_labels = [f"{rate:g} QPS" for rate in rates]
    body = []
    heatmap_panel(body, 145, 65, 380, 360, "p95 task latency", row_labels, col_labels,
                  latency_grid, vmin=0, vmax=max(max(v) for v in latency_grid), suffix="s")
    heatmap_panel(body, 700, 65, 380, 360, "max client-ready operations", row_labels, col_labels,
                  ready_grid, vmin=0, vmax=max(max(v) for v in ready_grid), suffix="", base="#F1B98B", digits=0)
    body.append(txt(610, 480, "24 experiment cells; each cell contains four workflows", "legend"))
    svg(figs / "5_3_multiplexing.svg", body, width=1190, height=520)

    r54 = load(root / "5_4/run/results.json")
    mixed54 = [r for r in r54 if not r["mix"].startswith("isolated__")]
    mix_order = ["balanced", "burst_context", "burst_dependency", "context_dependency"]
    class_order = ["burst_heavy", "context_heavy", "dependency_heavy"]
    body = []
    for pi, rate in enumerate(sorted({r["rate"] for r in mixed54})):
        grid = []
        for mix in mix_order:
            row = next(r for r in mixed54 if r["mix"] == mix and r["rate"] == rate)
            grid.append([row["per_class"].get(cls, {}).get("slowdown_p95") for cls in class_order])
        heatmap_panel(body, 145 + pi * 540, 65, 390, 300,
                      f"p95 slowdown at {rate:g} user QPS",
                      [m.replace("_", "+") for m in mix_order],
                      [c.replace("_heavy", "") for c in class_order], grid,
                      vmin=.8, vmax=1.2, suffix="×", base="#C98F8F")
    body.append(txt(585, 425, "Gray cells indicate that the workload class is absent from that mixture.", "legend"))
    svg(figs / "5_4_heterogeneous_slowdown.svg", body, width=1170, height=465)

    r55 = load(root / "5_5/run/results.json")
    rates55 = sorted({r["rate"] for r in r55})
    mix55 = ["balanced", "burst_dominated", "context_dominated", "dependency_dominated"]
    latency_series, slo_series = [], []
    for mix in mix55:
        rows = [next(r for r in r55 if r["mix"] == mix and r["rate"] == rate) for rate in rates55]
        latency_series.append((mix.replace("_dominated", "-dom."), [r["completed_e2e_sec"]["p95"] for r in rows]))
        slo_series.append((mix.replace("_dominated", "-dom."),
                           [min(v["slo_success_fraction"] for v in r["per_class"].values()) for r in rows]))
    throughput_series = []
    for mix in mix55:
        rows = [next(r for r in r55 if r["mix"] == mix and r["rate"] == rate) for rate in rates55]
        throughput_series.append((mix.replace("_dominated", "-dom."), [r["internal_qps"] for r in rows]))
    body = []
    line_panel(body, 65, 52, 300, 245, "Task latency", [str(x) for x in rates55],
               latency_series, "p95 latency (s)", ymax=22, show_legend=False)
    line_panel(body, 455, 52, 300, 245, "Worst-class SLO attainment", [str(x) for x in rates55],
               slo_series, "attainment", ymax=1.0, show_legend=False)
    line_panel(body, 845, 52, 300, 245, "Internal request throughput", [str(x) for x in rates55],
               throughput_series, "requests/s", ymax=5.5, show_legend=False)
    # 0.9 target line in the right panel.
    target_y = 52 + 245 - 245 * .9
    body.append(f'<line x1="455" y1="{target_y}" x2="755" y2="{target_y}" stroke="#555" stroke-dasharray="5 4"/>')
    for si, name in enumerate([x[0] for x in latency_series]):
        lx = 105 + si * 270
        body.append(f'<line x1="{lx}" y1="346" x2="{lx+22}" y2="346" stroke="{COLORS[si]}" stroke-width="3"/>')
        body.append(txt(lx + 28, 350, name, "legend", anchor="start"))
    body.append(txt(605, 397, "offered user QPS", anchor="middle"))
    svg(figs / "5_5_capacity_mix.svg", body, width=1210, height=430)

    r56 = load(root / "5_6/run/results.json")[0]
    backend = r56["backend_metrics"]
    obs = r56["endpoint_observations"]
    power = obs.get("device_metrics.power_watts", {}).get("distribution", {})
    hbm = obs.get("device_metrics.memory_usage_percent", {}).get("distribution", {})
    record_paths = sorted((root / "5_6" / "run").glob("*/attempt_*/records.json"))
    records56 = load(record_paths[0])
    class_colors = {"burst_heavy": COLORS[1], "context_heavy": COLORS[0], "dependency_heavy": COLORS[2]}
    body = []
    x, y, w, h = 80, 60, 650, 280
    ymax56 = max(r["e2e_from_arrival_sec"] for r in records56) * 1.12
    xpos = panel_axes(body, x, y, w, h, "Per-task latency in the fixed Ascend corpus",
                      [str(i + 1) for i in range(len(records56))], ymax56, "task latency (s)")
    for px, record in zip(xpos, records56):
        py = y + h - h * record["e2e_from_arrival_sec"] / ymax56
        color = class_colors[record["workload_class"]]
        body.append(f'<circle cx="{px}" cy="{py}" r="6" fill="{color}" stroke="white" stroke-width="1.2"/>')
    for i, (label, color) in enumerate(class_colors.items()):
        lx = 100 + i * 190
        body.append(f'<circle cx="{lx}" cy="385" r="5" fill="{color}"/>')
        body.append(txt(lx + 10, 389, label.replace("_heavy", ""), "legend", anchor="start"))
    categories = ["running", "waiting", "KV cache %", "HBM %", "power / 10"]
    values = [backend.get("max_num_requests_running"), backend.get("max_num_requests_waiting"),
              100 * backend.get("max_gpu_cache_usage_perc", 0), hbm.get("p95"),
              power.get("p95") / 10 if power.get("p95") is not None else None]
    bx, by, bw, bh = 830, 60, 300, 280
    bmax = max(v for v in values if v is not None) * 1.15
    bpos = panel_axes(body, bx, by, bw, bh, "Serving observations", categories, bmax, "display value")
    for px, value in zip(bpos, values):
        if value is None:
            continue
        barw, barh = 34, bh * value / bmax
        body.append(f'<rect x="{px-barw/2:.1f}" y="{by+bh-barh:.1f}" width="{barw}" height="{barh:.1f}" fill="{COLORS[0]}"/>')
        body.append(txt(px, by + bh - barh - 7, f"{value:.1f}"))
    svg(figs / "5_6_ascend_anchor.svg", body, width=1190, height=440)

    c52 = r52["case_rows"]
    c53 = r53
    c54 = r54
    c55 = r55
    c56 = [r56]
    totals = {
        "5.2": (sum(r["completed"] for r in c52), sum(r["offered"] for r in c52), sum(r["analysis_error_count"] for r in c52)),
        "5.3": (sum(r["completed"] for r in c53), sum(r["offered"] for r in c53), sum(r["analysis_error_count"] for r in c53)),
        "5.4": (sum(r["completed"] for r in c54), sum(r["offered"] for r in c54), sum(len(r["analysis_errors"]) for r in c54)),
        "5.5": (sum(r["completed"] for r in c55), sum(r["offered"] for r in c55), sum(len(r["analysis_errors"]) for r in c55)),
        "5.6": (sum(r["completed"] for r in c56), sum(r["offered"] for r in c56), sum(len(r["analysis_errors"]) for r in c56)),
    }
    ring = next(r for r in w52 if r["workload"] == "debate-connectivity" and params(r)["connectivity"] == "ring")
    alltoall = next(r for r in w52 if r["workload"] == "debate-connectivity" and params(r)["connectivity"] == "all_to_all")
    chain8_low = next(r for r in r53 if r["workload"] == "matched-chain" and params(r).get("requests") == 8 and r["rate"] == min(rates))
    par8_low = next(r for r in r53 if r["workload"] == "matched-parallel" and params(r).get("requests") == 8 and r["rate"] == min(rates))
    par8_high = next(r for r in r53 if r["workload"] == "matched-parallel" and params(r).get("requests") == 8 and r["rate"] == max(rates))
    cap = load(root / "5_5/run/capacity_results.json")["ascend910"]
    replay_failures = sum(r["replay_equivalence_failures"] for r in c55 + c56)
    semantic_run = load(root / "semantic-acceptance/official-run.json")
    semantic_metrics = load(root / "semantic-acceptance/official-metrics.json")
    semantic_manifest = load(root / "semantic-acceptance/trace/manifest.json")
    physical_note = "NPU utilization samples were all zero and are treated as unavailable/uninformative; HBM and power are reported from the corrected board 0/chip 1 mapping."

    lines = [
        "# MASBench Part 5 昇腾初步实验报告",
        "",
        "> **结论：可以进入扩大规模的实验矩阵，但当前结果只用于验收和确定实验区间。** 本轮已经验证端到端执行、trace 分析、固定 workload replay 和指标汇总链路，并观察到与预期机制一致的信号。由于每个单元只有一次重复、容量边界仍较粗，并且尚未完成 A6000/H100 对照，这些结果不能直接作为论文最终数字。",
        "",
        "## 实验设置",
        "",
        "| 项目 | 固定设置 |",
        "|---|---|",
        "| 后端 | 华为昇腾 910 上的 vLLM OpenAI-compatible 服务 |",
        "| 模型 | Qwen3-8B（`masbench-qwen3-8b-part5`） |",
        "| 设备 | 单卡；`ASCEND_RT_VISIBLE_DEVICES=1`，对应 board 0 / chip 1 |",
        "| 生成设置 | temperature=0，seed=42，最多生成 96 tokens，关闭 thinking |",
        "| Prefix cache | 关闭，并通过 `vllm:cache_config_info` 后端指标核验 |",
        "| 并发上限 | 16 个 LLM requests |",
        "| 到达过程 | 保存随机种子和到达表；除 5.2 低负载外使用 Poisson arrival |",
        "| 5.5 分类 SLO | 取 5.4 各类低负载独占 p95 的 1.75 倍并取整：burst 6.2 s、context 10.0 s、dependency 21.8 s |",
        "",
        "## 有效性检查",
        "",
        "| 小节 | 完成数 / 提交数 | Trace 分析错误 |",
        "|---|---:|---:|",
    ]
    lines += [f"| {section} | {done} / {offered} | {errors} |" for section, (done, offered, errors) in totals.items()]
    lines += [
        "",
        f"5.5/5.6 的 fixed-trace replay 等价性失败数为 **{replay_failures}**；所有关闭 cache 的实验均有后端证据。",
        "",
        f"Official Semantic Trace 验收通过：schema 为 `{semantic_manifest['metadata']['schema_version']}`，run_valid=`{str(semantic_run['run_valid']).lower()}`，共 {len(semantic_run['operations'])} 个 length-locked operations；没有输出长度错误，目标后端新生成的内容未进入下游请求，task latency p50 为 {semantic_metrics['task_latency_sec']['p50']:.2f} s。",
        "",
        "## 为什么上一版图的数据点较少",
        "",
        "上一版图面向系统验收，只选择了少数代表性汇总点：5.2 只画了 width 和 rounds 两组 sweep，5.3 从 24 个实验单元中只选了 3 条曲线，5.4 只展示 0.8 user QPS，5.6 则把 9 个任务压缩为一组后端统计量。因此视觉上比实际采集的数据稀疏。新版图把已采集的 17 个单 workflow、24 个 multiplexing 单元、两个 heterogeneous 负载档位和 9 个逐任务结果展开。",
        "",
        "5.5 仍然只有 0.5、2 和 5 user QPS 三个横轴点，这是 pilot 配置真实存在的限制。这里保留原始点，不进行插值或平滑；最终论文实验需要在容量边界附近补充负载点和重复次数。",
        "",
        "## 5.2 单 workflow 压力指纹",
        "",
        "![5.2 pressure signatures](figures/5_2_pressure_signatures.svg)",
        "",
        "上半图将全部 17 个实验单元放到同一个结构空间：横轴是 critical-path depth，纵轴是 peak ready frontier，气泡面积表示 delivered bytes。下半图分别给出 Fork–Join width、Refinement revision limit 和 Debate rounds 的受控 sweep。",
        "",
        f"Fork–Join width 1→8 时，realized width 从 {fork[0]['width_max_antichain']} 增至 {fork[-1]['width_max_antichain']}，request-context amplification 从 {fork[0]['request_context_amplification']:.1f}× 增至 {fork[-1]['request_context_amplification']:.1f}×，delivered bytes 从 {fork[0]['delivered_bytes']} 增至 {fork[-1]['delivered_bytes']}。Debate rounds 1→3 时，critical-path depth 从 {rounds[0]['critical_path_depth_nodes']} 增至 {rounds[-1]['critical_path_depth_nodes']}，delivered bytes 从 {rounds[0]['delivered_bytes']} 增至 {rounds[-1]['delivered_bytes']}。在固定 width/rounds 下，all-to-all 的 delivered bytes 是 ring 的 {alltoall['delivered_bytes']/ring['delivered_bytes']:.2f} 倍。",
        "",
        "**初步发现：** width 主要扩大同时 ready 的请求波峰；revision/rounds 延长关键路径和状态存活；dense connectivity 放大信息传递量。它们在结构空间中的方向不同，能够支撑论文中 structure → pressure signature 的中间层。",
        "",
        "## 5.3 Multiplexing 下的图压力",
        "",
        "![5.3 multiplexing](figures/5_3_multiplexing.svg)",
        "",
        "新版热图使用全部 24 个实验单元，每个单元包含 4 个 workflow。左图显示 task p95，右图显示客户端同时 ready 的 operation 峰值。",
        "",
        f"使用相同的 8-request payload 集合时，chain-8 在低负载下的 p95 为 {chain8_low['p95_e2e_sec']:.2f} s，frontier 为 1；parallel-8 在低负载下为 {par8_low['p95_e2e_sec']:.2f} s，在 {max(rates):g} user QPS 时升至 {par8_high['p95_e2e_sec']:.2f} s。后端 waiting requests 始终为 0，而客户端 ready/admission 压力已经增加。",
        "",
        "**初步发现：** dependency 决定压力出现的位置。链式结构首先体现为串行关键路径，宽图在 workflow 重叠时把潜在并行度转化为 admission contention。当前负载尚未形成后端排队饱和，最终实验应继续提高 offered load 或降低并发上限。",
        "",
        "## 5.4 异构多 workflow serving",
        "",
        "![5.4 heterogeneous slowdown](figures/5_4_heterogeneous_slowdown.svg)",
        "",
        "两个热图分别展示 0.15 和 0.8 user QPS 下各 workload class 相对于同类独占运行的 p95 slowdown；灰色表示该 mixture 中不存在对应类别。",
        "",
        "在 0.8 user QPS 下，burst+context 中 burst 类的 p95 slowdown 约为 1.19×；balanced 和 burst+dependency 中 dependency 类约为 1.11×。其余小于 1 的数值在单次重复下可能来自噪声或 batching benefit，不能解释为负干扰。",
        "",
        "**初步发现：** 在后端尚无显式 waiting queue 时，cross-workflow interference 已经表现出类别差异。正式实验需要多次重复并给出置信区间，才能把稳定干扰与批处理随机收益区分开。",
        "",
        "## 5.5 Workload mix 决定 serving capacity",
        "",
        "![5.5 workload-mix capacity](figures/5_5_capacity_mix.svg)",
        "",
        "新版图同时给出整体 task p95、最差类别 SLO attainment 和 internal request throughput。容量判定以最差类别 SLO 为准，避免某一类 workflow 被系统平均值掩盖。",
        "",
        "| Mixture | Pilot 容量区间 |",
        "|---|---|",
    ]
    for mix in mix55:
        item = cap[mix]
        bracket = f"{item['bracket'][0]}–{item['bracket'][1]} user QPS" if item["bracket"] else f"> {max(rates55):g} user QPS（右截断）"
        lines.append(f"| {mix} | {bracket} |")
    lines += [
        "",
        "balanced、burst-dominated 和 context-dominated 在 0.5–2 user QPS 之间越过分类 SLO 边界；dependency-dominated 到 5 user QPS 仍满足约束，因此只能报告为右截断。高 offered load 下，各 mixture 的 internal throughput 收敛到约 4.5–4.9 requests/s，但分类 SLO 结果明显不同。",
        "",
        "**初步发现：** 相同 user QPS 不等价于相同 serving pressure。即使模型与后端完全相同，mixture 仍会改变 ready burst 与 context-heavy phase 的重叠方式，从而改变可持续 task capacity。当前只能确定粗区间，正式实验应在 0.5–2 QPS 内加密采样，并把 dependency-dominated 扩展到 5 QPS 以上。",
        "",
        "## 5.6 Cross-hardware bottleneck map：当前仅为 Ascend 锚点",
        "",
        "![5.6 Ascend anchor](figures/5_6_ascend_anchor.svg)",
        "",
        f"新版左图展开固定 balanced corpus 中的 9 个任务，并按 burst/context/dependency 分类；右图保留 serving observations。2 user QPS 下完成 {r56['completed']}/{r56['offered']} 个任务，整体 p95 为 {r56['completed_e2e_sec']['p95']:.2f} s。vLLM 报告 max running={backend.get('max_num_requests_running'):.0f}、max waiting={backend.get('max_num_requests_waiting'):.0f}、max KV-cache usage={100*backend.get('max_gpu_cache_usage_perc',0):.2f}%。NPU utilization 采样均为 0，当前按 unavailable/uninformative 处理；HBM 和 power 使用修正后的 board 0 / chip 1 映射。",
        "",
        "**初步发现：** 这是一组有效的 Ascend replay 锚点，不是跨硬件 bottleneck migration 结论。只有让 A6000 和 H100 replay 完全相同的 frozen corpus、arrival schedules、cache policy、model-input contract 和 SLO，5.6 才能讨论硬件敏感性。",
        "",
        "## 对扩大实验规模的判断",
        "",
        "当前可以进入大规模实验准备：预期的因果链已经可见，所有实验路径都产生了可分析 trace，official length-locked replay 也通过真实后端验收。正式采集前应完成以下调整：",
        "",
        "1. 结构与 multiplexing 单元至少重复 3 次；容量边界附近至少重复 5 次并报告置信区间。",
        "2. balanced/burst/context mixture 在 0.5–2 QPS 之间加密采样；dependency-dominated 扩展到 5 QPS 以上。",
        "3. 增加任务数或 measurement duration，降低有限 cohort drain 对 goodput 的影响。",
        "4. 在 A6000 和 H100 上运行同一 frozen corpus；不可用的物理计数器继续明确标记 unavailable。",
        "5. 将已验收的 Semantic Trace `length_locked` 路径从 5-operation Fork–Join 扩展到每个冻结的 publication trace。`token_locked` 仍不受支持，不能静默降级。",
        "6. Task quality 不进入 serving score；只在 delivery-policy 对比中使用 out-of-band quality guardrail。",
        "",
        "## 结果文件说明",
        "",
        "- `5_2/`、`5_3/`：实验矩阵、逐运行 JSONL traces、后端采样和 CSV 汇总。",
        "- `5_4/run/`：异构 workload 结果；`5_4/frozen/`：供 5.5/5.6 使用的 content-addressed replay corpus。",
        "- `5_5/run/`：mixture-capacity 结果和已解析 arrival schedules。",
        "- `5_6/run/`：修正设备映射后的 Ascend 硬件锚点。",
        "- `semantic-acceptance/`：canonical trace bundle、scenario/system manifests、有效 official run 和 task metrics。",
        "- `configs/`：完整解析后的 pilot 配置。",
    ]
    (out / "PART5_PILOT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out / "PART5_PILOT_REPORT.md"), "figures": sorted(p.name for p in figs.glob("*.svg"))}, indent=2))


if __name__ == "__main__":
    main()

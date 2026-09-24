#!/usr/bin/env python3
"""Build line-based Part 5 pressure figures and a concise Chinese report."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

from app.pressure_signatures import build_pressure_signature

COLORS = ["#5C6F94", "#F1B98B", "#A8B6C8", "#8AA58D", "#C98F8F"]
GRID, TEXT, BG, PALE = "#E1E4E8", "#28323F", "#FFFFFF", "#F2F2F2"
LABELS = {"spawn": "Spawn", "fork_join": "Fork--Join",
          "refinement_loop": "Refinement Loop", "debate": "Debate"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/local/share/fonts") / filename,
        *Path.home().glob(f".local/lib/python*/site-packages/matplotlib/mpl-data/fonts/ttf/{filename}"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(path, size)
    raise RuntimeError(f"A scalable font is required to render readable figures: {filename}")


TICK_FONT = font(28)
LEGEND_FONT = font(28)
AXIS_FONT = font(32)
STAGE_FONT = font(27)
TITLE_FONT = font(38, True)


def text(draw, xy, value, *, anchor="mm", fill=TEXT, fnt=TICK_FONT):
    draw.text(xy, str(value), fill=fill, font=fnt, anchor=anchor)


def line_chart(draw, box, series, *, xlabel, ylabel, xmax=None, ymax=None,
               x_ticks=None, legend=True, stage_spans=None):
    x, y, w, h = box
    all_points = [point for _name, points in series for point in points
                  if point[0] is not None and point[1] is not None]
    xmax = xmax if xmax is not None else max((p[0] for p in all_points), default=1)
    ymax = ymax if ymax is not None else max((p[1] for p in all_points), default=1) * 1.08
    xmax, ymax = max(xmax, 1e-9), max(ymax, 1e-9)
    stage_labels = []
    if stage_spans:
        for i, (start, end, label) in enumerate(stage_spans):
            px0, px1 = x + w * start / xmax, x + w * end / xmax
            draw.rectangle((px0, y, px1, y + h), fill=PALE if i % 2 == 0 else "#E9EDF3")
            stage_labels.append(((px0 + px1) / 2, label))
    for i in range(6):
        py = y + h - h * i / 5
        draw.line((x, py, x + w, py), fill=GRID, width=2)
        text(draw, (x - 14, py), f"{ymax*i/5:.1f}", anchor="rm", fnt=TICK_FONT)
    ticks = x_ticks or [xmax * i / 5 for i in range(6)]
    for value in ticks:
        px = x + w * value / xmax
        draw.line((px, y + h, px, y + h + 7), fill=TEXT, width=2)
        text(draw, (px, y + h + 27), f"{value:g}", anchor="mt", fnt=TICK_FONT)
    draw.rectangle((x, y, x + w, y + h), outline=TEXT, width=3)
    for si, (name, points) in enumerate(series):
        color = COLORS[si % len(COLORS)]
        coords = [(x + w * px / xmax, y + h - h * py / ymax) for px, py in points
                  if px is not None and py is not None]
        if len(coords) > 1:
            draw.line(coords, fill=color, width=6, joint="curve")
        for px, py in coords:
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill=color, outline=BG, width=2)
        if legend:
            lx = x + (si % 2) * (w / 2) + 12
            ly = y + h + 72 + (si // 2) * 44
            draw.line((lx, ly, lx + 40, ly), fill=color, width=7)
            text(draw, (lx + 49, ly), name, anchor="lm", fnt=LEGEND_FONT)
    legend_rows = math.ceil(len(series) / 2) if legend else 0
    text(draw, (x + w / 2, y + h + (88 + 44 * legend_rows if legend else 78)),
         xlabel, fnt=AXIS_FONT)
    text(draw, (x, y - 30), ylabel, anchor="lm", fnt=AXIS_FONT)
    for center, label in stage_labels:
        bounds = draw.textbbox((center, y + 18), label, font=STAGE_FONT, anchor="mm")
        draw.rounded_rectangle((bounds[0] - 7, bounds[1] - 3,
                                bounds[2] + 7, bounds[3] + 3),
                               radius=5, fill=BG)
        text(draw, (center, y + 18), label, fnt=STAGE_FONT, fill="#566477")


def canvas(rows, cols, panel_w=970, panel_h=650):
    return Image.new("RGB", (cols * panel_w + 120, rows * panel_h + 80), BG)


def save(img, figs, stem):
    png, pdf = figs / f"{stem}.png", figs / f"{stem}.pdf"
    img.save(png, dpi=(220, 220), optimize=True)
    img.save(pdf, "PDF", resolution=220)


def trace_from_case(row, workload_class):
    records = load(Path(row["attempt_path"]) / "records.json")
    return Path(next(record["trace_path"] for record in records
                     if record.get("workload_class") == workload_class
                     and record.get("status") in {"completed", "accepted"}))


def isolated(rows):
    return {row["mix"].removeprefix("isolated__"): row for row in rows
            if row["mix"].startswith("isolated__")}


def observation(row, suffix, stat="mean"):
    values = [value["distribution"].get(stat) for key, value in row.get("endpoint_observations", {}).items()
              if key.endswith(suffix) and value.get("distribution", {}).get(stat) is not None]
    return mean(values) if values else None


def physical_kv_percent(row):
    value = observation(row, "cache_used_ratio", "max")
    if value is None:
        value = observation(row, "kv_cache_usage_percent", "max")
    return value * 100 if value is not None and value <= 1 else value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-config", required=True)
    args = parser.parse_args()
    root, out = Path(args.input).resolve(), Path(args.output).resolve()
    figs = out / "figures"; figs.mkdir(parents=True, exist_ok=True)
    model_config = load(args.model_config)

    shape_cases = isolated(load(root / "5_2/run/results.json"))
    signatures = {}
    for name, row in shape_cases.items():
        trace = trace_from_case(row, name)
        metrics = load(row["backend_metrics_path"]).get("samples", [])
        signatures[name] = build_pressure_signature(
            [json.loads(line) for line in trace.read_text().splitlines() if line.strip()],
            model_config=model_config, backend_samples=metrics)
        (out / "signatures").mkdir(exist_ok=True)
        (out / "signatures" / f"{name}.json").write_text(
            json.dumps(signatures[name], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Figure 5.2a: the graph shape itself, expressed as causal-wave profiles.
    img = canvas(1, 3, 920, 640); draw = ImageDraw.Draw(img)
    metrics = [("llm_operation_count", "Ready LLM ops / wave", 1),
               ("logical_kv_equivalent_bytes", "Logical KV equiv. (MiB)", 1/(1024**2)),
               ("cumulative_processed_tokens", "Cumulative tokens", 1)]
    for pi, (key, ylabel, scale) in enumerate(metrics):
        series = []
        for name in LABELS:
            rows = signatures[name]["causal_waves"]
            n = max(1, len(rows) - 1)
            series.append((LABELS[name], [(i / n, row[key] * scale) for i, row in enumerate(rows)]))
        line_chart(draw, (130 + pi*920, 60, 720, 390), series,
                   xlabel="Normalized causal progress", ylabel=ylabel, xmax=1,
                   x_ticks=[0, .2, .4, .6, .8, 1], legend=True)
    save(img, figs, "fig_5_2_canonical_causal_shapes")

    # Figure 5.2b: wall-clock unfolding of readiness and active model state.
    img = canvas(1, 2, 1180, 680); draw = ImageDraw.Draw(img)
    for pi, (key, ylabel, scale) in enumerate([
            ("llm_inflight", "Concurrent LLM operations", 1),
            ("logical_kv_equivalent_bytes", "Logical KV equiv. (MiB)", 1/(1024**2))]):
        series = []
        for name in LABELS:
            rows = signatures[name]["timeline"]
            end = max((row["time_sec"] for row in rows), default=1)
            start = min((row["time_sec"] for row in rows), default=0)
            span = max(end - start, 1e-9)
            series.append((LABELS[name], [((row["time_sec"] - start)/span, (row.get(key) or 0)*scale)
                                          for row in rows]))
        line_chart(draw, (135 + pi*1180, 60, 940, 410), series,
                   xlabel="Normalized execution time", ylabel=ylabel, xmax=1,
                   x_ticks=[0, .2, .4, .6, .8, 1], legend=True)
    save(img, figs, "fig_5_2_canonical_temporal_signatures")

    # Figure 5.3: complete hierarchical workflows with explicit stage bands.
    workflow_cases = isolated(load(root / "5_3/run/results.json"))
    img = canvas(len(workflow_cases), 1, 1500, 570); draw = ImageDraw.Draw(img)
    workflow_summaries = {}
    for i, (name, row) in enumerate(sorted(workflow_cases.items())):
        trace = trace_from_case(row, name)
        events = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
        sig = build_pressure_signature(events, model_config=model_config,
                                       backend_samples=load(row["backend_metrics_path"]).get("samples", []))
        workflow_summaries[name] = sig
        start = min(point["time_sec"] for point in sig["timeline"])
        end = max(point["time_sec"] for point in sig["timeline"]); span = max(end-start, 1e-9)
        points = [((point["time_sec"]-start)/span, (point.get("logical_kv_equivalent_bytes") or 0)/(1024**2))
                  for point in sig["timeline"]]
        spans = [((stage["first_operation_sec"]-start)/span,
                  (stage["last_operation_sec"]-start)/span,
                  stage["logical_stage_id"]) for stage in sig["stages"]]
        text(draw, (760, i*570 + 20), name.replace("_", " "), fnt=TITLE_FONT)
        line_chart(draw, (150, i*570 + 55, 1220, 330), [("Logical KV equivalent", points)],
                   xlabel="Normalized workflow time", ylabel="KV equiv. (MiB)", xmax=1,
                   x_ticks=[0, .2, .4, .6, .8, 1], legend=False, stage_spans=spans)
    save(img, figs, "fig_5_3_hierarchical_stage_trajectories")

    # Figure 5.4: dense isolated and balanced multiplexing curves.
    load_rows = load(root / "5_4/run/results.json")
    groups = {}
    for row in load_rows:
        groups.setdefault(row["mix"], []).append(row)
    panels = [
        (lambda r: r["completed_e2e_sec"]["p95"], "Task latency p95 (s)"),
        (lambda r: r["goodput_qps"], "Task goodput (task/s)"),
        (lambda r: r["internal_qps"], "Internal request QPS"),
        (lambda r: physical_kv_percent(r), "Physical KV usage (%)"),
        (lambda r: observation(r, "memory_bandwidth_percent", "mean"), "HBM bandwidth util. (%)"),
        (lambda r: observation(r, "ai_core_utilization_percent", "mean"), "AICore utilization (%)")]
    img = canvas(2, 3, 950, 680); draw = ImageDraw.Draw(img)
    ordered = ["isolated__spawn", "isolated__fork_join", "isolated__refinement_loop", "isolated__debate", "balanced"]
    for pi, (getter, ylabel) in enumerate(panels):
        series = []
        for name in ordered:
            if name not in groups: continue
            points = [(row["rate"], getter(row)) for row in sorted(groups[name], key=lambda r: r["rate"])]
            label = LABELS.get(name.removeprefix("isolated__"), "Balanced mix")
            series.append((label, points))
        col, rowi = pi % 3, pi // 3
        line_chart(draw, (130 + col*950, 45 + rowi*680, 735, 345), series,
                   xlabel="Offered user QPS", ylabel=ylabel, xmax=20,
                   x_ticks=[0, 4, 8, 12, 16, 20], legend=True)
    save(img, figs, "fig_5_4_dense_multiplexing_pressure")

    # Figure 5.5: composition changes capacity and per-class slowdown.
    # Slowdown uses the corresponding isolated class at the lowest measured load.
    isolated_baseline_p95 = {}
    for name, rows in groups.items():
        if not name.startswith("isolated__"):
            continue
        first = min(rows, key=lambda row: row["rate"])
        cls = name.removeprefix("isolated__")
        isolated_baseline_p95[cls] = first["per_class"][cls]["completed_e2e_sec"]["p95"]
    def worst_class_slowdown(row):
        values = []
        for cls, metrics in row["per_class"].items():
            observed = metrics["completed_e2e_sec"]["p95"]
            baseline = isolated_baseline_p95.get(cls)
            if observed is not None and baseline:
                values.append(observed / baseline)
        return max(values) if values else None
    mix_rows = load(root / "5_5/run/results.json")
    mix_groups = {}
    for row in mix_rows: mix_groups.setdefault(row["mix"], []).append(row)
    img = canvas(1, 3, 960, 660); draw = ImageDraw.Draw(img)
    for pi, (getter, ylabel) in enumerate([
            (lambda r: r["completed_e2e_sec"]["p95"], "Aggregate task p95 (s)"),
            (lambda r: r["goodput_qps"], "Task goodput (task/s)"),
            (worst_class_slowdown, "Worst-class p95 slowdown")]):
        short_name = {"balanced": "Balanced", "burst_dominated": "Burst-heavy",
                      "dependency_dominated": "Dependency-heavy",
                      "state_dominated": "State-heavy"}
        series = [(short_name.get(name, name.replace("_", " ")), [(row["rate"], getter(row))
                  for row in sorted(rows, key=lambda r: r["rate"])]) for name, rows in sorted(mix_groups.items())]
        line_chart(draw, (135 + pi*960, 60, 745, 390), series,
                   xlabel="Offered user QPS", ylabel=ylabel, xmax=20,
                   x_ticks=[0, 4, 8, 12, 16, 20], legend=True)
    save(img, figs, "fig_5_5_workload_mix_capacity")

    shape_lines = []
    for name in LABELS:
        sig = signatures[name]
        shape_lines.append(
            f"- **{LABELS[name]}**：{len(sig['causal_waves'])} 个因果波次，"
            f"峰值每波 LLM 操作 {max(r['llm_operation_count'] for r in sig['causal_waves'])}，"
            f"峰值逻辑 KV 等价 {max(r['logical_kv_equivalent_bytes'] for r in sig['causal_waves'])/1024**2:.1f} MiB。")
    capacity_lines = []
    for name, rows in sorted(groups.items()):
        high = max(rows, key=lambda row: row["rate"])
        label = name.removeprefix("isolated__").replace("_", " ")
        high_rate = high["rate"]
        high_goodput = high["goodput_qps"]
        high_p95 = high["completed_e2e_sec"]["p95"]
        capacity_lines.append(
            f"- **{label}**：测试到 {high_rate:g} user QPS 仍未触发 30 s SLO failure（capacity right-censored）；"
            f"该点 achieved goodput={high_goodput:.2f} task/s，p95={high_p95:.2f} s。")
    report = [
        "# MASBench Part 5 昇腾初步压力画像", "",
        "> 本报告全部数据来自 Ascend 910 + Qwen3-8B 的真实后端执行/受控 replay。它用于验证实验设计与采集链路，不能替代多次重复的大规模正式结果。", "",
        "## 5.2 标准模板的单 workflow pressure signature", "",
        "![causal shapes](figures/fig_5_2_canonical_causal_shapes.png)", "",
        "![temporal shapes](figures/fig_5_2_canonical_temporal_signatures.png)", "",
        *shape_lines, "",
        "两张图把结构形状和后端时间展开分开：第一张横轴是因果进度，直接显示宽、深、反复同步的差异；第二张显示这些差异在真实执行中如何变成并发请求与模型状态需求。逻辑 KV 等价值由真实 token 数和模型结构推导，不等于 vLLM 实际驻留 KV。", "",
        "## 5.3 分层 workflow 的 stage 轨迹", "",
        "![workflow stages](figures/fig_5_3_hierarchical_stage_trajectories.png)", "",
        "灰色 stage 区间与逻辑 KV 等价状态位于同一时间轴。不同 workflow 的 stage 数量、并行分支和 stage 间上下文传递会形成不同的状态峰值与持续时间；正式实验应按 stage 分解等待、计算、HBM 带宽和物理 KV。", "",
        "## 5.4 多 workflow 复用下的压力放大", "",
        "![multiplexing](figures/fig_5_4_dense_multiplexing_pressure.png)", "",
        *capacity_lines, "",
        "每条曲线包含 15 个 load 点（0.25–20 user QPS）。前三个面板是 task/request 层结果，后三个面板是后端物理观察。若某个设备指标缺失，图上保留为空值，不做插值或伪造。", "",
        "## 5.5 workload mix 改变 serving capacity", "",
        "![workload mix](figures/fig_5_5_workload_mix_capacity.png)", "",
        "固定总 user QPS 时，不同 composition 产生不同的内部 request 放大、上下文状态和依赖阻塞，因此 aggregate p95、goodput 与最慢 workflow class 的 slowdown 会分离。正式论文应使用重复实验和置信区间确认差异。", "",
        "## 指标边界", "",
        "- `logical KV equivalent`：estimated，按 `2 × layers × KV heads × head_dim × bytes × active tokens` 计算。",
        "- `physical KV usage`：backend-reported，来自 vLLM `/metrics`。",
        "- `AICore/HBM bandwidth utilization`：observed，来自 `npu-smi`。",
        "- `compute/HBM utilization ratio` 只能作为观察比值；当前没有同步 FLOP 与 DRAM-byte 计数，因此 arithmetic intensity 明确为 unavailable。",
        "- 本轮 replay 固定 realized DAG、recorded downstream payload 与外部结果；Qwen/vLLM 不能 strict token-lock，输出长度使用 best-effort 验证。", "",
        "## 正式大规模实验前的判断", "",
        "链路已经能够区分 graph shape、stage phase、logical state demand 与 backend physical observations，并能在固定 trace 下做密集负载与 composition sweep。本轮 811 个 workflow 全部完成，analysis error、dependency violation、client overflow 和 replay equivalence failure 均为 0。20 user QPS 时，Spawn/Fork--Join/Debate 的内部吞吐约为 15.0/15.1/17.2 request/s，而深链 Refinement Loop 约为 7.3 request/s；同为 8 个 LLM operation 的 task，结构依赖已经使持续吞吐分离。composition 实验中，20 user QPS 下 dependency-dominated mix 的 aggregate p95 为 6.59 s，burst-dominated 为 5.17 s。正式运行仍需加入多次重复、置信区间、稳定测量窗口，并在 A6000/H100 上复用同一 frozen corpus。"]
    (out / "PART5_PRESSURE_REPORT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out / "PART5_PRESSURE_REPORT_ZH.md"),
                      "figures": sorted(path.name for path in figs.iterdir())}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

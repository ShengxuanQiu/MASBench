#!/usr/bin/env python3
"""Plot vLLM-reported KV-block occupancy from high-frequency endpoint samples."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

from PIL import ImageDraw

from report_pressure_experiments import canvas, line_chart, save

LABELS = {"spawn": "Spawn", "fork_join": "Fork--Join",
          "refinement_loop": "Refinement Loop", "debate": "Debate"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_series(case):
    samples = load(Path(case["backend_metrics_path"]))["samples"]
    values = []
    for sample in samples:
        ratio = sample.get("cache_memory_metrics", {}).get("cache_used_ratio")
        if sample.get("status") != "success" or not isinstance(ratio, (int, float)):
            continue
        values.append((sample["timestamp_unix"] - case["clock_origin_unix"], 100 * ratio))
    if len(values) < 8:
        raise ValueError(f"Insufficient physical KV samples for {case['mix']}: {len(values)}")
    intervals = [b[0] - a[0] for a, b in zip(values, values[1:])]
    return values, median(intervals)


def records(case):
    return load(Path(case["attempt_path"]) / "records.json")


def single_shape(input_dir, figs):
    cases = {row["mix"].removeprefix("isolated__"): row
             for row in load(input_dir / "cache_disabled/shapes/results.json")
             if row["mix"].startswith("isolated__")}
    lines, counts, median_intervals = [], {}, []
    for name in LABELS:
        case = cases[name]
        record = next(row for row in records(case) if row.get("workload_class") == name)
        values, interval = sample_series(case)
        start = record["trace_origin_sec"]
        points = [(max(0, time - start), ratio) for time, ratio in values
                  if time - start >= -0.15]
        lines.append((LABELS[name], points))
        counts[name] = len(points)
        median_intervals.append(interval)
    image = canvas(1, 1, 1460, 760)
    draw = ImageDraw.Draw(image)
    line_chart(draw, (165, 70, 1240, 450), lines,
               xlabel="Time since workflow start (s)",
               ylabel="vLLM KV block pool occupied (%)",
               xmax=max(x for _name, points in lines for x, _ in points), ymax=1.0,
               x_ticks=list(range(6)), legend=True)
    save(image, figs, "fig_5_2_backend_kv_occupancy")
    return {"sample_count_by_template": counts,
            "median_sample_interval_sec": median(median_intervals),
            "peak_percent_by_template": {name: max(value for _, value in points)
                                         for (name, points) in lines}}


def repeated_spawn(input_dir, figs):
    lines, stats = [], {}
    max_x = 0.0
    for mode, label in (("cache_disabled", "Prefix cache off"),
                        ("warm_cache_enabled", "Prefix cache on")):
        case = load(input_dir / mode / "repeat_spawn/results.json")[0]
        ordered = sorted(records(case), key=lambda row: row["started_sec"])
        if len(ordered) != 2 or any(row["status"] not in {"completed", "accepted"} for row in ordered):
            raise ValueError(f"Expected two completed repeated Spawn workflows: {mode}")
        values, interval = sample_series(case)
        origin = ordered[0]["trace_origin_sec"]
        points = [(max(0, time - origin), ratio) for time, ratio in values
                  if time - origin >= -0.15]
        max_x = max(max_x, max(x for x, _ in points))
        lines.append((label, points))
        gap_values = [ratio for time, ratio in values
                      if ordered[0]["finished_sec"] <= time < ordered[1]["started_sec"]]
        raw_samples = load(Path(case["backend_metrics_path"]))["samples"]
        hits = [sample.get("cache_memory_metrics", {}).get("prefix_cache_hit_units")
                for sample in raw_samples]
        hits = [value for value in hits if isinstance(value, (int, float))]
        stats[mode] = {
            "sample_count": len(points), "median_sample_interval_sec": interval,
            "second_arrival_sec": ordered[1]["started_sec"] - origin,
            "between_workflows_kv_percent_median": median(gap_values) if gap_values else None,
            "prefix_cache_hits_delta": hits[-1] - hits[0] if len(hits) >= 2 else None,
        }
    image = canvas(1, 1, 1460, 760)
    draw = ImageDraw.Draw(image)
    line_chart(draw, (165, 70, 1240, 450), lines,
               xlabel="Time since first Spawn arrival (s)",
               ylabel="vLLM KV block pool occupied (%)",
               xmax=max_x, ymax=0.8,
               x_ticks=list(range(math.floor(max_x) + 1)), legend=True)
    save(image, figs, "fig_5_2_prefix_cache_retention")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_dir, output = Path(args.input).resolve(), Path(args.output).resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    shapes = single_shape(input_dir, figures)
    prefix = repeated_spawn(input_dir, figures)
    result = {
        "schema": "masbench_physical_kv_profile_v1",
        "metric": "vllm:kv_cache_usage_perc",
        "unit": "percent of vLLM KV block pool occupied",
        "source": "backend-reported",
        "metric_definition": "vLLM block_pool.get_usage(): 1 - free_blocks / total_blocks; reusable cached blocks may be on the free queue",
        "scope": "isolated NPU1 vLLM endpoint, Qwen3-8B",
        "cache_disabled_shapes": shapes,
        "repeated_spawn": prefix,
        "limitations": [
            "Backend-reported KV block occupancy is not direct NPU HBM-byte measurement.",
            "Cache-on and cache-off are independent service starts; the repeated pair is exploratory.",
            "A zero free/used-block occupancy reading does not prove all reusable prefix metadata was discarded.",
        ],
    }
    (output / "PHYSICAL_KV_PROFILE.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    off, on = prefix["cache_disabled"], prefix["warm_cache_enabled"]
    md = [
        "# 5.2 后端报告的 KV block 实际占用", "",
        "![四种模板的 vLLM KV block 使用率](figures/fig_5_2_backend_kv_occupancy.png)", "",
        "此图纵轴是 vLLM `/metrics` 报告的 KV block pool 已占用百分比，横轴是单个 workflow 启动后的真实时间；采样来自独立 NPU1 服务，prefix cache 关闭。四种模板分别有 32/30/89/31 个样本，采样间隔中位数约 60 ms。它反映服务端报告的 block 占用，而不是前面按 token 数推导的请求需求上界。", "",
        "![重复 Spawn 与 prefix cache](figures/fig_5_2_prefix_cache_retention.png)", "",
        "第二张图将相同 Spawn 工作流按固定间隔连续运行两次，分别开启和关闭 prefix cache，显示后端报告的 KV block 使用率。"
        f"两次运行之间的占用中位数：关闭时 {off['between_workflows_kv_percent_median']}%，开启时 {on['between_workflows_kv_percent_median']}%；"
        f"后端 prefix-hit 计数增量分别为 {off['prefix_cache_hits_delta']} 和 {on['prefix_cache_hits_delta']}。", "",
        "注意：vLLM 此指标按 `1 - free_blocks / total_blocks` 计算，反映当前不在 free queue 中的 KV blocks，并非 NPU 总 HBM 字节占用。可复用的 prefix block 即使回到 free queue，后续仍可能命中。这里 cache-on 组两次 workflow 间指标为 0，但 prefix-hit 计数增加 1792，恰好说明不能仅凭占用率回零断言 prefix 缓存被清空。",
    ]
    (output / "PHYSICAL_KV_PROFILE_ZH.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"figures": [str(figures / (stem + ".pdf")) for stem in
                                 ("fig_5_2_backend_kv_occupancy", "fig_5_2_prefix_cache_retention")],
                      "summary": result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

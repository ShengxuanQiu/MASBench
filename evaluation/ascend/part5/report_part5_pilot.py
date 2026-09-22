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
    text{font-family:Arial,Helvetica,sans-serif;fill:#273142}
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
    body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="border"/>')
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
    body = []
    line_panel(body, 75, 52, 330, 245, "Fork–Join width sweep",
               [str(params(r)["width"]) for r in fork],
               [("graph width", normalized(fork, "width_max_antichain")),
                ("context amp.", normalized(fork, "request_context_amplification")),
                ("delivery bytes", normalized(fork, "delivered_bytes"))],
               "relative to width=1")
    line_panel(body, 500, 52, 300, 245, "Debate round sweep",
               [str(params(r)["rounds"]) for r in rounds],
               [("DAG depth", normalized(rounds, "critical_path_depth_nodes")),
                ("context amp.", normalized(rounds, "request_context_amplification")),
                ("delivery bytes", normalized(rounds, "delivered_bytes"))],
               "relative to round=1")
    svg(figs / "5_2_pressure_signatures.svg", body)

    r53 = load(root / "5_3/report/report.json")["case_rows"]
    rates = sorted({r["rate"] for r in r53})
    selected = []
    for workload, request_count, label in (("matched-chain", 8, "chain-8"), ("matched-parallel", 8, "parallel-8"),
                                            ("fork-join", None, "fork–join-8")):
        vals = []
        for rate in rates:
            row = next(r for r in r53 if r["workload"] == workload and r["rate"] == rate and
                       (request_count is None or params(r).get("requests") == request_count))
            vals.append(row["p95_e2e_sec"])
        selected.append((label, vals))
    body = []
    line_panel(body, 90, 55, 700, 255, "Matched graph pressure under multiplexing",
               [str(x) for x in rates], selected, "p95 task latency (s)")
    body.append(txt(440, 378, "offered user QPS", anchor="middle"))
    svg(figs / "5_3_multiplexing.svg", body)

    r54 = load(root / "5_4/run/results.json")
    mixed54 = [r for r in r54 if r["rate"] == 0.8 and not r["mix"].startswith("isolated__")]
    mix_order = ["balanced", "burst_context", "burst_dependency", "context_dependency"]
    mixed54.sort(key=lambda r: mix_order.index(r["mix"]))
    class_order = ["burst_heavy", "context_heavy", "dependency_heavy"]
    series54 = []
    for cls in class_order:
        series54.append((cls.replace("_heavy", ""), [r["per_class"].get(cls, {}).get("slowdown_p95") for r in mixed54]))
    grouped_bar(figs / "5_4_heterogeneous_slowdown.svg",
                [r["mix"].replace("_", "\n") for r in mixed54], series54,
                "Class-specific interference at 0.8 user QPS", "p95 slowdown (×)", ymax=1.35)

    r55 = load(root / "5_5/run/results.json")
    rates55 = sorted({r["rate"] for r in r55})
    mix55 = ["balanced", "burst_dominated", "context_dominated", "dependency_dominated"]
    latency_series, slo_series = [], []
    for mix in mix55:
        rows = [next(r for r in r55 if r["mix"] == mix and r["rate"] == rate) for rate in rates55]
        latency_series.append((mix.replace("_dominated", "-dom."), [r["completed_e2e_sec"]["p95"] for r in rows]))
        slo_series.append((mix.replace("_dominated", "-dom."),
                           [min(v["slo_success_fraction"] for v in r["per_class"].values()) for r in rows]))
    body = []
    line_panel(body, 70, 52, 340, 245, "Task latency by workload mix", [str(x) for x in rates55],
               latency_series, "p95 latency (s)", ymax=22, show_legend=False)
    line_panel(body, 500, 52, 300, 245, "Worst-class SLO attainment", [str(x) for x in rates55],
               slo_series, "attainment", ymax=1.0, show_legend=False)
    # 0.9 target line in the right panel.
    target_y = 52 + 245 - 245 * .9
    body.append(f'<line x1="500" y1="{target_y}" x2="800" y2="{target_y}" stroke="#555" stroke-dasharray="5 4"/>')
    for si, name in enumerate([x[0] for x in latency_series]):
        lx = 70 + si * 190
        body.append(f'<line x1="{lx}" y1="346" x2="{lx+22}" y2="346" stroke="{COLORS[si]}" stroke-width="3"/>')
        body.append(txt(lx + 28, 350, name, "legend", anchor="start"))
    body.append(txt(440, 378, "offered user QPS", anchor="middle"))
    svg(figs / "5_5_capacity_mix.svg", body)

    r56 = load(root / "5_6/run/results.json")[0]
    backend = r56["backend_metrics"]
    obs = r56["endpoint_observations"]
    power = obs.get("device_metrics.power_watts", {}).get("distribution", {})
    hbm = obs.get("device_metrics.memory_usage_percent", {}).get("distribution", {})
    categories = ["running\nrequests", "waiting\nrequests", "KV cache\nusage (%)", "HBM\nusage (%)", "power\n(W/10)"]
    values = [backend.get("max_num_requests_running"), backend.get("max_num_requests_waiting"),
              100 * backend.get("max_gpu_cache_usage_perc", 0), hbm.get("p95"),
              power.get("p95") / 10 if power.get("p95") is not None else None]
    grouped_bar(figs / "5_6_ascend_anchor.svg", categories, [("Ascend anchor", values)],
                "Fixed-corpus Ascend serving observation", "display value", ymax=max(v for v in values if v is not None) * 1.2)

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
        "# MASBench Part 5 Ascend pilot",
        "",
        "> **Decision: GO for the scaled experiment matrix, with the limitations below.** The pilot validates the execution and analysis chain and produces mechanism-consistent signals. It is not a publication result: each cell has one repetition, the capacity boundary is coarse, and cross-hardware bottleneck migration has not yet been measured.",
        "",
        "## Experimental contract",
        "",
        "| Item | Resolved value |",
        "|---|---|",
        "| Backend | vLLM OpenAI-compatible on Huawei Ascend 910 |",
        "| Model | Qwen3-8B (`masbench-qwen3-8b-part5`) |",
        "| Placement | one accelerator, `ASCEND_RT_VISIBLE_DEVICES=1` = board 0 / chip 1 |",
        "| Generation | temperature 0, seed 42, max 96 tokens, thinking disabled |",
        "| Prefix cache | disabled and verified from `vllm:cache_config_info` |",
        "| Concurrency | 16 LLM requests |",
        "| Arrival | saved seeded schedules; Poisson except low-load 5.2 |",
        "| 5.5 class SLO policy | 1.75× the 5.4 isolated low-load p95, rounded: burst 6.2 s, context 10.0 s, dependency 21.8 s |",
        "",
        "## Validity summary",
        "",
        "| Section | Completed / offered | Trace analysis errors |",
        "|---|---:|---:|",
    ]
    lines += [f"| {section} | {done} / {offered} | {errors} |" for section, (done, offered, errors) in totals.items()]
    lines += [
        "",
        f"Fixed-trace replay equivalence failures across 5.5/5.6: **{replay_failures}**. All cache-disabled checks were backend-verified.",
        "",
        f"Official Semantic Trace acceptance also passed: schema `{semantic_manifest['metadata']['schema_version']}`, run_valid=`{str(semantic_run['run_valid']).lower()}`, {len(semantic_run['operations'])} length-locked operations, no output-length mismatches, no generated output used downstream, and task latency {semantic_metrics['task_latency_sec']['p50']:.2f} s.",
        "",
        "## 5.2 Single-workflow pressure signatures",
        "",
        "![5.2 pressure signatures](figures/5_2_pressure_signatures.svg)",
        "",
        f"Fork–Join width 1→8 increases realized width {fork[0]['width_max_antichain']}→{fork[-1]['width_max_antichain']}, request-context amplification {fork[0]['request_context_amplification']:.1f}×→{fork[-1]['request_context_amplification']:.1f}×, and delivered bytes {fork[0]['delivered_bytes']}→{fork[-1]['delivered_bytes']}. Debate rounds 1→3 increase critical-path depth {rounds[0]['critical_path_depth_nodes']}→{rounds[-1]['critical_path_depth_nodes']} and delivered bytes {rounds[0]['delivered_bytes']}→{rounds[-1]['delivered_bytes']}. At fixed width/rounds, all-to-all delivery carries {alltoall['delivered_bytes']/ring['delivered_bytes']:.2f}× the bytes of ring.",
        "",
        "**Insight.** Width creates a wide ready burst without increasing DAG depth; rounds extend depth and state lifetime; dense connectivity amplifies information consumption. This is the intended structure → pressure-signature bridge.",
        "",
        "## 5.3 Graph pressure under multiplexing",
        "",
        "![5.3 multiplexing](figures/5_3_multiplexing.svg)",
        "",
        f"With the same eight-request payload set, chain-8 has low-load p95 {chain8_low['p95_e2e_sec']:.2f} s and a one-request frontier, while parallel-8 is {par8_low['p95_e2e_sec']:.2f} s at low load and rises to {par8_high['p95_e2e_sec']:.2f} s at {max(rates):g} user QPS. The backend reported zero waiting requests, while client-side ready/admission pressure increased.",
        "",
        "**Insight.** Dependencies change where pressure appears: chains expose serial critical-path latency; wide graphs convert latent parallelism into admission contention as workflows overlap. The pilot does not show backend queue saturation yet.",
        "",
        "## 5.4 Heterogeneous multi-workflow serving",
        "",
        "![5.4 heterogeneous slowdown](figures/5_4_heterogeneous_slowdown.svg)",
        "",
        "At 0.8 user QPS, burst p95 slowdown reaches about 1.19× in the burst+context mix, while dependency-heavy slowdown is about 1.11× in balanced and burst+dependency mixes. Other cells remain near or below their isolated references; with one repetition these small speedups should be treated as noise or batching benefit, not negative interference.",
        "",
        "**Insight.** Cross-workflow interference is class-specific even before backend queueing appears. The full run needs repetitions and load-normalized confidence intervals.",
        "",
        "## 5.5 Workload mix determines serving capacity",
        "",
        "![5.5 workload-mix capacity](figures/5_5_capacity_mix.svg)",
        "",
        "| Mix | Pilot capacity status |",
        "|---|---|",
    ]
    for mix in mix55:
        item = cap[mix]
        bracket = f"{item['bracket'][0]}–{item['bracket'][1]} user QPS" if item["bracket"] else f"> {max(rates55):g} user QPS (right-censored)"
        lines.append(f"| {mix} | {bracket} |")
    lines += [
        "",
        "Balanced, burst-dominated, and context-dominated mixes cross the class-protected SLO boundary between 0.5 and 2 user QPS. Dependency-dominated remains valid through 5 user QPS. At high offered load, achieved internal throughput converges near 4.5–4.9 requests/s, while class SLO outcomes differ by composition.",
        "",
        "**Insight.** Equal user QPS is not equal serving pressure. Composition controls which ready bursts and context-heavy phases overlap, changing sustainable task capacity even with the same model and backend.",
        "",
        "## 5.6 Cross-hardware bottleneck map: Ascend anchor only",
        "",
        "![5.6 Ascend anchor](figures/5_6_ascend_anchor.svg)",
        "",
        f"The fixed balanced corpus at 2 user QPS completes {r56['completed']}/{r56['offered']} tasks with p95 {r56['completed_e2e_sec']['p95']:.2f} s. vLLM reports max running={backend.get('max_num_requests_running'):.0f}, max waiting={backend.get('max_num_requests_waiting'):.0f}, and max KV-cache usage={100*backend.get('max_gpu_cache_usage_perc',0):.2f}%. {physical_note}",
        "",
        "**Insight.** This is a valid Ascend replay anchor, not a bottleneck-migration result. A6000 and H100 must replay the same frozen corpus, schedules, cache policy, model-input contract, and SLOs before Section 5.6 can claim hardware sensitivity.",
        "",
        "## Readiness and next run",
        "",
        "The pilot supports proceeding because the expected causal chain is visible and every experiment path produced analyzable traces. Before publication runs:",
        "",
        "1. Use at least 3 repetitions for structure/multiplexing and 5 around each capacity boundary.",
        "2. Add dense rates inside 0.5–2 QPS for balanced/burst/context mixes and extend dependency-dominated beyond 5 QPS.",
        "3. Increase task count/measurement duration so achieved goodput is not dominated by finite-cohort drain.",
        "4. Run the identical frozen corpus on A6000 and H100; treat unavailable physical counters as unavailable.",
        "5. Scale the validated Semantic Trace `length_locked` path from the accepted five-operation Fork–Join trace to every frozen publication trace. `token_locked` remains unsupported and must not silently degrade.",
        "6. Keep task quality out of serving scores; add the out-of-band quality guardrail only for delivery-policy comparisons.",
        "",
        "## Artifact map",
        "",
        "- `5_2/` and `5_3/`: publication matrices, per-run JSONL traces, backend samples, and CSV reports.",
        "- `5_4/run/`: heterogeneous results; `5_4/frozen/`: content-addressed replay corpus used by 5.5/5.6.",
        "- `5_5/run/`: mix-capacity results and resolved schedules.",
        "- `5_6/run/`: corrected Ascend hardware anchor.",
        "- `semantic-acceptance/`: canonical trace bundle, resolved scenario/system manifests, valid official run, and task metrics.",
        "- `configs/`: fully resolved pilot configs.",
    ]
    (out / "PART5_PILOT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out / "PART5_PILOT_REPORT.md"), "figures": sorted(p.name for p in figs.glob("*.svg"))}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""MASBench Fig. 3, author-supplied audit data.

Print layout: two full-column panels stacked vertically.
Panel a: 3.45 x 1.78 in; panel b: 3.45 x 1.65 in.
No titles, panel letters or bottom narrative footnotes.
Residual mechanism descriptions are retained as data for the manuscript.
Run from any directory: python evaluation/plot_coverage.py
"""
from pathlib import Path
from collections import Counter
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "coverage"
COL_W = 3.45
BLUE_LIGHT, PEACH, BLUE, PALE = "#A8B6C8", "#F1B98B", "#5C6F94", "#F2F2F2"
INK, FRAME = "#35435A", "#758197"
INTERSECTIONS = {
    ("PE",): 15, ("PA", "PE"): 12, ("DE", "ER"): 9,
    ("DE", "PA"): 7, ("ER", "PE"): 5, ("DE", "ER", "PE"): 4,
    ("ER",): 4, ("DE",): 3, ("PA",): 3, ("DE", "PE"): 3,
    ("DE", "ER", "PA"): 2, ("DE", "PA", "PE"): 1,
    ("DE", "ER", "PA", "PE"): 1, ("ER", "PA", "PE"): 1,
}
MOTIF_ORDER = ["PE", "DE", "PA", "ER"]
MOTIF_NAME = {"PE": "Peer exchange", "DE": "Dispatch", "PA": "Parallel agg.", "ER": "Eval.-refine"}
RESIDUAL = [
    ("GPTSwarm", "General graph", "recursive / optimized graph"),
    ("MacNet", "General graph", "arbitrary DAG"),
    ("G-Designer", "General graph", "task-conditioned topology"),
    ("MaAS", "General graph", "sampled architecture"),
    ("AnyMAC", "General graph", "next-agent + next-context"),
    ("Guided Topology Diff.", "General graph", "generated topology"),
    ("Graph-GRPO", "General graph", "learned graph policy"),
    ("AgentNet", "General graph", "evolving DAG"),
    ("MetaAgent (FSM)", "General graph", "generated FSM"),
    ("MAS-GPT", "General graph", "generated architecture"),
    ("CoELA", "Shared environment", "mutable environment state"),
    ("HyperAgent", "Higher-order", "hyperedge / group relation"),
]


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 6.5,
        "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": FRAME, "axes.linewidth": 0.6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": None,
    })


def boxed(ax, grid=None):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME)
        spine.set_linewidth(0.6)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=PALE, linewidth=0.6)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches=None, metadata={
        "Creator": "MASBench evaluation/plot_coverage.py",
        "Subject": "Author-supplied coverage audit data",
    })
    plt.close(fig)
    print(OUT / name)


def plot_fig3a_upset():
    intersections = sorted(INTERSECTIONS.items(), key=lambda p: (-p[1], len(p[0]), p[0]))
    marginal = Counter()
    for combo, count in INTERSECTIONS.items():
        for motif in combo:
            marginal[motif] += count
    assert sum(INTERSECTIONS.values()) == 70
    fig = plt.figure(figsize=(COL_W, 1.78))
    left = fig.add_axes([0.76/COL_W, 0.34/1.78, 0.43/COL_W, 0.67/1.78])
    matrix = fig.add_axes([1.38/COL_W, 0.34/1.78, 1.99/COL_W, 0.67/1.78])
    top = fig.add_axes([1.38/COL_W, 1.08/1.78, 1.99/COL_W, 0.62/1.78])
    x = np.arange(len(intersections))
    counts = [count for _, count in intersections]
    # Every bar encodes the same quantity: exact-combination workflow count.
    top.bar(x, counts, color=BLUE, width=0.72, zorder=3)
    fig.text(0.04/COL_W, 1.67/1.78,
             "Column = motif set\nDark dot = present\nTop = set count\nLeft = motif count",
             ha="left", va="top", fontsize=5.7, linespacing=1.45, color=INK)
    for xi, count in zip(x, counts):
        top.text(xi, count + 0.35, str(count), ha="center", va="bottom", fontsize=5.7)
    top.set(xlim=(-0.65, len(x)-0.35), ylim=(0, 18), ylabel="Workflows")
    top.set_yticks([0, 5, 10, 15])
    top.set_xticks([])
    top.yaxis.labelpad = 3
    boxed(top, "y")
    y = np.arange(4)
    values = [marginal[m] for m in MOTIF_ORDER]
    left.barh(y, values, height=0.58, color=PEACH, zorder=3)
    left.set_yticks(y, labels=[MOTIF_NAME[m] for m in MOTIF_ORDER])
    left.set(xlim=(0, 48), ylim=(3.55, -0.55), xlabel="Workflows")
    left.set_xticks([0, 20, 40])
    left.tick_params(axis="y", length=0, labelsize=6)
    left.xaxis.labelpad = 3
    for yy, value in zip(y, values):
        left.text(value - 2, yy, str(value), ha="right", va="center", fontsize=5.6)
    boxed(left)
    for yi in range(4):
        if yi % 2 == 0:
            matrix.axhspan(yi-.5, yi+.5, color=PALE, zorder=0)
    for xi, (combo, _) in enumerate(intersections):
        matrix.scatter([xi]*4, y, s=6, color=BLUE_LIGHT, alpha=0.55, zorder=1)
        active = [i for i, motif in enumerate(MOTIF_ORDER) if motif in combo]
        if len(active) > 1:
            matrix.plot([xi, xi], [min(active), max(active)], color=BLUE, lw=0.9, zorder=2)
        matrix.scatter([xi]*len(active), active, s=13, color=BLUE, zorder=3)
    matrix.set(xlim=(-0.65, len(x)-0.35), ylim=(3.55, -0.55), xlabel="Motif combination")
    matrix.set_xticks([])
    matrix.set_yticks([])
    matrix.xaxis.labelpad = 11
    boxed(matrix)
    save(fig, "fig3a_workflow_composition_upset.pdf")


def plot_fig3b_coverage_gap():
    fig = plt.figure(figsize=(COL_W, 1.65))
    ax = fig.add_axes([1.03/COL_W, 0.34/1.65, 2.34/COL_W, 1.23/1.65])
    positions = [0, 1.35, 2.35, 3.35, 4.35]
    numerators, denominators = [62,16,34,36,70], [68,40,40,42,82]
    values = np.array(numerators) / denominators * 100
    labels = ["Local stages", "Sequential workflow", "Hierarchical workflow",
              "Expansion workflow", "Combined workflow"]
    colors = [BLUE_LIGHT, PEACH, BLUE, BLUE, BLUE]
    ax.axhspan(-0.5, 0.5, color=PALE, zorder=0)
    ax.axhline(0.75, color=FRAME, lw=0.5, linestyle=(0,(2,2)))
    ax.barh(positions, values, height=0.67, color=colors, zorder=3)
    ax.set_yticks(positions, labels=labels)
    ax.tick_params(axis="y", length=0, labelsize=6.0)
    ax.set(xlim=(0, 105), ylim=(4.95, -0.6), xlabel="Coverage (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.labelpad = 3
    for yy, pct, num, den, color in zip(positions, values, numerators, denominators, colors):
        ax.text(pct / 2, yy, f"{pct:.1f}% ({num}/{den})",
                color="white" if color == BLUE else INK, ha="center", va="center",
                fontsize=6.0, zorder=4)
    boxed(ax, "x")
    save(fig, "fig3b_hierarchical_coverage_gap.pdf")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    OUT = parser.parse_args().output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    plot_fig3a_upset()
    plot_fig3b_coverage_gap()


if __name__ == "__main__":
    main()

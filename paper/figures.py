"""Draw the paper's measurement figures from measurements/.

    python -m paper.figures                    # all of them, into paper/figures/
    python -m paper.figures --out DIR --only cross_target_decompositions

Values come from paper/values.py, which reads the archive through the specfloor
reports; this file only draws. slot_priority_trends refits the single-slot
oracle (about half a minute). Figure 1 is a diagram, not data, and is not here.
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from paper import values as V

INK = "#25324A"
BLUE = "#3F73A8"
GREEN = "#4A936A"
TEAL = "#72B7B2"
ORANGE = "#D89562"
PURPLE = "#8B6FA5"
FILL = "#E5F2EA"
GRID = "#E2E8F0"
# decomposition figures: floor, gap and exposure keep one colour each
FLOOR_COLOR, GAP_COLOR, EXPOSURE_COLOR = "#4C78A8", "#72B7B2", "#E07B73"
DECOMP_INK, DECOMP_GRID = "#303846", "#E1E6EB"
LABEL_BOX = {"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none",
             "alpha": 0.78}
SHORT = {"Qwen3-4B": "Q3-4B", "Qwen3-8B": "Q3-8B", "Qwen3-14B": "Q3-14B",
         "Gemma-4-12B": "G4-12B"}


def trend_style():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.2,
                         "axes.titlesize": 9.6, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
                         "pdf.fonttype": 42, "ps.fonttype": 42})


def decomposition_style():
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                         "axes.titlesize": 8.4, "axes.labelsize": 7.8,
                         "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
                         "legend.fontsize": 7.1, "pdf.fonttype": 42, "ps.fonttype": 42})


def clean(ax, ink=INK, grid=GRID):
    ax.grid(axis="y", color=grid, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink)
    ax.tick_params(colors=ink, length=3, width=0.7)


def floor_axis(ax, slots):
    ax.set_xticks(slots)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    clean(ax)
    ax.set_xlabel(r"Draft slot $k$", color=INK, labelpad=2)
    ax.set_ylabel("Information floor", color=INK, labelpad=5)


def decomposition_axis(ax):
    ax.set_ylim(0.0, 0.72)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    clean(ax, DECOMP_INK, DECOMP_GRID)
    ax.set_xlabel(r"Draft slot $k$", color=DECOMP_INK)


def line(ax, x, y, color, lw=2.1, ms=5.0):
    ax.plot(x, y, color=color, linewidth=lw, marker="o", markersize=ms,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.5, zorder=3)


def save(fig, out, stem):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.025)
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


# ------------------------------------------------------------- figures ------
def conditioning_order_comparison(out):
    trend_style()
    slots, t0, t1 = V.conditioning_order()
    removed = 1 - t1[1:] / t0[1:]
    fig, ax = plt.subplots(figsize=(7.05, 1.62))
    ax.fill_between(slots, t1, t0, color=FILL, alpha=0.95, zorder=1)
    line(ax, slots, t0, BLUE)
    line(ax, slots, t1, GREEN)
    for values, dy, va in ((t0, 7, "bottom"), (t1, -4, "top")):
        for s, v in zip(slots, values):
            ax.annotate(f"{v:.3f}", (s, v), xytext=(0, dy), textcoords="offset points",
                        ha="center", va=va, color=INK, fontsize=8.0, bbox=LABEL_BOX,
                        zorder=5)
    ax.text(3.6, 0.105, f"one realised token removes {100 * removed.min():.0f}–100%",
            ha="center", va="center", color="#376E50", fontsize=8.3, fontweight="semibold",
            bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none",
                  "alpha": 0.82}, zorder=4)
    ax.text(6.18, t0[-1] + 0.006, r"$T_k^{(0)}$", color=BLUE, va="center",
            fontweight="semibold")
    ax.text(6.18, t1[-1] + 0.003, r"$T_k^{(1)}$", color=GREEN, va="center",
            fontweight="semibold")
    ax.set_xlim(0.75, 6.72)
    ax.set_ylim(-0.065, 0.345)
    ax.set_title("One-step conditioning removes nearly all of the floor", loc="left",
                 color=INK, fontweight="semibold", pad=4)
    floor_axis(ax, slots)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.30, top=0.76)
    save(fig, out, "conditioning_order_comparison")


def domain_floor_trends(out):
    trend_style()
    slots, series = V.domain_floors()
    colors = (BLUE, ORANGE, GREEN, PURPLE)
    fig, ax = plt.subplots(figsize=(7.05, 1.42))
    for (name, vals), color in zip(series, colors):
        ax.plot(slots, vals, color=color, linewidth=2.0, marker="o", markersize=4.8,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.4,
                zorder=3, label=name)
    legend = ax.legend(loc="upper left", ncol=4, frameon=False, fontsize=8.0,
                       handlelength=1.4, handletextpad=0.4, columnspacing=1.1,
                       borderaxespad=0.1)
    for text, color in zip(legend.get_texts(), colors):
        text.set_color(color)
        text.set_fontweight("semibold")
    ax.set_xlim(0.72, 6.28)
    ax.set_ylim(-0.015, 0.40)
    ax.set_title("Open-ended domains incur higher information floors", loc="left",
                 color=INK, fontweight="semibold", pad=4)
    floor_axis(ax, slots)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    ax.set_ylabel("Order-0 floor")
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.31, top=0.75)
    save(fig, out, "domain_floor_trends")


def prototype_loss_trends(out):
    trend_style()
    slots, series = V.prototypes()
    fig, ax = plt.subplots(figsize=(7.05, 1.34))
    for (W, vals), color in zip(series, (BLUE, GREEN, ORANGE)):
        line(ax, slots, vals, color, lw=2.0, ms=4.8)
        ax.text(6.17, vals[-1], f"{W} prototype" + ("s" if W > 1 else ""), color=color,
                va="center", fontsize=8.3, fontweight="semibold", zorder=5)
    ax.set_xlim(0.75, 7.10)
    ax.set_ylim(-0.005, 0.315)
    ax.set_title("A few path-dependent modes capture most of the floor", loc="left",
                 color=INK, fontweight="semibold", pad=4)
    floor_axis(ax, slots)
    ax.set_ylabel("Loss")
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.32, top=0.73)
    save(fig, out, "prototype_loss_trends")


def cross_target_decompositions(out):
    trend_style()
    names, dfl, dfr, dsf, dsr = V.cross_target()
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 1.35), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    for ax, floor, risk, title in ((axes[0], dfl, dfr, "(a) DFlash · order 0"),
                                   (axes[1], dsf, dsr, "(b) DSpark · order 1")):
        ax.bar(x, floor, width=0.68, color=BLUE, edgecolor="white", linewidth=0.6, zorder=2)
        ax.bar(x, risk - floor, width=0.68, bottom=floor, color=TEAL, edgecolor="white",
               linewidth=0.6, zorder=2)
        ax.bar(x, risk, width=0.68, facecolor="none", edgecolor=INK, linewidth=0.8, zorder=3)
        ax.set_title(title, loc="left", color=INK, fontweight="semibold", pad=2)
        ax.set_xlim(-0.55, 3.55)
        ax.set_ylim(0.0, 0.72)
        ax.set_xticks(x, labels=[SHORT[n] for n in names])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6])
        clean(ax)
    axes[0].set_ylabel("Risk at slot 6", color=INK, labelpad=4)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.27, top=0.74)
    save(fig, out, "cross_target_decompositions")


def slot_priority_trends(out):
    trend_style()
    floor, risk = V.dflash_decomposition()
    gain = V.single_slot_gains("br0")
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 1.45), gridspec_kw={"wspace": 0.20})
    panels = ((axes[0], risk - floor, TEAL, "(a) Model gap grows", "Model gap",
               (0.10, 0.39), [0.1, 0.2, 0.3]),
              (axes[1], gain, ORANGE, "(b) Single-slot serving value falls", "Serving gain",
               (0.02, 0.27), [0.0, 0.1, 0.2]))
    for ax, vals, color, title, ylabel, ylim, yticks in panels:
        line(ax, V.SLOTS, vals, color, lw=2.0, ms=4.7)
        for s, v in zip(V.SLOTS, vals):
            ax.annotate(f"{v:.3f}", (s, v), xytext=(0, 7), textcoords="offset points",
                        ha="center", va="bottom", color=INK, fontsize=7.2, zorder=5)
        ax.set_xlim(-0.25, 6.25)
        ax.set_ylim(*ylim)
        ax.set_xticks(V.SLOTS)
        ax.set_yticks(yticks)
        clean(ax)
        ax.set_title(title, loc="left", x=0.11, color=INK, fontweight="semibold", pad=4)
        ax.set_xlabel(r"Draft slot $k$", color=INK, labelpad=2)
        ax.set_ylabel(ylabel, color=INK, labelpad=4)
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.31, top=0.73)
    save(fig, out, "slot_priority_trends")


def stacked(ax, x, parts, total, width=0.72):
    bottom = np.zeros_like(total)
    for vals, color in parts:
        ax.bar(x, vals, width=width, bottom=bottom, color=color, edgecolor="white",
               linewidth=0.6, zorder=2)
        bottom = bottom + vals
    ax.bar(x, total, width=width, facecolor="none", edgecolor=DECOMP_INK, linewidth=0.8,
           zorder=3)
    for s, t in zip(x, total):
        ax.text(s, t + 0.013, f"{t:.2f}", ha="center", va="bottom", color=DECOMP_INK,
                fontsize=6.7, zorder=5)


def drafter_decomposition(out):
    decomposition_style()
    dfl, dfr = V.dflash_decomposition()
    dsf, dso, dss = V.dspark_decomposition()
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 1.62), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    stacked(axes[0], V.SLOTS, ((dfl, FLOOR_COLOR), (dfr - dfl, GAP_COLOR)), dfr)
    stacked(axes[1], V.SLOTS, ((dsf, FLOOR_COLOR), (dso - dsf, GAP_COLOR),
                               (dss - dso, EXPOSURE_COLOR)), dss)
    axes[0].set_title("(a) DFlash: product measure", loc="left", color=DECOMP_INK,
                      fontweight="semibold")
    axes[0].set_ylabel("Rejection risk", color=DECOMP_INK)
    axes[1].set_title("(b) DSpark: order-1 chain", loc="left", color=DECOMP_INK,
                      fontweight="semibold")
    axes[1].tick_params(axis="y", left=False)
    for ax in axes:
        ax.set_xticks(V.SLOTS)
        decomposition_axis(ax)
    fig.legend(handles=[Patch(facecolor=FLOOR_COLOR, edgecolor="none",
                              label=r"information floor $T^{(m)}$"),
                        Patch(facecolor=GAP_COLOR, edgecolor="none", label="model gap"),
                        Patch(facecolor=EXPOSURE_COLOR, edgecolor="none",
                              label="exposure difference")],
               loc="upper center", bbox_to_anchor=(0.52, 1.025), ncol=3, frameon=False,
               handlelength=1.25, columnspacing=1.6)
    fig.subplots_adjust(left=0.078, right=0.995, bottom=0.30, top=0.69)
    save(fig, out, "drafter_decomposition")


def solution_decomposition(out):
    decomposition_style()
    d = V.solution()
    x = np.array(sorted(d))
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.62), sharey=True,
                             gridspec_kw={"wspace": .14})
    panels = ((("model_gap",), "(a) Model gap", GAP_COLOR),
              (("exposure",), "(b) Exposure difference", EXPOSURE_COLOR),
              (("model_gap", "exposure"), "(c) Total gap", PURPLE))
    for ax, (metrics, title, color) in zip(axes, panels):
        for arm, offset, fill, hatch in ((0, -.19, to_rgba(color, .3), "///"),
                                         (1, .19, color, None)):
            ax.bar(x + offset, [sum(d[k][m][arm] for m in metrics) for k in x], width=.34,
                   color=fill, edgecolor=DECOMP_INK, hatch=hatch, linewidth=.5, zorder=2)
        ax.set_xticks(x)
        ax.set_title(title, loc="left", color=DECOMP_INK, fontweight="semibold")
        decomposition_axis(ax)
        ax.set_ylim(0, .60)
        ax.set_yticks([0, .2, .4, .6])
    axes[0].set_ylabel("Risk difference", color=DECOMP_INK)
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False)
    fig.legend(handles=[Patch(facecolor="white", edgecolor=DECOMP_INK, linewidth=.5,
                              hatch="///", label="DSpark"),
                        Patch(facecolor="#9BA3AD", edgecolor=DECOMP_INK, linewidth=.5,
                              label="Ours")],
               loc="upper center", bbox_to_anchor=(.52, 1.025), ncol=2, frameon=False,
               handlelength=1.25, columnspacing=1.6)
    fig.subplots_adjust(left=.078, right=.995, bottom=.30, top=.69)
    save(fig, out, "solution_decomposition")


FIGURES = {f.__name__: f for f in (
    conditioning_order_comparison, domain_floor_trends, drafter_decomposition,
    cross_target_decompositions, slot_priority_trends, solution_decomposition,
    prototype_loss_trends)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent / "figures")
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES))
    args = ap.parse_args()
    for name in args.only or FIGURES:
        FIGURES[name](args.out)
        print(f"wrote {args.out / name}.pdf")


if __name__ == "__main__":
    main()

"""Matplotlib-only plotting (no seaborn). Every figure is saved as PNG (300 dpi) and PDF.

Models are distinguished by colour AND marker; families by line style (solid for
linear/trees, dashed for deep). Two figure widths are used: a single IEEE column
(3.4 in) for the CD diagrams and dot-whisker plots, and a double column (7 in) for
the multi-panel learning curves.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.constants import FAMILY, FAMILY_LINESTYLE, MODEL_COLOR, MODEL_MARKER

SINGLE_COL_IN = 3.4
DOUBLE_COL_IN = 7.0


def _save(fig: plt.Figure, output_dir: Path, name: str) -> tuple[Path, Path]:
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def plot_learning_curves(
    panels: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    fractions: list[float],
    models: list[str],
    ylabel: str,
    output_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    """panels: {panel_title: {model: (means, los, his)}}, one subplot per panel."""
    n_panels = len(panels)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(DOUBLE_COL_IN, 2.3 * n_rows), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (title, curves) in zip(axes_flat, panels.items()):
        for model in models:
            means, los, his = curves[model]
            color = MODEL_COLOR[model]
            ax.plot(
                fractions,
                means,
                color=color,
                marker=MODEL_MARKER[model],
                linestyle=FAMILY_LINESTYLE[FAMILY[model]],
                markersize=4,
                linewidth=1.2,
                label=model,
            )
            ax.fill_between(fractions, los, his, color=color, alpha=0.15, linewidth=0)
        ax.set_title(title, fontsize=8)
        ax.set_xticks(fractions)
        ax.tick_params(labelsize=7)

    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    for ax in axes_flat[: min(n_cols, n_panels)]:
        pass
    for row in range(n_rows):
        axes_flat[row * n_cols].set_ylabel(ylabel, fontsize=8)
    for ax in axes_flat[max(0, n_panels - n_cols) : n_panels]:
        ax.set_xlabel("Training fraction", fontsize=8)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return _save(fig, output_dir, name)


def plot_cd_diagram(
    mean_rank_by_model: dict[str, float],
    cliques: list[tuple[int, int]],
    cd_nemenyi: float,
    title: str,
    output_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    """Horizontal critical-difference-style diagram. Models are placed at their mean
    rank on a shared axis; thick bars connect the Wilcoxon-Holm non-significant cliques
    (see figure_captions.md: the bar is NOT the Nemenyi bar, which is printed separately).
    """
    order = sorted(mean_rank_by_model, key=lambda m: mean_rank_by_model[m])
    ranks = [mean_rank_by_model[m] for m in order]
    k = len(order)

    # Greedily assign each label a vertical "lane" so that labels whose mean ranks
    # are closer than `min_gap` never land in the same lane (avoids the text
    # collisions that occur when 2-3 models have nearly identical mean ranks).
    min_gap = 0.45
    lanes: list[list[float]] = []
    lane_of_point: list[int] = []
    for x in ranks:
        lane = 0
        while True:
            if lane >= len(lanes):
                lanes.append([])
            if all(abs(x - placed) > min_gap for placed in lanes[lane]):
                lanes[lane].append(x)
                lane_of_point.append(lane)
                break
            lane += 1
    n_lanes = len(lanes)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 2.0 + 0.22 * n_lanes))
    ax.hlines(0, 1, k, color="black", linewidth=1)
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-(0.35 + 0.22 * n_lanes), 0.45 + 0.22 * max(1, len(cliques)))

    for i, model in enumerate(order):
        color = MODEL_COLOR[model]
        ax.plot(ranks[i], 0, marker=MODEL_MARKER[model], color=color, markersize=6, zorder=3)
        y_text = -0.20 - 0.22 * lane_of_point[i]
        ax.annotate(
            model,
            (ranks[i], 0),
            xytext=(ranks[i], y_text),
            ha="center",
            va="top",
            fontsize=7,
        )

    for level, (i, j) in enumerate(cliques):
        y = 0.18 + level * 0.18
        ax.plot([ranks[i], ranks[j]], [y, y], color="black", linewidth=3, solid_capstyle="butt")

    ax.set_xlabel("Mean rank (1 = best)", fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"{title}  (Nemenyi CD = {cd_nemenyi:.3f}, not drawn)", fontsize=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return _save(fig, output_dir, name)


def plot_diff_dotwhisker(
    diff_all5: dict[str, tuple[float, float, float]],
    diff_no_ckd: dict[str, tuple[float, float, float]],
    models: list[str],
    title: str,
    output_dir: Path,
    name: str,
) -> tuple[Path, Path]:
    """Dot-and-whisker plot of the dataset-averaged difference to logreg, ALL5 vs NO_CKD."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, 0.5 * len(models) + 0.6))
    y_positions = np.arange(len(models))[::-1]
    offset = 0.15

    for y, model in zip(y_positions, models):
        for which, offs, marker_fill in ((diff_all5, offset, True), (diff_no_ckd, -offset, False)):
            point, lo, hi = which[model]
            color = MODEL_COLOR[model]
            ax.errorbar(
                point,
                y + offs,
                xerr=[[point - lo], [hi - point]],
                fmt=MODEL_MARKER[model],
                color=color,
                markerfacecolor=color if marker_fill else "white",
                markeredgecolor=color,
                capsize=2,
                linewidth=1,
            )

    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(models, fontsize=7)
    ax.set_xlabel("AUROC difference vs. logreg", fontsize=8)
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    return _save(fig, output_dir, name)

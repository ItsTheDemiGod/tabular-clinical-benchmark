"""Phase 4c: the two main paper figures, drawn at their FINAL physical size.

Reads ONLY result CSVs (never images). Writes paper/figures/<name>.png (300 dpi) and
.pdf (TrueType fonts embedded, no Type 3), plus paper/figures/SOURCES.txt listing every
source CSV with its SHA-256 and the filter used. Fits/changes nothing.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.constants import (
    DATASETS_NO_CKD, FAMILY, FAMILY_LINESTYLE, MODEL_COLOR, MODEL_MARKER, MODELS,
)

OUT_DIR = PROJECT_ROOT / "paper" / "figures"
DOUBLE_COL_IN = 7.16
MIN_FONT_PT = 7
FRACTIONS = [0.1, 0.25, 0.5, 1.0]
Y_LO, Y_HI = 0.03, 3.5  # slope axis (log); the lowest CI limit is 0.044

P3A_CI = Path("results/phase3a/ci_dataset_averaged.csv")
P3B_METRICS = Path("results/phase3b/calibration_metrics.csv")
P3B_CI = Path("results/phase3b/calibration_bootstrap_ci.csv")
P3B_AVG = Path("results/phase3b/calibration_averaged.csv")
P3C_AVG = Path("results/phase3c/robustness_averaged.csv")
P3D_AVG = Path("results/phase3d/noise_continuous/noise_cont_averaged.csv")

# figure name -> (width_in, height_in)
FIGURE_SIZES: dict[str, tuple[float, float]] = {
    "fig_main_learning_calibration": (DOUBLE_COL_IN, 2.4),
    "fig_main_robustness": (DOUBLE_COL_IN, 2.2),
}


def _style() -> None:
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,  # embed TrueType, never Type 3
        "font.size": MIN_FONT_PT, "axes.labelsize": MIN_FONT_PT, "axes.titlesize": MIN_FONT_PT,
        "xtick.labelsize": MIN_FONT_PT, "ytick.labelsize": MIN_FONT_PT,
        "legend.fontsize": MIN_FONT_PT, "figure.dpi": 100, "axes.linewidth": 0.6,
        "lines.linewidth": 1.0, "lines.markersize": 3.5,
    })


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / path)


def sha256_of(path: Path) -> str:
    return hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()


def _letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.02, 1.03, f"({letter})", transform=ax.transAxes, fontsize=MIN_FONT_PT,
            fontweight="bold", ha="right", va="bottom")


def _model_legend(fig: plt.Figure, extra: list[Line2D] | None = None) -> None:
    handles = [Line2D([], [], color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ls="none", label=m)
               for m in MODELS] + (extra or [])
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles),
               frameon=False, handlelength=1.3, handletextpad=0.3, columnspacing=0.9, borderaxespad=0.1)


def build_fig_learning_calibration() -> tuple[plt.Figure, list[dict]]:
    """Fig. 1: (a) AUROC and (b) log loss vs fraction (ALL5, 95% CI bands); (c) NO_CKD slope at f=0.1."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["fig_main_learning_calibration"],
                             layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1.05]})
    ci = _read(P3A_CI)
    for ax, metric, ylabel, letter in ((axes[0], "auroc", "AUROC (ALL5 mean)", "a"),
                                       (axes[1], "log_loss", "Log loss (ALL5 mean)", "b")):
        for m in MODELS:
            d = ci[(ci.model == m) & (ci.metric == metric) & (ci.dataset_set == "ALL5")].set_index("fraction").loc[FRACTIONS]
            ls = FAMILY_LINESTYLE[FAMILY[m]]
            ax.fill_between(FRACTIONS, d.ci_lo, d.ci_hi, color=MODEL_COLOR[m], alpha=0.15, lw=0)
            ax.plot(FRACTIONS, d.point, color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ls=ls)
        ax.set_xscale("log")
        ax.set_xticks(FRACTIONS)
        ax.set_xticklabels(["0.1", "0.25", "0.5", "1.0"])
        ax.minorticks_off()
        ax.set_xlabel("Training fraction")
        ax.set_ylabel(ylabel)
        _letter(ax, letter)

    ax = axes[2]
    met = _read(P3B_METRICS)
    cci = _read(P3B_CI)
    avg = _read(P3B_AVG)
    offsets = np.linspace(-0.24, 0.24, len(DATASETS_NO_CKD))
    clipped = 0
    for i, m in enumerate(MODELS):
        for off, ds in zip(offsets, DATASETS_NO_CKD):
            r = met[(met.dataset == ds) & (met.model == m) & (met.fraction == 0.1)].iloc[0]
            c = cci[(cci.dataset == ds) & (cci.model == m) & (cci.fraction == 0.1)].iloc[0]
            ax.errorbar(i + off, r.slope, yerr=[[r.slope - c.slope_lo], [c.slope_hi - r.slope]],
                        color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ms=3, lw=0.6, capsize=0, elinewidth=0.6)
            clipped += int(not (Y_LO <= c.slope_lo and c.slope_hi <= Y_HI))
        mean = avg[(avg.model == m) & (avg.fraction == 0.1) & (avg.dataset_set == "NO_CKD")].iloc[0].slope_mean
        ax.hlines(mean, i - 0.34, i + 0.34, color="black", lw=1.6, zorder=5)
    ax.axhline(1.0, color="gray", lw=0.7, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2])
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax.minorticks_off()
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=45, ha="right")
    ax.set_ylabel("Calibration slope (f = 0.1)")
    _letter(ax, "c")
    assert clipped == 0, f"{clipped} slope CI(s) fall outside the y-range"

    _model_legend(fig, [Line2D([], [], color="black", lw=1.6, label="NO_CKD mean")])
    sources = [
        {"file": P3A_CI, "filter": "metric in {auroc, log_loss}, dataset_set == ALL5, fractions 0.1/0.25/0.5/1.0, bands = ci_lo/ci_hi (panels a, b)"},
        {"file": P3B_METRICS, "filter": "fraction == 0.1, dataset in NO_CKD (pima_diabetes, heart_cleveland, ilpd, breast_wisconsin); column slope = one dot per dataset (panel c)"},
        {"file": P3B_CI, "filter": "fraction == 0.1, same datasets; whiskers = slope_lo/slope_hi (per-dataset bootstrap CI)"},
        {"file": P3B_AVG, "filter": "fraction == 0.1, dataset_set == NO_CKD; column slope_mean = black bar. No CI exists for the NO_CKD average, so none is drawn."},
    ]
    return fig, sources


def build_fig_robustness() -> tuple[plt.Figure, list[dict]]:
    """Fig. 2: (a) AUROC degradation at 30% missing / noise 0.5; (b) log-loss increase at noise 0.5, original vs continuous-only."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["fig_main_robustness"],
                             layout="constrained", gridspec_kw={"width_ratios": [1, 1.15]})
    rob = _read(P3C_AVG)
    cont = _read(P3D_AVG)
    rob = rob[rob.dataset_set == "ALL5"]
    cont = cont[cont.dataset_set == "ALL5"]

    ax = axes[0]
    offs = np.linspace(-0.3, 0.3, len(MODELS))
    for gi, (kind, level) in enumerate((("missing", 0.3), ("noise", 0.5))):
        for off, m in zip(offs, MODELS):
            r = rob[(rob.model == m) & (rob.kind == kind) & (np.isclose(rob.level, level)) & (rob.metric == "auroc")].iloc[0]
            ax.errorbar(gi + off, r.degradation_mean, yerr=[[r.degradation_mean - r.degradation_ci_lo], [r.degradation_ci_hi - r.degradation_mean]],
                        color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ms=3.2, lw=0.6, capsize=0, elinewidth=0.6, ls="none")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["30% missing", "Noise 0.5"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("AUROC change")
    _letter(ax, "a")

    ax = axes[1]
    for i, m in enumerate(MODELS):
        for off, (src, level_kind, filled) in zip((-0.15, 0.15), ((rob, "noise", True), (cont, "noise_cont", False))):
            r = src[(src.model == m) & (src.kind == level_kind) & (np.isclose(src.level, 0.5)) & (src.metric == "log_loss")].iloc[0]
            ax.errorbar(i + off, r.degradation_mean, yerr=[[r.degradation_mean - r.degradation_ci_lo], [r.degradation_ci_hi - r.degradation_mean]],
                        color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ms=3.2, lw=0.6, capsize=0, elinewidth=0.6, ls="none",
                        mfc=MODEL_COLOR[m] if filled else "white")
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODELS, rotation=45, ha="right")
    ax.set_ylabel("Log-loss change")
    _letter(ax, "b")

    _model_legend(fig, [
        Line2D([], [], color="gray", marker="o", ls="none", label="original noise (b, filled)"),
        Line2D([], [], color="gray", marker="o", mfc="white", ls="none", label="continuous-only (b, open)"),
    ])
    sources = [
        {"file": P3C_AVG, "filter": "dataset_set == ALL5; panel a: kind==missing level==0.3 and kind==noise level==0.5, metric==auroc, degradation_mean with degradation_ci_lo/hi; panel b: kind==noise level==0.5, metric==log_loss (filled markers)"},
        {"file": P3D_AVG, "filter": "dataset_set == ALL5, kind==noise_cont level==0.5, metric==log_loss (open markers, panel b)"},
    ]
    return fig, sources


BUILDERS = {
    "fig_main_learning_calibration": build_fig_learning_calibration,
    "fig_main_robustness": build_fig_robustness,
}


def main() -> None:
    print(f"sys.executable = {sys.executable}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Figure sources (generated by scripts/make_paper_figures.py; do not edit by hand)", ""]
    for name, build in BUILDERS.items():
        fig, sources = build()
        w, h = FIGURE_SIZES[name]
        fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
        fig.savefig(OUT_DIR / f"{name}.pdf")
        plt.close(fig)
        lines.append(f"[{name}]  size {w} x {h} in  files: {name}.png (300 dpi), {name}.pdf")
        for s in sources:
            lines.append(f"  source: {s['file'].as_posix()}  sha256={sha256_of(s['file'])}")
            lines.append(f"    filter: {s['filter']}")
        lines.append("")
        print(f"wrote {name}")
    (OUT_DIR / "SOURCES.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()

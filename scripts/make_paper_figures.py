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
    DATASETS_ALL5, DATASETS_NO_CKD, FAMILY, FAMILY_LINESTYLE, MODEL_COLOR, MODEL_MARKER, MODELS,
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

# Fig. 1(c): datasets are identified by marker SHAPE (colour stays = model).
DATASET_MARKER = {"pima_diabetes": "o", "heart_cleveland": "s", "ilpd": "^", "breast_wisconsin": "D"}

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


def _model_legend(fig: plt.Figure, extra: list[Line2D] | None = None, ncol: int | None = None) -> None:
    handles = [Line2D([], [], color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ls="none", label=m)
               for m in MODELS] + (extra or [])
    fig.legend(handles=handles, loc="outside upper center", ncol=ncol or len(handles),
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
                        color=MODEL_COLOR[m], marker=DATASET_MARKER[ds], ms=3, lw=0.6, capsize=0, elinewidth=0.6)
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

    _model_legend(fig, [Line2D([], [], color="black", lw=1.6, label="NO_CKD mean (c)")]
                  + [Line2D([], [], color="gray", marker=DATASET_MARKER[d], ls="none", label=d)
                     for d in DATASETS_NO_CKD], ncol=6)
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


# ---------------------------------------------------------------------------
# Phase 4d: Fig. 3 (explanation agreement and plausibility) and Fig. 4 (reliability).
# ---------------------------------------------------------------------------
from analysis.calibration import ece_equal_mass, wilson_interval  # noqa: E402
from scripts.phase3b_calibration import _reliability_bins  # noqa: E402  (Phase 3b's own equal-mass binning, reused)

P3D_AGREE = Path("results/phase3d/agreement_per_dataset.csv")
P3D_BASE = Path("results/phase3d/baselines.csv")
P3D_PLAUS = Path("results/phase3d/plausibility.csv")
MANIFEST = Path("results/MANIFEST_sha256.txt")
PRED_DIR = Path("results/predictions")
N_SPLITS = 10

SHAP_MODELS = ["logreg", "rf", "mlp"]  # pre-registered family rule (CLAUDE.md section 7, decision 14)
PAIRS = [("logreg", "rf"), ("logreg", "mlp"), ("rf", "mlp")]
PAIR_MARKER = {PAIRS[0]: "o", PAIRS[1]: "^", PAIRS[2]: "D"}
PAIR_COLOR = {PAIRS[0]: "#444444", PAIRS[1]: "#777777", PAIRS[2]: "#AAAAAA"}  # neutral greys
DAGGER_THRESHOLD = 0.60  # chance precision >= this: plausibility is non-informative (claim map C10)
DS_TICK = {"pima_diabetes": "Pima", "heart_cleveland": "Cleveland", "ckd": "CKD", "ilpd": "ILPD",
           "breast_wisconsin": "Breast W."}

FIGURE_SIZES_4D: dict[str, tuple[float, float]] = {
    "fig_main_explanations": (DOUBLE_COL_IN, 2.4),
    "fig_main_reliability": (3.45, 2.7),
}


def dagger_label(label: str, chance: float) -> str:
    """Tick label with a dagger when the chance level is at least DAGGER_THRESHOLD."""
    return label + "†" if chance >= DAGGER_THRESHOLD - 1e-12 else label


def reliability_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Equal-mass ECE recomputed from the bins that are plotted (must equal ece_equal_mass)."""
    mean_p, obs, _, _, n_g = _reliability_bins(np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64), n_bins)
    return float(np.sum(n_g / n_g.sum() * np.abs(obs - mean_p)))


def build_fig_explanations() -> tuple[plt.Figure, list[dict]]:
    """Fig. 3: (a) cross-model top-5 Jaccard vs within-model stability and chance; (b) plausibility precision."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES_4D["fig_main_explanations"], layout="constrained")
    agree, base, plaus = _read(P3D_AGREE), _read(P3D_BASE), _read(P3D_PLAUS)
    xs = np.arange(len(DATASETS_ALL5))

    ax = axes[0]
    offs = np.linspace(-0.3, 0.3, 5)  # 3 pairs, within-model stability, chance
    for i, ds in enumerate(DATASETS_ALL5):
        for off, pair in zip(offs[:3], PAIRS):
            r = agree[(agree.dataset == ds) & (agree.model_a == pair[0]) & (agree.model_b == pair[1])].iloc[0]
            ax.plot(i + off, r.jaccard_mean, marker=PAIR_MARKER[pair], color=PAIR_COLOR[pair], ls="none", ms=3.6)
        w = base[(base.dataset == ds) & (base.type == "within_model_mean_over_models")].iloc[0]
        ax.plot(i + offs[3], w.jaccard_mean, marker="s", mfc="white", mec="black", color="black", ls="none", ms=3.6)
        c = base[(base.dataset == ds) & (base.type == "chance")].iloc[0]
        ax.plot(i + offs[4], c.jaccard_mean, marker="x", color="black", ls="none", ms=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([DS_TICK[d] for d in DATASETS_ALL5])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-5 Jaccard")
    _letter(ax, "a")

    ax = axes[1]
    offs_m = np.linspace(-0.27, 0.15, 3)
    chances = {}
    for i, ds in enumerate(DATASETS_ALL5):
        for off, m in zip(offs_m, SHAP_MODELS):
            r = plaus[(plaus.dataset == ds) & (plaus.model == m)].iloc[0]
            ax.errorbar(i + off, r.precision_mean, yerr=[[r.precision_mean - r.precision_ci_lo], [r.precision_ci_hi - r.precision_mean]],
                        color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ms=3.4, lw=0.6, capsize=0, elinewidth=0.6, ls="none")
        chance = float(plaus[plaus.dataset == ds].chance_precision.iloc[0])
        chances[ds] = chance
        ax.plot(i + 0.3, chance, marker="x", color="black", ls="none", ms=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([dagger_label(DS_TICK[d], chances[d]) for d in DATASETS_ALL5])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Precision vs clinical list")
    _letter(ax, "b")

    handles = [Line2D([], [], color=PAIR_COLOR[p], marker=PAIR_MARKER[p], ls="none", label=f"{p[0]}-{p[1]} (a)") for p in PAIRS]
    handles += [Line2D([], [], color="black", marker="s", mfc="white", ls="none", label="within-model stability (a)"),
                Line2D([], [], color="black", marker="x", ls="none", label="chance level")]
    handles += [Line2D([], [], color=MODEL_COLOR[m], marker=MODEL_MARKER[m], ls="none", label=f"{m} (b)") for m in SHAP_MODELS]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, handlelength=1.3,
               handletextpad=0.3, columnspacing=0.9, borderaxespad=0.1)
    sources = [
        {"file": P3D_AGREE, "filter": "all 5 datasets; model pairs logreg-rf, logreg-mlp, rf-mlp; column jaccard_mean = dots (panel a)"},
        {"file": P3D_BASE, "filter": "type == within_model_mean_over_models (squares, jaccard_mean) and type == chance (crosses, jaccard_mean) (panel a)"},
        {"file": P3D_PLAUS, "filter": "models logreg, rf, mlp; precision_mean with precision_ci_lo/hi (dots, 95% CI) and chance_precision (crosses) (panel b). "
                                      "DAGGER (†) on a dataset tick label of panel b = chance precision >= 0.60, i.e. plausibility non-informative (claim map C10); "
                                      "define it in the caption"},
    ]
    return fig, sources


def load_pooled_f01(model: str, datasets: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per dataset: (y_true, prob_0.1) concatenated over the 10 splits (stored y_true, no recomputation)."""
    out = {}
    for ds in datasets:
        ys, ps = [], []
        for s in range(N_SPLITS):
            z = np.load(PROJECT_ROOT / PRED_DIR / f"{ds}__{model}__split{s}.npz")
            ys.append(z["y_true"].astype(np.float64))
            ps.append(z["prob_0.1"].astype(np.float64))
        out[ds] = (np.concatenate(ys), np.concatenate(ps))
    return out


RELIABILITY_CHECK: dict = {}


def build_fig_reliability() -> tuple[plt.Figure, list[dict]]:
    """Fig. 4: equal-mass 10-bin reliability, 4 NO_CKD datasets pooled, f=0.1, logreg/rf/mlp, Wilson 95% CIs."""
    _style()
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZES_4D["fig_main_reliability"], layout="constrained")
    met = _read(P3B_METRICS)
    max_diff = 0.0
    for m in SHAP_MODELS:
        per_ds = load_pooled_f01(m, DATASETS_NO_CKD)
        # Reproduce Phase 3b's equal-mass ECE for every (dataset, model) from the plotted binning.
        for ds, (y, p) in per_ds.items():
            ref = float(met[(met.dataset == ds) & (met.model == m) & (met.fraction == 0.1)].iloc[0].ece_em)
            max_diff = max(max_diff, abs(reliability_ece(y, p) - ref))
        y_all = np.concatenate([v[0] for v in per_ds.values()])
        p_all = np.concatenate([v[1] for v in per_ds.values()])
        max_diff = max(max_diff, abs(reliability_ece(y_all, p_all) - ece_equal_mass(y_all, p_all)))
        mean_p, obs, lo, hi, _ = _reliability_bins(y_all, p_all)
        ax.errorbar(mean_p, obs, yerr=[obs - lo, hi - obs], color=MODEL_COLOR[m], marker=MODEL_MARKER[m],
                    ms=3.4, lw=0.8, capsize=0, elinewidth=0.6, label=m)
    RELIABILITY_CHECK["max_abs_diff"] = max_diff
    if max_diff > 1e-9:
        raise SystemExit(f"reliability bins do not reproduce Phase 3b equal-mass ECE: max diff {max_diff}")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=0.7, label="perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (equal-mass bins)")
    ax.set_ylabel("Observed fraction positive")
    ax.set_aspect("equal", adjustable="box")
    handles = [Line2D([], [], color=MODEL_COLOR[m], marker=MODEL_MARKER[m], lw=0.8, ms=3.4, label=m) for m in SHAP_MODELS]
    handles.append(Line2D([], [], color="gray", ls="--", lw=0.7, label="perfect calibration"))
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False,
               handlelength=1.3, handletextpad=0.3, columnspacing=0.9, borderaxespad=0.1)
    pat = [ln for ln in (PROJECT_ROOT / MANIFEST).read_text(encoding="utf-8").splitlines()
           if any(f"results/predictions/{ds}__{m}__split" in ln for ds in DATASETS_NO_CKD for m in SHAP_MODELS)]
    sources = [
        {"file": P3B_METRICS, "filter": "fraction == 0.1, 4 NO_CKD datasets x logreg/rf/mlp, column ece_em: reference for the reproduction check only (not plotted)"},
        {"pattern": "results/predictions/{pima_diabetes,heart_cleveland,ilpd,breast_wisconsin}__{logreg,rf,mlp}__split{0..9}.npz",
         "filter": "keys y_true and prob_0.1; the 4 datasets pooled per model (n = 2 x rows per dataset, summed); equal-mass 10 bins "
                   "(scripts.phase3b_calibration._reliability_bins), Wilson 95% intervals (analysis.calibration.wilson_interval)",
         "manifest_lines": pat},
    ]
    return fig, sources


BUILDERS_4D = {"fig_main_explanations": build_fig_explanations, "fig_main_reliability": build_fig_reliability}


BUILDERS = {
    "fig_main_learning_calibration": build_fig_learning_calibration,
    "fig_main_robustness": build_fig_robustness,
}


def main() -> None:
    print(f"sys.executable = {sys.executable}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["# Figure sources (generated by scripts/make_paper_figures.py; do not edit by hand)", ""]
    sizes = {**FIGURE_SIZES, **FIGURE_SIZES_4D}
    for name, build in {**BUILDERS, **BUILDERS_4D}.items():
        fig, sources = build()
        w, h = sizes[name]
        fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
        fig.savefig(OUT_DIR / f"{name}.pdf")
        plt.close(fig)
        lines.append(f"[{name}]  size {w} x {h} in  files: {name}.png (300 dpi), {name}.pdf")
        for s in sources:
            if "file" in s:
                lines.append(f"  source: {s['file'].as_posix()}  sha256={sha256_of(s['file'])}")
            else:
                lines.append(f"  source files (pattern): {s['pattern']}")
            lines.append(f"    filter: {s['filter']}")
            if "manifest_lines" in s:
                lines.append(f"    matching lines of {MANIFEST.as_posix()} ({len(s['manifest_lines'])} files):")
                lines += [f"      manifest: {ln}" for ln in s["manifest_lines"]]
        if name == "fig_main_reliability":
            lines.append(f"  reproduction check: max |binned equal-mass ECE - Phase 3b ece_em| = {RELIABILITY_CHECK['max_abs_diff']:.3e}")
        lines.append("")
        print(f"wrote {name}")
    print(f"reliability reproduction max abs diff = {RELIABILITY_CHECK['max_abs_diff']:.3e}")
    (OUT_DIR / "SOURCES.txt").write_text(chr(10).join(lines), encoding="utf-8", newline=chr(10))


if __name__ == "__main__":
    main()

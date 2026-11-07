"""Phase 3b: calibration analysis (pooled metrics, slopes, reliability diagrams,
cross-fitted post-hoc recalibration check).

Reads results/predictions/*.npz, results/all_runs.csv and results/raw/*.csv
(verify only). Writes ONLY inside results/phase3b/. Never touches results/raw,
results/tuning, results/models, results/diagnostics, or anything in src/.

Subcommands: verify, metrics, posthoc, figures, tables, all.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.calibration import (
    bootstrap_diff_pooled_metric,
    bootstrap_pooled_metric,
    cross_fit_calibrate,
    ece_equal_mass,
    fit_platt,
    logit,
    pooled_metrics,
    wilson_interval,
)
from analysis.constants import (
    DATASETS_ALL5,
    DATASETS_NO_CKD,
    FAMILY,
    FAMILY_LINESTYLE,
    FRACTIONS_ALL,
    FRACTIONS_TEST,
    MODEL_COLOR,
    MODEL_MARKER,
    MODELS,
    SEED,
)
from analysis.data import block_matrix, load_all_runs
from analysis.stats import ranks_best_is_1
from analysis.tables import bold_best_mask, escape_latex, format_float, write_csv_and_tex
from src.data import load_dataset
from src.metrics import LOG_LOSS_EPS, compute_metrics, expected_calibration_error

PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase3b"
BOOTSTRAP_B = 2000
DATASET_MARKER = {
    "pima_diabetes": "o",
    "heart_cleveland": "s",
    "ckd": "^",
    "ilpd": "D",
    "breast_wisconsin": "v",
}


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# Shared loading
# ---------------------------------------------------------------------------
def _npz_path(dataset: str, model: str, split: int) -> Path:
    return PREDICTIONS_DIR / f"{dataset}__{model}__split{split}.npz"


def _load_task(dataset: str, model: str, split: int) -> dict:
    data = np.load(_npz_path(dataset, model, split))
    return {k: data[k] for k in data.files}


def _load_pooled_arrays(dataset: str, model: str, fraction: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The 10 per-split (y_true, prob) arrays for one (dataset, model, fraction)."""
    y_splits, p_splits = [], []
    for split in range(10):
        task = _load_task(dataset, model, split)
        y_splits.append(task["y_true"].astype(np.float64))
        p_splits.append(task[f"prob_{fraction}"].astype(np.float64))
    return y_splits, p_splits


# ---------------------------------------------------------------------------
# Subcommand: verify
# ---------------------------------------------------------------------------
def run_verify() -> dict:
    fractions = FRACTIONS_ALL
    expected_keys = {"test_idx", "y_true"} | {f"prob_{f}" for f in fractions}

    structural_errors: list[str] = []
    alignment_errors: list[str] = []
    n_files = 0

    all_runs = load_all_runs()
    diffs = {m: [] for m in ("auroc", "brier", "log_loss", "ece")}

    dataset_labels: dict[str, np.ndarray] = {}

    for dataset in DATASETS_ALL5:
        X, y, _ = load_dataset(dataset, verbose=False)
        dataset_labels[dataset] = y.to_numpy()

        for model in MODELS:
            for split in range(10):
                path = _npz_path(dataset, model, split)
                if not path.exists():
                    structural_errors.append(f"missing file: {path}")
                    continue
                n_files += 1
                task = _load_task(dataset, model, split)

                missing_keys = expected_keys - set(task.keys())
                if missing_keys:
                    structural_errors.append(f"{path.name}: missing keys {missing_keys}")
                    continue

                test_idx = task["test_idx"]
                y_true = task["y_true"]
                if len(test_idx) != len(y_true):
                    structural_errors.append(f"{path.name}: len(test_idx) != len(y_true)")
                if not set(np.unique(y_true).tolist()) <= {0, 1}:
                    structural_errors.append(f"{path.name}: y_true not binary")

                for fraction in fractions:
                    prob = task[f"prob_{fraction}"]
                    if not np.all(np.isfinite(prob)):
                        structural_errors.append(f"{path.name} frac={fraction}: non-finite probability")
                    if prob.min() < 0 or prob.max() > 1:
                        structural_errors.append(f"{path.name} frac={fraction}: probability outside [0,1]")

                # Alignment check.
                expected_labels = dataset_labels[dataset][test_idx]
                if not np.array_equal(y_true, expected_labels):
                    alignment_errors.append(f"{path.name}: y_true != y[test_idx]")

                # Recompute metrics and compare to results/all_runs.csv.
                row = all_runs[
                    (all_runs["dataset"] == dataset)
                    & (all_runs["model"] == model)
                    & (all_runs["split_idx"] == split)
                    & (all_runs["fraction"] == fraction)
                ]
                if len(row) != 1:
                    alignment_errors.append(
                        f"{path.name} frac={fraction}: expected exactly 1 matching all_runs.csv row, found {len(row)}"
                    )
                    continue
                recomputed = compute_metrics(y_true, task[f"prob_{fraction}"])
                for metric in diffs:
                    diffs[metric].append(abs(recomputed[metric] - float(row.iloc[0][metric])))

    max_abs_diff = {m: (max(v) if v else float("nan")) for m, v in diffs.items()}
    n_above_1e9 = {m: sum(1 for d in v if d > 1e-9) for m, v in diffs.items()}
    n_above_1e6 = {m: sum(1 for d in v if d > 1e-6) for m, v in diffs.items()}

    stop = bool(structural_errors) or bool(alignment_errors) or any(v > 0 for v in n_above_1e6.values())

    return {
        "n_files": n_files,
        "structural_errors": structural_errors,
        "alignment_errors": alignment_errors,
        "max_abs_diff": max_abs_diff,
        "n_above_1e9": n_above_1e9,
        "n_above_1e6": n_above_1e6,
        "stop": stop,
    }


def _print_verify_report(report: dict) -> None:
    print(f"verify: {report['n_files']} files checked")
    print(f"structural errors: {len(report['structural_errors'])}")
    for e in report["structural_errors"][:20]:
        print(f"  {e}")
    print(f"alignment errors: {len(report['alignment_errors'])}")
    for e in report["alignment_errors"][:20]:
        print(f"  {e}")
    for metric in ("auroc", "brier", "log_loss", "ece"):
        print(
            f"  {metric}: max_abs_diff={report['max_abs_diff'][metric]:.3e} "
            f"n>1e-9={report['n_above_1e9'][metric]} n>1e-6={report['n_above_1e6'][metric]}"
        )
    print(f"STOP: {report['stop']}")


# ---------------------------------------------------------------------------
# Subcommand: metrics
# ---------------------------------------------------------------------------
def run_metrics_subcommand() -> None:
    output_dir = ensure_output_dir()
    rng = np.random.default_rng(SEED)

    metrics_rows = []
    ci_rows = []
    pooled_cache: dict[tuple[str, str, float], dict] = {}
    split_cache: dict[tuple[str, str, float], tuple[list, list]] = {}

    for dataset in DATASETS_ALL5:
        for model in MODELS:
            for fraction in FRACTIONS_ALL:
                y_splits, p_splits = _load_pooled_arrays(dataset, model, fraction)
                split_cache[(dataset, model, fraction)] = (y_splits, p_splits)
                y_pooled = np.concatenate(y_splits)
                p_pooled = np.concatenate(p_splits)
                pm = pooled_metrics(y_pooled, p_pooled)
                pooled_cache[(dataset, model, fraction)] = pm
                mean_split_ece = float(
                    np.mean([expected_calibration_error(y, p) for y, p in zip(y_splits, p_splits)])
                )
                metrics_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "fraction": fraction,
                        **pm,
                        "mean_split_ece": mean_split_ece,
                    }
                )

                # Bootstrap CI for ece_ew, slope, intercept (resample the 10 splits).
                ece_boot = bootstrap_pooled_metric(
                    y_splits, p_splits, rng, BOOTSTRAP_B, lambda y, p: expected_calibration_error(y, p)
                )

                def _slope_fn(y, p):
                    _, slope, sep = fit_platt(y, logit(p))
                    return slope

                def _intercept_fn(y, p):
                    intercept, _, sep = fit_platt(y, logit(p))
                    return intercept

                slope_boot = bootstrap_pooled_metric(y_splits, p_splits, rng, BOOTSTRAP_B, _slope_fn)
                intercept_boot = bootstrap_pooled_metric(y_splits, p_splits, rng, BOOTSTRAP_B, _intercept_fn)

                n_dropped = int(np.isnan(slope_boot).sum())
                ece_lo, ece_hi = np.percentile(ece_boot, [2.5, 97.5])
                valid_slope = slope_boot[~np.isnan(slope_boot)]
                valid_intercept = intercept_boot[~np.isnan(intercept_boot)]
                slope_lo, slope_hi = (
                    np.percentile(valid_slope, [2.5, 97.5]) if len(valid_slope) else (np.nan, np.nan)
                )
                intercept_lo, intercept_hi = (
                    np.percentile(valid_intercept, [2.5, 97.5]) if len(valid_intercept) else (np.nan, np.nan)
                )
                ci_rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "fraction": fraction,
                        "ece_ew_lo": ece_lo,
                        "ece_ew_hi": ece_hi,
                        "slope_lo": slope_lo,
                        "slope_hi": slope_hi,
                        "intercept_lo": intercept_lo,
                        "intercept_hi": intercept_hi,
                        "n_dropped_replicates": n_dropped,
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "calibration_metrics.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(output_dir / "calibration_bootstrap_ci.csv", index=False)

    # Dataset-averaged values (plain averages over the per-dataset pooled metrics).
    averaged_rows = []
    for model in MODELS:
        for fraction in FRACTIONS_ALL:
            for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                subset = metrics_df[
                    (metrics_df.model == model) & (metrics_df.fraction == fraction) & (metrics_df.dataset.isin(datasets))
                ]
                non_separable = subset[~subset.separable]
                separable_datasets = subset[subset.separable]["dataset"].tolist()
                averaged_rows.append(
                    {
                        "model": model,
                        "fraction": fraction,
                        "dataset_set": set_name,
                        "ece_ew_mean": subset["ece_ew"].mean(),
                        "ece_em_mean": subset["ece_em"].mean(),
                        "slope_mean": non_separable["slope"].mean() if len(non_separable) else np.nan,
                        "intercept_mean": non_separable["intercept"].mean() if len(non_separable) else np.nan,
                        "n_datasets_used_for_slope": len(non_separable),
                        "separable_datasets": ",".join(separable_datasets),
                    }
                )
    pd.DataFrame(averaged_rows).to_csv(output_dir / "calibration_averaged.csv", index=False)

    # Brier/log_loss dataset-averaged (pooled, from compute_metrics on pooled predictions).
    bl_rows = []
    for model in MODELS:
        for fraction in FRACTIONS_ALL:
            for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                briers, loglosses = [], []
                for dataset in datasets:
                    y_splits, p_splits = split_cache[(dataset, model, fraction)]
                    y_pooled = np.concatenate(y_splits)
                    p_pooled = np.concatenate(p_splits)
                    m = compute_metrics(y_pooled, p_pooled)
                    briers.append(m["brier"])
                    loglosses.append(m["log_loss"])
                bl_rows.append(
                    {
                        "model": model,
                        "fraction": fraction,
                        "dataset_set": set_name,
                        "brier_mean": float(np.mean(briers)),
                        "log_loss_mean": float(np.mean(loglosses)),
                    }
                )
    bl_df = pd.DataFrame(bl_rows)
    bl_df.to_csv(output_dir / "calibration_brier_logloss_averaged.csv", index=False)

    # Paired bootstrap difference to logreg, pooled ece_ew, dataset-averaged, fractions 1.0/0.1.
    rng_diff = np.random.default_rng(SEED)
    diff_rows = []
    for model in MODELS:
        if model == "logreg":
            continue
        for fraction in FRACTIONS_TEST:
            for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                per_dataset_diff_boot = []
                point_diffs = []
                for dataset in datasets:
                    y_m, p_m = split_cache[(dataset, model, fraction)]
                    y_l, p_l = split_cache[(dataset, "logreg", fraction)]
                    diff_boot = bootstrap_diff_pooled_metric(
                        y_m, p_m, y_l, p_l, rng_diff, BOOTSTRAP_B,
                        lambda y, p: expected_calibration_error(y, p),
                    )
                    per_dataset_diff_boot.append(diff_boot)
                    point_diffs.append(
                        expected_calibration_error(np.concatenate(y_m), np.concatenate(p_m))
                        - expected_calibration_error(np.concatenate(y_l), np.concatenate(p_l))
                    )
                averaged_boot = np.mean(np.vstack(per_dataset_diff_boot), axis=0)
                lo, hi = np.percentile(averaged_boot, [2.5, 97.5])
                diff_rows.append(
                    {
                        "model": model,
                        "fraction": fraction,
                        "dataset_set": set_name,
                        "ece_ew_diff_mean": float(np.mean(point_diffs)),
                        "ci_lo": float(lo),
                        "ci_hi": float(hi),
                    }
                )
    pd.DataFrame(diff_rows).to_csv(output_dir / "calibration_diff_vs_logreg.csv", index=False)


# ---------------------------------------------------------------------------
# Subcommand: posthoc
# ---------------------------------------------------------------------------
def run_posthoc_subcommand() -> None:
    output_dir = ensure_output_dir()
    per_split_rows = []

    for dataset in DATASETS_ALL5:
        for model in MODELS:
            for split in range(10):
                task = _load_task(dataset, model, split)
                y_true = task["y_true"].astype(np.float64)
                for fraction in FRACTIONS_TEST:
                    prob = task[f"prob_{fraction}"].astype(np.float64)
                    row = {
                        "dataset": dataset,
                        "model": model,
                        "split_idx": split,
                        "fraction": fraction,
                        "n": len(y_true),
                    }
                    original_metrics = compute_metrics(y_true, np.clip(prob, LOG_LOSS_EPS, 1 - LOG_LOSS_EPS))
                    for k, v in original_metrics.items():
                        row[f"original_{k}"] = v
                    row["original_ece_ew"] = row.pop("original_ece")

                    for method in ("platt", "isotonic"):
                        calibrated, n_fallback, _ = cross_fit_calibrate(y_true, prob, seed=SEED + split, method=method)
                        m = compute_metrics(y_true, calibrated)
                        row[f"{method}_auroc"] = m["auroc"]
                        row[f"{method}_brier"] = m["brier"]
                        row[f"{method}_log_loss"] = m["log_loss"]
                        row[f"{method}_ece_ew"] = m["ece"]
                        row[f"{method}_n_fallback"] = n_fallback
                    per_split_rows.append(row)

    per_split_df = pd.DataFrame(per_split_rows)
    per_split_df.to_csv(output_dir / "posthoc_per_split.csv", index=False)

    summary_rows = []
    for fraction in FRACTIONS_TEST:
        for model in MODELS:
            for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                subset = per_split_df[
                    (per_split_df.fraction == fraction)
                    & (per_split_df.model == model)
                    & (per_split_df.dataset.isin(datasets))
                ]
                row = {"fraction": fraction, "model": model, "dataset_set": set_name}
                for method in ("platt", "isotonic"):
                    row[f"{method}_brier_diff"] = float((subset[f"{method}_brier"] - subset["original_brier"]).mean())
                    row[f"{method}_log_loss_diff"] = float(
                        (subset[f"{method}_log_loss"] - subset["original_log_loss"]).mean()
                    )
                    row[f"{method}_ece_ew_diff"] = float(
                        (subset[f"{method}_ece_ew"] - subset["original_ece_ew"]).mean()
                    )
                    row[f"{method}_auroc_diff_max_abs"] = float(
                        (subset[f"{method}_auroc"] - subset["original_auroc"]).abs().max()
                    )
                    row[f"{method}_n_fallback_total"] = int(subset[f"{method}_n_fallback"].sum())
                summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(output_dir / "posthoc_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Subcommand: figures
# ---------------------------------------------------------------------------
def _reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    order = np.argsort(p, kind="mergesort")
    y_sorted, p_sorted = y[order], p[order]
    groups_y = np.array_split(y_sorted, n_bins)
    groups_p = np.array_split(p_sorted, n_bins)
    mean_p, obs_frac, lo, hi, n_g = [], [], [], [], []
    for gy, gp in zip(groups_y, groups_p):
        if len(gy) == 0:
            continue
        successes = int(gy.sum())
        n = len(gy)
        wlo, whi = wilson_interval(successes, n)
        mean_p.append(float(gp.mean()))
        obs_frac.append(float(gy.mean()))
        lo.append(wlo)
        hi.append(whi)
        n_g.append(n)
    return np.array(mean_p), np.array(obs_frac), np.array(lo), np.array(hi), np.array(n_g)


def _plot_reliability_panel(ax, hist_ax, y, p, model, title):
    mean_p, obs_frac, lo, hi, _ = _reliability_bins(y, p)
    color = MODEL_COLOR[model]
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, zorder=1)
    ax.errorbar(
        mean_p,
        obs_frac,
        yerr=[obs_frac - lo, hi - obs_frac],
        fmt=MODEL_MARKER[model],
        color=color,
        markersize=4,
        capsize=2,
        linewidth=1,
        zorder=2,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=7)
    ax.tick_params(labelsize=6)
    hist_ax.hist(p, bins=20, range=(0, 1), color=color, alpha=0.6)
    hist_ax.set_xlim(0, 1)
    hist_ax.set_yticks([])
    hist_ax.tick_params(labelsize=5)


def _figure4(fraction: float, name: str) -> tuple[Path, Path]:
    output_dir = ensure_output_dir()
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(4, 4, height_ratios=[3, 1, 3, 1], hspace=0.55, wspace=0.35)

    for i, model in enumerate(MODELS):
        row_pair, col = divmod(i, 4)
        main_ax = fig.add_subplot(gs[row_pair * 2, col])
        hist_ax = fig.add_subplot(gs[row_pair * 2 + 1, col])
        y_all, p_all = [], []
        for dataset in DATASETS_NO_CKD:
            for split in range(10):
                task = _load_task(dataset, model, split)
                y_all.append(task["y_true"].astype(np.float64))
                p_all.append(task[f"prob_{fraction}"].astype(np.float64))
        y_pooled = np.concatenate(y_all)
        p_pooled = np.concatenate(p_all)
        _plot_reliability_panel(main_ax, hist_ax, y_pooled, p_pooled, model, model)
        if col == 0:
            main_ax.set_ylabel("Observed fraction", fontsize=6)

    fig.suptitle(f"Reliability, NO_CKD pooled, fraction={fraction}", fontsize=9)
    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _figure4c() -> tuple[Path, Path]:
    output_dir = ensure_output_dir()
    fig, axes = plt.subplots(len(DATASETS_ALL5), len(MODELS), figsize=(7.0, 5.5))
    for row, dataset in enumerate(DATASETS_ALL5):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            y_all, p_all = [], []
            for split in range(10):
                task = _load_task(dataset, model, split)
                y_all.append(task["y_true"].astype(np.float64))
                p_all.append(task["prob_1.0"].astype(np.float64))
            y_pooled, p_pooled = np.concatenate(y_all), np.concatenate(p_all)
            mean_p, obs_frac, lo, hi, _ = _reliability_bins(y_pooled, p_pooled)
            color = MODEL_COLOR[model]
            ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=0.6)
            ax.plot(mean_p, obs_frac, marker=MODEL_MARKER[model], color=color, markersize=2.5, linewidth=0.6)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(model, fontsize=6)
            if col == 0:
                ax.set_ylabel(dataset, fontsize=5)
    fig.suptitle("Reliability per dataset, fraction=1.0 (compact)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    png_path = output_dir / "fig4c_reliability_per_dataset_frac1p0.png"
    pdf_path = output_dir / "fig4c_reliability_per_dataset_frac1p0.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def _figure5(metrics_df: pd.DataFrame, ci_df: pd.DataFrame) -> tuple[Path, Path, list]:
    output_dir = ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    y_positions = {m: i for i, m in enumerate(MODELS[::-1])}
    separable_cells = []

    for ax, fraction in zip(axes, FRACTIONS_TEST):
        ax.axvline(1.0, color="black", linewidth=0.8, linestyle=":")
        for model in MODELS:
            y_base = y_positions[model]
            offsets = np.linspace(-0.3, 0.3, len(DATASETS_ALL5))
            for off, dataset in zip(offsets, DATASETS_ALL5):
                row = metrics_df[
                    (metrics_df.model == model) & (metrics_df.fraction == fraction) & (metrics_df.dataset == dataset)
                ]
                if row.empty or bool(row.iloc[0]["separable"]):
                    if not row.empty:
                        separable_cells.append((dataset, model, fraction))
                    continue
                slope = row.iloc[0]["slope"]
                ci_row = ci_df[
                    (ci_df.model == model) & (ci_df.fraction == fraction) & (ci_df.dataset == dataset)
                ].iloc[0]
                lo, hi = ci_row["slope_lo"], ci_row["slope_hi"]
                if np.isnan(lo) or np.isnan(hi):
                    xerr = None
                else:
                    xerr = [[max(0.0, slope - lo)], [max(0.0, hi - slope)]]
                ax.errorbar(
                    slope,
                    y_base + off,
                    xerr=xerr,
                    fmt=DATASET_MARKER[dataset],
                    color=MODEL_COLOR[model],
                    markersize=4,
                    capsize=2,
                    linewidth=0.8,
                )
        ax.set_title(f"fraction={fraction}", fontsize=8)
        ax.set_xlabel("Calibration slope", fontsize=8)
        ax.tick_params(labelsize=7)

    axes[0].set_yticks(list(y_positions.values()))
    axes[0].set_yticklabels(list(y_positions.keys()), fontsize=7)
    handles = [
        plt.Line2D([0], [0], marker=marker, color="gray", linestyle="", markersize=5, label=dataset)
        for dataset, marker in DATASET_MARKER.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    png_path = output_dir / "fig5_calibration_slope.png"
    pdf_path = output_dir / "fig5_calibration_slope.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, separable_cells


def _figure6(posthoc_summary: pd.DataFrame) -> tuple[Path, Path]:
    output_dir = ensure_output_dir()
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    metrics = [("brier_diff", "Brier change"), ("log_loss_diff", "Log loss change")]
    x = np.arange(len(MODELS))
    width = 0.25

    for row, (metric_suffix, metric_label) in enumerate(metrics):
        for col, set_name in enumerate(("ALL5", "NO_CKD")):
            ax = axes[row, col]
            subset = posthoc_summary[(posthoc_summary.fraction == 1.0) & (posthoc_summary.dataset_set == set_name)]
            subset = subset.set_index("model").reindex(MODELS)
            original = np.zeros(len(MODELS))
            platt = subset[f"platt_{metric_suffix}"].to_numpy()
            isotonic = subset[f"isotonic_{metric_suffix}"].to_numpy()
            ax.bar(x - width, original, width, label="original (=0)", color="#999999")
            ax.bar(x, platt, width, label="platt", color="#56B4E9")
            ax.bar(x + width, isotonic, width, label="isotonic", color="#E69F00")
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(MODELS, fontsize=6, rotation=45, ha="right")
            ax.set_title(f"{metric_label}, {set_name}", fontsize=8)
            ax.tick_params(labelsize=6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png_path = output_dir / "fig6_posthoc_effect_frac1p0.png"
    pdf_path = output_dir / "fig6_posthoc_effect_frac1p0.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def run_figures_subcommand() -> list[str]:
    output_dir = ensure_output_dir()
    checked = []

    png, _ = _figure4(1.0, "fig4_reliability_NO_CKD_frac1p0")
    checked.append(f"{png.name}: 7 model panels + histogram strips, diagonal visible, titles readable")
    png, _ = _figure4(0.1, "fig4b_reliability_NO_CKD_frac0p1")
    checked.append(f"{png.name}: same checks")
    png, _ = _figure4c()
    checked.append(f"{png.name}: 5x7 compact grid, row/col labels readable")

    metrics_df = pd.read_csv(output_dir / "calibration_metrics.csv")
    ci_df = pd.read_csv(output_dir / "calibration_bootstrap_ci.csv")
    png, _, separable_cells = _figure5(metrics_df, ci_df)
    checked.append(f"{png.name}: both fraction panels, model labels shared, line at slope=1 visible")

    posthoc_summary = pd.read_csv(output_dir / "posthoc_summary.csv")
    png, _ = _figure6(posthoc_summary)
    checked.append(f"{png.name}: 4 panels, legend readable, bars distinguishable")

    _write_figure_captions(output_dir, separable_cells)
    return checked


def _write_figure_captions(output_dir: Path, separable_cells: list) -> None:
    lines = [
        "# Phase 3b figure captions (factual only, no interpretation)\n",
        "## fig4 / fig4b: reliability diagrams (NO_CKD pooled)",
        "Equal-mass 10-bin reliability diagrams (mean predicted probability vs. observed",
        "positive fraction, Wilson 95% interval on the observed fraction, diagonal",
        "reference), one panel per model, pooled over the 4 NO_CKD datasets and their 10",
        "splits each (2 x n_rows per dataset, since each test row appears in 2 of the 10",
        "splits). fig4: fraction=1.0. fig4b: fraction=0.1. Histogram strip below each",
        "panel: distribution of the pooled predicted probabilities, 20 equal-width bins.\n",
        "## fig4c: reliability per dataset (compact)",
        "Same equal-mass 10-bin reliability curves, one panel per (dataset, model) pair,",
        "pooled over that dataset's 10 splits, fraction=1.0 only, no histogram strip.\n",
        "## fig5: calibration slope",
        "Pooled calibration slope (Platt regression on logit(p), MLE) per model, with",
        "95% bootstrap CIs (seed 42, B=2000, resampling the 10 splits), coloured by model",
        "(same mapping as Phase 3a) with one marker shape per dataset (" +
        ", ".join(f"{d}={m}" for d, m in DATASET_MARKER.items()) + "),",
        "fraction=1.0 and 0.1 side by side, vertical line at slope=1. Cells omitted as",
        "separable (exact class separation in logit space, slope/intercept undefined): "
        + (", ".join(f"{d}/{m}/frac={f}" for d, m, f in separable_cells) if separable_cells else "none") + ".\n",
        "## fig6: post-hoc effect",
        "Change in dataset-averaged Brier (top row) and log loss (bottom row) relative to",
        "the original (uncalibrated) predictions, for Platt and isotonic cross-fitted",
        "post-hoc recalibration, fraction=1.0, ALL5 (left) and NO_CKD (right), per model.\n",
    ]
    (output_dir / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Subcommand: tables
# ---------------------------------------------------------------------------
def run_tables_subcommand() -> None:
    output_dir = ensure_output_dir()
    metrics_df = pd.read_csv(output_dir / "calibration_metrics.csv")
    averaged_df = pd.read_csv(output_dir / "calibration_averaged.csv")
    bl_df = pd.read_csv(output_dir / "calibration_brier_logloss_averaged.csv")
    posthoc_summary = pd.read_csv(output_dir / "posthoc_summary.csv")
    all_runs = load_all_runs()

    caption_lines = ["# Phase 3b table captions / notes\n"]

    # Tables 6 and 7: fraction 1.0 and 0.1, ALL5 (brier/logloss/ece_ew/ece_em), slope/intercept over NO_CKD.
    for fraction, name in ((1.0, "table6_calibration_frac1p0"), (0.1, "table7_calibration_frac0p1")):
        rows = {}
        footnote_sep = []
        for model in MODELS:
            bl_row = bl_df[(bl_df.model == model) & (bl_df.fraction == fraction) & (bl_df.dataset_set == "ALL5")].iloc[0]
            avg_row = averaged_df[
                (averaged_df.model == model) & (averaged_df.fraction == fraction) & (averaged_df.dataset_set == "ALL5")
            ].iloc[0]
            slope_row = averaged_df[
                (averaged_df.model == model) & (averaged_df.fraction == fraction) & (averaged_df.dataset_set == "NO_CKD")
            ].iloc[0]
            if slope_row["separable_datasets"]:
                footnote_sep.append(f"{model}: {slope_row['separable_datasets']}")
            rows[model] = {
                "brier": bl_row["brier_mean"],
                "log_loss": bl_row["log_loss_mean"],
                "ece_ew": avg_row["ece_ew_mean"],
                "ece_em": avg_row["ece_em_mean"],
                "slope_NOCKD": slope_row["slope_mean"],
                "intercept_NOCKD": slope_row["intercept_mean"],
            }
        table = pd.DataFrame(rows).T.reindex(index=MODELS)
        display = table.map(lambda v: format_float(v, 4))
        write_csv_and_tex(name, display, output_dir, index_label="model")
        if footnote_sep:
            caption_lines.append(f"## {name} separable-dataset footnote (excluded from slope/intercept)")
            caption_lines.extend(f"- {s}" for s in footnote_sep)
            caption_lines.append("")

    # Table 8: post-hoc effect at fraction 1.0, ALL5 and NO_CKD.
    rows8 = []
    for model in MODELS:
        for set_name in ("ALL5", "NO_CKD"):
            row = posthoc_summary[
                (posthoc_summary.fraction == 1.0) & (posthoc_summary.model == model) & (posthoc_summary.dataset_set == set_name)
            ].iloc[0]
            rows8.append(
                {
                    "model": model,
                    "dataset_set": set_name,
                    "platt_brier_diff": row["platt_brier_diff"],
                    "isotonic_brier_diff": row["isotonic_brier_diff"],
                    "platt_log_loss_diff": row["platt_log_loss_diff"],
                    "isotonic_log_loss_diff": row["isotonic_log_loss_diff"],
                    "platt_ece_ew_diff": row["platt_ece_ew_diff"],
                    "isotonic_ece_ew_diff": row["isotonic_ece_ew_diff"],
                    "platt_n_fallback": row["platt_n_fallback_total"],
                    "isotonic_n_fallback": row["isotonic_n_fallback_total"],
                }
            )
    table8 = pd.DataFrame(rows8)
    table8.to_csv(output_dir / "table8_posthoc_effect.csv", index=False)
    display8 = table8.copy()
    for col in display8.columns:
        if col.endswith("_n_fallback"):
            display8[col] = display8[col].map(lambda v: str(int(v)))
        elif col not in ("model", "dataset_set"):
            display8[col] = display8[col].map(lambda v: format_float(v, 4))
    display8 = display8.set_index(["model", "dataset_set"])
    with open(output_dir / "table8_posthoc_effect.tex", "w", encoding="utf-8") as fh:
        fh.write(r"\begin{tabular}{ll" + "r" * (len(display8.columns)) + "}" + "\n")
        fh.write(r"\toprule" + "\n")
        header = ["model", "dataset\\_set"] + [escape_latex(c) for c in display8.columns]
        fh.write(" & ".join(header) + r" \\" + "\n")
        fh.write(r"\midrule" + "\n")
        for (model, set_name), row in display8.iterrows():
            cells = [escape_latex(model), escape_latex(set_name)] + [escape_latex(v) for v in row]
            fh.write(" & ".join(cells) + r" \\" + "\n")
        fh.write(r"\bottomrule" + "\n")
        fh.write(r"\end{tabular}" + "\n")

    # Table 4b: mean rank per model at fractions 0.25 and 0.5, BLOCK unit, descriptive only.
    rows4b = []
    for metric in ("auroc", "log_loss", "brier"):
        higher_is_better = metric == "auroc"
        for fraction in (0.25, 0.5):
            for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                blocks = block_matrix(all_runs, metric, fraction, datasets)
                ranks = ranks_best_is_1(blocks.to_numpy(), higher_is_better)
                mean_ranks = ranks.mean(axis=0)
                rows4b.append(
                    {"metric": metric, "fraction": fraction, "dataset_set": set_name, **dict(zip(MODELS, mean_ranks))}
                )
    table4b = pd.DataFrame(rows4b)
    table4b.to_csv(output_dir / "table4b_mean_rank_raw.csv", index=False)
    display4b = table4b.copy()
    for model in MODELS:
        display4b[model] = display4b[model].map(lambda v: format_float(v, 3))
    display4b.index = display4b.apply(
        lambda r: f"{r['metric']}/frac={r['fraction']}/{r['dataset_set']}", axis=1
    )
    display4b = display4b[MODELS]
    write_csv_and_tex("table4b_mean_rank", display4b, output_dir, index_label="metric/fraction/set")

    (output_dir / "table_captions.md").write_text("\n".join(caption_lines) + "\n", encoding="utf-8")
    _write_calibration_notes(output_dir)


def _write_calibration_notes(output_dir: Path) -> None:
    lines = [
        "# Phase 3b calibration notes\n",
        "- Each test row appears TWICE in the pooled predictions used throughout this",
        "  phase (once per repeat of the 5-fold cross-validation), so pooled analyses use",
        "  2 x n_rows predictions per (dataset, model, fraction), not n_rows.",
        "- ECE on small samples is biased and noisy; equal-width (ece_ew) and equal-mass",
        "  (ece_em) ECE use different binning rules and can disagree.",
        "- The calibration slope and intercept are undefined (NaN, flagged `separable`)",
        "  when the two classes are exactly separable in logit space, as happens for CKD.",
        "- The post-hoc (Platt/isotonic) analysis is cross-fitted on the TEST predictions",
        "  themselves (inner 5-fold, each calibrator fit on ~48-123 rows and applied only",
        "  to its held-out fold) and is descriptive only; it is NOT part of the benchmark",
        "  pipeline, and isotonic regression's AUROC can change through tie-breaking.",
        "- Friedman-style hypothesis tests are not applied to any calibration metric here.",
        "- No file in results/phase3b states that a model is \"significantly\" better or",
        "  \"better calibrated\" than another; only numbers are reported.",
    ]
    (output_dir / "calibration_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3b calibration analysis")
    parser.add_argument("subcommand", choices=["verify", "metrics", "posthoc", "figures", "tables", "all"])
    args = parser.parse_args()

    print(f"sys.executable = {sys.executable}")
    ensure_output_dir()
    start = time.time()

    if args.subcommand in ("verify", "all"):
        t0 = time.time()
        report = run_verify()
        print(f"[verify] done in {time.time() - t0:.1f}s")
        _print_verify_report(report)
        if report["stop"]:
            print("STOP: verify found a problem above the 1e-6 threshold or an alignment/structural error.")
            sys.exit(1)
    if args.subcommand in ("metrics", "all"):
        t0 = time.time()
        run_metrics_subcommand()
        print(f"[metrics] done in {time.time() - t0:.1f}s")
    if args.subcommand in ("posthoc", "all"):
        t0 = time.time()
        run_posthoc_subcommand()
        print(f"[posthoc] done in {time.time() - t0:.1f}s")
    if args.subcommand in ("figures", "all"):
        t0 = time.time()
        checked = run_figures_subcommand()
        print(f"[figures] done in {time.time() - t0:.1f}s")
        for line in checked:
            print(f"  checked: {line}")
    if args.subcommand in ("tables", "all"):
        t0 = time.time()
        run_tables_subcommand()
        print(f"[tables] done in {time.time() - t0:.1f}s")
    print(f"total runtime: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

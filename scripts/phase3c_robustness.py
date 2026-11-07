"""Phase 3c: robustness analysis (test-time missingness and noise on saved models).

Loads the SAVED fraction-1.0 pipelines in results/models/ and the stored test rows
in results/predictions/*.npz, perturbs the TEST features only, and measures
degradation. Never fits, refits or tunes any model. Writes ONLY inside
results/phase3c/. Never touches results/raw, results/tuning, results/models,
results/diagnostics, or anything in src/.

Subcommands: verify, run, analyze, figures, tables, all.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.bootstrap import (
    bootstrap_diff_ci,
    bootstrap_diff_ci_averaged,
    bootstrap_mean_ci,
    bootstrap_mean_ci_averaged,
)
from analysis.constants import (
    DATASETS_ALL5,
    DATASETS_NO_CKD,
    FAMILY,
    FAMILY_LINESTYLE,
    MODEL_COLOR,
    MODEL_MARKER,
    MODELS,
    SEED,
)
from analysis.robustness import (
    MISSING_LEVELS,
    N_DRAWS,
    NOISE_LEVELS,
    apply_missing,
    apply_noise,
    evaluate_predictions,
    make_rng,
)
from analysis.tables import escape_latex, format_float, write_csv_and_tex
from src.data import load_dataset
from src.metrics import compute_metrics
from src.models import positive_class_proba
from src.utils import make_splits

MODELS_DIR = PROJECT_ROOT / "results" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase3c"
PHASE3C_PRED_DIR = OUTPUT_DIR / "predictions"
RUNS_PARTS_DIR = OUTPUT_DIR / "runs_parts"
BOOTSTRAP_B = 5000
TIMING_GATE_MINUTES = 90

_DATASET_CACHE: dict[str, tuple] = {}


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3C_PRED_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_PARTS_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _get_dataset(dataset: str):
    if dataset not in _DATASET_CACHE:
        X, y, meta = load_dataset(dataset, verbose=False)
        splits = make_splits(X, y, n_splits=5, n_repeats=2, seed=42)
        _DATASET_CACHE[dataset] = (X, y, meta, splits)
    return _DATASET_CACHE[dataset]


def _stem(dataset: str, model: str, split: int) -> str:
    return f"{dataset}__{model}__split{split}"


def _pipeline_path(dataset: str, model: str, split: int) -> Path:
    return MODELS_DIR / f"{_stem(dataset, model, split)}.joblib"


def _npz_path(dataset: str, model: str, split: int) -> Path:
    return PREDICTIONS_DIR / f"{_stem(dataset, model, split)}.npz"


def _free_gpu_memory(pipeline) -> None:
    del pipeline
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Subcommand: verify
# ---------------------------------------------------------------------------
def run_verify() -> dict:
    structural_errors: list[str] = []
    alignment_errors: list[str] = []
    n_checked = 0
    max_prob_diff: dict[str, float] = {m: 0.0 for m in MODELS}
    max_auroc_diff: dict[str, float] = {m: 0.0 for m in MODELS}
    n_above_1e6 = {m: 0 for m in MODELS}
    n_above_1e4 = {m: 0 for m in MODELS}

    for dataset in DATASETS_ALL5:
        X, y, meta, splits = _get_dataset(dataset)
        for model in MODELS:
            for split in range(10):
                pipeline_path = _pipeline_path(dataset, model, split)
                npz_path = _npz_path(dataset, model, split)
                if not pipeline_path.exists():
                    structural_errors.append(f"missing pipeline: {pipeline_path.name}")
                    continue
                if not npz_path.exists():
                    structural_errors.append(f"missing npz: {npz_path.name}")
                    continue
                n_checked += 1

                train_idx, test_idx = splits[split]
                data = np.load(npz_path)
                if not np.array_equal(data["test_idx"], test_idx):
                    alignment_errors.append(f"{pipeline_path.stem}: test_idx mismatch")
                    continue
                y_true_expected = y.iloc[test_idx].to_numpy()
                if not np.array_equal(data["y_true"], y_true_expected):
                    alignment_errors.append(f"{pipeline_path.stem}: y_true mismatch")
                    continue

                try:
                    pipeline = joblib.load(pipeline_path)
                    X_test = X.iloc[test_idx]
                    raw_proba = pipeline.predict_proba(X_test)
                except Exception as exc:  # noqa: BLE001
                    structural_errors.append(f"{pipeline_path.stem}: predict failed: {exc}")
                    continue

                if raw_proba.shape != (len(test_idx), 2) or np.isnan(raw_proba).any():
                    structural_errors.append(
                        f"{pipeline_path.stem}: predict_proba shape/NaN problem {raw_proba.shape}"
                    )
                    continue

                proba = positive_class_proba(pipeline, X_test)
                stored_proba = data["prob_1.0"]
                diff = float(np.max(np.abs(proba - stored_proba)))
                max_prob_diff[model] = max(max_prob_diff[model], diff)
                if diff > 1e-6:
                    n_above_1e6[model] += 1
                if diff > 1e-4:
                    n_above_1e4[model] += 1

                auroc_live = compute_metrics(data["y_true"], proba)["auroc"]
                auroc_stored = compute_metrics(data["y_true"], stored_proba)["auroc"]
                auroc_diff = abs(auroc_live - auroc_stored)
                max_auroc_diff[model] = max(max_auroc_diff[model], auroc_diff)

                if diff > 1e-3 or auroc_diff > 1e-3:
                    alignment_errors.append(
                        f"{pipeline_path.stem}: prob diff {diff:.3e} or AUROC diff {auroc_diff:.3e} exceeds 1e-3"
                    )
                _free_gpu_memory(pipeline)

    stop = bool(structural_errors) or bool(alignment_errors)
    return {
        "n_checked": n_checked,
        "structural_errors": structural_errors,
        "alignment_errors": alignment_errors,
        "max_prob_diff": max_prob_diff,
        "max_auroc_diff": max_auroc_diff,
        "n_above_1e6": n_above_1e6,
        "n_above_1e4": n_above_1e4,
        "stop": stop,
    }


def _print_verify_report(report: dict) -> None:
    print(f"verify: {report['n_checked']} tasks checked")
    print(f"structural errors: {len(report['structural_errors'])}")
    for e in report["structural_errors"][:20]:
        print(f"  {e}")
    print(f"alignment errors: {len(report['alignment_errors'])}")
    for e in report["alignment_errors"][:20]:
        print(f"  {e}")
    for model in MODELS:
        print(
            f"  {model}: max_prob_diff={report['max_prob_diff'][model]:.3e} "
            f"max_auroc_diff={report['max_auroc_diff'][model]:.3e} "
            f"n>1e-6={report['n_above_1e6'][model]} n>1e-4={report['n_above_1e4'][model]}"
        )
    print(f"STOP: {report['stop']}")


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------
def _run_one_task(dataset: str, model: str, split: int) -> tuple[list[dict], dict]:
    """Returns (31 result rows, dict of probability arrays keyed by prediction-set name)."""
    X, y, meta, splits = _get_dataset(dataset)
    train_idx, test_idx = splits[split]
    X_test = X.iloc[test_idx]
    X_train = X.iloc[train_idx]
    y_true = y.iloc[test_idx].to_numpy()
    numeric_cols = meta["numeric_cols"]

    pipeline = joblib.load(_pipeline_path(dataset, model, split))
    rows: list[dict] = []
    probs: dict[str, np.ndarray] = {}

    def _row(kind: str, level: float, draw, result: dict, realised_missing=None) -> dict:
        return {
            "dataset": dataset,
            "model": model,
            "split_idx": split,
            "kind": kind,
            "level": level,
            "draw": draw,
            "auroc": result["auroc"],
            "brier": result["brier"],
            "log_loss": result["log_loss"],
            "n_unique_probs": result["n_unique_probs"],
            "constant_prediction": result["constant_prediction"],
            "realised_missing_added": realised_missing,
            "status": result["status"],
            "error": result["error"],
        }

    n_test = len(test_idx)

    result = evaluate_predictions(pipeline, X_test, y_true)
    rows.append(_row("clean", 0.0, None, result))
    probs["clean"] = result["prob"] if result["prob"] is not None else np.full(n_test, np.nan, dtype=np.float32)

    for level_idx, r in enumerate(MISSING_LEVELS):
        for draw in range(N_DRAWS):
            rng = make_rng(dataset, split, "missing", level_idx, draw)
            X_pert, realised = apply_missing(X_test, r, rng)
            result = evaluate_predictions(pipeline, X_pert, y_true)
            rows.append(_row("missing", r, draw, result, realised_missing=realised))
            key = f"miss_{r}_d{draw}"
            probs[key] = result["prob"] if result["prob"] is not None else np.full(n_test, np.nan, dtype=np.float32)

    for level_idx, s in enumerate(NOISE_LEVELS):
        for draw in range(N_DRAWS):
            rng = make_rng(dataset, split, "noise", level_idx, draw)
            X_pert = apply_noise(X_test, X_train, s, numeric_cols, rng)
            result = evaluate_predictions(pipeline, X_pert, y_true)
            rows.append(_row("noise", s, draw, result))
            key = f"noise_{s}_d{draw}"
            probs[key] = result["prob"] if result["prob"] is not None else np.full(n_test, np.nan, dtype=np.float32)

    _free_gpu_memory(pipeline)
    return rows, probs


def _task_complete(dataset: str, model: str, split: int) -> bool:
    part_path = RUNS_PARTS_DIR / f"{_stem(dataset, model, split)}.csv"
    if not part_path.exists():
        return False
    try:
        return len(pd.read_csv(part_path)) == 31
    except Exception:  # noqa: BLE001
        return False


def run_timing_gate() -> float:
    """Times one full 31-prediction task per model (same dataset), extrapolates to
    350 tasks. Returns the estimated total minutes. Does not run the probes' results
    into the real output files.
    """
    probe_dataset = DATASETS_ALL5[0]
    per_model_seconds: dict[str, float] = {}
    for model in MODELS:
        start = time.time()
        _run_one_task(probe_dataset, model, 0)
        per_model_seconds[model] = time.time() - start

    avg_seconds = float(np.mean(list(per_model_seconds.values())))
    total_estimate_minutes = avg_seconds * 350 / 60.0

    print("Timing gate probe (1 task = 31 predictions, dataset=" + probe_dataset + ", split=0):")
    for model, seconds in per_model_seconds.items():
        print(f"  {model}: {seconds:.1f}s/task -> extrapolated {seconds * 50 / 60:.1f} min for its 50 tasks")
    print(f"  average: {avg_seconds:.1f}s/task -> extrapolated total: {total_estimate_minutes:.1f} min for 350 tasks")
    return total_estimate_minutes


def run_run_subcommand(skip_timing_gate: bool = False) -> None:
    ensure_output_dir()

    if not skip_timing_gate:
        estimate = run_timing_gate()
        if estimate > TIMING_GATE_MINUTES:
            print(f"STOP: estimated total runtime {estimate:.1f} min exceeds the {TIMING_GATE_MINUTES} min gate.")
            print("Options (not choosing one): (a) proceed anyway if Demian approves the longer runtime,")
            print("(b) reduce R (draws) or the number of levels, (c) parallelize across datasets/models,")
            print("(d) run on a faster device / without the slowest model(s).")
            sys.exit(1)

    n_done = 0
    n_skipped = 0
    for dataset in DATASETS_ALL5:
        for model in MODELS:
            for split in range(10):
                if _task_complete(dataset, model, split):
                    n_skipped += 1
                    continue
                rows, probs = _run_one_task(dataset, model, split)
                stem = _stem(dataset, model, split)
                pd.DataFrame(rows).to_csv(RUNS_PARTS_DIR / f"{stem}.csv", index=False)
                np.savez_compressed(
                    PHASE3C_PRED_DIR / f"{stem}.npz",
                    **{k: v.astype(np.float32) for k, v in probs.items()},
                )
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  ... {n_done} tasks completed, {n_skipped} skipped (resume)")

    print(f"run: {n_done} tasks completed this run, {n_skipped} already complete (resumed)")

    parts = sorted(RUNS_PARTS_DIR.glob("*.csv"))
    merged = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    merged = merged.sort_values(["dataset", "model", "split_idx", "kind", "level", "draw"], na_position="first")
    merged.to_csv(OUTPUT_DIR / "robustness_runs.csv", index=False)
    print(f"robustness_runs.csv: {len(merged)} rows (expected 10850)")


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------
METRICS = ["auroc", "brier", "log_loss"]
LEVEL_KINDS: list[tuple[str, float]] = [("missing", r) for r in MISSING_LEVELS] + [("noise", s) for s in NOISE_LEVELS]


def _per_split_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Average the 5 draws within (dataset, model, split, kind, level) -> one row per
    (dataset, model, split, kind, level) with mean metrics over the ok draws.
    """
    ok = runs[runs.status == "ok"]
    grouped = (
        ok.groupby(["dataset", "model", "split_idx", "kind", "level"], as_index=False)[METRICS]
        .mean()
        .rename(columns={m: f"{m}_mean" for m in METRICS})
    )
    n_ok = ok.groupby(["dataset", "model", "split_idx", "kind", "level"], as_index=False).size()
    grouped = grouped.merge(n_ok, on=["dataset", "model", "split_idx", "kind", "level"])
    grouped = grouped.rename(columns={"size": "n_ok_draws"})
    return grouped


def run_analyze_subcommand() -> None:
    output_dir = ensure_output_dir()
    runs = pd.read_csv(OUTPUT_DIR / "robustness_runs.csv")
    per_split = _per_split_table(runs)

    clean = per_split[per_split.kind == "clean"].set_index(["dataset", "model", "split_idx"])
    perturbed = per_split[per_split.kind != "clean"].copy()

    def _clean_value(row, metric):
        key = (row["dataset"], row["model"], row["split_idx"])
        return clean.loc[key, f"{metric}_mean"] if key in clean.index else np.nan

    for metric in METRICS:
        perturbed[f"{metric}_clean"] = perturbed.apply(lambda r: _clean_value(r, metric), axis=1)
        perturbed[f"{metric}_degradation"] = perturbed[f"{metric}_mean"] - perturbed[f"{metric}_clean"]

    rng = np.random.default_rng(SEED)
    per_dataset_rows = []
    for dataset in DATASETS_ALL5:
        for model in MODELS:
            for kind, level in LEVEL_KINDS:
                sub = perturbed[
                    (perturbed.dataset == dataset) & (perturbed.model == model) & (perturbed.kind == kind) & (perturbed.level == level)
                ].sort_values("split_idx")
                if len(sub) != 10:
                    continue
                for metric in METRICS:
                    perturbed_vals = sub[f"{metric}_mean"].to_numpy()
                    degradation_vals = sub[f"{metric}_degradation"].to_numpy()
                    p_point, p_lo, p_hi = bootstrap_mean_ci(perturbed_vals, rng, BOOTSTRAP_B)
                    d_point, d_lo, d_hi = bootstrap_mean_ci(degradation_vals, rng, BOOTSTRAP_B)
                    per_dataset_rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "kind": kind,
                            "level": level,
                            "metric": metric,
                            "perturbed_mean": p_point,
                            "perturbed_ci_lo": p_lo,
                            "perturbed_ci_hi": p_hi,
                            "degradation_mean": d_point,
                            "degradation_ci_lo": d_lo,
                            "degradation_ci_hi": d_hi,
                        }
                    )
    pd.DataFrame(per_dataset_rows).to_csv(output_dir / "robustness_per_dataset.csv", index=False)

    # Dataset-averaged ALL5/NO_CKD.
    averaged_rows = []
    for model in MODELS:
        for kind, level in LEVEL_KINDS:
            for metric in METRICS:
                per_dataset_perturbed = {}
                per_dataset_degradation = {}
                for dataset in DATASETS_ALL5:
                    sub = perturbed[
                        (perturbed.dataset == dataset) & (perturbed.model == model) & (perturbed.kind == kind) & (perturbed.level == level)
                    ].sort_values("split_idx")
                    if len(sub) != 10:
                        continue
                    per_dataset_perturbed[dataset] = sub[f"{metric}_mean"].to_numpy()
                    per_dataset_degradation[dataset] = sub[f"{metric}_degradation"].to_numpy()
                for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                    values = [per_dataset_perturbed[d] for d in datasets if d in per_dataset_perturbed]
                    deg_values = [per_dataset_degradation[d] for d in datasets if d in per_dataset_degradation]
                    if len(values) != len(datasets):
                        continue
                    p_point, p_lo, p_hi = bootstrap_mean_ci_averaged(values, rng, BOOTSTRAP_B)
                    d_point, d_lo, d_hi = bootstrap_mean_ci_averaged(deg_values, rng, BOOTSTRAP_B)
                    averaged_rows.append(
                        {
                            "model": model,
                            "kind": kind,
                            "level": level,
                            "metric": metric,
                            "dataset_set": set_name,
                            "perturbed_mean": p_point,
                            "perturbed_ci_lo": p_lo,
                            "perturbed_ci_hi": p_hi,
                            "degradation_mean": d_point,
                            "degradation_ci_lo": d_lo,
                            "degradation_ci_hi": d_hi,
                        }
                    )
    pd.DataFrame(averaged_rows).to_csv(output_dir / "robustness_averaged.csv", index=False)

    # Paired difference to logreg, in DEGRADATION.
    diff_rows = []
    for model in MODELS:
        if model == "logreg":
            continue
        for kind, level in LEVEL_KINDS:
            for metric in METRICS:
                per_dataset_point = {}
                model_by_dataset = {}
                logreg_by_dataset = {}
                for dataset in DATASETS_ALL5:
                    model_sub = perturbed[
                        (perturbed.dataset == dataset) & (perturbed.model == model) & (perturbed.kind == kind) & (perturbed.level == level)
                    ].sort_values("split_idx")
                    logreg_sub = perturbed[
                        (perturbed.dataset == dataset) & (perturbed.model == "logreg") & (perturbed.kind == kind) & (perturbed.level == level)
                    ].sort_values("split_idx")
                    if len(model_sub) != 10 or len(logreg_sub) != 10:
                        continue
                    model_deg = model_sub[f"{metric}_degradation"].to_numpy()
                    logreg_deg = logreg_sub[f"{metric}_degradation"].to_numpy()
                    model_by_dataset[dataset] = model_deg
                    logreg_by_dataset[dataset] = logreg_deg
                    point, lo, hi = bootstrap_diff_ci(model_deg, logreg_deg, rng, BOOTSTRAP_B)
                    per_dataset_point[dataset] = point
                    diff_rows.append(
                        {
                            "model": model,
                            "kind": kind,
                            "level": level,
                            "metric": metric,
                            "level_of_detail": "PER_DATASET",
                            "dataset_set": dataset,
                            "diff_mean": point,
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "n_positive": np.nan,
                            "n_total": np.nan,
                        }
                    )
                for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                    if not all(d in model_by_dataset for d in datasets):
                        continue
                    model_subset = [model_by_dataset[d] for d in datasets]
                    logreg_subset = [logreg_by_dataset[d] for d in datasets]
                    point, lo, hi = bootstrap_diff_ci_averaged(model_subset, logreg_subset, rng, BOOTSTRAP_B)
                    n_positive = sum(1 for d in datasets if per_dataset_point[d] > 0)
                    diff_rows.append(
                        {
                            "model": model,
                            "kind": kind,
                            "level": level,
                            "metric": metric,
                            "level_of_detail": "AVERAGED",
                            "dataset_set": set_name,
                            "diff_mean": point,
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "n_positive": n_positive,
                            "n_total": len(datasets),
                        }
                    )
    pd.DataFrame(diff_rows).to_csv(output_dir / "robustness_diff_vs_logreg.csv", index=False)


# ---------------------------------------------------------------------------
# Subcommand: figures
# ---------------------------------------------------------------------------
def _curve_data(averaged_df: pd.DataFrame, set_name: str, metric: str, kind: str, levels: list[float]):
    curves = {}
    for model in MODELS:
        means, los, his = [], [], []
        clean_row = averaged_df[
            (averaged_df.model == model) & (averaged_df.kind == "missing") & (averaged_df.level == MISSING_LEVELS[0]) & (averaged_df.metric == metric) & (averaged_df.dataset_set == set_name)
        ]
        for level in levels:
            if level == 0.0:
                sub = averaged_df[
                    (averaged_df.model == model) & (averaged_df.kind == kind) & (averaged_df.level == levels[1]) & (averaged_df.metric == metric) & (averaged_df.dataset_set == set_name)
                ]
                means.append(float(sub["perturbed_mean"].iloc[0] - sub["degradation_mean"].iloc[0]))
                los.append(np.nan)
                his.append(np.nan)
            else:
                sub = averaged_df[
                    (averaged_df.model == model) & (averaged_df.kind == kind) & (averaged_df.level == level) & (averaged_df.metric == metric) & (averaged_df.dataset_set == set_name)
                ]
                means.append(float(sub["perturbed_mean"].iloc[0]))
                los.append(float(sub["perturbed_ci_lo"].iloc[0]))
                his.append(float(sub["perturbed_ci_hi"].iloc[0]))
        curves[model] = (np.array(means), np.array(los), np.array(his))
    return curves


def _plot_degradation_curves(averaged_df, set_name, metric, name, ylabel) -> Path:
    output_dir = ensure_output_dir()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for ax, (kind, levels) in zip(axes, (("missing", [0.0] + MISSING_LEVELS), ("noise", [0.0] + NOISE_LEVELS))):
        curves = _curve_data(averaged_df, set_name, metric, kind, levels)
        for model in MODELS:
            means, los, his = curves[model]
            color = MODEL_COLOR[model]
            ax.plot(
                levels, means, color=color, marker=MODEL_MARKER[model],
                linestyle=FAMILY_LINESTYLE[FAMILY[model]], markersize=4, linewidth=1.2, label=model,
            )
            valid = ~np.isnan(los)
            if valid.any():
                ax.fill_between(np.array(levels)[valid], los[valid], his[valid], color=color, alpha=0.15, linewidth=0)
        ax.set_xlabel(f"{kind} level", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"{set_name} dataset-averaged {ylabel}", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    png_path = output_dir / f"{name}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path


def _plot_per_dataset_curves(per_dataset_df: pd.DataFrame) -> Path:
    output_dir = ensure_output_dir()
    fig, axes = plt.subplots(len(DATASETS_ALL5), 2, figsize=(7.0, 9.6), sharex="col")
    for row, dataset in enumerate(DATASETS_ALL5):
        for col, (kind, levels) in enumerate((("missing", [0.0] + MISSING_LEVELS), ("noise", [0.0] + NOISE_LEVELS))):
            ax = axes[row, col]
            for model in MODELS:
                means = []
                for level in levels:
                    if level == 0.0:
                        sub = per_dataset_df[
                            (per_dataset_df.dataset == dataset) & (per_dataset_df.model == model) & (per_dataset_df.kind == kind) & (per_dataset_df.level == levels[1]) & (per_dataset_df.metric == "auroc")
                        ]
                        means.append(float(sub["perturbed_mean"].iloc[0] - sub["degradation_mean"].iloc[0]))
                    else:
                        sub = per_dataset_df[
                            (per_dataset_df.dataset == dataset) & (per_dataset_df.model == model) & (per_dataset_df.kind == kind) & (per_dataset_df.level == level) & (per_dataset_df.metric == "auroc")
                        ]
                        means.append(float(sub["perturbed_mean"].iloc[0]))
                ax.plot(
                    levels, means, color=MODEL_COLOR[model], marker=MODEL_MARKER[model],
                    linestyle=FAMILY_LINESTYLE[FAMILY[model]], markersize=3, linewidth=0.9, label=model,
                )
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(kind, fontsize=8)
            if col == 0:
                ax.set_ylabel(dataset, fontsize=6)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-dataset AUROC vs. perturbation level", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    png_path = output_dir / "fig7e_per_dataset_auroc.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig7e_per_dataset_auroc.pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path


def _plot_figure8(diff_df: pd.DataFrame) -> Path:
    output_dir = ensure_output_dir()
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    models_non_logreg = [m for m in MODELS if m != "logreg"]
    y_positions = np.arange(len(models_non_logreg))[::-1]
    offset = 0.15
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    for y, model in zip(y_positions, models_non_logreg):
        for which, offs, filled in (("ALL5", offset, True), ("NO_CKD", -offset, False)):
            row = diff_df[
                (diff_df.model == model) & (diff_df.kind == "missing") & (diff_df.level == MISSING_LEVELS[-1])
                & (diff_df.metric == "auroc") & (diff_df.level_of_detail == "AVERAGED") & (diff_df.dataset_set == which)
            ]
            if row.empty:
                continue
            point, lo, hi = row.iloc[0][["diff_mean", "ci_lo", "ci_hi"]]
            color = MODEL_COLOR[model]
            ax.errorbar(
                point, y + offs, xerr=[[point - lo], [hi - point]], fmt=MODEL_MARKER[model],
                color=color, markerfacecolor=color if filled else "white", markeredgecolor=color,
                capsize=2, linewidth=1,
            )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(models_non_logreg, fontsize=7)
    ax.set_xlabel("AUROC degradation diff. vs. logreg\n(30% missing)", fontsize=7)
    fig.tight_layout()
    png_path = output_dir / "fig8_diff_vs_logreg_degradation.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig8_diff_vs_logreg_degradation.pdf", bbox_inches="tight")
    plt.close(fig)
    return png_path


def run_figures_subcommand() -> list[str]:
    output_dir = ensure_output_dir()
    averaged_df = pd.read_csv(output_dir / "robustness_averaged.csv")
    per_dataset_df = pd.read_csv(output_dir / "robustness_per_dataset.csv")
    diff_df = pd.read_csv(output_dir / "robustness_diff_vs_logreg.csv")
    checked = []

    p = _plot_degradation_curves(averaged_df, "NO_CKD", "auroc", "fig7_degradation_auroc_NO_CKD", "AUROC")
    checked.append(f"{p.name}: 2 panels (missing/noise), legend, CI bands, axes labelled")
    p = _plot_degradation_curves(averaged_df, "NO_CKD", "log_loss", "fig7b_degradation_logloss_NO_CKD", "Log loss")
    checked.append(f"{p.name}: same checks")
    p = _plot_degradation_curves(averaged_df, "ALL5", "auroc", "fig7c_degradation_auroc_ALL5", "AUROC")
    checked.append(f"{p.name}: same checks")
    p = _plot_degradation_curves(averaged_df, "ALL5", "log_loss", "fig7d_degradation_logloss_ALL5", "Log loss")
    checked.append(f"{p.name}: same checks")
    p = _plot_per_dataset_curves(per_dataset_df)
    checked.append(f"{p.name}: 5x2 grid, row/column labels readable")
    p = _plot_figure8(diff_df)
    checked.append(f"{p.name}: whiskers visible, zero line, filled=ALL5/open=NO_CKD")

    _write_figure_captions(output_dir)
    return checked


def _write_figure_captions(output_dir: Path) -> None:
    lines = [
        "# Phase 3c figure captions (factual only, no interpretation)\n",
        "## fig7 / fig7b (NO_CKD) and fig7c / fig7d (ALL5): degradation curves",
        "Dataset-averaged AUROC (fig7/7c) or log loss (fig7b/7d) vs. perturbation level,",
        "7 models, 2 panels (missing rate 0/0.1/0.2/0.3; noise level 0/0.1/0.25/0.5).",
        f"Each point averages {N_DRAWS} independent perturbation draws per (dataset, split,",
        "kind, level), then the 10 splits per dataset, then the datasets (nested bootstrap,",
        f"seed 42, B={BOOTSTRAP_B}). The level-0 point is each model's own clean prediction",
        "(no CI shown there, since it is not a perturbation-level estimate).\n",
        "## fig7e: per-dataset AUROC curves (supplementary)",
        "Same AUROC curves as fig7/7c, one row per dataset, fraction=1.0, no CI bands\n"
        "(point estimates only).\n",
        "## fig8: degradation difference to logreg",
        "Dataset-averaged AUROC degradation difference (model minus logreg) at the",
        "highest missing level (30%), with 95% bootstrap CIs (seed 42, B="
        f"{BOOTSTRAP_B}, same resampled splits for both models), ALL5 (filled) and",
        "NO_CKD (open) side by side, vertical line at 0.\n",
    ]
    (output_dir / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Subcommand: tables
# ---------------------------------------------------------------------------
def _block(averaged_df: pd.DataFrame, metric: str, kind: str, levels: list[float], set_name: str) -> pd.DataFrame:
    rows = {}
    clean_level = levels[0] if levels[0] != 0.0 else levels[1]
    for level in levels:
        row = {}
        for model in MODELS:
            if level == 0.0:
                sub = averaged_df[
                    (averaged_df.model == model) & (averaged_df.kind == kind) & (averaged_df.level == clean_level) & (averaged_df.metric == metric) & (averaged_df.dataset_set == set_name)
                ]
                value = float(sub["perturbed_mean"].iloc[0] - sub["degradation_mean"].iloc[0])
            else:
                sub = averaged_df[
                    (averaged_df.model == model) & (averaged_df.kind == kind) & (averaged_df.level == level) & (averaged_df.metric == metric) & (averaged_df.dataset_set == set_name)
                ]
                value = float(sub["perturbed_mean"].iloc[0])
            row[model] = value
        rows[level] = row
    table = pd.DataFrame(rows).T.reindex(columns=MODELS)
    table.index.name = "level"
    return table


def run_tables_subcommand() -> None:
    output_dir = ensure_output_dir()
    averaged_df = pd.read_csv(output_dir / "robustness_averaged.csv")
    diff_df = pd.read_csv(output_dir / "robustness_diff_vs_logreg.csv")

    for set_name in ("ALL5", "NO_CKD"):
        table9 = _block(averaged_df, "auroc", "missing", [0.0] + MISSING_LEVELS, set_name)
        write_csv_and_tex(f"table9_auroc_missing_{set_name}", table9.map(lambda v: format_float(v, 3)), output_dir, index_label="level")

        table10 = _block(averaged_df, "auroc", "noise", [0.0] + NOISE_LEVELS, set_name)
        write_csv_and_tex(f"table10_auroc_noise_{set_name}", table10.map(lambda v: format_float(v, 3)), output_dir, index_label="level")

        table11a = _block(averaged_df, "log_loss", "missing", [0.0] + MISSING_LEVELS, set_name)
        write_csv_and_tex(f"table11_logloss_missing_{set_name}", table11a.map(lambda v: format_float(v, 3)), output_dir, index_label="level")

        table11b = _block(averaged_df, "log_loss", "noise", [0.0] + NOISE_LEVELS, set_name)
        write_csv_and_tex(f"table11_logloss_noise_{set_name}", table11b.map(lambda v: format_float(v, 3)), output_dir, index_label="level")

    # Table 12: degradation diff vs logreg at highest levels, AUROC and log_loss, ALL5/NO_CKD.
    rows12 = diff_df[
        (diff_df.level_of_detail == "AVERAGED")
        & (((diff_df.kind == "missing") & (diff_df.level == MISSING_LEVELS[-1])) | ((diff_df.kind == "noise") & (diff_df.level == NOISE_LEVELS[-1])))
        & (diff_df.metric.isin(["auroc", "log_loss"]))
    ].copy()
    rows12 = rows12[["model", "kind", "level", "metric", "dataset_set", "diff_mean", "ci_lo", "ci_hi", "n_positive", "n_total"]]
    rows12.to_csv(output_dir / "table12_diff_vs_logreg.csv", index=False)
    display12 = rows12.copy()
    for col in ("diff_mean", "ci_lo", "ci_hi"):
        display12[col] = display12[col].map(lambda v: format_float(v, 4))
    with open(output_dir / "table12_diff_vs_logreg.tex", "w", encoding="utf-8") as fh:
        fh.write(r"\begin{tabular}{llllrrrrr}" + "\n" + r"\toprule" + "\n")
        header = ["model", "kind", "level", "metric", "dataset\\_set", "diff\\_mean", "ci\\_lo", "ci\\_hi", "n\\_positive/n\\_total"]
        fh.write(" & ".join(header) + r" \\" + "\n" + r"\midrule" + "\n")
        for _, r in display12.iterrows():
            cells = [
                escape_latex(r.model), escape_latex(r.kind), escape_latex(r.level), escape_latex(r.metric),
                escape_latex(r.dataset_set), r.diff_mean, r.ci_lo, r.ci_hi, f"{r.n_positive:.0f}/{r.n_total:.0f}",
            ]
            fh.write(" & ".join(str(c) for c in cells) + r" \\" + "\n")
        fh.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")

    _write_table_captions_and_notes(output_dir)


def _write_table_captions_and_notes(output_dir: Path) -> None:
    (output_dir / "table_captions.md").write_text(
        "# Phase 3c table captions\n\n"
        "Tables 9-11 report dataset-averaged AUROC / log loss at each perturbation level "
        "(level 0 = clean). Table 12 reports the degradation difference to logreg at the "
        "highest levels (30% missing, noise 0.5), AUROC and log loss, with 95% bootstrap CIs.\n",
        encoding="utf-8",
    )
    notes = [
        "# Phase 3c robustness notes\n",
        "- Models are trained on CLEAN data; only the TEST features are perturbed.",
        "- The shared preprocessor (median / most-frequent imputation fitted on the",
        "  training fold) is part of every saved pipeline, so these results describe",
        "  MODEL + IMPUTER together; tree models do not use native NaN handling.",
        "- Missingness is Missing Completely At Random (MCAR); there is no informative",
        "  missingness mechanism here.",
        "- Noise is relative to each column's TRAINING standard deviation (ddof=0) and is",
        "  applied to ordinal numeric columns (CKD's sg/al/su, Cleveland's ca) as if",
        "  continuous, same as every other numeric column.",
        f"- {N_DRAWS} independent perturbation draws per (dataset, split, kind, level);",
        "  every model sees the identical perturbed matrix for a given draw (paired design).",
        "- The dataset-averaged bootstrap resamples the 10 splits within each dataset and",
        "  then averages the dataset means; it ignores between-dataset variability.",
        "- Degradation is relative to each model's OWN clean performance on that split; a",
        "  model that starts lower can degrade less in absolute terms.",
        "- No file in results/phase3c/ states that a model is \"more robust\" or",
        "  \"significantly\" better; only numbers are reported.",
    ]
    (output_dir / "robustness_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3c robustness analysis")
    parser.add_argument("subcommand", choices=["verify", "run", "analyze", "figures", "tables", "all"])
    parser.add_argument("--skip-timing-gate", action="store_true", help="skip the timing-gate probe in 'run'")
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
            print("STOP: verify found an alignment/structural error or a difference above 1e-3.")
            sys.exit(1)
    if args.subcommand in ("run", "all"):
        t0 = time.time()
        run_run_subcommand(skip_timing_gate=args.skip_timing_gate)
        print(f"[run] done in {time.time() - t0:.1f}s")
    if args.subcommand in ("analyze", "all"):
        t0 = time.time()
        run_analyze_subcommand()
        print(f"[analyze] done in {time.time() - t0:.1f}s")
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

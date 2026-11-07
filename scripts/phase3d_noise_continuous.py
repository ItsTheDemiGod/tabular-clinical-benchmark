"""Phase 3d, Part 3: supplementary robustness variant -- Gaussian noise on
CONTINUOUS columns only (excludes CKD's sg/al/su and Cleveland's ca). Loads the
SAVED fraction-1.0 pipelines; never fits, refits or tunes anything. Degradation is
computed against the CLEAN rows already in results/phase3c/robustness_runs.csv.
Writes ONLY inside results/phase3d/noise_continuous/.

Subcommands: diagnostic, run, analyze, figures, tables, all.
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

from analysis.bootstrap import bootstrap_diff_ci, bootstrap_diff_ci_averaged, bootstrap_mean_ci, bootstrap_mean_ci_averaged
from analysis.constants import DATASETS_ALL5, DATASETS_NO_CKD, FAMILY, FAMILY_LINESTYLE, MODEL_COLOR, MODEL_MARKER, MODELS, SEED
from analysis.robustness import NOISE_LEVELS, N_DRAWS, apply_noise, evaluate_predictions, make_rng
from analysis.robustness_supp import EXCLUDED_ORDINAL_COLUMNS, apply_noise_excluding, make_rng_noise_cont
from analysis.tables import escape_latex, format_float, write_csv_and_tex
from src.data import load_dataset
from src.utils import make_splits

MODELS_DIR = PROJECT_ROOT / "results" / "models"
PHASE3C_DIR = PROJECT_ROOT / "results" / "phase3c"
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase3d" / "noise_continuous"
RUNS_PARTS_DIR = OUTPUT_DIR / "runs_parts"
BOOTSTRAP_B = 5000
TIMING_GATE_MINUTES = 30

_DATASET_CACHE: dict[str, tuple] = {}


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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


def _exclude_cols_for(dataset: str) -> list[str]:
    return EXCLUDED_ORDINAL_COLUMNS.get(dataset, [])


# ---------------------------------------------------------------------------
# Subcommand: diagnostic (read-only, cheap)
# ---------------------------------------------------------------------------
def run_diagnostic() -> pd.DataFrame:
    rows = []
    for dataset in DATASETS_ALL5:
        X, y, meta, splits = _get_dataset(dataset)
        numeric_cols = meta["numeric_cols"]
        exclude_cols = _exclude_cols_for(dataset)
        # "Physically non-negative" columns: whole-dataset minimum >= 0.
        nonneg_cols = [c for c in numeric_cols if X[c].min(skipna=True) >= 0]

        for level_idx, s in enumerate(NOISE_LEVELS):
            frac_negative = {"noise": [], "noise_cont": []}
            for split in range(10):
                train_idx, test_idx = splits[split]
                X_test = X.iloc[test_idx]
                X_train = X.iloc[train_idx]
                for draw in range(N_DRAWS):
                    rng_orig = make_rng(dataset, split, "noise", level_idx, draw)
                    pert_orig = apply_noise(X_test, X_train, s, numeric_cols, rng_orig)
                    rng_cont = make_rng_noise_cont(dataset, split, level_idx, draw)
                    pert_cont = apply_noise_excluding(X_test, X_train, s, numeric_cols, exclude_cols, rng_cont)

                    for key, pert in (("noise", pert_orig), ("noise_cont", pert_cont)):
                        if not nonneg_cols:
                            continue
                        values = pert[nonneg_cols].to_numpy(dtype=np.float64)
                        valid = ~np.isnan(values)
                        n_valid = int(valid.sum())
                        n_negative = int((values[valid] < 0).sum()) if n_valid else 0
                        frac_negative[key].append(n_negative / n_valid if n_valid else np.nan)

            rows.append(
                {
                    "dataset": dataset,
                    "level": s,
                    "nonneg_columns": ",".join(nonneg_cols),
                    "frac_negative_noise": float(np.nanmean(frac_negative["noise"])) if frac_negative["noise"] else np.nan,
                    "frac_negative_noise_cont": float(np.nanmean(frac_negative["noise_cont"])) if frac_negative["noise_cont"] else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    output_dir = ensure_output_dir()
    df.to_csv(output_dir / "negative_value_diagnostic.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------
def _run_one_task(dataset: str, model: str, split: int) -> tuple[list[dict], dict]:
    X, y, meta, splits = _get_dataset(dataset)
    train_idx, test_idx = splits[split]
    X_test = X.iloc[test_idx]
    X_train = X.iloc[train_idx]
    y_true = y.iloc[test_idx].to_numpy()
    numeric_cols = meta["numeric_cols"]
    exclude_cols = _exclude_cols_for(dataset)
    n_test = len(test_idx)

    pipeline = joblib.load(_pipeline_path(dataset, model, split))
    rows: list[dict] = []
    probs: dict[str, np.ndarray] = {}

    for level_idx, s in enumerate(NOISE_LEVELS):
        for draw in range(N_DRAWS):
            rng = make_rng_noise_cont(dataset, split, level_idx, draw)
            X_pert = apply_noise_excluding(X_test, X_train, s, numeric_cols, exclude_cols, rng)
            result = evaluate_predictions(pipeline, X_pert, y_true)
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "split_idx": split,
                    "kind": "noise_cont",
                    "level": s,
                    "draw": draw,
                    "auroc": result["auroc"],
                    "brier": result["brier"],
                    "log_loss": result["log_loss"],
                    "n_unique_probs": result["n_unique_probs"],
                    "constant_prediction": result["constant_prediction"],
                    "status": result["status"],
                    "error": result["error"],
                }
            )
            key = f"noise_cont_{s}_d{draw}"
            probs[key] = result["prob"] if result["prob"] is not None else np.full(n_test, np.nan, dtype=np.float32)

    del pipeline
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    return rows, probs


def _task_complete(dataset: str, model: str, split: int) -> bool:
    part_path = RUNS_PARTS_DIR / f"{_stem(dataset, model, split)}.csv"
    if not part_path.exists():
        return False
    try:
        return len(pd.read_csv(part_path)) == 15
    except Exception:  # noqa: BLE001
        return False


def run_timing_gate() -> float:
    probe_dataset = DATASETS_ALL5[0]
    per_model_seconds: dict[str, float] = {}
    for model in MODELS:
        start = time.time()
        _run_one_task(probe_dataset, model, 0)
        per_model_seconds[model] = time.time() - start
    avg_seconds = float(np.mean(list(per_model_seconds.values())))
    total_estimate_minutes = avg_seconds * 350 / 60.0
    print(f"Timing gate probe (1 task = 15 predictions, dataset={probe_dataset}, split=0):")
    for model, seconds in per_model_seconds.items():
        print(f"  {model}: {seconds:.1f}s/task")
    print(f"  average: {avg_seconds:.1f}s/task -> extrapolated total: {total_estimate_minutes:.1f} min for 350 tasks")
    return total_estimate_minutes


def run_run_subcommand(skip_timing_gate: bool = False) -> None:
    ensure_output_dir()
    if not skip_timing_gate:
        estimate = run_timing_gate()
        if estimate > TIMING_GATE_MINUTES:
            print(f"STOP: estimated total runtime {estimate:.1f} min exceeds the {TIMING_GATE_MINUTES} min gate.")
            sys.exit(1)

    n_done, n_skipped = 0, 0
    for dataset in DATASETS_ALL5:
        for model in MODELS:
            for split in range(10):
                if _task_complete(dataset, model, split):
                    n_skipped += 1
                    continue
                rows, probs = _run_one_task(dataset, model, split)
                stem = _stem(dataset, model, split)
                pd.DataFrame(rows).to_csv(RUNS_PARTS_DIR / f"{stem}.csv", index=False)
                np.savez_compressed(OUTPUT_DIR / f"{stem}.npz", **{k: v.astype(np.float32) for k, v in probs.items()})
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  ... {n_done} tasks completed, {n_skipped} skipped (resume)")
    print(f"run: {n_done} tasks completed this run, {n_skipped} already complete (resumed)")

    parts = sorted(RUNS_PARTS_DIR.glob("*.csv"))
    merged = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    merged = merged.sort_values(["dataset", "model", "split_idx", "level", "draw"], na_position="first")
    merged.to_csv(OUTPUT_DIR / "robustness_runs_noise_cont.csv", index=False)
    print(f"robustness_runs_noise_cont.csv: {len(merged)} rows (expected 5250)")


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------
METRICS = ["auroc", "brier", "log_loss"]


def _clean_lookup() -> pd.DataFrame:
    phase3c_runs = pd.read_csv(PHASE3C_DIR / "robustness_runs.csv")
    clean = phase3c_runs[phase3c_runs.kind == "clean"]
    return clean.set_index(["dataset", "model", "split_idx"])


def run_analyze_subcommand() -> None:
    output_dir = ensure_output_dir()
    runs = pd.read_csv(output_dir / "robustness_runs_noise_cont.csv")
    ok = runs[runs.status == "ok"]
    per_split = (
        ok.groupby(["dataset", "model", "split_idx", "level"], as_index=False)[METRICS]
        .mean()
        .rename(columns={m: f"{m}_mean" for m in METRICS})
    )
    clean = _clean_lookup()

    def _clean_value(row, metric):
        key = (row["dataset"], row["model"], row["split_idx"])
        return clean.loc[key, metric] if key in clean.index else np.nan

    for metric in METRICS:
        per_split[f"{metric}_clean"] = per_split.apply(lambda r: _clean_value(r, metric), axis=1)
        per_split[f"{metric}_degradation"] = per_split[f"{metric}_mean"] - per_split[f"{metric}_clean"]

    rng = np.random.default_rng(SEED)
    per_dataset_rows = []
    for dataset in DATASETS_ALL5:
        for model in MODELS:
            for level in NOISE_LEVELS:
                sub = per_split[(per_split.dataset == dataset) & (per_split.model == model) & (per_split.level == level)].sort_values("split_idx")
                if len(sub) != 10:
                    continue
                for metric in METRICS:
                    p_point, p_lo, p_hi = bootstrap_mean_ci(sub[f"{metric}_mean"].to_numpy(), rng, BOOTSTRAP_B)
                    d_point, d_lo, d_hi = bootstrap_mean_ci(sub[f"{metric}_degradation"].to_numpy(), rng, BOOTSTRAP_B)
                    per_dataset_rows.append(
                        {
                            "dataset": dataset, "model": model, "kind": "noise_cont", "level": level, "metric": metric,
                            "perturbed_mean": p_point, "perturbed_ci_lo": p_lo, "perturbed_ci_hi": p_hi,
                            "degradation_mean": d_point, "degradation_ci_lo": d_lo, "degradation_ci_hi": d_hi,
                        }
                    )
    pd.DataFrame(per_dataset_rows).to_csv(output_dir / "noise_cont_per_dataset.csv", index=False)

    averaged_rows = []
    for model in MODELS:
        for level in NOISE_LEVELS:
            for metric in METRICS:
                per_dataset_perturbed, per_dataset_degradation = {}, {}
                for dataset in DATASETS_ALL5:
                    sub = per_split[(per_split.dataset == dataset) & (per_split.model == model) & (per_split.level == level)].sort_values("split_idx")
                    if len(sub) != 10:
                        continue
                    per_dataset_perturbed[dataset] = sub[f"{metric}_mean"].to_numpy()
                    per_dataset_degradation[dataset] = sub[f"{metric}_degradation"].to_numpy()
                for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                    if not all(d in per_dataset_perturbed for d in datasets):
                        continue
                    values = [per_dataset_perturbed[d] for d in datasets]
                    deg_values = [per_dataset_degradation[d] for d in datasets]
                    p_point, p_lo, p_hi = bootstrap_mean_ci_averaged(values, rng, BOOTSTRAP_B)
                    d_point, d_lo, d_hi = bootstrap_mean_ci_averaged(deg_values, rng, BOOTSTRAP_B)
                    averaged_rows.append(
                        {
                            "model": model, "kind": "noise_cont", "level": level, "metric": metric, "dataset_set": set_name,
                            "perturbed_mean": p_point, "perturbed_ci_lo": p_lo, "perturbed_ci_hi": p_hi,
                            "degradation_mean": d_point, "degradation_ci_lo": d_lo, "degradation_ci_hi": d_hi,
                        }
                    )
    pd.DataFrame(averaged_rows).to_csv(output_dir / "noise_cont_averaged.csv", index=False)

    # Paired difference to logreg at noise=0.5.
    diff_rows = []
    level = NOISE_LEVELS[-1]
    for model in MODELS:
        if model == "logreg":
            continue
        for metric in METRICS:
            model_by_dataset, logreg_by_dataset, per_dataset_point = {}, {}, {}
            for dataset in DATASETS_ALL5:
                model_sub = per_split[(per_split.dataset == dataset) & (per_split.model == model) & (per_split.level == level)].sort_values("split_idx")
                logreg_sub = per_split[(per_split.dataset == dataset) & (per_split.model == "logreg") & (per_split.level == level)].sort_values("split_idx")
                if len(model_sub) != 10 or len(logreg_sub) != 10:
                    continue
                model_deg = model_sub[f"{metric}_degradation"].to_numpy()
                logreg_deg = logreg_sub[f"{metric}_degradation"].to_numpy()
                model_by_dataset[dataset] = model_deg
                logreg_by_dataset[dataset] = logreg_deg
                point, lo, hi = bootstrap_diff_ci(model_deg, logreg_deg, rng, BOOTSTRAP_B)
                per_dataset_point[dataset] = point
                diff_rows.append(
                    {"model": model, "level": level, "metric": metric, "level_of_detail": "PER_DATASET",
                     "dataset_set": dataset, "diff_mean": point, "ci_lo": lo, "ci_hi": hi, "n_positive": np.nan, "n_total": np.nan}
                )
            for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                if not all(d in model_by_dataset for d in datasets):
                    continue
                model_subset = [model_by_dataset[d] for d in datasets]
                logreg_subset = [logreg_by_dataset[d] for d in datasets]
                point, lo, hi = bootstrap_diff_ci_averaged(model_subset, logreg_subset, rng, BOOTSTRAP_B)
                n_positive = sum(1 for d in datasets if per_dataset_point[d] > 0)
                diff_rows.append(
                    {"model": model, "level": level, "metric": metric, "level_of_detail": "AVERAGED",
                     "dataset_set": set_name, "diff_mean": point, "ci_lo": lo, "ci_hi": hi, "n_positive": n_positive, "n_total": len(datasets)}
                )
    pd.DataFrame(diff_rows).to_csv(output_dir / "noise_cont_diff_vs_logreg.csv", index=False)


# ---------------------------------------------------------------------------
# Subcommand: figures
# ---------------------------------------------------------------------------
def run_figures_subcommand() -> list[str]:
    output_dir = ensure_output_dir()
    cont_averaged = pd.read_csv(output_dir / "noise_cont_averaged.csv")
    orig_averaged = pd.read_csv(PHASE3C_DIR / "robustness_averaged.csv")
    checked = []

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True)
    for ax, set_name in zip(axes, ("ALL5", "NO_CKD")):
        for model in MODELS:
            color = MODEL_COLOR[model]
            cont_sub = cont_averaged[(cont_averaged.model == model) & (cont_averaged.metric == "log_loss") & (cont_averaged.dataset_set == set_name)].sort_values("level")
            orig_sub = orig_averaged[(orig_averaged.model == model) & (orig_averaged.kind == "noise") & (orig_averaged.metric == "log_loss") & (orig_averaged.dataset_set == set_name)].sort_values("level")
            ax.plot(cont_sub.level, cont_sub.perturbed_mean, color=color, marker=MODEL_MARKER[model], linestyle="-", markersize=4, linewidth=1.2, label=model)
            ax.plot(orig_sub.level, orig_sub.perturbed_mean, color=color, marker=MODEL_MARKER[model], linestyle=":", markersize=4, linewidth=1.0)
        ax.set_title(set_name, fontsize=9)
        ax.set_xlabel("noise level", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("Log loss", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Log loss vs. noise level: continuous-only (solid) vs. Phase 3c (dotted)", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    png_path = output_dir / "fig9_logloss_noise_continuous_vs_original.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig9_logloss_noise_continuous_vs_original.pdf", bbox_inches="tight")
    plt.close(fig)
    checked.append(f"{png_path.name}: 2 panels, legend, solid/dotted distinguishable, axes labelled")

    (output_dir / "figure_captions.md").write_text(
        "# Phase 3d (Part 3) figure captions (factual only, no interpretation)\n\n"
        "## fig9: log loss vs. noise level, continuous-only vs. Phase 3c noise\n"
        "Dataset-averaged log loss vs. noise level (0.1, 0.25, 0.5), 7 models, ALL5 and "
        "NO_CKD panels. Solid line = continuous-only noise (Phase 3d, CKD's sg/al/su and "
        "Cleveland's ca excluded from noise); dotted line = the original Phase 3c noise "
        f"(all numeric columns noised). {N_DRAWS} draws per (dataset, split, level); "
        "bootstrap seed 42, B=5000, resampling the 10 splits within each dataset.\n",
        encoding="utf-8",
    )
    return checked


# ---------------------------------------------------------------------------
# Subcommand: tables
# ---------------------------------------------------------------------------
def run_tables_subcommand() -> None:
    output_dir = ensure_output_dir()
    cont_averaged = pd.read_csv(output_dir / "noise_cont_averaged.csv")
    orig_averaged = pd.read_csv(PHASE3C_DIR / "robustness_averaged.csv")
    cont_diff = pd.read_csv(output_dir / "noise_cont_diff_vs_logreg.csv")
    orig_diff = pd.read_csv(PHASE3C_DIR / "robustness_diff_vs_logreg.csv")

    # Table 13: AUROC and log loss at noise_cont vs original noise, ALL5/NO_CKD.
    for metric in ("auroc", "log_loss"):
        for set_name in ("ALL5", "NO_CKD"):
            rows = {}
            for level in NOISE_LEVELS:
                row = {}
                for model in MODELS:
                    cont_val = cont_averaged[(cont_averaged.model == model) & (cont_averaged.metric == metric) & (cont_averaged.level == level) & (cont_averaged.dataset_set == set_name)]["perturbed_mean"]
                    orig_val = orig_averaged[(orig_averaged.model == model) & (orig_averaged.kind == "noise") & (orig_averaged.metric == metric) & (orig_averaged.level == level) & (orig_averaged.dataset_set == set_name)]["perturbed_mean"]
                    row[f"{model}_cont"] = float(cont_val.iloc[0]) if len(cont_val) else np.nan
                    row[f"{model}_orig"] = float(orig_val.iloc[0]) if len(orig_val) else np.nan
                rows[level] = row
            table = pd.DataFrame(rows).T
            table.index.name = "level"
            write_csv_and_tex(f"table13_{metric}_noise_{set_name}", table.map(lambda v: format_float(v, 3)), output_dir, index_label="level")

    # Table 14: degradation diff vs logreg at noise=0.5, noise_cont vs original noise.
    level = NOISE_LEVELS[-1]
    rows14 = []
    for model in MODELS:
        if model == "logreg":
            continue
        for metric in ("auroc", "log_loss"):
            for set_name in ("ALL5", "NO_CKD"):
                cont_row = cont_diff[(cont_diff.model == model) & (cont_diff.metric == metric) & (cont_diff.level_of_detail == "AVERAGED") & (cont_diff.dataset_set == set_name)]
                orig_row = orig_diff[(orig_diff.model == model) & (orig_diff.metric == metric) & (orig_diff.kind == "noise") & (orig_diff.level == level) & (orig_diff.level_of_detail == "AVERAGED") & (orig_diff.dataset_set == set_name)]
                rows14.append(
                    {
                        "model": model, "metric": metric, "dataset_set": set_name,
                        "cont_diff_mean": float(cont_row["diff_mean"].iloc[0]) if len(cont_row) else np.nan,
                        "cont_ci_lo": float(cont_row["ci_lo"].iloc[0]) if len(cont_row) else np.nan,
                        "cont_ci_hi": float(cont_row["ci_hi"].iloc[0]) if len(cont_row) else np.nan,
                        "orig_diff_mean": float(orig_row["diff_mean"].iloc[0]) if len(orig_row) else np.nan,
                        "orig_ci_lo": float(orig_row["ci_lo"].iloc[0]) if len(orig_row) else np.nan,
                        "orig_ci_hi": float(orig_row["ci_hi"].iloc[0]) if len(orig_row) else np.nan,
                    }
                )
    table14 = pd.DataFrame(rows14)
    table14.to_csv(output_dir / "table14_diff_vs_logreg_noise_cont_vs_original.csv", index=False)
    display14 = table14.copy()
    for col in display14.columns:
        if col not in ("model", "metric", "dataset_set"):
            display14[col] = display14[col].map(lambda v: format_float(v, 4))
    with open(output_dir / "table14_diff_vs_logreg_noise_cont_vs_original.tex", "w", encoding="utf-8") as fh:
        fh.write(r"\begin{tabular}{lll" + "r" * 6 + "}\n" + r"\toprule" + "\n")
        header = ["model", "metric", "dataset\\_set", "cont\\_diff", "cont\\_lo", "cont\\_hi", "orig\\_diff", "orig\\_lo", "orig\\_hi"]
        fh.write(" & ".join(header) + r" \\" + "\n" + r"\midrule" + "\n")
        for _, r in display14.iterrows():
            cells = [escape_latex(r.model), escape_latex(r.metric), escape_latex(r.dataset_set)] + [str(r[c]) for c in display14.columns[3:]]
            fh.write(" & ".join(cells) + r" \\" + "\n")
        fh.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")

    (output_dir / "table_captions.md").write_text(
        "# Phase 3d (Part 3) table captions\n\n"
        "Table 13: dataset-averaged AUROC and log loss at each noise level, continuous-only "
        "(_cont suffix) next to the original Phase 3c noise (_orig suffix), ALL5 and NO_CKD.\n"
        "Table 14: degradation difference to logreg at noise=0.5, continuous-only vs. "
        "original noise, with 95% bootstrap CIs.\n",
        encoding="utf-8",
    )
    notes = [
        "# Phase 3d (Part 3) notes: continuous-only noise variant\n",
        "- Identical to the Phase 3c NOISE perturbation, except CKD's sg/al/su and",
        "  Cleveland's ca (ordinal numeric scales) receive NO noise at all.",
        "- CKD's bp column also has <=10 distinct values but is intentionally kept IN",
        "  (noised as ordinary continuous blood pressure), per an explicit decision.",
        f"- {N_DRAWS} draws per (dataset, split, level); seeded by (dataset, split,",
        "  'noise_cont', level, draw), never by model.",
        "- Degradation is computed against the CLEAN rows in",
        "  results/phase3c/robustness_runs.csv (not recomputed).",
        "- The bootstrap resamples the 10 splits within each dataset and ignores",
        "  between-dataset variability. Descriptive only; no hypothesis tests.",
    ]
    (output_dir / "robustness_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3d continuous-only noise variant")
    parser.add_argument("subcommand", choices=["diagnostic", "run", "analyze", "figures", "tables", "all"])
    parser.add_argument("--skip-timing-gate", action="store_true")
    args = parser.parse_args()

    print(f"sys.executable = {sys.executable}")
    ensure_output_dir()
    start = time.time()

    if args.subcommand in ("diagnostic", "all"):
        t0 = time.time()
        df = run_diagnostic()
        print(f"[diagnostic] done in {time.time() - t0:.1f}s")
        print(df.to_string(index=False))
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

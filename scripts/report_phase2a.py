"""Print the Phase 2a verification checks, result tables and suspicious-pattern flags.

This only reads files and prints. Run from the project root, after the full run:
    python scripts/report_phase2a.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.runner import RESULT_SUBDIRS, is_constant_prediction

RESULTS_DIR = PROJECT_ROOT / "results"
MODELS: List[str] = ["logreg", "rf", "xgb", "lgbm"]
DATASETS: List[str] = ["pima_diabetes", "heart_cleveland", "ckd", "ilpd", "breast_wisconsin"]
FRACTIONS: List[float] = [0.1, 0.25, 0.5, 1.0]

# Flag thresholds.
HIGH_AUROC = 0.99
DECREASE_TOLERANCE = 0.005


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def print_verification(runs: pd.DataFrame) -> None:
    section("Verification")
    print(f"Rows in phase2a_all_runs.csv: {len(runs)} (expected 800)")
    print(f"Duplicate task/fraction rows: {runs.duplicated(['dataset', 'model', 'split_idx', 'fraction']).sum()}")
    print(f"Status counts: {runs['status'].value_counts().to_dict()}")

    failed = runs[runs["status"] != "ok"]
    print(f"Failed rows: {len(failed)}")
    for _, row in failed.iterrows():
        print(f"  {row['dataset']} {row['model']} split={row['split_idx']} fraction={row['fraction']}: {row['error']}")

    constant = runs[is_constant_prediction(runs)]
    print(f"Rows with constant_prediction True: {len(constant)}")
    if len(constant) > 0:
        print(constant[["dataset", "model", "split_idx", "fraction", "n_train"]].to_string(index=False))

    for subdir in RESULT_SUBDIRS:
        print(f"Files in {subdir}: {sum(p.is_file() for p in (RESULTS_DIR / subdir).iterdir())} (expected 200)")
    size = sum(p.stat().st_size for p in (RESULTS_DIR / "models").iterdir())
    print(f"Disk size of results/models: {size / 1024 ** 2:.1f} MB")


def print_table_1(ok: pd.DataFrame) -> None:
    section("Table 1: mean +/- std AUROC at fraction 1.0 over the 10 splits")
    full = ok[ok["fraction"] == 1.0]
    summary = full.groupby(["dataset", "model"])["auroc"].agg(["mean", "std"])
    table = summary.apply(lambda r: f"{r['mean']:.4f} +/- {r['std']:.4f}", axis=1).unstack("model")
    print(table.loc[DATASETS, MODELS].to_string())


def print_table_2(ok: pd.DataFrame) -> None:
    section("Table 2: mean AUROC per model and fraction")
    for dataset in DATASETS:
        table = (
            ok[ok["dataset"] == dataset]
            .pivot_table(index="model", columns="fraction", values="auroc", aggfunc="mean")
            .loc[MODELS, FRACTIONS]
        )
        print(f"\n{dataset}")
        print(table.round(4).to_string())


def print_table_3(ok: pd.DataFrame) -> None:
    section("Table 3: fraction 1.0, averaged over datasets (fit times from 4 parallel workers, rough)")
    full = ok[ok["fraction"] == 1.0]
    for label, rows in (("all rows", full), ("excluding constant_prediction", full[~is_constant_prediction(full)])):
        # Averages per dataset first, so each dataset counts equally.
        per_dataset = rows.groupby(["model", "dataset"])[["ece", "brier", "fit_time_s"]].mean()
        table = per_dataset.groupby("model").mean().loc[MODELS]
        print(f"\n{label} (n={len(rows)})")
        print(table.round(4).to_string())


def print_table_4(ok: pd.DataFrame) -> None:
    section("Table 4: constant_prediction rows by model and fraction")
    flagged = ok.assign(constant=is_constant_prediction(ok))
    table = flagged.pivot_table(index="model", columns="fraction", values="constant", aggfunc="sum")
    print(table.loc[MODELS, FRACTIONS].astype(int).to_string())
    if flagged["constant"].any():
        print("\nBy dataset:")
        print(flagged[flagged["constant"]].groupby(["dataset", "model", "fraction"]).size().to_string())


def load_tuning() -> pd.DataFrame:
    records = []
    for path in sorted((RESULTS_DIR / "tuning").glob("*.json")):
        with open(path, encoding="utf-8") as f:
            record = json.load(f)
        info = record.get("info") or {}
        records.append(
            {
                "dataset": record["dataset"],
                "model": record["model"],
                "split_idx": record["split_idx"],
                "n_failed_trials": info.get("n_failed_trials"),
                "best_value": info.get("best_value"),
                "failed_trial_errors": info.get("failed_trial_errors", []),
            }
        )
    return pd.DataFrame(records)


def print_tuning_summary(tuning: pd.DataFrame) -> None:
    section("Tuning summary")
    print("Mean n_failed_trials per model:")
    print(tuning.groupby("model")["n_failed_trials"].mean().loc[MODELS].to_string())
    print("\nInner-CV AUROC (best_value) per dataset:")
    print(tuning.groupby("dataset")["best_value"].agg(["min", "max"]).loc[DATASETS].round(4).to_string())


def print_flags(ok: pd.DataFrame, tuning: pd.DataFrame) -> None:
    """List patterns worth a person's attention. Nothing is changed because of them."""
    section("Suspicious patterns (flagged only)")

    means = ok.groupby(["dataset", "model", "fraction"])["auroc"].mean()
    high = means[(means > HIGH_AUROC) & (means.index.get_level_values("dataset") != "breast_wisconsin")]
    print(f"\nMean AUROC above {HIGH_AUROC} outside breast_wisconsin: {len(high)}")
    if len(high) > 0:
        print(high.round(4).to_string())

    high_single = ok[(ok["auroc"] > HIGH_AUROC) & (ok["dataset"] != "breast_wisconsin")]
    print(f"\nSingle rows above {HIGH_AUROC} outside breast_wisconsin, by dataset: "
          f"{high_single['dataset'].value_counts().to_dict()}")
    perfect = ok[ok["auroc"] == 1.0]
    print(f"Rows with AUROC exactly 1.0, by dataset: {perfect['dataset'].value_counts().to_dict()}")

    low = ok[ok["auroc"] < 0.5]
    print(f"\nRows with AUROC below 0.5: {len(low)}")
    if len(low) > 0:
        print(low[["dataset", "model", "split_idx", "fraction", "auroc"]].to_string(index=False))

    exactly_half = ok[ok["auroc"] == 0.5]
    print(f"Rows with AUROC exactly 0.5: {len(exactly_half)}")

    identical = [
        key
        for key, group in ok.groupby(["dataset", "model", "split_idx"])
        if len(group) == len(FRACTIONS) and group["auroc"].nunique() == 1
    ]
    print(f"\nTasks whose AUROC is identical across all fractions: {len(identical)}")
    by_group = pd.Series([key[:2] for key in identical]).value_counts()
    if len(by_group) > 0:
        print(by_group.to_string())

    print(f"\nMean learning-curve drops larger than {DECREASE_TOLERANCE} AUROC as training size grows:")
    n_drops = 0
    for (dataset, model), group in means.groupby(level=["dataset", "model"]):
        values = group.droplevel(["dataset", "model"]).loc[FRACTIONS]
        for smaller, larger in zip(FRACTIONS, FRACTIONS[1:]):
            drop = values[smaller] - values[larger]
            if drop > DECREASE_TOLERANCE:
                n_drops += 1
                print(f"  {dataset} {model}: {smaller} -> {larger}: {values[smaller]:.4f} -> {values[larger]:.4f}")
    print(f"  total: {n_drops}")

    failed_trials = tuning[tuning["n_failed_trials"].fillna(0) > 0]
    print(f"\nTasks with failed tuning trials: {len(failed_trials)}")
    for _, row in failed_trials.iterrows():
        print(f"  {row['dataset']} {row['model']} split={row['split_idx']}: "
              f"{row['n_failed_trials']} failed, e.g. {row['failed_trial_errors'][:1]}")

    print("\nMean inner-CV best_value minus mean outer test AUROC at fraction 1.0 (optimism):")
    full = ok[ok["fraction"] == 1.0].groupby(["dataset", "model"])["auroc"].mean()
    inner = tuning.groupby(["dataset", "model"])["best_value"].mean()
    gap = (inner - full).unstack("model").loc[DATASETS, MODELS]
    print(gap.round(4).to_string())


def main() -> None:
    runs = pd.read_csv(RESULTS_DIR / "phase2a_all_runs.csv")
    ok = runs[runs["status"] == "ok"]
    tuning = load_tuning()
    pd.set_option("display.width", 200)

    print_verification(runs)
    print_table_1(ok)
    print_table_2(ok)
    print_table_3(ok)
    print_table_4(ok)
    print_tuning_summary(tuning)
    print_flags(ok, tuning)


if __name__ == "__main__":
    main()

"""Loading results/all_runs.csv and building the BLOCK / DATASET analysis-unit matrices.

This module only ever reads from results/all_runs.csv and results/raw/*.csv
(read-only, for the Phase 3a precondition check); it never writes anything.
"""
from __future__ import annotations

import glob

import pandas as pd

from analysis.constants import ALL_RUNS_PATH, MODELS, RAW_DIR


def load_all_runs() -> pd.DataFrame:
    """Load results/all_runs.csv unchanged."""
    return pd.read_csv(ALL_RUNS_PATH)


def load_raw_concat() -> pd.DataFrame:
    """Concatenate every results/raw/*.csv file, for the precondition cross-check."""
    files = sorted(glob.glob(str(RAW_DIR / "*.csv")))
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def block_matrix(df: pd.DataFrame, metric: str, fraction: float, datasets: list[str]) -> pd.DataFrame:
    """Return the BLOCK-unit matrix: rows = (dataset, split_idx), columns = models (fixed order).

    One row per (dataset, split_idx) block at the given fraction, restricted to `datasets`.
    """
    sub = df[(df["fraction"] == fraction) & (df["dataset"].isin(datasets))]
    pivot = sub.pivot_table(index=["dataset", "split_idx"], columns="model", values=metric)
    pivot = pivot.reindex(columns=MODELS)
    # Keep dataset blocks ordered by the fixed dataset list, then by split_idx.
    pivot = pivot.sort_index(
        key=lambda idx: idx if idx.name == "split_idx" else idx.map({d: i for i, d in enumerate(datasets)})
    )
    return pivot


def dataset_matrix(blocks: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    """Return the DATASET-unit matrix: rows = dataset-level mean over its 10 splits."""
    grouped = blocks.groupby(level="dataset").mean()
    return grouped.reindex(index=datasets, columns=MODELS)


def get_split_values(df: pd.DataFrame, dataset: str, fraction: float, model: str, metric: str) -> pd.Series:
    """The 10 per-split values for one (dataset, fraction, model), ordered by split_idx."""
    sub = df[(df["dataset"] == dataset) & (df["fraction"] == fraction) & (df["model"] == model)]
    return sub.sort_values("split_idx")[metric].reset_index(drop=True)

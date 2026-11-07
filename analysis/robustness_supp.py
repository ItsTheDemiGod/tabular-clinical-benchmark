"""Phase 3d supplementary robustness helper: Gaussian noise identical to Phase 3c's
NOISE perturbation, except that a given set of ordinal-scale numeric columns receive
NO noise at all (left untouched, like a categorical column).

This is a NEW module -- analysis/robustness.py is only imported from, never modified.
Reuses analysis.robustness.DATASET_INDEX for dataset seeding so the two modules stay
consistent, but defines its own kind index (3) since "noise_cont" does not exist in
analysis.robustness.KIND_INDEX.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.robustness import DATASET_INDEX

__all__ = ["EXCLUDED_ORDINAL_COLUMNS", "NOISE_CONT_KIND_INDEX", "make_rng_noise_cont", "apply_noise_excluding"]

# Fixed before any run, per the Phase 3d ordinal-column check (CLAUDE.md section 7):
# only these columns are excluded. CKD's `bp` also has <=10 distinct values but is
# kept IN (as ordinary noisy continuous blood pressure), per Demian's explicit choice.
EXCLUDED_ORDINAL_COLUMNS: dict[str, list[str]] = {
    "ckd": ["sg", "al", "su"],
    "heart_cleveland": ["ca"],
}

NOISE_CONT_KIND_INDEX: int = 3  # distinct from clean=0, missing=1, noise=2 in analysis.robustness


def make_rng_noise_cont(dataset: str, split_idx: int, level_idx: int, draw: int) -> np.random.Generator:
    """Reproducible Generator for the "noise_cont" kind, keyed only by
    (dataset, split, level, draw) -- never by model.
    """
    seed_sequence = np.random.SeedSequence(
        [42, DATASET_INDEX[dataset], split_idx, NOISE_CONT_KIND_INDEX, level_idx, draw]
    )
    return np.random.default_rng(seed_sequence)


def apply_noise_excluding(
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    s: float,
    numeric_cols: list[str],
    exclude_cols: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Identical to analysis.robustness.apply_noise, except columns in `exclude_cols`
    are skipped entirely (no noise, and no rng state consumed for them), so calling
    this with exclude_cols=[] reproduces analysis.robustness.apply_noise exactly for
    the same rng state and column order.
    """
    perturbed = X.copy()
    n = len(X)
    for col in numeric_cols:
        if col in exclude_cols:
            continue
        sigma = float(X_train[col].std(ddof=0))
        if not np.isfinite(sigma):
            sigma = 0.0
        noise = rng.normal(0.0, s * sigma, size=n)
        non_missing = X[col].notna().to_numpy()
        values = perturbed[col].to_numpy(dtype=np.float64, copy=True)
        values[non_missing] = values[non_missing] + noise[non_missing]
        perturbed[col] = values
    return perturbed

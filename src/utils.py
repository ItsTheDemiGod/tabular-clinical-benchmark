"""Reproducibility and resampling utilities shared across the benchmark.

Keeping these in one place ensures every experiment in later phases uses the
exact same splitting and subsampling logic, which is itself a precondition
for a fair, leakage-safe comparison between models.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


def set_seed(seed: int) -> None:
    """Seed the `random` and `numpy` global RNGs for reproducibility.

    Args:
        seed: Seed value applied to both RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 2,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate repeated stratified K-fold train/test index splits.

    Using `RepeatedStratifiedKFold` (rather than a single K-fold) gives a
    more stable estimate of model performance on small clinical datasets,
    where a single split can be dominated by sampling noise.

    Args:
        X: Feature matrix (only `len(X)` and index alignment with `y` matter).
        y: Binary target aligned with `X`.
        n_splits: Number of folds per repeat.
        n_repeats: Number of times the K-fold procedure is repeated with
            different randomization.
        seed: Random seed controlling the fold assignment.

    Returns:
        A list of `(train_idx, test_idx)` positional index arrays, one per
        fold across all repeats (length `n_splits * n_repeats`).
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=seed
    )
    return list(rskf.split(X, y))


def subsample_train(
    train_idx: np.ndarray,
    y: pd.Series,
    fraction: float,
    seed: int,
) -> np.ndarray:
    """Stratified subsample of a training fold's indices.

    Used to study how model performance scales with training-set size
    (fractions 0.1, 0.25, 0.5, 1.0) while never touching the test fold.

    Args:
        train_idx: Positional indices of the training fold (as returned by
            `make_splits`), relative to the original `X`/`y` arrays.
        y: Full target series (indexed positionally, same length as the
            original `X` passed to `make_splits`).
        fraction: Fraction of `train_idx` to keep (0 < fraction <= 1).
        seed: Random seed for the subsampling draw.

    Returns:
        A subsampled array of indices (a subset of `train_idx`), with at
        least 2 samples of each class guaranteed whenever `train_idx`
        itself contains at least 2 samples of each class.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    rng = np.random.RandomState(seed)
    y_train = np.asarray(y)[train_idx]

    classes = np.unique(y_train)
    min_per_class = 2

    selected: List[np.ndarray] = []
    for cls in classes:
        cls_idx = train_idx[y_train == cls]
        n_keep = max(min_per_class, int(round(len(cls_idx) * fraction)))
        n_keep = min(n_keep, len(cls_idx))
        chosen = rng.choice(cls_idx, size=n_keep, replace=False)
        selected.append(chosen)

    result = np.concatenate(selected)
    rng.shuffle(result)
    return result

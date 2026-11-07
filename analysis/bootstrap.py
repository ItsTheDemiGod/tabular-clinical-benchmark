"""Percentile bootstrap confidence intervals (seed 42, B = 5000 by default).

Per-dataset CIs resample the 10 splits of that dataset with replacement.
Dataset-averaged CIs use NESTED resampling: each bootstrap replicate resamples the
splits independently within every dataset and then averages the per-dataset means;
this ignores between-dataset variability (reported as a caveat, not fixed here).
Paired differences to logreg always reuse the SAME resampled split indices for the
model and for logreg, so the pairing by split_idx is preserved under resampling.
"""
from __future__ import annotations

import numpy as np


def _resample_indices(rng: np.random.Generator, n: int, b: int) -> np.ndarray:
    return rng.integers(0, n, size=(b, n))


def bootstrap_mean_ci(
    values: np.ndarray, rng: np.random.Generator, b: int
) -> tuple[float, float, float]:
    """CI for the mean of one dataset's 10 split values. Returns (point, lo95, hi95)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    idx = _resample_indices(rng, n, b)
    boot_means = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_mean_ci_averaged(
    values_by_dataset: list[np.ndarray], rng: np.random.Generator, b: int
) -> tuple[float, float, float]:
    """Nested-resampling CI for the dataset-averaged mean across several datasets."""
    boot_per_dataset = []
    for values in values_by_dataset:
        values = np.asarray(values, dtype=float)
        idx = _resample_indices(rng, len(values), b)
        boot_per_dataset.append(values[idx].mean(axis=1))
    boot_averaged = np.mean(np.vstack(boot_per_dataset), axis=0)
    point = float(np.mean([np.mean(v) for v in values_by_dataset]))
    lo, hi = np.percentile(boot_averaged, [2.5, 97.5])
    return point, float(lo), float(hi)


def bootstrap_diff_ci(
    model_values: np.ndarray, logreg_values: np.ndarray, rng: np.random.Generator, b: int
) -> tuple[float, float, float]:
    """CI for the paired difference (model - logreg) within one dataset."""
    model_values = np.asarray(model_values, dtype=float)
    logreg_values = np.asarray(logreg_values, dtype=float)
    n = len(model_values)
    idx = _resample_indices(rng, n, b)
    model_boot = model_values[idx].mean(axis=1)
    logreg_boot = logreg_values[idx].mean(axis=1)
    diff_boot = model_boot - logreg_boot
    point = float(model_values.mean() - logreg_values.mean())
    lo, hi = np.percentile(diff_boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def bootstrap_diff_ci_averaged(
    model_by_dataset: list[np.ndarray],
    logreg_by_dataset: list[np.ndarray],
    rng: np.random.Generator,
    b: int,
) -> tuple[float, float, float]:
    """Nested-resampling CI for the dataset-averaged paired difference (model - logreg)."""
    diff_boot_per_dataset = []
    for model_values, logreg_values in zip(model_by_dataset, logreg_by_dataset):
        model_values = np.asarray(model_values, dtype=float)
        logreg_values = np.asarray(logreg_values, dtype=float)
        idx = _resample_indices(rng, len(model_values), b)
        model_boot = model_values[idx].mean(axis=1)
        logreg_boot = logreg_values[idx].mean(axis=1)
        diff_boot_per_dataset.append(model_boot - logreg_boot)
    diff_boot_averaged = np.mean(np.vstack(diff_boot_per_dataset), axis=0)
    point = float(
        np.mean(
            [
                np.mean(m) - np.mean(l)
                for m, l in zip(model_by_dataset, logreg_by_dataset)
            ]
        )
    )
    lo, hi = np.percentile(diff_boot_averaged, [2.5, 97.5])
    return point, float(lo), float(hi)

"""Phase 3b calibration helpers: equal-mass ECE, the Platt calibration regression
(fit by unregularised MLE via scipy.optimize, not sklearn, per the Phase 3b brief),
separability detection, the Wilson confidence interval, pooled-metric bootstrapping,
and cross-fitted post-hoc recalibration (Platt / isotonic).

Nothing here touches a file; everything operates on arrays already loaded by
scripts/phase3b_calibration.py. `src.metrics.expected_calibration_error` is reused
directly for the equal-width ECE so Phase 3b's "ece_ew" is byte-identical to the
benchmark's own definition.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from src.metrics import LOG_LOSS_EPS, expected_calibration_error

__all__ = [
    "ece_equal_mass",
    "logit",
    "is_separable",
    "fit_platt",
    "wilson_interval",
    "bootstrap_pooled_metric",
    "bootstrap_diff_pooled_metric",
    "cross_fit_calibrate",
    "pooled_metrics",
]


def logit(p: np.ndarray) -> np.ndarray:
    """Logit of probabilities, clipped to the same epsilon as src.metrics' log loss."""
    clipped = np.clip(np.asarray(p, dtype=np.float64), LOG_LOSS_EPS, 1 - LOG_LOSS_EPS)
    return np.log(clipped / (1 - clipped))


def ece_equal_mass(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Equal-mass (equal-count) ECE: sort by probability, split into `n_bins` near-equal
    groups with numpy.array_split, ECE = sum over groups of (n_g/n) * |mean(y) - mean(p)|.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    order = np.argsort(y_prob, kind="mergesort")
    y_sorted = y_true[order]
    p_sorted = y_prob[order]
    n = len(y_true)
    ece = 0.0
    for group_y, group_p in zip(np.array_split(y_sorted, n_bins), np.array_split(p_sorted, n_bins)):
        if len(group_y) == 0:
            continue
        ece += (len(group_y) / n) * abs(group_y.mean() - group_p.mean())
    return float(ece)


def is_separable(y: np.ndarray, z: np.ndarray) -> bool:
    """True if the two classes are perfectly separable in z (either direction)."""
    y = np.asarray(y)
    z = np.asarray(z, dtype=np.float64)
    z0, z1 = z[y == 0], z[y == 1]
    if len(z0) == 0 or len(z1) == 0:
        return True
    return bool(z0.max() < z1.min() or z1.max() < z0.min())


def fit_platt(y: np.ndarray, z: np.ndarray) -> tuple[float, float, bool]:
    """Unregularised 2-parameter logistic regression y ~ sigmoid(intercept + slope*z),
    fit by maximum likelihood with scipy.optimize (analytic gradient, BFGS).

    Returns (intercept, slope, separable). If y has fewer than 2 classes or the
    classes are exactly separable in z, returns (nan, nan, True) without optimizing.
    """
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if len(np.unique(y)) < 2 or is_separable(y, z):
        return float("nan"), float("nan"), True

    def neg_log_likelihood(params: np.ndarray) -> float:
        a, b = params
        logits = a + b * z
        # log(sigmoid(x)) = -log(1+exp(-x)) = -logaddexp(0, -x); numerically stable.
        log_p1 = -np.logaddexp(0.0, -logits)
        log_p0 = -np.logaddexp(0.0, logits)
        return float(-np.sum(y * log_p1 + (1 - y) * log_p0))

    def gradient(params: np.ndarray) -> np.ndarray:
        a, b = params
        logits = a + b * z
        p = 1.0 / (1.0 + np.exp(-logits))
        residual = p - y
        return np.array([residual.sum(), float(np.sum(residual * z))])

    result = minimize(neg_log_likelihood, x0=np.array([0.0, 1.0]), jac=gradient, method="BFGS")
    intercept, slope = result.x
    return float(intercept), float(slope), False


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (default 95%, z=1.96)."""
    if n == 0:
        return float("nan"), float("nan")
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return float(center - margin), float(center + margin)


def bootstrap_pooled_metric(
    y_splits: list[np.ndarray],
    p_splits: list[np.ndarray],
    rng: np.random.Generator,
    b: int,
    metric_fn,
) -> np.ndarray:
    """Resample the n splits (e.g. the 10 CV splits of a dataset) with replacement B
    times; each replicate pools the resampled splits (with repetition) and applies
    `metric_fn(y_pooled, p_pooled) -> float` (nan allowed, e.g. for separable data).
    Returns a length-B array (may contain nan).
    """
    n = len(y_splits)
    idx = rng.integers(0, n, size=(b, n))
    results = np.full(b, np.nan)
    for i in range(b):
        chosen = idx[i]
        y_pooled = np.concatenate([y_splits[j] for j in chosen])
        p_pooled = np.concatenate([p_splits[j] for j in chosen])
        results[i] = metric_fn(y_pooled, p_pooled)
    return results


def bootstrap_diff_pooled_metric(
    y_model_splits: list[np.ndarray],
    p_model_splits: list[np.ndarray],
    y_logreg_splits: list[np.ndarray],
    p_logreg_splits: list[np.ndarray],
    rng: np.random.Generator,
    b: int,
    metric_fn,
) -> np.ndarray:
    """Like `bootstrap_pooled_metric`, but for the paired difference (model - logreg),
    using the SAME resampled split indices for both, so the pairing by split is kept.
    """
    n = len(y_model_splits)
    idx = rng.integers(0, n, size=(b, n))
    diffs = np.full(b, np.nan)
    for i in range(b):
        chosen = idx[i]
        y_m = np.concatenate([y_model_splits[j] for j in chosen])
        p_m = np.concatenate([p_model_splits[j] for j in chosen])
        y_l = np.concatenate([y_logreg_splits[j] for j in chosen])
        p_l = np.concatenate([p_logreg_splits[j] for j in chosen])
        diffs[i] = metric_fn(y_m, p_m) - metric_fn(y_l, p_l)
    return diffs


def _calibrate_fold(
    y_fit: np.ndarray, p_fit: np.ndarray, p_apply: np.ndarray, method: str
) -> tuple[np.ndarray, bool]:
    """Fit one calibrator on (y_fit, p_fit) and apply it to p_apply.

    Returns (calibrated_p_apply, fallback). `fallback` is True (and the identity map
    is used, clipped to [1e-7, 1-1e-7]) when `y_fit` has a single class, or, for
    "platt", when `y_fit` is exactly separable in logit(p_fit).
    """
    if method == "platt":
        z_fit = logit(p_fit)
        if len(np.unique(y_fit)) < 2 or is_separable(y_fit, z_fit):
            return np.clip(p_apply, LOG_LOSS_EPS, 1 - LOG_LOSS_EPS), True
        intercept, slope, _ = fit_platt(y_fit, z_fit)
        z_apply = logit(p_apply)
        logits = intercept + slope * z_apply
        calibrated = 1.0 / (1.0 + np.exp(-logits))
        return np.clip(calibrated, LOG_LOSS_EPS, 1 - LOG_LOSS_EPS), False
    if method == "isotonic":
        if len(np.unique(y_fit)) < 2:
            return np.clip(p_apply, LOG_LOSS_EPS, 1 - LOG_LOSS_EPS), True
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_fit, y_fit)
        return np.clip(iso.predict(p_apply), LOG_LOSS_EPS, 1 - LOG_LOSS_EPS), False
    raise ValueError(f"unknown method {method!r}")


def cross_fit_calibrate(
    y: np.ndarray, p: np.ndarray, seed: int, method: str
) -> tuple[np.ndarray, int, list[tuple[np.ndarray, np.ndarray]]]:
    """Cross-fitted post-hoc recalibration: inner stratified 5-fold; each fold's
    calibrator is fit on the other 4 folds and applied only to the held-out fold, so
    no row calibrates itself.

    method: "platt" or "isotonic".
    Returns (calibrated_probs, n_fallback, fold_indices), where fold_indices is the
    list of (fit_idx, apply_idx) index-array pairs used, for direct inspection in
    tests (no monkeypatching needed to verify row disjointness/coverage).
    """
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    out = np.full(len(p), np.nan, dtype=np.float64)
    n_fallback = 0
    fold_indices: list[tuple[np.ndarray, np.ndarray]] = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fit_idx, apply_idx in skf.split(p, y):
        fold_indices.append((fit_idx, apply_idx))
        calibrated, fallback = _calibrate_fold(y[fit_idx], p[fit_idx], p[apply_idx], method)
        out[apply_idx] = calibrated
        n_fallback += int(fallback)

    return out, n_fallback, fold_indices


def pooled_metrics(y_pooled: np.ndarray, p_pooled: np.ndarray) -> dict:
    """ece_ew, ece_em, intercept, slope, separable and n_pooled on one pooled sample."""
    ece_ew = expected_calibration_error(y_pooled, p_pooled)
    ece_em = ece_equal_mass(y_pooled, p_pooled)
    z_pooled = logit(p_pooled)
    intercept, slope, separable = fit_platt(y_pooled, z_pooled)
    return {
        "ece_ew": ece_ew,
        "ece_em": ece_em,
        "intercept": intercept,
        "slope": slope,
        "separable": separable,
        "n_pooled": len(y_pooled),
    }

"""Evaluation metrics shared by every model in the benchmark."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

N_ECE_BINS: int = 10
LOG_LOSS_EPS: float = 1e-7


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Binary expected calibration error on the positive-class probability.

    Uses 10 equal-width bins over [0, 1]:

        ECE = sum over bins of (n_bin / n) * |mean(y_true in bin) - mean(y_prob in bin)|

    Empty bins are skipped. Bins include their lower edge and exclude their upper
    edge, except the last bin, which also includes 1.0.

    Args:
        y_true: Binary labels (0/1).
        y_prob: Predicted probabilities of the positive class, within [0, 1].

    Returns:
        The ECE, between 0 and 1.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    # Multiplying instead of comparing to np.linspace edges avoids rounding surprises
    # such as linspace's 0.30000000000000004, which would put 0.3 in the wrong bin.
    bin_ids = np.minimum(np.floor(y_prob * N_ECE_BINS).astype(int), N_ECE_BINS - 1)

    ece = 0.0
    for bin_id in range(N_ECE_BINS):
        in_bin = bin_ids == bin_id
        n_bin = int(in_bin.sum())
        if n_bin == 0:
            continue
        gap = abs(float(y_true[in_bin].mean()) - float(y_prob[in_bin].mean()))
        ece += (n_bin / y_true.size) * gap
    return float(ece)


def prediction_diversity(y_prob: np.ndarray) -> Tuple[int, bool]:
    """Report how many distinct predictions a model made.

    A constant prediction cannot rank patients at all (AUROC 0.5), yet it can still look
    well calibrated, so calibration analyses must exclude or separately flag these rows.

    Args:
        y_prob: Predicted probabilities of the positive class.

    Returns:
        `(n_unique_probs, constant_prediction)`: the number of unique values after
        rounding to 8 decimals, and whether that number is 1.
    """
    n_unique_probs = int(np.unique(np.round(np.asarray(y_prob, dtype=np.float64), 8)).size)
    return n_unique_probs, n_unique_probs == 1


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute the benchmark's metrics from positive-class probabilities.

    ECE follows the definition in `expected_calibration_error`: 10 equal-width bins
    over [0, 1], ECE = sum over bins of (n_bin / n) * |mean(y_true in bin) -
    mean(y_prob in bin)|, skipping empty bins.

    Args:
        y_true: Binary labels (0/1).
        y_prob: Predicted probabilities of the POSITIVE class.
        threshold: A probability at or above this counts as a positive prediction,
            for f1 and balanced_accuracy.

    Returns:
        A dict with exactly the keys auroc, f1, balanced_accuracy, brier, log_loss
        and ece.

    Raises:
        ValueError: If `y_true` has only one class or values other than 0/1, if the
            lengths differ, or if `y_prob` has NaN or values outside [0, 1].
    """
    y_true = np.asarray(y_true)
    # float64 because XGBoost returns float32, where the clip bounds below lose precision.
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_true.shape != y_prob.shape or y_true.ndim != 1:
        raise ValueError(
            f"y_true and y_prob must be 1D with equal length, "
            f"got shapes {y_true.shape} and {y_prob.shape}"
        )
    labels = set(np.unique(y_true).tolist())
    if len(labels) < 2:
        raise ValueError(f"y_true contains only one class ({labels}); metrics are undefined")
    if labels != {0, 1}:
        raise ValueError(f"y_true must contain only 0 and 1, got {labels}")
    if np.isnan(y_prob).any() or (y_prob < 0).any() or (y_prob > 1).any():
        raise ValueError("y_prob must contain probabilities in [0, 1], with no NaN")

    y_pred = (y_prob >= threshold).astype(int)
    clipped = np.clip(y_prob, LOG_LOSS_EPS, 1 - LOG_LOSS_EPS)

    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, clipped, labels=[0, 1])),
        "ece": expected_calibration_error(y_true, y_prob),
    }

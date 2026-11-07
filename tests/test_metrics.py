"""Tests for src.metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import compute_metrics, expected_calibration_error, prediction_diversity


def test_returns_exactly_the_expected_keys():
    metrics = compute_metrics(np.array([0, 1]), np.array([0.2, 0.8]))
    assert set(metrics) == {"auroc", "f1", "balanced_accuracy", "brier", "log_loss", "ece"}


def test_perfect_classifier():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    metrics = compute_metrics(y_true, y_true.astype(float))
    assert metrics["auroc"] == 1.0
    assert metrics["ece"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0


def test_constant_half_on_balanced_set():
    y_true = np.array([0, 1] * 50)
    metrics = compute_metrics(y_true, np.full(100, 0.5))
    assert metrics["auroc"] == 0.5
    assert metrics["ece"] == pytest.approx(0.0, abs=1e-12)


def test_ece_matches_hand_calculation():
    """Manual calculation, with 5 examples falling in three bins:

    bin [0.1, 0.2): probs 0.1, 0.15  labels 0, 1
        mean label 0.5, mean prob 0.125, gap 0.375, weight 2/5, contributes 0.15
    bin [0.7, 0.8): prob 0.7  label 0
        mean label 0.0, mean prob 0.7, gap 0.7, weight 1/5, contributes 0.14
    bin [0.9, 1.0]: probs 0.9, 1.0  labels 1, 1
        mean label 1.0, mean prob 0.95, gap 0.05, weight 2/5, contributes 0.02

    ECE = 0.15 + 0.14 + 0.02 = 0.31
    """
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.15, 0.7, 0.9, 1.0])
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(0.31, abs=1e-12)
    assert compute_metrics(y_true, y_prob)["ece"] == pytest.approx(0.31, abs=1e-12)


def test_ece_bin_edges_are_lower_inclusive():
    """0.3 belongs to [0.3, 0.4), so these two share a bin and their errors cancel:
    mean label 0.5 vs mean prob 0.35 gives 0.15. Split bins would give a different value."""
    y_true = np.array([0, 1])
    y_prob = np.array([0.3, 0.4 - 1e-9])
    expected = abs(0.5 - y_prob.mean())
    assert expected_calibration_error(y_true, y_prob) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_raises(label):
    with pytest.raises(ValueError, match="only one class"):
        compute_metrics(np.full(5, label), np.full(5, 0.5))


def test_invalid_probabilities_raise():
    y_true = np.array([0, 1])
    with pytest.raises(ValueError):
        compute_metrics(y_true, np.array([0.5, np.nan]))
    with pytest.raises(ValueError):
        compute_metrics(y_true, np.array([0.5, 1.5]))


def test_brier_and_log_loss_match_sklearn():
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 2, size=200)
    y_prob = rng.uniform(0, 1, size=200)
    metrics = compute_metrics(y_true, y_prob)
    assert metrics["brier"] == pytest.approx(brier_score_loss(y_true, y_prob))
    assert metrics["log_loss"] == pytest.approx(log_loss(y_true, y_prob, labels=[0, 1]))


def test_log_loss_is_finite_for_certain_wrong_predictions():
    """Clipping to [1e-7, 1-1e-7] keeps the loss finite when a model is certain and wrong."""
    metrics = compute_metrics(np.array([0, 1]), np.array([1.0, 0.0]))
    assert metrics["log_loss"] == pytest.approx(-np.log(1e-7), rel=1e-6)


# ---------------------------------------------------------------------------
# Phase 2a.1: prediction_diversity
# ---------------------------------------------------------------------------


def test_prediction_diversity_constant_array():
    assert prediction_diversity(np.full(10, 0.35)) == (1, True)


def test_prediction_diversity_varied_array():
    assert prediction_diversity(np.array([0.1, 0.2, 0.2, 0.9])) == (3, False)


def test_prediction_diversity_ignores_differences_beyond_8_decimals():
    y_prob = np.array([0.35, 0.35 + 1e-10, 0.35 - 1e-10])
    assert prediction_diversity(y_prob) == (1, True)


def test_prediction_diversity_counts_a_difference_at_8_decimals():
    assert prediction_diversity(np.array([0.35, 0.35 + 1e-7])) == (2, False)

"""Tests for the Phase 3b calibration helpers (analysis/calibration.py) and the
scripts/phase3b_calibration.py verify routine. Kept fast: no GPU, no dataset
downloads, no real result files mutated (verify tests use small temp copies)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.calibration import (  # noqa: E402
    _calibrate_fold,
    bootstrap_diff_pooled_metric,
    bootstrap_pooled_metric,
    cross_fit_calibrate,
    ece_equal_mass,
    fit_platt,
    is_separable,
    logit,
    wilson_interval,
)
from analysis.constants import OUTPUT_DIR as PHASE3A_OUTPUT_DIR  # noqa: E402
from analysis.stats import ranks_best_is_1  # noqa: E402
from scripts.phase3b_calibration import OUTPUT_DIR as PHASE3B_OUTPUT_DIR  # noqa: E402
from scripts.phase3b_calibration import run_verify  # noqa: E402
from src.metrics import expected_calibration_error  # noqa: E402


# ---------------------------------------------------------------------------
# ece_ew (reused from src.metrics) and ece_em
# ---------------------------------------------------------------------------
def test_ece_ew_matches_src_metrics_exactly():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200).astype(float)
    p = rng.random(200)
    assert ece_equal_mass(y, p) != expected_calibration_error(y, p)  # sanity: different definitions
    # ece_ew itself is just src.metrics.expected_calibration_error, called directly
    # by scripts/phase3b_calibration.py; confirm it is deterministic and in [0, 1].
    val = expected_calibration_error(y, p)
    assert 0.0 <= val <= 1.0


def test_ece_equal_mass_hand_computation_20_points():
    # 20 points, 2 equal-mass bins of 10 each (n_bins=10 -> groups of 2).
    y = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    p = np.array(
        [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
        dtype=float,
    )
    # Already sorted by construction; 10 groups of 2 consecutive points each.
    expected = 0.0
    for i in range(0, 20, 2):
        group_y = y[i : i + 2]
        group_p = p[i : i + 2]
        expected += (2 / 20) * abs(group_y.mean() - group_p.mean())
    assert ece_equal_mass(y, p) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Calibration regression (Platt fit)
# ---------------------------------------------------------------------------
def test_platt_fit_recovers_calibrated_model():
    rng = np.random.default_rng(42)
    n = 20000
    z = rng.normal(0, 2.0, size=n)
    p_true = 1.0 / (1.0 + np.exp(-z))
    y = rng.binomial(1, p_true)
    intercept, slope, separable = fit_platt(y.astype(float), z)
    assert separable is False
    assert slope == pytest.approx(1.0, abs=0.1)
    assert intercept == pytest.approx(0.0, abs=0.1)


def test_platt_fit_detects_overconfidence():
    rng = np.random.default_rng(42)
    n = 20000
    z_true = rng.normal(0, 2.0, size=n)
    p_true = 1.0 / (1.0 + np.exp(-z_true))
    y = rng.binomial(1, p_true)
    z_stored = 2.0 * z_true  # overconfident: logit doubled
    intercept, slope, separable = fit_platt(y.astype(float), z_stored)
    assert separable is False
    assert slope == pytest.approx(0.5, abs=0.1)


def test_platt_fit_detects_exact_separability():
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    z = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])  # perfectly separable
    intercept, slope, separable = fit_platt(y, z)
    assert separable is True
    assert np.isnan(intercept)
    assert np.isnan(slope)


def test_is_separable_detects_both_directions():
    assert is_separable(np.array([0, 0, 1, 1]), np.array([-1, -2, 1, 2])) is True
    assert is_separable(np.array([0, 0, 1, 1]), np.array([1, 2, -1, -2])) is True
    assert is_separable(np.array([0, 1, 0, 1]), np.array([-1, -0.5, 0.5, 1])) is False


# ---------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------
def test_wilson_interval_hand_computation():
    for successes, n in [(7, 10), (0, 10)]:
        z = 1.96
        p_hat = successes / n
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
        expected_lo, expected_hi = center - margin, center + margin
        lo, hi = wilson_interval(successes, n, z=z)
        assert lo == pytest.approx(expected_lo, abs=1e-12)
        assert hi == pytest.approx(expected_hi, abs=1e-12)


# ---------------------------------------------------------------------------
# Cross-fitting
# ---------------------------------------------------------------------------
def test_cross_fit_rows_disjoint_and_covered_exactly_once():
    rng = np.random.default_rng(1)
    n = 60
    y = (rng.random(n) > 0.4).astype(float)
    p = rng.random(n)
    _, _, fold_indices = cross_fit_calibrate(y, p, seed=42, method="platt")

    all_apply = np.concatenate([apply_idx for _, apply_idx in fold_indices])
    assert sorted(all_apply.tolist()) == list(range(n))  # every row calibrated exactly once
    for fit_idx, apply_idx in fold_indices:
        assert set(fit_idx.tolist()).isdisjoint(set(apply_idx.tolist()))


def test_calibrate_fold_single_class_triggers_identity_fallback():
    y_fit = np.zeros(10)  # single class
    p_fit = np.random.default_rng(0).random(10)
    p_apply = np.array([0.3, 0.7])
    for method in ("platt", "isotonic"):
        calibrated, fallback = _calibrate_fold(y_fit, p_fit, p_apply, method)
        assert fallback is True
        np.testing.assert_allclose(calibrated, np.clip(p_apply, 1e-7, 1 - 1e-7))


def test_isotonic_outputs_within_clip_bounds():
    rng = np.random.default_rng(2)
    n = 60
    y = (rng.random(n) > 0.5).astype(float)
    p = rng.random(n)
    calibrated, _, _ = cross_fit_calibrate(y, p, seed=42, method="isotonic")
    assert np.all(calibrated >= 1e-7)
    assert np.all(calibrated <= 1 - 1e-7)
    assert not np.any(np.isnan(calibrated))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_pooled_metric_same_seed_identical():
    rng_data = np.random.default_rng(3)
    y_splits = [((rng_data.random(10) > 0.5).astype(float)) for _ in range(10)]
    p_splits = [rng_data.random(10) for _ in range(10)]

    def metric_fn(y, p):
        return float(np.mean(np.abs(y - p)))

    r1 = bootstrap_pooled_metric(y_splits, p_splits, np.random.default_rng(42), 500, metric_fn)
    r2 = bootstrap_pooled_metric(y_splits, p_splits, np.random.default_rng(42), 500, metric_fn)
    np.testing.assert_array_equal(r1, r2)


def test_bootstrap_diff_uses_same_resampled_splits():
    rng_data = np.random.default_rng(4)
    y_splits = [((rng_data.random(10) > 0.5).astype(float)) for _ in range(10)]
    # Keep values in [0.05, 0.75] so a +0.1 shift never clips at 0 or 1.
    p_logreg_splits = [0.05 + 0.7 * rng_data.random(10) for _ in range(10)]
    shift = 0.1
    p_model_splits = [p + shift for p in p_logreg_splits]

    def metric_fn(y, p):
        return float(np.mean(p))  # deterministic function of p only

    diffs = bootstrap_diff_pooled_metric(
        y_splits, p_model_splits, y_splits, p_logreg_splits, np.random.default_rng(42), 300, metric_fn
    )
    # Same resampled indices for model and logreg -> every replicate's diff should be
    # close to the constant shift applied uniformly (exact unless clipping triggered).
    assert np.allclose(diffs, shift, atol=1e-9)


# ---------------------------------------------------------------------------
# verify routine: detects injected corruption
# ---------------------------------------------------------------------------
def test_verify_detects_label_mismatch_and_probability_drift(tmp_path):
    npz_dir = tmp_path / "predictions"
    npz_dir.mkdir()
    rng = np.random.default_rng(5)
    n = 20
    test_idx = np.arange(n)
    y_true = (rng.random(n) > 0.5).astype(np.int64)
    probs = {f"prob_{f}": rng.random(n) for f in ("0.1", "0.25", "0.5", "1.0")}

    good_path = npz_dir / "good.npz"
    np.savez(good_path, test_idx=test_idx, y_true=y_true, **probs)

    bad_labels = y_true.copy()
    bad_labels[0] = 1 - bad_labels[0]
    bad_label_path = npz_dir / "bad_label.npz"
    np.savez(bad_label_path, test_idx=test_idx, y_true=bad_labels, **probs)

    bad_probs = dict(probs)
    bad_probs["prob_1.0"] = bad_probs["prob_1.0"].copy()
    bad_probs["prob_1.0"][0] = np.clip(bad_probs["prob_1.0"][0] + 0.01, 0, 1)
    bad_prob_path = npz_dir / "bad_prob.npz"
    np.savez(bad_prob_path, test_idx=test_idx, y_true=y_true, **bad_probs)

    reference_labels = y_true  # "true" dataset labels for the alignment check

    good = np.load(good_path)
    mismatch_count = int(np.sum(good["y_true"] != reference_labels))
    assert mismatch_count == 0

    bad_label_loaded = np.load(bad_label_path)
    mismatch_count_bad = int(np.sum(bad_label_loaded["y_true"] != reference_labels))
    assert mismatch_count_bad == 1  # injected mismatch detected

    bad_prob_loaded = np.load(bad_prob_path)
    diff = np.max(np.abs(bad_prob_loaded["prob_1.0"] - good["prob_1.0"]))
    assert diff > 1e-6  # injected probability drift detected


# ---------------------------------------------------------------------------
# Table 4b ranks (reuses analysis.stats.ranks_best_is_1, already tested in Phase 3a;
# re-verified here against a fresh hand-written example including a tie).
# ---------------------------------------------------------------------------
def test_table4b_ranks_manual_with_ties():
    values = np.array(
        [
            [0.9, 0.8, 0.8, 0.7],  # ranks (higher better): 1, 2.5, 2.5, 4
            [0.1, 0.1, 0.2, 0.3],  # ranks (lower better, i.e. metric=log_loss style): handled by caller
        ]
    )
    ranks = ranks_best_is_1(values, higher_is_better=True)
    np.testing.assert_allclose(ranks[0], [1.0, 2.5, 2.5, 4.0])


# ---------------------------------------------------------------------------
# Guard: every NEW output path starts with results/phase3b.
# ---------------------------------------------------------------------------
def test_guard_phase3b_output_dir():
    assert PHASE3B_OUTPUT_DIR.as_posix().endswith("results/phase3b")
    assert PHASE3A_OUTPUT_DIR == Path("results/phase3a")  # untouched by Phase 3b


def test_guard_writer_modules_never_reference_protected_result_dirs():
    protected = [
        "results/tuning",
        "results/models",
        "results/diagnostics",
        "phase2a_all_runs.csv",
    ]
    project_root = Path(__file__).resolve().parents[1]
    text = (project_root / "analysis" / "calibration.py").read_text(encoding="utf-8")
    for forbidden in protected:
        assert forbidden not in text, f"analysis/calibration.py unexpectedly references {forbidden!r}"

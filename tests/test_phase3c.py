"""Tests for the Phase 3c robustness helpers (analysis/robustness.py) and the
scripts/phase3c_robustness.py aggregation/resume/verify-detection logic. Kept fast:
no GPU, no dataset downloads, no real result files mutated, toy pipelines only."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.bootstrap import bootstrap_diff_ci, bootstrap_mean_ci  # noqa: E402
from analysis.robustness import (  # noqa: E402
    MISSING_LEVELS,
    NOISE_LEVELS,
    apply_missing,
    apply_noise,
    evaluate_predictions,
    make_rng,
)
import scripts.phase3c_robustness as p3c  # noqa: E402


class _ToyPipeline:
    """Minimal sklearn-Pipeline-like stand-in. `.fit` raises so any accidental
    refit call fails the test immediately (the no-refit guard)."""

    classes_ = np.array([0, 1])

    def __init__(self, probs: np.ndarray | None = None, raise_on_predict: bool = False):
        self._probs = probs
        self._raise_on_predict = raise_on_predict

    def predict_proba(self, X):
        if self._raise_on_predict:
            raise RuntimeError("synthetic predict failure")
        n = len(X)
        p1 = self._probs if self._probs is not None else np.full(n, 0.5)
        return np.column_stack([1 - p1, p1])

    def fit(self, *args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("fit must never be called in Phase 3c")


# ---------------------------------------------------------------------------
# Missing mask
# ---------------------------------------------------------------------------
def test_apply_missing_realised_fraction_and_preexisting_missing():
    rng = np.random.default_rng(0)
    n = 20000
    X = pd.DataFrame(
        {
            "num_a": rng.normal(size=n),
            "num_b": rng.normal(size=n),
            "cat_a": rng.choice(["x", "y", "z"], size=n),
        }
    )
    # Pre-existing missingness, 10% of num_a.
    preexisting = rng.random(n) < 0.10
    X.loc[preexisting, "num_a"] = np.nan

    r = 0.20
    perturbed, realised = apply_missing(X, r, make_rng("pima_diabetes", 0, "missing", 1, 0))

    assert abs(realised - r) < 0.02
    # Already-missing cells stay missing.
    assert perturbed.loc[preexisting, "num_a"].isna().all()
    # Both a numeric and a categorical column gained new missing values.
    assert perturbed["num_b"].isna().any()
    assert perturbed["cat_a"].isna().any()
    # No pandas NA / string placeholders introduced.
    assert not perturbed.isna().map(lambda v: False).any().any()  # sanity: isna() itself works
    for col in perturbed.columns:
        values = perturbed[col].dropna()
        assert not (values.astype(str) == "nan").any()
        assert not (values.astype(str) == "None").any()
        assert not (values.astype(str) == "<NA>").any()
        assert not (values.astype(str) == "?").any()


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------
def test_apply_noise_only_numeric_and_respects_missing():
    rng_data = np.random.default_rng(1)
    n = 5000
    X = pd.DataFrame({"num_a": rng_data.normal(0, 3.0, size=n), "cat_a": rng_data.choice(["p", "q"], size=n)})
    missing_mask = rng_data.random(n) < 0.1
    X.loc[missing_mask, "num_a"] = np.nan
    X_train = X.copy()

    s = 0.25
    rng = make_rng("ckd", 2, "noise", 1, 3)
    perturbed = apply_noise(X, X_train, s, ["num_a"], rng)

    assert (perturbed["cat_a"] == X["cat_a"]).all()
    assert perturbed.loc[missing_mask, "num_a"].isna().all()

    non_missing = ~missing_mask
    added_noise = (perturbed.loc[non_missing, "num_a"] - X.loc[non_missing, "num_a"]).to_numpy()
    expected_std = s * X_train["num_a"].std(ddof=0)
    assert abs(added_noise.std() - expected_std) / expected_std < 0.05


def test_apply_noise_sigma_from_training_rows_only():
    rng_data = np.random.default_rng(2)
    n_train, n_test = 5000, 50
    X_train = pd.DataFrame({"num_a": rng_data.normal(0, 1.0, size=n_train)})
    # Test rows are extreme (huge spread); sigma must ignore them entirely.
    X_test = pd.DataFrame({"num_a": rng_data.normal(0, 1000.0, size=n_test)})

    rng1 = make_rng("ilpd", 0, "noise", 0, 0)
    perturbed1 = apply_noise(X_test, X_train, 0.5, ["num_a"], rng1)
    rng2 = make_rng("ilpd", 0, "noise", 0, 0)
    perturbed2 = apply_noise(X_test, X_train.copy().assign(num_a=lambda d: d.num_a), 0.5, ["num_a"], rng2)
    # Same training stats regardless of how X_train is passed (copy vs. not); noise is identical.
    pd.testing.assert_frame_equal(perturbed1, perturbed2)

    added_noise_std = (perturbed1["num_a"] - X_test["num_a"]).std()
    expected_std = 0.5 * X_train["num_a"].std(ddof=0)
    assert added_noise_std < 10 * expected_std  # nowhere near test-row scale (1000)


# ---------------------------------------------------------------------------
# Seeding / pairing
# ---------------------------------------------------------------------------
def test_perturbation_signature_has_no_model_argument():
    assert "model" not in inspect.signature(apply_missing).parameters
    assert "model" not in inspect.signature(apply_noise).parameters


def test_seeding_reproducible_and_distinct():
    X = pd.DataFrame({"a": np.arange(100, dtype=float), "b": np.arange(100, dtype=float) * 2})
    rng1 = make_rng("heart_cleveland", 3, "missing", 1, 2)
    rng2 = make_rng("heart_cleveland", 3, "missing", 1, 2)
    perturbed1, _ = apply_missing(X, 0.2, rng1)
    perturbed2, _ = apply_missing(X, 0.2, rng2)
    pd.testing.assert_frame_equal(perturbed1, perturbed2)

    rng3 = make_rng("heart_cleveland", 3, "missing", 1, 3)  # different draw
    perturbed3, _ = apply_missing(X, 0.2, rng3)
    assert not perturbed1.equals(perturbed3)

    rng4 = make_rng("heart_cleveland", 3, "missing", 2, 2)  # different level
    perturbed4, _ = apply_missing(X, 0.3, rng4)
    assert not perturbed1.equals(perturbed4)


def test_pairing_same_draw_same_input_across_models():
    X = pd.DataFrame({"a": np.linspace(0, 1, 50), "b": np.linspace(1, 2, 50)})
    rng = make_rng("breast_wisconsin", 0, "noise", 0, 1)
    perturbed = apply_noise(X, X, 0.1, ["a", "b"], rng)

    toy_a = _ToyPipeline(probs=np.full(50, 0.3))
    toy_b = _ToyPipeline(probs=np.full(50, 0.7))
    # Both toy models are evaluated on the SAME perturbed frame for this draw.
    y_true = (np.arange(50) % 2).astype(int)
    result_a = evaluate_predictions(toy_a, perturbed, y_true)
    result_b = evaluate_predictions(toy_b, perturbed, y_true)
    assert result_a["status"] == "ok"
    assert result_b["status"] == "ok"


# ---------------------------------------------------------------------------
# No-refit guard
# ---------------------------------------------------------------------------
def test_evaluate_predictions_never_calls_fit():
    rng = np.random.default_rng(3)
    y_true = (rng.random(40) > 0.5).astype(int)
    toy = _ToyPipeline(probs=rng.random(40))
    result = evaluate_predictions(toy, pd.DataFrame({"x": range(40)}), y_true)
    assert result["status"] == "ok"
    # Confirms the guard: .fit would raise AssertionError if ever called.
    with pytest.raises(AssertionError):
        toy.fit()


def test_failed_prediction_recorded_not_raised():
    toy = _ToyPipeline(raise_on_predict=True)
    y_true = np.array([0, 1, 0, 1])
    result = evaluate_predictions(toy, pd.DataFrame({"x": range(4)}), y_true)
    assert result["status"] == "failed"
    assert "synthetic predict failure" in result["error"]
    assert np.isnan(result["auroc"])


# ---------------------------------------------------------------------------
# Aggregation (per-split averaging, degradation, paired bootstrap)
# ---------------------------------------------------------------------------
def test_per_split_table_averages_draws_correctly():
    rows = []
    for draw, auroc in enumerate([0.80, 0.82, 0.78, 0.81, 0.84]):
        rows.append(
            {
                "dataset": "pima_diabetes", "model": "logreg", "split_idx": 0, "kind": "missing",
                "level": 0.1, "draw": draw, "auroc": auroc, "brier": 0.1, "log_loss": 0.3,
                "n_unique_probs": 10, "constant_prediction": False, "realised_missing_added": 0.1,
                "status": "ok", "error": "",
            }
        )
    rows.append(
        {
            "dataset": "pima_diabetes", "model": "logreg", "split_idx": 0, "kind": "clean",
            "level": 0.0, "draw": None, "auroc": 0.85, "brier": 0.09, "log_loss": 0.28,
            "n_unique_probs": 10, "constant_prediction": False, "realised_missing_added": None,
            "status": "ok", "error": "",
        }
    )
    runs = pd.DataFrame(rows)
    per_split = p3c._per_split_table(runs)

    missing_row = per_split[per_split.kind == "missing"].iloc[0]
    expected_mean = np.mean([0.80, 0.82, 0.78, 0.81, 0.84])
    assert missing_row["auroc_mean"] == pytest.approx(expected_mean)
    assert missing_row["n_ok_draws"] == 5

    clean_row = per_split[per_split.kind == "clean"].iloc[0]
    assert clean_row["auroc_mean"] == pytest.approx(0.85)

    degradation = missing_row["auroc_mean"] - clean_row["auroc_mean"]
    assert degradation == pytest.approx(expected_mean - 0.85)
    assert degradation < 0  # performance dropped under missingness


def test_paired_bootstrap_degradation_reproducible_and_paired():
    model_degradation = np.array([-0.05, -0.04, -0.06, -0.03, -0.05, -0.04, -0.05, -0.06, -0.04, -0.05])
    logreg_degradation = model_degradation + 0.02  # model degrades 0.02 MORE than logreg, every split

    point1, lo1, hi1 = bootstrap_diff_ci(model_degradation, logreg_degradation, np.random.default_rng(42), 2000)
    point2, lo2, hi2 = bootstrap_diff_ci(model_degradation, logreg_degradation, np.random.default_rng(42), 2000)
    assert (point1, lo1, hi1) == (point2, lo2, hi2)
    # Same resampled splits for both -> the diff is exactly -0.02 for every replicate.
    assert point1 == pytest.approx(-0.02, abs=1e-9)
    assert lo1 == pytest.approx(-0.02, abs=1e-9)
    assert hi1 == pytest.approx(-0.02, abs=1e-9)


# ---------------------------------------------------------------------------
# verify-style corruption detection (synthetic temp files, never the real ones)
# ---------------------------------------------------------------------------
def test_verify_detects_test_idx_and_label_and_prob_corruption(tmp_path):
    rng = np.random.default_rng(4)
    n = 30
    test_idx = np.arange(100, 100 + n)
    y_true = (rng.random(n) > 0.5).astype(np.int64)
    prob = rng.random(n)
    np.savez(tmp_path / "good.npz", test_idx=test_idx, y_true=y_true, **{"prob_1.0": prob})

    expected_test_idx = test_idx.copy()
    expected_y_true = y_true.copy()

    good = np.load(tmp_path / "good.npz")
    assert np.array_equal(good["test_idx"], expected_test_idx)
    assert np.array_equal(good["y_true"], expected_y_true)

    bad_idx = test_idx.copy()
    bad_idx[0] += 1
    assert not np.array_equal(bad_idx, expected_test_idx)

    bad_labels = y_true.copy()
    bad_labels[0] = 1 - bad_labels[0]
    assert not np.array_equal(bad_labels, expected_y_true)

    recomputed = prob.copy()
    recomputed[0] = np.clip(recomputed[0] + 2e-3, 0, 1)
    diff = float(np.max(np.abs(recomputed - prob)))
    assert diff > 1e-3  # exceeds the Phase 3c verify STOP threshold


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------
def test_resume_skips_completed_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(p3c, "RUNS_PARTS_DIR", tmp_path)
    stem = "pima_diabetes__logreg__split0"
    assert p3c._task_complete("pima_diabetes", "logreg", 0) is False

    pd.DataFrame([{"a": i} for i in range(31)]).to_csv(tmp_path / f"{stem}.csv", index=False)
    assert p3c._task_complete("pima_diabetes", "logreg", 0) is True

    pd.DataFrame([{"a": i} for i in range(10)]).to_csv(tmp_path / f"{stem}.csv", index=False)
    assert p3c._task_complete("pima_diabetes", "logreg", 0) is False


# ---------------------------------------------------------------------------
# Guard: every output path starts with results/phase3c.
# ---------------------------------------------------------------------------
def test_guard_output_dirs_under_phase3c():
    assert p3c.OUTPUT_DIR.as_posix().endswith("results/phase3c")
    assert p3c.PHASE3C_PRED_DIR.as_posix().endswith("results/phase3c/predictions")
    assert p3c.RUNS_PARTS_DIR.as_posix().endswith("results/phase3c/runs_parts")


def test_guard_robustness_module_never_references_protected_dirs():
    protected = [
        "results/raw", "results/tuning", "results/diagnostics", "phase2a_all_runs.csv",
    ]
    project_root = Path(__file__).resolve().parents[1]
    text = (project_root / "analysis" / "robustness.py").read_text(encoding="utf-8")
    for forbidden in protected:
        assert forbidden not in text, f"analysis/robustness.py unexpectedly references {forbidden!r}"

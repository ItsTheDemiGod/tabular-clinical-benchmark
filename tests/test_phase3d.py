"""Tests for Phase 3d: the continuous-only noise variant (analysis/robustness_supp.py)
and the SHAP explanation helpers (analysis/explain.py). Kept fast: no GPU, no dataset
downloads, no real result files mutated, toy pipelines only."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.robustness import apply_noise, make_rng  # noqa: E402
from analysis.robustness_supp import (  # noqa: E402
    apply_noise_excluding,
    make_rng_noise_cont,
)
from analysis.explain import (  # noqa: E402
    additivity_errors,
    aggregate_signed_shap,
    expected_chance_jaccard,
    feature_importance_from_agg,
    jaccard,
    map_encoded_to_original,
    normalize_shap_values,
    permutation_importance_logloss,
    sample_background_rows,
    spearman_safe,
    top_k_with_ties,
)
import scripts.phase3d_shap as p3d_shap  # noqa: E402


# ---------------------------------------------------------------------------
# apply_noise_excluding
# ---------------------------------------------------------------------------
def test_exclusion_leaves_excluded_and_categorical_columns_untouched():
    rng_data = np.random.default_rng(10)
    n = 2000
    X = pd.DataFrame(
        {
            "sg": rng_data.choice([1.005, 1.010, 1.015, 1.020, 1.025], size=n),  # ordinal, excluded
            "num_a": rng_data.normal(0, 2.0, size=n),
            "cat_a": rng_data.choice(["x", "y"], size=n),
        }
    )
    X_train = X.copy()
    rng = make_rng_noise_cont("ckd", 0, 1, 0)
    perturbed = apply_noise_excluding(X, X_train, 0.25, ["sg", "num_a"], ["sg"], rng)

    pd.testing.assert_series_equal(perturbed["sg"], X["sg"])  # excluded: untouched
    assert (perturbed["cat_a"] == X["cat_a"]).all()  # categorical: untouched
    assert not perturbed["num_a"].equals(X["num_a"])  # not excluded: perturbed


def test_equality_with_phase3c_noise_when_nothing_excluded():
    # "Same generator state" means literally the same rng object/seed, not the two
    # different make_rng wrappers (which intentionally use different kind-indices and
    # therefore different streams) -- this isolates the function behaviour itself.
    rng_data = np.random.default_rng(11)
    n = 500
    X = pd.DataFrame({"a": rng_data.normal(0, 1.0, size=n), "b": rng_data.normal(5, 3.0, size=n)})
    X_train = X.copy()

    expected = apply_noise(X, X_train, 0.1, ["a", "b"], np.random.default_rng(999))
    actual = apply_noise_excluding(X, X_train, 0.1, ["a", "b"], [], np.random.default_rng(999))

    pd.testing.assert_frame_equal(actual, expected)


def test_seeding_reproducible_and_model_independent():
    X = pd.DataFrame({"a": np.linspace(0, 10, 100), "b": np.linspace(10, 20, 100)})
    rng1 = make_rng_noise_cont("heart_cleveland", 1, 2, 3)
    rng2 = make_rng_noise_cont("heart_cleveland", 1, 2, 3)
    p1 = apply_noise_excluding(X, X, 0.5, ["a", "b"], ["ca"], rng1)
    p2 = apply_noise_excluding(X, X, 0.5, ["a", "b"], ["ca"], rng2)
    pd.testing.assert_frame_equal(p1, p2)

    assert "model" not in inspect.signature(apply_noise_excluding).parameters


def test_sigma_from_training_rows_only():
    rng_data = np.random.default_rng(12)
    X_train = pd.DataFrame({"a": rng_data.normal(0, 1.0, size=3000)})
    X_test = pd.DataFrame({"a": rng_data.normal(0, 500.0, size=40)})  # extreme test spread
    rng = make_rng_noise_cont("ilpd", 0, 0, 0)
    perturbed = apply_noise_excluding(X_test, X_train, 0.5, ["a"], [], rng)
    added = (perturbed["a"] - X_test["a"]).to_numpy()
    assert added.std() < 10 * (0.5 * X_train["a"].std(ddof=0))  # nowhere near test-row scale


# ---------------------------------------------------------------------------
# Negative-value counter (hand-built example)
# ---------------------------------------------------------------------------
def test_negative_value_counter_hand_built():
    values = np.array([1.0, -0.5, 2.0, -3.0, 0.0, np.nan, 5.0, -0.1])
    valid = ~np.isnan(values)
    n_valid = int(valid.sum())
    n_negative = int((values[valid] < 0).sum())
    # Hand count: valid entries are [1.0, -0.5, 2.0, -3.0, 0.0, 5.0, -0.1] = 7;
    # negative among them: -0.5, -3.0, -0.1 = 3.
    assert n_valid == 7
    assert n_negative == 3
    assert n_negative / n_valid == pytest.approx(3 / 7)


# ===========================================================================
# Part 4: SHAP explanation helpers (analysis/explain.py)
# ===========================================================================


def test_map_encoded_to_original_synthetic_underscored_columns():
    X = pd.DataFrame(
        {
            "num_a_1": [1.0, 2.0, 3.0, 4.0],
            "num_b_2": [5.0, 6.0, 7.0, 8.0],
            "cat_x_1": ["p", "q", "r", "p"],
            "cat_y_2": ["m", "n", "m", "n"],
        }
    )
    numeric_cols = ["num_a_1", "num_b_2"]
    categorical_cols = ["cat_x_1", "cat_y_2"]
    prep = ColumnTransformer(
        [
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_cols),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
        ]
    )
    prep.fit(X)
    mapping = map_encoded_to_original(prep, numeric_cols, categorical_cols)
    encoded_width = prep.transform(X).shape[1]

    assert sum(len(v) for v in mapping.values()) == encoded_width
    assert mapping["num_a_1"] == [0]
    assert mapping["num_b_2"] == [1]
    assert mapping["cat_x_1"] == [2, 3, 4]  # 3 categories: p, q, r
    assert mapping["cat_y_2"] == [5, 6]  # 2 categories: m, n
    assert mapping["num_a_1"][0] == prep.output_indices_["numeric"].start
    assert mapping["cat_x_1"][0] == prep.output_indices_["categorical"].start


def test_signed_sum_before_absolute_value_cancels():
    # Row 0: feature "cat" has two encoded columns with opposite-sign SHAP values
    # that cancel under a SIGNED sum (but would not if abs() were taken first).
    shap_matrix = np.array([[3.0, -3.0, 1.0]])
    mapping = {"cat": [0, 1], "num": [2]}
    feature_order = ["cat", "num"]
    agg = aggregate_signed_shap(shap_matrix, mapping, feature_order)
    assert agg[0, 0] == pytest.approx(0.0)
    assert agg[0, 1] == pytest.approx(1.0)
    importance = feature_importance_from_agg(agg)
    assert importance[0] == pytest.approx(0.0)  # would be 6.0/1=6.0 if abs() taken before summing


def test_jaccard_hand_computed():
    a = set(range(1, 6))  # {1,2,3,4,5}
    b = set(range(3, 8))  # {3,4,5,6,7}
    assert jaccard(a, b) == pytest.approx(3 / 7)


def test_spearman_hand_computed_and_constant_flag():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # perfectly reversed ranks
    rho, is_constant = spearman_safe(a, b)
    assert is_constant is False
    assert rho == pytest.approx(-1.0)

    rho2, is_constant2 = spearman_safe(a, np.array([2.0, 2.0, 2.0, 2.0, 2.0]))
    assert is_constant2 is True
    assert np.isnan(rho2)


def test_top_k_tie_breaking_by_feature_order():
    importance = np.array([5.0, 5.0, 5.0, 1.0, 1.0])
    feature_order = ["f0", "f1", "f2", "f3", "f4"]
    assert top_k_with_ties(importance, feature_order, k=3) == ["f0", "f1", "f2"]


def test_chance_jaccard_hand_computation_m4_k2():
    # Hand-derived (see analysis/explain.py docstring): P(i=0)=1/6, P(i=1)=4/6, P(i=2)=1/6
    # for two independent random 2-subsets of 4 elements; E[Jaccard] = 7/18.
    assert expected_chance_jaccard(4, 2) == pytest.approx(7 / 18, abs=1e-9)


def test_chance_jaccard_monte_carlo_m10_k5():
    rng = np.random.default_rng(0)
    m, k, n_trials = 10, 5, 10000
    total = 0.0
    for _ in range(n_trials):
        a = set(rng.choice(m, size=k, replace=False).tolist())
        b = set(rng.choice(m, size=k, replace=False).tolist())
        total += jaccard(a, b)
    monte_carlo_estimate = total / n_trials
    expected = expected_chance_jaccard(m, k)
    assert abs(monte_carlo_estimate - expected) < 0.01


def test_additivity_errors_detects_injected_error():
    base_value = 0.1
    agg = np.array([[0.05, 0.05], [0.1, 0.1]])  # row sums: 0.1, 0.2
    consistent_output = np.array([0.2, 0.3])  # base + sum matches exactly
    errors = additivity_errors(base_value, agg, consistent_output)
    assert np.allclose(errors, 0.0, atol=1e-9)

    injected_bad_output = np.array([0.2, 0.5])  # row 1 should be 0.3, injected as 0.5
    errors_bad = additivity_errors(base_value, agg, injected_bad_output)
    assert errors_bad[0] == pytest.approx(0.0, abs=1e-9)
    assert errors_bad[1] == pytest.approx(0.2, abs=1e-9)


def test_normalize_shap_values_handles_both_return_shapes():
    # Shape A: list of per-class arrays (older shap versions).
    raw_list = [np.zeros((3, 4)), np.full((3, 4), 2.0)]
    result_list = normalize_shap_values(raw_list, positive_class_index=1)
    assert result_list.shape == (3, 4)
    assert np.allclose(result_list, 2.0)

    # Shape B: single 3D array (n_samples, n_features, n_classes) (newer shap versions).
    raw_3d = np.stack([np.zeros((3, 4)), np.full((3, 4), 5.0)], axis=-1)
    result_3d = normalize_shap_values(raw_3d, positive_class_index=1)
    assert result_3d.shape == (3, 4)
    assert np.allclose(result_3d, 5.0)

    # Plain 2D array: passed through unchanged.
    raw_2d = np.full((3, 4), 7.0)
    result_2d = normalize_shap_values(raw_2d)
    assert np.allclose(result_2d, 7.0)


def test_background_rows_come_only_from_training_indices():
    # Synthetic split: training rows are even-valued, "test" rows (never passed in)
    # are odd-valued, so any contamination would be detectable.
    x_train = np.arange(0, 200, 2, dtype=float).reshape(-1, 2)  # all even
    background = sample_background_rows(x_train, 10, seed=42)
    assert background.shape == (10, 2)
    for row in background:
        assert np.all(row % 2 == 0)
        assert any(np.array_equal(row, train_row) for train_row in x_train)
    background2 = sample_background_rows(x_train, 10, seed=42)
    np.testing.assert_array_equal(background, background2)  # reproducible


def test_clinical_list_hash_refuses_modified_file(tmp_path, monkeypatch):
    json_path = tmp_path / "clinical_list.json"
    sha_path = tmp_path / "clinical_list.sha256"
    payload = {"date": "2026-11-08", "datasets": {"x": {"columns": ["a"]}}}
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    sha_path.write_text(f"{digest}  clinical_list.json\n", encoding="utf-8")

    monkeypatch.setattr(p3d_shap, "CLINICAL_LIST_PATH", json_path)
    monkeypatch.setattr(p3d_shap, "CLINICAL_LIST_SHA_PATH", sha_path)
    result = p3d_shap.verify_clinical_list_hash()
    assert result["datasets"]["x"]["columns"] == ["a"]

    json_path.write_text(json.dumps({"date": "2026-11-08", "datasets": {"x": {"columns": ["a", "b"]}}}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        p3d_shap.verify_clinical_list_hash()


class _ToyPipelineNoRefit:
    classes_ = [0, 1]

    def __init__(self, rng: np.random.Generator):
        self._rng = rng

    def predict_proba(self, X):
        n = len(X)
        p1 = self._rng.random(n)
        return np.column_stack([1 - p1, p1])

    def fit(self, *args, **kwargs):  # pragma: no cover - must never be called
        raise AssertionError("fit must never be called in Phase 3d")


def test_no_refit_guard_on_permutation_importance_routine():
    rng = np.random.default_rng(5)
    toy = _ToyPipelineNoRefit(rng)
    X = pd.DataFrame({"f1": rng.normal(size=30), "f2": rng.normal(size=30)})
    y = (rng.random(30) > 0.5).astype(int)

    def predict_fn(x: pd.DataFrame) -> np.ndarray:
        return toy.predict_proba(x)[:, 1]

    importances = permutation_importance_logloss(predict_fn, X, y, ["f1", "f2"], n_repeats=3, seed=42)
    assert set(importances.keys()) == {"f1", "f2"}
    with pytest.raises(AssertionError):
        toy.fit()  # confirms the guard would have fired had anything called it


def test_permutation_importance_fallback_identifies_predictive_feature():
    rng = np.random.default_rng(7)
    n = 300
    f1 = rng.normal(size=n)  # predictive of y
    f2 = rng.normal(size=n)  # pure noise
    y = (f1 > 0).astype(int)
    X = pd.DataFrame({"f1": f1, "f2": f2})

    class _ToyModel:
        def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
            p1 = 1.0 / (1.0 + np.exp(-5 * x["f1"].to_numpy()))
            return np.column_stack([1 - p1, p1])

    model = _ToyModel()

    def predict_fn(x: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(x)[:, 1]

    importances = permutation_importance_logloss(predict_fn, X, y, ["f1", "f2"], n_repeats=5, seed=1)
    assert importances["f1"] > importances["f2"]


def test_guard_phase3d_output_paths():
    import scripts.phase3d_noise_continuous as p3d_noise

    assert p3d_shap.OUTPUT_DIR.as_posix().endswith("results/phase3d")
    assert p3d_shap.SHAP_DIR.as_posix().endswith("results/phase3d/shap")
    assert p3d_noise.OUTPUT_DIR.as_posix().endswith("results/phase3d/noise_continuous")

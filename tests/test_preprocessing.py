"""Tests for src.preprocessing and src.utils: splits, subsampling, and leakage."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset
from src.preprocessing import build_preprocessor
from src.utils import make_splits, set_seed, subsample_train

DATASET_NAMES = [
    "pima_diabetes",
    "heart_cleveland",
    "ckd",
    "ilpd",
    "breast_wisconsin",
]


# ---------------------------------------------------------------------------
# make_splits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_make_splits_structure(name):
    X, y, _ = load_dataset(name)
    splits = make_splits(X, y, n_splits=5, n_repeats=2, seed=42)

    assert len(splits) == 10

    for train_idx, test_idx in splits:
        # No overlap between train and test.
        assert set(train_idx).isdisjoint(set(test_idx))

        # Every test set contains both classes.
        y_test = y.iloc[test_idx]
        assert set(y_test.unique().tolist()) == {0, 1}


# ---------------------------------------------------------------------------
# subsample_train
# ---------------------------------------------------------------------------


def test_subsample_train_min_per_class_rule_triggers():
    """Synthetic imbalanced case where fraction=0.1 alone would violate the
    minimum-2-per-class rule, to directly exercise the guarantee."""
    # 18 majority (class 0), 2 minority (class 1) -> 0.1 * 2 = 0.2, would
    # round down to 0 without the floor of 2.
    y = pd.Series([0] * 18 + [1] * 2)
    train_idx = np.arange(len(y))

    sub_idx = subsample_train(train_idx, y, fraction=0.1, seed=42)
    sub_y = y.iloc[sub_idx]

    assert (sub_y == 1).sum() >= 2
    assert (sub_y == 0).sum() >= 2


@pytest.mark.parametrize("fraction", [0.1, 0.25, 0.5, 1.0])
def test_subsample_train_balance_on_smallest_dataset(fraction):
    """heart_cleveland (303 rows) is the smallest of the five datasets."""
    X, y, meta = load_dataset("heart_cleveland")
    assert meta["n_rows"] == min(
        load_dataset(n)[2]["n_rows"] for n in DATASET_NAMES
    )

    set_seed(42)
    train_idx, _ = make_splits(X, y, n_splits=5, n_repeats=1, seed=42)[0]

    sub_idx = subsample_train(train_idx, y, fraction=fraction, seed=42)

    full_balance = y.iloc[train_idx].mean()
    sub_balance = y.iloc[sub_idx].mean()

    # Class balance should be roughly preserved (loose tolerance since small
    # fractions on a small dataset round coarsely).
    assert abs(full_balance - sub_balance) < 0.15

    # Minimum-per-class guarantee.
    sub_y = y.iloc[sub_idx]
    assert (sub_y == 0).sum() >= 2
    assert (sub_y == 1).sum() >= 2

    # Subsample must be a strict subset of the training fold.
    assert set(sub_idx).issubset(set(train_idx))


# ---------------------------------------------------------------------------
# Leakage test for build_preprocessor
# ---------------------------------------------------------------------------


def test_preprocessor_statistics_come_from_training_fold_only():
    """Fit on a train fold only; verify imputation/scaling statistics match a
    manual computation from the train fold, and are NOT influenced by the
    test fold's (very different) values."""
    rng = np.random.RandomState(0)

    n_train, n_test = 100, 20
    train_numeric = rng.normal(loc=10.0, scale=2.0, size=n_train)
    # Test fold has a wildly different distribution -- if the preprocessor
    # leaked, the imputer/scaler statistics would shift toward this.
    test_numeric = rng.normal(loc=1000.0, scale=50.0, size=n_test)

    # Inject missing values into both folds.
    train_numeric_with_nan = train_numeric.copy()
    train_numeric_with_nan[:5] = np.nan

    train_cat = rng.choice(["a", "b"], size=n_train)
    test_cat = rng.choice(["a", "b", "c"], size=n_test)  # "c" unseen in training

    X_train = pd.DataFrame({"num": train_numeric_with_nan, "cat": train_cat})
    X_test = pd.DataFrame({"num": test_numeric, "cat": test_cat})

    preprocessor = build_preprocessor(
        numeric_cols=["num"], categorical_cols=["cat"], scale=True
    )

    # Fit ONLY on the training fold.
    preprocessor.fit(X_train)

    # Manual ground truth computed from the training fold only.
    expected_median = np.nanmedian(train_numeric_with_nan)
    train_imputed = np.where(
        np.isnan(train_numeric_with_nan), expected_median, train_numeric_with_nan
    )
    expected_mean = train_imputed.mean()
    expected_std = train_imputed.std()

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    fitted_median = numeric_pipeline.named_steps["imputer"].statistics_[0]
    fitted_mean = numeric_pipeline.named_steps["scaler"].mean_[0]
    fitted_std = numeric_pipeline.named_steps["scaler"].scale_[0]

    assert np.isclose(fitted_median, expected_median)
    assert np.isclose(fitted_mean, expected_mean)
    assert np.isclose(fitted_std, expected_std)

    # Transforming the test fold must not raise (unknown category "c" is
    # ignored) and must not change the fitted statistics.
    transformed_test = preprocessor.transform(X_test)
    assert transformed_test.shape[0] == n_test

    fitted_median_after = numeric_pipeline.named_steps["imputer"].statistics_[0]
    fitted_mean_after = numeric_pipeline.named_steps["scaler"].mean_[0]
    assert fitted_median_after == fitted_median
    assert fitted_mean_after == fitted_mean


# ---------------------------------------------------------------------------
# Phase 1.1: Cleveland routing through build_preprocessor
# ---------------------------------------------------------------------------


def test_cleveland_preprocessor_routes_categoricals_and_outputs_no_nan():
    X, _, meta = load_dataset("heart_cleveland")

    preprocessor = build_preprocessor(
        numeric_cols=meta["numeric_cols"],
        categorical_cols=meta["categorical_cols"],
        scale=True,
    )
    output = preprocessor.fit_transform(X)
    if hasattr(output, "toarray"):
        output = output.toarray()

    assert not np.isnan(output).any()

    # Each categorical column must be one-hot encoded on its observed codes.
    encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    categories = dict(zip(meta["categorical_cols"], encoder.categories_))
    assert set(categories["thal"]) == {"3", "6", "7"}
    assert set(categories["cp"]) == {"1", "2", "3", "4"}

    n_expected = len(meta["numeric_cols"]) + sum(len(c) for c in encoder.categories_)
    assert output.shape == (meta["n_rows"], n_expected)


# ---------------------------------------------------------------------------
# Phase 1.2: the preprocessor must work on every real dataset
# ---------------------------------------------------------------------------


def _assert_clean_numeric_output(output, n_rows: int) -> np.ndarray:
    """Check a transformed fold is a finite, numeric 2D array with `n_rows` rows."""
    array = output.toarray() if hasattr(output, "toarray") else np.asarray(output)
    assert array.ndim == 2
    assert np.issubdtype(array.dtype, np.number)
    assert np.isfinite(array).all(), "output contains NaN or inf"
    assert array.shape[0] == n_rows
    return array


@pytest.mark.parametrize("scale", [True, False])
@pytest.mark.parametrize("name", DATASET_NAMES)
def test_preprocessor_fits_on_all_datasets(name, scale):
    X, y, meta = load_dataset(name)
    train_idx, test_idx = make_splits(X, y, n_splits=5, n_repeats=1, seed=42)[0]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

    preprocessor = build_preprocessor(meta["numeric_cols"], meta["categorical_cols"], scale=scale)
    preprocessor.fit(X_train)

    train_output = _assert_clean_numeric_output(preprocessor.transform(X_train), len(X_train))
    test_output = _assert_clean_numeric_output(preprocessor.transform(X_test), len(X_test))
    assert train_output.shape[1] == test_output.shape[1]

    print(
        f"SHAPE {name} scale={scale}: "
        f"train_input={X_train.shape} train_output={train_output.shape} "
        f"test_input={X_test.shape} test_output={test_output.shape}"
    )


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_preprocessor_fits_on_small_training_fraction(name):
    """A tiny training subset can miss rare categories that the test fold contains."""
    X, y, meta = load_dataset(name)
    train_idx, test_idx = make_splits(X, y, n_splits=5, n_repeats=1, seed=42)[0]
    sub_idx = subsample_train(train_idx, y, fraction=0.1, seed=42)

    preprocessor = build_preprocessor(meta["numeric_cols"], meta["categorical_cols"], scale=True)
    preprocessor.fit(X.iloc[sub_idx])

    _assert_clean_numeric_output(preprocessor.transform(X.iloc[test_idx]), len(test_idx))

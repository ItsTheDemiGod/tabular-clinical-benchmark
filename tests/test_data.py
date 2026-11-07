"""Tests for src.data: dataset loading correctness."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.data
from src.data import _deduplicate, _normalize_categoricals, load_dataset

DATASET_NAMES = [
    "pima_diabetes",
    "heart_cleveland",
    "ckd",
    "ilpd",
    "breast_wisconsin",
]


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_dataset_loads_and_is_well_formed(name):
    X, y, meta = load_dataset(name)

    # Non-empty, aligned X/y.
    assert len(X) == len(y)
    assert len(X) > 0

    # y is exactly {0, 1} and both classes present.
    unique_y = set(y.unique().tolist())
    assert unique_y == {0, 1}, f"{name}: expected y in {{0,1}}, got {unique_y}"

    # No duplicate column names.
    assert not X.columns.duplicated().any(), f"{name}: duplicate column names found"

    # Required meta keys.
    for key in (
        "name",
        "n_rows",
        "n_features",
        "class_balance",
        "numeric_cols",
        "categorical_cols",
        "missing_fraction",
        "source",
    ):
        assert key in meta, f"{name}: missing meta key '{key}'"

    assert meta["n_rows"] == len(X)
    assert meta["n_features"] == X.shape[1]
    assert 0.0 < meta["class_balance"] < 1.0


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_dataset("not_a_real_dataset")


# ---------------------------------------------------------------------------
# Phase 1.1: Pima hidden missing values
# ---------------------------------------------------------------------------

PIMA_ZERO_COLUMNS = ["plas", "pres", "skin", "insu", "mass"]


def test_pima_zeros_replaced_with_nan():
    X, _, meta = load_dataset("pima_diabetes")

    for col in PIMA_ZERO_COLUMNS:
        assert (X[col] == 0).sum() == 0, f"{col} still contains zeros"
        # Every replaced zero must show up as missing (no rows were deduplicated).
        assert X[col].isna().sum() == meta["zeros_replaced_with_nan"][col]
        # meta's missing fractions are computed after the replacement.
        assert meta["missing_fraction"][col] == X[col].isna().mean()

    assert X.isna().to_numpy().mean() > 0


def test_pima_real_zero_values_untouched():
    """0 pregnancies is a real value, so preg keeps its zeros and has no NaN."""
    X, _, meta = load_dataset("pima_diabetes")
    assert (X["preg"] == 0).sum() > 0
    assert X[["preg", "pedi", "age"]].isna().sum().sum() == 0
    assert set(meta["zeros_replaced_with_nan"]) == set(PIMA_ZERO_COLUMNS)


# ---------------------------------------------------------------------------
# Phase 1.1: Cleveland categorical columns
# ---------------------------------------------------------------------------

CLEVELAND_CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
CLEVELAND_NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]


def test_cleveland_column_split():
    _, _, meta = load_dataset("heart_cleveland")
    assert set(meta["categorical_cols"]) == set(CLEVELAND_CATEGORICAL)
    assert set(meta["numeric_cols"]) == set(CLEVELAND_NUMERIC)


def test_cleveland_categoricals_are_clean_strings():
    X, _, _ = load_dataset("heart_cleveland")
    for col in CLEVELAND_CATEGORICAL:
        present = X[col].dropna()
        assert (present != "nan").all(), f"{col} contains the string 'nan'"
        assert present.map(lambda v: isinstance(v, str)).all(), f"{col} has non-strings"
        # One consistent format: "3", not a mix of "3" and "3.0".
        assert present.str.fullmatch(r"\d+").all(), f"{col} has inconsistent codes"


def test_cleveland_missing_counts_preserved():
    """The raw data has 2 missing thal and 4 missing ca values."""
    X, _, _ = load_dataset("heart_cleveland")
    assert X["thal"].isna().sum() == 2
    assert X["ca"].isna().sum() == 4


# ---------------------------------------------------------------------------
# Phase 1.1: duplicate rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_no_exact_duplicates_and_dedup_meta(name):
    X, y, meta = load_dataset(name)

    assert X.assign(__target__=y.values).duplicated().sum() == 0

    for key in (
        "n_duplicates_found",
        "n_rows_before_dedup",
        "n_rows_after_dedup",
        "n_conflicting_label_rows",
    ):
        assert key in meta, f"{name}: missing meta key '{key}'"

    assert meta["n_rows"] == meta["n_rows_after_dedup"] == len(X)
    assert meta["n_rows_before_dedup"] - meta["n_duplicates_found"] == meta["n_rows"]
    # X and y must keep a clean, shared index after rows are dropped.
    assert list(X.index) == list(range(len(X)))
    assert list(y.index) == list(range(len(y)))


def test_deduplicate_drops_exact_copies_and_keeps_conflicts():
    """Synthetic case, since the real datasets may not contain every situation."""
    X = pd.DataFrame(
        {
            "a": [1.0, 1.0, 2.0, 2.0, np.nan, np.nan],
            "b": ["x", "x", "y", "y", "z", "z"],
        },
        # A non-default index, to confirm the result is reset and stays aligned.
        index=[10, 11, 12, 13, 14, 15],
    )
    y = pd.Series([0, 0, 0, 1, 1, 1], index=X.index)

    X_out, y_out, stats = _deduplicate(X, y)

    # Rows 0/1 are exact copies, and so are 4/5 (matching NaNs count as equal).
    # Rows 2/3 have the same features but different labels, so both stay.
    assert stats == {
        "n_duplicates_found": 2,
        "n_rows_before_dedup": 6,
        "n_rows_after_dedup": 4,
        "n_conflicting_label_rows": 2,
    }
    assert y_out.tolist() == [0, 0, 1, 1]
    assert X_out["a"].tolist()[:3] == [1.0, 2.0, 2.0]
    assert list(X_out.index) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Phase 1.2: categorical storage and CKD ordinal columns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_categoricals_are_object_dtype_without_pd_na(name):
    X, _, meta = load_dataset(name)

    for col in meta["categorical_cols"]:
        assert X[col].dtype == object, f"{name}.{col} has dtype {X[col].dtype}"
        assert not any(value is pd.NA for value in X[col]), f"{name}.{col} contains pd.NA"
        for placeholder in ("nan", "None", "<NA>", "?"):
            assert (X[col] != placeholder).all(), f"{name}.{col} contains '{placeholder}'"


def test_ckd_ordinal_columns_numeric():
    _, _, meta = load_dataset("ckd")

    for col in ("sg", "al", "su"):
        assert col in meta["numeric_cols"]
        assert col not in meta["categorical_cols"]
    assert meta["ordinal_as_numeric"] == ["sg", "al", "su"]


def test_normalize_categoricals_converts_missing_markers_and_keeps_counts():
    """Synthetic case covering pd.NA, None and NaN, leaving numeric columns alone."""
    X = pd.DataFrame(
        {
            "string_col": pd.Series(["a", pd.NA, "b"], dtype="string"),
            "object_col": pd.Series(["x", None, np.nan], dtype=object),
            "numeric_col": [1.0, np.nan, 3.0],
        }
    )

    out = _normalize_categoricals(X)

    assert out["string_col"].dtype == object
    assert out["object_col"].dtype == object
    assert out["numeric_col"].dtype == np.float64
    assert out.isna().sum().to_dict() == X.isna().sum().to_dict()
    # Every missing categorical value is now exactly the float NaN.
    for col in ("string_col", "object_col"):
        for value in out[col][out[col].isna()]:
            assert isinstance(value, float) and np.isnan(value)


# ---------------------------------------------------------------------------
# Phase 2a.1: quiet logging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_verbose_false_prints_nothing(name, capsys):
    capsys.readouterr()
    load_dataset(name, verbose=False)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("name", DATASET_NAMES)
def test_default_still_prints(name, capsys):
    capsys.readouterr()
    load_dataset(name)
    assert "duplicates found and dropped" in capsys.readouterr().out


def test_conflicting_label_warning_prints_even_when_quiet(monkeypatch, capsys):
    """No real dataset has conflicting labels, so the deduplication stats are faked."""
    original = src.data._deduplicate

    def with_conflicts(X, y):
        X_out, y_out, stats = original(X, y)
        return X_out, y_out, {**stats, "n_conflicting_label_rows": 2}

    monkeypatch.setattr(src.data, "_deduplicate", with_conflicts)
    capsys.readouterr()
    load_dataset("ilpd", verbose=False)
    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "conflicting labels" in output
    assert "duplicates found" not in output


def test_normalize_categoricals_rejects_placeholder_strings():
    """Placeholders must raise, not silently change the dataset's missing counts."""
    X = pd.DataFrame({"col": pd.Series(["a", "?"], dtype=object)})
    with pytest.raises(RuntimeError, match="placeholder"):
        _normalize_categoricals(X)

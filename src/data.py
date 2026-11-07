"""Dataset loading for the clinical tabular benchmark.

Five small clinical binary-classification datasets are loaded through a
single `load_dataset(name)` entry point. Each raw download is cached to
`data/` (as a pickle of the untouched source objects) so repeated runs,
including the test suite, do not hit the network every time. Target
binarization and all cleaning happen on every call, after the cache is read,
so changing that logic does not require re-downloading and never alters the
cached files.

Verified dataset identities (checked against the live sources, not assumed).
Row counts here are RAW counts; `meta["n_rows"]` is the count after duplicates
are dropped:
  - "pima_diabetes": OpenML dataset name="diabetes", version=1 (OpenML id 37),
    768 rows x 8 features, target "class" in {tested_positive, tested_negative}.
  - "heart_cleveland": UCI "Heart Disease", ucimlrepo id=45, 303 rows x 13
    features, target "num" in {0,1,2,3,4} (0 = no disease).
  - "ckd": UCI "Chronic Kidney Disease", ucimlrepo id=336, 400 rows x 24
    features, target "class" in {"ckd", "ckd\t", "notckd"} (raw strings need
    stripping).
  - "ilpd": UCI "ILPD (Indian Liver Patient Dataset)", ucimlrepo id=225,
    583 rows x 10 features, target "Selector" in {1, 2} (1 = liver patient).
  - "breast_wisconsin": UCI "Breast Cancer Wisconsin (Diagnostic)",
    ucimlrepo id=17, 569 rows x 30 features, target "Diagnosis" in {M, B}.

Cleaning applied on load (Phase 1.1):
  - Pima: zeros that mean "not recorded" become NaN (see PIMA_ZERO_AS_MISSING).
  - Cleveland: integer-coded categoricals become strings, keeping NaN as NaN.
  - All datasets: exact duplicate rows are dropped; rows with identical
    features but conflicting labels are counted and kept.

Cleaning applied on load (Phase 1.2):
  - All datasets: categorical columns are stored as plain object dtype with
    np.nan for missing values, which is the form SimpleImputer can handle.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Expected source schemas. A mismatch raises instead of silently cleaning the
# wrong columns.
PIMA_EXPECTED_COLUMNS: List[str] = [
    "preg", "plas", "pres", "skin", "insu", "mass", "pedi", "age", "class",
]
CLEVELAND_EXPECTED_COLUMNS: List[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# Pima columns where 0 is physiologically impossible and means "not recorded".
# preg is excluded (0 pregnancies is real), as are pedi and age.
PIMA_ZERO_AS_MISSING: List[str] = ["plas", "pres", "skin", "insu", "mass"]

# Cleveland columns that are integer-coded categories, not quantities. ca (a
# count of major vessels) stays numeric.
CLEVELAND_CATEGORICAL: List[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
CLEVELAND_NUMERIC: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

# CKD columns kept numeric because they are ordered scales and one-hot encoding would discard the order.
CKD_ORDINAL_AS_NUMERIC: List[str] = ["sg", "al", "su"]

# Strings that must never stand in for a missing value in a categorical column.
FORBIDDEN_CATEGORY_STRINGS: FrozenSet[str] = frozenset({"nan", "None", "<NA>", "?"})

# What a private loader hands back to load_dataset: cleaned X, binary y, the
# source description, and any dataset-specific meta entries.
LoaderResult = Tuple[pd.DataFrame, pd.Series, str, Dict[str, Any]]


def _cache_path(name: str) -> Path:
    """Return the raw-download cache path for `name`, ensuring data/ exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / f"{name}_raw.pkl"


def _load_or_fetch_raw(name: str, fetch_fn: Callable[[], Tuple[pd.DataFrame, pd.DataFrame]]):
    """Return a cached raw (X, y_raw) pair for `name`, fetching+caching if absent."""
    path = _cache_path(name)
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    raw = fetch_fn()
    with open(path, "wb") as f:
        pickle.dump(raw, f)
    return raw


def _infer_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Split columns into (numeric_cols, categorical_cols) by pandas dtype."""
    categorical_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    return numeric_cols, categorical_cols


def _build_meta(name: str, X: pd.DataFrame, y: pd.Series, source: str) -> Dict[str, Any]:
    """Assemble the metadata dict required for every loaded dataset."""
    numeric_cols, categorical_cols = _infer_column_types(X)
    return {
        "name": name,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_balance": float(y.mean()),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "missing_fraction": X.isna().mean().to_dict(),
        "source": source,
    }


def _check_columns(name: str, actual: List[str], expected: List[str]) -> None:
    """Raise if the columns of `name` differ from the schema the cleaning assumes."""
    if actual != expected:
        raise RuntimeError(
            f"{name}: column mismatch.\n  expected: {expected}\n  actual:   {actual}"
        )


def _replace_zeros_with_nan(
    X: pd.DataFrame, columns: List[str]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Return a copy of `X` with 0 replaced by NaN in `columns`, plus per-column counts."""
    X = X.copy()
    counts: Dict[str, int] = {}
    for col in columns:
        is_zero = X[col] == 0
        counts[col] = int(is_zero.sum())
        X[col] = X[col].mask(is_zero, np.nan)
    return X, counts


def _integer_codes_to_strings(column: pd.Series) -> pd.Series:
    """Convert integer-coded values to strings ("1", never "1.0"), keeping NaN as NaN.

    Going through int makes the format the same whether pandas stored the column
    as int64 or, because it has missing values, as float64 (like thal).
    """
    values: List[Any] = []
    for value in column:
        if pd.isna(value):
            values.append(np.nan)
        elif float(value).is_integer():
            values.append(str(int(value)))
        else:
            raise RuntimeError(
                f"Column '{column.name}': expected integer codes, found {value!r}"
            )
    return pd.Series(values, index=column.index, name=column.name, dtype=object)


def _normalize_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of `X` whose categorical columns are object dtype with np.nan.

    SimpleImputer crashes on the pd.NA marker used by pandas "string" dtype
    ("boolean value of NA is ambiguous"), so every missing marker becomes np.nan.
    Only non-numeric columns are touched, and each missing value stays missing.

    Raises:
        RuntimeError: If a column contains a string from
            FORBIDDEN_CATEGORY_STRINGS. Treating it as missing would change the
            dataset's missing counts, so that decision is left to a person.
    """
    X = X.copy()
    _, categorical_cols = _infer_column_types(X)
    for col in categorical_cols:
        forbidden = sorted(set(X[col].dropna()) & FORBIDDEN_CATEGORY_STRINGS)
        if forbidden:
            raise RuntimeError(f"Column '{col}' contains placeholder strings: {forbidden}")
        values = [np.nan if pd.isna(value) else value for value in X[col]]
        X[col] = pd.Series(values, index=X.index, name=col, dtype=object)
    return X


def _deduplicate(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    """Drop exact duplicate rows (features + target), keeping the first occurrence.

    pandas treats NaNs in the same position as equal here, so two rows that are
    missing the same values and agree everywhere else count as duplicates.

    Rows with identical features but different labels are kept and counted. Exact
    duplicates are already gone at that point, so any repeated feature vector left
    must have conflicting labels, and the count refers to rows still in the data.

    Returns:
        The deduplicated `X` and `y` with reset indexes, and a dict with
        "n_duplicates_found", "n_rows_before_dedup", "n_rows_after_dedup" and
        "n_conflicting_label_rows".
    """
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    is_duplicate = X.assign(__target__=y).duplicated(keep="first")
    X_deduped = X.loc[~is_duplicate].reset_index(drop=True)
    y_deduped = y.loc[~is_duplicate].reset_index(drop=True)

    stats = {
        "n_duplicates_found": int(is_duplicate.sum()),
        "n_rows_before_dedup": int(len(X)),
        "n_rows_after_dedup": int(len(X_deduped)),
        "n_conflicting_label_rows": int(X_deduped.duplicated(keep=False).sum()),
    }
    return X_deduped, y_deduped, stats


def _load_pima_diabetes(verbose: bool = True) -> LoaderResult:
    import openml

    def fetch():
        ds = openml.datasets.get_dataset("diabetes", version=1)
        if ds.name != "diabetes" or ds.version != 1:
            raise RuntimeError(
                f"OpenML dataset mismatch: expected name='diabetes' version=1, "
                f"got name='{ds.name}' version={ds.version}"
            )
        X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        return X, y

    X, y_raw = _load_or_fetch_raw("pima_diabetes", fetch)

    # OpenML returns the target separately, so check it along with the features.
    actual_columns = list(X.columns) + [y_raw.name]
    if verbose:
        print(f"[pima_diabetes] columns from OpenML: {actual_columns}")
    _check_columns("pima_diabetes", actual_columns, PIMA_EXPECTED_COLUMNS)

    X, zeros_replaced = _replace_zeros_with_nan(X, PIMA_ZERO_AS_MISSING)

    y = (y_raw == "tested_positive").astype(int)
    y.name = "target"
    extra_meta = {"zeros_replaced_with_nan": zeros_replaced}
    return X, y, "OpenML diabetes v1 (id 37)", extra_meta


def _load_heart_cleveland(verbose: bool = True) -> LoaderResult:
    from ucimlrepo import fetch_ucirepo

    def fetch():
        d = fetch_ucirepo(id=45)
        if d.metadata.name != "Heart Disease":
            raise RuntimeError(
                f"UCI id=45 mismatch: expected 'Heart Disease', got '{d.metadata.name}'"
            )
        return d.data.features.copy(), d.data.targets.copy()

    X, y_raw = _load_or_fetch_raw("heart_cleveland", fetch)

    if verbose:
        print(f"[heart_cleveland] columns and dtypes from UCI: {X.dtypes.astype(str).to_dict()}")
    _check_columns("heart_cleveland", list(X.columns), CLEVELAND_EXPECTED_COLUMNS)

    # Object dtype is what routes these to the one-hot branch: _infer_column_types
    # reports them as categorical, and meta's lists feed build_preprocessor.
    X = X.copy()
    for col in CLEVELAND_CATEGORICAL:
        X[col] = _integer_codes_to_strings(X[col])

    y = (y_raw["num"].astype(int) > 0).astype(int)
    y.name = "target"
    return X, y, "UCI Heart Disease, Cleveland subset (id 45)", {}


def _load_ckd(verbose: bool = True) -> LoaderResult:
    from ucimlrepo import fetch_ucirepo

    def fetch():
        d = fetch_ucirepo(id=336)
        if d.metadata.name != "Chronic Kidney Disease":
            raise RuntimeError(
                f"UCI id=336 mismatch: expected 'Chronic Kidney Disease', got '{d.metadata.name}'"
            )
        return d.data.features.copy(), d.data.targets.copy()

    X, y_raw = _load_or_fetch_raw("ckd", fetch)
    X = X.copy()

    # Raw CKD strings carry inconsistent whitespace (e.g. "ckd\t", "\tno") that
    # would otherwise create spurious extra categories after one-hot encoding.
    numeric_cols, categorical_cols = _infer_column_types(X)
    for col in categorical_cols:
        X[col] = X[col].astype("string").str.strip()

    cls = y_raw["class"].astype("string").str.strip()
    y = (cls == "ckd").astype(int)
    y.name = "target"
    extra_meta = {"ordinal_as_numeric": list(CKD_ORDINAL_AS_NUMERIC)}
    return X, y, "UCI Chronic Kidney Disease (id 336)", extra_meta


def _load_ilpd(verbose: bool = True) -> LoaderResult:
    from ucimlrepo import fetch_ucirepo

    def fetch():
        d = fetch_ucirepo(id=225)
        if d.metadata.name != "ILPD (Indian Liver Patient Dataset)":
            raise RuntimeError(
                f"UCI id=225 mismatch: expected 'ILPD (Indian Liver Patient Dataset)', "
                f"got '{d.metadata.name}'"
            )
        return d.data.features.copy(), d.data.targets.copy()

    X, y_raw = _load_or_fetch_raw("ilpd", fetch)
    y = (y_raw["Selector"].astype(int) == 1).astype(int)
    y.name = "target"
    return X, y, "UCI ILPD Indian Liver Patient Dataset (id 225)", {}


def _load_breast_wisconsin(verbose: bool = True) -> LoaderResult:
    from ucimlrepo import fetch_ucirepo

    def fetch():
        d = fetch_ucirepo(id=17)
        if d.metadata.name != "Breast Cancer Wisconsin (Diagnostic)":
            raise RuntimeError(
                f"UCI id=17 mismatch: expected 'Breast Cancer Wisconsin (Diagnostic)', "
                f"got '{d.metadata.name}'"
            )
        return d.data.features.copy(), d.data.targets.copy()

    X, y_raw = _load_or_fetch_raw("breast_wisconsin", fetch)
    y = (y_raw["Diagnosis"] == "M").astype(int)
    y.name = "target"
    return X, y, "UCI Breast Cancer Wisconsin Diagnostic (id 17)", {}


_LOADERS: Dict[str, Callable[[bool], LoaderResult]] = {
    "pima_diabetes": _load_pima_diabetes,
    "heart_cleveland": _load_heart_cleveland,
    "ckd": _load_ckd,
    "ilpd": _load_ilpd,
    "breast_wisconsin": _load_breast_wisconsin,
}


def load_dataset(
    name: str, verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Load one of the five clinical benchmark datasets.

    Dataset-specific cleaning runs first, then deduplication, then `meta` is
    built, so every value in `meta` describes the data actually returned.

    Args:
        name: One of "pima_diabetes", "heart_cleveland", "ckd", "ilpd",
            "breast_wisconsin".
        verbose: Whether to print progress lines. The warning about conflicting
            labels is printed either way.

    Returns:
        A tuple `(X, y, meta)`:
          - `X`: feature DataFrame with a reset index.
          - `y`: Series of 0/1 ints (1 = positive / disease-present class).
          - `meta`: dict with keys "name", "n_rows" (after deduplication),
            "n_features", "class_balance" (positive fraction),
            "numeric_cols", "categorical_cols", "missing_fraction"
            (per-column dict), "source", "n_duplicates_found",
            "n_rows_before_dedup", "n_rows_after_dedup" and
            "n_conflicting_label_rows". For "pima_diabetes" it also has
            "zeros_replaced_with_nan" (column -> number of zeros replaced).
            For "ckd" it also has "ordinal_as_numeric" (ordered-scale columns
            deliberately kept numeric).

    Raises:
        ValueError: If `name` is not one of the five supported datasets.
        RuntimeError: If the fetched dataset's identity does not match what
            was expected (wrong OpenML name/version or wrong UCI dataset
            name for the given id), if Pima or Cleveland columns differ from
            the expected schema, or if a categorical column contains a
            placeholder string such as "?".
    """
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Supported: {sorted(_LOADERS)}")

    X, y, source, extra_meta = _LOADERS[name](verbose)
    X = _normalize_categoricals(X)
    X, y, dedup_stats = _deduplicate(X, y)

    if verbose:
        print(
            f"[{name}] duplicates found and dropped: {dedup_stats['n_duplicates_found']} "
            f"({dedup_stats['n_rows_before_dedup']} -> {dedup_stats['n_rows_after_dedup']} rows)"
        )
    if dedup_stats["n_conflicting_label_rows"] > 0:
        print(
            f"WARNING [{name}]: {dedup_stats['n_conflicting_label_rows']} rows have "
            f"identical features but conflicting labels (kept, not dropped)."
        )

    meta = _build_meta(name, X, y, source)
    meta.update(dedup_stats)
    meta.update(extra_meta)
    return X, y, meta

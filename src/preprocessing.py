"""Leakage-safe preprocessing layer.

The single rule this module exists to enforce: every statistic used to
transform data (an imputation median/mode, a scaler mean/std, a one-hot
encoder's category vocabulary) must be learned ONLY from the training fold.
We achieve this not by being careful every time we preprocess, but by
returning an *unfitted* `ColumnTransformer` that the caller places inside a
`sklearn.pipeline.Pipeline` alongside the model. Because a `Pipeline` calls
`.fit()` on each step using only the data it is given, and cross-validation
code calls `.fit()` on the pipeline with only the training fold, the
preprocessor can never see the test fold's values during fitting -- leakage
is structurally impossible as long as callers fit the whole pipeline at
once, rather than fitting the preprocessor separately on the full dataset.
"""

from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    scale: bool,
) -> ColumnTransformer:
    """Build an unfitted ColumnTransformer for leakage-safe preprocessing.

    Numeric columns: median imputation, optionally followed by
    `StandardScaler`. Categorical columns: most-frequent imputation
    followed by one-hot encoding with `handle_unknown="ignore"` (so a
    category seen only in the test fold does not raise an error -- it is
    simply encoded as all-zeros).

    This function must NEVER be called with `.fit()` or `.fit_transform()`
    on anything other than a training fold. The returned object is meant to
    be placed as a step inside a `sklearn.pipeline.Pipeline` together with
    the estimator, so that cross-validation fits the preprocessor and the
    model together on each training fold only.

    Args:
        numeric_cols: Names of numeric feature columns.
        categorical_cols: Names of categorical feature columns.
        scale: Whether to apply `StandardScaler` after imputing numeric
            columns.

    Returns:
        An unfitted `ColumnTransformer` ready to be used inside a
        `Pipeline`.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_cols:
        transformers.append(("numeric", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("categorical", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers=transformers)

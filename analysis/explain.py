"""Phase 3d explanation helpers: encoded-to-original-feature mapping (via the fitted
ColumnTransformer's output_indices_ and OneHotEncoder.categories_, never by parsing
column-name strings), signed-sum SHAP aggregation, top-k selection with tie-breaking
by feature order, Jaccard/Spearman agreement, the exact chance-level Jaccard, an
additivity check, and a permutation-importance fallback.

Nothing here fits, refits or tunes a model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import spearmanr

from src.metrics import compute_metrics

__all__ = [
    "map_encoded_to_original",
    "aggregate_signed_shap",
    "feature_importance_from_agg",
    "top_k_with_ties",
    "jaccard",
    "expected_chance_jaccard",
    "spearman_safe",
    "normalize_shap_values",
    "additivity_errors",
    "permutation_importance_logloss",
    "sample_background_rows",
]


def map_encoded_to_original(
    fitted_preprocessor, numeric_cols: list[str], categorical_cols: list[str]
) -> dict[str, list[int]]:
    """Original feature name -> list of its encoded-column indices, read ONLY from
    the fitted ColumnTransformer's `output_indices_` and the one-hot encoder's
    `categories_` -- never by parsing encoded column-name strings.
    """
    mapping: dict[str, list[int]] = {}
    indices = fitted_preprocessor.output_indices_

    if numeric_cols:
        num_slice = indices["numeric"]
        for i, col in enumerate(numeric_cols):
            mapping[col] = [num_slice.start + i]

    if categorical_cols:
        cat_slice = indices["categorical"]
        onehot = fitted_preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        offset = cat_slice.start
        for col, categories in zip(categorical_cols, onehot.categories_):
            n = len(categories)
            mapping[col] = list(range(offset, offset + n))
            offset += n

    return mapping


def aggregate_signed_shap(shap_matrix: np.ndarray, mapping: dict[str, list[int]], feature_order: list[str]) -> np.ndarray:
    """Sum the SIGNED SHAP values over each feature's encoded columns, per row,
    BEFORE any absolute value is taken. Returns an (n_rows, n_original_features) array
    in `feature_order`.
    """
    n_rows = shap_matrix.shape[0]
    agg = np.zeros((n_rows, len(feature_order)), dtype=np.float64)
    for j, feature in enumerate(feature_order):
        cols = mapping[feature]
        agg[:, j] = shap_matrix[:, cols].sum(axis=1)
    return agg


def feature_importance_from_agg(agg: np.ndarray) -> np.ndarray:
    """Importance of each original feature = mean over rows of the absolute
    aggregated (signed-summed) SHAP value.
    """
    return np.abs(agg).mean(axis=0)


def top_k_with_ties(importance: np.ndarray, feature_order: list[str], k: int = 5) -> list[str]:
    """Top-k feature names by importance (descending); ties broken by feature order
    (earlier in `feature_order` wins), via a stable sort on the negated importance.
    """
    order = np.argsort(-importance, kind="stable")
    return [feature_order[i] for i in order[:k]]


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def expected_chance_jaccard(m: int, k: int = 5) -> float:
    """Exact expected Jaccard of two INDEPENDENT random k-subsets of an m-element
    set. P(|A intersect B| = i) is hypergeometric (population m, k "successes" from
    the fixed set B, k draws for A): C(k,i)*C(m-k,k-i)/C(m,k). Jaccard given
    intersection i is i / (2k - i) (undefined at i=0=k=0, not reached for k>=1).
    """
    if k == 0:
        return 1.0
    total = 0.0
    denom_choose_m_k = comb(m, k, exact=True)
    for i in range(0, k + 1):
        if k - i > m - k:
            continue  # C(m-k, k-i) would be 0
        p = comb(k, i, exact=True) * comb(m - k, k - i, exact=True) / denom_choose_m_k
        union = 2 * k - i
        total += p * (i / union) if union > 0 else 0.0
    return total


def spearman_safe(a: np.ndarray, b: np.ndarray) -> tuple[float, bool]:
    """Spearman rank correlation; returns (rho, is_constant). If either array is
    constant, rho is undefined (NaN) and is_constant is True.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan"), True
    rho, _ = spearmanr(a, b)
    return float(rho), False


def normalize_shap_values(raw, positive_class_index: int = 1) -> np.ndarray:
    """Normalize a shap explainer's return value, in EITHER of shap's two historical
    shapes, to a single (n_samples, n_features) array for the positive class:
      - a list of per-class arrays, each (n_samples, n_features);
      - a single 3D array (n_samples, n_features, n_classes).
    A plain 2D array (n_samples, n_features) is returned unchanged (e.g. a binary
    linear/kernel explainer that already returns one class's values).
    """
    if isinstance(raw, list):
        return np.asarray(raw[positive_class_index])
    arr = np.asarray(raw)
    if arr.ndim == 3:
        return arr[:, :, positive_class_index]
    return arr


def additivity_errors(base_value: float, shap_rows: np.ndarray, model_output: np.ndarray) -> np.ndarray:
    """|base_value + sum(shap_row) - model_output| per row."""
    predicted = base_value + shap_rows.sum(axis=1)
    return np.abs(predicted - np.asarray(model_output))


def sample_background_rows(x_train_encoded: np.ndarray, n: int, seed: int) -> np.ndarray:
    """n rows sampled without replacement from x_train_encoded (TRAINING rows only)."""
    rng = np.random.default_rng(seed)
    n_available = x_train_encoded.shape[0]
    idx = rng.choice(n_available, size=min(n, n_available), replace=False)
    return x_train_encoded[idx]


def permutation_importance_logloss(
    predict_fn,
    x_raw: pd.DataFrame,
    y_true: np.ndarray,
    feature_cols: list[str],
    n_repeats: int,
    seed: int,
) -> dict[str, float]:
    """Fallback importance: mean increase in log loss when one ORIGINAL feature
    column is permuted (n_repeats times, seeded), applied to the raw feature before
    the pipeline's own preprocessing. Never calls .fit.
    """
    baseline_prob = predict_fn(x_raw)
    baseline_logloss = compute_metrics(y_true, baseline_prob)["log_loss"]

    rng = np.random.default_rng(seed)
    importances: dict[str, float] = {}
    for col in feature_cols:
        increases = []
        for _ in range(n_repeats):
            permuted = x_raw.copy()
            permuted[col] = rng.permutation(permuted[col].to_numpy())
            prob = predict_fn(permuted)
            logloss = compute_metrics(y_true, prob)["log_loss"]
            increases.append(logloss - baseline_logloss)
        importances[col] = float(np.mean(increases))
    return importances

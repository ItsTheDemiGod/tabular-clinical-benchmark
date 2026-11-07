"""Phase 3c robustness helpers: deterministic test-time perturbations (missingness,
Gaussian noise) and a no-refit evaluation routine.

Nothing here ever calls `.fit` on a pipeline or model; every function here only
reads an already-fitted pipeline and already-loaded data. Perturbations never
depend on the model being evaluated (same perturbed frame is reused across models
for a given (dataset, split, kind, level, draw) -- a paired design).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.constants import DATASETS_ALL5
from src.metrics import compute_metrics, prediction_diversity
from src.models import positive_class_proba

__all__ = [
    "DATASET_INDEX",
    "KIND_INDEX",
    "MISSING_LEVELS",
    "NOISE_LEVELS",
    "N_DRAWS",
    "make_rng",
    "apply_missing",
    "apply_noise",
    "evaluate_predictions",
]

# Fixed integer indices, for deterministic seeding only (never used for ordering
# in tables/figures, which always use the MODELS / DATASETS_ALL5 lists directly).
DATASET_INDEX: dict[str, int] = {name: i for i, name in enumerate(DATASETS_ALL5)}
KIND_INDEX: dict[str, int] = {"clean": 0, "missing": 1, "noise": 2}

MISSING_LEVELS: list[float] = [0.10, 0.20, 0.30]
NOISE_LEVELS: list[float] = [0.10, 0.25, 0.50]
N_DRAWS: int = 5


def make_rng(dataset: str, split_idx: int, kind: str, level_idx: int, draw: int) -> np.random.Generator:
    """A reproducible Generator keyed only by (dataset, split, kind, level, draw) --
    never by model, so every model sees the identical perturbed test matrix.
    """
    seed_sequence = np.random.SeedSequence(
        [42, DATASET_INDEX[dataset], split_idx, KIND_INDEX[kind], level_idx, draw]
    )
    return np.random.default_rng(seed_sequence)


def apply_missing(X: pd.DataFrame, r: float, rng: np.random.Generator) -> tuple[pd.DataFrame, float]:
    """Set each cell to missing independently with probability r (MCAR). Cells that
    are already missing stay missing. Returns (perturbed_frame, realised_added_fraction),
    where the fraction is newly-missing cells / total cells.
    """
    already_missing = X.isna()
    draw_mask = pd.DataFrame(rng.random(X.shape) < r, index=X.index, columns=X.columns)
    new_missing = draw_mask & ~already_missing
    perturbed = X.mask(new_missing)
    realised_added_fraction = float(new_missing.to_numpy().sum() / X.size)
    return perturbed, realised_added_fraction


def apply_noise(
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    s: float,
    numeric_cols: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add N(0, (s * sigma_j)^2) to every NON-missing value of each numeric column j,
    with sigma_j the population std (ddof=0) of that column on the TRAINING rows.
    Categorical columns and already-missing cells are left untouched. Columns are
    perturbed in the given fixed order, so the same rng always yields the same result.
    """
    perturbed = X.copy()
    n = len(X)
    for col in numeric_cols:
        sigma = float(X_train[col].std(ddof=0))
        if not np.isfinite(sigma):
            sigma = 0.0
        noise = rng.normal(0.0, s * sigma, size=n)
        non_missing = X[col].notna().to_numpy()
        values = perturbed[col].to_numpy(dtype=np.float64, copy=True)
        values[non_missing] = values[non_missing] + noise[non_missing]
        perturbed[col] = values
    return perturbed


def evaluate_predictions(pipeline, X_test: pd.DataFrame, y_true: np.ndarray) -> dict:
    """Predict with the already-fitted `pipeline` (never calls .fit) and compute
    auroc/brier/log_loss/n_unique_probs/constant_prediction. On any exception,
    returns status="failed" with the error text instead of raising.
    """
    try:
        proba = positive_class_proba(pipeline, X_test)
        n_unique, constant = prediction_diversity(proba)
        metrics = compute_metrics(y_true, proba)
        return {
            "status": "ok",
            "error": "",
            "auroc": metrics["auroc"],
            "brier": metrics["brier"],
            "log_loss": metrics["log_loss"],
            "n_unique_probs": n_unique,
            "constant_prediction": constant,
            "prob": proba,
        }
    except Exception as exc:  # noqa: BLE001 - must record, never crash the run
        return {
            "status": "failed",
            "error": str(exc),
            "auroc": float("nan"),
            "brier": float("nan"),
            "log_loss": float("nan"),
            "n_unique_probs": 0,
            "constant_prediction": False,
            "prob": None,
        }

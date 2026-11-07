"""Shared Optuna tuning. Every model gets the same budget and the same objective."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.models import MODEL_REGISTRY, build_pipeline, positive_class_proba


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_cols: List[str],
    categorical_cols: List[str],
    seed: int,
    n_trials: int = 20,
    inner_splits: int = 3,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Find hyperparameters by maximizing mean inner-CV AUROC on the training data.

    This function receives only training data and never test indices. A new pipeline
    is built for every inner fold, so imputers, scalers and encoders are fitted on that
    fold's training part only.

    Args:
        model_name: A key of MODEL_REGISTRY.
        X_train: Training features.
        y_train: Training labels, aligned with `X_train`.
        numeric_cols: Numeric feature columns.
        categorical_cols: Categorical feature columns.
        seed: Seed for the sampler, the inner folds and the model.
        n_trials: Number of Optuna trials.
        inner_splits: Number of inner cross-validation folds.

    Returns:
        `(best_params, info)`. `info` has n_trials (including failed ones),
        n_failed_trials, best_value (inner-CV AUROC), tuning_time_s and
        failed_trial_errors (the error text of each failed trial). If the
        model's registry entry has `tunable=False`, no Optuna study runs at
        all: this returns `({}, info)` with n_trials=0, n_failed_trials=0,
        best_value=None, tuning_time_s=0.0, failed_trial_errors=[].

    Raises:
        ValueError: If `model_name` is not in the registry.
        RuntimeError: If every trial failed.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Supported: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]

    if not spec.tunable:
        return {}, {
            "n_trials": 0,
            "n_failed_trials": 0,
            "best_value": None,
            "tuning_time_s": 0.0,
            "failed_trial_errors": [],
        }

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    y_values = np.asarray(y_train)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    folds = list(inner_cv.split(X_train, y_values))
    failed_trial_errors: List[str] = []

    def objective(trial: optuna.trial.Trial) -> float:
        # Record the reason, then re-raise so Optuna marks this trial as failed.
        try:
            params = spec.suggest_params(trial)
            fold_scores = []
            for fit_idx, val_idx in folds:
                pipeline = build_pipeline(model_name, params, numeric_cols, categorical_cols, seed)
                pipeline.fit(X_train.iloc[fit_idx], y_values[fit_idx])
                val_prob = positive_class_proba(pipeline, X_train.iloc[val_idx])
                fold_scores.append(roc_auc_score(y_values[val_idx], val_prob))
            return float(np.mean(fold_scores))
        except Exception as exc:
            failed_trial_errors.append(f"trial {trial.number}: {type(exc).__name__}: {exc}")
            raise

    start = time.perf_counter()
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    tuning_time_s = time.perf_counter() - start

    # Counted from trial states, so a trial Optuna rejects (e.g. a NaN score) counts too.
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    if n_complete == 0:
        raise RuntimeError(
            f"All {len(study.trials)} trials failed for '{model_name}'. "
            f"First error: {failed_trial_errors[0] if failed_trial_errors else 'unknown'}"
        )

    info: Dict[str, Any] = {
        "n_trials": len(study.trials),
        "n_failed_trials": len(study.trials) - n_complete,
        "best_value": float(study.best_value),
        "tuning_time_s": tuning_time_s,
        "failed_trial_errors": failed_trial_errors,
    }
    return dict(study.best_trial.params), info

"""Registry of the classical models and their shared pipeline construction.

Every model receives the output of the same preprocessor, including median
imputation. No model sees NaN, so XGBoost's and LightGBM's native missing-value
handling is never used and the comparison stays fair. class_weight and early
stopping are deliberately not used either.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor

# scikit-learn 1.8+ warns about penalty="l2" on every fit (the argument is removed in
# 1.10). The result is identical to the default, and requirements.txt pins 1.9.1.
warnings.filterwarnings("ignore", message="'penalty' was deprecated", category=FutureWarning)

Params = Dict[str, Any]

# Models whose fit() produces a `training_info_` attribute and needs a GPU.
# Used by src/runner.py to decide whether to write a diagnostics file, cap
# torch threads, and clear CUDA memory after each fit.
DEEP_MODEL_NAMES: frozenset = frozenset({"mlp", "ftt", "tabnet"})


@dataclass(frozen=True)
class ModelSpec:
    """One registry entry: how to build a model and how to tune it.

    `tunable` exists so a future non-tunable model can be added without
    changing the harness: when False, `tune_model` (src/tuning.py) skips
    Optuna entirely and returns empty params.
    """

    name: str
    make_model: Callable[[Params, int], Any]
    suggest_params: Callable[[optuna.trial.BaseTrial], Params]
    needs_scaling: bool
    tunable: bool = True


def _make_logreg(params: Params, seed: int) -> LogisticRegression:
    return LogisticRegression(
        penalty="l2", solver="lbfgs", max_iter=5000, random_state=seed, **params
    )


def _suggest_logreg(trial: optuna.trial.BaseTrial) -> Params:
    return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}


def _make_rf(params: Params, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_jobs=1, random_state=seed, **params)


def _suggest_rf(trial: optuna.trial.BaseTrial) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }


def _make_xgb(params: Params, seed: int) -> XGBClassifier:
    return XGBClassifier(
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=1,
        verbosity=0,
        random_state=seed,
        **params,
    )


def _suggest_xgb(trial: optuna.trial.BaseTrial) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def _make_lgbm(params: Params, seed: int) -> LGBMClassifier:
    return LGBMClassifier(n_jobs=1, verbose=-1, subsample_freq=1, random_state=seed, **params)


def _suggest_lgbm(trial: optuna.trial.BaseTrial) -> Params:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "num_leaves": trial.suggest_int("num_leaves", 4, 31),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 2, 30),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10, log=True),
    }


def _make_mlp(params: Params, seed: int) -> Any:
    # Imported lazily so this module (and the classical models/tests) work in
    # an environment without torch installed.
    from src.deep_models import MLPClassifier

    return MLPClassifier(seed=seed, **params)


def _suggest_mlp(trial: optuna.trial.BaseTrial) -> Params:
    return {
        "n_layers": trial.suggest_int("n_layers", 1, 3),
        "hidden": trial.suggest_int("hidden", 16, 256, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }


def _make_ftt(params: Params, seed: int) -> Any:
    from src.deep_models import FTTransformerClassifier

    return FTTransformerClassifier(seed=seed, **params)


def _suggest_ftt(trial: optuna.trial.BaseTrial) -> Params:
    return {
        "d_token": trial.suggest_categorical("d_token", [16, 32, 64]),
        "n_blocks": trial.suggest_int("n_blocks", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }


def _make_tabnet(params: Params, seed: int) -> Any:
    from src.deep_models import TabNetClassifierWrapper

    return TabNetClassifierWrapper(seed=seed, **params)


def _suggest_tabnet(trial: optuna.trial.BaseTrial) -> Params:
    return {
        "n_d": trial.suggest_categorical("n_d", [8, 16, 32]),
        "n_steps": trial.suggest_int("n_steps", 2, 5),
        "gamma": trial.suggest_float("gamma", 1.0, 2.0),
        "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-5, 1e-2, log=True),
        "lr": trial.suggest_float("lr", 1e-3, 3e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
    }


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "logreg": ModelSpec("logreg", _make_logreg, _suggest_logreg, needs_scaling=True, tunable=True),
    "rf": ModelSpec("rf", _make_rf, _suggest_rf, needs_scaling=False, tunable=True),
    "xgb": ModelSpec("xgb", _make_xgb, _suggest_xgb, needs_scaling=False, tunable=True),
    "lgbm": ModelSpec("lgbm", _make_lgbm, _suggest_lgbm, needs_scaling=False, tunable=True),
    "mlp": ModelSpec("mlp", _make_mlp, _suggest_mlp, needs_scaling=True, tunable=True),
    "ftt": ModelSpec("ftt", _make_ftt, _suggest_ftt, needs_scaling=True, tunable=True),
    "tabnet": ModelSpec("tabnet", _make_tabnet, _suggest_tabnet, needs_scaling=True, tunable=True),
}


def build_pipeline(
    model_name: str,
    params: Params,
    numeric_cols: List[str],
    categorical_cols: List[str],
    seed: int,
) -> Pipeline:
    """Return an unfitted preprocessor + classifier pipeline for `model_name`.

    The input width of any model must come from the fitted preprocessor output, never
    from a hardcoded number: one-hot width depends on the categories in the training
    data, so it changes with the training subset.

    Args:
        model_name: A key of MODEL_REGISTRY.
        params: Hyperparameters passed to the model's constructor.
        numeric_cols: Numeric feature columns.
        categorical_cols: Categorical feature columns.
        seed: Random seed for the model.

    Raises:
        ValueError: If `model_name` is not in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Supported: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[model_name]
    return Pipeline(
        [
            ("prep", build_preprocessor(numeric_cols, categorical_cols, scale=spec.needs_scaling)),
            ("clf", spec.make_model(params, seed)),
        ]
    )


def effective_params(model_name: str, params: Params, fraction: float) -> Params:
    """Return the parameters to fit with at `fraction`, as a new dict.

    Hyperparameters are tuned once at 100% training size and reused for smaller
    fractions. Leaf-size constraints measured in absolute rows or hessian units would
    otherwise forbid any split on small subsets and produce a constant prediction, so
    we keep their RELATIVE size constant by scaling them with the fraction:

      - "xgb": min_child_weight = max(1.0, min_child_weight * fraction), kept as an
        unrounded float because it is a hessian threshold.
      - "lgbm": min_child_samples = max(1, round(min_child_samples * fraction)).
      - "rf": min_samples_leaf = max(1, round(min_samples_leaf * fraction)).
      - "logreg": unchanged, because C already scales with sample size.
      - "mlp", "ftt", "tabnet": unchanged. Their own training schedule
        (src/deep_models.py: `training_schedule`) adapts batch size, epoch
        count and patience to the fraction's actual row count internally,
        and that adaptation is recorded in `training_info_`, not here.

    `round` is Python's built-in, so exact halves go to the nearest even number
    (2.5 becomes 2). No other parameter changes, and the input is never mutated.

    Args:
        model_name: A key of MODEL_REGISTRY.
        params: The tuned hyperparameters.
        fraction: Training fraction, in (0, 1]. At 1.0 the result equals `params`.

    Raises:
        ValueError: If `model_name` is unknown or `fraction` is outside (0, 1].
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Supported: {sorted(MODEL_REGISTRY)}")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    effective = dict(params)
    # An early return keeps values exactly as tuned, including int vs float.
    if fraction == 1.0:
        return effective

    if model_name == "xgb":
        effective["min_child_weight"] = max(1.0, float(params["min_child_weight"]) * fraction)
    elif model_name == "lgbm":
        effective["min_child_samples"] = max(1, round(params["min_child_samples"] * fraction))
    elif model_name == "rf":
        effective["min_samples_leaf"] = max(1, round(params["min_samples_leaf"] * fraction))
    return effective


def positive_class_proba(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Return predicted probabilities of class 1 as a float64 array."""
    positive_column = list(pipeline.classes_).index(1)
    return np.asarray(pipeline.predict_proba(X)[:, positive_column], dtype=np.float64)

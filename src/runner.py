"""Run one benchmark task: one (dataset, model, outer split).

A task tunes once on the full training fold, then fits and evaluates at each training
fraction with those same hyperparameters. The raw CSV is written last and atomically,
so its presence means every other file for that task was already finished. An
interrupted task therefore has no CSV and simply runs again on resume.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.data import load_dataset
from src.metrics import compute_metrics, prediction_diversity
from src.models import DEEP_MODEL_NAMES, build_pipeline, effective_params, positive_class_proba
from src.tuning import tune_model
from src.utils import make_splits, subsample_train

# The outer splits are fixed, so every model is evaluated on identical folds.
OUTER_SPLIT_SEED: int = 42
OUTER_N_SPLITS: int = 5
OUTER_N_REPEATS: int = 2

METRIC_COLUMNS: List[str] = ["auroc", "f1", "balanced_accuracy", "brier", "log_loss", "ece"]
DIVERSITY_COLUMNS: List[str] = ["n_unique_probs", "constant_prediction"]
RAW_COLUMNS: List[str] = [
    "dataset", "model", "split_idx", "fraction", "seed", "n_train", "n_test",
    "n_features_encoded", *METRIC_COLUMNS, "fit_time_s", "predict_time_s",
    *DIVERSITY_COLUMNS, "status", "error", "params_json",
]
RESULT_SUBDIRS: Tuple[str, ...] = ("raw", "tuning", "predictions", "models", "diagnostics")


def task_stem(dataset: str, model_name: str, split_idx: int) -> str:
    """Return the file name stem shared by all of a task's output files."""
    return f"{dataset}__{model_name}__split{split_idx}"


def make_result_dirs(results_dir: Union[str, Path]) -> Path:
    """Create the results subfolders if missing and return `results_dir` as a Path."""
    root = Path(results_dir)
    for subdir in RESULT_SUBDIRS:
        (root / subdir).mkdir(parents=True, exist_ok=True)
    return root


def is_constant_prediction(rows: pd.DataFrame) -> pd.Series:
    """Return a mask of rows flagged constant.

    Read through str because the column holds empty values if any row in a CSV failed,
    in which case pandas does not read it as a bool column.
    """
    return rows["constant_prediction"].astype(str) == "True"


def _is_task_complete(
    raw_path: Path, n_fractions: int, diagnostics_path: Optional[Path] = None
) -> bool:
    """A task is complete if its CSV is in the current format with one "ok" row per fraction.

    Args:
        raw_path: Path to the task's raw CSV.
        n_fractions: Expected number of rows (one per training fraction).
        diagnostics_path: For deep models (src.models.DEEP_MODEL_NAMES), the
            diagnostics JSON must also exist; pass None for classical models,
            which never write one.
    """
    if not raw_path.exists():
        return False
    if diagnostics_path is not None and not diagnostics_path.exists():
        return False
    rows = pd.read_csv(raw_path)
    # An older CSV lacks the current columns, so that task must run again.
    if not set(RAW_COLUMNS).issubset(rows.columns):
        return False
    return len(rows) == n_fractions and (rows["status"] == "ok").all()


def _describe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _evaluate_fraction(
    row: Dict[str, Any],
    probabilities: Dict[float, np.ndarray],
    fraction: float,
    X: pd.DataFrame,
    y: pd.Series,
    meta: Dict[str, Any],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    tuned_params: Dict[str, Any],
    task_seed: int,
    model_path: Optional[Path],
) -> None:
    """Fit and evaluate one fraction, updating `row` and `probabilities` as it goes.

    Results land in `row` step by step, so a failed row still shows what was reached.
    """
    fit_params = effective_params(model_name, tuned_params, fraction)
    row["params_json"] = json.dumps(
        {"tuned": tuned_params, "effective": fit_params, "fraction": fraction}
    )

    sub_idx = subsample_train(train_idx, y, fraction, task_seed)
    row["n_train"] = len(sub_idx)

    pipeline = build_pipeline(
        model_name, fit_params, meta["numeric_cols"], meta["categorical_cols"], task_seed
    )
    start = time.perf_counter()
    pipeline.fit(X.iloc[sub_idx], y.iloc[sub_idx])
    row["fit_time_s"] = time.perf_counter() - start

    row["n_features_encoded"] = pipeline.named_steps["prep"].transform(X.iloc[sub_idx[:1]]).shape[1]

    start = time.perf_counter()
    y_prob = positive_class_proba(pipeline, X.iloc[test_idx])
    row["predict_time_s"] = time.perf_counter() - start
    probabilities[fraction] = y_prob

    row["n_unique_probs"], row["constant_prediction"] = prediction_diversity(y_prob)
    row.update(compute_metrics(y.iloc[test_idx].to_numpy(), y_prob))

    # Deep models (mlp/ftt/tabnet) expose training_info_ on the fitted estimator;
    # classical models don't, so this stays unset (and no diagnostics file is
    # written) for them. Not a RAW_COLUMNS key, so it never reaches the raw CSV.
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "training_info_"):
        row["_training_info"] = dict(clf.training_info_)

    # Inside this function so that a failed save shows up as a failed row.
    if model_path is not None:
        joblib.dump(pipeline, model_path, compress=3)


def run_task(
    dataset: str,
    model_name: str,
    split_idx: int,
    seed: int = 42,
    fractions: Sequence[float] = (0.1, 0.25, 0.5, 1.0),
    n_trials: int = 20,
    results_dir: Union[str, Path] = "results",
    save_models: bool = True,
) -> Dict[str, Any]:
    """Tune once, then fit and evaluate at each training fraction.

    Writes, under `results_dir`:
      - raw/{stem}.csv: one row per fraction.
      - tuning/{stem}.json: best parameters and tuning info.
      - predictions/{stem}.npz: test_idx, y_true and prob_{fraction} arrays. A failed
        fraction's array is all NaN, so every fraction always has an array.
      - models/{stem}.joblib: the fitted fraction 1.0 pipeline, if `save_models`.

    The hyperparameters are tuned once on the full training fold and reused for the
    smaller fractions, with only leaf-size settings scaled by `effective_params`. The
    tuning JSON holds the tuned values; each row's params_json holds both sets.

    Args:
        dataset: A dataset name accepted by `load_dataset`.
        model_name: A key of MODEL_REGISTRY.
        split_idx: Outer split index, 0 to 9.
        seed: Base seed. The task uses `seed + split_idx` for tuning, subsampling and
            the model, and that value goes in the rows' "seed" column. The outer splits
            always use OUTER_SPLIT_SEED.
        fractions: Training fractions to evaluate.
        n_trials: Number of Optuna trials.
        results_dir: Root folder for output files.
        save_models: Whether to save the fraction 1.0 pipeline.

    Returns:
        A dict with "skipped", "n_ok" and "n_failed".
    """
    root = make_result_dirs(results_dir)
    stem = task_stem(dataset, model_name, split_idx)
    raw_path = root / "raw" / f"{stem}.csv"
    is_deep_model = model_name in DEEP_MODEL_NAMES
    diagnostics_path = root / "diagnostics" / f"{stem}.json"

    if is_deep_model:
        # Lazily imported here (not at module level) so classical-only
        # environments without torch can still import and use this module.
        import torch

        torch.set_num_threads(1)

    if _is_task_complete(raw_path, len(fractions), diagnostics_path if is_deep_model else None):
        print(f"Skipping {stem}: already complete")
        return {"skipped": True, "n_ok": len(fractions), "n_failed": 0}

    X, y, meta = load_dataset(dataset, verbose=False)
    splits = make_splits(
        X, y, n_splits=OUTER_N_SPLITS, n_repeats=OUTER_N_REPEATS, seed=OUTER_SPLIT_SEED
    )
    train_idx, test_idx = splits[split_idx]
    task_seed = seed + split_idx

    # Tune once, on the full training fold only. If that fails, every fraction gets a
    # failed row, so the task is reported rather than skipped.
    best_params: Optional[Dict[str, Any]] = None
    tuning_error = ""
    tuning_record: Dict[str, Any] = {
        "dataset": dataset, "model": model_name, "split_idx": split_idx, "seed": task_seed,
    }
    try:
        best_params, info = tune_model(
            model_name,
            X.iloc[train_idx],
            y.iloc[train_idx],
            meta["numeric_cols"],
            meta["categorical_cols"],
            task_seed,
            n_trials=n_trials,
        )
        tuning_record.update(best_params=best_params, info=info)
    except Exception as exc:
        tuning_error = f"tuning failed: {_describe_error(exc)}"
        tuning_record.update(best_params=None, error=tuning_error)
    with open(root / "tuning" / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(tuning_record, f, indent=2)

    probabilities: Dict[float, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []

    for fraction in fractions:
        row: Dict[str, Any] = {column: None for column in RAW_COLUMNS}
        row.update(
            dataset=dataset,
            model=model_name,
            split_idx=split_idx,
            fraction=fraction,
            seed=task_seed,
            n_test=len(test_idx),
            params_json="",
            status="ok",
            error="",
        )
        model_path = (
            root / "models" / f"{stem}.joblib" if save_models and fraction == 1.0 else None
        )
        try:
            if best_params is None:
                raise RuntimeError(tuning_error)
            _evaluate_fraction(
                row, probabilities, fraction, X, y, meta, train_idx, test_idx,
                model_name, best_params, task_seed, model_path,
            )
        except Exception as exc:
            # These columns stay empty on a failed row, even if only a later step failed.
            row.update({column: None for column in METRIC_COLUMNS + DIVERSITY_COLUMNS})
            row.update(status="failed", error=_describe_error(exc))
        else:
            # Still an "ok" row: the run is valid, just degenerate.
            if row["constant_prediction"]:
                print(
                    f"WARNING: constant predictions: dataset={dataset}, model={model_name}, "
                    f"split={split_idx}, fraction={fraction}"
                )
        finally:
            # The fitted pipeline (if any) is local to _evaluate_fraction and already
            # out of scope here; this just reclaims the GPU memory it held.
            if is_deep_model and "torch" in sys.modules:
                sys.modules["torch"].cuda.empty_cache()
        rows.append(row)

    if is_deep_model:
        diagnostics = {
            str(row["fraction"]): row.pop("_training_info", {"error": row["error"]})
            for row in rows
        }
        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)

    # A model that cannot be faithfully reloaded would silently corrupt every later
    # analysis that loads results/models/*.joblib, so this is a hard stop, not a
    # per-fraction failure: it propagates out of run_task uncaught.
    model_path_1_0 = root / "models" / f"{stem}.joblib" if save_models else None
    if model_path_1_0 is not None and model_path_1_0.exists() and 1.0 in probabilities:
        reloaded = joblib.load(model_path_1_0)
        reloaded_prob = positive_class_proba(reloaded, X.iloc[test_idx])
        max_diff = float(np.max(np.abs(reloaded_prob - probabilities[1.0])))
        if not np.allclose(reloaded_prob, probabilities[1.0], atol=1e-4):
            raise RuntimeError(
                f"{stem}: saved model did not reproduce predictions after a "
                f"joblib reload (max abs diff {max_diff:.6g} > 1e-4)."
            )

    arrays: Dict[str, np.ndarray] = {
        "test_idx": test_idx,
        "y_true": y.iloc[test_idx].to_numpy(),
    }
    for fraction in fractions:
        arrays[f"prob_{fraction}"] = probabilities.get(
            fraction, np.full(len(test_idx), np.nan, dtype=np.float64)
        )
    np.savez(root / "predictions" / f"{stem}.npz", **arrays)

    # Written last, and atomically: this file is what marks the task as finished.
    temporary_path = raw_path.with_suffix(".csv.tmp")
    pd.DataFrame(rows, columns=RAW_COLUMNS).to_csv(temporary_path, index=False)
    os.replace(temporary_path, raw_path)

    n_ok = sum(row["status"] == "ok" for row in rows)
    return {"skipped": False, "n_ok": n_ok, "n_failed": len(rows) - n_ok}


def run_task_safe(**task_kwargs: Any) -> Dict[str, Any]:
    """Run a task for the command-line script, reporting a crash instead of raising.

    One task crashing outside the per-fraction handling (e.g. a dataset that fails to
    load) must not stop the others, so it comes back in the result as "crash".
    """
    start = time.perf_counter()
    result: Dict[str, Any] = {"skipped": False, "n_ok": 0, "n_failed": 0, "crash": ""}
    try:
        result.update(run_task(**task_kwargs))
    except Exception as exc:
        result["crash"] = _describe_error(exc)
    result["stem"] = task_stem(
        task_kwargs["dataset"], task_kwargs["model_name"], task_kwargs["split_idx"]
    )
    result["elapsed_s"] = time.perf_counter() - start
    return result

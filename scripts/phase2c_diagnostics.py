"""Phase 2c: read-only diagnostics on the existing Phase 2a/2b results, plus a
schedule-sensitivity check that re-fits the deep models with already-tuned
hyperparameters (no tuning).

This script never writes to results/raw, results/tuning, results/predictions,
results/models, results/diagnostics, results/phase2a_all_runs.csv or
results/all_runs.csv. All new outputs go to results/phase2c/. It never edits
src/: the sensitivity check patches src.deep_models's module-level constants
(MAX_STEPS, PATIENCE_STEPS, VALIDATION_FRACTION) only inside this process, and
always restores them afterwards (see `patched_schedule`).

Subcommands: endpoints, tabnet, sensitivity, xgb-wide-range, all
("all" runs endpoints + tabnet only; sensitivity and xgb-wide-range are each
run on their own because of the STOP gates described in the project brief).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset
from src.metrics import compute_metrics, prediction_diversity
from src.models import DEEP_MODEL_NAMES, MODEL_REGISTRY, build_pipeline, effective_params, positive_class_proba
from src.utils import make_splits, subsample_train

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "phase2c"
RAW_DIR = RESULTS_DIR / "raw"
TUNING_DIR = RESULTS_DIR / "tuning"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"

ALL_DATASETS: List[str] = ["pima_diabetes", "heart_cleveland", "ckd", "ilpd", "breast_wisconsin"]
ALL_MODELS: List[str] = sorted(MODEL_REGISTRY)

OUTER_SPLIT_SEED = 42
OUTER_N_SPLITS = 5
OUTER_N_REPEATS = 2
N_SPLITS_TOTAL = OUTER_N_SPLITS * OUTER_N_REPEATS  # 10

ENDPOINT_REL_TOL = 0.02


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def task_stem(dataset: str, model_name: str, split_idx: int) -> str:
    return f"{dataset}__{model_name}__split{split_idx}"


def load_tuned_params(dataset: str, model_name: str, split_idx: int) -> Dict[str, Any]:
    """Load the tuned `best_params` for one task from results/tuning/*.json."""
    path = TUNING_DIR / f"{task_stem(dataset, model_name, split_idx)}.json"
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)
    best_params = record.get("best_params")
    if best_params is None:
        raise ValueError(f"{path}: tuning failed for this task (best_params is null)")
    return best_params


def load_raw(dataset: str, model_name: str, split_idx: int) -> pd.DataFrame:
    path = RAW_DIR / f"{task_stem(dataset, model_name, split_idx)}.csv"
    return pd.read_csv(path)


def load_diagnostics(dataset: str, model_name: str, split_idx: int) -> Dict[str, Any]:
    path = DIAGNOSTICS_DIR / f"{task_stem(dataset, model_name, split_idx)}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 2a. endpoints
# ---------------------------------------------------------------------------


class _RecordingTrial:
    """Stand-in for optuna.trial.Trial: records suggest_* calls instead of sampling.

    Returned values are arbitrary (the low end / first choice) since only the
    *search space* (name, bounds, log-scale, choices) is needed here, never an
    actual sampled value.
    """

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def suggest_int(self, name: str, low: int, high: int, *, step: int = 1, log: bool = False) -> int:
        self.records.append(
            {"name": name, "low": low, "high": high, "log": log, "type": "int", "choices": None}
        )
        return low

    def suggest_float(
        self, name: str, low: float, high: float, *, step: Optional[float] = None, log: bool = False
    ) -> float:
        self.records.append(
            {"name": name, "low": low, "high": high, "log": log, "type": "float", "choices": None}
        )
        return low

    def suggest_categorical(self, name: str, choices: Sequence[Any]) -> Any:
        choices = list(choices)
        self.records.append(
            {"name": name, "low": None, "high": None, "log": False, "type": "categorical", "choices": choices}
        )
        return choices[0]


def get_search_space() -> Dict[str, List[Dict[str, Any]]]:
    """Return {model_name: [param_record, ...]} by replaying each suggest_params."""
    space: Dict[str, List[Dict[str, Any]]] = {}
    for name, spec in MODEL_REGISTRY.items():
        trial = _RecordingTrial()
        spec.suggest_params(trial)
        space[name] = trial.records
    return space


def classify_param(record: Dict[str, Any], value: Any) -> Optional[str]:
    """Classify a tuned `value` against its search-space `record` as "lower", "upper" or None.

    Integers and numeric-categorical choices: exact equality to the bound.
    Continuous (float) parameters: within ENDPOINT_REL_TOL (2%) of the range,
    measured on a log scale when the parameter is log-uniform.
    A categorical parameter whose choices are not ALL numeric has no meaningful
    endpoint and always returns None (e.g. rf's max_features).
    """
    if record["type"] == "categorical":
        choices = record["choices"]
        if len(choices) < 2 or not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in choices):
            return None
        lo, hi = min(choices), max(choices)
        if value == lo:
            return "lower"
        if value == hi:
            return "upper"
        return None

    lo, hi = record["low"], record["high"]
    if record["type"] == "int":
        if value == lo:
            return "lower"
        if value == hi:
            return "upper"
        return None

    # Continuous (float).
    if record.get("log"):
        if lo <= 0 or hi <= 0:
            return None
        log_lo, log_hi, log_v = math.log(lo), math.log(hi), math.log(value)
        span = log_hi - log_lo
    else:
        log_lo = log_hi = log_v = None
        span = hi - lo
    if span <= 0:
        return None
    if record.get("log"):
        if (log_v - log_lo) <= ENDPOINT_REL_TOL * span:
            return "lower"
        if (log_hi - log_v) <= ENDPOINT_REL_TOL * span:
            return "upper"
    else:
        if (value - lo) <= ENDPOINT_REL_TOL * span:
            return "lower"
        if (hi - value) <= ENDPOINT_REL_TOL * span:
            return "upper"
    return None


def run_endpoints(datasets: Sequence[str] = ALL_DATASETS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read-only: classify every tuned hyperparameter against its search-space endpoints.

    Returns (overall_df, by_dataset_df); also writes them to results/phase2c/.
    """
    space = get_search_space()
    overall_rows: List[Dict[str, Any]] = []
    by_dataset_rows: List[Dict[str, Any]] = []

    for model_name in ALL_MODELS:
        records = space[model_name]
        for record in records:
            if record["type"] == "categorical":
                choices = record["choices"]
                numeric = len(choices) >= 2 and all(
                    isinstance(c, (int, float)) and not isinstance(c, bool) for c in choices
                )
                low = min(choices) if numeric else None
                high = max(choices) if numeric else None
                if not numeric:
                    # No meaningful endpoint concept (e.g. rf's max_features); skip the row.
                    continue
            else:
                low, high = record["low"], record["high"]

            per_dataset_counts: Dict[str, Dict[str, int]] = {}
            n_lower = n_upper = n_tasks = 0
            for dataset in datasets:
                d_lower = d_upper = d_tasks = 0
                for split_idx in range(N_SPLITS_TOTAL):
                    try:
                        tuned = load_tuned_params(dataset, model_name, split_idx)
                    except (FileNotFoundError, ValueError):
                        continue
                    if record["name"] not in tuned:
                        continue
                    d_tasks += 1
                    verdict = classify_param(record, tuned[record["name"]])
                    if verdict == "lower":
                        d_lower += 1
                    elif verdict == "upper":
                        d_upper += 1
                per_dataset_counts[dataset] = {"n_at_lower": d_lower, "n_at_upper": d_upper, "n_tasks": d_tasks}
                n_lower += d_lower
                n_upper += d_upper
                n_tasks += d_tasks
                by_dataset_rows.append(
                    {
                        "model": model_name,
                        "parameter": record["name"],
                        "dataset": dataset,
                        "low": low,
                        "high": high,
                        "n_at_lower": d_lower,
                        "n_at_upper": d_upper,
                        "n_tasks": d_tasks,
                    }
                )

            overall_rows.append(
                {
                    "model": model_name,
                    "parameter": record["name"],
                    "low": low,
                    "high": high,
                    "n_at_lower": n_lower,
                    "n_at_upper": n_upper,
                    "n_tasks": n_tasks,
                }
            )

    overall_df = pd.DataFrame(overall_rows)
    overall_df["n_at_endpoint"] = overall_df["n_at_lower"] + overall_df["n_at_upper"]
    overall_df = overall_df.sort_values("n_at_endpoint", ascending=False).drop(columns="n_at_endpoint")
    by_dataset_df = pd.DataFrame(by_dataset_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(OUT_DIR / "endpoints_overall.csv", index=False)
    by_dataset_df.to_csv(OUT_DIR / "endpoints_by_dataset.csv", index=False)
    return overall_df, by_dataset_df


# ---------------------------------------------------------------------------
# 2b. tabnet
# ---------------------------------------------------------------------------


def _load_per_split_auroc(dataset: str, model_name: str) -> pd.DataFrame:
    """Long-format (dataset, model, split_idx, fraction, auroc) from results/raw."""
    rows = []
    for split_idx in range(N_SPLITS_TOTAL):
        df = load_raw(dataset, model_name, split_idx)
        for _, r in df.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "model": model_name,
                    "split_idx": split_idx,
                    "fraction": r["fraction"],
                    "auroc": r["auroc"],
                }
            )
    return pd.DataFrame(rows)


def find_collapsed_splits(
    per_split_1_0: pd.DataFrame, thresholds: Dict[str, float]
) -> pd.DataFrame:
    """Rows of `per_split_1_0` (dataset, split_idx, auroc) whose auroc is below
    `thresholds[dataset]`. `per_split_1_0` must already be filtered to fraction 1.0
    and the model of interest.
    """
    out = []
    for _, r in per_split_1_0.iterrows():
        threshold = thresholds.get(r["dataset"])
        if threshold is not None and r["auroc"] < threshold:
            out.append(r)
    if not out:
        return per_split_1_0.iloc[0:0].copy()
    return pd.DataFrame(out).reset_index(drop=True)


def run_tabnet_diagnostics(
    datasets: Sequence[str] = ("heart_cleveland", "ckd"),
    comparison_models: Sequence[str] = ("tabnet", "logreg", "mlp"),
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read-only: per-split AUROC tables plus full diagnostics for TabNet's collapsed splits."""
    if thresholds is None:
        thresholds = {"ckd": 0.90, "heart_cleveland": 0.75}

    per_split_frames = []
    for dataset in datasets:
        for model_name in comparison_models:
            per_split_frames.append(_load_per_split_auroc(dataset, model_name))
    per_split_df = pd.concat(per_split_frames, ignore_index=True)

    tabnet_1_0 = per_split_df[(per_split_df["model"] == "tabnet") & (per_split_df["fraction"] == 1.0)]
    collapsed = find_collapsed_splits(tabnet_1_0, thresholds)

    detail_rows = []
    for _, row in collapsed.iterrows():
        dataset, split_idx = row["dataset"], int(row["split_idx"])
        tuned = load_tuned_params(dataset, "tabnet", split_idx)
        diag = load_diagnostics(dataset, "tabnet", split_idx)
        diag_1_0 = diag.get("1.0", {})
        diag_0_1 = diag.get("0.1", {})

        npz_path = PREDICTIONS_DIR / f"{task_stem(dataset, 'tabnet', split_idx)}.npz"
        arrays = np.load(npz_path)
        prob = arrays["prob_1.0"]
        finite = prob[np.isfinite(prob)]
        n_unique, _ = prediction_diversity(finite) if finite.size else (0, True)

        logreg_raw = load_raw(dataset, "logreg", split_idx)
        logreg_auroc = float(logreg_raw.loc[logreg_raw["fraction"] == 1.0, "auroc"].iloc[0])

        detail_rows.append(
            {
                "dataset": dataset,
                "split_idx": split_idx,
                "auroc_1_0": float(row["auroc"]),
                "tuned_params_json": json.dumps(tuned),
                "diag_1_0_n_fit": diag_1_0.get("n_fit"),
                "diag_1_0_n_train_part": diag_1_0.get("n_train_part"),
                "diag_1_0_n_val": diag_1_0.get("n_val"),
                "diag_1_0_early_stopping": diag_1_0.get("early_stopping"),
                "diag_1_0_effective_batch": diag_1_0.get("effective_batch"),
                "diag_1_0_epochs_run": diag_1_0.get("epochs_run"),
                "diag_1_0_best_epoch": diag_1_0.get("best_epoch"),
                "diag_1_0_best_val_logloss": diag_1_0.get("best_val_logloss"),
                "diag_1_0_max_epochs": diag_1_0.get("max_epochs"),
                "diag_1_0_patience_epochs": diag_1_0.get("patience_epochs"),
                "diag_0_1_n_fit": diag_0_1.get("n_fit"),
                "diag_0_1_n_train_part": diag_0_1.get("n_train_part"),
                "diag_0_1_n_val": diag_0_1.get("n_val"),
                "diag_0_1_early_stopping": diag_0_1.get("early_stopping"),
                "diag_0_1_effective_batch": diag_0_1.get("effective_batch"),
                "diag_0_1_epochs_run": diag_0_1.get("epochs_run"),
                "diag_0_1_best_epoch": diag_0_1.get("best_epoch"),
                "diag_0_1_best_val_logloss": diag_0_1.get("best_val_logloss"),
                "diag_0_1_max_epochs": diag_0_1.get("max_epochs"),
                "diag_0_1_patience_epochs": diag_0_1.get("patience_epochs"),
                "pred_n_unique_probs": n_unique,
                "pred_min": float(finite.min()) if finite.size else None,
                "pred_max": float(finite.max()) if finite.size else None,
                "pred_mean": float(finite.mean()) if finite.size else None,
                "pred_frac_positive": float((finite >= 0.5).mean()) if finite.size else None,
                "logreg_auroc_same_split": logreg_auroc,
            }
        )
    collapsed_detail_df = pd.DataFrame(detail_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_split_df.to_csv(OUT_DIR / "tabnet_per_split.csv", index=False)
    collapsed_detail_df.to_csv(OUT_DIR / "tabnet_collapsed_splits.csv", index=False)
    return per_split_df, collapsed_detail_df


# ---------------------------------------------------------------------------
# 2c. sensitivity
# ---------------------------------------------------------------------------

SENSITIVITY_MODELS: List[str] = ["mlp", "ftt", "tabnet"]
SENSITIVITY_SPLITS: List[int] = [0, 1, 2]
SENSITIVITY_FRACTIONS: List[float] = [0.1, 1.0]
VARIANTS: Dict[str, Dict[str, float]] = {
    "baseline": {"max_steps": 600, "patience_steps": 60, "validation_fraction": 0.15},
    "A_long": {"max_steps": 2000, "patience_steps": 150, "validation_fraction": 0.15},
    "B_bigval": {"max_steps": 600, "patience_steps": 60, "validation_fraction": 0.30},
    "C_long_bigval": {"max_steps": 2000, "patience_steps": 150, "validation_fraction": 0.30},
}
SENSITIVITY_COLUMNS: List[str] = [
    "dataset", "model", "split_idx", "fraction", "variant",
    "auroc", "brier", "log_loss", "ece", "n_unique_probs", "fit_time_s",
    "n_train", "n_val", "early_stopping", "effective_batch", "max_epochs",
    "patience_epochs", "epochs_run", "best_epoch", "best_val_logloss",
    "hit_cap", "nan_loss", "status", "error",
]


@contextlib.contextmanager
def patched_schedule(max_steps: int, patience_steps: int, validation_fraction: float):
    """Temporarily patch src.deep_models's module-level training-schedule constants.

    These constants are read as bare globals at call time inside
    `_run_training_loop` and `TabNetClassifierWrapper.fit` (by design, so that
    this kind of monkeypatching works), never bound as function-default values,
    so reassigning the module attributes here actually changes the next fit's
    behaviour. Always restores the original values, including on an exception.
    """
    import src.deep_models as dm

    original = (dm.MAX_STEPS, dm.PATIENCE_STEPS, dm.VALIDATION_FRACTION)
    dm.MAX_STEPS = max_steps
    dm.PATIENCE_STEPS = patience_steps
    dm.VALIDATION_FRACTION = validation_fraction
    try:
        yield dm
    finally:
        dm.MAX_STEPS, dm.PATIENCE_STEPS, dm.VALIDATION_FRACTION = original


class ScheduleVerificationError(RuntimeError):
    """Raised when a patched schedule constant did not actually take effect."""


def _verify_schedule_applied(
    dm: Any,
    variant: Dict[str, float],
    tuned_batch_size: int,
    y_fit: np.ndarray,
    task_seed: int,
    info: Dict[str, Any],
) -> None:
    """Recompute the expected schedule/validation split and compare to `info`.

    Raises ScheduleVerificationError if the patched constants did not take
    effect for this fit (max_epochs, patience_epochs or n_val mismatch).

    When a fit has fewer than 4 samples of either class (`early_stopping` is
    False), both the MLP/FTT training loop and the TabNet wrapper fall back to
    a fixed step budget keyed on FIXED_STEPS_NO_VALIDATION, which does not use
    MAX_STEPS/PATIENCE_STEPS at all -- so the max_epochs/patience_epochs check
    only applies when `early_stopping` is True. n_val is still checked in
    every case (it must be 0 regardless of variant when there is no split).
    """
    if info.get("early_stopping"):
        expected = dm.training_schedule(
            info["n_train_part"], tuned_batch_size,
            max_steps=int(variant["max_steps"]), patience_steps=int(variant["patience_steps"]),
        )
        if expected["max_epochs"] != info["max_epochs"] or expected["patience_epochs"] != info["patience_epochs"]:
            raise ScheduleVerificationError(
                f"max_epochs/patience_epochs mismatch: expected {expected}, "
                f"got max_epochs={info['max_epochs']} patience_epochs={info['patience_epochs']}"
            )
    split = dm.split_validation(y_fit, task_seed, variant["validation_fraction"])
    expected_n_val = 0 if split is None else len(split[1])
    if expected_n_val != info["n_val"]:
        raise ScheduleVerificationError(
            f"n_val mismatch: expected {expected_n_val} (validation_fraction="
            f"{variant['validation_fraction']}), got {info['n_val']}"
        )


def _dataset_cache() -> Dict[str, Tuple[pd.DataFrame, pd.Series, Dict[str, Any], List[Tuple[np.ndarray, np.ndarray]]]]:
    cache: Dict[str, Any] = {}

    def get(dataset: str):
        if dataset not in cache:
            X, y, meta = load_dataset(dataset, verbose=False)
            splits = make_splits(X, y, n_splits=OUTER_N_SPLITS, n_repeats=OUTER_N_REPEATS, seed=OUTER_SPLIT_SEED)
            cache[dataset] = (X, y, meta, splits)
        return cache[dataset]

    return get


def _fit_one(
    dataset: str, model_name: str, split_idx: int, fraction: float, variant_name: str,
    variant: Dict[str, float], get_dataset,
) -> Dict[str, Any]:
    """Reproduce one runner-style fit under a patched training schedule."""
    import torch

    row: Dict[str, Any] = {column: None for column in SENSITIVITY_COLUMNS}
    row.update(dataset=dataset, model=model_name, split_idx=split_idx, fraction=fraction, variant=variant_name)

    try:
        X, y, meta, splits = get_dataset(dataset)
        train_idx, test_idx = splits[split_idx]
        task_seed = OUTER_SPLIT_SEED + split_idx

        tuned = load_tuned_params(dataset, model_name, split_idx)
        fit_params = effective_params(model_name, tuned, fraction)
        sub_idx = subsample_train(train_idx, y, fraction, task_seed)
        row["n_train"] = len(sub_idx)
        y_fit = y.iloc[sub_idx].to_numpy()

        pipeline = build_pipeline(model_name, fit_params, meta["numeric_cols"], meta["categorical_cols"], task_seed)
        with patched_schedule(**variant) as dm:
            start = time.perf_counter()
            pipeline.fit(X.iloc[sub_idx], y.iloc[sub_idx])
            row["fit_time_s"] = time.perf_counter() - start

            clf = pipeline.named_steps["clf"]
            info = dict(clf.training_info_)
            _verify_schedule_applied(dm, variant, int(fit_params["batch_size"]), y_fit, task_seed, info)

        y_prob = positive_class_proba(pipeline, X.iloc[test_idx])
        metrics = compute_metrics(y.iloc[test_idx].to_numpy(), y_prob)
        n_unique, _ = prediction_diversity(y_prob)

        row.update(
            auroc=metrics["auroc"], brier=metrics["brier"], log_loss=metrics["log_loss"], ece=metrics["ece"],
            n_unique_probs=n_unique,
            n_val=info.get("n_val"), early_stopping=info.get("early_stopping"),
            effective_batch=info.get("effective_batch"), max_epochs=info.get("max_epochs"),
            patience_epochs=info.get("patience_epochs"), epochs_run=info.get("epochs_run"),
            best_epoch=info.get("best_epoch"), best_val_logloss=info.get("best_val_logloss"),
            nan_loss=info.get("nan_loss", False), status="ok", error="",
        )
        row["hit_cap"] = bool(info.get("epochs_run") is not None and info.get("max_epochs") is not None
                               and info["epochs_run"] >= info["max_epochs"])
        del pipeline
    except ScheduleVerificationError:
        raise
    except Exception as exc:  # noqa: BLE001 - recorded, never silently swallowed
        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        # Covers both the success path (pipeline already deleted above) and a
        # failure path where `pipeline` was built but the fit/predict raised
        # before reaching that `del`: drop it here too, before freeing the
        # CUDA cache, so a failed fit cannot pin GPU memory into the next one.
        if "pipeline" in locals():
            del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def _existing_sensitivity_keys(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df["dataset"], df["model"], df["split_idx"], df["fraction"], df["variant"]))


def _append_sensitivity_row(csv_path: Path, row: Dict[str, Any]) -> None:
    df = pd.DataFrame([row], columns=SENSITIVITY_COLUMNS)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


def all_sensitivity_combos() -> List[Tuple[str, str, int, float, str]]:
    combos = []
    for variant_name in VARIANTS:
        for dataset in ALL_DATASETS:
            for model_name in SENSITIVITY_MODELS:
                for split_idx in SENSITIVITY_SPLITS:
                    for fraction in SENSITIVITY_FRACTIONS:
                        combos.append((dataset, model_name, split_idx, fraction, variant_name))
    return combos


def run_baseline_check(get_dataset, csv_path: Path) -> pd.DataFrame:
    """Run (or reuse) the 90 baseline-variant fits and compare them to results/raw."""
    existing = _existing_sensitivity_keys(csv_path)
    baseline_combos = [c for c in all_sensitivity_combos() if c[4] == "baseline"]
    for dataset, model_name, split_idx, fraction, variant_name in baseline_combos:
        key = (dataset, model_name, split_idx, fraction, variant_name)
        if key in existing:
            continue
        row = _fit_one(dataset, model_name, split_idx, fraction, variant_name, VARIANTS[variant_name], get_dataset)
        _append_sensitivity_row(csv_path, row)

    df = pd.read_csv(csv_path)
    baseline_df = df[df["variant"] == "baseline"].copy()

    diffs = []
    for _, r in baseline_df.iterrows():
        if r["status"] != "ok":
            diffs.append(np.nan)
            continue
        raw = load_raw(r["dataset"], r["model"], int(r["split_idx"]))
        ref = raw.loc[raw["fraction"] == r["fraction"], "auroc"]
        diffs.append(abs(float(ref.iloc[0]) - float(r["auroc"])) if len(ref) else np.nan)
    baseline_df["abs_diff_vs_raw"] = diffs
    return baseline_df


def run_timing_gate(get_dataset, n_per_model: int = 2) -> Dict[str, Any]:
    """Time `n_per_model` A_long fits per model and extrapolate to the full 360."""
    variant = VARIANTS["A_long"]
    timings: Dict[str, List[float]] = {m: [] for m in SENSITIVITY_MODELS}
    for model_name in SENSITIVITY_MODELS:
        count = 0
        for dataset in ALL_DATASETS:
            for split_idx in SENSITIVITY_SPLITS:
                for fraction in SENSITIVITY_FRACTIONS:
                    if count >= n_per_model:
                        break
                    row = _fit_one(dataset, model_name, split_idx, fraction, "A_long_timing_probe", variant, get_dataset)
                    if row["status"] == "ok":
                        timings[model_name].append(row["fit_time_s"])
                    count += 1
                if count >= n_per_model:
                    break
            if count >= n_per_model:
                break

    per_model_mean = {m: (float(np.mean(v)) if v else None) for m, v in timings.items()}
    total_combos = len(all_sensitivity_combos())
    n_per_model_total = total_combos // len(SENSITIVITY_MODELS)
    estimate_s = sum(
        (per_model_mean[m] or 0.0) * n_per_model_total for m in SENSITIVITY_MODELS
    )
    return {
        "per_model_mean_s": per_model_mean,
        "timings_raw_s": timings,
        "n_per_model_total": n_per_model_total,
        "total_combos": total_combos,
        "estimate_total_s": estimate_s,
    }


def run_sensitivity(csv_path: Path = OUT_DIR / "sensitivity_runs.csv", timing_gate_hours: float = 2.0) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    get_dataset = _dataset_cache()

    print("=== Step 1: baseline reproduction check (90 fits) ===")
    baseline_df = run_baseline_check(get_dataset, csv_path)
    ok = baseline_df[baseline_df["status"] == "ok"]
    max_diff = float(ok["abs_diff_vs_raw"].max()) if len(ok) else float("nan")
    n_above_1e4 = int((ok["abs_diff_vs_raw"] > 1e-4).sum())
    n_above_5e3 = int((ok["abs_diff_vs_raw"] > 5e-3).sum())
    n_failed = int((baseline_df["status"] != "ok").sum())
    print(f"max abs diff vs results/raw: {max_diff}")
    print(f"fits with diff > 1e-4: {n_above_1e4} / {len(baseline_df)}")
    print(f"fits with diff > 5e-3: {n_above_5e3} / {len(baseline_df)}")
    print(f"failed baseline fits: {n_failed}")
    if n_above_5e3 > 0.10 * len(baseline_df):
        print("STOP: more than 10% of baseline fits differ from results/raw by more than 5e-3.")
        print("This means the sensitivity harness does not replicate the runner. Aborting.")
        sys.exit(1)

    print("=== Step 2: timing gate (6 A_long probe fits) ===")
    gate = run_timing_gate(get_dataset)
    print(json.dumps(gate, indent=2, default=str))
    estimate_hours = gate["estimate_total_s"] / 3600.0
    print(f"Estimated total runtime for {gate['total_combos']} fits: {estimate_hours:.2f} hours")
    if estimate_hours > timing_gate_hours:
        print(f"STOP: estimated runtime exceeds the {timing_gate_hours}h gate. Not running the full sensitivity check.")
        sys.exit(1)

    print("=== Step 3: full sensitivity run ===")
    existing = _existing_sensitivity_keys(csv_path)
    combos = all_sensitivity_combos()
    for i, (dataset, model_name, split_idx, fraction, variant_name) in enumerate(combos, start=1):
        key = (dataset, model_name, split_idx, fraction, variant_name)
        if key in existing:
            continue
        row = _fit_one(dataset, model_name, split_idx, fraction, variant_name, VARIANTS[variant_name], get_dataset)
        _append_sensitivity_row(csv_path, row)
        print(f"[{i}/{len(combos)}] {key} -> status={row['status']} auroc={row.get('auroc')}")
    print("Sensitivity run complete.")


# ---------------------------------------------------------------------------
# 3d. xgb-wide-range (descriptive only; does not change src/models.py)
# ---------------------------------------------------------------------------


def _suggest_xgb_wide(trial: Any) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10, log=True),
    }


def _tune_with_custom_suggester(
    model_name: str, suggest_fn, X_train: pd.DataFrame, y_train: pd.Series,
    numeric_cols: List[str], categorical_cols: List[str], seed: int,
    n_trials: int = 20, inner_splits: int = 3,
) -> Dict[str, Any]:
    """A copy of src.tuning.tune_model's objective loop with a custom suggest_fn.

    Kept local to this script (not in src/) because it exists only to probe a
    wider xgb search space for a one-off, descriptive diagnostic.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.ERROR)
    y_values = np.asarray(y_train)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
    folds = list(inner_cv.split(X_train, y_values))

    def objective(trial: Any) -> float:
        params = suggest_fn(trial)
        fold_scores = []
        for fit_idx, val_idx in folds:
            pipeline = build_pipeline(model_name, params, numeric_cols, categorical_cols, seed)
            pipeline.fit(X_train.iloc[fit_idx], y_values[fit_idx])
            val_prob = positive_class_proba(pipeline, X_train.iloc[val_idx])
            fold_scores.append(roc_auc_score(y_values[val_idx], val_prob))
        return float(np.mean(fold_scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, catch=(Exception,))
    return dict(study.best_trial.params)


def run_xgb_wide_range(
    datasets: Sequence[str] = ("pima_diabetes", "ilpd", "heart_cleveland"),
    splits: Sequence[int] = (0, 1, 2),
) -> pd.DataFrame:
    rows = []
    for dataset in datasets:
        X, y, meta = load_dataset(dataset, verbose=False)
        outer_splits = make_splits(X, y, n_splits=OUTER_N_SPLITS, n_repeats=OUTER_N_REPEATS, seed=OUTER_SPLIT_SEED)
        for split_idx in splits:
            train_idx, test_idx = outer_splits[split_idx]
            task_seed = OUTER_SPLIT_SEED + split_idx
            wide_best = _tune_with_custom_suggester(
                "xgb", _suggest_xgb_wide, X.iloc[train_idx], y.iloc[train_idx],
                meta["numeric_cols"], meta["categorical_cols"], task_seed,
            )
            original_best = load_tuned_params(dataset, "xgb", split_idx)
            original_raw = load_raw(dataset, "xgb", split_idx)

            for fraction in (0.1, 1.0):
                fit_params = effective_params("xgb", wide_best, fraction)
                sub_idx = subsample_train(train_idx, y, fraction, task_seed)
                pipeline = build_pipeline("xgb", fit_params, meta["numeric_cols"], meta["categorical_cols"], task_seed)
                pipeline.fit(X.iloc[sub_idx], y.iloc[sub_idx])
                y_prob = positive_class_proba(pipeline, X.iloc[test_idx])
                wide_auroc = roc_auc_score(y.iloc[test_idx].to_numpy(), y_prob)
                original_auroc = float(original_raw.loc[original_raw["fraction"] == fraction, "auroc"].iloc[0])
                rows.append(
                    {
                        "dataset": dataset, "split_idx": split_idx, "fraction": fraction,
                        "original_min_child_weight": original_best.get("min_child_weight"),
                        "wide_min_child_weight": wide_best.get("min_child_weight"),
                        "original_auroc": original_auroc,
                        "wide_auroc": wide_auroc,
                    }
                )
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "xgb_wide_range.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Sensitivity reporting (Tables S1-S5): reads results/phase2c/sensitivity_runs.csv,
# never re-fits anything. Not one of the four subcommands named in the brief, but
# added so the report tables are reproducible without re-running the 360 fits.
# ---------------------------------------------------------------------------


def build_sensitivity_report(csv_path: Path = OUT_DIR / "sensitivity_runs.csv") -> Dict[str, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df["constant_prediction"] = df["n_unique_probs"] == 1

    # S1: baseline reproduction check.
    baseline = df[df["variant"] == "baseline"].copy()
    diffs = []
    for _, r in baseline.iterrows():
        if r["status"] != "ok":
            diffs.append(np.nan)
            continue
        raw = load_raw(r["dataset"], r["model"], int(r["split_idx"]))
        ref = raw.loc[raw["fraction"] == r["fraction"], "auroc"]
        diffs.append(abs(float(ref.iloc[0]) - float(r["auroc"])) if len(ref) else np.nan)
    baseline["abs_diff_vs_raw"] = diffs
    ok = baseline[baseline["status"] == "ok"]
    s1 = pd.DataFrame(
        [
            {
                "max_abs_diff": float(ok["abs_diff_vs_raw"].max()) if len(ok) else float("nan"),
                "n_above_1e4": int((ok["abs_diff_vs_raw"] > 1e-4).sum()),
                "n_above_5e3": int((ok["abs_diff_vs_raw"] > 5e-3).sum()),
                "n_failed": int((baseline["status"] != "ok").sum()),
                "n_total": len(baseline),
            }
        ]
    )

    ok_df = df[df["status"] == "ok"].copy()

    # S2: dataset-averaged-first, then averaged-over-datasets, mean per model/fraction/variant,
    # plus the difference to baseline.
    per_dataset_mean = ok_df.groupby(["model", "fraction", "variant", "dataset"])[["auroc", "log_loss", "brier"]].mean()
    s2 = per_dataset_mean.groupby(["model", "fraction", "variant"]).mean().reset_index()
    baseline_s2 = s2[s2["variant"] == "baseline"][["model", "fraction", "auroc", "log_loss", "brier"]]
    baseline_s2 = baseline_s2.rename(columns={"auroc": "auroc_baseline", "log_loss": "log_loss_baseline", "brier": "brier_baseline"})
    s2 = s2.merge(baseline_s2, on=["model", "fraction"], how="left")
    s2["auroc_diff_vs_baseline"] = s2["auroc"] - s2["auroc_baseline"]
    s2["log_loss_diff_vs_baseline"] = s2["log_loss"] - s2["log_loss_baseline"]
    s2["brier_diff_vs_baseline"] = s2["brier"] - s2["brier_baseline"]

    # S3: per dataset/model/fraction, baseline vs each variant, mean + std over the 3 splits.
    s3 = ok_df.groupby(["dataset", "model", "fraction", "variant"])["auroc"].agg(["mean", "std"]).reset_index()
    s3 = s3.rename(columns={"mean": "auroc_mean", "std": "auroc_std_over_splits"})

    # S4: schedule/timing behaviour per model, fraction, variant.
    s4 = ok_df.groupby(["model", "fraction", "variant"]).agg(
        frac_early_stopping=("early_stopping", "mean"),
        frac_hit_cap=("hit_cap", "mean"),
        mean_epochs_run=("epochs_run", "mean"),
        mean_n_val=("n_val", "mean"),
        mean_fit_time_s=("fit_time_s", "mean"),
    ).reset_index()

    # S5: failure/NaN/constant-prediction counts per variant and model.
    s5 = df.groupby(["model", "variant"]).agg(
        n_failed=("status", lambda s: int((s != "ok").sum())),
        n_nan_loss=("nan_loss", lambda s: int(s.fillna(False).astype(bool).sum())),
        n_constant_prediction=("constant_prediction", lambda s: int(s.fillna(False).astype(bool).sum())),
        n_total=("status", "size"),
    ).reset_index()

    tables = {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S5": s5}
    for name, table in tables.items():
        table.to_csv(OUT_DIR / f"sensitivity_report_{name}.csv", index=False)
    return tables


def print_sensitivity_flags(tables: Dict[str, pd.DataFrame]) -> None:
    s2 = tables["S2"]
    print("--- FLAG: variants changing dataset-averaged AUROC by > 0.01 vs baseline ---")
    flagged = s2[(s2["variant"] != "baseline") & (s2["auroc_diff_vs_baseline"].abs() > 0.01)]
    print(flagged[["model", "fraction", "variant", "auroc", "auroc_baseline", "auroc_diff_vs_baseline"]].to_string(index=False)
          if len(flagged) else "(none)")

    s3 = tables["S3"]
    print("--- FLAG: TabNet/ckd fraction=1.0 collapse pattern across variants ---")
    ckd_tabnet = s3[(s3["dataset"] == "ckd") & (s3["model"] == "tabnet") & (s3["fraction"] == 1.0)]
    print(ckd_tabnet.to_string(index=False) if len(ckd_tabnet) else "(no ckd/tabnet/1.0 rows)")

    s4 = tables["S4"]
    print("--- FLAG: variants with > 20% of fits hitting the cap ---")
    flagged_cap = s4[s4["frac_hit_cap"] > 0.20]
    print(flagged_cap.to_string(index=False) if len(flagged_cap) else "(none)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("endpoints")
    sub.add_parser("tabnet")
    sub.add_parser("sensitivity")
    sub.add_parser("sensitivity-report")
    sub.add_parser("xgb-wide-range")
    sub.add_parser("all")
    args = parser.parse_args()

    print(f"sys.executable: {sys.executable}")

    if args.command in ("endpoints", "all"):
        overall, _ = run_endpoints()
        print(overall.to_string(index=False))
    if args.command in ("tabnet", "all"):
        per_split, collapsed = run_tabnet_diagnostics()
        print(f"collapsed splits found: {len(collapsed)}")
        print(collapsed.to_string(index=False) if len(collapsed) else "(none)")
    if args.command == "sensitivity":
        run_sensitivity()
    if args.command == "sensitivity-report":
        tables = build_sensitivity_report()
        for name, table in tables.items():
            print(f"--- Table {name} ---")
            print(table.to_string(index=False))
        print_sensitivity_flags(tables)
    if args.command == "xgb-wide-range":
        df = run_xgb_wide_range()
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()

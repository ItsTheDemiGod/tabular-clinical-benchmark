"""Tests for src.deep_models (mlp, ftt, tabnet) and their integration into the harness.

Requires torch + CUDA (this project's deep models have no CPU fallback by
design). MAX_STEPS/PATIENCE_STEPS/FIXED_STEPS_NO_VALIDATION are monkeypatched
down for every test in this file (see `_small_training_budget` below) so the
suite stays fast; the exact production defaults are exercised separately by
`test_training_schedule_worked_examples`, which passes them explicitly.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.deep_models as deep_models
import src.runner as runner_module
from src.data import load_dataset
from src.deep_models import (
    FTTransformerClassifier,
    MLPClassifier,
    TabNetClassifierWrapper,
    split_validation,
    training_schedule,
)
from src.metrics import prediction_diversity
from src.models import MODEL_REGISTRY, build_pipeline, positive_class_proba
from src.runner import RAW_COLUMNS, run_task, task_stem
from src.tuning import tune_model
from src.utils import make_splits, subsample_train

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_NAMES = ["pima_diabetes", "heart_cleveland", "ckd", "ilpd", "breast_wisconsin"]
DEEP_MODEL_NAMES = ["mlp", "ftt", "tabnet"]

DEEP_MIDPOINT_PARAMS = {
    "mlp": {"n_layers": 2, "hidden": 64, "dropout": 0.25, "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 32},
    "ftt": {"d_token": 32, "n_blocks": 2, "dropout": 0.2, "lr": 5.5e-4, "weight_decay": 1e-4, "batch_size": 32},
    "tabnet": {"n_d": 16, "n_steps": 4, "gamma": 1.5, "lambda_sparse": 3.2e-4, "lr": 5.5e-3, "batch_size": 32},
}


@pytest.fixture(autouse=True)
def _small_training_budget(monkeypatch):
    """Shrink the gradient-step budget so the GPU suite runs quickly."""
    monkeypatch.setattr(deep_models, "MAX_STEPS", 30)
    monkeypatch.setattr(deep_models, "PATIENCE_STEPS", 10)
    monkeypatch.setattr(deep_models, "FIXED_STEPS_NO_VALIDATION", 20)


def _first_split(name):
    X, y, meta = load_dataset(name)
    train_idx, test_idx = make_splits(X, y, n_splits=5, n_repeats=2, seed=42)[0]
    return X, y, meta, train_idx, test_idx


def _pima_train():
    X, y, meta, train_idx, _ = _first_split("pima_diabetes")
    return X.iloc[train_idx], y.iloc[train_idx], meta


# ---------------------------------------------------------------------------
# split_validation
# ---------------------------------------------------------------------------


def test_split_validation_partitions_without_overlap_and_has_min_counts():
    y = np.array([0] * 20 + [1] * 15)
    train_idx, val_idx = split_validation(y, seed=1)
    assert set(train_idx.tolist()) & set(val_idx.tolist()) == set()
    assert set(train_idx.tolist()) | set(val_idx.tolist()) == set(range(len(y)))
    for cls in (0, 1):
        assert (y[val_idx] == cls).sum() >= 2
        assert (y[train_idx] == cls).sum() >= 2


def test_split_validation_deterministic_for_same_seed():
    y = np.array([0] * 20 + [1] * 15)
    a = split_validation(y, seed=5)
    b = split_validation(y, seed=5)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_split_validation_none_when_a_class_has_fewer_than_four():
    assert split_validation(np.array([0] * 3 + [1] * 20), seed=0) is None
    assert split_validation(np.array([0] * 4 + [1] * 20), seed=0) is not None


# ---------------------------------------------------------------------------
# training_schedule
# ---------------------------------------------------------------------------


def test_training_schedule_worked_examples():
    # Worked by hand against the production defaults (max_steps=600, patience_steps=60).
    a = training_schedule(n_train_part=20, batch_size=64)
    assert a == {"effective_batch": 8, "steps_per_epoch": 3, "max_epochs": 200, "patience_epochs": 20}

    b = training_schedule(n_train_part=100, batch_size=16)
    assert b == {"effective_batch": 16, "steps_per_epoch": 7, "max_epochs": 86, "patience_epochs": 9}

    c = training_schedule(n_train_part=8, batch_size=32)
    assert c == {"effective_batch": 8, "steps_per_epoch": 1, "max_epochs": 500, "patience_epochs": 60}


def test_training_schedule_effective_batch_bounds():
    for n_train_part in (4, 8, 20, 100, 1000):
        for batch_size in (16, 32, 64, 128):
            schedule = training_schedule(n_train_part, batch_size)
            assert schedule["effective_batch"] <= batch_size
            if n_train_part >= 32:
                assert schedule["effective_batch"] >= 8
            assert schedule["patience_epochs"] >= 5

    assert training_schedule(20, 64)["effective_batch"] == 8


def test_training_schedule_epoch_budget_close_to_max_steps():
    for n_train_part in (8, 20, 57, 123, 400):
        for batch_size in (16, 32, 64, 128):
            schedule = training_schedule(n_train_part, batch_size)
            total_possible_steps = schedule["max_epochs"] * schedule["steps_per_epoch"]
            if schedule["max_epochs"] < 500:
                assert abs(total_possible_steps - 600) <= schedule["steps_per_epoch"]


# ---------------------------------------------------------------------------
# Early stopping / no-validation / NaN handling
# ---------------------------------------------------------------------------


def test_early_stopping_restores_the_best_epoch_weights():
    rng = np.random.RandomState(0)
    n = 200
    X = rng.randn(n, 4).astype(np.float32)
    y = ((X[:, 0] + 0.5 * X[:, 1]) > 0).astype(np.float64)
    seed = 3

    clf = MLPClassifier(n_layers=1, hidden=8, dropout=0.0, lr=1e-2, weight_decay=0.0, batch_size=16, seed=seed)
    clf.fit(X, y)
    info = clf.training_info_
    assert info["early_stopping"] is True
    assert info["best_val_logloss"] is not None

    import torch

    train_idx, val_idx = split_validation(y, seed)
    model = clf._get_eval_model()
    X_val_t = torch.as_tensor(X[val_idx], dtype=torch.float32, device="cuda")
    y_val_t = torch.as_tensor(y[val_idx], dtype=torch.float32, device="cuda")
    with torch.no_grad():
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(X_val_t), y_val_t).item()
    assert abs(loss - info["best_val_logloss"]) < 1e-6


def test_no_validation_split_trains_fixed_steps_without_early_stopping():
    rng = np.random.RandomState(1)
    X = np.vstack([rng.randn(30, 3), rng.randn(3, 3) + 5]).astype(np.float32)
    y = np.array([0] * 30 + [1] * 3, dtype=np.float64)  # class 1 has only 3 -> no validation split

    clf = MLPClassifier(n_layers=1, hidden=8, batch_size=8, seed=0)
    clf.fit(X, y)
    info = clf.training_info_

    assert info["early_stopping"] is False
    assert info["best_val_logloss"] is None
    steps_per_epoch = math.ceil(info["n_train_part"] / info["effective_batch"])
    total_possible_steps = steps_per_epoch * info["epochs_run"]
    assert total_possible_steps >= deep_models.FIXED_STEPS_NO_VALIDATION
    assert total_possible_steps - steps_per_epoch < deep_models.FIXED_STEPS_NO_VALIDATION


def test_nan_loss_is_recorded_and_never_yields_nan_probabilities():
    # A literal infinite learning rate makes the very first AdamW step produce
    # non-finite weights deterministically (empirically, large-but-finite lr
    # values like 1e8-1e18 never go non-finite within this tiny step budget,
    # while anything from roughly 1e20 up does on step one -- there is no
    # middle regime here to exercise the "nan mid-training, keep earlier
    # weights" branch, so this test covers the other explicitly allowed
    # outcome: no finite weights were ever produced, and fit() raises).
    X = np.random.RandomState(2).randn(50, 3).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float64)
    clf = MLPClassifier(n_layers=1, hidden=8, lr=math.inf, weight_decay=0.0, batch_size=8, seed=0)
    try:
        clf.fit(X, y)
    except RuntimeError as exc:
        assert "non-finite" in str(exc)
        return
    assert clf.training_info_["nan_loss"] is True
    proba = clf.predict_proba(X)
    assert not np.isnan(proba).any()


# ---------------------------------------------------------------------------
# All three wrappers: fit/predict on every dataset, tiny-data, pickling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", DATASET_NAMES)
@pytest.mark.parametrize("model", DEEP_MODEL_NAMES)
def test_deep_pipeline_fits_and_predicts(dataset, model):
    X, y, meta, train_idx, test_idx = _first_split(dataset)
    pipeline = build_pipeline(
        model, DEEP_MIDPOINT_PARAMS[model], meta["numeric_cols"], meta["categorical_cols"], seed=0
    )
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    proba = pipeline.predict_proba(X.iloc[test_idx])

    assert proba.shape == (len(test_idx), 2)
    assert not np.isnan(proba).any()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    input_width = pipeline.named_steps["prep"].transform(X.iloc[train_idx[:1]]).shape[1]
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "input_dim_"):
        assert clf.input_dim_ == input_width


@pytest.mark.parametrize("model", DEEP_MODEL_NAMES)
def test_tiny_data_fit_on_heart_cleveland_ten_percent(model):
    X, y, meta, train_idx, test_idx = _first_split("heart_cleveland")
    sub_idx = subsample_train(train_idx, y, 0.1, seed=42)
    pipeline = build_pipeline(
        model, DEEP_MIDPOINT_PARAMS[model], meta["numeric_cols"], meta["categorical_cols"], seed=0
    )
    pipeline.fit(X.iloc[sub_idx], y.iloc[sub_idx])
    proba = positive_class_proba(pipeline, X.iloc[test_idx])
    assert not np.isnan(proba).any()
    n_unique, constant = prediction_diversity(proba)
    print(f"{model} on {len(sub_idx)}-row subsample: n_unique_probs={n_unique}, constant={constant}")


@pytest.mark.parametrize("model", DEEP_MODEL_NAMES)
def test_pickling_round_trip_reproduces_predict_proba(model, tmp_path):
    X, y, meta, train_idx, test_idx = _first_split("pima_diabetes")
    pipeline = build_pipeline(
        model, DEEP_MIDPOINT_PARAMS[model], meta["numeric_cols"], meta["categorical_cols"], seed=0
    )
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    proba_before = pipeline.predict_proba(X.iloc[test_idx])

    path = tmp_path / f"{model}.joblib"
    joblib.dump(pipeline, path, compress=3)
    print(f"{model} pickle size: {path.stat().st_size} bytes")

    reloaded = joblib.load(path)
    proba_after = reloaded.predict_proba(X.iloc[test_idx])
    assert np.allclose(proba_before, proba_after, atol=1e-4)


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", DEEP_MODEL_NAMES)
def test_tune_model_returns_params_matching_the_search_space(model):
    X_train, y_train, meta = _pima_train()
    best_params, info = tune_model(
        model, X_train, y_train, meta["numeric_cols"], meta["categorical_cols"], seed=1, n_trials=3
    )
    suggested = MODEL_REGISTRY[model].suggest_params(optuna.trial.FixedTrial(best_params))
    assert set(best_params) == set(suggested)
    assert info["n_trials"] == 3


def test_tunable_false_registry_entry_skips_optuna(monkeypatch):
    import dataclasses

    import src.tuning as tuning_module

    spec = dataclasses.replace(MODEL_REGISTRY["logreg"], tunable=False)
    monkeypatch.setitem(MODEL_REGISTRY, "logreg", spec)

    def boom(*args, **kwargs):
        raise AssertionError("optuna.create_study should not be called for a non-tunable model")

    monkeypatch.setattr(tuning_module.optuna, "create_study", boom)

    X_train, y_train, meta = _pima_train()
    best_params, info = tune_model(
        "logreg", X_train, y_train, meta["numeric_cols"], meta["categorical_cols"], seed=1, n_trials=5
    )
    assert best_params == {}
    assert info == {
        "n_trials": 0, "n_failed_trials": 0, "best_value": None,
        "tuning_time_s": 0.0, "failed_trial_errors": [],
    }


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def test_mlp_leakage_corrupted_test_fold_does_not_change_training(tmp_path, monkeypatch):
    dataset, model = "pima_diabetes", "mlp"
    args = {"dataset": dataset, "model_name": model, "split_idx": 0, "n_trials": 2}
    stem = task_stem(dataset, model, 0)
    clean_dir, corrupted_dir = tmp_path / "clean", tmp_path / "corrupted"

    run_task(results_dir=clean_dir, **args)

    def load_corrupted(name, verbose=True):
        X, y, meta = load_dataset(name, verbose=verbose)
        _, test_idx = make_splits(X, y, n_splits=5, n_repeats=2, seed=42)[0]
        X = X.copy()
        for col in meta["numeric_cols"]:
            X.iloc[test_idx, X.columns.get_loc(col)] = 1e9
        return X, y, meta

    monkeypatch.setattr(runner_module, "load_dataset", load_corrupted)
    run_task(results_dir=corrupted_dir, **args)
    monkeypatch.undo()

    def read_tuning(root):
        with open(root / "tuning" / f"{stem}.json", encoding="utf-8") as f:
            return json.load(f)

    assert read_tuning(clean_dir)["best_params"] == read_tuning(corrupted_dir)["best_params"]

    X, y, _ = load_dataset(dataset)
    train_idx, _ = make_splits(X, y, n_splits=5, n_repeats=2, seed=42)[0]
    clean_model = joblib.load(clean_dir / "models" / f"{stem}.joblib")
    corrupted_model = joblib.load(corrupted_dir / "models" / f"{stem}.joblib")
    clean_pred = positive_class_proba(clean_model, X.iloc[train_idx])
    corrupted_pred = positive_class_proba(corrupted_model, X.iloc[train_idx])
    # GPU float ops are not bit-exact across runs the way the classical (CPU,
    # tree/closed-form) models are, so this allows a small tolerance.
    assert np.allclose(clean_pred, corrupted_pred, atol=1e-5)

    with np.load(clean_dir / "predictions" / f"{stem}.npz") as clean_p, np.load(
        corrupted_dir / "predictions" / f"{stem}.npz"
    ) as corrupted_p:
        assert not np.array_equal(clean_p["prob_1.0"], corrupted_p["prob_1.0"])


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", DEEP_MODEL_NAMES)
def test_deep_model_runner_outputs_and_resume(model, tmp_path, capsys):
    stem = task_stem("pima_diabetes", model, 0)
    args = {"dataset": "pima_diabetes", "model_name": model, "split_idx": 0, "n_trials": 2}

    run_task(results_dir=tmp_path, **args)
    rows = pd.read_csv(tmp_path / "raw" / f"{stem}.csv")
    assert list(rows.columns) == RAW_COLUMNS
    assert len(rows) == 4
    assert (rows["status"] == "ok").all()

    with open(tmp_path / "diagnostics" / f"{stem}.json", encoding="utf-8") as f:
        diagnostics = json.load(f)
    assert set(diagnostics) == {"0.1", "0.25", "0.5", "1.0"}
    for entry in diagnostics.values():
        assert "n_fit" in entry

    with np.load(tmp_path / "predictions" / f"{stem}.npz") as predictions:
        assert sum(1 for k in predictions.files if k.startswith("prob_")) == 4

    assert joblib.load(tmp_path / "models" / f"{stem}.joblib") is not None

    capsys.readouterr()
    result = run_task(results_dir=tmp_path, **args)
    assert result["skipped"] is True
    assert "Skipping" in capsys.readouterr().out


@pytest.mark.parametrize("model", DEEP_MODEL_NAMES)
def test_deep_model_forced_failure_only_affects_that_fraction(model, tmp_path, monkeypatch):
    original = runner_module.subsample_train

    def fail_at_quarter(train_idx, y, fraction, seed):
        if fraction == 0.25:
            raise RuntimeError("forced failure")
        return original(train_idx, y, fraction, seed)

    monkeypatch.setattr(runner_module, "subsample_train", fail_at_quarter)
    run_task(results_dir=tmp_path, dataset="pima_diabetes", model_name=model, split_idx=0, n_trials=2)

    stem = task_stem("pima_diabetes", model, 0)
    rows = pd.read_csv(tmp_path / "raw" / f"{stem}.csv").set_index("fraction")
    assert rows.loc[0.25, "status"] == "failed"
    for fraction in (0.1, 0.5, 1.0):
        assert rows.loc[fraction, "status"] == "ok"

    with open(tmp_path / "diagnostics" / f"{stem}.json", encoding="utf-8") as f:
        diagnostics = json.load(f)
    assert diagnostics["0.25"] == {"error": rows.loc[0.25, "error"]}


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", DEEP_MODEL_NAMES)
def test_deep_model_determinism(model, tmp_path):
    args = {"dataset": "pima_diabetes", "model_name": model, "split_idx": 0, "n_trials": 2}
    stem = task_stem("pima_diabetes", model, 0)
    run_task(results_dir=tmp_path / "a", **args)
    run_task(results_dir=tmp_path / "b", **args)
    a = pd.read_csv(tmp_path / "a" / "raw" / f"{stem}.csv").set_index("fraction")["auroc"]
    b = pd.read_csv(tmp_path / "b" / "raw" / f"{stem}.csv").set_index("fraction")["auroc"]
    max_diff = float((a - b).abs().max())
    print(f"{model} determinism: max AUROC diff across two runs = {max_diff:.2e}")
    assert max_diff < 1e-6, f"{model}: max AUROC diff {max_diff:.2e} exceeds the 1e-6 target"


# ---------------------------------------------------------------------------
# Classical regression (must still hold after the Phase 2b registry/runner changes)
# ---------------------------------------------------------------------------


def test_classical_regression_unaffected_by_phase2b_changes(tmp_path):
    run_task(results_dir=tmp_path, dataset="pima_diabetes", model_name="logreg", split_idx=0, n_trials=20)
    stem = task_stem("pima_diabetes", "logreg", 0)
    rows = pd.read_csv(tmp_path / "raw" / f"{stem}.csv").set_index("fraction")
    baseline = pd.read_csv(PROJECT_ROOT / "results" / "raw" / f"{stem}.csv").set_index("fraction")
    for fraction in (0.1, 0.25, 0.5, 1.0):
        assert rows.loc[fraction, "auroc"] == pytest.approx(baseline.loc[fraction, "auroc"], abs=1e-9)


# ---------------------------------------------------------------------------
# Torch-free classical path
# ---------------------------------------------------------------------------


def test_classical_path_does_not_import_torch_in_a_fresh_subprocess():
    script = (
        "import sys\n"
        f"sys.path.insert(0, r'{PROJECT_ROOT}')\n"
        "from src.models import build_pipeline\n"
        "from src.data import load_dataset\n"
        "from src.utils import make_splits\n"
        "X, y, meta = load_dataset('pima_diabetes', verbose=False)\n"
        "train_idx, test_idx = make_splits(X, y, n_splits=5, n_repeats=2, seed=42)[0]\n"
        "for name in ('logreg', 'rf', 'xgb', 'lgbm'):\n"
        "    pipeline = build_pipeline(name, {}, meta['numeric_cols'], meta['categorical_cols'], seed=0)\n"
        "    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])\n"
        "assert 'torch' not in sys.modules, 'torch was imported by the classical-only path'\n"
        "print('OK')\n"
    )
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout

"""Tests for src.runner: outputs, resume, failure handling, leakage and determinism."""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.runner
from src.data import load_dataset
from src.models import positive_class_proba
from src.runner import RAW_COLUMNS, run_task, task_stem
from src.utils import make_splits

STEM = task_stem("pima_diabetes", "logreg", 0)
ARGS = {"dataset": "pima_diabetes", "model_name": "logreg", "split_idx": 0, "n_trials": 2}


def _read_rows(results_dir, stem=STEM):
    return pd.read_csv(Path(results_dir) / "raw" / f"{stem}.csv")


def _hash_all_files(results_dir):
    return {
        str(path.relative_to(results_dir)): (
            hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns,
        )
        for path in Path(results_dir).rglob("*")
        if path.is_file()
    }


@pytest.fixture(scope="module")
def finished_task(tmp_path_factory):
    """One completed pima_diabetes + logreg + split 0 task, shared by the output tests."""
    results_dir = tmp_path_factory.mktemp("results")
    run_task(results_dir=results_dir, **ARGS)
    return results_dir


def test_produces_four_ok_rows_with_expected_columns(finished_task):
    rows = _read_rows(finished_task)
    assert list(rows.columns) == [
        "dataset", "model", "split_idx", "fraction", "seed", "n_train", "n_test",
        "n_features_encoded", "auroc", "f1", "balanced_accuracy", "brier", "log_loss", "ece",
        "fit_time_s", "predict_time_s", "n_unique_probs", "constant_prediction",
        "status", "error", "params_json",
    ]
    assert len(rows) == 4
    assert (rows["status"] == "ok").all()
    assert rows["error"].isna().all()
    assert rows["fraction"].tolist() == [0.1, 0.25, 0.5, 1.0]
    assert rows["auroc"].notna().all()
    assert (rows["n_features_encoded"] == 8).all()
    assert (rows["fit_time_s"] > 0).all()
    assert (rows["predict_time_s"] > 0).all()


def test_npz_has_four_probability_arrays(finished_task):
    rows = _read_rows(finished_task)
    with np.load(finished_task / "predictions" / f"{STEM}.npz") as predictions:
        assert set(predictions.files) == {
            "test_idx", "y_true", "prob_0.1", "prob_0.25", "prob_0.5", "prob_1.0",
        }
        n_test = len(predictions["test_idx"])
        assert n_test == rows["n_test"].iloc[0]
        for key in ("prob_0.1", "prob_0.25", "prob_0.5", "prob_1.0"):
            assert predictions[key].shape == (n_test,)
            assert not np.isnan(predictions[key]).any()


def test_model_loads_and_reproduces_saved_predictions(finished_task):
    X, y, _ = load_dataset("pima_diabetes")
    model = joblib.load(finished_task / "models" / f"{STEM}.joblib")
    with np.load(finished_task / "predictions" / f"{STEM}.npz") as predictions:
        reloaded = positive_class_proba(model, X.iloc[predictions["test_idx"]])
        assert np.array_equal(reloaded, predictions["prob_1.0"])


def test_tuning_json_has_params_and_info(finished_task):
    with open(finished_task / "tuning" / f"{STEM}.json", encoding="utf-8") as f:
        record = json.load(f)
    assert set(record["best_params"]) == {"C"}
    for key in ("n_trials", "n_failed_trials", "best_value", "tuning_time_s"):
        assert key in record["info"]


def test_smaller_fraction_has_fewer_training_rows(finished_task):
    rows = _read_rows(finished_task).set_index("fraction")
    assert rows.loc[0.1, "n_train"] < rows.loc[1.0, "n_train"]


def test_same_params_are_used_for_every_fraction(finished_task):
    tuned = [json.loads(value)["tuned"] for value in _read_rows(finished_task)["params_json"]]
    assert len(tuned) == 4
    assert all(params == tuned[0] for params in tuned)


def test_rerun_skips_and_modifies_nothing(finished_task, capsys):
    before = _hash_all_files(finished_task)
    capsys.readouterr()

    result = run_task(results_dir=finished_task, **ARGS)

    assert "Skipping" in capsys.readouterr().out
    assert result["skipped"] is True
    assert _hash_all_files(finished_task) == before


def test_incomplete_task_is_not_skipped(tmp_path, monkeypatch):
    """A task with a failed row must run again on resume, not count as complete."""
    original = src.runner.compute_metrics

    def always_broken(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(src.runner, "compute_metrics", always_broken)
    run_task(results_dir=tmp_path, **ARGS)
    assert (_read_rows(tmp_path)["status"] == "failed").all()

    monkeypatch.setattr(src.runner, "compute_metrics", original)
    result = run_task(results_dir=tmp_path, **ARGS)
    assert result["skipped"] is False
    assert (_read_rows(tmp_path)["status"] == "ok").all()


def test_forced_failure_only_affects_that_fraction(tmp_path, monkeypatch):
    original = src.runner.subsample_train

    def fail_at_quarter(train_idx, y, fraction, seed):
        if fraction == 0.25:
            raise RuntimeError("forced failure")
        return original(train_idx, y, fraction, seed)

    monkeypatch.setattr(src.runner, "subsample_train", fail_at_quarter)
    run_task(results_dir=tmp_path, **ARGS)

    rows = _read_rows(tmp_path).set_index("fraction")
    assert len(rows) == 4
    assert rows.loc[0.25, "status"] == "failed"
    assert "forced failure" in rows.loc[0.25, "error"]
    for metric in ("auroc", "f1", "balanced_accuracy", "brier", "log_loss", "ece"):
        assert pd.isna(rows.loc[0.25, metric])
    for fraction in (0.1, 0.5, 1.0):
        assert rows.loc[fraction, "status"] == "ok"
        assert pd.notna(rows.loc[fraction, "auroc"])

    # A failed fraction still has its array, filled with NaN.
    with np.load(tmp_path / "predictions" / f"{STEM}.npz") as predictions:
        assert np.isnan(predictions["prob_0.25"]).all()
        assert not np.isnan(predictions["prob_0.5"]).any()


def test_failure_in_metrics_leaves_metric_columns_empty(tmp_path, monkeypatch):
    def always_broken(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(src.runner, "compute_metrics", always_broken)
    run_task(results_dir=tmp_path, **ARGS)
    rows = _read_rows(tmp_path)
    assert rows[["auroc", "f1", "ece"]].isna().all().all()


def test_failed_tuning_gives_failed_rows_instead_of_skipping(tmp_path, monkeypatch):
    def broken_tuning(*args, **kwargs):
        raise RuntimeError("forced tuning failure")

    monkeypatch.setattr(src.runner, "tune_model", broken_tuning)
    result = run_task(results_dir=tmp_path, **ARGS)

    assert result["n_failed"] == 4
    rows = _read_rows(tmp_path)
    assert len(rows) == 4
    assert (rows["status"] == "failed").all()
    assert rows["error"].str.contains("forced tuning failure").all()


def test_save_models_false_writes_no_model(tmp_path):
    run_task(results_dir=tmp_path, save_models=False, fractions=(0.5, 1.0), **ARGS)
    assert not (tmp_path / "models" / f"{STEM}.joblib").exists()


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------


def _corrupted_loader(dataset):
    """Return a load_dataset replacement with extreme values in the first split's test fold."""

    def load_corrupted(name, verbose=True):
        X, y, meta = load_dataset(name, verbose=verbose)
        _, test_idx = make_splits(X, y, n_splits=5, n_repeats=2, seed=42)[0]
        X = X.copy()
        for col in meta["numeric_cols"]:
            X.iloc[test_idx, X.columns.get_loc(col)] = 1e9
        for col in meta["categorical_cols"]:
            X.iloc[test_idx, X.columns.get_loc(col)] = "CORRUPTED"
        return X, y, meta

    return load_corrupted


@pytest.mark.parametrize("dataset", ["pima_diabetes", "heart_cleveland"])
@pytest.mark.parametrize("model", ["logreg", "xgb"])
def test_corrupting_test_fold_changes_nothing_learned_from_training(
    dataset, model, tmp_path, monkeypatch
):
    """Run the whole task twice, once with extreme values in the test fold.

    Tuned parameters, inner-CV score and training-set predictions must be identical,
    while test predictions must differ, which shows the corruption reached the task.
    """
    args = {"dataset": dataset, "model_name": model, "split_idx": 0, "n_trials": 3}
    stem = task_stem(dataset, model, 0)
    clean_dir, corrupted_dir = tmp_path / "clean", tmp_path / "corrupted"

    run_task(results_dir=clean_dir, **args)
    monkeypatch.setattr(src.runner, "load_dataset", _corrupted_loader(dataset))
    run_task(results_dir=corrupted_dir, **args)
    monkeypatch.undo()

    def read_tuning(root):
        with open(root / "tuning" / f"{stem}.json", encoding="utf-8") as f:
            return json.load(f)

    clean_tuning, corrupted_tuning = read_tuning(clean_dir), read_tuning(corrupted_dir)
    assert clean_tuning["best_params"] == corrupted_tuning["best_params"]
    assert clean_tuning["info"]["best_value"] == corrupted_tuning["info"]["best_value"]

    X, y, _ = load_dataset(dataset)
    train_idx, _ = make_splits(X, y, n_splits=5, n_repeats=2, seed=42)[0]
    clean_model = joblib.load(clean_dir / "models" / f"{stem}.joblib")
    corrupted_model = joblib.load(corrupted_dir / "models" / f"{stem}.joblib")
    assert np.array_equal(
        positive_class_proba(clean_model, X.iloc[train_idx]),
        positive_class_proba(corrupted_model, X.iloc[train_idx]),
    )

    with np.load(clean_dir / "predictions" / f"{stem}.npz") as clean, np.load(
        corrupted_dir / "predictions" / f"{stem}.npz"
    ) as corrupted:
        assert not np.array_equal(clean["prob_1.0"], corrupted["prob_1.0"])


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["logreg", "rf", "xgb", "lgbm"])
def test_same_seed_gives_identical_auroc(model, tmp_path):
    args = {"dataset": "pima_diabetes", "model_name": model, "split_idx": 0, "n_trials": 2}
    stem = task_stem("pima_diabetes", model, 0)

    run_task(results_dir=tmp_path / "first", **args)
    run_task(results_dir=tmp_path / "second", **args)

    first, second = _read_rows(tmp_path / "first", stem), _read_rows(tmp_path / "second", stem)
    assert first["auroc"].tolist() == second["auroc"].tolist()
    assert first["params_json"].tolist() == second["params_json"].tolist()


# ---------------------------------------------------------------------------
# Phase 2a.1: new columns, params_json format and resume
# ---------------------------------------------------------------------------

XGB_STEM = task_stem("pima_diabetes", "xgb", 0)
XGB_ARGS = {"dataset": "pima_diabetes", "model_name": "xgb", "split_idx": 0, "n_trials": 2}
XGB_TUNED = {
    "n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.75,
    "colsample_bytree": 0.75, "reg_lambda": 1.0, "min_child_weight": 8,
}


def _fixed_tuning(*args, **kwargs):
    info = {
        "n_trials": 2, "n_failed_trials": 0, "best_value": 0.5,
        "tuning_time_s": 0.0, "failed_trial_errors": [],
    }
    return dict(XGB_TUNED), info


def test_params_json_records_tuned_and_effective(tmp_path, monkeypatch):
    monkeypatch.setattr(src.runner, "tune_model", _fixed_tuning)
    run_task(results_dir=tmp_path, **XGB_ARGS)

    rows = _read_rows(tmp_path, XGB_STEM).set_index("fraction")
    assert {"n_unique_probs", "constant_prediction"}.issubset(rows.columns)

    for fraction, value in rows["params_json"].items():
        parsed = json.loads(value)
        assert set(parsed) == {"tuned", "effective", "fraction"}
        assert parsed["fraction"] == fraction
        assert parsed["tuned"] == XGB_TUNED

    small = json.loads(rows.loc[0.1, "params_json"])
    assert small["effective"] != small["tuned"]
    assert small["effective"]["min_child_weight"] == 1.0

    full = json.loads(rows.loc[1.0, "params_json"])
    assert full["effective"] == full["tuned"]

    # The tuning file keeps the tuned value, not a scaled one.
    with open(tmp_path / "tuning" / f"{XGB_STEM}.json", encoding="utf-8") as f:
        assert json.load(f)["best_params"] == XGB_TUNED


def test_model_is_fit_with_the_effective_params(tmp_path, monkeypatch):
    """The value that reached the model matters, not only the value that was recorded."""
    fitted_with = {}
    original = src.runner.build_pipeline

    def recording_build_pipeline(model_name, params, numeric_cols, categorical_cols, seed):
        fitted_with[len(fitted_with)] = params["min_child_weight"]
        return original(model_name, params, numeric_cols, categorical_cols, seed)

    monkeypatch.setattr(src.runner, "tune_model", _fixed_tuning)
    monkeypatch.setattr(src.runner, "build_pipeline", recording_build_pipeline)
    run_task(results_dir=tmp_path, **XGB_ARGS)

    assert list(fitted_with.values()) == [1.0, 2.0, 4.0, 8]


def test_old_format_csv_is_not_skipped(finished_task, tmp_path):
    results_dir = tmp_path / "results"
    run_task(results_dir=results_dir, fractions=(0.5, 1.0), **ARGS)
    raw_path = results_dir / "raw" / f"{STEM}.csv"
    _read_rows(results_dir).drop(columns=["n_unique_probs", "constant_prediction"]).to_csv(
        raw_path, index=False
    )

    result = run_task(results_dir=results_dir, fractions=(0.5, 1.0), **ARGS)

    assert result["skipped"] is False
    assert "n_unique_probs" in _read_rows(results_dir).columns


def test_constant_prediction_stays_ok_and_prints_a_warning(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        src.runner, "positive_class_proba", lambda pipeline, X: np.full(len(X), 0.35)
    )
    capsys.readouterr()
    run_task(results_dir=tmp_path, fractions=(0.5, 1.0), **ARGS)

    output = capsys.readouterr().out
    warnings = [line for line in output.splitlines() if line.startswith("WARNING")]
    assert len(warnings) == 2
    for text in ("dataset=pima_diabetes", "model=logreg", "split=0", "fraction=0.5"):
        assert text in warnings[0]

    rows = _read_rows(tmp_path)
    assert (rows["status"] == "ok").all()
    assert rows["constant_prediction"].all()
    assert (rows["n_unique_probs"] == 1).all()


def test_normal_row_is_not_flagged_constant(finished_task):
    rows = _read_rows(finished_task)
    assert not rows["constant_prediction"].any()
    assert (rows["n_unique_probs"] > 1).all()


def test_failed_row_leaves_new_columns_empty(tmp_path, monkeypatch):
    def always_broken(*args, **kwargs):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(src.runner, "compute_metrics", always_broken)
    run_task(results_dir=tmp_path, **ARGS)
    rows = _read_rows(tmp_path)
    assert (rows["status"] == "failed").all()
    assert rows[["n_unique_probs", "constant_prediction"]].isna().all().all()

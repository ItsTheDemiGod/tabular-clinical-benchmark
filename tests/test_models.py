"""Tests for src.models and src.tuning."""

import dataclasses
import inspect
import sys
from pathlib import Path

import numpy as np
import optuna
import pytest
import scipy.sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_dataset
from src.metrics import prediction_diversity
from src.models import MODEL_REGISTRY, build_pipeline, effective_params, positive_class_proba
from src.tuning import tune_model
from src.utils import make_splits, subsample_train

DATASET_NAMES = ["pima_diabetes", "heart_cleveland", "ckd", "ilpd", "breast_wisconsin"]
MODEL_NAMES = ["logreg", "rf", "xgb", "lgbm"]

# Midpoints of each search space: arithmetic for linear ranges, geometric for log ranges
# (rounded), and the middle option for categoricals.
MIDPOINT_PARAMS = {
    "logreg": {"C": 0.316},
    "rf": {"n_estimators": 200, "max_depth": 11, "min_samples_leaf": 5, "max_features": "log2"},
    "xgb": {
        "n_estimators": 225, "max_depth": 5, "learning_rate": 0.055, "subsample": 0.75,
        "colsample_bytree": 0.75, "reg_lambda": 0.316, "min_child_weight": 5,
    },
    "lgbm": {
        "n_estimators": 225, "num_leaves": 17, "learning_rate": 0.055, "min_child_samples": 16,
        "subsample": 0.75, "colsample_bytree": 0.75, "reg_lambda": 0.316,
    },
}


def _first_split(name):
    X, y, meta = load_dataset(name)
    train_idx, test_idx = make_splits(X, y, n_splits=5, n_repeats=2, seed=42)[0]
    return X, y, meta, train_idx, test_idx


ALL_MODEL_NAMES = MODEL_NAMES + ["mlp", "ftt", "tabnet"]


def test_registry_contents():
    # Phase 2b added mlp, ftt and tabnet (src/deep_models.py) alongside the four
    # classical models; this test now covers all seven registry entries and
    # the new `tunable` field (src/models.py ModelSpec).
    assert list(MODEL_REGISTRY) == ALL_MODEL_NAMES
    assert {name: spec.needs_scaling for name, spec in MODEL_REGISTRY.items()} == {
        "logreg": True, "rf": False, "xgb": False, "lgbm": False,
        "mlp": True, "ftt": True, "tabnet": True,
    }
    assert {name: spec.tunable for name, spec in MODEL_REGISTRY.items()} == {
        name: True for name in ALL_MODEL_NAMES
    }
    for name, spec in MODEL_REGISTRY.items():
        assert spec.name == name


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_midpoints_lie_in_the_search_space(model):
    """FixedTrial raises if a value is outside the space, or if a key is missing."""
    trial = optuna.trial.FixedTrial(MIDPOINT_PARAMS[model])
    assert MODEL_REGISTRY[model].suggest_params(trial) == MIDPOINT_PARAMS[model]


def test_model_settings_match_the_specification():
    logreg = MODEL_REGISTRY["logreg"].make_model({"C": 1.0}, 7)
    assert (logreg.penalty, logreg.solver, logreg.max_iter) == ("l2", "lbfgs", 5000)

    rf = MODEL_REGISTRY["rf"].make_model({}, 7)
    assert rf.n_jobs == 1

    xgb = MODEL_REGISTRY["xgb"].make_model({}, 7)
    assert (xgb.tree_method, xgb.eval_metric, xgb.n_jobs, xgb.verbosity) == ("hist", "logloss", 1, 0)

    lgbm = MODEL_REGISTRY["lgbm"].make_model({}, 7)
    assert (lgbm.n_jobs, lgbm.verbose, lgbm.subsample_freq) == (1, -1, 1)

    # Same seed everywhere, and no class weighting anywhere it can be set.
    for model in (logreg, rf, xgb, lgbm):
        assert model.get_params()["random_state"] == 7
    for model in (logreg, rf, lgbm):
        assert model.get_params()["class_weight"] is None
    assert xgb.get_params()["early_stopping_rounds"] is None


@pytest.mark.parametrize("dataset", DATASET_NAMES)
@pytest.mark.parametrize("model", MODEL_NAMES)
def test_pipeline_fits_and_predicts(dataset, model):
    X, y, meta, train_idx, test_idx = _first_split(dataset)
    pipeline = build_pipeline(
        model, MIDPOINT_PARAMS[model], meta["numeric_cols"], meta["categorical_cols"], seed=0
    )
    pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
    proba = pipeline.predict_proba(X.iloc[test_idx])

    assert proba.shape == (len(test_idx), 2)
    assert not np.isnan(proba).any()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    # XGBoost treats the zeros of a sparse matrix as missing, which would break the
    # rule that no model uses native missing-value handling. Models must get dense,
    # NaN-free input.
    transformed = pipeline.named_steps["prep"].transform(X.iloc[train_idx])
    assert not scipy.sparse.issparse(transformed)
    assert not np.isnan(transformed).any()


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------


def _pima_train():
    X, y, meta, train_idx, _ = _first_split("pima_diabetes")
    return X.iloc[train_idx], y.iloc[train_idx], meta


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_tune_model_returns_params_matching_the_search_space(model):
    X_train, y_train, meta = _pima_train()
    best_params, info = tune_model(
        model, X_train, y_train, meta["numeric_cols"], meta["categorical_cols"], seed=1, n_trials=3
    )

    suggested = MODEL_REGISTRY[model].suggest_params(optuna.trial.FixedTrial(best_params))
    assert set(best_params) == set(suggested)
    assert set(best_params) == set(MIDPOINT_PARAMS[model])

    assert info["n_trials"] == 3
    assert info["n_failed_trials"] == 0
    assert 0.0 <= info["best_value"] <= 1.0
    assert info["tuning_time_s"] > 0


def test_tune_model_signature_takes_no_test_data():
    parameters = list(inspect.signature(tune_model).parameters)
    assert parameters == [
        "model_name", "X_train", "y_train", "numeric_cols", "categorical_cols",
        "seed", "n_trials", "inner_splits",
    ]
    assert not any("test" in parameter for parameter in parameters)


def test_default_tuning_budget():
    parameters = inspect.signature(tune_model).parameters
    assert parameters["n_trials"].default == 20
    assert parameters["inner_splits"].default == 3


def _registry_with_suggest(monkeypatch, model, suggest_params):
    spec = dataclasses.replace(MODEL_REGISTRY[model], suggest_params=suggest_params)
    monkeypatch.setitem(MODEL_REGISTRY, model, spec)


def test_broken_trial_is_counted_and_tuning_still_returns(monkeypatch):
    original = MODEL_REGISTRY["logreg"].suggest_params

    def broken_on_second_trial(trial):
        if trial.number == 1:
            raise RuntimeError("deliberately broken trial")
        return original(trial)

    _registry_with_suggest(monkeypatch, "logreg", broken_on_second_trial)

    X_train, y_train, meta = _pima_train()
    best_params, info = tune_model(
        "logreg", X_train, y_train, meta["numeric_cols"], meta["categorical_cols"], seed=1, n_trials=3
    )

    assert info["n_trials"] == 3
    assert info["n_failed_trials"] == 1
    assert "deliberately broken trial" in info["failed_trial_errors"][0]
    assert set(best_params) == {"C"}


def test_all_trials_failing_raises(monkeypatch):
    def always_broken(trial):
        raise RuntimeError("deliberately broken trial")

    _registry_with_suggest(monkeypatch, "logreg", always_broken)

    X_train, y_train, meta = _pima_train()
    with pytest.raises(RuntimeError, match="trials failed"):
        tune_model(
            "logreg", X_train, y_train, meta["numeric_cols"], meta["categorical_cols"], seed=1, n_trials=2
        )


# ---------------------------------------------------------------------------
# Phase 2a.1: effective_params
# ---------------------------------------------------------------------------

FRACTIONS = (0.1, 0.25, 0.5, 1.0)
LEAF_SETTINGS = {
    # model: (parameter, smallest search-space value, largest, minimum after scaling)
    "xgb": ("min_child_weight", 1, 10, 1.0),
    "lgbm": ("min_child_samples", 2, 30, 1),
    "rf": ("min_samples_leaf", 1, 10, 1),
}


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_effective_params_identity_at_full_fraction(model):
    params = MIDPOINT_PARAMS[model]
    result = effective_params(model, params, 1.0)
    assert result == params
    assert result is not params
    # Exactly equal includes the type: an int must not come back as a float.
    assert {k: type(v) for k, v in result.items()} == {k: type(v) for k, v in params.items()}


@pytest.mark.parametrize(
    "model, value, fraction, expected",
    [
        ("xgb", 8, 0.1, 1.0),
        ("xgb", 10, 0.5, 5.0),
        ("lgbm", 27, 0.1, 3),
        ("lgbm", 19, 0.1, 2),
        ("rf", 10, 0.1, 1),
    ],
)
def test_effective_params_known_values(model, value, fraction, expected):
    parameter = LEAF_SETTINGS[model][0]
    params = {**MIDPOINT_PARAMS[model], parameter: value}
    assert effective_params(model, params, fraction)[parameter] == expected


def test_xgb_effective_min_child_weight_is_an_unrounded_float():
    params = {**MIDPOINT_PARAMS["xgb"], "min_child_weight": 7}
    result = effective_params("xgb", params, 0.25)["min_child_weight"]
    assert isinstance(result, float)
    assert result == 1.75


@pytest.mark.parametrize("model", ["xgb", "lgbm", "rf"])
@pytest.mark.parametrize("fraction", FRACTIONS)
def test_effective_params_never_below_minimum(model, fraction):
    parameter, low, high, minimum = LEAF_SETTINGS[model]
    for value in range(low, high + 1):
        params = {**MIDPOINT_PARAMS[model], parameter: value}
        assert effective_params(model, params, fraction)[parameter] >= minimum


@pytest.mark.parametrize("fraction", FRACTIONS)
def test_effective_params_logreg_unchanged(fraction):
    assert effective_params("logreg", {"C": 0.316}, fraction) == {"C": 0.316}


@pytest.mark.parametrize("model", MODEL_NAMES)
@pytest.mark.parametrize("fraction", FRACTIONS)
def test_effective_params_does_not_mutate_input(model, fraction):
    params = dict(MIDPOINT_PARAMS[model])
    effective_params(model, params, fraction)
    assert params == MIDPOINT_PARAMS[model]


@pytest.mark.parametrize("model", ["xgb", "lgbm", "rf"])
@pytest.mark.parametrize("fraction", FRACTIONS)
def test_effective_params_changes_only_the_leaf_setting(model, fraction):
    parameter = LEAF_SETTINGS[model][0]
    params = MIDPOINT_PARAMS[model]
    result = effective_params(model, params, fraction)
    assert set(result) == set(params)
    assert {k: v for k, v in result.items() if k != parameter} == {
        k: v for k, v in params.items() if k != parameter
    }


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_effective_params_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        effective_params("xgb", MIDPOINT_PARAMS["xgb"], fraction)


# ---------------------------------------------------------------------------
# Phase 2a.1: regression tests for the constant predictions found in Phase 2a
# ---------------------------------------------------------------------------


def _diversity_on_test_fold(dataset, model, params, fraction):
    """Fit on the subsample with exactly `params` and predict the full test fold."""
    task_seed = 42 + 0
    X, y, meta = load_dataset(dataset, verbose=False)
    train_idx, test_idx = make_splits(X, y, n_splits=5, n_repeats=2, seed=42)[0]
    sub_idx = subsample_train(train_idx, y, fraction, task_seed)
    pipeline = build_pipeline(
        model, params, meta["numeric_cols"], meta["categorical_cols"], task_seed
    )
    pipeline.fit(X.iloc[sub_idx], y.iloc[sub_idx])
    return prediction_diversity(positive_class_proba(pipeline, X.iloc[test_idx]))


@pytest.mark.parametrize(
    "dataset, model, parameter, value, fraction, unscaled_is_constant",
    [
        ("pima_diabetes", "xgb", "min_child_weight", 8, 0.1, True),
        ("heart_cleveland", "xgb", "min_child_weight", 3, 0.1, True),
        ("heart_cleveland", "lgbm", "min_child_samples", 19, 0.1, True),
        ("ilpd", "xgb", "min_child_weight", 9, 0.1, True),
        # With midpoint settings this one splits even unscaled, so it would pass without
        # the fix. The next test covers this failure with the actual tuned parameters.
        ("ilpd", "xgb", "min_child_weight", 9, 0.25, False),
        ("ilpd", "lgbm", "min_child_samples", 27, 0.1, True),
    ],
)
def test_effective_params_prevent_the_known_constant_predictions(
    dataset, model, parameter, value, fraction, unscaled_is_constant
):
    tuned = {**MIDPOINT_PARAMS[model], parameter: value}

    # Records whether the unscaled params reproduce the failure, i.e. whether this case
    # would really catch the bug.
    assert _diversity_on_test_fold(dataset, model, tuned, fraction)[1] is unscaled_is_constant

    scaled = effective_params(model, tuned, fraction)
    n_unique, constant = _diversity_on_test_fold(dataset, model, scaled, fraction)
    assert constant is False
    assert n_unique > 1


def test_effective_params_fix_ilpd_xgb_with_the_actual_tuned_params():
    """The parameters Phase 2a tuned for ilpd + xgb on split 0, which gave constant
    predictions at fraction 0.25 as well as 0.1."""
    tuned = {
        "n_estimators": 102,
        "max_depth": 4,
        "learning_rate": 0.010727906707673717,
        "subsample": 0.611761862540941,
        "colsample_bytree": 0.6890990071944861,
        "reg_lambda": 0.37943538282298367,
        "min_child_weight": 9,
    }
    for fraction in (0.1, 0.25):
        assert _diversity_on_test_fold("ilpd", "xgb", tuned, fraction)[1] is True
        scaled = effective_params("xgb", tuned, fraction)
        assert _diversity_on_test_fold("ilpd", "xgb", scaled, fraction)[1] is False

"""Tests for scripts/phase2c_diagnostics.py: endpoint classification, the
schedule-patching helper, the sensitivity runner's resume logic, and the
collapsed-split selector. Kept fast: no GPU fit, no dataset downloads."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.phase2c_diagnostics import (  # noqa: E402
    VARIANTS,
    _existing_sensitivity_keys,
    all_sensitivity_combos,
    classify_param,
    find_collapsed_splits,
    patched_schedule,
)


# ---------------------------------------------------------------------------
# Endpoint classification
# ---------------------------------------------------------------------------


def test_int_at_lower_endpoint_is_exact_equality():
    record = {"name": "n_d", "low": 8, "high": 32, "log": False, "type": "int", "choices": None}
    assert classify_param(record, 8) == "lower"
    assert classify_param(record, 32) == "upper"
    assert classify_param(record, 20) is None


def test_int_near_but_not_at_endpoint_is_none():
    record = {"name": "n_steps", "low": 2, "high": 5, "log": False, "type": "int", "choices": None}
    # 3 is close to the lower bound of 2 but not equal to it: integers require exact equality.
    assert classify_param(record, 3) is None


def test_log_uniform_within_two_percent_on_log_scale():
    record = {"name": "lr", "low": 1e-4, "high": 1e-2, "log": True, "type": "float", "choices": None}
    import math

    log_lo, log_hi = math.log(1e-4), math.log(1e-2)
    span = log_hi - log_lo
    just_inside = math.exp(log_lo + 0.019 * span)
    just_outside = math.exp(log_lo + 0.021 * span)
    middle = math.exp(log_lo + 0.5 * span)
    assert classify_param(record, just_inside) == "lower"
    assert classify_param(record, just_outside) is None
    assert classify_param(record, middle) is None
    just_inside_upper = math.exp(log_hi - 0.019 * span)
    assert classify_param(record, just_inside_upper) == "upper"


def test_linear_continuous_within_two_percent():
    record = {"name": "dropout", "low": 0.0, "high": 0.5, "log": False, "type": "float", "choices": None}
    assert classify_param(record, 0.0) == "lower"
    assert classify_param(record, 0.009) == "lower"  # 1.8% of range
    assert classify_param(record, 0.011) is None  # 2.2% of range
    assert classify_param(record, 0.25) is None
    assert classify_param(record, 0.5) == "upper"


def test_categorical_numeric_choices():
    record = {"name": "batch_size", "low": None, "high": None, "log": False, "type": "categorical",
              "choices": [16, 32, 64, 128]}
    assert classify_param(record, 16) == "lower"
    assert classify_param(record, 128) == "upper"
    assert classify_param(record, 32) is None


def test_categorical_non_numeric_choices_never_classified():
    record = {"name": "max_features", "low": None, "high": None, "log": False, "type": "categorical",
              "choices": ["sqrt", "log2", 0.5]}
    assert classify_param(record, "sqrt") is None
    assert classify_param(record, 0.5) is None


def test_lower_and_upper_are_counted_separately():
    record = {"name": "n_estimators", "low": 50, "high": 400, "log": False, "type": "int", "choices": None}
    values = [50, 50, 400, 200, 50]
    verdicts = [classify_param(record, v) for v in values]
    assert verdicts.count("lower") == 3
    assert verdicts.count("upper") == 1
    assert verdicts.count(None) == 1


# ---------------------------------------------------------------------------
# patched_schedule
# ---------------------------------------------------------------------------


def test_patched_schedule_changes_training_schedule_output():
    import src.deep_models as dm

    original = (dm.MAX_STEPS, dm.PATIENCE_STEPS, dm.VALIDATION_FRACTION)
    baseline = dm.training_schedule(100, 16, max_steps=dm.MAX_STEPS, patience_steps=dm.PATIENCE_STEPS)

    with patched_schedule(max_steps=2000, patience_steps=150, validation_fraction=0.30) as patched_dm:
        assert patched_dm.MAX_STEPS == 2000
        assert patched_dm.PATIENCE_STEPS == 150
        assert patched_dm.VALIDATION_FRACTION == 0.30
        patched = dm.training_schedule(100, 16, max_steps=dm.MAX_STEPS, patience_steps=dm.PATIENCE_STEPS)
        assert patched != baseline
        assert patched["max_epochs"] >= baseline["max_epochs"]

    assert (dm.MAX_STEPS, dm.PATIENCE_STEPS, dm.VALIDATION_FRACTION) == original


def test_patched_schedule_restores_on_exception():
    import src.deep_models as dm

    original = (dm.MAX_STEPS, dm.PATIENCE_STEPS, dm.VALIDATION_FRACTION)
    with pytest.raises(RuntimeError):
        with patched_schedule(max_steps=1, patience_steps=1, validation_fraction=0.99):
            assert dm.MAX_STEPS == 1
            raise RuntimeError("simulated failure mid-fit")

    assert (dm.MAX_STEPS, dm.PATIENCE_STEPS, dm.VALIDATION_FRACTION) == original


def test_variants_match_the_spec():
    assert VARIANTS["baseline"] == {"max_steps": 600, "patience_steps": 60, "validation_fraction": 0.15}
    assert VARIANTS["A_long"] == {"max_steps": 2000, "patience_steps": 150, "validation_fraction": 0.15}
    assert VARIANTS["B_bigval"] == {"max_steps": 600, "patience_steps": 60, "validation_fraction": 0.30}
    assert VARIANTS["C_long_bigval"] == {"max_steps": 2000, "patience_steps": 150, "validation_fraction": 0.30}


# ---------------------------------------------------------------------------
# Resume logic
# ---------------------------------------------------------------------------


def test_existing_sensitivity_keys_empty_when_file_missing(tmp_path):
    assert _existing_sensitivity_keys(tmp_path / "does_not_exist.csv") == set()


def test_resume_skips_rows_already_present(tmp_path):
    csv_path = tmp_path / "sensitivity_runs.csv"
    existing_rows = pd.DataFrame(
        [
            {"dataset": "pima_diabetes", "model": "mlp", "split_idx": 0, "fraction": 0.1, "variant": "baseline",
             "status": "ok"},
            {"dataset": "ckd", "model": "tabnet", "split_idx": 1, "fraction": 1.0, "variant": "A_long",
             "status": "failed"},
        ]
    )
    existing_rows.to_csv(csv_path, index=False)

    keys = _existing_sensitivity_keys(csv_path)
    assert ("pima_diabetes", "mlp", 0, 0.1, "baseline") in keys
    assert ("ckd", "tabnet", 1, 1.0, "A_long") in keys
    assert len(keys) == 2

    all_combos = all_sensitivity_combos()
    pending = [c for c in all_combos if c not in keys]
    assert len(pending) == len(all_combos) - 2
    assert ("pima_diabetes", "mlp", 0, 0.1, "baseline") not in pending
    assert ("ckd", "tabnet", 1, 1.0, "A_long") not in pending


def test_all_sensitivity_combos_has_expected_total():
    combos = all_sensitivity_combos()
    # 4 variants x 5 datasets x 3 models x 3 splits x 2 fractions = 360
    assert len(combos) == 360
    assert len(set(combos)) == 360  # no duplicates


# ---------------------------------------------------------------------------
# Collapsed-split selector
# ---------------------------------------------------------------------------


def test_find_collapsed_splits_picks_exactly_splits_below_threshold():
    table = pd.DataFrame(
        [
            {"dataset": "ckd", "split_idx": 0, "auroc": 0.999},
            {"dataset": "ckd", "split_idx": 1, "auroc": 0.759},  # below 0.90 -> collapsed
            {"dataset": "ckd", "split_idx": 2, "auroc": 0.901},  # just above threshold
            {"dataset": "heart_cleveland", "split_idx": 0, "auroc": 0.830},
            {"dataset": "heart_cleveland", "split_idx": 1, "auroc": 0.740},  # below 0.75 -> collapsed
        ]
    )
    thresholds = {"ckd": 0.90, "heart_cleveland": 0.75}
    collapsed = find_collapsed_splits(table, thresholds)
    pairs = set(zip(collapsed["dataset"], collapsed["split_idx"]))
    assert pairs == {("ckd", 1), ("heart_cleveland", 1)}
    assert len(collapsed) == 2


def test_find_collapsed_splits_returns_empty_when_none_collapsed():
    table = pd.DataFrame([{"dataset": "ckd", "split_idx": 0, "auroc": 0.999}])
    collapsed = find_collapsed_splits(table, {"ckd": 0.90})
    assert len(collapsed) == 0

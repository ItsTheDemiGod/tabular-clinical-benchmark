"""Tests for the Phase 3a statistical-analysis helpers (analysis/*.py and
scripts/phase3a_stats.py). Kept fast: no file I/O into results/phase3a except
inside tmp_path, no GPU, no dataset downloads."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import friedmanchisquare, t as t_dist, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.bootstrap import bootstrap_diff_ci, bootstrap_mean_ci  # noqa: E402
from analysis.constants import ALL_RUNS_PATH, OUTPUT_DIR, RAW_DIR  # noqa: E402
from analysis.stats import (  # noqa: E402
    detect_cliques,
    friedman_test,
    holm_correction,
    iman_davenport,
    nadeau_bengio_ttest,
    nemenyi_cd,
    ranks_best_is_1,
    wilcoxon_pair,
)
from analysis.tables import write_csv_and_tex  # noqa: E402


# ---------------------------------------------------------------------------
# Holm step-down
# ---------------------------------------------------------------------------
def test_holm_matches_manual_calculation():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.5])
    # Manual Holm step-down (already sorted ascending): multiplier = m - k for k=0..4.
    # raw = [0.01*5, 0.02*4, 0.03*3, 0.04*2, 0.5*1] = [0.05, 0.08, 0.09, 0.08, 0.5]
    # running max (monotone): [0.05, 0.08, 0.09, 0.09, 0.5]
    expected = np.array([0.05, 0.08, 0.09, 0.09, 0.5])
    adjusted = holm_correction(p)
    np.testing.assert_allclose(adjusted, expected)


def test_holm_is_monotone_and_capped_at_one():
    p = np.array([0.9, 0.3, 0.95, 0.01, 0.5, 0.02])
    adjusted = holm_correction(p)
    order = np.argsort(p)
    assert np.all(np.diff(adjusted[order]) >= -1e-12)
    assert np.all(adjusted <= 1.0)


# ---------------------------------------------------------------------------
# Friedman / Iman-Davenport
# ---------------------------------------------------------------------------
def test_friedman_matches_scipy():
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 3.0],
            [1.0, 3.0, 2.0],
            [3.0, 2.0, 1.0],
        ]
    )
    statistic, p_value, n_blocks, n_models = friedman_test(values)
    expected_stat, expected_p = friedmanchisquare(values[:, 0], values[:, 1], values[:, 2])
    assert statistic == pytest.approx(expected_stat, abs=1e-9)
    assert p_value == pytest.approx(expected_p, abs=1e-9)
    assert n_blocks == 4
    assert n_models == 3


def test_iman_davenport_matches_manual_formula():
    statistic, _, n_blocks, n_models = friedman_test(
        np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0], [1.0, 3.0, 2.0], [3.0, 2.0, 1.0]])
    )
    f_statistic, p_value, df1, df2 = iman_davenport(statistic, n_blocks, n_models)
    expected_f = ((n_blocks - 1) * statistic) / (n_blocks * (n_models - 1) - statistic)
    assert f_statistic == pytest.approx(expected_f, abs=1e-9)
    assert df1 == n_models - 1
    assert df2 == (n_models - 1) * (n_blocks - 1)
    assert 0.0 <= p_value <= 1.0


# ---------------------------------------------------------------------------
# Ranks with ties
# ---------------------------------------------------------------------------
def test_ranks_average_on_ties_and_distinct_block():
    values = np.array(
        [
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # all 7 tied
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],  # strictly increasing
        ]
    )
    ranks = ranks_best_is_1(values, higher_is_better=True)
    np.testing.assert_allclose(ranks[0], [4.0] * 7)  # average of ranks 1..7
    np.testing.assert_allclose(ranks[1], [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0])


# ---------------------------------------------------------------------------
# Nemenyi
# ---------------------------------------------------------------------------
def test_nemenyi_q_alpha_matches_demsar_table():
    _, q_alpha = nemenyi_cd(k=7, n_blocks=50)
    assert q_alpha == pytest.approx(2.949, abs=0.002)


def test_nemenyi_cd_matches_manual_formula():
    cd, q_alpha = nemenyi_cd(k=7, n_blocks=50)
    expected_cd = q_alpha * np.sqrt(7 * 8 / (6.0 * 50))
    assert cd == pytest.approx(expected_cd, abs=1e-9)


# ---------------------------------------------------------------------------
# Wilcoxon wrapper
# ---------------------------------------------------------------------------
def test_wilcoxon_all_zero_differences():
    stat, p, n_nonzero = wilcoxon_pair(np.zeros(10))
    assert np.isnan(stat)
    assert p == 1.0
    assert n_nonzero == 0


def test_wilcoxon_known_case_matches_scipy():
    d = np.array([1, 2, 3, -1, -2, 0, 4, -3, 5, -4], dtype=float)
    stat, p, n_nonzero = wilcoxon_pair(d)
    expected_stat, expected_p = wilcoxon(d, zero_method="wilcox", method="auto")
    assert stat == pytest.approx(expected_stat)
    assert p == pytest.approx(expected_p)
    assert n_nonzero == 9  # one exact zero among the 10


# ---------------------------------------------------------------------------
# Nadeau-Bengio corrected t-test
# ---------------------------------------------------------------------------
def test_nadeau_bengio_matches_manual_calculation():
    d = np.array([0.1, 0.2, -0.1, 0.3, 0.0, 0.15, 0.25, -0.05, 0.1, 0.2])
    ratio = 0.25
    j = len(d)
    mean_d = d.mean()
    var_d = d.var(ddof=1)
    expected_t = mean_d / np.sqrt((1.0 / j + ratio) * var_d)
    expected_p = 2 * t_dist.sf(abs(expected_t), df=j - 1)

    t_statistic, p_value, flagged = nadeau_bengio_ttest(d, ratio)
    assert t_statistic == pytest.approx(expected_t, abs=1e-9)
    assert p_value == pytest.approx(expected_p, abs=1e-9)
    assert flagged is False


def test_nadeau_bengio_zero_variance_zero_mean():
    t_statistic, p_value, flagged = nadeau_bengio_ttest(np.zeros(10), 0.25)
    assert t_statistic == 0.0
    assert p_value == 1.0
    assert flagged is False


def test_nadeau_bengio_zero_variance_nonzero_mean_is_flagged():
    t_statistic, p_value, flagged = nadeau_bengio_ttest(np.full(10, 0.3), 0.25)
    assert flagged is True
    assert p_value == 0.0
    assert np.isposinf(t_statistic)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def test_bootstrap_same_seed_gives_identical_ci():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    point1, lo1, hi1 = bootstrap_mean_ci(values, np.random.default_rng(42), 1000)
    point2, lo2, hi2 = bootstrap_mean_ci(values, np.random.default_rng(42), 1000)
    assert (point1, lo1, hi1) == (point2, lo2, hi2)


def test_bootstrap_ci_contains_the_mean():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    point, lo, hi = bootstrap_mean_ci(values, np.random.default_rng(42), 2000)
    assert point == pytest.approx(values.mean())
    assert lo <= point <= hi


def test_bootstrap_diff_uses_same_resampled_indices():
    logreg_values = np.array([0.80, 0.81, 0.79, 0.82, 0.78, 0.80, 0.83, 0.77, 0.80, 0.81])
    constant_shift = 0.05
    model_values = logreg_values + constant_shift
    point, lo, hi = bootstrap_diff_ci(model_values, logreg_values, np.random.default_rng(42), 2000)
    # If the same resampled indices are used for both arrays, EVERY bootstrap
    # replicate's difference is exactly the constant shift, so the CI collapses
    # to a point at that shift.
    assert point == pytest.approx(constant_shift, abs=1e-9)
    assert lo == pytest.approx(constant_shift, abs=1e-9)
    assert hi == pytest.approx(constant_shift, abs=1e-9)


# ---------------------------------------------------------------------------
# Clique detection
# ---------------------------------------------------------------------------
def test_detect_cliques_on_hand_written_matrix():
    # Models 0,1,2 are mutually non-significant (p_holm >= 0.05); model 3 is
    # significantly different from all of them.
    matrix = np.array(
        [
            [1.0, 0.90, 0.90, 0.01],
            [0.90, 1.0, 0.90, 0.02],
            [0.90, 0.90, 1.0, 0.03],
            [0.01, 0.02, 0.03, 1.0],
        ]
    )
    cliques = detect_cliques(matrix)
    assert cliques == [(0, 2)]


def test_detect_cliques_no_clique_when_all_significant():
    matrix = np.array(
        [
            [1.0, 0.01, 0.01],
            [0.01, 1.0, 0.01],
            [0.01, 0.01, 1.0],
        ]
    )
    assert detect_cliques(matrix) == []


# ---------------------------------------------------------------------------
# Guard: Phase 3a code only reads results/all_runs.csv & results/raw, and only
# ever writes inside results/phase3a/.
# ---------------------------------------------------------------------------
def test_guard_constants_point_only_at_allowed_locations():
    assert OUTPUT_DIR == Path("results/phase3a")
    assert ALL_RUNS_PATH == Path("results/all_runs.csv")
    assert RAW_DIR == Path("results/raw")


def test_guard_writer_modules_never_reference_protected_result_dirs():
    # analysis/data.py legitimately mentions results/raw (read-only precondition
    # cross-check) and results/all_runs.csv; every OTHER module that writes output
    # (tables, plotting, bootstrap, stats) has no business reading or writing
    # results/raw, tuning, predictions, models or diagnostics, and should never
    # even name them as string literals.
    protected = [
        "results/raw",
        "results/tuning",
        "results/predictions",
        "results/models",
        "results/diagnostics",
        "phase2a_all_runs.csv",
    ]
    project_root = Path(__file__).resolve().parents[1]
    writer_files = [
        project_root / "analysis" / "tables.py",
        project_root / "analysis" / "plotting.py",
        project_root / "analysis" / "bootstrap.py",
        project_root / "analysis" / "stats.py",
    ]
    for path in writer_files:
        text = path.read_text(encoding="utf-8")
        for forbidden in protected:
            assert forbidden not in text, f"{path} unexpectedly references {forbidden!r}"


def test_write_csv_and_tex_only_writes_into_given_directory(tmp_path):
    display = pd.DataFrame({"model_a": ["1.000"]}, index=["dataset_x"])
    csv_path, tex_path = write_csv_and_tex("sometable", display, tmp_path, index_label="dataset")
    assert csv_path.parent == tmp_path
    assert tex_path.parent == tmp_path
    assert csv_path.exists()
    assert tex_path.exists()

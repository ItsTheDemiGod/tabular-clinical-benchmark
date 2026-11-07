"""Statistical primitives for Phase 3a: ranks, Friedman/Iman-Davenport, Holm-adjusted
Wilcoxon, the Nadeau-Bengio corrected t-test, the Nemenyi critical difference, and
clique detection for the critical-difference diagrams.

No function here touches a file; everything operates on numpy arrays / pandas frames
already loaded by analysis.data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import friedmanchisquare
from scipy.stats import rankdata
from scipy.stats import studentized_range
from scipy.stats import t as t_dist
from scipy.stats import wilcoxon


def ranks_best_is_1(matrix: np.ndarray, higher_is_better: bool) -> np.ndarray:
    """Per-row ranks (rank 1 = best), ties broken by averaging, for a (blocks x models) matrix."""
    data = -matrix if higher_is_better else matrix
    return np.apply_along_axis(lambda row: rankdata(row, method="average"), 1, data)


def friedman_test(matrix: np.ndarray) -> tuple[float, float, int, int]:
    """Friedman chi-squared test on a (blocks x models) matrix. Returns (statistic, p, N, k)."""
    n_blocks, n_models = matrix.shape
    columns = [matrix[:, i] for i in range(n_models)]
    statistic, p_value = friedmanchisquare(*columns)
    return float(statistic), float(p_value), n_blocks, n_models


def iman_davenport(statistic: float, n_blocks: int, n_models: int) -> tuple[float, float, int, int]:
    """Iman-Davenport F correction of the Friedman statistic. Returns (F, p, df1, df2)."""
    df1 = n_models - 1
    df2 = (n_models - 1) * (n_blocks - 1)
    # Guard against the degenerate case where the Friedman statistic equals its
    # maximum possible value N*(k-1), which would divide by zero.
    denom = n_blocks * (n_models - 1) - statistic
    if denom <= 0:
        return float("inf"), 0.0, df1, df2
    f_statistic = ((n_blocks - 1) * statistic) / denom
    p_value = float(f_dist.sf(f_statistic, df1, df2))
    return f_statistic, p_value, df1, df2


def holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment. Returns adjusted p-values in the ORIGINAL order,
    monotone non-decreasing along the sorted order and capped at 1.
    """
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]
    adjusted_sorted = np.empty(m)
    running_max = 0.0
    for k in range(m):
        candidate = min(1.0, sorted_p[k] * (m - k))
        running_max = max(running_max, candidate)
        adjusted_sorted[k] = running_max
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted


def wilcoxon_pair(differences: np.ndarray) -> tuple[float, float, int]:
    """Two-sided Wilcoxon signed-rank test on paired differences.

    Returns (statistic, p_value, n_nonzero). All-zero differences return
    (nan, 1.0, 0) rather than raising, per the Phase 3a brief.
    """
    differences = np.asarray(differences, dtype=float)
    n_nonzero = int(np.sum(differences != 0))
    if n_nonzero == 0:
        return float("nan"), 1.0, 0
    statistic, p_value = wilcoxon(differences, zero_method="wilcox", method="auto")
    return float(statistic), float(p_value), n_nonzero


def nadeau_bengio_ttest(differences: np.ndarray, n_test_over_n_train: float) -> tuple[float, float, bool]:
    """Nadeau-Bengio corrected resampled paired t-test.

    t = mean(d) / sqrt((1/J + n_test/n_train) * var(d)), J = len(differences), var with ddof=1.
    Returns (t_statistic, p_value, flagged). flagged is True only for the degenerate
    case var(d) == 0 with mean(d) != 0 (all differences identical but nonzero).
    """
    differences = np.asarray(differences, dtype=float)
    j = len(differences)
    mean_d = float(differences.mean())
    var_d = float(differences.var(ddof=1))
    # Treat floating-point noise around zero (e.g. identical float64 values whose
    # variance computes to ~1e-33 rather than exactly 0.0) as the degenerate case.
    if var_d < 1e-12:
        if mean_d == 0.0:
            return 0.0, 1.0, False
        return float(np.sign(mean_d) * np.inf), 0.0, True
    denom = np.sqrt((1.0 / j + n_test_over_n_train) * var_d)
    t_statistic = mean_d / denom
    p_value = float(2 * t_dist.sf(abs(t_statistic), df=j - 1))
    return float(t_statistic), p_value, False


def nemenyi_cd(k: int, n_blocks: int) -> tuple[float, float]:
    """Classical Nemenyi critical distance. Returns (CD, q_alpha)."""
    q_alpha = float(studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2))
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_blocks))
    return cd, q_alpha


def detect_cliques(p_holm_matrix: np.ndarray, alpha: float = 0.05) -> list[tuple[int, int]]:
    """Maximal cliques of consecutive indices (in rank order) where every pair has
    p_holm >= alpha. `p_holm_matrix` must already be reordered to rank order and be
    symmetric. Returns a sorted list of (start, end) inclusive index pairs with end > start
    (single-model cliques are not reported, since they are not drawn).
    """
    n = p_holm_matrix.shape[0]
    valid = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i, n):
            ok = True
            for a in range(i, j + 1):
                for b in range(a + 1, j + 1):
                    if p_holm_matrix[a, b] < alpha:
                        ok = False
                        break
                if not ok:
                    break
            valid[i, j] = ok

    candidates = [(i, j) for i in range(n) for j in range(i + 1, n) if valid[i, j]]
    maximal = [
        (i, j)
        for (i, j) in candidates
        if not any((i2 <= i and j2 >= j and (i2, j2) != (i, j)) for (i2, j2) in candidates)
    ]
    return sorted(set(maximal))


def pairwise_wilcoxon_holm(block_values: np.ndarray, models: list[str]) -> pd.DataFrame:
    """All k*(k-1)/2 pairwise Wilcoxon tests on a (blocks x models) matrix, Holm-adjusted."""
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            diffs = block_values[:, i] - block_values[:, j]
            _, p_raw, n_nonzero = wilcoxon_pair(diffs)
            rows.append(
                {
                    "model_a": models[i],
                    "model_b": models[j],
                    "mean_diff": float(np.mean(diffs)),
                    "n_blocks": block_values.shape[0],
                    "n_nonzero": n_nonzero,
                    "p_raw": p_raw,
                }
            )
    table = pd.DataFrame(rows)
    table["p_holm"] = holm_correction(table["p_raw"].to_numpy())
    table["significant_holm_0.05"] = table["p_holm"] < 0.05
    return table


def symmetric_p_holm_matrix(pairwise: pd.DataFrame, models: list[str]) -> np.ndarray:
    """Dense k x k symmetric matrix of Holm-adjusted p-values (diagonal = 1.0)."""
    n = len(models)
    index = {model: i for i, model in enumerate(models)}
    matrix = np.ones((n, n))
    for _, row in pairwise.iterrows():
        i, j = index[row["model_a"]], index[row["model_b"]]
        matrix[i, j] = row["p_holm"]
        matrix[j, i] = row["p_holm"]
    return matrix

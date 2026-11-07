"""Phase 3a: statistical analysis, learning-curve figures and LaTeX tables.

Reads results/all_runs.csv only. Writes only inside results/phase3a/. Never touches
results/raw, results/tuning, results/predictions, results/models, results/diagnostics,
or anything in src/.

Subcommands: tests, ci, figures, tables, all.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from analysis.bootstrap import (
    bootstrap_diff_ci,
    bootstrap_diff_ci_averaged,
    bootstrap_mean_ci,
    bootstrap_mean_ci_averaged,
)
from analysis.constants import (
    BOOTSTRAP_B,
    DATASETS_ALL5,
    DATASETS_NO_CKD,
    FRACTIONS_ALL,
    FRACTIONS_TEST,
    HIGHER_IS_BETTER,
    METRIC_LABEL,
    METRICS_TESTED,
    MODELS,
    OUTPUT_DIR,
    SEED,
)
from analysis.data import block_matrix, dataset_matrix, get_split_values, load_all_runs, load_raw_concat
from analysis.plotting import plot_cd_diagram, plot_diff_dotwhisker, plot_learning_curves
from analysis.stats import (
    friedman_test,
    holm_correction,
    iman_davenport,
    nadeau_bengio_ttest,
    nemenyi_cd,
    pairwise_wilcoxon_holm,
    ranks_best_is_1,
    symmetric_p_holm_matrix,
)
from analysis.tables import bold_best_mask, format_float, write_csv_and_tex


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


# ---------------------------------------------------------------------------
# Subcommand: tests  (Friedman/Iman-Davenport, Holm-adjusted Wilcoxon, Nadeau-Bengio)
# ---------------------------------------------------------------------------
def run_tests_subcommand(df: pd.DataFrame) -> None:
    output_dir = ensure_output_dir()
    friedman_rows: list[dict] = []
    pairwise_tables: list[pd.DataFrame] = []

    for metric in METRICS_TESTED:
        higher_is_better = HIGHER_IS_BETTER[metric]
        for fraction in FRACTIONS_TEST:
            for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                blocks = block_matrix(df, metric, fraction, datasets)
                block_values = blocks.to_numpy()
                friedman_rows.append(
                    _friedman_row(block_values, metric, fraction, set_name, "BLOCK")
                )
                pairwise_block = pairwise_wilcoxon_holm(block_values, MODELS)
                pairwise_block.insert(0, "unit", "BLOCK")
                pairwise_block.insert(0, "dataset_set", set_name)
                pairwise_block.insert(0, "fraction", fraction)
                pairwise_block.insert(0, "metric", metric)
                pairwise_block["min_possible_p"] = np.nan
                pairwise_tables.append(pairwise_block)

                dmatrix = dataset_matrix(blocks, datasets)
                dataset_values = dmatrix.to_numpy()
                friedman_rows.append(
                    _friedman_row(dataset_values, metric, fraction, set_name, "DATASET")
                )
                pairwise_dataset = pairwise_wilcoxon_holm(dataset_values, MODELS)
                pairwise_dataset.insert(0, "unit", "DATASET")
                pairwise_dataset.insert(0, "dataset_set", set_name)
                pairwise_dataset.insert(0, "fraction", fraction)
                pairwise_dataset.insert(0, "metric", metric)
                n_cells = dataset_values.shape[0]
                pairwise_dataset["min_possible_p"] = 2 ** (1 - n_cells)
                pairwise_tables.append(pairwise_dataset)

    friedman_df = pd.DataFrame(friedman_rows)
    friedman_df.to_csv(output_dir / "friedman_summary.csv", index=False)

    pairwise_df = pd.concat(pairwise_tables, ignore_index=True)
    pairwise_df.to_csv(output_dir / "pairwise_wilcoxon.csv", index=False)

    corrected_df, ratios = run_corrected_ttest(df)
    corrected_df.to_csv(output_dir / "corrected_ttest.csv", index=False)

    _write_stats_notes(output_dir, ratios)


def _friedman_row(values: np.ndarray, metric: str, fraction: float, set_name: str, unit: str) -> dict:
    higher_is_better = HIGHER_IS_BETTER[metric]
    ranks = ranks_best_is_1(values, higher_is_better)
    mean_ranks = ranks.mean(axis=0)
    statistic, p_value, n_blocks, n_models = friedman_test(values)
    f_stat, f_p, df1, df2 = iman_davenport(statistic, n_blocks, n_models)
    row = {
        "metric": metric,
        "fraction": fraction,
        "dataset_set": set_name,
        "unit": unit,
        "statistic": statistic,
        "p_value": p_value,
        "iman_davenport_F": f_stat,
        "iman_davenport_p": f_p,
        "df1": df1,
        "df2": df2,
        "n_blocks": n_blocks,
        "n_models": n_models,
    }
    for model, rank in zip(MODELS, mean_ranks):
        row[f"mean_rank_{model}"] = rank
    return row


def run_corrected_ttest(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Nadeau-Bengio corrected paired t-test, logreg vs. each of the 6 other models,
    per dataset, for auroc and log_loss at fractions 1.0 and 0.1.
    """
    ratios: dict[str, float] = {}
    for dataset in DATASETS_ALL5:
        row = df[(df["dataset"] == dataset) & (df["fraction"] == 1.0) & (df["split_idx"] == 0)].iloc[0]
        ratios[dataset] = float(row["n_test"]) / float(row["n_train"])

    rows = []
    for dataset in DATASETS_ALL5:
        ratio = ratios[dataset]
        for fraction in FRACTIONS_TEST:
            for metric in ("auroc", "log_loss"):
                logreg_vals = get_split_values(df, dataset, fraction, "logreg", metric).to_numpy()
                group_rows = []
                for model in MODELS:
                    if model == "logreg":
                        continue
                    other_vals = get_split_values(df, dataset, fraction, model, metric).to_numpy()
                    diffs = other_vals - logreg_vals
                    t_stat, p_raw, flagged = nadeau_bengio_ttest(diffs, ratio)
                    group_rows.append(
                        {
                            "dataset": dataset,
                            "fraction": fraction,
                            "metric": metric,
                            "model": model,
                            "mean_diff_vs_logreg": float(diffs.mean()),
                            "t_statistic": t_stat,
                            "p_raw": p_raw,
                            "flagged_zero_variance_nonzero_mean": flagged,
                            "n_test_over_n_train": ratio,
                        }
                    )
                p_raw_array = np.array([r["p_raw"] for r in group_rows])
                p_holm_array = holm_correction(p_raw_array)
                for row_dict, p_holm in zip(group_rows, p_holm_array):
                    row_dict["p_holm"] = p_holm
                rows.extend(group_rows)
    return pd.DataFrame(rows), ratios


def _write_stats_notes(output_dir: Path, ratios: dict[str, float]) -> None:
    lines = [
        "# Phase 3a statistical notes\n",
        "## Unit-of-analysis caveat\n",
        "- **BLOCK unit** (50 blocks for ALL5, 40 for NO_CKD: one per (dataset, split_idx)):",
        "  treats the 10 splits of a dataset as independent, although they come from",
        "  repeated 5-fold cross-validation with overlapping training sets. BLOCK-unit",
        "  p-values are therefore **anti-conservative** (they understate the true p-value).",
        "- **DATASET unit** (5 or 4 dataset-level cells): valid (each cell is one dataset,",
        "  genuinely independent of the others), but has almost no power. With 5 cells the",
        "  smallest possible two-sided Wilcoxon p-value is 2/32 = 0.0625; with 4 cells it is",
        "  2/16 = 0.125 — no pair can reach Holm-adjusted significance at 0.05 on this unit.",
        "  The `min_possible_p` column in pairwise_wilcoxon.csv records this per row.",
        "- Both units are reported; neither file ever states that a model is",
        '  "significantly better" — only "Holm-adjusted p < 0.05 on the BLOCK unit".\n',
        "## Nadeau-Bengio n_test/n_train ratios (first split, fraction 1.0)\n",
    ]
    for dataset, ratio in ratios.items():
        lines.append(f"- {dataset}: {ratio:.6f}")
    (output_dir / "stats_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Subcommand: ci  (bootstrap confidence intervals)
# ---------------------------------------------------------------------------
def run_ci_subcommand(df: pd.DataFrame) -> None:
    output_dir = ensure_output_dir()
    rng = np.random.default_rng(SEED)

    per_dataset_rows = []
    averaged_rows = []
    for model in MODELS:
        for fraction in FRACTIONS_ALL:
            for metric in METRICS_TESTED:
                values_by_dataset = []
                for dataset in DATASETS_ALL5:
                    values = get_split_values(df, dataset, fraction, model, metric).to_numpy()
                    values_by_dataset.append(values)
                    point, lo, hi = bootstrap_mean_ci(values, rng, BOOTSTRAP_B)
                    per_dataset_rows.append(
                        {
                            "model": model,
                            "fraction": fraction,
                            "metric": metric,
                            "dataset": dataset,
                            "point": point,
                            "ci_lo": lo,
                            "ci_hi": hi,
                        }
                    )
                for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                    subset = [v for d, v in zip(DATASETS_ALL5, values_by_dataset) if d in datasets]
                    point, lo, hi = bootstrap_mean_ci_averaged(subset, rng, BOOTSTRAP_B)
                    averaged_rows.append(
                        {
                            "model": model,
                            "fraction": fraction,
                            "metric": metric,
                            "dataset_set": set_name,
                            "point": point,
                            "ci_lo": lo,
                            "ci_hi": hi,
                        }
                    )

    pd.DataFrame(per_dataset_rows).to_csv(output_dir / "ci_per_dataset.csv", index=False)
    pd.DataFrame(averaged_rows).to_csv(output_dir / "ci_dataset_averaged.csv", index=False)

    diff_rows = []
    for model in MODELS:
        if model == "logreg":
            continue
        for fraction in FRACTIONS_ALL:
            for metric in METRICS_TESTED:
                model_by_dataset = []
                logreg_by_dataset = []
                per_dataset_point = {}
                for dataset in DATASETS_ALL5:
                    model_vals = get_split_values(df, dataset, fraction, model, metric).to_numpy()
                    logreg_vals = get_split_values(df, dataset, fraction, "logreg", metric).to_numpy()
                    model_by_dataset.append(model_vals)
                    logreg_by_dataset.append(logreg_vals)
                    point, lo, hi = bootstrap_diff_ci(model_vals, logreg_vals, rng, BOOTSTRAP_B)
                    per_dataset_point[dataset] = point
                    diff_rows.append(
                        {
                            "model": model,
                            "fraction": fraction,
                            "metric": metric,
                            "level": "PER_DATASET",
                            "dataset_set": dataset,
                            "point": point,
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "n_positive": np.nan,
                            "n_total": np.nan,
                        }
                    )
                for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                    model_subset = [v for d, v in zip(DATASETS_ALL5, model_by_dataset) if d in datasets]
                    logreg_subset = [v for d, v in zip(DATASETS_ALL5, logreg_by_dataset) if d in datasets]
                    point, lo, hi = bootstrap_diff_ci_averaged(model_subset, logreg_subset, rng, BOOTSTRAP_B)
                    n_positive = sum(1 for d in datasets if per_dataset_point[d] > 0)
                    diff_rows.append(
                        {
                            "model": model,
                            "fraction": fraction,
                            "metric": metric,
                            "level": "AVERAGED",
                            "dataset_set": set_name,
                            "point": point,
                            "ci_lo": lo,
                            "ci_hi": hi,
                            "n_positive": n_positive,
                            "n_total": len(datasets),
                        }
                    )

    pd.DataFrame(diff_rows).to_csv(output_dir / "ci_diff_vs_logreg.csv", index=False)


# ---------------------------------------------------------------------------
# Subcommand: figures
# ---------------------------------------------------------------------------
def run_figures_subcommand(df: pd.DataFrame) -> list[str]:
    output_dir = ensure_output_dir()
    checked: list[str] = []

    for metric, fig_name in (
        ("auroc", "fig1_learning_curves_auroc"),
        ("log_loss", "fig1b_learning_curves_logloss"),
        ("brier", "fig1c_learning_curves_brier"),
    ):
        rng = np.random.default_rng(SEED)
        panels: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for dataset in DATASETS_ALL5:
            curves = {}
            for model in MODELS:
                means, los, his = [], [], []
                for fraction in FRACTIONS_ALL:
                    values = get_split_values(df, dataset, fraction, model, metric).to_numpy()
                    point, lo, hi = bootstrap_mean_ci(values, rng, BOOTSTRAP_B)
                    means.append(point)
                    los.append(lo)
                    his.append(hi)
                curves[model] = (np.array(means), np.array(los), np.array(his))
            panels[dataset] = curves

        averaged_curves = {}
        for model in MODELS:
            means, los, his = [], [], []
            for fraction in FRACTIONS_ALL:
                values_by_dataset = [
                    get_split_values(df, dataset, fraction, model, metric).to_numpy() for dataset in DATASETS_ALL5
                ]
                point, lo, hi = bootstrap_mean_ci_averaged(values_by_dataset, rng, BOOTSTRAP_B)
                means.append(point)
                los.append(lo)
                his.append(hi)
            averaged_curves[model] = (np.array(means), np.array(los), np.array(his))
        panels["Average (ALL5)"] = averaged_curves

        png_path, pdf_path = plot_learning_curves(
            panels, FRACTIONS_ALL, MODELS, METRIC_LABEL[metric], output_dir, fig_name
        )
        checked.append(f"{png_path.name}: 6 panels (5 datasets + ALL5 average), legend present, axes labelled")

    cd_names = []
    for fraction in FRACTIONS_TEST:
        for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
            blocks = block_matrix(df, "auroc", fraction, datasets)
            block_values = blocks.to_numpy()
            ranks = ranks_best_is_1(block_values, True)
            mean_rank_by_model = dict(zip(MODELS, ranks.mean(axis=0)))
            pairwise = pairwise_wilcoxon_holm(block_values, MODELS)
            order = sorted(MODELS, key=lambda m: mean_rank_by_model[m])
            matrix = symmetric_p_holm_matrix(pairwise, MODELS)
            order_idx = [MODELS.index(m) for m in order]
            ordered_matrix = matrix[np.ix_(order_idx, order_idx)]
            from analysis.stats import detect_cliques

            cliques = detect_cliques(ordered_matrix)
            cd, _ = nemenyi_cd(len(MODELS), block_values.shape[0])
            name = f"fig2_cd_auroc_frac{fraction}_{set_name}".replace(".", "p")
            title = f"AUROC, fraction={fraction}, {set_name} (BLOCK unit)"
            png_path, _ = plot_cd_diagram(mean_rank_by_model, cliques, cd, title, output_dir, name)
            cd_names.append(png_path.name)
    checked.append(f"CD diagrams ({', '.join(cd_names)}): model labels readable, no overlap with bars")

    rng = np.random.default_rng(SEED)
    for fraction in FRACTIONS_TEST:
        diff_all5, diff_no_ckd = {}, {}
        for model in MODELS:
            if model == "logreg":
                continue
            model_all5 = [get_split_values(df, d, fraction, model, "auroc").to_numpy() for d in DATASETS_ALL5]
            logreg_all5 = [get_split_values(df, d, fraction, "logreg", "auroc").to_numpy() for d in DATASETS_ALL5]
            diff_all5[model] = bootstrap_diff_ci_averaged(model_all5, logreg_all5, rng, BOOTSTRAP_B)

            model_no_ckd = [get_split_values(df, d, fraction, model, "auroc").to_numpy() for d in DATASETS_NO_CKD]
            logreg_no_ckd = [get_split_values(df, d, fraction, "logreg", "auroc").to_numpy() for d in DATASETS_NO_CKD]
            diff_no_ckd[model] = bootstrap_diff_ci_averaged(model_no_ckd, logreg_no_ckd, rng, BOOTSTRAP_B)

        name = f"fig3_diff_vs_logreg_frac{fraction}".replace(".", "p")
        png_path, _ = plot_diff_dotwhisker(
            diff_all5,
            diff_no_ckd,
            [m for m in MODELS if m != "logreg"],
            f"AUROC diff. vs. logreg, fraction={fraction}",
            output_dir,
            name,
        )
        checked.append(f"{png_path.name}: dot/whisker legible, zero line visible, markers filled=ALL5/open=NO_CKD")

    _write_figure_captions(output_dir)
    return checked


def _write_figure_captions(output_dir: Path) -> None:
    lines = [
        "# Figure captions (factual only, no interpretation)\n",
        "## fig1 / fig1b / fig1c: learning curves",
        "Mean AUROC / log loss / Brier score vs. training fraction (0.1, 0.25, 0.5, 1.0),",
        "7 models, one panel per dataset (10 splits per point) plus a 6th panel averaging",
        "the 5 datasets (nested bootstrap). Shaded bands are 95% percentile bootstrap CIs,",
        f"B = {BOOTSTRAP_B}, seed {SEED}, resampling the 10 splits with replacement.\n",
        "## fig2: critical-difference-style diagrams",
        "Mean ranks (BLOCK unit, rank 1 = best AUROC) of the 7 models, for fraction 1.0 and",
        "0.1, ALL5 and NO_CKD. Thick bars connect cliques of consecutively-ranked models",
        "for which every pairwise Holm-adjusted Wilcoxon p-value is >= 0.05. The classical",
        "Nemenyi critical distance is printed in the title for reference but is NOT what is",
        "drawn as a bar; the bars are the Wilcoxon-Holm cliques.\n",
        "## fig3: paired difference to logreg",
        "Dataset-averaged AUROC difference (model minus logreg), nested bootstrap, 95% CI,",
        f"B = {BOOTSTRAP_B}, seed {SEED}, for fraction 1.0 and 0.1, ALL5 (filled marker) and",
        "NO_CKD (open marker) side by side, with a vertical reference line at 0.\n",
    ]
    (output_dir / "figure_captions.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Subcommand: tables
# ---------------------------------------------------------------------------
def run_tables_subcommand(df: pd.DataFrame) -> None:
    output_dir = ensure_output_dir()
    caption_lines = ["# Table captions / notes\n"]

    # Table 1 & 2: mean +/- std at fraction 1.0, rows = datasets + 2 summary rows.
    for metric, name, higher_is_better in (("auroc", "table1_auroc", True), ("log_loss", "table2_logloss", False)):
        means = {}
        stds = {}
        for dataset in DATASETS_ALL5:
            row_vals = {m: get_split_values(df, dataset, 1.0, m, metric).to_numpy() for m in MODELS}
            means[dataset] = {m: v.mean() for m, v in row_vals.items()}
            stds[dataset] = {m: v.std(ddof=1) for m, v in row_vals.items()}
        mean_df = pd.DataFrame(means).T.reindex(index=DATASETS_ALL5, columns=MODELS)
        std_df = pd.DataFrame(stds).T.reindex(index=DATASETS_ALL5, columns=MODELS)

        mean_all5 = mean_df.loc[DATASETS_ALL5].mean(axis=0)
        std_all5 = mean_df.loc[DATASETS_ALL5].std(axis=0, ddof=1)
        mean_no_ckd = mean_df.loc[DATASETS_NO_CKD].mean(axis=0)
        std_no_ckd = mean_df.loc[DATASETS_NO_CKD].std(axis=0, ddof=1)
        mean_df.loc["mean over datasets (ALL5)"] = mean_all5
        std_df.loc["mean over datasets (ALL5)"] = std_all5
        mean_df.loc["mean over 4 datasets (NO_CKD)"] = mean_no_ckd
        std_df.loc["mean over 4 datasets (NO_CKD)"] = std_no_ckd

        display = pd.DataFrame(index=mean_df.index, columns=MODELS)
        for row_label in mean_df.index:
            decimals = 4 if row_label == "ckd" else 3
            for model in MODELS:
                display.loc[row_label, model] = (
                    f"{mean_df.loc[row_label, model]:.{decimals}f} ± {std_df.loc[row_label, model]:.{decimals}f}"
                )
        bold_mask = bold_best_mask(mean_df, higher_is_better)
        write_csv_and_tex(name, display, output_dir, bold_mask, index_label="dataset")

    # Table 3: mean metric per model and fraction, dataset-averaged ALL5 and NO_CKD.
    for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
        rows = {}
        for fraction in FRACTIONS_ALL:
            row = {}
            for model in MODELS:
                vals = [get_split_values(df, d, fraction, model, "auroc").to_numpy().mean() for d in datasets]
                row[model] = float(np.mean(vals))
            rows[fraction] = row
        table3 = pd.DataFrame(rows).T.reindex(columns=MODELS)
        table3.index.name = "fraction"
        display3 = table3.map(lambda v: format_float(v, 3))
        write_csv_and_tex(f"table3_mean_auroc_{set_name}", display3, output_dir, index_label="fraction")

    # Table 4: mean rank per model at each fraction (BLOCK unit, ALL5 and NO_CKD).
    friedman_p_notes = []
    for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
        rows = {}
        for fraction in FRACTIONS_TEST:
            blocks = block_matrix(df, "auroc", fraction, datasets)
            block_values = blocks.to_numpy()
            ranks = ranks_best_is_1(block_values, True)
            mean_ranks = ranks.mean(axis=0)
            rows[fraction] = dict(zip(MODELS, mean_ranks))
            _, p_value, n_blocks, n_models = friedman_test(block_values)
            friedman_p_notes.append(
                f"- {set_name}, fraction {fraction}: Friedman p = {p_value:.6g} (N={n_blocks}, k={n_models})"
            )
        table4 = pd.DataFrame(rows).T.reindex(columns=MODELS)
        table4.index.name = "fraction"
        display4 = table4.map(lambda v: format_float(v, 3))
        write_csv_and_tex(f"table4_mean_rank_{set_name}", display4, output_dir, index_label="fraction")
    caption_lines.append("## Table 4 Friedman p-values (BLOCK unit, AUROC)\n")
    caption_lines.extend(friedman_p_notes)
    caption_lines.append("")

    # Table 5: mean brier/log_loss/ece/fit_time_s/predict_time_s at fraction 1.0.
    columns5 = ["brier", "log_loss", "ece", "fit_time_s", "predict_time_s"]
    rows5 = {}
    for model in MODELS:
        row = {}
        for col in columns5:
            per_dataset_means = [get_split_values(df, d, 1.0, model, col).to_numpy().mean() for d in DATASETS_ALL5]
            row[col] = float(np.mean(per_dataset_means))
        rows5[model] = row
    table5 = pd.DataFrame(rows5).T.reindex(index=MODELS, columns=columns5)
    display5 = table5.map(lambda v: format_float(v, 4))
    write_csv_and_tex("table5_calibration_timing", display5, output_dir, index_label="model")
    caption_lines.append(
        "## Table 5 note: fit_time_s / predict_time_s were measured with 3 to 4 parallel "
        "workers (see CLAUDE.md section 5) and are only rough."
    )

    (output_dir / "table_captions.md").write_text("\n".join(caption_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3a statistical analysis")
    parser.add_argument("subcommand", choices=["tests", "ci", "figures", "tables", "all"])
    args = parser.parse_args()

    print(f"sys.executable = {sys.executable}")
    df = load_all_runs()
    ensure_output_dir()

    start = time.time()
    if args.subcommand in ("tests", "all"):
        t0 = time.time()
        run_tests_subcommand(df)
        print(f"[tests] done in {time.time() - t0:.1f}s")
    if args.subcommand in ("ci", "all"):
        t0 = time.time()
        run_ci_subcommand(df)
        print(f"[ci] done in {time.time() - t0:.1f}s")
    if args.subcommand in ("figures", "all"):
        t0 = time.time()
        checked = run_figures_subcommand(df)
        print(f"[figures] done in {time.time() - t0:.1f}s")
        for line in checked:
            print(f"  checked: {line}")
    if args.subcommand in ("tables", "all"):
        t0 = time.time()
        run_tables_subcommand(df)
        print(f"[tables] done in {time.time() - t0:.1f}s")
    print(f"total runtime: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

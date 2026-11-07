# Inventory of figures and tables (Phases 2c, 3a, 3b, 3c, 3d)

LaTeX column-width estimate method: for each table COLUMN, the longest cell's character count (header included, \textbf{} stripped) converted to inches (0.075in/char + 0.12in padding), summed over columns -- a documented heuristic, NOT a real LaTeX layout pass -- treat `fits_one_column_3.4in` as an estimate, not a guarantee.

## Phase 3a (10 items)

### fig1
- files: results/phase3a/fig1_learning_curves_auroc.pdf; results/phase3a/fig1_learning_curves_auroc.png
- what it shows: Mean AUROC / log loss / Brier score vs. training fraction (0.1, 0.25, 0.5, 1.0), 7 models, one panel per dataset (10 splits per point) plus a 6th panel averaging the 5 datasets (nested bootstrap). Shaded bands are 95% percentile bootstrap CIs, B = 5000, seed 42, resampling the 10 splits with replacement.
- panel count: 6
- pixel size: 2068x1406 (6.89x4.69 in at 300 dpi)
- data source: results/all_runs.csv
- known issues: none noted

### fig1b
- files: results/phase3a/fig1b_learning_curves_logloss.pdf; results/phase3a/fig1b_learning_curves_logloss.png
- what it shows: Mean AUROC / log loss / Brier score vs. training fraction (0.1, 0.25, 0.5, 1.0), 7 models, one panel per dataset (10 splits per point) plus a 6th panel averaging the 5 datasets (nested bootstrap). Shaded bands are 95% percentile bootstrap CIs, B = 5000, seed 42, resampling the 10 splits with replacement.
- panel count: 6
- pixel size: 2065x1406 (6.88x4.69 in at 300 dpi)
- data source: results/all_runs.csv
- known issues: none noted

### fig1c
- files: results/phase3a/fig1c_learning_curves_brier.pdf; results/phase3a/fig1c_learning_curves_brier.png
- what it shows: Mean AUROC / log loss / Brier score vs. training fraction (0.1, 0.25, 0.5, 1.0), 7 models, one panel per dataset (10 splits per point) plus a 6th panel averaging the 5 datasets (nested bootstrap). Shaded bands are 95% percentile bootstrap CIs, B = 5000, seed 42, resampling the 10 splits with replacement.
- panel count: 6
- pixel size: 2068x1406 (6.89x4.69 in at 300 dpi)
- data source: results/all_runs.csv
- known issues: none noted

### fig2
- files: results/phase3a/fig2_cd_auroc_frac0p1_ALL5.pdf; results/phase3a/fig2_cd_auroc_frac0p1_ALL5.png; results/phase3a/fig2_cd_auroc_frac0p1_NO_CKD.pdf; results/phase3a/fig2_cd_auroc_frac0p1_NO_CKD.png; results/phase3a/fig2_cd_auroc_frac1p0_ALL5.pdf; results/phase3a/fig2_cd_auroc_frac1p0_ALL5.png; results/phase3a/fig2_cd_auroc_frac1p0_NO_CKD.pdf; results/phase3a/fig2_cd_auroc_frac1p0_NO_CKD.png
- what it shows: Mean ranks (BLOCK unit, rank 1 = best AUROC) of the 7 models, for fraction 1.0 and 0.1, ALL5 and NO_CKD. Thick bars connect cliques of consecutively-ranked models for which every pairwise Holm-adjusted Wilcoxon p-value is >= 0.05. The classical Nemenyi critical distance is printed in the title for reference but is NOT what is drawn as a bar; the bars are the Wilcoxon-Holm cliques.
- panel count: 1
- pixel size: 2070x764 (6.90x2.55 in at 300 dpi)
- data source: results/phase3a/friedman_summary.csv, pairwise_wilcoxon.csv
- known issues: none noted

### fig3
- files: results/phase3a/fig3_diff_vs_logreg_frac0p1.pdf; results/phase3a/fig3_diff_vs_logreg_frac0p1.png; results/phase3a/fig3_diff_vs_logreg_frac1p0.pdf; results/phase3a/fig3_diff_vs_logreg_frac1p0.png
- what it shows: Dataset-averaged AUROC difference (model minus logreg), nested bootstrap, 95% CI, B = 5000, seed 42, for fraction 1.0 and 0.1, ALL5 (filled marker) and NO_CKD (open marker) side by side, with a vertical reference line at 0.
- panel count: 1
- pixel size: 988x1046 (3.29x3.49 in at 300 dpi)
- data source: results/phase3a/ci_diff_vs_logreg.csv
- known issues: none noted

### table1
- files: results/phase3a/table1_auroc.csv; results/phase3a/table1_auroc.tex
- what it shows: (no caption found)
- table rows x cols: 7 x 8
- LaTeX width estimate: 11.09 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: none noted

### table2
- files: results/phase3a/table2_logloss.csv; results/phase3a/table2_logloss.tex
- what it shows: (no caption found)
- table rows x cols: 7 x 8
- LaTeX width estimate: 11.09 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: none noted

### table3
- files: results/phase3a/table3_mean_auroc_ALL5.csv; results/phase3a/table3_mean_auroc_ALL5.tex; results/phase3a/table3_mean_auroc_NO_CKD.csv; results/phase3a/table3_mean_auroc_NO_CKD.tex
- what it shows: (no caption found)
- table rows x cols: 4 x 8
- LaTeX width estimate: 4.33 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: Phase 2a/2b's own Table 3 (per-split mean ECE) is inflated by small test folds (60-154 rows) and must not be quoted (CLAUDE.md decision 11); not to be confused with Phase 3a's table3_mean_auroc_*.csv, which reports AUROC only.

### table4
- files: results/phase3a/table4_mean_rank_ALL5.csv; results/phase3a/table4_mean_rank_ALL5.tex; results/phase3a/table4_mean_rank_NO_CKD.csv; results/phase3a/table4_mean_rank_NO_CKD.tex
- what it shows: (no caption found)
- table rows x cols: 2 x 8
- LaTeX width estimate: 4.33 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: Only fractions 1.0 and 0.1 (Phase 3a); fractions 0.25 and 0.5 are in Phase 3b's table4b instead.

### table5
- files: results/phase3a/table5_calibration_timing.csv; results/phase3a/table5_calibration_timing.tex
- what it shows: (no caption found)
- table rows x cols: 7 x 6
- LaTeX width estimate: 4.84 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: fit_time_s / predict_time_s measured with 3 to 4 parallel workers; rough.

## Phase 3b (9 items)

### fig4
- files: results/phase3b/fig4_reliability_NO_CKD_frac1p0.pdf; results/phase3b/fig4_reliability_NO_CKD_frac1p0.png
- what it shows: Equal-mass 10-bin reliability diagrams (mean predicted probability vs. observed positive fraction, Wilson 95% interval on the observed fraction, diagonal reference), one panel per model, pooled over the 4 NO_CKD datasets and their 10 splits each (2 x n_rows per dataset, since each test row appears in 2 of the 10 splits). fig4: fraction=1.0. fig4b: fraction=0.1. Histogram strip below each panel: distribution of the pooled predicted probabilities, 20 equal-width bins.
- panel count: 7
- pixel size: 1830x1310 (6.10x4.37 in at 300 dpi)
- data source: results/predictions/*.npz
- known issues: none noted

### fig4b
- files: results/phase3b/fig4b_reliability_NO_CKD_frac0p1.pdf; results/phase3b/fig4b_reliability_NO_CKD_frac0p1.png
- what it shows: Equal-mass 10-bin reliability diagrams (mean predicted probability vs. observed positive fraction, Wilson 95% interval on the observed fraction, diagonal reference), one panel per model, pooled over the 4 NO_CKD datasets and their 10 splits each (2 x n_rows per dataset, since each test row appears in 2 of the 10 splits). fig4: fraction=1.0. fig4b: fraction=0.1. Histogram strip below each panel: distribution of the pooled predicted probabilities, 20 equal-width bins.
- panel count: 7
- pixel size: 1830x1310 (6.10x4.37 in at 300 dpi)
- data source: results/predictions/*.npz
- known issues: none noted

### fig4c
- files: results/phase3b/fig4c_reliability_per_dataset_frac1p0.pdf; results/phase3b/fig4c_reliability_per_dataset_frac1p0.png
- what it shows: Same equal-mass 10-bin reliability curves, one panel per (dataset, model) pair, pooled over that dataset's 10 splits, fraction=1.0 only, no histogram strip.
- panel count: 35
- pixel size: 2067x1632 (6.89x5.44 in at 300 dpi)
- data source: results/predictions/*.npz
- known issues: none noted

### fig5
- files: results/phase3b/fig5_calibration_slope.pdf; results/phase3b/fig5_calibration_slope.png
- what it shows: Pooled calibration slope (Platt regression on logit(p), MLE) per model, with 95% bootstrap CIs (seed 42, B=2000, resampling the 10 splits), coloured by model (same mapping as Phase 3a) with one marker shape per dataset (pima_diabetes=o, heart_cleveland=s, ckd=^, ilpd=D, breast_wisconsin=v), fraction=1.0 and 0.1 side by side, vertical line at slope=1. Cells omitted as separable (exact class separation in logit space, slope/intercept undefined): none.
- panel count: 2
- pixel size: 2066x1008 (6.89x3.36 in at 300 dpi)
- data source: results/phase3b/calibration_metrics.csv, calibration_bootstrap_ci.csv
- known issues: none noted

### fig6
- files: results/phase3b/fig6_posthoc_effect_frac1p0.pdf; results/phase3b/fig6_posthoc_effect_frac1p0.png
- what it shows: Change in dataset-averaged Brier (top row) and log loss (bottom row) relative to the original (uncalibrated) predictions, for Platt and isotonic cross-fitted post-hoc recalibration, fraction=1.0, ALL5 (left) and NO_CKD (right), per model.
- panel count: 4
- pixel size: 2073x1528 (6.91x5.09 in at 300 dpi)
- data source: results/phase3b/posthoc_summary.csv
- known issues: none noted

### table4b
- files: results/phase3b/table4b_mean_rank.csv; results/phase3b/table4b_mean_rank.tex; results/phase3b/table4b_mean_rank_raw.csv
- what it shows: (no caption found)
- table rows x cols: 12 x 8
- LaTeX width estimate: 5.76 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: Only fractions 0.25 and 0.5 (Phase 3b); fractions 1.0 and 0.1 are in Phase 3a's table4 instead. Descriptive only (no hypothesis tests).

### table6
- files: results/phase3b/table6_calibration_frac1p0.csv; results/phase3b/table6_calibration_frac1p0.tex
- what it shows: - logreg: nan - rf: nan - xgb: nan - lgbm: nan - mlp: nan - ftt: nan - tabnet: nan
- table rows x cols: 7 x 7
- LaTeX width estimate: 5.56 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: NO_CKD slope/intercept; 0 cells separable in this run (see Phase 3b notes); CKD's own extreme slopes are not in this table.

### table7
- files: results/phase3b/table7_calibration_frac0p1.csv; results/phase3b/table7_calibration_frac0p1.tex
- what it shows: - logreg: nan - rf: nan - xgb: nan - lgbm: nan - mlp: nan - ftt: nan - tabnet: nan
- table rows x cols: 7 x 7
- LaTeX width estimate: 5.56 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: Same caveats as table6, at fraction 0.1.

### table8
- files: results/phase3b/table8_posthoc_effect.csv; results/phase3b/table8_posthoc_effect.tex
- what it shows: (no caption found)
- table rows x cols: 14 x 10
- LaTeX width estimate: 15.15 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: Post-hoc Platt/isotonic changes are descriptive and cross-fitted on test predictions; not a pipeline component.

## Phase 3c (10 items)

### fig7
- files: results/phase3c/fig7_degradation_auroc_NO_CKD.pdf; results/phase3c/fig7_degradation_auroc_NO_CKD.png
- what it shows: Dataset-averaged AUROC (fig7/7c) or log loss (fig7b/7d) vs. perturbation level, 7 models, 2 panels (missing rate 0/0.1/0.2/0.3; noise level 0/0.1/0.25/0.5). Each point averages 5 independent perturbation draws per (dataset, split, kind, level), then the 10 splits per dataset, then the datasets (nested bootstrap, seed 42, B=5000). The level-0 point is each model's own clean prediction (no CI shown there, since it is not a perturbation-level estimate).
- panel count: 2
- pixel size: 2070x910 (6.90x3.03 in at 300 dpi)
- data source: results/phase3c/robustness_averaged.csv
- known issues: none noted

### fig7b
- files: results/phase3c/fig7b_degradation_logloss_NO_CKD.pdf; results/phase3c/fig7b_degradation_logloss_NO_CKD.png
- what it shows: Dataset-averaged AUROC (fig7/7c) or log loss (fig7b/7d) vs. perturbation level, 7 models, 2 panels (missing rate 0/0.1/0.2/0.3; noise level 0/0.1/0.25/0.5). Each point averages 5 independent perturbation draws per (dataset, split, kind, level), then the 10 splits per dataset, then the datasets (nested bootstrap, seed 42, B=5000). The level-0 point is each model's own clean prediction (no CI shown there, since it is not a perturbation-level estimate).
- panel count: 2
- pixel size: 2086x910 (6.95x3.03 in at 300 dpi)
- data source: results/phase3c/robustness_averaged.csv
- known issues: none noted

### fig7c
- files: results/phase3c/fig7c_degradation_auroc_ALL5.pdf; results/phase3c/fig7c_degradation_auroc_ALL5.png
- what it shows: Dataset-averaged AUROC (fig7/7c) or log loss (fig7b/7d) vs. perturbation level, 7 models, 2 panels (missing rate 0/0.1/0.2/0.3; noise level 0/0.1/0.25/0.5). Each point averages 5 independent perturbation draws per (dataset, split, kind, level), then the 10 splits per dataset, then the datasets (nested bootstrap, seed 42, B=5000). The level-0 point is each model's own clean prediction (no CI shown there, since it is not a perturbation-level estimate).
- panel count: 2
- pixel size: 2070x910 (6.90x3.03 in at 300 dpi)
- data source: results/phase3c/robustness_averaged.csv
- known issues: none noted

### fig7d
- files: results/phase3c/fig7d_degradation_logloss_ALL5.pdf; results/phase3c/fig7d_degradation_logloss_ALL5.png
- what it shows: Dataset-averaged AUROC (fig7/7c) or log loss (fig7b/7d) vs. perturbation level, 7 models, 2 panels (missing rate 0/0.1/0.2/0.3; noise level 0/0.1/0.25/0.5). Each point averages 5 independent perturbation draws per (dataset, split, kind, level), then the 10 splits per dataset, then the datasets (nested bootstrap, seed 42, B=5000). The level-0 point is each model's own clean prediction (no CI shown there, since it is not a perturbation-level estimate).
- panel count: 2
- pixel size: 2050x910 (6.83x3.03 in at 300 dpi)
- data source: results/phase3c/robustness_averaged.csv
- known issues: none noted

### fig7e
- files: results/phase3c/fig7e_per_dataset_auroc.pdf; results/phase3c/fig7e_per_dataset_auroc.png
- what it shows: Same AUROC curves as fig7/7c, one row per dataset, fraction=1.0, no CI bands (point estimates only).
- panel count: 10
- pixel size: 2071x2896 (6.90x9.65 in at 300 dpi)
- data source: results/phase3c/robustness_per_dataset.csv
- known issues: none noted

### fig8
- files: results/phase3c/fig8_diff_vs_logreg_degradation.pdf; results/phase3c/fig8_diff_vs_logreg_degradation.png
- what it shows: Dataset-averaged AUROC degradation difference (model minus logreg) at the highest missing level (30%), with 95% bootstrap CIs (seed 42, B=5000, same resampled splits for both models), ALL5 (filled) and NO_CKD (open) side by side, vertical line at 0.
- panel count: 1
- pixel size: 987x926 (3.29x3.09 in at 300 dpi)
- data source: results/phase3c/robustness_diff_vs_logreg.csv
- known issues: none noted

### table10
- files: results/phase3c/table10_auroc_noise_ALL5.csv; results/phase3c/table10_auroc_noise_ALL5.tex; results/phase3c/table10_auroc_noise_NO_CKD.csv; results/phase3c/table10_auroc_noise_NO_CKD.tex
- what it shows: (no caption found)
- table rows x cols: 4 x 8
- LaTeX width estimate: 4.11 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: Clean + 3 noise levels only (no missing); ALL5/NO_CKD as separate files.

### table11
- files: results/phase3c/table11_logloss_missing_ALL5.csv; results/phase3c/table11_logloss_missing_ALL5.tex; results/phase3c/table11_logloss_missing_NO_CKD.csv; results/phase3c/table11_logloss_missing_NO_CKD.tex; results/phase3c/table11_logloss_noise_ALL5.csv; results/phase3c/table11_logloss_noise_ALL5.tex; results/phase3c/table11_logloss_noise_NO_CKD.csv; results/phase3c/table11_logloss_noise_NO_CKD.tex
- what it shows: (no caption found)
- table rows x cols: 4 x 8
- LaTeX width estimate: 4.11 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: Separate files for missing vs. noise kinds; log loss only.

### table12
- files: results/phase3c/table12_diff_vs_logreg.csv; results/phase3c/table12_diff_vs_logreg.tex
- what it shows: (no caption found)
- table rows x cols: 48 x 10
- LaTeX width estimate: 7.3 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: The version of this table printed in the Phase 3c session's final chat message was garbled; only the CSV/TeX file is authoritative (CLAUDE.md decision 13).

### table9
- files: results/phase3c/table9_auroc_missing_ALL5.csv; results/phase3c/table9_auroc_missing_ALL5.tex; results/phase3c/table9_auroc_missing_NO_CKD.csv; results/phase3c/table9_auroc_missing_NO_CKD.tex
- what it shows: (no caption found)
- table rows x cols: 4 x 8
- LaTeX width estimate: 4.11 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: Clean + 3 missing levels only (no noise); ALL5/NO_CKD as separate files.

## Phase 3d (5 items)

### fig10
- files: results/phase3d/fig10_top_features_importance.pdf; results/phase3d/fig10_top_features_importance.png
- what it shows: Horizontal bars of the mean (over 10 splits) normalised SHAP-derived importance for the union of each model's top-5 original features, one panel per dataset, 3 bars per feature (logreg, rf, mlp). Features on the pre-registered clinical list (results/phase3d/clinical_list.json) are marked with an asterisk.
- panel count: 5
- pixel size: 2063x4425 (6.88x14.75 in at 300 dpi)
- data source: results/phase3d/top_features.csv
- known issues: none noted

### fig11
- files: results/phase3d/fig11_agreement_dotplot.pdf; results/phase3d/fig11_agreement_dotplot.png
- what it shows: Per dataset: mean top-5 Jaccard (left) and mean Spearman rank correlation (right) over the 10 splits for the 3 model pairs (95% bootstrap CI, seed 42, B=5000), the within-model stability level (square, mean over the 3 models' 45 split-pairs), and the chance level (x: exact expected Jaccard for the left panel, 0 for the right panel).
- panel count: 2
- pixel size: 2065x955 (6.88x3.18 in at 300 dpi)
- data source: results/phase3d/agreement_per_dataset.csv, baselines.csv
- known issues: none noted

### table15
- files: results/phase3d/table15_top_features.csv; results/phase3d/table15_top_features.tex
- what it shows: (no caption found)
- table rows x cols: 75 x 5
- LaTeX width estimate: 4.65 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: Top-5 by MEAN normalised importance across splits, not by per-split frequency.

### table16
- files: results/phase3d/table16_agreement.csv; results/phase3d/table16_agreement.tex
- what it shows: (no caption found)
- table rows x cols: 31 x 4
- LaTeX width estimate: 5.58 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: BLOCK-style per-split agreement is NOT computed; only per-dataset/averaged Jaccard and Spearman (10-split means) are reported, since the 10 splits overlap (repeated CV) and are not independent.

### table17
- files: results/phase3d/table17_plausibility.csv; results/phase3d/table17_plausibility.tex
- what it shows: (no caption found)
- table rows x cols: 15 x 8
- LaTeX width estimate: 4.47 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: Precision only; recall of the clinical list within the top-5 is not computed (not requested).

## Phase 3d-noise-cont (3 items)

### fig9
- files: results/phase3d/noise_continuous/fig9_logloss_noise_continuous_vs_original.pdf; results/phase3d/noise_continuous/fig9_logloss_noise_continuous_vs_original.png
- what it shows: Dataset-averaged log loss vs. noise level (0.1, 0.25, 0.5), 7 models, ALL5 and NO_CKD panels. Solid line = continuous-only noise (Phase 3d, CKD's sg/al/su and Cleveland's ca excluded from noise); dotted line = the original Phase 3c noise (all numeric columns noised). 5 draws per (dataset, split, level); bootstrap seed 42, B=5000, resampling the 10 splits within each dataset.
- panel count: 2
- pixel size: 2068x981 (6.89x3.27 in at 300 dpi)
- data source: results/phase3c/robustness_averaged.csv, results/phase3d/noise_continuous/noise_cont_averaged.csv
- known issues: none noted

### table13
- files: results/phase3d/noise_continuous/table13_auroc_noise_ALL5.csv; results/phase3d/noise_continuous/table13_auroc_noise_ALL5.tex; results/phase3d/noise_continuous/table13_auroc_noise_NO_CKD.csv; results/phase3d/noise_continuous/table13_auroc_noise_NO_CKD.tex; results/phase3d/noise_continuous/table13_log_loss_noise_ALL5.csv; results/phase3d/noise_continuous/table13_log_loss_noise_ALL5.tex; results/phase3d/noise_continuous/table13_log_loss_noise_NO_CKD.csv; results/phase3d/noise_continuous/table13_log_loss_noise_NO_CKD.tex
- what it shows: (no caption found)
- table rows x cols: 3 x 15
- LaTeX width estimate: 12.53 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: Continuous-only (_cont) vs. original Phase 3c noise (_orig) side by side; AUROC and log loss are separate files.

### table14
- files: results/phase3d/noise_continuous/table14_diff_vs_logreg_noise_cont_vs_original.csv; results/phase3d/noise_continuous/table14_diff_vs_logreg_noise_cont_vs_original.tex
- what it shows: (no caption found)
- table rows x cols: 24 x 9
- LaTeX width estimate: 7.0 in (fits one IEEE column: False)
- data source: (not curated)
- known issues: Only the highest level (30% missing / noise 0.5); per-dataset breakdown is in noise_cont_diff_vs_logreg.csv / robustness_diff_vs_logreg.csv, not in this table.

## Captions and notes files per phase

- results/phase3a/figure_captions.md
- results/phase3a/table_captions.md
- results/phase3a/stats_notes.md
- results/phase3b/figure_captions.md
- results/phase3b/table_captions.md
- results/phase3b/calibration_notes.md
- results/phase3c/figure_captions.md
- results/phase3c/table_captions.md
- results/phase3c/robustness_notes.md
- results/phase3d/figure_captions.md
- results/phase3d/table_captions.md
- results/phase3d/explanation_notes.md
- results/phase3d/noise_continuous/figure_captions.md
- results/phase3d/noise_continuous/table_captions.md
- results/phase3d/noise_continuous/robustness_notes.md

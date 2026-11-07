# Phase 3d (Part 4) notes: SHAP agreement and clinical plausibility

- Explanations are computed on ENCODED columns and aggregated to ORIGINAL
  features by summing signed SHAP values per row, then taking the mean
  absolute value over test rows.
- logreg is explained in LOG-ODDS (decision_function); rf and mlp are explained
  in PROBABILITY of the positive class. Only RANKINGS (top-5 sets, Spearman
  correlation) are compared across models, never raw SHAP magnitudes.
- The mlp KernelExplainer uses a 50-row TRAINING background
  (seed 42 + split_idx) and nsamples="auto".
- The 10 splits of a dataset come from repeated cross-validation with
  overlapping training sets, so they are not independent.
- The clinical feature list was fixed BEFORE any SHAP value was computed
  (results/phase3d/clinical_list.json, SHA-256 verified at analysis time).
- No direction-of-effect (positive/negative SHAP) check is performed here.
- Correlated features, especially Breast Wisconsin's mean/standard-error/worst
  triplets, can substitute for one another in a top-5 set.
- No file here states that models "agree" or that a feature is "plausible";
  only numbers (Jaccard, Spearman, precision) are reported.

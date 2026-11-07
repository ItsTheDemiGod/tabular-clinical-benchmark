# Phase 3b calibration notes

- Each test row appears TWICE in the pooled predictions used throughout this
  phase (once per repeat of the 5-fold cross-validation), so pooled analyses use
  2 x n_rows predictions per (dataset, model, fraction), not n_rows.
- ECE on small samples is biased and noisy; equal-width (ece_ew) and equal-mass
  (ece_em) ECE use different binning rules and can disagree.
- The calibration slope and intercept are undefined (NaN, flagged `separable`)
  when the two classes are exactly separable in logit space, as happens for CKD.
- The post-hoc (Platt/isotonic) analysis is cross-fitted on the TEST predictions
  themselves (inner 5-fold, each calibrator fit on ~48-123 rows and applied only
  to its held-out fold) and is descriptive only; it is NOT part of the benchmark
  pipeline, and isotonic regression's AUROC can change through tie-breaking.
- Friedman-style hypothesis tests are not applied to any calibration metric here.
- No file in results/phase3b states that a model is "significantly" better or
  "better calibrated" than another; only numbers are reported.

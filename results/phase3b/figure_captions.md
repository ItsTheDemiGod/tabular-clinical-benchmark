# Phase 3b figure captions (factual only, no interpretation)

## fig4 / fig4b: reliability diagrams (NO_CKD pooled)
Equal-mass 10-bin reliability diagrams (mean predicted probability vs. observed
positive fraction, Wilson 95% interval on the observed fraction, diagonal
reference), one panel per model, pooled over the 4 NO_CKD datasets and their 10
splits each (2 x n_rows per dataset, since each test row appears in 2 of the 10
splits). fig4: fraction=1.0. fig4b: fraction=0.1. Histogram strip below each
panel: distribution of the pooled predicted probabilities, 20 equal-width bins.

## fig4c: reliability per dataset (compact)
Same equal-mass 10-bin reliability curves, one panel per (dataset, model) pair,
pooled over that dataset's 10 splits, fraction=1.0 only, no histogram strip.

## fig5: calibration slope
Pooled calibration slope (Platt regression on logit(p), MLE) per model, with
95% bootstrap CIs (seed 42, B=2000, resampling the 10 splits), coloured by model
(same mapping as Phase 3a) with one marker shape per dataset (pima_diabetes=o, heart_cleveland=s, ckd=^, ilpd=D, breast_wisconsin=v),
fraction=1.0 and 0.1 side by side, vertical line at slope=1. Cells omitted as
separable (exact class separation in logit space, slope/intercept undefined): none.

## fig6: post-hoc effect
Change in dataset-averaged Brier (top row) and log loss (bottom row) relative to
the original (uncalibrated) predictions, for Platt and isotonic cross-fitted
post-hoc recalibration, fraction=1.0, ALL5 (left) and NO_CKD (right), per model.

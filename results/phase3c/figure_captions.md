# Phase 3c figure captions (factual only, no interpretation)

## fig7 / fig7b (NO_CKD) and fig7c / fig7d (ALL5): degradation curves
Dataset-averaged AUROC (fig7/7c) or log loss (fig7b/7d) vs. perturbation level,
7 models, 2 panels (missing rate 0/0.1/0.2/0.3; noise level 0/0.1/0.25/0.5).
Each point averages 5 independent perturbation draws per (dataset, split,
kind, level), then the 10 splits per dataset, then the datasets (nested bootstrap,
seed 42, B=5000). The level-0 point is each model's own clean prediction
(no CI shown there, since it is not a perturbation-level estimate).

## fig7e: per-dataset AUROC curves (supplementary)
Same AUROC curves as fig7/7c, one row per dataset, fraction=1.0, no CI bands
(point estimates only).

## fig8: degradation difference to logreg
Dataset-averaged AUROC degradation difference (model minus logreg) at the
highest missing level (30%), with 95% bootstrap CIs (seed 42, B=5000, same resampled splits for both models), ALL5 (filled) and
NO_CKD (open) side by side, vertical line at 0.

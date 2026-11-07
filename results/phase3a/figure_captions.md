# Figure captions (factual only, no interpretation)

## fig1 / fig1b / fig1c: learning curves
Mean AUROC / log loss / Brier score vs. training fraction (0.1, 0.25, 0.5, 1.0),
7 models, one panel per dataset (10 splits per point) plus a 6th panel averaging
the 5 datasets (nested bootstrap). Shaded bands are 95% percentile bootstrap CIs,
B = 5000, seed 42, resampling the 10 splits with replacement.

## fig2: critical-difference-style diagrams
Mean ranks (BLOCK unit, rank 1 = best AUROC) of the 7 models, for fraction 1.0 and
0.1, ALL5 and NO_CKD. Thick bars connect cliques of consecutively-ranked models
for which every pairwise Holm-adjusted Wilcoxon p-value is >= 0.05. The classical
Nemenyi critical distance is printed in the title for reference but is NOT what is
drawn as a bar; the bars are the Wilcoxon-Holm cliques.

## fig3: paired difference to logreg
Dataset-averaged AUROC difference (model minus logreg), nested bootstrap, 95% CI,
B = 5000, seed 42, for fraction 1.0 and 0.1, ALL5 (filled marker) and
NO_CKD (open marker) side by side, with a vertical reference line at 0.

# Phase 3d (Part 4) figure captions (factual only, no interpretation)

## fig10: top-5 union feature importance
Horizontal bars of the mean (over 10 splits) normalised SHAP-derived importance for the union of each model's top-5 original features, one panel per dataset, 3 bars per feature (logreg, rf, mlp). Features on the pre-registered clinical list (results/phase3d/clinical_list.json) are marked with an asterisk.

## fig11: agreement dot plot
Per dataset: mean top-5 Jaccard (left) and mean Spearman rank correlation (right) over the 10 splits for the 3 model pairs (95% bootstrap CI, seed 42, B=5000), the within-model stability level (square, mean over the 3 models' 45 split-pairs), and the chance level (x: exact expected Jaccard for the left panel, 0 for the right panel).

# Phase 3c robustness notes

- Models are trained on CLEAN data; only the TEST features are perturbed.
- The shared preprocessor (median / most-frequent imputation fitted on the
  training fold) is part of every saved pipeline, so these results describe
  MODEL + IMPUTER together; tree models do not use native NaN handling.
- Missingness is Missing Completely At Random (MCAR); there is no informative
  missingness mechanism here.
- Noise is relative to each column's TRAINING standard deviation (ddof=0) and is
  applied to ordinal numeric columns (CKD's sg/al/su, Cleveland's ca) as if
  continuous, same as every other numeric column.
- 5 independent perturbation draws per (dataset, split, kind, level);
  every model sees the identical perturbed matrix for a given draw (paired design).
- The dataset-averaged bootstrap resamples the 10 splits within each dataset and
  then averages the dataset means; it ignores between-dataset variability.
- Degradation is relative to each model's OWN clean performance on that split; a
  model that starts lower can degrade less in absolute terms.
- No file in results/phase3c/ states that a model is "more robust" or
  "significantly" better; only numbers are reported.

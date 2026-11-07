# Phase 3a statistical notes

## Unit-of-analysis caveat

- **BLOCK unit** (50 blocks for ALL5, 40 for NO_CKD: one per (dataset, split_idx)):
  treats the 10 splits of a dataset as independent, although they come from
  repeated 5-fold cross-validation with overlapping training sets. BLOCK-unit
  p-values are therefore **anti-conservative** (they understate the true p-value).
- **DATASET unit** (5 or 4 dataset-level cells): valid (each cell is one dataset,
  genuinely independent of the others), but has almost no power. With 5 cells the
  smallest possible two-sided Wilcoxon p-value is 2/32 = 0.0625; with 4 cells it is
  2/16 = 0.125 — no pair can reach Holm-adjusted significance at 0.05 on this unit.
  The `min_possible_p` column in pairwise_wilcoxon.csv records this per row.
- Both units are reported; neither file ever states that a model is
  "significantly better" — only "Holm-adjusted p < 0.05 on the BLOCK unit".

## Nadeau-Bengio n_test/n_train ratios (first split, fraction 1.0)

- pima_diabetes: 0.250814
- heart_cleveland: 0.252066
- ckd: 0.250000
- ilpd: 0.250000
- breast_wisconsin: 0.250549

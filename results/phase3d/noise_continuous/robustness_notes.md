# Phase 3d (Part 3) notes: continuous-only noise variant

- Identical to the Phase 3c NOISE perturbation, except CKD's sg/al/su and
  Cleveland's ca (ordinal numeric scales) receive NO noise at all.
- CKD's bp column also has <=10 distinct values but is intentionally kept IN
  (noised as ordinary continuous blood pressure), per an explicit decision.
- 5 draws per (dataset, split, level); seeded by (dataset, split,
  'noise_cont', level, draw), never by model.
- Degradation is computed against the CLEAN rows in
  results/phase3c/robustness_runs.csv (not recomputed).
- The bootstrap resamples the 10 splits within each dataset and ignores
  between-dataset variability. Descriptive only; no hypothesis tests.

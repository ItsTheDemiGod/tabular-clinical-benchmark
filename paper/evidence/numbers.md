# numbers.json / numbers.tex -- usage and rounding rules

## Usage in the paper
Write `\N{key}` in the LaTeX source (after `\input{paper/evidence/numbers.tex}`).
An undefined key makes LaTeX fail to compile -- this is intended, not a bug: it
means that number is not (yet) in the numbers book, so it must not be hand-typed.

## Rounding rules
- AUROC, Brier, log loss, ECE, slopes, intercepts, their differences and CI limits: 3 decimals.
- p-values: 3 decimals, or `<0.001` when below 0.001.
- Mean ranks: 2 decimals.
- Jaccard, Spearman, precision: 2 decimals.
- Percentages: 1 decimal.
- Times (seconds): 2 decimals.
- Counts: integers.
- Negative zero (e.g. -0.0001 rounded to 3 dp) prints as `0.000`, never `-0.000`.

## LaTeX safety
Formatted strings are escaped for `% # & _`; a leading minus sign is written as
`$-$` (so it renders correctly in text mode); `<0.001` is written as `$<$0.001`.

## Key naming scheme
Lowercase `a-z0-9.-` only. Dataset-set labels: `ALL5` -> `all5`, `NO_CKD` -> `nockd`.
Examples:
- `auroc-mean-logreg-all5-f1.0` -- dataset-averaged ALL5 mean AUROC, logreg, fraction 1.0.
- `jaccard-logreg-rf-ckd` -- top-5 Jaccard between logreg and rf on ckd.
- `n-holm-sig-block-pairs` -- count of Holm-significant BLOCK-unit pairs (Phase 3a).

Total entries: 2017.

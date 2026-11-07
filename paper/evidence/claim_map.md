# Claim map (Phase 4b)

Mechanical SUPPORTED/PARTLY/NOT SUPPORTED/DESCRIPTIVE status, fixed rules (see scripts/build_claim_map.py docstring), applied literally. No paper wording here.

## C1. At fraction 1.0, no model's dataset-averaged AUROC exceeds logreg's

sign column = number of datasets whose per-dataset mean diff agrees in sign with the dataset-averaged point estimate (threshold: >=3/5 ALL5, >=3/4 NO_CKD).

| model | ALL5 diff | ALL5 CI | ALL5 sign agree | NO_CKD diff | NO_CKD CI | NO_CKD sign agree | status |
|---|---|---|---|---|---|---|---|
| rf | -0.0025 | [-0.0062, 0.0014] | 4/5 | -0.0031 | [-0.0079, 0.002] | 3/4 | NOT SUPPORTED |
| xgb | -0.0043 | [-0.0087, 0.0004] | 4/5 | -0.005 | [-0.0105, 0.0007] | 3/4 | NOT SUPPORTED |
| lgbm | -0.007 | [-0.0112, -0.0028] | 4/5 | -0.0087 | [-0.0136, -0.0034] | 3/4 | SUPPORTED |
| mlp | -0.0014 | [-0.0047, 0.0017] | 3/5 | -0.0018 | [-0.0059, 0.002] | 3/4 | NOT SUPPORTED |
| ftt | -0.0084 | [-0.0136, -0.0035] | 5/5 | -0.0101 | [-0.0167, -0.0038] | 4/4 | SUPPORTED |
| tabnet | -0.0414 | [-0.0563, -0.0284] | 5/5 | -0.0456 | [-0.0597, -0.0328] | 4/4 | SUPPORTED |

3/6 non-logreg models are SUPPORTED (by the fixed rule) as having a dataset-averaged AUROC below logreg's at fraction 1.0; the per-cell exceptions below must still be stated alongside this, since the dataset-averaged claim does not rule out individual (dataset, model) cells where the model wins.

Exception cells at fraction 1.0 (5 of 30 (dataset,model) cells, model's per-dataset mean AUROC > logreg's):

| dataset | model | model mean AUROC | logreg mean AUROC | excess |
|---|---|---|---|---|
| pima_diabetes | rf | 0.8391 | 0.8341 | 0.005 |
| pima_diabetes | xgb | 0.8416 | 0.8341 | 0.0075 |
| pima_diabetes | lgbm | 0.8386 | 0.8341 | 0.0045 |
| pima_diabetes | mlp | 0.8357 | 0.8341 | 0.0016 |
| ckd | mlp | 1.0 | 0.9999 | 0.0001 |

## C2. The same at fraction 0.1

| model | ALL5 diff | ALL5 CI | ALL5 sign agree | NO_CKD diff | NO_CKD CI | NO_CKD sign agree | status |
|---|---|---|---|---|---|---|---|
| rf | -0.0021 | [-0.0082, 0.004] | 2/5 | -0.0026 | [-0.0101, 0.005] | 2/4 | NOT SUPPORTED |
| xgb | -0.0117 | [-0.0187, -0.0049] | 4/5 | -0.0123 | [-0.0209, -0.0039] | 3/4 | SUPPORTED |
| lgbm | -0.013 | [-0.0203, -0.0058] | 5/5 | -0.0146 | [-0.0237, -0.0056] | 4/4 | SUPPORTED |
| mlp | -0.011 | [-0.0191, -0.0044] | 3/5 | -0.0141 | [-0.0238, -0.0058] | 3/4 | SUPPORTED |
| ftt | -0.0251 | [-0.0392, -0.014] | 5/5 | -0.0308 | [-0.048, -0.0171] | 4/4 | SUPPORTED |
| tabnet | -0.162 | [-0.2035, -0.1228] | 5/5 | -0.1389 | [-0.17, -0.1083] | 4/4 | SUPPORTED |

5/6 non-logreg models are SUPPORTED as having a dataset-averaged AUROC below logreg's at fraction 0.1; see the per-cell exceptions below.

Exception cells at fraction 0.1 (6 of 30 (dataset,model) cells):

| dataset | model | model mean AUROC | logreg mean AUROC | excess |
|---|---|---|---|---|
| heart_cleveland | rf | 0.887 | 0.8851 | 0.0019 |
| ckd | rf | 0.9975 | 0.9973 | 0.0002 |
| ckd | mlp | 0.9987 | 0.9973 | 0.0013 |
| ilpd | rf | 0.7152 | 0.7003 | 0.0149 |
| ilpd | xgb | 0.7056 | 0.7003 | 0.0053 |
| breast_wisconsin | mlp | 0.99 | 0.9898 | 0.0002 |

## C3. Gap to logreg is larger at fraction 0.1 than 1.0, for xgb/lgbm/mlp/ftt/tabnet but not rf
(new quantity: diff-of-differences, group 4b, nested bootstrap seed=42 B=5000)

| model | ALL5 diff-of-diff | ALL5 CI | NO_CKD diff-of-diff | NO_CKD CI | status | note |
|---|---|---|---|---|---|---|
| rf | 0.0005 | [-0.005, 0.0057] | 0.0005 | [-0.0065, 0.007] | NOT SUPPORTED | expected ~flat (claimed exception) |
| xgb | -0.0074 | [-0.015, -0.0002] | -0.0073 | [-0.0166, 0.0018] | PARTLY | expected negative (gap larger/more negative at f0.1) |
| lgbm | -0.006 | [-0.0144, 0.0021] | -0.0059 | [-0.0165, 0.0044] | NOT SUPPORTED | expected negative (gap larger/more negative at f0.1) |
| mlp | -0.0097 | [-0.0171, -0.0028] | -0.0124 | [-0.0216, -0.0036] | SUPPORTED | expected negative (gap larger/more negative at f0.1) |
| ftt | -0.0167 | [-0.0309, -0.0043] | -0.0207 | [-0.0391, -0.0054] | SUPPORTED | expected negative (gap larger/more negative at f0.1) |
| tabnet | -0.1207 | [-0.166, -0.0757] | -0.0933 | [-0.1269, -0.0608] | SUPPORTED | expected negative (gap larger/more negative at f0.1) |

Negative diff-of-diff means the gap to logreg is MORE negative (larger) at fraction 0.1 than at 1.0, matching the claim's direction for that model.

## C4. Per-dataset corrected t-tests (Nadeau-Bengio, Holm)

Total cells: 120 (5 datasets x 2 fractions x 2 metrics x 6 models). Significant (p_holm<0.05): 1.

The 1 significant cell: dataset=ckd, model=xgb, metric=auroc, fraction=0.1, mean_diff_vs_logreg=-0.0092, t=-3.5739, p_holm=0.0359. Status: **DESCRIPTIVE**.

## C5. Calibration at fraction 0.1 and 1.0: pooled NO_CKD slope per model

| fraction | model | NO_CKD pooled slope | per-dataset slope [CI] |
|---|---|---|---|
| 1.0 | logreg | 1.1376 | pima_diabetes=1.0933 [0.9589,1.2648]; heart_cleveland=1.2207 [1.0009,1.5572]; ckd=10.7766 [9.3661,17.6163]; ilpd=0.914 [0.7436,1.136]; breast_wisconsin=1.3224 [0.9707,2.0111] |
| 1.0 | rf | 1.2785 | pima_diabetes=1.1197 [0.978,1.289]; heart_cleveland=1.5761 [1.2718,1.9471]; ckd=3.4214 [3.0615,8.0781]; ilpd=0.9512 [0.8226,1.1544]; breast_wisconsin=1.467 [1.297,1.7359] |
| 1.0 | xgb | 1.0659 | pima_diabetes=1.292 [1.1655,1.4336]; heart_cleveland=0.8756 [0.657,1.1665]; ckd=1.2714 [1.0539,1.8369]; ilpd=1.2013 [0.9125,1.7505]; breast_wisconsin=0.8946 [0.8028,1.0546] |
| 1.0 | lgbm | 0.8073 | pima_diabetes=1.0796 [0.9375,1.2389]; heart_cleveland=0.8265 [0.578,1.1822]; ckd=0.8643 [0.6843,1.3965]; ilpd=0.7413 [0.5597,1.0634]; breast_wisconsin=0.5818 [0.5019,0.7375] |
| 1.0 | mlp | 0.9805 | pima_diabetes=0.9885 [0.8831,1.1003]; heart_cleveland=1.034 [0.836,1.3818]; ckd=1.3801 [1.0966,3.246]; ilpd=0.9203 [0.7531,1.1737]; breast_wisconsin=0.9794 [0.7309,1.319] |
| 1.0 | ftt | 0.9086 | pima_diabetes=0.8496 [0.7348,0.9691]; heart_cleveland=0.7871 [0.674,0.9555]; ckd=0.8714 [0.7574,1.1595]; ilpd=1.0202 [0.8915,1.2162]; breast_wisconsin=0.9778 [0.8172,1.1896] |
| 1.0 | tabnet | 0.8222 | pima_diabetes=0.6782 [0.5546,0.8353]; heart_cleveland=0.7614 [0.6081,0.9812]; ckd=1.0951 [0.9499,2.0295]; ilpd=0.8954 [0.6983,1.2166]; breast_wisconsin=0.9539 [0.8132,1.1408] |
| 0.1 | logreg | 1.3413 | pima_diabetes=1.2987 [0.9224,1.8983]; heart_cleveland=2.104 [1.7117,2.7562]; ckd=1.7988 [1.2189,4.6177]; ilpd=0.3901 [0.2976,0.5993]; breast_wisconsin=1.5723 [1.1391,2.375] |
| 0.1 | rf | 1.1513 | pima_diabetes=0.9365 [0.8257,1.0695]; heart_cleveland=1.5191 [1.3431,1.7749]; ckd=3.6445 [3.18,4.8067]; ilpd=0.6462 [0.5108,0.8899]; breast_wisconsin=1.5036 [1.3459,1.7881] |
| 0.1 | xgb | 0.9752 | pima_diabetes=0.9919 [0.9017,1.1062]; heart_cleveland=1.0017 [0.8581,1.2507]; ckd=1.7288 [1.5108,2.1294]; ilpd=0.7559 [0.5064,1.5514]; breast_wisconsin=1.1511 [1.0759,1.3087] |
| 0.1 | lgbm | 0.5478 | pima_diabetes=0.7644 [0.6511,0.9477]; heart_cleveland=0.5839 [0.4582,0.8683]; ckd=0.7301 [0.5948,0.9928]; ilpd=0.3349 [0.2163,0.5699]; breast_wisconsin=0.5081 [0.4404,0.6072] |
| 0.1 | mlp | 0.4491 | pima_diabetes=0.3469 [0.1714,1.0361]; heart_cleveland=0.5989 [0.3835,1.3228]; ckd=0.8147 [0.4963,2.3039]; ilpd=0.2047 [0.1314,0.6193]; breast_wisconsin=0.646 [0.4711,0.9278] |
| 0.1 | ftt | 0.3661 | pima_diabetes=0.3254 [0.2085,0.5947]; heart_cleveland=0.2589 [0.2019,0.5033]; ckd=0.5096 [0.4386,0.6928]; ilpd=0.3382 [0.2305,0.5503]; breast_wisconsin=0.5418 [0.4082,0.8533] |
| 0.1 | tabnet | 0.2724 | pima_diabetes=0.1387 [0.1193,0.1855]; heart_cleveland=0.1252 [0.0441,0.1719]; ckd=0.4712 [0.3006,0.7876]; ilpd=0.2251 [0.173,0.2996]; breast_wisconsin=0.6006 [0.4698,0.8193] |

Status: **DESCRIPTIVE**. At fraction 0.1, mlp/ftt/tabnet's pooled NO_CKD slopes are below 1 as claimed; note xgb's pooled slope (0.975) is ALSO below 1 at fraction 0.1, so the logreg/rf/xgb-vs-mlp/ftt/tabnet grouping is not perfectly clean for xgb.

## C6. Lowest dataset-averaged log loss model, fraction 1.0 and 0.1, ALL5 and NO_CKD, vs. second-best

| fraction | dataset_set | leader | leader mean log loss | second | second mean log loss | leader-second diff | CI |
|---|---|---|---|---|---|---|---|
| 1.0 | ALL5 | mlp | 0.3011 | xgb | 0.31 | -0.009 | [-0.0213, 0.0018] |
| 1.0 | NO_CKD | logreg | 0.3678 | mlp | 0.3719 | -0.0042 | [-0.0112, 0.0026] |
| 0.1 | ALL5 | rf | 0.3771 | xgb | 0.3931 | -0.0159 | [-0.0246, -0.0072] |
| 0.1 | NO_CKD | rf | 0.4294 | logreg | 0.4433 | -0.0139 | [-0.026, -0.0019] |

Status: **DESCRIPTIVE**. Note the leader is NOT the same model across all 4 cells (mlp at f1.0/ALL5, logreg at f1.0/NO_CKD, rf at f0.1/ALL5 and f0.1/NO_CKD).

## C7. Range of dataset-averaged AUROC degradation across the 7 models

| perturbation | dataset_set | smallest drop (model) | largest drop (model) | range |
|---|---|---|---|---|
| 30% missing | ALL5 | -0.0252 (tabnet) | -0.0196 (ftt) | 0.0056 |
| 30% missing | NO_CKD | -0.0319 (tabnet) | -0.0236 (ftt) | 0.0084 |
| noise 0.5 (original) | ALL5 | -0.0295 (lgbm) | -0.017 (mlp) | 0.0125 |
| noise 0.5 (original) | NO_CKD | -0.0327 (tabnet) | -0.0205 (mlp) | 0.0123 |
| noise 0.5 (continuous-only) | ALL5 | -0.0236 (tabnet) | -0.0125 (mlp) | 0.0112 |
| noise 0.5 (continuous-only) | NO_CKD | -0.029 (tabnet) | -0.0153 (mlp) | 0.0137 |

Status: **DESCRIPTIVE**. FLAG: at 30% missing / ALL5, the full-precision smallest-magnitude degradation belongs to ftt (-0.019576), not mlp (-0.019990) as stated in an earlier CLAUDE.md summary sentence -- both round to -0.020 at 3dp, which is why the earlier session reported mlp; the raw (unrounded) values differ. See the Part 7 FLAGS section.

## C8. Robustness log loss and AUROC degradation diff to logreg, per model, at 30% missing / noise 0.5

See claim_map.csv rows with claim_id=C8 for the per-(perturbation,metric,model) SUPPORTED/PARTLY/NOT SUPPORTED status (CI-excludes-0 in ALL5 and/or NO_CKD; full point/CI values in claim_map_detail.json).

## C9. Explanation agreement vs. within-model stability and chance

Pairs with cross-model Jaccard BELOW within-model stability: 13 of 15.
  - pima_diabetes/logreg-rf: 0.8 < 0.921
  - pima_diabetes/rf-mlp: 0.8 < 0.921
  - heart_cleveland/logreg-rf: 0.5476 < 0.7175
  - heart_cleveland/logreg-mlp: 0.6667 < 0.7175
  - heart_cleveland/rf-mlp: 0.5298 < 0.7175
  - ckd/logreg-rf: 0.5298 < 0.7496
  - ckd/logreg-mlp: 0.7095 < 0.7496
  - ckd/rf-mlp: 0.5095 < 0.7496
  - ilpd/logreg-rf: 0.3631 < 0.6683
  - ilpd/logreg-mlp: 0.3988 < 0.6683
  - ilpd/rf-mlp: 0.4048 < 0.6683
  - breast_wisconsin/logreg-rf: 0.4008 < 0.6344
  - breast_wisconsin/logreg-mlp: 0.3115 < 0.6344

Pairs NOT below within-model stability (exceptions): 2.
  - breast_wisconsin/rf-mlp: 0.644 >= 0.6344
  - pima_diabetes/logreg-mlp: 1.0 >= 0.921

Pairs within 0.05 of chance: 2.
  - ilpd/logreg-rf: jaccard=0.3631, chance=0.3504, diff=0.0127
  - ilpd/logreg-mlp: jaccard=0.3988, chance=0.3504, diff=0.0484

Breast Wisconsin distinct top-5 sets (of 10 splits): {'logreg': 8, 'rf': 1, 'mlp': 9}.

Status: **DESCRIPTIVE**.

## C10. Clinical plausibility precision vs. chance

| dataset | model | precision | CI | chance |
|---|---|---|---|---|
| pima_diabetes | logreg | 1.0 | [1.0, 1.0] | 0.625 |
| pima_diabetes | rf | 0.88 | [0.82, 0.94] | 0.625 |
| pima_diabetes | mlp | 1.0 | [1.0, 1.0] | 0.625 |
| heart_cleveland | logreg | 0.98 | [0.94, 1.0] | 0.6154 |
| heart_cleveland | rf | 0.88 | [0.82, 0.94] | 0.6154 |
| heart_cleveland | mlp | 0.96 | [0.9, 1.0] | 0.6154 |
| ckd | logreg | 0.96 | [0.9, 1.0] | 0.3333 |
| ckd | rf | 1.0 | [1.0, 1.0] | 0.3333 |
| ckd | mlp | 1.0 | [1.0, 1.0] | 0.3333 |
| ilpd | logreg | 0.7 | [0.64, 0.76] | 0.6 |
| ilpd | rf | 0.98 | [0.94, 1.0] | 0.6 |
| ilpd | mlp | 0.64 | [0.56, 0.72] | 0.6 |
| breast_wisconsin | logreg | 0.72 | [0.62, 0.78] | 0.5 |
| breast_wisconsin | rf | 1.0 | [1.0, 1.0] | 0.5 |
| breast_wisconsin | mlp | 0.94 | [0.88, 1.0] | 0.5 |

Datasets flagged non-informative (chance >= 0.60): ['heart_cleveland', 'ilpd', 'pima_diabetes'].

Status: **DESCRIPTIVE**.

## C11. Fit-time ratios to logreg, dataset-averaged ALL5, fraction 1.0

logreg fit time: 0.0244 s. Caveat: fit_time_s measured with 3 to 4 parallel workers; rough, not a controlled single-process timing.

| model | fit time (s) | ratio to logreg |
|---|---|---|
| lgbm | 0.0618 | 2.533x |
| xgb | 0.0658 | 2.697x |
| rf | 0.3607 | 14.783x |
| mlp | 1.4581 | 59.758x |
| ftt | 2.1276 | 87.197x |
| tabnet | 6.6636 | 273.098x |

Status: **DESCRIPTIVE**.

## STATEMENTS THAT MUST NOT BE MADE

- **Deep learning never helps on this benchmark.**
  - Reason: The MLP ties or beats logreg on several metrics (lowest dataset-averaged Brier/log loss at fraction 1.0); only 1/120 Nadeau-Bengio cells is significant either way, so 'never helps' overstates both directions.
- **Any model is 'significantly' better than another, citing the Friedman test or the BLOCK-unit Wilcoxon test.**
  - Reason: The BLOCK unit treats the 10 overlapping-training-set splits as independent, which is anti-conservative (CLAUDE.md decision 6); these tests are descriptive only.
- **Any model is 'significantly' better than another, citing the DATASET-unit Wilcoxon test.**
  - Reason: With 5 (or 4) datasets the DATASET-unit test has essentially no power (minimum possible two-sided p is 0.0625 with 5 datasets, 0.125 with 4); 0/252 DATASET-unit pairs reached Holm significance.
- **TabNet is generally poor as a model family.**
  - Reason: Only under THIS training schedule: the Phase 2c sensitivity check (3 splits, fractions 0.1/1.0 only) showed TabNet's ckd collapse resolves under a longer schedule, so the result is schedule-specific, not a verdict on TabNet in general.
- **Any model is 'best calibrated', based on ECE alone.**
  - Reason: ECE on 60-154-row test folds is noisy and biased; report log loss, Brier and the calibration slope alongside it, with CIs.
- **The SHAP explanations are 'correct' or identify the 'true' important features.**
  - Reason: No direction-of-effect check was performed; features are often correlated (e.g. Breast Wisconsin's mean/SE/worst triplets); the MLP's KernelExplainer uses only a 50-row training background.
- **Any claim about CKD's AUROC DIFFERENCES between models.**
  - Reason: CKD is saturated (mean AUROC 0.9999 for logreg/rf); it separates models poorly on AUROC. Use log loss/Brier/ECE for CKD, and always report ALL5 and NO_CKD separately.
- **Any claim based on the Phase 2a/2b per-split mean ECE (the old Table 3).**
  - Reason: That figure is inflated by the small test folds (60-154 rows); CLAUDE.md decision 11 says to use the Phase 3b pooled ECE values instead.
- **Any claim that post-hoc recalibration (Platt/isotonic) does or does not help in general.**
  - Reason: The Phase 3b post-hoc check is exploratory and cross-fitted on the test predictions themselves (not a pipeline component); isotonic has a clipping artefact that inflates log loss on misses (CLAUDE.md decision 10).

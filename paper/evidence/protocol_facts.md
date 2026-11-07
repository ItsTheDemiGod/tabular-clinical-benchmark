# Protocol facts (generated programmatically, never hand-typed)

## Datasets

| dataset | source | raw_rows | dup_dropped | rows_used | raw_feat | numeric | categorical | encoded_width | positive% | missing% | n_test (min-max) | min_train@0.1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pima_diabetes | OpenML diabetes v1 (id 37) | 768 | 0 | 768 | 8 | 8 | 0 | 8-8 | 34.9 | 10.61 | 153-154 | 61 |
| heart_cleveland | UCI Heart Disease, Cleveland subset (id 45) | 303 | 0 | 303 | 13 | 6 | 7 | 25-25 | 45.9 | 0.15 | 60-61 | 24 |
| ckd | UCI Chronic Kidney Disease (id 336) | 400 | 0 | 400 | 24 | 14 | 10 | 34-34 | 62.5 | 10.54 | 80-80 | 32 |
| ilpd | UCI ILPD Indian Liver Patient Dataset (id 225) | 583 | 13 | 570 | 10 | 9 | 1 | 11-11 | 71.2 | 0.07 | 114-114 | 45 |
| breast_wisconsin | UCI Breast Cancer Wisconsin Diagnostic (id 17) | 569 | 0 | 569 | 30 | 30 | 0 | 30-30 | 37.3 | 0.00 | 113-114 | 45 |

## Protocol constants

- **n_models**: 7
- **models**: ['logreg', 'rf', 'xgb', 'lgbm', 'mlp', 'ftt', 'tabnet']
- **n_datasets**: 5
- **n_datasets_no_ckd**: 4
- **outer_splits_total**: 10
- **outer_splits_description**: 5-fold x 2 repeats (RepeatedStratifiedKFold)
- **fractions**: [0.1, 0.25, 0.5, 1.0]
- **n_tasks**: 350
- **n_result_rows**: 1400
- **tuning_n_trials**: 20
- **tuning_inner_folds**: 3
- **tuning_sampler**: TPESampler
- **tuning_objective**: mean inner-CV AUROC
- **global_seed**: 42
- **deep_max_steps**: 600
- **deep_patience_steps**: 60
- **deep_validation_fraction**: 0.15
- **deep_fixed_steps_no_validation**: 200
- **deep_max_epochs_cap**: 500
- **deep_effective_batch_rule**: min(batch_size, max(8, n_train_part // 4))
- **ece_definition**: Binary expected calibration error on the positive-class probability.

## Search space (per model)

### logreg
| name | type | low | high | log | choices |
|---|---|---|---|---|---|
| C | float | 0.001 | 100.0 | True | None |
### rf
| name | type | low | high | log | choices |
|---|---|---|---|---|---|
| n_estimators | int | 100 | 300 | False | None |
| max_depth | int | 2 | 20 | False | None |
| min_samples_leaf | int | 1 | 10 | False | None |
| max_features | categorical | None | None | False | ['sqrt', 'log2', 0.5] |
### xgb
| name | type | low | high | log | choices |
|---|---|---|---|---|---|
| n_estimators | int | 50 | 400 | False | None |
| max_depth | int | 2 | 8 | False | None |
| learning_rate | float | 0.01 | 0.3 | True | None |
| subsample | float | 0.5 | 1.0 | False | None |
| colsample_bytree | float | 0.5 | 1.0 | False | None |
| reg_lambda | float | 0.01 | 10 | True | None |
| min_child_weight | int | 1 | 10 | False | None |
### lgbm
| name | type | low | high | log | choices |
|---|---|---|---|---|---|
| n_estimators | int | 50 | 400 | False | None |
| num_leaves | int | 4 | 31 | False | None |
| learning_rate | float | 0.01 | 0.3 | True | None |
| min_child_samples | int | 2 | 30 | False | None |
| subsample | float | 0.5 | 1.0 | False | None |
| colsample_bytree | float | 0.5 | 1.0 | False | None |
| reg_lambda | float | 0.01 | 10 | True | None |
### mlp
| name | type | low | high | log | choices |
|---|---|---|---|---|---|
| n_layers | int | 1 | 3 | False | None |
| hidden | int | 16 | 256 | True | None |
| dropout | float | 0.0 | 0.5 | False | None |
| lr | float | 0.0001 | 0.01 | True | None |
| weight_decay | float | 1e-06 | 0.01 | True | None |
| batch_size | categorical | None | None | False | [16, 32, 64, 128] |
### ftt
| name | type | low | high | log | choices |
|---|---|---|---|---|---|
| d_token | categorical | None | None | False | [16, 32, 64] |
| n_blocks | int | 1 | 3 | False | None |
| dropout | float | 0.0 | 0.4 | False | None |
| lr | float | 0.0001 | 0.003 | True | None |
| weight_decay | float | 1e-06 | 0.01 | True | None |
| batch_size | categorical | None | None | False | [16, 32, 64, 128] |
### tabnet
| name | type | low | high | log | choices |
|---|---|---|---|---|---|
| n_d | categorical | None | None | False | [8, 16, 32] |
| n_steps | int | 2 | 5 | False | None |
| gamma | float | 1.0 | 2.0 | False | None |
| lambda_sparse | float | 1e-05 | 0.01 | True | None |
| lr | float | 0.001 | 0.03 | True | None |
| batch_size | categorical | None | None | False | [16, 32, 64, 128] |

## Software / hardware

- **python**: 3.12.4
- **numpy**: 2.5.4
- **pandas**: 3.1.0
- **scipy**: 1.18.1
- **sklearn**: 1.9.1
- **xgboost**: 3.4.1
- **lightgbm**: 4.7.0
- **optuna**: 5.0.0
- **torch**: 2.15.0+cu132
- **joblib**: 1.6.0
- **matplotlib**: 3.11.2
- **pytorch_tabnet**: 4.1.0
- **shap**: 0.54.1
- **torch_cuda**: 13.2
- **gpu_name**: NVIDIA GeForce RTX 4060 Laptop GPU
- **nvidia_smi**: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB, 617.14
- **environment_split**: .venv = classical only (no torch); .venv-deep = classical + torch + pytorch-tabnet + shap

## Fit / predict time (fraction 1.0, dataset-averaged)

| model | mean_fit_time_s | mean_predict_time_s |
|---|---|---|
| logreg | 0.024 | 0.0085 |
| rf | 0.361 | 0.0281 |
| xgb | 0.066 | 0.0067 |
| lgbm | 0.062 | 0.0091 |
| mlp | 1.458 | 0.0097 |
| ftt | 2.128 | 0.0232 |
| tabnet | 6.664 | 0.0457 |

Note: fit/predict times measured with 3 to 4 parallel workers; rough.


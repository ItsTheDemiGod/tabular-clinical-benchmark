# tabular-clinical-benchmark

Supporting code for the research paper **"When Does Deep Learning Help on
Small Clinical Tabular Data?"**

## Project goal

This project benchmarks the performance of deep learning versus classical
machine learning models on small, real-world clinical tabular datasets,
across varying training-set sizes. It is built in phases:

- **Phase 1 (this phase):** project setup, dataset loading, and a
  leakage-safe preprocessing layer. No models, hyperparameter tuning, or
  experiments yet.
- **Later phases:** model implementations, tuning, and the experiments
  themselves.

## Datasets

Five small clinical binary-classification datasets, loaded via
`src.data.load_dataset(name)`:

| name               | source                                             | rows | features |
|--------------------|-----------------------------------------------------|------|----------|
| `pima_diabetes`    | OpenML "diabetes" v1 (OpenML id 37)                 | 768  | 8        |
| `heart_cleveland`  | UCI "Heart Disease" (ucimlrepo id 45)               | 303  | 13       |
| `ckd`              | UCI "Chronic Kidney Disease" (ucimlrepo id 336)     | 400  | 24       |
| `ilpd`             | UCI "ILPD (Indian Liver Patient Dataset)" (id 225)  | 570  | 10       |
| `breast_wisconsin` | UCI "Breast Cancer Wisconsin (Diagnostic)" (id 17)  | 569  | 30       |

Raw downloads are cached under `data/` so repeated loads (and the test
suite) don't hit the network every time. See `results/dataset_summary.csv`
for the full summary table (rows, duplicates, features, numeric/categorical
counts, positive-class fraction, missing-value fraction, conflicting-label
rows). Regenerate it with `python -m src.make_dataset_summary`.

## Environment setup

```bash
# From the project root:
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the tests

```bash
pytest tests/ -v
```

## Repository layout

```
tabular-clinical-benchmark/
  README.md
  CLAUDE.md               # project context and working protocol for Claude Code
  requirements.txt        # pins .venv (classical models only, no torch)
  requirements-deep.txt   # pins .venv-deep (classical models + torch + pytorch-tabnet)
  .gitignore
  src/
    __init__.py
    data.py              # load_dataset(name) -> (X, y, meta)
    preprocessing.py     # build_preprocessor(...) -> unfitted ColumnTransformer
    utils.py             # set_seed, make_splits, subsample_train
    make_dataset_summary.py  # regenerates results/dataset_summary.csv
    metrics.py           # compute_metrics, prediction_diversity
    models.py            # MODEL_REGISTRY, build_pipeline, effective_params
    tuning.py            # tune_model
    runner.py            # run_task
    deep_models.py       # MLP / FT-Transformer / TabNet wrappers (Phase 2b)
  scripts/
    run_phase2a.py        # classical-model command-line runner (smoke test and full run)
    run_phase2b.py        # deep-model command-line runner
    report_phase2a.py     # verification checks, tables and flags
    phase2c_diagnostics.py  # Phase 2c: endpoints / tabnet / sensitivity / sensitivity-report / xgb-wide-range / all
    phase3a_stats.py        # Phase 3a: tests / ci / figures / tables / all
    phase3b_calibration.py  # Phase 3b: verify / metrics / posthoc / figures / tables / all
    phase3c_robustness.py   # Phase 3c: verify / run / analyze / figures / tables / all
  analysis/                # Phase 3a/3b/3c helper package (ranks, Friedman/Wilcoxon/Nadeau-Bengio, bootstrap, tables, plotting, calibration, robustness)
  data/                   # cached raw dataset downloads (*_raw.pkl), tracked in git
  results/
    raw/                  # one CSV per (dataset, model, split) task
    tuning/                # one JSON per task: tuned hyperparameters
    predictions/           # one .npz per task: test_idx, y_true, prob_{fraction}
    models/                 # one .joblib per task (fraction 1.0 only), tracked in git
    diagnostics/            # one JSON per deep-model task: training_info_ per fraction
    phase2a_all_runs.csv    # frozen Phase 2a/2a.1 classical snapshot (800 rows)
    all_runs.csv            # Phase 2a+2b aggregate (1400 rows)
    dataset_summary.csv     # per-dataset summary table
    MANIFEST_sha256.txt      # SHA-256 of every file above, for integrity checks
    phase2c/                # Phase 2c diagnostic outputs (see "Phase 2c" below)
    phase3a/                 # Phase 3a stats/CI/figures/tables outputs (see "Phase 3a" below)
    phase3b/                 # Phase 3b calibration outputs (see "Phase 3b" below)
    phase3c/                 # Phase 3c robustness outputs (see "Phase 3c" below)
  results_smoke/            # smoke-test output (classical), gitignored, disposable
  results_smoke_2b/          # smoke-test output (deep models), gitignored, disposable
  tests/
    test_data.py
    test_preprocessing.py
    test_metrics.py
    test_models.py
    test_runner.py
    test_deep_models.py      # needs torch; run from .venv-deep
    test_phase2c.py           # fast, no GPU fit required
    test_phase3a.py           # fast, no GPU fit required
    test_phase3b.py           # fast, no GPU fit required
    test_phase3c.py           # fast, no GPU fit required
  notebooks/                 # reserved for future exploratory analysis
```

## Phase 1 complete

- Project structure, virtual environment, and pinned `requirements.txt`.
- `src/data.py`: `load_dataset(name)` for all five datasets, with identity
  checks against the live OpenML/UCI sources (fails loudly on a mismatch
  rather than silently substituting data), raw-download caching, target
  binarization, and full `meta` dicts (row/feature counts, class balance,
  numeric/categorical column lists, per-column missing fraction, source).
- `src/preprocessing.py`: `build_preprocessor(numeric_cols, categorical_cols,
  scale)` returning an unfitted `ColumnTransformer` (median/most-frequent
  imputation, optional scaling, one-hot encoding with
  `handle_unknown="ignore"`) designed to be fit only inside a `Pipeline` on a
  single training fold.
- `src/utils.py`: `set_seed`, `make_splits` (RepeatedStratifiedKFold), and
  `subsample_train` (stratified subsampling with a minimum-2-per-class
  guarantee).
- 17 passing tests covering dataset loading, split structure, subsampling
  behavior, and a leakage test verifying imputation/scaling statistics are
  computed from the training fold only.
- `results/dataset_summary.csv` generated from live data.

Not yet implemented (future phases): models, hyperparameter tuning,
experiment loops, and result analysis.

## Phase 1.1 patch

Three data-quality fixes, all applied in `load_dataset` after the cache is read,
so the raw downloads in `data/` stay untouched.

1. **Pima hidden missing values.** In `plas`, `pres`, `skin`, `insu` and `mass`,
   a 0 means "not recorded" and is physiologically impossible, so it now becomes
   NaN (652 values in total, listed in `meta["zeros_replaced_with_nan"]`). Left as
   zeros, these looked like real measurements and hid the missingness: 48.7% of
   `insu` and 29.6% of `skin` are actually missing. `preg`, `pedi` and `age` are
   unchanged, because 0 pregnancies is a real value.
2. **Cleveland categorical columns.** `sex`, `cp`, `fbs`, `restecg`, `exang`,
   `slope` and `thal` are category codes, not quantities, but were being scaled as
   numbers. They are now strings and go to the one-hot branch, with missing values
   kept as NaN. `age`, `trestbps`, `chol`, `thalach`, `oldpeak` and `ca` stay numeric.
3. **Duplicate rows.** Exact duplicates (all features plus the target) are dropped
   in every dataset, keeping the first, because a copy could land in both the train
   and test folds and inflate scores. Only ILPD had any: 13 rows, taking it from 583
   to 570. Rows with identical features but conflicting labels would be kept and
   counted, with a warning; there are currently none.

## Phase 1.2 patch

**Fix: CKD preprocessor crash.** `build_preprocessor` raised
`TypeError: boolean value of NA is ambiguous` when fit on `ckd`, because the loader
stored categorical columns in pandas `string` dtype, whose `pd.NA` missing marker
`SimpleImputer` cannot handle. A shared helper in `load_dataset`,
`_normalize_categoricals`, now stores categorical columns in all five datasets as
plain object dtype with `np.nan` for missing values. Missing counts, numeric
columns, row counts and the raw caches are unchanged. The helper raises an error if
it finds a placeholder string (`"nan"`, `"None"`, `"<NA>"`, `"?"`), so that a
placeholder is never silently counted as missing or kept as a category.

**Why `sg`, `al` and `su` stay numeric.** These CKD columns are ordered scales with
5 to 6 distinct values, and one-hot encoding would discard the order. They are
listed in `meta["ordinal_as_numeric"]`.

**New tests:**

- `test_preprocessor_fits_on_all_datasets`: 5 datasets x scale on/off. Fits on the
  train fold only and checks both folds for finite, numeric output of the right size.
  No earlier test fit the preprocessor on CKD, which is how the crash went unnoticed.
- `test_preprocessor_fits_on_small_training_fraction`: fits on a 10% training subset
  and transforms the full test fold, including categories the subset never saw.
- `test_categoricals_are_object_dtype_without_pd_na`: checks categorical storage in
  all five datasets.
- `test_ckd_ordinal_columns_numeric`: checks where `sg`, `al` and `su` are routed.
- Two synthetic `_normalize_categoricals` tests: missing markers and placeholder
  rejection, neither of which the real data fully exercises.

## Phase 2a

**What was built:**

- `src/metrics.py`: `compute_metrics` (AUROC, F1, balanced accuracy, Brier, log loss,
  ECE).
- `src/models.py`: `MODEL_REGISTRY` with logistic regression, random forest, XGBoost and
  LightGBM, plus `build_pipeline`.
- `src/tuning.py`: `tune_model`, the shared Optuna setup. Every model gets 20 TPE trials
  scored by mean AUROC over 3 inner folds, using training data only.
- `src/runner.py`: `run_task`, which runs one (dataset, model, outer split) task and
  supports resume.
- `scripts/run_phase2a.py`: the command-line runner.
- `scripts/report_phase2a.py`: read-only verification checks, tables and flags.

**How to run (from the project root, with the virtual environment active):**

```bash
# Smoke test: pima_diabetes, logreg and xgb, split 0, 3 trials, writes to results_smoke/
python scripts/run_phase2a.py --smoke

# Full run: 200 tasks, writes to results/. Re-run the same command to resume after an
# interruption; finished tasks are skipped.
python scripts/run_phase2a.py --workers 4

# Verification, tables and flags for the full run
python scripts/report_phase2a.py
```

**ECE definition.** Binary expected calibration error on the positive-class probability,
with 10 equal-width bins over [0, 1]:
`ECE = sum over bins of (n_bin / n) * |mean(y_true in bin) - mean(y_prob in bin)|`,
skipping empty bins. Bins include their lower edge and exclude their upper edge, except
the last, which also includes 1.0.

**Conventions:**

- Hyperparameters are tuned once, on the full (100%) training fold, and reused for the
  smaller fractions. Phase 2a.1 scales the leaf-size settings; everything else is reused
  unchanged.
- No model uses native missing-value handling. All four receive the output of the shared
  preprocessor (median imputation), which is dense and contains no NaN. `class_weight` and
  early stopping are not used.
- Model input widths come from the fitted preprocessor output, never from a hardcoded
  number, because one-hot width depends on the categories in the training data.

## Phase 2a.1 patch

**Why leaf-size settings are scaled.** Leaf-size constraints tuned at 100% training size
are measured in absolute rows or hessian units. Reused on a small subset, they can forbid
every split, and the model then predicts the same value for every patient (AUROC exactly
0.5). `effective_params(model_name, params, fraction)` keeps their relative size constant:

| model | rule |
|-------|------|
| `xgb` | `min_child_weight = max(1.0, min_child_weight * fraction)`, an unrounded float |
| `lgbm` | `min_child_samples = max(1, round(min_child_samples * fraction))` |
| `rf` | `min_samples_leaf = max(1, round(min_samples_leaf * fraction))` |
| `logreg` | unchanged, because `C` already scales with sample size |

At fraction 1.0 the parameters are exactly the tuned ones. `round` is Python's built-in, so
exact halves go to the nearest even number. Tuning itself is unchanged, and
`results/tuning/*.json` holds the tuned values.

**New columns.** `results/raw/*.csv` gains `n_unique_probs` (unique predicted probabilities
after rounding to 8 decimals) and `constant_prediction` (true when that number is 1), right
after `predict_time_s`. Both are empty on failed rows. A constant row keeps status `ok`,
prints a warning, and is listed at the end of the run. CSVs without these columns count as
incomplete and rerun on resume.

**New `params_json` format.** A JSON object with exactly three keys: `tuned` (from tuning
at 100%), `effective` (actually used for that fraction) and `fraction`.

**Verbose option.** `load_dataset(name, verbose=False)` prints nothing, except the warning
about conflicting labels, which always prints. The runner uses `verbose=False`, and the
default behaviour is unchanged.

**Calibration convention.** A constant prediction can look well calibrated while being
unable to rank patients, so analyses of calibration must exclude or separately flag rows
where `constant_prediction` is true. The full Phase 2a run has no such rows.

## Phase 2b

**What was built.** Three deep models were added to the same harness, tuning protocol and
result-CSV format as Phase 2a/2a.1: MLP, FT-Transformer and TabNet. They receive the output
of the same shared preprocessor (median imputation, one-hot, scaling) as the classical
models, so the comparison stays fair. TabPFN is **not** part of this phase: it is not
imported, installed, configured or tested anywhere in this codebase.

**Models and search spaces** (Optuna, `TPESampler`, 20 trials, 3 inner folds — identical
budget to the classical models):

| model | hyperparameters tuned |
|-------|------------------------|
| `mlp` | `n_layers` (1-3), `hidden` (16-256, log), `dropout` (0-0.5), `lr` (1e-4 to 1e-2, log), `weight_decay` (1e-6 to 1e-2, log), `batch_size` (16/32/64/128) |
| `ftt` | `d_token` (16/32/64), `n_blocks` (1-3), `dropout` (0-0.4), `lr` (1e-4 to 3e-3, log), `weight_decay` (1e-6 to 1e-2, log), `batch_size` (16/32/64/128) |
| `tabnet` | `n_d` = `n_a` (8/16/32), `n_steps` (2-5), `gamma` (1.0-2.0), `lambda_sparse` (1e-5 to 1e-2, log), `lr` (1e-3 to 3e-2, log), `batch_size` (16/32/64/128) |

MLP is a plain stack of `n_layers` equal-width `Linear -> ReLU -> Dropout` blocks (no batch
norm — batches on the small training fractions are tiny) ending in one logit. FT-Transformer
follows Gorishniy et al. (2021): a learnable CLS token, `n_blocks` pre-norm Transformer blocks
(4 attention heads, feed-forward width `2 * d_token`), and a linear head on the CLS token.
**Simplification:** every encoded column (including one-hot columns) is tokenized as a
numeric feature — there is no separate categorical-embedding path. TabNet wraps
`pytorch_tabnet.tab_model.TabNetClassifier` directly.

**Training-schedule rule.** `training_schedule(n_train_part, batch_size)` keeps the number of
gradient steps roughly constant across the four training fractions, instead of a constant
epoch count: `effective_batch = min(batch_size, max(8, n_train_part // 4))`,
`steps_per_epoch = ceil(n_train_part / effective_batch)`, `max_epochs = min(500,
ceil(600 / steps_per_epoch))`, `patience_epochs = max(5, ceil(60 / steps_per_epoch))`. Early
stopping monitors validation log loss on a 15%-of-training-fold split
(`split_validation`), restoring the best weights at the end. When a class has fewer than 4
samples in the fit set no validation split is possible, so the model trains for a fixed 200
steps instead and records `early_stopping=False`. This is an asymmetry with the tree models
(`rf`/`xgb`/`lgbm`) and `logreg`, which have no early stopping and no validation carve-out at
all — their full tuned training fold is used for fitting at every fraction.

**TabNet singleton-batch rule.** `pytorch_tabnet`'s internal network uses `BatchNorm1d`, which
raises on a batch of exactly 1 sample in training mode. Because `effective_batch` is derived
from the training-fold size and the tuned `batch_size`, a remainder of exactly 1
(`n_train_part % effective_batch == 1`) produces a final mini-batch of 1 row and crashes.
`_avoid_singleton_remainder(total, chunk)` bumps `chunk` up by one whenever this would happen,
applied to both `effective_batch` (against `n_train_part`) and `virtual_batch_size` (against
`effective_batch`, Ghost Batch Normalization's internal chunk size). This was found live
during timing runs (not by the unit tests' tiny synthetic datasets) on
`breast_wisconsin__tabnet` and `ilpd__tabnet`.

**Environments.** The deep models run only in `.venv-deep` (torch 2.15.0+cu132,
pytorch-tabnet 4.1.0, plus the same classical package versions as `.venv`). The four
classical models run identically in either environment; `requirements.txt` pins `.venv`
(classical-only, no torch) and `requirements-deep.txt` pins the full `.venv-deep` freeze.
`src/models.py` imports `src.deep_models` (and therefore torch) lazily, only inside the three
new models' `_make_*` functions, so the classical registry, classical pipelines and classical
tests work unchanged in an environment without torch at all — verified by a dedicated test
that checks `torch` is absent from `sys.modules` after building and fitting the four
classical pipelines in a fresh subprocess.

GPU: NVIDIA RTX 4060 (CUDA 13.2, driver matching torch 2.15.0+cu132).

**How to run (from the project root, `.venv-deep\Scripts\python.exe` explicitly):**

```bash
# Smoke test: pima_diabetes, all three deep models, split 0, 3 trials, writes to results_smoke_2b/
.venv-deep\Scripts\python.exe scripts/run_phase2b.py --smoke

# Full run: 150 deep-model tasks (5 datasets x 3 models x 10 splits), writes into results/
# alongside the classical results. Re-run the same command to resume after an interruption;
# finished tasks (raw + tuning + predictions + models + diagnostics all present) are skipped.
.venv-deep\Scripts\python.exe scripts/run_phase2b.py --workers 3

# Or split into two resumable pieces, run separately:
.venv-deep\Scripts\python.exe scripts/run_phase2b.py --models mlp ftt --workers 3
.venv-deep\Scripts\python.exe scripts/run_phase2b.py --models tabnet --workers 3
```

**New result files.** `results/diagnostics/{dataset}__{model}__split{split}.json`, written
only for the three deep models, keyed by fraction, holding each `training_info_` dict
(`n_fit`, `n_train_part`, `n_val`, `early_stopping`, `effective_batch`, `steps_per_epoch`,
`max_epochs`, `patience_epochs`, `epochs_run`, `best_epoch`, `best_val_logloss`,
`n_parameters`, `device`, `peak_gpu_mem_mb`, `nan_loss`). The `results/raw/*.csv` schema is
unchanged from Phase 2a.1. `results/all_runs.csv` aggregates all seven models'
`results/raw/*.csv` files (1400 rows: 350 tasks x 4 fractions); `results/phase2a_all_runs.csv`
is untouched.

**TabPFN.** Not part of this phase. It is excluded from every environment-verification step,
is not imported or invoked anywhere in `src/` or `scripts/`, and no model weights or API
tokens were obtained for it.

## Phase 2c

Read-only diagnostics on the Phase 2a/2b results, plus a schedule-sensitivity check for the
deep models. Phase 2c never modifies `results/raw`, `results/tuning`, `results/predictions`,
`results/models`, `results/diagnostics`, `results/phase2a_all_runs.csv` or `results/all_runs.csv`,
and never changes anything in `src/`. All new outputs are under `results/phase2c/`.

**What was run**, via `scripts/phase2c_diagnostics.py` (`.venv-deep\Scripts\python.exe`):

```bash
.venv-deep\Scripts\python.exe scripts/phase2c_diagnostics.py endpoints
.venv-deep\Scripts\python.exe scripts/phase2c_diagnostics.py tabnet
.venv-deep\Scripts\python.exe scripts/phase2c_diagnostics.py xgb-wide-range
.venv-deep\Scripts\python.exe scripts/phase2c_diagnostics.py sensitivity
.venv-deep\Scripts\python.exe scripts/phase2c_diagnostics.py sensitivity-report
```

- **`endpoints`** (read-only): replays every model's `suggest_params` against a recording
  stand-in for the Optuna trial to recover each hyperparameter's search range, then classifies
  every tuned value in `results/tuning/*.json` as at the lower endpoint, the upper endpoint, or
  neither (integers/numeric-categorical choices: exact equality; continuous: within 2% of the
  range, on a log scale for log-uniform parameters). Writes `results/phase2c/endpoints_overall.csv`
  and `endpoints_by_dataset.csv`.
- **`tabnet`** (read-only): per-split AUROC tables for tabnet/logreg/mlp on `ckd` and
  `heart_cleveland` (`results/phase2c/tabnet_per_split.csv`), plus full tuning/diagnostics/prediction
  detail for every "collapsed" TabNet split (AUROC < 0.90 on ckd or < 0.75 on heart_cleveland at
  fraction 1.0) in `tabnet_collapsed_splits.csv`.
- **`xgb-wide-range`** (descriptive only, does not touch `src/models.py`): re-tunes xgb with
  `min_child_weight` searched log-uniformly over 0.1-10 instead of 1-10, on 3 datasets x 3 splits,
  and compares test AUROC at fractions 0.1 and 1.0 to the original `results/raw` values. Writes
  `results/phase2c/xgb_wide_range.csv`. 3 splits are too few for a conclusion; the results are
  descriptive only.
- **`sensitivity`**: re-fits mlp/ftt/tabnet with their already-tuned hyperparameters (no tuning)
  on all 5 datasets, splits 0-2, fractions 0.1 and 1.0, under 4 schedule variants (patching
  `src/deep_models.MAX_STEPS`/`PATIENCE_STEPS`/`VALIDATION_FRACTION` in-process only, always
  restored afterwards): baseline (600/60/0.15, the current production schedule), `A_long`
  (2000/150/0.15), `B_bigval` (600/60/0.30), `C_long_bigval` (2000/150/0.30). Before running the
  full 360 fits it (1) reproduces the 90 baseline fits and checks them against `results/raw`
  (max abs AUROC difference was 0.0 across all 90), and (2) times 6 probe fits and STOPs if the
  extrapolated total exceeds 2 hours (estimated 1.64h; actual run completed all 360/360 with
  `status=ok`, 0 failures, 0 NaN losses, 0 constant predictions). Resumable: re-running skips
  rows already in `results/phase2c/sensitivity_runs.csv`.
- **`sensitivity-report`** (read-only): rebuilds the five summary tables (S1-S5: baseline check,
  per-model/fraction mean metrics per variant, per-dataset AUROC with split-to-split std, early
  stopping/cap/timing behaviour, failure counts) from `sensitivity_runs.csv` without re-fitting
  anything, as `results/phase2c/sensitivity_report_S1.csv` .. `S5.csv`.
- **`all`** runs `endpoints` + `tabnet` only (`sensitivity` and `xgb-wide-range` are run on their
  own because of the STOP gates above).

**Headline findings** (see `results/phase2c/` for the full tables): `xgb`'s `min_child_weight`
sits at the lower end of its tuned range (1) in 23/50 classical tasks and `rf`'s
`min_samples_leaf` in 18/50 -- a mild constraint, not a hard floor (see CLAUDE.md section 7).
TabNet's known "collapse" on `ckd` at fraction 1.0 (mean AUROC 0.919, split std 0.138 under the
current schedule) disappears under all three longer/bigger-validation variants (mean AUROC
0.999-1.000, std <= 0.0015) -- but this was only tested on splits 0-2, not the full 10, and MLP
and FT-Transformer barely move either way (all changes under 0.01 AUROC). Deciding whether to
rerun the full deep-model benchmark with a different schedule is a decision for Demian; see
CLAUDE.md section 8.

**Git.** The project is under git, with a GitHub remote `origin` added 2026-11-07 (see "Code
availability" below). The first push is Demian's to make; after that, Claude Code may push with
plain `git push origin main` at the end of a phase, only once tests pass and the secret scan is
clean, never with `--force` or to any branch but `main`, and never rewriting history.
`results/models/` and `data/*.pkl` are tracked (not gitignored) as of 2026-11-07.
`results/MANIFEST_sha256.txt` (created 2026-11-07, 1552 file hashes) records the SHA-256 of
every file in `results/raw`, `tuning`, `predictions`, `models`, `diagnostics`, plus
`phase2a_all_runs.csv` and `all_runs.csv`, so a future session can verify nothing in those
folders has changed.

## Code availability

This repository is at <https://github.com/ItsTheDemiGod/tabular-clinical-benchmark.git>.
`results/models/` (the saved fraction-1.0 pipelines) and the raw dataset caches in `data/` are
included in the repository. Not included: the `.venv`/`.venv-deep` Python environments,
`results_smoke`/`results_smoke_2b` (disposable smoke-test output), and local editor/session
settings (`.claude/settings.local.json`).

## Phase 3a

Statistical analysis of the existing `results/all_runs.csv` (1400 rows), plus learning-curve
figures and the main LaTeX tables. Phase 3a only reads `results/all_runs.csv` (and, once, for a
precondition check, `results/raw/*.csv`); it never modifies any existing result file and never
fits a model. New code lives in `analysis/` (helper package) and `scripts/phase3a_stats.py`
(CLI); all new outputs are under `results/phase3a/`.

**What was computed**, via `.venv-deep\Scripts\python.exe scripts/phase3a_stats.py <subcommand>`:

```bash
.venv-deep\Scripts\python.exe scripts/phase3a_stats.py tests     # Friedman/Iman-Davenport, Holm-Wilcoxon, Nadeau-Bengio
.venv-deep\Scripts\python.exe scripts/phase3a_stats.py ci        # bootstrap confidence intervals (seed 42, B=5000)
.venv-deep\Scripts\python.exe scripts/phase3a_stats.py figures   # learning curves, CD diagrams, diff-vs-logreg plots
.venv-deep\Scripts\python.exe scripts/phase3a_stats.py tables    # Tables 1-5, CSV + LaTeX
.venv-deep\Scripts\python.exe scripts/phase3a_stats.py all       # all four, in order
```

- **`tests`**: for AUROC, log loss and Brier, at fractions 1.0 and 0.1, over ALL5 (all 5
  datasets) and NO_CKD (CKD excluded): per-block and per-dataset mean ranks, a Friedman test with
  the Iman-Davenport F correction, and all 21 pairwise Wilcoxon signed-rank tests with a
  hand-implemented Holm step-down correction. Also a Nadeau-Bengio corrected resampled paired
  t-test (logreg vs. each of the other 6 models) per dataset. Writes `friedman_summary.csv`,
  `pairwise_wilcoxon.csv`, `corrected_ttest.csv`, `stats_notes.md`.
- **`ci`**: 95% percentile bootstrap CIs (seed 42, 5000 resamples) per model/fraction/metric, per
  dataset and dataset-averaged (nested resampling), plus the paired difference to logreg using the
  *same* resampled split indices for both. Writes `ci_per_dataset.csv`, `ci_dataset_averaged.csv`,
  `ci_diff_vs_logreg.csv`.
- **`figures`**: learning curves (AUROC/log loss/Brier vs. training fraction, 95% CI bands),
  Wilcoxon-Holm critical-difference-style diagrams (AUROC, BLOCK unit), and dot-and-whisker plots
  of the AUROC difference to logreg. PNG (300 dpi) and PDF, captions in `figure_captions.md`.
- **`tables`**: Tables 1-5 (mean +/- std AUROC/log loss at fraction 1.0, mean AUROC by fraction,
  mean rank by fraction, and mean Brier/log loss/ECE/fit time/predict time at fraction 1.0), each
  as `.csv` and a hand-written booktabs `.tex`. Notes in `table_captions.md`.

**Unit-of-analysis caveat** (see `results/phase3a/stats_notes.md` for the full statement): the
BLOCK unit (one row per dataset x split, 50 or 40 rows) treats the 10 splits of a dataset as
independent, although they come from repeated cross-validation with overlapping training sets, so
BLOCK-unit p-values are anti-conservative. The DATASET unit (5 or 4 dataset-level means) is valid
but has almost no power (the smallest possible two-sided Wilcoxon p-value is 0.0625 with 5 cells,
0.125 with 4), so no DATASET-unit pair is ever significant at 0.05 after Holm. Neither file states
that a model is "significantly better" -- only "Holm-adjusted p < 0.05 on the BLOCK unit".

**Decisions recorded after Phase 2c/3a** (2026-11-07) are in CLAUDE.md sections 6-8: no
deep-model rerun with a longer schedule (the baseline schedule stays primary); the xgb
`min_child_weight` wide-range check showed no consistent AUROC change; the endpoint findings by
the same-endpoint rule; TabNet's heart_cleveland collapses are not fixed by a longer schedule; the
"MLP hits a hard 500-epoch cap" claim is unsupported by the actual mean epochs run and is recorded
as unexplained.

## Phase 3b

Calibration analysis of the existing predictions in `results/predictions/*.npz` (350 files: 5
datasets x 7 models x 10 splits). Phase 3b only reads `results/predictions`, `results/all_runs.csv`
and (read-only) `results/phase2c/sensitivity_runs.csv` / `results/diagnostics/*.json`; it never
modifies any existing result file and never fits or reruns any of the 7 benchmark models (the
cross-fitted post-hoc recalibration check is a separate, descriptive, in-memory analysis, not a
pipeline change). New code: `analysis/calibration.py` (helper module) and
`scripts/phase3b_calibration.py` (CLI). All new outputs are under `results/phase3b/`.

**Pooling caveat:** every pooled analysis below pools the 10 CV splits of a dataset, so each test
row appears **twice** in the pool (once per repeat of the 5-fold cross-validation) -- pooled
analyses use 2 x n_rows predictions per (dataset, model, fraction), not n_rows.

**What was computed**, via `.venv-deep\Scripts\python.exe scripts/phase3b_calibration.py <subcommand>`:

```bash
.venv-deep\Scripts\python.exe scripts/phase3b_calibration.py verify   # STOP-gated recomputation check
.venv-deep\Scripts\python.exe scripts/phase3b_calibration.py metrics  # pooled ECE/slope/intercept + bootstrap CIs
.venv-deep\Scripts\python.exe scripts/phase3b_calibration.py posthoc  # cross-fitted Platt/isotonic check
.venv-deep\Scripts\python.exe scripts/phase3b_calibration.py figures  # reliability diagrams, slope plot, post-hoc bars
.venv-deep\Scripts\python.exe scripts/phase3b_calibration.py tables   # Tables 4b, 6, 7, 8
.venv-deep\Scripts\python.exe scripts/phase3b_calibration.py all      # all five, in order
```

- **`verify`** (STOP-gated): loads every prediction file, checks array presence/shape/range,
  confirms `y_true` matches `y[test_idx]` from a fresh `load_dataset` call, and recomputes
  auroc/brier/log_loss/ece from the stored probabilities for all 1400 (file x fraction)
  combinations, comparing to `results/all_runs.csv`. STOPs on any structural/alignment problem or
  any recomputed difference above 1e-6 (none found: max absolute difference was exactly 0.0 for
  every metric).
- **`metrics`**: per (dataset, model, fraction) pooled `ece_ew` (equal-width, identical to
  `src.metrics`), `ece_em` (equal-mass, 10 groups), the Platt calibration slope/intercept (MLE via
  `scipy.optimize`, undefined/NaN when the two classes are exactly separable in logit space), and
  95% bootstrap CIs (seed 42, B=2000, resampling the 10 splits). Also dataset-averaged values
  (ALL5/NO_CKD) and the paired bootstrap difference to logreg for pooled `ece_ew`. Writes
  `calibration_metrics.csv`, `calibration_bootstrap_ci.csv`, `calibration_averaged.csv`,
  `calibration_brier_logloss_averaged.csv`, `calibration_diff_vs_logreg.csv`.
- **`posthoc`** (descriptive, NOT part of the pipeline): inner stratified 5-fold cross-fitting on
  each split's test predictions (seed 42 + split_idx); each fold's Platt/isotonic calibrator is
  fit on the other 4 folds (about 48-123 rows) and applied only to the held-out fold, with an
  identity-map fallback when a fit fold has a single class (or, for Platt, is separable). Writes
  `posthoc_per_split.csv` and `posthoc_summary.csv`.
- **`figures`**: Figure 4/4b (equal-mass reliability diagrams, NO_CKD pooled, fractions 1.0/0.1),
  Figure 4c (per-dataset compact grid, fraction 1.0), Figure 5 (pooled calibration slope with 95%
  CIs, coloured by model, one marker per dataset), Figure 6 (post-hoc Brier/log-loss change,
  ALL5/NO_CKD). PNG (300 dpi) and PDF, captions in `figure_captions.md`.
- **`tables`**: Table 6/7 (fraction 1.0/0.1: Brier, log loss, ece_ew, ece_em, NO_CKD slope and
  intercept), Table 8 (post-hoc Brier/log-loss/ece_ew change and fallback counts), Table 4b (mean
  rank at fractions 0.25/0.5, BLOCK unit, descriptive only, reusing Phase 3a's ranking helper).
  Notes in `table_captions.md` and `calibration_notes.md`.

**Calibration caveats** (full statement in `results/phase3b/calibration_notes.md`): ECE on small
samples is biased and noisy; equal-width and equal-mass ECE can disagree; the calibration slope
and intercept are undefined when the two classes are exactly separable in logit space (this never
happened at the pooled point-estimate level in this run, even for CKD, but several CKD bootstrap
replicates were separable and dropped from that cell's CI); the post-hoc analysis is cross-fitted
on the test predictions themselves and is descriptive only, not a pipeline component, and
isotonic regression's AUROC can change through tie-breaking. No file in `results/phase3b/` states
that a model is "significantly better" or "better calibrated" -- only numbers are reported.

**Epoch-cap re-check** (`results/phase3b/epoch_cap_check.csv`, see CLAUDE.md section 7 item 5):
under the baseline training schedule, every mlp cap hit is a "budget hit" (the 600-step budget
translated into a per-task `max_epochs` below 500); the literal 500-epoch hard limit is only ever
hit under the Phase 2c sensitivity check's longer-budget variants, and only for mlp/ftt at
fraction 0.1.

## Phase 3c

Robustness analysis on the SAVED fraction-1.0 pipelines in `results/models/*.joblib` (350
fitted pipelines). Phase 3c perturbs only the TEST features and never fits, refits or tunes
any model; it only reads `results/models`, `results/predictions`, `results/all_runs.csv` and
(via `load_dataset`/`make_splits`) the raw datasets. New code: `analysis/robustness.py` (helper
module) and `scripts/phase3c_robustness.py` (CLI). All new outputs are under `results/phase3c/`.

**Design:** two perturbation kinds, each with 3 levels and 5 independent draws per
(dataset, split, kind, level) -- the SAME perturbed test matrix is reused across all 7 models for
a given draw (paired design; the perturbation functions take no model argument).
- **missing**: every cell (numeric and categorical) is independently set to NaN with probability
  r in (0.10, 0.20, 0.30) -- Missing Completely At Random. Cells already missing stay missing.
- **noise**: Gaussian noise `N(0, (s * sigma_j)^2)` is added to every non-missing value of each
  numeric column j, s in (0.10, 0.25, 0.50), with sigma_j the TRAINING-fold standard deviation
  (ddof=0) of that column -- never the test rows. CKD's `sg`/`al`/`su` and Cleveland's `ca` are
  ordinal numeric scales and receive continuous noise like every other numeric column.

**What was computed**, via `.venv-deep\Scripts\python.exe scripts/phase3c_robustness.py <subcommand>`:

```bash
.venv-deep\Scripts\python.exe scripts/phase3c_robustness.py verify   # STOP-gated recomputation check
.venv-deep\Scripts\python.exe scripts/phase3c_robustness.py run      # timing-gated, resumable, writes robustness_runs.csv
.venv-deep\Scripts\python.exe scripts/phase3c_robustness.py analyze  # per-dataset/averaged/diff-vs-logreg tables
.venv-deep\Scripts\python.exe scripts/phase3c_robustness.py figures  # degradation curves, diff-vs-logreg plot
.venv-deep\Scripts\python.exe scripts/phase3c_robustness.py tables   # Tables 9-12
.venv-deep\Scripts\python.exe scripts/phase3c_robustness.py all      # all five, in order (verify first STOPs before the rest if it fails)
```

- **`verify`** (STOP-gated): loads every one of the 350 pipelines, predicts on the CLEAN test
  rows and compares to the stored `prob_1.0`, and checks the split/label alignment against a
  fresh `load_dataset`/`make_splits` call. STOPs on any structural/alignment problem or any
  difference above 1e-3 (none found: max difference was exactly 0.0 for every model).
- **`run`**: times one full 31-prediction task (1 clean + 3 missing levels x 5 draws + 3 noise
  levels x 5 draws) per model on one dataset first and STOPs if the extrapolated total for all
  350 tasks exceeds 90 minutes (estimated 4.5 min; actual full run took ~3 min). Resumable (skips
  tasks already written to `results/phase3c/runs_parts/`); frees GPU memory between tasks. Writes
  `robustness_runs.csv` (10850 rows = 350 tasks x 31 prediction sets) and one compressed npz of
  probabilities per task under `results/phase3c/predictions/`.
- **`analyze`**: averages the 5 draws per (dataset, model, split, kind, level), computes
  degradation relative to each model's own clean performance on that split, and gives 95%
  bootstrap CIs (seed 42, B=5000) per dataset, dataset-averaged (ALL5/NO_CKD), and as a paired
  difference to logreg in degradation. Writes `robustness_per_dataset.csv`,
  `robustness_averaged.csv`, `robustness_diff_vs_logreg.csv`.
- **`figures`**: Figure 7/7b (NO_CKD AUROC/log-loss degradation curves) and 7c/7d (ALL5
  versions), Figure 7e (per-dataset AUROC, supplementary), Figure 8 (AUROC degradation
  difference to logreg at the highest levels). PNG (300 dpi) and PDF, captions in
  `figure_captions.md`.
- **`tables`**: Table 9 (AUROC vs. missing level), Table 10 (AUROC vs. noise level), Table 11
  (log loss, both kinds), Table 12 (degradation difference to logreg at the highest levels), each
  ALL5 and NO_CKD. Notes in `robustness_notes.md`.

**Robustness caveats** (full statement in `results/phase3c/robustness_notes.md`): the saved
pipeline's preprocessor (median/most-frequent imputation fitted on the training fold) is part of
every result, so these numbers describe MODEL + IMPUTER together, not the model alone; tree
models use no native NaN handling; missingness is MCAR only (no informative-missingness
mechanism is simulated); the bootstrap resamples the 10 splits within each dataset and ignores
between-dataset variability; degradation is always relative to that model's own clean
performance, so a model that starts lower can degrade less in absolute terms. No file states
that a model is "more robust" or "significantly better" -- only numbers are reported.

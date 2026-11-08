r"""Phase 4d: builds paper/evidence/drafting_sheet.md and .csv (facts only, NO paper prose).

One row per planned fact for each section of the paper. Every numeric value is read from
paper/evidence/numbers.json (column `keys`: the numbers-book key; column `values`: its formatted
value; `raw_values`: its unrounded value). A key may instead carry one of three prefixes for
non-numeric / verbatim facts, which are resolved programmatically and checked by the tests:
  pf:<dot.path>    value inside paper/evidence/protocol_facts.json
  meta:<ds>.<key>  value inside src.data.load_dataset(<ds>).meta
  claude:<id>      an exact substring of CLAUDE.md (registry CLAUDE_FACTS below)
Caveats are exact substrings of CLAUDE.md or claim_map.md (registry CAVEATS; asserted at build time).
Claim ids/statuses are read from paper/evidence/claim_map.csv. Deterministic (byte-identical on rerun).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from analysis.constants import DATASETS_ALL5, DATASETS_NO_CKD, MODELS

EVID = PROJECT_ROOT / "paper" / "evidence"
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"
CLAIM_MD = EVID / "claim_map.md"
CLAIM_CSV = EVID / "claim_map.csv"
OUT_MD = EVID / "drafting_sheet.md"
OUT_CSV = EVID / "drafting_sheet.csv"
COLUMNS = ["item_id", "section", "fact_label", "keys", "values", "raw_values", "claim_status",
           "caveat_ids", "caveat_text", "source_file", "shown_in"]
NON_LOGREG = [m for m in MODELS if m != "logreg"]
DS_KEY = lambda ds: ds.replace("_", "-")  # noqa: E731  (numbers-book key style)
SETS = (("all5", "ALL5"), ("nockd", "NO_CKD"))

NUM = {e["key"]: e for e in json.loads((EVID / "numbers.json").read_text(encoding="utf-8"))}
PF = json.loads((EVID / "protocol_facts.json").read_text(encoding="utf-8"))
CLAUDE = CLAUDE_MD.read_text(encoding="utf-8")
CLAIM_TXT = CLAIM_MD.read_text(encoding="utf-8")
CLAIM = pd.read_csv(CLAIM_CSV)

# Exact substrings of CLAUDE.md copied into the sheet as verbatim protocol facts.
CLAUDE_FACTS: dict[str, str] = {
    "preprocessing": "Preprocessing: ONE shared pipeline for all models (median imputation + optional `StandardScaler` for numeric, most-frequent imputation + one-hot with `handle_unknown=\"ignore\"` for categorical), always fitted on the training fold only",
    "subsample": "Training fractions: 0.1, 0.25, 0.5, 1.0 of each training fold, stratified, with at least 2 samples per class",
    "tuning": "Tuning: Optuna TPE, 20 trials, 3 inner folds, mean inner-CV AUROC, same budget for every tunable model, done ONCE at fraction 1.0 on the training fold only, and the tuned hyperparameters are reused for the smaller fractions.",
    "effective_params": "to keep relative regularisation constant at small fractions, `xgb min_child_weight = max(1.0, value * fraction)` (float), `lgbm min_child_samples` and `rf min_samples_leaf = max(1, round(value * fraction))`; `logreg` and all deep models unchanged. At fraction 1.0 params are identical.",
    "metrics": "`auroc`, `f1` and `balanced_accuracy` at threshold 0.5, `brier`, `log_loss` (probabilities clipped to `[1e-7, 1-1e-7]`), `ece` = binary ECE on the positive-class probability with 10 equal-width bins on `[0,1]`, computed manually (not sklearn's calibration module).",
    "deep_schedule": "validation split of 15% (`VALIDATION_FRACTION`) taken from the training subset only (never the test fold), via `split_validation` which requires at least 4 samples per class or returns `None`; early stopping on validation log loss with best-weight restore",
    "no_early_stop_trees": "The tree models have no early stopping (asymmetry, state it in the paper).",
    "ftt": "FT-Transformer is our own implementation and treats ALL encoded columns as numeric tokens (simplification).",
    "tabnet_batch": "TabNet: `_avoid_singleton_remainder` nudges `effective_batch` and `virtual_batch_size` up by 1 when the last batch would have exactly 1 row (BatchNorm crash found in Phase 2b).",
    "no_native_nan": "No model uses native missing-value handling (XGBoost and LightGBM get imputed data too).",
    "ilpd_dedup": "The dropped rows may be different patients with identical values (no patient ID); dropping is the safer choice against train and test leakage.",
    "ckd_ordinal": "`sg`, `al`, `su` are ordered scales and are kept NUMERIC on purpose",
    "ckd_saturated": "This dataset is nearly saturated",
    "hardware": "a Windows **laptop** with an NVIDIA GeForce RTX 4060 **Laptop GPU**",
    "task_seed": "`task_seed = 42 + split_idx`",
    "cleveland_target": "target binarised as `(num > 0)` (any disease = 1)",
}

# Exact-substring caveats: name -> (source text, substring).
CAVEATS: dict[str, tuple[str, str]] = {
    "UNIT": (CLAUDE, "the Friedman tests and critical-difference diagrams are **descriptive only**, because the BLOCK unit is anti-conservative (the 10 splits of a dataset come from overlapping-training-set repeated cross-validation) and the DATASET unit has essentially no power (minimum possible two-sided Wilcoxon p = 0.0625 with 5 datasets, 0.125 with 4)"),
    "SCOPE": (CLAUDE, "Every claim in the paper must be scoped to \"on these 5 small datasets, with 20 tuning trials per model and a fixed training schedule.\""),
    "EXC": (CLAIM_TXT, "the per-cell exceptions below must still be stated alongside this, since the dataset-averaged claim does not rule out individual (dataset, model) cells where the model wins."),
    "CKD": (CLAUDE, "The paper's statements about CKD must say it is saturated, and results must be reported both with and without it"),
    "MECH": (CLAUDE, "the claim map's SUPPORTED/PARTLY/NOT SUPPORTED status is a MECHANICAL rule (CI-excludes-0 in both ALL5 and NO_CKD AND per-dataset sign agreement >=3/5 and >=3/4), fixed before any value was computed"),
    "CAL1": (CLAUDE, "every pooled calibration analysis uses 2 x n_rows predictions per (dataset, model, fraction), since each test row appears in 2 of the 10 splits (one per CV repeat). ECE on small samples is biased and noisy"),
    "CAL2": (CLAUDE, "use Brier and log loss as primary calibration measures, with bootstrap confidence intervals."),
    "ECE_ONLY": (CLAIM_TXT, "ECE on 60-154-row test folds is noisy and biased; report log loss, Brier and the calibration slope alongside it, with CIs."),
    "POSTHOC": (CLAIM_TXT, "The Phase 3b post-hoc check is exploratory and cross-fitted on the test predictions themselves (not a pipeline component); isotonic has a clipping artefact that inflates log loss on misses (CLAUDE.md decision 10)."),
    "ROB1": (CLAUDE, "every result describes MODEL + the saved pipeline's IMPUTER together (median/most-frequent imputation fitted on the training fold is part of the pipeline), not the model in isolation; tree models use no native NaN handling. Missingness tested is MCAR only"),
    "ROB2": (CLAUDE, "Degradation is always relative to each model's OWN clean performance on that split, so a model with lower clean performance can show smaller absolute degradation."),
    "NOISE": (CLAUDE, "Noise is relative to each column's TRAINING standard deviation and applied to ordinal numeric columns (CKD `sg`/`al`/`su`, Cleveland `ca`) as if continuous."),
    "ILPDART": (CLAUDE, "The largest single-split AUROC drop (`ilpd`, split 1, noise=0.5, draw 2 — the same cell for all 7 models) is a perturbation artefact: Gaussian noise can create physically impossible values (e.g. negative lab values) since no clipping was applied in Phase 3c."),
    "EXPL": (CLAUDE, "so only RANKINGS (top-5 sets, Spearman correlation) are ever compared across models, never raw SHAP magnitudes."),
    "DIR": (CLAUDE, "No direction-of-effect (positive/negative SHAP) check was performed."),
    "CORR": (CLAUDE, "Correlated features — especially Breast Wisconsin's mean/standard-error/worst triplets — can substitute for one another in a top-5 set"),
    "TIMING": (CLAIM_TXT, "Caveat: fit_time_s measured with 3 to 4 parallel workers; rough, not a controlled single-process timing."),
    "SCHED": (CLAUDE, "with the caveats that it used 3 splits (0-2, not all 10), fractions 0.1 and 1.0 only (not 0.25/0.5), no re-tuning of hyperparameters under the new schedule, and that the schedule variants were not chosen by looking at test AUROC."),
    "TABNET": (CLAIM_TXT, "Only under THIS training schedule: the Phase 2c sensitivity check (3 splits, fractions 0.1/1.0 only) showed TabNet's ckd collapse resolves under a longer schedule, so the result is schedule-specific, not a verdict on TabNet in general."),
    "ENDPT": (CLAUDE, "For xgb this is a **mild constraint, not a hard floor**"),
    "C6": (CLAIM_TXT, "Note the leader is NOT the same model across all 4 cells (mlp at f1.0/ALL5, logreg at f1.0/NO_CKD, rf at f0.1/ALL5 and f0.1/NO_CKD)."),
    "C7": (CLAIM_TXT, "FLAG: at 30% missing / ALL5, the full-precision smallest-magnitude degradation belongs to ftt (-0.019576), not mlp (-0.019990) as stated in an earlier CLAUDE.md summary sentence"),
    "C10": (CLAIM_TXT, "Datasets flagged non-informative (chance >= 0.60): "),
    "C9": (CLAIM_TXT, "Pairs with cross-model Jaccard BELOW within-model stability: 13 of 15."),
    "ENDPT2": (CLAUDE, "a limitation of possible truncation applies only to `ftt d_token`"),
    "DESC": (CLAUDE, "Evidence framing for the paper (2026-11-07):"),
}

ROWS: list[dict] = []
_COUNTER: dict[str, int] = {}


def _resolve(ref: str) -> tuple[str, str, str]:
    """(formatted, raw, source_file) for one key reference."""
    if ref in NUM:
        e = NUM[ref]
        assert not e["undefined"], ref
        return str(e["formatted"]), repr(e["value"]), e["source_file"]
    if ref.startswith("pf:"):
        v = PF
        for part in ref[3:].split("."):
            v = v[part]
        return str(v), "", "paper/evidence/protocol_facts.json"
    if ref.startswith("meta:"):
        from src.data import load_dataset
        ds, key = ref[5:].split(".", 1)
        v = load_dataset(ds, verbose=False)[2][key]
        return str(v), "", f"src/data.py (load_dataset('{ds}').meta['{key}'])"
    if ref.startswith("claude:"):
        s = CLAUDE_FACTS[ref[7:]]
        assert s in CLAUDE, ref
        return s, "", "CLAUDE.md"
    raise KeyError(f"unknown key reference: {ref}")


def claim_status(claim: str | None, items: list[str] | None = None) -> str:
    if claim is None:
        return "n/a"
    sub = CLAIM[CLAIM.claim_id == claim]
    if items is not None:
        sub = sub[sub["item"].isin(items)]
    assert len(sub), (claim, items)
    return "; ".join(f"{claim}:{r.item}={r.status}" for r in sub.itertuples())


def add(sec: str, label: str, refs: list[str] | None = None, claim: str | None = None, items: list[str] | None = None,
        caveats: list[str] | None = None, shown: str = "text only", source: str | None = None) -> None:
    refs = refs or []
    vals, raws, srcs = [], [], []
    for r in refs:
        f, raw, s = _resolve(r)
        vals.append(f)
        raws.append(raw)
        if s not in srcs:
            srcs.append(s)
    caveats = caveats or []
    texts = []
    for c in caveats:
        text, sub = CAVEATS[c]
        assert sub in text, f"caveat {c} not found verbatim"
        texts.append(sub)
    _COUNTER[sec] = _COUNTER.get(sec, 0) + 1
    ROWS.append({
        "item_id": f"{sec}-{_COUNTER[sec]:03d}", "section": SECTIONS[sec], "fact_label": label,
        "keys": "|".join(refs), "values": "|".join(vals), "raw_values": "|".join(raws),
        "claim_status": claim_status(claim, items), "caveat_ids": "|".join(caveats),
        "caveat_text": " || ".join(texts), "source_file": source or "; ".join(srcs) or "n/a", "shown_in": shown,
    })


SECTIONS = {
    "ABS": "Abstract and I. Introduction", "REL": "II. Related Work (role list, no numbers)",
    "DAT": "III. Datasets and Experimental Protocol", "MOD": "IV. Models and Training",
    "VA": "V.A Discrimination and learning curves", "VB": "V.B Calibration",
    "VC": "V.C Robustness to missing values and noise", "VD": "V.D Explanation agreement",
    "LIM": "VI. Discussion and Limitations", "CON": "VII. Conclusion", "CMP": "Compute (fit times)",
}
FRAC = {"1.0": "f1.0", "0.1": "f0.1"}


def diff_refs(model: str, frac: str, sets=SETS) -> list[str]:
    return [f"auroc-diff-logreg-{model}-{s}-f{frac}-{p}" for s, _ in sets for p in ("point", "ci-lo", "ci-hi", "npos")]


def build() -> None:
    ROWS.clear()
    _COUNTER.clear()
    # ---------------------------------------------------------------- Abstract / Introduction
    for key, label in (("n-datasets", "Number of datasets"), ("n-datasets-no-ckd", "Number of datasets without CKD (NO_CKD)"),
                       ("n-models", "Number of models"), ("n-outer-splits", "Outer splits per dataset"),
                       ("n-fractions", "Number of training fractions"), ("n-tasks", "Number of tasks (dataset x model x split)"),
                       ("n-result-rows", "Number of result rows (tasks x fractions)"),
                       ("tuning-n-trials", "Optuna trials per tunable model")):
        add("ABS", label, [key])
    rows_keys = {ds: f"rows-used-{DS_KEY(ds)}" for ds in DATASETS_ALL5}
    small = min(rows_keys, key=lambda d: NUM[rows_keys[d]]["value"])
    large = max(rows_keys, key=lambda d: NUM[rows_keys[d]]["value"])
    add("ABS", f"Smallest dataset ({small}): rows after de-duplication", [rows_keys[small]], shown="Table I")
    add("ABS", f"Largest dataset ({large}): rows after de-duplication", [rows_keys[large]], shown="Table I")
    for tag, claim, frac in (("C1", "C1", "1.0"), ("C2", "C2", "0.1")):
        for m in NON_LOGREG:
            add("ABS", f"AUROC difference to logreg, {m} minus logreg, fraction {frac}: ALL5 (point, ci-lo, ci-hi, datasets positive), then NO_CKD",
                diff_refs(m, frac), claim, [m], ["UNIT", "CKD", "MECH", "EXC"], "Table II; Fig. 1(a)")
    for m in NON_LOGREG:
        base = [f"auroc-dod-f0.1-f1.0-{m}-{s}-{p}" for s, _ in SETS for p in ("point", "ci-lo", "ci-hi")]
        add("ABS", f"C3 diff-of-differences in AUROC gap to logreg (gap at fraction 0.1 minus gap at fraction 1.0), {m}: ALL5 then NO_CKD (point, ci-lo, ci-hi)",
            base, "C3", [m], ["MECH", "CKD"], "text only")
    for m in MODELS:
        add("ABS", f"C5 pooled NO_CKD calibration slope at fraction 0.1, {m}", [f"slope-{m}-nockd-f0.1"], "C5", None, ["CAL1", "CAL2"], "Fig. 1(c); Table II")
    for fr, s, cell in (("1.0", "all5", "f1.0-ALL5-mlp-vs-xgb"), ("1.0", "nockd", "f1.0-NO_CKD-logreg-vs-mlp"),
                        ("0.1", "all5", "f0.1-ALL5-rf-vs-xgb"), ("0.1", "nockd", "f0.1-NO_CKD-rf-vs-logreg")):
        leader, second = {"f1.0-ALL5-mlp-vs-xgb": ("mlp", "xgb"), "f1.0-NO_CKD-logreg-vs-mlp": ("logreg", "mlp"),
                          "f0.1-ALL5-rf-vs-xgb": ("rf", "xgb"), "f0.1-NO_CKD-rf-vs-logreg": ("rf", "logreg")}[cell]
        refs = [f"logloss-mean-{leader}-{s}-f{fr}", f"logloss-mean-{second}-{s}-f{fr}"]
        if "logreg" in (leader, second):
            other = second if leader == "logreg" else leader
            refs += [f"logloss-diff-logreg-{other}-{s}-f{fr}-{p}" for p in ("point", "ci-lo", "ci-hi")]
            direction = f"diff = {other} minus logreg (as in the numbers book)"
        else:
            refs += [f"logloss-diff-leader-2nd-{leader}-{second}-{s}-f{fr}-{p}" for p in ("point", "ci-lo", "ci-hi")]
            direction = f"diff = {leader} minus {second}"
        add("ABS", f"C6 lowest dataset-averaged log loss, fraction {fr}, {s.upper().replace('NOCKD', 'NO_CKD')}: leader {leader} vs second {second} (means, then {direction})",
            refs, "C6", [cell], ["C6", "CAL2"], "Table II")
    for m in NON_LOGREG:
        refs = [f"auroc-diff-logreg-degradation-miss30-{m}-all5-{p}" for p in ("point", "ci-lo", "ci-hi")]
        add("ABS", f"C8 AUROC degradation difference to logreg at 30% missing, {m} minus logreg, ALL5 (point, ci-lo, ci-hi)", refs, "C8",
            [f"miss30|auroc|{m}"], ["ROB1", "ROB2"], "Fig. 2(a)")
    add("ABS", "C4 Nadeau-Bengio corrected t-tests: number of Holm-significant cells and total cells",
        ["nb-n-sig-cells", "nb-n-total-cells"], "C4", None, ["UNIT"], "text only")
    add("ABS", "C4 the one significant cell (ckd, xgb, AUROC, fraction 0.1): mean difference to logreg, t, p_holm",
        ["nb-sig-0-ckd-xgb-auroc-f0.1-meandiff", "nb-sig-0-ckd-xgb-auroc-f0.1-t", "nb-sig-0-ckd-xgb-auroc-f0.1-pholm"], "C4", None, ["UNIT", "CKD"], "text only")

    # ---------------------------------------------------------------- Related work (role list)
    for key, role in (("christodoulou2019", "clinical LR vs ML; calibration rarely assessed"),
                      ("grinsztajn2022", "tree models on medium-sized data"),
                      ("gorishniy2021", "FT-Transformer; no universally superior model"),
                      ("arik2021", "TabNet"), ("guo2017", "calibration of neural networks; image and document classification only"),
                      ("chen2016", "XGBoost"), ("ke2017", "LightGBM"), ("breiman2001", "random forests"),
                      ("lundberg2017", "SHAP"), ("akiba2019", "Optuna"), ("nadeau2003", "corrected resampled t-test"),
                      ("pedregosa2011", "scikit-learn"), ("dua2019", "UCI repository"),
                      ("shwartzziv2022", "XGBoost vs deep models"), ("kadra2021", "regularized MLPs"),
                      ("mcelfresh2023", "NN vs GBDT debate; neural nets and small datasets"), ("borisov2024", "survey")):
        add("REL", f"{key}: {role}")

    # ---------------------------------------------------------------- III Datasets and protocol
    per_ds = (("rows-used", "rows after de-duplication"), ("n-rows-raw", "raw rows"), ("n-duplicates-dropped", "exact duplicates dropped"),
              ("raw-features", "raw features"), ("n-numeric", "numeric features"), ("n-categorical", "categorical features"),
              ("positive-fraction", "positive class (%)"), ("missing-fraction", "missing values (%)"),
              ("encoded-width-min", "encoded width, minimum over splits"), ("encoded-width-max", "encoded width, maximum over splits"),
              ("min-train-f0.1", "smallest training set at fraction 0.1"), ("n-test-min", "smallest test fold"), ("n-test-max", "largest test fold"))
    for ds in DATASETS_ALL5:
        add("DAT", f"{ds}: " + "; ".join(lbl for _, lbl in per_ds) + " (in this order)",
            [f"{k}-{DS_KEY(ds)}" for k, _ in per_ds], None, None, ["CKD"] if ds == "ckd" else [], "Table I")
    for col in ("plas", "pres", "skin", "insu", "mass"):
        add("DAT", f"pima_diabetes: zeros replaced by missing in column {col}", [f"zeros-replaced-pima-diabetes-{col}"])
    add("DAT", "pima_diabetes: missing share (%) of insulin and skin after replacement",
        ["missing-col-pima-diabetes-insu", "missing-col-pima-diabetes-skin"])
    add("DAT", "heart_cleveland: categorical columns", ["meta:heart_cleveland.categorical_cols"])
    add("DAT", "heart_cleveland: numeric columns", ["meta:heart_cleveland.numeric_cols"])
    add("DAT", "heart_cleveland: target binarisation", ["claude:cleveland_target"])
    add("DAT", "ckd: ordinal columns kept numeric on purpose, and their count", ["meta:ckd.ordinal_as_numeric", "n-ordinal-as-numeric-ckd", "claude:ckd_ordinal"], None, None, ["CKD"])
    add("DAT", "ckd: saturation statement", ["claude:ckd_saturated"], None, None, ["CKD"])
    add("DAT", "ilpd: duplicates dropped (rows raw, dropped, kept) and the caveat", ["n-rows-raw-ilpd", "n-duplicates-dropped-ilpd", "rows-used-ilpd", "claude:ilpd_dedup"])
    add("DAT", "Outer CV: description, number of folds, repeats, total splits, seed", ["pf:protocol.outer_splits_description", "n-cv-folds", "n-cv-repeats", "n-outer-splits", "global-seed"])
    add("DAT", "Task seed rule", ["claude:task_seed"])
    add("DAT", "Training fractions (values) and the subsampling rule", ["pf:protocol.fractions", "claude:subsample"])
    add("DAT", "Shared preprocessing", ["claude:preprocessing", "claude:no_native_nan"])
    add("DAT", "Tuning: trials, inner folds, sampler, objective, rule", ["tuning-n-trials", "tuning-inner-folds", "pf:protocol.tuning_sampler", "pf:protocol.tuning_objective", "claude:tuning"])
    add("DAT", "Metric definitions", ["claude:metrics", "pf:protocol.ece_definition"])
    for k in ("python", "numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "optuna", "torch", "joblib", "matplotlib", "pytorch_tabnet", "shap", "torch_cuda"):
        add("DAT", f"Software version: {k}", [f"pf:software_hardware.{k}"])
    add("DAT", "Hardware: GPU name and nvidia-smi line", ["pf:software_hardware.gpu_name", "pf:software_hardware.nvidia_smi", "claude:hardware"])

    # ---------------------------------------------------------------- IV Models and training
    add("MOD", "The seven models, in the fixed order", ["pf:protocol.models"])
    add("MOD", "Effective-parameter scaling rule at small fractions", ["claude:effective_params"])
    for key, lbl in (("deep-max-steps", "deep max steps"), ("deep-patience-steps", "deep patience steps"),
                     ("deep-validation-fraction", "deep validation fraction (%)"), ("deep-fixed-steps-no-validation", "fixed steps without validation"),
                     ("deep-max-epochs-cap", "deep epoch hard cap")):
        add("MOD", f"Deep-model schedule constant: {lbl}", [key])
    add("MOD", "Deep-model effective batch rule and validation rule", ["pf:protocol.deep_effective_batch_rule", "claude:deep_schedule"])
    add("MOD", "Early-stopping asymmetry between tree and deep models", ["claude:no_early_stop_trees"])
    add("MOD", "FT-Transformer simplification", ["claude:ftt"])
    add("MOD", "TabNet singleton-batch rule", ["claude:tabnet_batch"])
    ends = sorted(k for k in NUM if k.startswith("endpoint-") and (k.endswith("-n-lower") or k.endswith("-n-upper")))
    pairs = sorted({k[: -len("-n-lower")] for k in ends if k.endswith("-n-lower")})
    for p in pairs:
        add("MOD", f"Search-space endpoint hits (of 50 tasks): {p[len('endpoint-'):]} lower then upper", [p + "-n-lower", p + "-n-upper"], None, None, ["ENDPT", "ENDPT2"], "Table S (search spaces)")
    add("MOD", "Number of parameters at the same endpoint in at least 15 of 50 tasks", ["n-high-endpoint-params"], None, None, ["ENDPT"])
    add("MOD", "Number of TabNet collapsed splits (Phase 2c definition)", ["n-tabnet-collapsed-splits"], None, None, ["TABNET", "SCHED"])
    for ds, sp in (("ckd", 1),):
        add("MOD", f"TabNet {ds} split {sp}, fraction 1.0, AUROC: baseline, A_long, B_bigval, C_long_bigval",
            [f"tabnet-{ds}-split{sp}-f1.0-{v}" for v in ("baseline", "a-long", "b-bigval", "c-long-bigval")], None, None, ["TABNET", "SCHED"])
    for m in ("mlp", "ftt", "tabnet"):
        for f in ("0.1", "1.0"):
            add("MOD", f"Sensitivity check: dataset-averaged AUROC change of {m} under the longer schedule A_long, fraction {f}",
                [f"auroc-sens-diff-{m}-a-long-f{f}"], None, None, ["SCHED"])

    # ---------------------------------------------------------------- V.A
    from scripts.make_paper_tables import COLUMNS as TCOLS
    for m in MODELS:
        add("VA", f"Table II row for {m}: AUROC f1.0 ALL5, NO_CKD; f0.1 ALL5, NO_CKD; log loss f1.0 ALL5; log loss f0.1 ALL5; slope f0.1 NO_CKD",
            [t.format(m=m) for _, _, t, _ in TCOLS], None, None, ["CAL1", "CKD"], "Table II")
    for m in MODELS:
        add("VA", f"Learning curve, {m}: ALL5 mean AUROC at fractions 0.1, 0.25, 0.5, 1.0, then ALL5 mean log loss at the same fractions",
            [f"auroc-mean-{m}-all5-f{f}" for f in ("0.1", "0.25", "0.5", "1.0")] + [f"logloss-mean-{m}-all5-f{f}" for f in ("0.1", "0.25", "0.5", "1.0")],
            None, None, ["CKD"], "Fig. 1(a,b)")
        add("VA", f"Change in ALL5 AUROC from fraction 0.1 to fraction 1.0, {m}", [f"auroc-change-f0.1-f1.0-{m}-all5"], None, None, [], "Fig. 1(a)")
    for frac, claim in (("1.0", "C1"), ("0.1", "C2")):
        for ds in DATASETS_ALL5:
            for m in NON_LOGREG:
                k = f"auroc-excess-over-logreg-{DS_KEY(ds)}-{m}-f{frac}"
                if NUM[k]["value"] > 0:
                    add("VA", f"{claim} exception cell, fraction {frac}: {ds}, {m} mean AUROC exceeds logreg's by (model mean, logreg mean, excess)",
                        [f"auroc-mean-{m}-{DS_KEY(ds)}-f{frac}", f"auroc-mean-logreg-{DS_KEY(ds)}-f{frac}", k], claim, None, ["EXC", "CKD"], "text only")
    for m in NON_LOGREG:
        add("VA", f"C3 diff-of-differences, {m}: ALL5 (point, ci-lo, ci-hi, datasets agreeing in sign, datasets in total), then NO_CKD",
            [f"auroc-dod-f0.1-f1.0-{m}-{s}-{p}" for s, _ in SETS for p in ("point", "ci-lo", "ci-hi", "nsign-agree", "ntotal")], "C3", [m], ["MECH"], "text only")
    add("VA", "C4 corrected t-test: significant cells, total cells", ["nb-n-sig-cells", "nb-n-total-cells"], "C4", None, ["UNIT"], "text only")
    add("VA", "Holm-significant pairs: BLOCK unit and DATASET unit (descriptive)", ["n-holm-sig-block-pairs", "n-holm-sig-dataset-pairs"], None, None, ["UNIT"], "text only")
    for s, _ in SETS:
        for f in ("1.0", "0.1"):
            add("VA", f"Friedman test AUROC (descriptive only), {s}, fraction {f}: chi-square, p", [f"friedman-chi2-{s}-f{f}", f"friedman-p-{s}-f{f}"], None, None, ["UNIT"], "text only")

    # ---------------------------------------------------------------- V.B
    for f in ("1.0", "0.1"):
        for m in MODELS:
            add("VB", f"NO_CKD pooled calibration slope, {m}, fraction {f}", [f"slope-{m}-nockd-f{f}"], "C5", None, ["CAL1", "CAL2"], "Fig. 1(c); Table II" if f == "0.1" else "text only")
        for ds in DATASETS_ALL5:
            for m in MODELS:
                add("VB", f"Per-dataset pooled slope (point, ci-lo, ci-hi), {ds}, {m}, fraction {f}",
                    [f"slope-per-dataset-{DS_KEY(ds)}-{m}-f{f}-{p}" for p in ("point", "ci-lo", "ci-hi")], "C5", None, ["CAL1", "CKD"], "Table S (slopes)")
    for s in ("all5",):  # pooled ECE/Brier/log-loss keys exist for ALL5 only (Phase 3b table 6)
        for f in ("1.0", "0.1"):
            for m in MODELS:
                add("VB", f"Pooled calibration, {m}, {s}, fraction {f}: ECE equal-width, ECE equal-mass, Brier, log loss",
                    [f"ece-ew-{m}-{s}-f{f}", f"ece-em-{m}-{s}-f{f}", f"brier-pooled-{m}-{s}-f{f}", f"logloss-pooled-{m}-{s}-f{f}"], None, None, ["CAL1", "CAL2", "ECE_ONLY"], "text only")
    for f in ("1.0", "0.1"):
        for m in NON_LOGREG:
            add("VB", f"Pooled ECE equal-width difference to logreg, {m} minus logreg, ALL5, fraction {f} (point, ci-lo, ci-hi)",
                [f"ece-ew-diff-logreg-{m}-all5-f{f}-{p}" for p in ("point", "ci-lo", "ci-hi")], None, None, ["CAL1", "ECE_ONLY"], "text only")
    for s, _ in SETS:
        for m in MODELS:
            add("VB", f"EXPLORATORY post-hoc recalibration, {m}, {s}: Platt Brier diff, Platt log-loss diff, Platt fallbacks, isotonic Brier diff, isotonic log-loss diff, isotonic fallbacks",
                [f"posthoc-{m}-{s}-{c}-{q}" for c in ("platt", "isotonic") for q in (("brier-diff", "logloss-diff", "n-fallback"))], None, None, ["POSTHOC"], "supplementary only")

    # ---------------------------------------------------------------- V.C
    for s, _ in SETS:
        for m in MODELS:
            add("VC", f"C7 dataset-averaged AUROC and log-loss change under perturbation, {m}, {s}: 30% missing (AUROC, log loss), noise 0.5 original (AUROC, log loss), noise 0.5 continuous-only (AUROC, log loss)",
                [f"auroc-miss30-degradation-{m}-{s}", f"logloss-miss30-degradation-{m}-{s}", f"auroc-noise50-orig-degradation-{m}-{s}",
                 f"logloss-noise50-orig-degradation-{m}-{s}", f"auroc-noise50-cont-degradation-{m}-{s}", f"logloss-noise50-cont-degradation-{m}-{s}"],
                "C7", [f"{lab}-{'ALL5' if s == 'all5' else 'NO_CKD'}" for lab in ("30% missing", "noise 0.5 (original)", "noise 0.5 (continuous-only)")], ["ROB1", "ROB2", "NOISE", "C7"], "Fig. 2")
    for pert, ptag in (("miss30", "30% missing"), ("noise50-orig", "noise 0.5 (original)"), ("noise50-cont", "noise 0.5 (continuous-only)")):
        for metric, mname in (("auroc", "auroc"), ("logloss", "log_loss")):
            for m in NON_LOGREG:
                refs = [f"{metric}-diff-logreg-degradation-{pert}-{m}-{s}-{p}" for s, _ in SETS for p in ("point", "ci-lo", "ci-hi")]
                add("VC", f"C8 degradation difference to logreg, {m} minus logreg, {mname}, {ptag}: ALL5 (point, ci-lo, ci-hi) then NO_CKD",
                    refs, "C8", [f"{pert}|{mname}|{m}"], ["ROB1", "ROB2", "NOISE", "MECH"], "Fig. 2" if pert != "noise50-cont" or mname == "log_loss" else "supplementary")
    for ds in DATASETS_ALL5:
        add("VC", f"Share of non-negative columns that went negative at noise 0.5 (%), {ds}: original noise, continuous-only noise",
            [f"negrate-{DS_KEY(ds)}-level0.5-orig", f"negrate-{DS_KEY(ds)}-level0.5-cont"], None, None, ["NOISE"], "text only")
    add("VC", "Largest single-split AUROC drop (ilpd, split 1, noise 0.5, draw 2): perturbation artefact", ["claude:ilpd_dedup"], None, None, ["ILPDART"], "text only")

    # ---------------------------------------------------------------- V.D
    for a, b in (("logreg", "rf"), ("logreg", "mlp"), ("rf", "mlp")):
        add("VD", f"C9 cross-model top-5 Jaccard and Spearman, {a} vs {b}: ALL5, NO_CKD (Jaccard), ALL5, NO_CKD (Spearman)",
            [f"jaccard-{a}-{b}-all5", f"jaccard-{a}-{b}-nockd", f"spearman-{a}-{b}-all5", f"spearman-{a}-{b}-nockd"], "C9", None, ["EXPL", "DIR", "CORR"], "Fig. 3(a)")
        for ds in DATASETS_ALL5:
            add("VD", f"C9 per-dataset top-5 Jaccard and Spearman, {a} vs {b}, {ds}", [f"jaccard-{a}-{b}-{DS_KEY(ds)}", f"spearman-{a}-{b}-{DS_KEY(ds)}"], "C9", None, ["C9", "EXPL", "CORR"], "Fig. 3(a)")
    for ds in DATASETS_ALL5:
        add("VD", f"C9 baselines, {ds}: within-model stability Jaccard, stability Spearman, chance Jaccard, number of features",
            [f"stability-jaccard-{DS_KEY(ds)}", f"stability-spearman-{DS_KEY(ds)}", f"chance-jaccard-{DS_KEY(ds)}", f"m-features-{DS_KEY(ds)}"], "C9", None, ["C9"], "Fig. 3(a)")
    for ds in DATASETS_ALL5:
        for m in ("logreg", "rf", "mlp"):
            add("VD", f"C10 plausibility precision vs the pre-registered clinical list, {ds}, {m}: precision, ci-lo, ci-hi, chance",
                [f"plausibility-{m}-{DS_KEY(ds)}-{p}" for p in ("precision", "ci-lo", "ci-hi", "chance")], "C10", None, ["C10", "DIR"], "Fig. 3(b)")
    flagged = [ds for ds in DATASETS_ALL5 if NUM[f"plausibility-logreg-{DS_KEY(ds)}-chance"]["value"] >= 0.60 - 1e-12]
    add("VD", f"C10 datasets with chance precision >= 0.60 (non-informative for plausibility; dagger in Fig. 3(b)): {', '.join(flagged)}",
        [f"plausibility-logreg-{DS_KEY(ds)}-chance" for ds in flagged], "C10", None, ["C10"], "Fig. 3(b)")
    for m in ("logreg", "rf", "mlp"):
        add("VD", f"Number of distinct top-5 sets over the 10 splits, breast_wisconsin, {m}", [f"n-distinct-top5-breast-wisconsin-{m}"], "C9", None, ["CORR"], "text only")

    # ---------------------------------------------------------------- VI Limitations (verbatim)
    sec7 = CLAUDE.split("## 7.", 1)[1].split("## 8.", 1)[0]
    sec7 = sec7.split("\n", 1)[1]
    items = re.split(r"\n(?=- )", sec7.strip("\n"))
    for i, it in enumerate(items, 1):
        _COUNTER["LIM"] = _COUNTER.get("LIM", 0) + 1
        ROWS.append({"item_id": f"LIM-{i:03d}", "section": SECTIONS["LIM"], "fact_label": f"Limitations item L{i} (CLAUDE.md section 7, verbatim)",
                     "keys": "", "values": it, "raw_values": "", "claim_status": "n/a", "caveat_ids": "", "caveat_text": "",
                     "source_file": "CLAUDE.md", "shown_in": "text only"})
    mn = CLAIM_TXT.split("## STATEMENTS THAT MUST NOT BE MADE", 1)[1].strip("\n")
    for j, it in enumerate(re.split(r"\n(?=- \*\*)", mn.strip("\n")), 1):
        ROWS.append({"item_id": f"LIM-M{j:02d}", "section": SECTIONS["LIM"], "fact_label": f"Statement that must not be made M{j} (claim_map.md, verbatim)",
                     "keys": "", "values": it.strip("\n"), "raw_values": "", "claim_status": "n/a", "caveat_ids": "", "caveat_text": "",
                     "source_file": "paper/evidence/claim_map.md", "shown_in": "text only"})

    # ---------------------------------------------------------------- VII Conclusion
    for claim, lbl in (("C1", "C1 status per model (fraction 1.0, AUROC below logreg)"), ("C2", "C2 status per model (fraction 0.1)"),
                       ("C3", "C3 status per model (gap larger at fraction 0.1)"), ("C5", "C5 calibration slopes"),
                       ("C6", "C6 log-loss leaders"), ("C9", "C9 explanation agreement"), ("C10", "C10 plausibility")):
        add("CON", lbl, [], claim, None, ["UNIT", "SCOPE"] if claim in ("C1", "C2", "C3") else ["SCOPE"], "text only", "paper/evidence/claim_map.csv")
    for m in NON_LOGREG:
        add("CON", f"C8 AUROC degradation difference to logreg at 30% missing, {m}, status", [], "C8", [f"miss30|auroc|{m}"], ["ROB1", "SCOPE"], "text only", "paper/evidence/claim_map.csv")
    add("CON", "Evidence framing for the paper", [], None, None, ["DESC", "SCOPE"], "text only", "CLAUDE.md")

    # ---------------------------------------------------------------- Compute
    for m in MODELS:
        add("CMP", f"Fit time (s) and predict time (s), {m}, fraction 1.0, dataset-averaged ALL5", [f"fit-time-{m}", f"predict-time-{m}"], None, None, ["TIMING"], "text only")
    for m in NON_LOGREG:
        add("CMP", f"C11 fit-time ratio to logreg, {m}", [f"fit-time-ratio-logreg-{m}"], "C11", [m], ["TIMING"], "text only")


LEGEND = """# Drafting sheet (Phase 4d): facts only, no paper prose

How to read it. Every row is one fact for one section. `keys` lists numbers-book keys (use them as `\\N{key}` in the paper); `values` are their formatted values
(read from `paper/evidence/numbers.json`, never typed); `raw_values` are the unrounded values. Keys starting with `pf:`, `meta:` or `claude:` are verbatim non-numeric
facts from `protocol_facts.json`, the dataset metadata, or CLAUDE.md. `caveat_text` is an exact copy from CLAUDE.md or `claim_map.md`.

Status words (mechanical rule of the claim map, fixed before any value was computed):
- SUPPORTED: the confidence interval excludes 0 in both ALL5 and NO_CKD and the per-dataset signs agree (>=3/5 and >=3/4).
- PARTLY: only part of that rule holds.
- NOT SUPPORTED: the rule fails.
- DESCRIPTIVE: a quantity that is reported, not tested.

Rules: a claim with status NOT SUPPORTED or PARTLY may only be stated with its caveat attached. DESCRIPTIVE items may be stated as numbers only.
Digits in a cell must come from a key's formatted value or a verbatim copy; the `fact_label` column carries only fixed design labels (see tests/test_phase4d.py).
"""


def write() -> None:
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        w.writerows(ROWS)
    lines = [LEGEND]
    cav_used: dict[str, str] = {}
    for sec, title in SECTIONS.items():
        rows = [r for r in ROWS if r["section"] == title]
        lines += [f"\n## {title} ({len(rows)} rows)\n"]
        if sec == "LIM":
            for r in rows:
                lines += [f"**{r['item_id']}** {r['fact_label']}\n", "```text", r["values"], "```\n"]
            continue
        lines += ["| id | fact | keys | values | claim:status | caveats | source | shown in |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            for c in r["caveat_ids"].split("|"):
                if c:
                    cav_used[c] = CAVEATS[c][1]
            esc = lambda s: str(s).replace("|", "/").replace("\n", " ")  # noqa: E731
            lines.append(f"| {r['item_id']} | {esc(r['fact_label'])} | {esc(r['keys']).replace('/', ', ')} | {esc(r['values']).replace('/', ', ')} | "
                         f"{esc(r['claim_status'])} | {esc(r['caveat_ids']).replace('/', ', ')} | {esc(r['source_file'])} | {esc(r['shown_in'])} |")
    lines += ["\n## Caveat texts (verbatim; ids used in the tables above)\n"]
    for c in sorted(cav_used):
        lines.append(f"- **{c}**: {cav_used[c]}")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    print(f"sys.executable = {sys.executable}")
    build()
    write()
    per = {}
    for r in ROWS:
        per[r["section"]] = per.get(r["section"], 0) + 1
    for k, v in per.items():
        print(f"{v:4d}  {k}")
    print(f"total {len(ROWS)} rows")


if __name__ == "__main__":
    main()

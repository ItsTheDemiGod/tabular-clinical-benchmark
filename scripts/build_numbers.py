"""Phase 4a: builds the evidence pack entirely from existing result files.

Produces (all under paper/evidence/):
  - protocol_facts.json / protocol_facts.md   (Part 2: dataset + protocol facts)
  - numbers.json / numbers.csv / numbers.tex / numbers.md / numbers_index.md  (Part 3)
  - inventory.csv / inventory.md              (Part 4: every figure/table produced)

Never hand-types a result number: every numbers.json entry is read programmatically
from a source file and carries (source_file, selector, source_sha256). Read-only
except for writing into paper/evidence/. Never fits, refits or tunes anything.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from analysis.constants import DATASETS_ALL5, DATASETS_NO_CKD, MODELS, SEED
from scripts.phase2c_diagnostics import get_search_space
from src.data import load_dataset
import src.deep_models as deep_models_module
import src.metrics as metrics_module
import src.tuning as tuning_module

OUTPUT_DIR = PROJECT_ROOT / "paper" / "evidence"

# ---------------------------------------------------------------------------
# Key naming, hashing, rounding
# ---------------------------------------------------------------------------
_KEY_INVALID = re.compile(r"[^a-z0-9.\-]")


def ds_label(name: str) -> str:
    """Dataset-set label for keys: ALL5 -> all5, NO_CKD -> nockd, else lower()."""
    return {"ALL5": "all5", "NO_CKD": "nockd"}.get(name, str(name).lower())


def make_key(*parts) -> str:
    raw = "-".join(str(p) for p in parts).lower().replace("_", "-").replace(" ", "-")
    raw = _KEY_INVALID.sub("-", raw)
    return re.sub(r"-+", "-", raw).strip("-")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_DECIMALS = {
    "auroc": 3, "brier": 3, "logloss": 3, "ece": 3, "slope": 3, "intercept": 3,
    "diff": 3, "ci": 3, "rank": 2, "jaccard": 2, "spearman": 2, "precision": 2,
    "percentage": 1, "time": 2,
}


def format_value(value, kind: str) -> str:
    if kind == "count":
        return str(int(round(value)))
    if kind == "pvalue":
        return "<0.001" if value < 0.001 else f"{value:.3f}"
    if kind == "string":
        return str(value)
    decimals = _DECIMALS[kind]
    s = f"{value:.{decimals}f}"
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]  # negative zero -> "0.000"
    return s


class NumbersBook:
    """Accumulates numbers.json entries with provenance; enforces unique keys."""

    def __init__(self) -> None:
        self.entries: list[dict] = []
        self._keys: set[str] = set()
        self._sha_cache: dict[str, str] = {}

    def _sha(self, source_file: Path) -> str:
        key = str(source_file)
        if key not in self._sha_cache:
            if not source_file.exists():
                raise FileNotFoundError(f"numbers.json source file missing: {source_file}")
            self._sha_cache[key] = file_sha256(source_file)
        return self._sha_cache[key]

    def add(
        self,
        key: str,
        value,
        kind: str,
        unit: str,
        role: str,
        source_file: Path,
        selector: str,
        phase: str,
        undefined_reason: str | None = None,
    ) -> None:
        if not re.fullmatch(r"[a-z0-9.\-]+", key):
            raise ValueError(f"invalid key (must match [a-z0-9.-]+): {key!r}")
        if key in self._keys:
            raise ValueError(f"duplicate numbers.json key: {key}")
        if role not in ("primary", "descriptive", "supplementary"):
            raise ValueError(f"invalid role {role!r} for key {key}")
        source_file = Path(source_file)
        sha = self._sha(source_file)

        if undefined_reason is not None:
            entry_value, formatted = None, None
        else:
            if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                raise ValueError(f"key {key}: value is NaN/inf/None but undefined_reason was not given")
            entry_value = float(value) if isinstance(value, (int, float, np.integer, np.floating)) else value
            formatted = format_value(value, kind)

        self._keys.add(key)
        self.entries.append(
            {
                "key": key,
                "value": entry_value,
                "formatted": formatted,
                "unit": unit,
                "role": role,
                "source_file": str(source_file.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "selector": selector,
                "source_sha256": sha,
                "phase": phase,
                "undefined": undefined_reason is not None,
                "undefined_reason": undefined_reason,
            }
        )


def _one_row(df: pd.DataFrame, mask: "pd.Series[bool]", context: str) -> pd.Series:
    sub = df[mask]
    if len(sub) != 1:
        raise RuntimeError(f"expected exactly 1 row for {context}, got {len(sub)}")
    return sub.iloc[0]


# ---------------------------------------------------------------------------
# Part 2: dataset and protocol facts (protocol_facts.json / .md)
# ---------------------------------------------------------------------------
def build_protocol_facts(book: NumbersBook) -> dict:
    all_runs_path = PROJECT_ROOT / "results" / "all_runs.csv"
    all_runs = pd.read_csv(all_runs_path)
    dataset_summary_path = PROJECT_ROOT / "results" / "dataset_summary.csv"
    dataset_summary = pd.read_csv(dataset_summary_path)

    datasets = {}
    for ds in DATASETS_ALL5:
        row = _one_row(dataset_summary, dataset_summary.dataset == ds, f"dataset_summary[{ds}]")
        f1 = all_runs[(all_runs.dataset == ds) & (all_runs.fraction == 1.0)]
        all_frac = all_runs[all_runs.dataset == ds]
        f01 = all_runs[(all_runs.dataset == ds) & (all_runs.fraction == 0.1)]
        facts = {
            "name": ds,
            "source": row["source"],
            "raw_rows": int(row["n_rows_before_dedup"]),
            "duplicates_dropped": int(row["n_duplicates_found"]),
            "rows_used": int(row["n_rows"]),
            "raw_features": int(row["n_features"]),
            "numeric_features": int(row["n_numeric"]),
            "categorical_features": int(row["n_categorical"]),
            "encoded_width_min": int(f1.n_features_encoded.min()),
            "encoded_width_max": int(f1.n_features_encoded.max()),
            "positive_fraction": float(row["positive_fraction"]),
            "missing_fraction": float(row["missing_fraction"]),
            "test_fold_n_min": int(all_frac.n_test.min()),
            "test_fold_n_max": int(all_frac.n_test.max()),
            "min_train_n_at_frac0.1": int(f01.n_train.min()),
        }
        datasets[ds] = facts

        book.add(make_key("rows-used", ds), facts["rows_used"], "count", "rows", "descriptive",
                  dataset_summary_path, f"dataset_summary.csv row dataset=={ds}, column n_rows", "3a")
        book.add(make_key("raw-features", ds), facts["raw_features"], "count", "features", "descriptive",
                  dataset_summary_path, f"dataset_summary.csv row dataset=={ds}, column n_features", "3a")
        book.add(make_key("positive-fraction", ds), facts["positive_fraction"] * 100, "percentage", "%", "descriptive",
                  dataset_summary_path, f"dataset_summary.csv row dataset=={ds}, column positive_fraction*100", "3a")
        book.add(make_key("missing-fraction", ds), facts["missing_fraction"] * 100, "percentage", "%", "descriptive",
                  dataset_summary_path, f"dataset_summary.csv row dataset=={ds}, column missing_fraction*100", "3a")
        book.add(make_key("encoded-width-min", ds), facts["encoded_width_min"], "count", "columns", "descriptive",
                  all_runs_path, f"all_runs.csv dataset=={ds} & fraction==1.0, min(n_features_encoded)", "3a")
        book.add(make_key("encoded-width-max", ds), facts["encoded_width_max"], "count", "columns", "descriptive",
                  all_runs_path, f"all_runs.csv dataset=={ds} & fraction==1.0, max(n_features_encoded)", "3a")

    # Protocol constants, read programmatically (never hand-typed).
    tune_sig = inspect.signature(tuning_module.tune_model)
    n_trials = tune_sig.parameters["n_trials"].default
    inner_folds = tune_sig.parameters["inner_splits"].default
    sampler_source = inspect.getsource(tuning_module.tune_model)
    sampler_match = re.search(r"sampler=optuna\.samplers\.(\w+)\(", sampler_source)
    sampler_name = sampler_match.group(1) if sampler_match else "UNKNOWN"

    schedule_source = inspect.getsource(deep_models_module.training_schedule)
    max_epochs_cap_match = re.search(r"max_epochs = min\((\d+),", schedule_source)
    max_epochs_cap = int(max_epochs_cap_match.group(1))
    batch_rule_match = re.search(r"effective_batch = (.+)\n", schedule_source)
    batch_rule_text = batch_rule_match.group(1).strip()

    ece_docstring = inspect.getdoc(metrics_module.expected_calibration_error) or ""

    protocol = {
        "n_models": len(MODELS),
        "models": list(MODELS),
        "n_datasets": len(DATASETS_ALL5),
        "n_datasets_no_ckd": len(DATASETS_NO_CKD),
        "outer_splits_total": 10,
        "outer_splits_description": "5-fold x 2 repeats (RepeatedStratifiedKFold)",
        "fractions": [0.1, 0.25, 0.5, 1.0],
        "n_tasks": len(MODELS) * len(DATASETS_ALL5) * 10,
        "n_result_rows": len(all_runs),
        "tuning_n_trials": n_trials,
        "tuning_inner_folds": inner_folds,
        "tuning_sampler": sampler_name,
        "tuning_objective": "mean inner-CV AUROC",
        "global_seed": SEED,
        "deep_max_steps": deep_models_module.MAX_STEPS,
        "deep_patience_steps": deep_models_module.PATIENCE_STEPS,
        "deep_validation_fraction": deep_models_module.VALIDATION_FRACTION,
        "deep_fixed_steps_no_validation": deep_models_module.FIXED_STEPS_NO_VALIDATION,
        "deep_max_epochs_cap": max_epochs_cap,
        "deep_effective_batch_rule": batch_rule_text,
        "ece_definition": ece_docstring.split("\n\n")[0] if ece_docstring else "",
    }

    for key_name, value, kind, unit in [
        ("n-models", protocol["n_models"], "count", "models"),
        ("n-datasets", protocol["n_datasets"], "count", "datasets"),
        ("n-outer-splits", protocol["outer_splits_total"], "count", "splits"),
        ("n-tasks", protocol["n_tasks"], "count", "tasks"),
        ("n-result-rows", protocol["n_result_rows"], "count", "rows"),
        ("tuning-n-trials", protocol["tuning_n_trials"], "count", "trials"),
        ("tuning-inner-folds", protocol["tuning_inner_folds"], "count", "folds"),
        ("global-seed", protocol["global_seed"], "count", "seed"),
        ("deep-max-steps", protocol["deep_max_steps"], "count", "steps"),
        ("deep-patience-steps", protocol["deep_patience_steps"], "count", "steps"),
        ("deep-max-epochs-cap", protocol["deep_max_epochs_cap"], "count", "epochs"),
        ("deep-fixed-steps-no-validation", protocol["deep_fixed_steps_no_validation"], "count", "steps"),
    ]:
        book.add(make_key(key_name), value, kind, unit, "descriptive", all_runs_path,
                  f"programmatic constant ({key_name})", "3a")
    book.add(make_key("deep-validation-fraction"), protocol["deep_validation_fraction"] * 100, "percentage", "%",
              "descriptive", all_runs_path, "src.deep_models.VALIDATION_FRACTION * 100", "3a")

    # Search space (reference table, not individual numbers.json entries).
    search_space = get_search_space()

    # Software / hardware versions (printed only, no BibTeX).
    import numpy, scipy, sklearn, scipy as _scipy
    versions = {"python": platform.python_version()}
    for mod_name in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "optuna", "torch", "joblib", "matplotlib"]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[mod_name] = "not installed"
    try:
        import importlib.metadata as _metadata

        versions["pytorch_tabnet"] = _metadata.version("pytorch-tabnet")
    except Exception:  # noqa: BLE001
        versions["pytorch_tabnet"] = "not installed"
    try:
        import shap as shap_module
        versions["shap"] = getattr(shap_module, "__version__", "unknown")
    except ImportError:
        versions["shap"] = "not installed"
    try:
        import torch
        versions["torch_cuda"] = torch.version.cuda if torch.cuda.is_available() else "unavailable"
        versions["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    except Exception:  # noqa: BLE001
        versions["torch_cuda"] = "unavailable"
        versions["gpu_name"] = "unavailable"
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        versions["nvidia_smi"] = nvidia_smi.stdout.strip() if nvidia_smi.returncode == 0 else "nvidia-smi failed"
    except Exception:  # noqa: BLE001
        versions["nvidia_smi"] = "nvidia-smi unavailable"
    versions["environment_split"] = ".venv = classical only (no torch); .venv-deep = classical + torch + pytorch-tabnet + shap"

    # Fit / predict times at fraction 1.0, averaged within dataset then over datasets.
    times = {}
    f1 = all_runs[all_runs.fraction == 1.0]
    for model in MODELS:
        per_dataset_fit, per_dataset_predict = [], []
        for ds in DATASETS_ALL5:
            sub = f1[(f1.dataset == ds) & (f1.model == model)]
            per_dataset_fit.append(sub.fit_time_s.mean())
            per_dataset_predict.append(sub.predict_time_s.mean())
        mean_fit = float(np.mean(per_dataset_fit))
        mean_predict = float(np.mean(per_dataset_predict))
        times[model] = {"mean_fit_time_s": mean_fit, "mean_predict_time_s": mean_predict}
        book.add(make_key("fit-time", model), mean_fit, "time", "s", "descriptive", all_runs_path,
                  f"all_runs.csv fraction==1.0 model=={model}, mean(mean-per-dataset fit_time_s)", "3a")
        book.add(make_key("predict-time", model), mean_predict, "time", "s", "descriptive", all_runs_path,
                  f"all_runs.csv fraction==1.0 model=={model}, mean(mean-per-dataset predict_time_s)", "3a")
    times["note"] = "fit/predict times measured with 3 to 4 parallel workers; rough"

    return {
        "datasets": datasets,
        "protocol": protocol,
        "search_space": search_space,
        "software_hardware": versions,
        "timing": times,
    }


def write_protocol_facts(facts: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "protocol_facts.json").write_text(json.dumps(facts, indent=2, default=str) + "\n", encoding="utf-8")

    lines = ["# Protocol facts (generated programmatically, never hand-typed)\n", "## Datasets\n"]
    lines.append("| dataset | source | raw_rows | dup_dropped | rows_used | raw_feat | numeric | categorical | encoded_width | positive% | missing% | n_test (min-max) | min_train@0.1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ds, d in facts["datasets"].items():
        lines.append(
            f"| {ds} | {d['source']} | {d['raw_rows']} | {d['duplicates_dropped']} | {d['rows_used']} | "
            f"{d['raw_features']} | {d['numeric_features']} | {d['categorical_features']} | "
            f"{d['encoded_width_min']}-{d['encoded_width_max']} | {d['positive_fraction']*100:.1f} | "
            f"{d['missing_fraction']*100:.2f} | {d['test_fold_n_min']}-{d['test_fold_n_max']} | {d['min_train_n_at_frac0.1']} |"
        )
    lines.append("\n## Protocol constants\n")
    for k, v in facts["protocol"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Search space (per model)\n")
    for model, records in facts["search_space"].items():
        lines.append(f"### {model}")
        lines.append("| name | type | low | high | log | choices |")
        lines.append("|---|---|---|---|---|---|")
        for r in records:
            lines.append(f"| {r['name']} | {r['type']} | {r['low']} | {r['high']} | {r['log']} | {r['choices']} |")
    lines.append("\n## Software / hardware\n")
    for k, v in facts["software_hardware"].items():
        lines.append(f"- **{k}**: {v}")
    lines.append("\n## Fit / predict time (fraction 1.0, dataset-averaged)\n")
    lines.append("| model | mean_fit_time_s | mean_predict_time_s |")
    lines.append("|---|---|---|")
    for model in MODELS:
        t = facts["timing"][model]
        lines.append(f"| {model} | {t['mean_fit_time_s']:.3f} | {t['mean_predict_time_s']:.4f} |")
    lines.append(f"\nNote: {facts['timing']['note']}.\n")

    (OUTPUT_DIR / "protocol_facts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Part 3, Group 1: dataset/protocol facts already added by build_protocol_facts.
# Part 3, Group 2: primary performance (Phase 3a)
# ---------------------------------------------------------------------------
FRACTIONS_ALL = [0.1, 0.25, 0.5, 1.0]
FRACTIONS_TEST = [1.0, 0.1]
METRICS3 = ["auroc", "log_loss", "brier"]
METRIC_KIND = {"auroc": "auroc", "log_loss": "logloss", "brier": "brier"}


def build_group2_primary_performance(book: NumbersBook) -> None:
    all_runs_path = PROJECT_ROOT / "results" / "all_runs.csv"
    all_runs = pd.read_csv(all_runs_path)

    # Per-dataset mean/std AUROC at fractions 1.0 and 0.1, every model.
    for model in MODELS:
        for ds in DATASETS_ALL5:
            for frac in FRACTIONS_TEST:
                vals = all_runs[(all_runs.dataset == ds) & (all_runs.model == model) & (all_runs.fraction == frac)]["auroc"]
                if len(vals) != 10:
                    raise RuntimeError(f"expected 10 split values for {ds}/{model}/frac={frac}, got {len(vals)}")
                frac_tag = "f1.0" if frac == 1.0 else "f0.1"
                book.add(make_key("auroc-mean", model, ds, frac_tag), float(vals.mean()), "auroc", "", "primary",
                          all_runs_path, f"all_runs.csv dataset=={ds} model=={model} fraction=={frac}, mean(auroc)", "3a")
                book.add(make_key("auroc-std", model, ds, frac_tag), float(vals.std(ddof=1)), "auroc", "", "descriptive",
                          all_runs_path, f"all_runs.csv dataset=={ds} model=={model} fraction=={frac}, std(auroc,ddof=1)", "3a")

    # Dataset-averaged ALL5/NOCKD AUROC/log_loss/Brier at all 4 fractions.
    for model in MODELS:
        for metric in METRICS3:
            for frac in FRACTIONS_ALL:
                for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
                    per_dataset_means = []
                    for ds in datasets:
                        vals = all_runs[(all_runs.dataset == ds) & (all_runs.model == model) & (all_runs.fraction == frac)][metric]
                        per_dataset_means.append(vals.mean())
                    avg = float(np.mean(per_dataset_means))
                    frac_tag = f"f{frac}"
                    book.add(
                        make_key(metric.replace("_", ""), "mean", model, ds_label(set_name), frac_tag),
                        avg, METRIC_KIND[metric], "", "primary", all_runs_path,
                        f"all_runs.csv model=={model} fraction=={frac} datasets in {set_name}, mean over datasets of per-dataset mean({metric})", "3a",
                    )

    # Dataset-averaged AUROC/log_loss/Brier difference to logreg, with CI + n_positive.
    ci_diff_path = PROJECT_ROOT / "results" / "phase3a" / "ci_diff_vs_logreg.csv"
    ci_diff = pd.read_csv(ci_diff_path)
    for model in MODELS:
        if model == "logreg":
            continue
        for metric in METRICS3:
            for frac in FRACTIONS_TEST:
                for set_name in ("ALL5", "NO_CKD"):
                    row = _one_row(
                        ci_diff,
                        (ci_diff.model == model) & (ci_diff.metric == metric) & (ci_diff.fraction == frac) & (ci_diff.dataset_set == set_name),
                        f"ci_diff_vs_logreg[{model},{metric},{frac},{set_name}]",
                    )
                    frac_tag = "f1.0" if frac == 1.0 else "f0.1"
                    base = make_key(metric.replace("_", ""), "diff-logreg", model, ds_label(set_name), frac_tag)
                    book.add(f"{base}-point", float(row.point), "diff", "", "primary", ci_diff_path,
                              f"ci_diff_vs_logreg.csv model=={model} metric=={metric} fraction=={frac} dataset_set=={set_name}, point", "3a")
                    book.add(f"{base}-ci-lo", float(row.ci_lo), "ci", "", "primary", ci_diff_path,
                              f"... ci_lo", "3a")
                    book.add(f"{base}-ci-hi", float(row.ci_hi), "ci", "", "primary", ci_diff_path,
                              f"... ci_hi", "3a")
                    book.add(f"{base}-npos", int(row.n_positive), "count", "datasets", "descriptive", ci_diff_path,
                              f"... n_positive", "3a")

    # Mean ranks (BLOCK unit) at all 4 fractions, ALL5/NOCKD: Phase 3a for {1.0,0.1}, Phase 3b Table 4b for {0.25,0.5}.
    friedman_path = PROJECT_ROOT / "results" / "phase3a" / "friedman_summary.csv"
    friedman = pd.read_csv(friedman_path)
    table4b_path = PROJECT_ROOT / "results" / "phase3b" / "table4b_mean_rank_raw.csv"
    table4b = pd.read_csv(table4b_path)
    for model in MODELS:
        for set_name in ("ALL5", "NO_CKD"):
            for frac in FRACTIONS_TEST:
                row = _one_row(friedman, (friedman.metric == "auroc") & (friedman.unit == "BLOCK") & (friedman.fraction == frac) & (friedman.dataset_set == set_name),
                                f"friedman[auroc,BLOCK,{frac},{set_name}]")
                frac_tag = "f1.0" if frac == 1.0 else "f0.1"
                book.add(make_key("rank", model, ds_label(set_name), frac_tag), float(row[f"mean_rank_{model}"]), "rank", "", "primary",
                          friedman_path, f"friedman_summary.csv metric==auroc unit==BLOCK fraction=={frac} dataset_set=={set_name}, mean_rank_{model}", "3a")
            for frac in (0.25, 0.5):
                row = _one_row(table4b, (table4b.metric == "auroc") & (table4b.fraction == frac) & (table4b.dataset_set == set_name),
                                f"table4b[auroc,{frac},{set_name}]")
                frac_tag = f"f{frac}"
                book.add(make_key("rank", model, ds_label(set_name), frac_tag), float(row[model]), "rank", "", "primary",
                          table4b_path, f"table4b_mean_rank_raw.csv metric==auroc fraction=={frac} dataset_set=={set_name}, column {model}", "3b")

    # Friedman statistic/p for AUROC at fractions 1.0, 0.1 (descriptive).
    for set_name in ("ALL5", "NO_CKD"):
        for frac in FRACTIONS_TEST:
            row = _one_row(friedman, (friedman.metric == "auroc") & (friedman.unit == "BLOCK") & (friedman.fraction == frac) & (friedman.dataset_set == set_name),
                            f"friedman-stat[{frac},{set_name}]")
            frac_tag = "f1.0" if frac == 1.0 else "f0.1"
            book.add(make_key("friedman-chi2", ds_label(set_name), frac_tag), float(row.statistic), "diff", "", "descriptive",
                      friedman_path, f"friedman_summary.csv metric==auroc unit==BLOCK fraction=={frac} dataset_set=={set_name}, statistic", "3a")
            book.add(make_key("friedman-p", ds_label(set_name), frac_tag), float(row.p_value), "pvalue", "", "descriptive",
                      friedman_path, f"... p_value", "3a")

    # Holm-significant BLOCK pairs count, and the (zero) DATASET-unit count.
    pairwise_path = PROJECT_ROOT / "results" / "phase3a" / "pairwise_wilcoxon.csv"
    pairwise = pd.read_csv(pairwise_path)
    n_sig_block = int(pairwise[pairwise.unit == "BLOCK"]["significant_holm_0.05"].sum())
    n_sig_dataset = int(pairwise[pairwise.unit == "DATASET"]["significant_holm_0.05"].sum())
    book.add(make_key("n-holm-sig-block-pairs"), n_sig_block, "count", "pairs", "primary", pairwise_path,
              "pairwise_wilcoxon.csv unit==BLOCK, sum(significant_holm_0.05)", "3a")
    book.add(make_key("n-holm-sig-dataset-pairs"), n_sig_dataset, "count", "pairs", "primary", pairwise_path,
              "pairwise_wilcoxon.csv unit==DATASET, sum(significant_holm_0.05)", "3a")

    # Nadeau-Bengio corrected t-test: total cells, number Holm-significant, details of each.
    corrected_path = PROJECT_ROOT / "results" / "phase3a" / "corrected_ttest.csv"
    corrected = pd.read_csv(corrected_path)
    n_total_cells = len(corrected)
    sig = corrected[corrected.p_holm < 0.05]
    book.add(make_key("nb-n-total-cells"), n_total_cells, "count", "cells", "primary", corrected_path,
              "corrected_ttest.csv, len(df)", "3a")
    book.add(make_key("nb-n-sig-cells"), int(len(sig)), "count", "cells", "primary", corrected_path,
              "corrected_ttest.csv, count(p_holm < 0.05)", "3a")
    for i, (_, row) in enumerate(sig.iterrows()):
        base = make_key("nb-sig", i, row["dataset"], row["model"], row["metric"], f"f{row['fraction']}")
        book.add(f"{base}-meandiff", float(row.mean_diff_vs_logreg), "diff", "", "primary", corrected_path,
                  f"corrected_ttest.csv row {i} (significant), mean_diff_vs_logreg", "3a")
        book.add(f"{base}-t", float(row.t_statistic), "diff", "", "primary", corrected_path,
                  f"... t_statistic", "3a")
        book.add(f"{base}-pholm", float(row.p_holm), "pvalue", "", "primary", corrected_path,
                  f"... p_holm", "3a")


# ---------------------------------------------------------------------------
# Part 3, Group 3: learning-curve summary (AUROC change frac 0.1 -> 1.0)
# ---------------------------------------------------------------------------
def build_group3_learning_curve(book: NumbersBook) -> None:
    all_runs_path = PROJECT_ROOT / "results" / "all_runs.csv"
    all_runs = pd.read_csv(all_runs_path)
    for model in MODELS:
        for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
            means_at = {}
            for frac in (0.1, 1.0):
                per_dataset = [all_runs[(all_runs.dataset == d) & (all_runs.model == model) & (all_runs.fraction == frac)]["auroc"].mean() for d in datasets]
                means_at[frac] = float(np.mean(per_dataset))
            change = means_at[1.0] - means_at[0.1]
            book.add(make_key("auroc-change-f0.1-f1.0", model, ds_label(set_name)), change, "diff", "", "descriptive",
                      all_runs_path, f"all_runs.csv model=={model} datasets in {set_name}, dataset-avg mean(auroc)@frac1.0 minus @frac0.1", "3a")


# ---------------------------------------------------------------------------
# Part 3, Group 4: calibration (Phase 3b)
# ---------------------------------------------------------------------------
def build_group4_calibration(book: NumbersBook) -> None:
    p3b = PROJECT_ROOT / "results" / "phase3b"
    metrics_path = p3b / "calibration_metrics.csv"
    metrics_df = pd.read_csv(metrics_path)
    averaged_path = p3b / "calibration_averaged.csv"
    averaged = pd.read_csv(averaged_path)
    bl_path = p3b / "calibration_brier_logloss_averaged.csv"
    bl = pd.read_csv(bl_path)
    diff_path = p3b / "calibration_diff_vs_logreg.csv"
    diff = pd.read_csv(diff_path)
    posthoc_path = p3b / "posthoc_summary.csv"
    posthoc = pd.read_csv(posthoc_path)

    for model in MODELS:
        for frac in FRACTIONS_TEST:
            frac_tag = "f1.0" if frac == 1.0 else "f0.1"
            ece_row = _one_row(averaged, (averaged.model == model) & (averaged.fraction == frac) & (averaged.dataset_set == "ALL5"),
                                f"calibration_averaged[{model},{frac},ALL5]")
            book.add(make_key("ece-ew", model, "all5", frac_tag), float(ece_row.ece_ew_mean), "ece", "", "primary",
                      averaged_path, f"calibration_averaged.csv model=={model} fraction=={frac} dataset_set==ALL5, ece_ew_mean", "3b")
            book.add(make_key("ece-em", model, "all5", frac_tag), float(ece_row.ece_em_mean), "ece", "", "descriptive",
                      averaged_path, f"... ece_em_mean", "3b")
            bl_row = _one_row(bl, (bl.model == model) & (bl.fraction == frac) & (bl.dataset_set == "ALL5"), f"bl[{model},{frac}]")
            book.add(make_key("brier-pooled", model, "all5", frac_tag), float(bl_row.brier_mean), "brier", "", "primary",
                      bl_path, f"calibration_brier_logloss_averaged.csv model=={model} fraction=={frac} dataset_set==ALL5, brier_mean", "3b")
            book.add(make_key("logloss-pooled", model, "all5", frac_tag), float(bl_row.log_loss_mean), "logloss", "", "primary",
                      bl_path, f"... log_loss_mean", "3b")
            nockd_row = _one_row(averaged, (averaged.model == model) & (averaged.fraction == frac) & (averaged.dataset_set == "NO_CKD"),
                                  f"calibration_averaged[{model},{frac},NO_CKD]")
            slope_val = nockd_row.slope_mean
            if pd.isna(slope_val):
                book.add(make_key("slope", model, "nockd", frac_tag), None, "slope", "", "primary", averaged_path,
                          f"calibration_averaged.csv model=={model} fraction=={frac} dataset_set==NO_CKD, slope_mean",
                          "3b", undefined_reason="all NO_CKD datasets separable in this cell")
            else:
                book.add(make_key("slope", model, "nockd", frac_tag), float(slope_val), "slope", "", "primary",
                          averaged_path, f"calibration_averaged.csv model=={model} fraction=={frac} dataset_set==NO_CKD, slope_mean", "3b")
            intercept_val = nockd_row.intercept_mean
            if pd.isna(intercept_val):
                book.add(make_key("intercept", model, "nockd", frac_tag), None, "intercept", "", "descriptive", averaged_path,
                          f"calibration_averaged.csv model=={model} fraction=={frac} dataset_set==NO_CKD, intercept_mean",
                          "3b", undefined_reason="all NO_CKD datasets separable in this cell")
            else:
                book.add(make_key("intercept", model, "nockd", frac_tag), float(intercept_val), "intercept", "", "descriptive",
                          averaged_path, f"calibration_averaged.csv model=={model} fraction=={frac} dataset_set==NO_CKD, intercept_mean", "3b")

        if model == "logreg":
            continue
        for frac in FRACTIONS_TEST:
            frac_tag = "f1.0" if frac == 1.0 else "f0.1"
            for set_name in ("ALL5", "NO_CKD"):
                row = _one_row(diff, (diff.model == model) & (diff.fraction == frac) & (diff.dataset_set == set_name),
                                f"calibration_diff[{model},{frac},{set_name}]")
                base = make_key("ece-ew-diff-logreg", model, ds_label(set_name), frac_tag)
                book.add(f"{base}-point", float(row.ece_ew_diff_mean), "diff", "", "primary", diff_path,
                          f"calibration_diff_vs_logreg.csv model=={model} fraction=={frac} dataset_set=={set_name}, ece_ew_diff_mean", "3b")
                book.add(f"{base}-ci-lo", float(row.ci_lo), "ci", "", "primary", diff_path, "... ci_lo", "3b")
                book.add(f"{base}-ci-hi", float(row.ci_hi), "ci", "", "primary", diff_path, "... ci_hi", "3b")

    n_separable = int(metrics_df["separable"].sum())
    book.add(make_key("n-separable-cells"), n_separable, "count", "cells", "primary", metrics_path,
              "calibration_metrics.csv, sum(separable)", "3b")

    for model in MODELS:
        for set_name in ("ALL5", "NO_CKD"):
            row = _one_row(posthoc, (posthoc.fraction == 1.0) & (posthoc.model == model) & (posthoc.dataset_set == set_name),
                            f"posthoc[{model},{set_name}]")
            base = make_key("posthoc", model, ds_label(set_name))
            for method in ("platt", "isotonic"):
                book.add(f"{base}-{method}-brier-diff", float(row[f"{method}_brier_diff"]), "diff", "", "supplementary",
                          posthoc_path, f"posthoc_summary.csv fraction==1.0 model=={model} dataset_set=={set_name}, {method}_brier_diff", "3b")
                book.add(f"{base}-{method}-logloss-diff", float(row[f"{method}_log_loss_diff"]), "diff", "", "supplementary",
                          posthoc_path, f"... {method}_log_loss_diff", "3b")
                book.add(f"{base}-{method}-n-fallback", int(row[f"{method}_n_fallback_total"]), "count", "fold-fits", "supplementary",
                          posthoc_path, f"... {method}_n_fallback_total", "3b")


# ---------------------------------------------------------------------------
# Part 3, Group 5: robustness (Phases 3c and 3d)
# ---------------------------------------------------------------------------
def build_group5_robustness(book: NumbersBook) -> None:
    p3c = PROJECT_ROOT / "results" / "phase3c"
    averaged_path = p3c / "robustness_averaged.csv"
    averaged = pd.read_csv(averaged_path)
    diff_path = p3c / "robustness_diff_vs_logreg.csv"
    diff = pd.read_csv(diff_path)

    p3d_noise = PROJECT_ROOT / "results" / "phase3d" / "noise_continuous"
    cont_averaged_path = p3d_noise / "noise_cont_averaged.csv"
    cont_averaged = pd.read_csv(cont_averaged_path)
    cont_diff_path = p3d_noise / "noise_cont_diff_vs_logreg.csv"
    cont_diff = pd.read_csv(cont_diff_path)
    diag_path = p3d_noise / "negative_value_diagnostic.csv"
    diag = pd.read_csv(diag_path)

    for model in MODELS:
        for metric in ("auroc", "log_loss"):
            kind = METRIC_KIND[metric]
            for set_name in ("ALL5", "NO_CKD"):
                # Clean value: perturbed_mean - degradation_mean at any level (they share the same clean baseline).
                any_row = _one_row(averaged, (averaged.model == model) & (averaged.kind == "missing") & (averaged.level == 0.1) & (averaged.metric == metric) & (averaged.dataset_set == set_name),
                                    f"clean-lookup[{model},{metric},{set_name}]")
                clean_val = float(any_row.perturbed_mean - any_row.degradation_mean)
                base = make_key(kind, "clean", model, ds_label(set_name))
                book.add(base, clean_val, kind, "", "primary", averaged_path,
                          f"robustness_averaged.csv model=={model} kind==missing level==0.1 metric=={metric} dataset_set=={set_name}, perturbed_mean - degradation_mean", "3c")

                miss_row = _one_row(averaged, (averaged.model == model) & (averaged.kind == "missing") & (averaged.level == 0.3) & (averaged.metric == metric) & (averaged.dataset_set == set_name),
                                     f"miss30[{model},{metric},{set_name}]")
                book.add(make_key(kind, "miss30", model, ds_label(set_name)), float(miss_row.perturbed_mean), kind, "", "primary",
                          averaged_path, f"robustness_averaged.csv model=={model} kind==missing level==0.3 metric=={metric} dataset_set=={set_name}, perturbed_mean", "3c")
                book.add(make_key(kind, "miss30-degradation", model, ds_label(set_name)), float(miss_row.degradation_mean), "diff", "", "primary",
                          averaged_path, f"... degradation_mean", "3c")

                noise_row = _one_row(averaged, (averaged.model == model) & (averaged.kind == "noise") & (averaged.level == 0.5) & (averaged.metric == metric) & (averaged.dataset_set == set_name),
                                      f"noise50[{model},{metric},{set_name}]")
                book.add(make_key(kind, "noise50-orig", model, ds_label(set_name)), float(noise_row.perturbed_mean), kind, "", "primary",
                          averaged_path, f"robustness_averaged.csv model=={model} kind==noise level==0.5 metric=={metric} dataset_set=={set_name}, perturbed_mean", "3c")
                book.add(make_key(kind, "noise50-orig-degradation", model, ds_label(set_name)), float(noise_row.degradation_mean), "diff", "", "primary",
                          averaged_path, f"... degradation_mean", "3c")

                cont_row = _one_row(cont_averaged, (cont_averaged.model == model) & (cont_averaged.level == 0.5) & (cont_averaged.metric == metric) & (cont_averaged.dataset_set == set_name),
                                     f"noise50cont[{model},{metric},{set_name}]")
                book.add(make_key(kind, "noise50-cont", model, ds_label(set_name)), float(cont_row.perturbed_mean), kind, "", "primary",
                          cont_averaged_path, f"noise_cont_averaged.csv model=={model} level==0.5 metric=={metric} dataset_set=={set_name}, perturbed_mean", "3d")
                book.add(make_key(kind, "noise50-cont-degradation", model, ds_label(set_name)), float(cont_row.degradation_mean), "diff", "", "primary",
                          cont_averaged_path, f"... degradation_mean", "3d")

        if model == "logreg":
            continue
        for metric in ("auroc", "log_loss"):
            kind = METRIC_KIND[metric]
            for set_name in ("ALL5", "NO_CKD"):
                miss_diff = _one_row(diff, (diff.model == model) & (diff.kind == "missing") & (diff.level == 0.3) & (diff.metric == metric) & (diff.level_of_detail == "AVERAGED") & (diff.dataset_set == set_name),
                                      f"miss30diff[{model},{metric},{set_name}]")
                base = make_key(kind, "diff-logreg-degradation-miss30", model, ds_label(set_name))
                book.add(f"{base}-point", float(miss_diff.diff_mean), "diff", "", "primary", diff_path, f"robustness_diff_vs_logreg.csv model=={model} kind==missing level==0.3 metric=={metric} level_of_detail==AVERAGED dataset_set=={set_name}, diff_mean", "3c")
                book.add(f"{base}-ci-lo", float(miss_diff.ci_lo), "ci", "", "primary", diff_path, "... ci_lo", "3c")
                book.add(f"{base}-ci-hi", float(miss_diff.ci_hi), "ci", "", "primary", diff_path, "... ci_hi", "3c")

                noise_diff = _one_row(diff, (diff.model == model) & (diff.kind == "noise") & (diff.level == 0.5) & (diff.metric == metric) & (diff.level_of_detail == "AVERAGED") & (diff.dataset_set == set_name),
                                       f"noise50diff[{model},{metric},{set_name}]")
                base2 = make_key(kind, "diff-logreg-degradation-noise50-orig", model, ds_label(set_name))
                book.add(f"{base2}-point", float(noise_diff.diff_mean), "diff", "", "primary", diff_path, f"robustness_diff_vs_logreg.csv model=={model} kind==noise level==0.5 metric=={metric} level_of_detail==AVERAGED dataset_set=={set_name}, diff_mean", "3c")
                book.add(f"{base2}-ci-lo", float(noise_diff.ci_lo), "ci", "", "primary", diff_path, "... ci_lo", "3c")
                book.add(f"{base2}-ci-hi", float(noise_diff.ci_hi), "ci", "", "primary", diff_path, "... ci_hi", "3c")

                cont_diff_row = _one_row(cont_diff, (cont_diff.model == model) & (cont_diff.metric == metric) & (cont_diff.level_of_detail == "AVERAGED") & (cont_diff.dataset_set == set_name),
                                          f"noise50contdiff[{model},{metric},{set_name}]")
                base3 = make_key(kind, "diff-logreg-degradation-noise50-cont", model, ds_label(set_name))
                book.add(f"{base3}-point", float(cont_diff_row.diff_mean), "diff", "", "primary", cont_diff_path, f"noise_cont_diff_vs_logreg.csv model=={model} metric=={metric} level_of_detail==AVERAGED dataset_set=={set_name}, diff_mean", "3d")
                book.add(f"{base3}-ci-lo", float(cont_diff_row.ci_lo), "ci", "", "primary", cont_diff_path, "... ci_lo", "3d")
                book.add(f"{base3}-ci-hi", float(cont_diff_row.ci_hi), "ci", "", "primary", cont_diff_path, "... ci_hi", "3d")

    # Negative-value diagnostic rates (Phase 3d).
    for _, row in diag.iterrows():
        base = make_key("negrate", row["dataset"], f"level{row['level']}")
        book.add(f"{base}-orig", float(row.frac_negative_noise) * 100, "percentage", "%", "supplementary", diag_path,
                  f"negative_value_diagnostic.csv dataset=={row['dataset']} level=={row['level']}, frac_negative_noise*100", "3d")
        book.add(f"{base}-cont", float(row.frac_negative_noise_cont) * 100, "percentage", "%", "supplementary", diag_path,
                  f"... frac_negative_noise_cont*100", "3d")


# ---------------------------------------------------------------------------
# Part 3, Group 6: explanations (Phase 3d SHAP)
# ---------------------------------------------------------------------------
SHAP_MODELS = ["logreg", "rf", "mlp"]
SHAP_PAIRS = [("logreg", "rf"), ("logreg", "mlp"), ("rf", "mlp")]


def build_group6_explanations(book: NumbersBook) -> None:
    p3d = PROJECT_ROOT / "results" / "phase3d"
    per_dataset_path = p3d / "agreement_per_dataset.csv"
    per_dataset = pd.read_csv(per_dataset_path)
    averaged_path = p3d / "agreement_averaged.csv"
    averaged = pd.read_csv(averaged_path)
    baselines_path = p3d / "baselines.csv"
    baselines = pd.read_csv(baselines_path)
    plausibility_path = p3d / "plausibility.csv"
    plausibility = pd.read_csv(plausibility_path)

    for model_a, model_b in SHAP_PAIRS:
        pair_tag = f"{model_a}-{model_b}"
        for ds in DATASETS_ALL5:
            row = _one_row(per_dataset, (per_dataset.dataset == ds) & (per_dataset.model_a == model_a) & (per_dataset.model_b == model_b),
                            f"agreement_per_dataset[{ds},{pair_tag}]")
            book.add(make_key("jaccard", pair_tag, ds), float(row.jaccard_mean), "jaccard", "", "primary", per_dataset_path,
                      f"agreement_per_dataset.csv dataset=={ds} model_a=={model_a} model_b=={model_b}, jaccard_mean", "3d")
            book.add(make_key("spearman", pair_tag, ds), float(row.spearman_mean), "spearman", "", "primary", per_dataset_path,
                      f"... spearman_mean", "3d")
        for set_name in ("ALL5", "NO_CKD"):
            row = _one_row(averaged, (averaged.model_a == model_a) & (averaged.model_b == model_b) & (averaged.dataset_set == set_name),
                            f"agreement_averaged[{pair_tag},{set_name}]")
            book.add(make_key("jaccard", pair_tag, ds_label(set_name)), float(row.jaccard_mean), "jaccard", "", "primary", averaged_path,
                      f"agreement_averaged.csv model_a=={model_a} model_b=={model_b} dataset_set=={set_name}, jaccard_mean", "3d")
            book.add(make_key("spearman", pair_tag, ds_label(set_name)), float(row.spearman_mean), "spearman", "", "primary", averaged_path,
                      f"... spearman_mean", "3d")

    for ds in DATASETS_ALL5:
        stab = _one_row(baselines, (baselines.dataset == ds) & (baselines.type == "within_model_mean_over_models"), f"stability[{ds}]")
        book.add(make_key("stability-jaccard", ds), float(stab.jaccard_mean), "jaccard", "", "primary", baselines_path,
                  f"baselines.csv dataset=={ds} type==within_model_mean_over_models, jaccard_mean", "3d")
        book.add(make_key("stability-spearman", ds), float(stab.spearman_mean), "spearman", "", "primary", baselines_path,
                  f"... spearman_mean", "3d")
        chance = _one_row(baselines, (baselines.dataset == ds) & (baselines.type == "chance"), f"chance[{ds}]")
        book.add(make_key("chance-jaccard", ds), float(chance.chance_jaccard), "jaccard", "", "primary", baselines_path,
                  f"baselines.csv dataset=={ds} type==chance, chance_jaccard", "3d")
        book.add(make_key("m-features", ds), int(chance.m_features), "count", "features", "primary", baselines_path,
                  f"baselines.csv dataset=={ds} type==chance, m_features", "3d")

    for ds in DATASETS_ALL5:
        for model in SHAP_MODELS:
            row = _one_row(plausibility, (plausibility.dataset == ds) & (plausibility.model == model), f"plausibility[{ds},{model}]")
            base = make_key("plausibility", model, ds)
            book.add(f"{base}-precision", float(row.precision_mean), "precision", "", "primary", plausibility_path,
                      f"plausibility.csv dataset=={ds} model=={model}, precision_mean", "3d")
            book.add(f"{base}-ci-lo", float(row.precision_ci_lo), "ci", "", "primary", plausibility_path, "... precision_ci_lo", "3d")
            book.add(f"{base}-ci-hi", float(row.precision_ci_hi), "ci", "", "primary", plausibility_path, "... precision_ci_hi", "3d")
            book.add(f"{base}-chance", float(row.chance_precision), "precision", "", "primary", plausibility_path,
                      f"... chance_precision", "3d")

    # Number of distinct top-5 sets per model for Breast Wisconsin (recomputed from the per-task SHAP npz).
    from analysis.explain import top_k_with_ties

    for model in SHAP_MODELS:
        top5_sets = set()
        for split in range(10):
            npz_path = p3d / "shap" / f"breast_wisconsin__{model}__split{split}.npz"
            data = np.load(npz_path, allow_pickle=True)
            feat = [str(f) for f in data["feature_names"]]
            imp = data["importance"]
            top5_sets.add(frozenset(top_k_with_ties(imp, feat, 5)))
        book.add(make_key("n-distinct-top5-breast-wisconsin", model), len(top5_sets), "count", "sets", "primary",
                  p3d / "shap" / "breast_wisconsin__rf__split0.npz",
                  f"recomputed from results/phase3d/shap/breast_wisconsin__{model}__split*.npz (10 files), number of distinct top-5 feature sets", "3d")


# ---------------------------------------------------------------------------
# Part 3, Group 7: sensitivity and diagnostics (Phase 2c)
# ---------------------------------------------------------------------------
DEEP_MODELS = ["mlp", "ftt", "tabnet"]
VARIANTS = ["A_long", "B_bigval", "C_long_bigval"]


def build_group7_sensitivity(book: NumbersBook) -> None:
    p2c = PROJECT_ROOT / "results" / "phase2c"
    sens_path = p2c / "sensitivity_runs.csv"
    sens = pd.read_csv(sens_path)
    sens_ok = sens[sens.status == "ok"]

    for model in DEEP_MODELS:
        for frac in FRACTIONS_TEST:
            frac_tag = "f1.0" if frac == 1.0 else "f0.1"
            baseline = sens_ok[(sens_ok.model == model) & (sens_ok.variant == "baseline") & (sens_ok.fraction == frac)]
            baseline_keyed = baseline.set_index(["dataset", "split_idx"])
            for variant in VARIANTS:
                var_rows = sens_ok[(sens_ok.model == model) & (sens_ok.variant == variant) & (sens_ok.fraction == frac)]
                diffs = {"auroc": [], "log_loss": [], "brier": []}
                for _, row in var_rows.iterrows():
                    key = (row["dataset"], row["split_idx"])
                    if key not in baseline_keyed.index:
                        continue
                    base_row = baseline_keyed.loc[key]
                    for metric in diffs:
                        diffs[metric].append(row[metric] - base_row[metric])
                for metric, vals in diffs.items():
                    if not vals:
                        continue
                    kind = METRIC_KIND[metric]
                    book.add(make_key(kind, "sens-diff", model, variant, frac_tag), float(np.mean(vals)), "diff", "", "supplementary",
                              sens_path, f"sensitivity_runs.csv model=={model} variant=={variant} fraction=={frac} minus matching baseline rows, mean({metric} diff)", "2c")

    # TabNet on CKD split 1, fraction 1.0, baseline + each variant.
    for variant in ["baseline"] + VARIANTS:
        row = sens_ok[(sens_ok.model == "tabnet") & (sens_ok.dataset == "ckd") & (sens_ok.split_idx == 1) & (sens_ok.fraction == 1.0) & (sens_ok.variant == variant)]
        if len(row) == 0:
            continue
        row = row.iloc[0]
        book.add(make_key("tabnet-ckd-split1-f1.0", variant), float(row.auroc), "auroc", "", "primary", sens_path,
                  f"sensitivity_runs.csv model==tabnet dataset==ckd split_idx==1 fraction==1.0 variant=={variant}, auroc", "2c")

    collapsed_path = p2c / "tabnet_collapsed_splits.csv"
    collapsed = pd.read_csv(collapsed_path)
    book.add(make_key("n-tabnet-collapsed-splits"), len(collapsed), "count", "splits", "primary", collapsed_path,
              "tabnet_collapsed_splits.csv, len(df)", "2c")

    endpoints_path = p2c / "endpoints_overall.csv"
    endpoints = pd.read_csv(endpoints_path)
    high_endpoint = endpoints[(endpoints.n_at_lower >= 15) | (endpoints.n_at_upper >= 15)]
    book.add(make_key("n-high-endpoint-params"), len(high_endpoint), "count", "parameters", "primary", endpoints_path,
              "endpoints_overall.csv, count(n_at_lower>=15 or n_at_upper>=15)", "2c")
    for _, row in high_endpoint.iterrows():
        base = make_key("endpoint", row["model"], row["parameter"])
        book.add(f"{base}-n-lower", int(row.n_at_lower), "count", "tasks", "primary", endpoints_path,
                  f"endpoints_overall.csv model=={row['model']} parameter=={row['parameter']}, n_at_lower", "2c")
        book.add(f"{base}-n-upper", int(row.n_at_upper), "count", "tasks", "primary", endpoints_path,
                  f"... n_at_upper", "2c")

    # Epoch-cap facts (Phase 3b, section (ii) of epoch_cap_check.csv: baseline mlp, all 10 splits).
    epoch_cap_path = PROJECT_ROOT / "results" / "phase3b" / "epoch_cap_check.csv"
    raw_text = epoch_cap_path.read_text(encoding="utf-8")
    section_ii_start = raw_text.index("# (ii)")
    section_iii_start = raw_text.index("# (iii)")
    section_ii_text = raw_text[section_ii_start:section_iii_start].split("\n", 1)[1]
    import io

    section_ii = pd.read_csv(io.StringIO(section_ii_text))
    for _, row in section_ii.iterrows():
        frac_tag = "f1.0" if row["fraction"] == 1.0 else "f0.1"
        base = make_key("epochcap-mlp-baseline", frac_tag)
        book.add(f"{base}-n-hit-cap", int(row.n_hit_cap), "count", "tasks", "descriptive", epoch_cap_path,
                  f"epoch_cap_check.csv section (ii) fraction=={row['fraction']}, n_hit_cap", "3b")
        book.add(f"{base}-mean-epochs-run", float(row.mean_epochs_run), "diff", "", "descriptive", epoch_cap_path,
                  f"... mean_epochs_run", "3b")
        book.add(f"{base}-max-epochs-seen", int(row.max_epochs_seen), "count", "epochs", "descriptive", epoch_cap_path,
                  f"... max_epochs_seen", "3b")


# ---------------------------------------------------------------------------
# Phase 4b: new quantities needed by the claim map, group "4b".
#
# C3 needs the dataset-averaged DIFFERENCE OF DIFFERENCES in AUROC gap to
# logreg between fraction 0.1 and fraction 1.0, per model. C6 needs the
# paired AUROC/log-loss... (log loss only, per the claim text) difference
# between the dataset-averaged log-loss LEADER and the SECOND-best model, at
# each (fraction, dataset_set), for the two leader/second-best pairs that do
# NOT already involve logreg (the two that do -- f1.0/NO_CKD: logreg vs mlp;
# f0.1/NO_CKD: rf vs logreg -- reuse the existing logloss-diff-logreg-* keys
# instead of being recomputed here).
#
# Both use the SAME nested-bootstrap procedure as Phases 3a/3c: seed 42,
# B = 5000, resample the 10 splits WITHIN each dataset (same resampled
# indices reused for every series compared for that dataset, so pairing is
# preserved under resampling), then average the per-dataset bootstrap means.
# ---------------------------------------------------------------------------
def _diff_of_diff_bootstrap(
    all_runs: pd.DataFrame, model: str, metric: str, datasets: list[str], seed: int, b: int = 5000,
) -> tuple[float, float, float, list[float]]:
    """CI for dataset-averaged (model-logreg diff @ f0.1) - (model-logreg diff @ f1.0).

    Returns (point, ci_lo, ci_hi, point_per_dataset) -- the last for the
    per-dataset sign-agreement count used by the claim-map status rule.
    """
    rng = np.random.default_rng(seed)
    boot_per_dataset = []
    point_per_dataset = []
    for ds in datasets:
        series = {}
        for frac in (0.1, 1.0):
            for who in (model, "logreg"):
                vals = (
                    all_runs[(all_runs.dataset == ds) & (all_runs.model == who) & (all_runs.fraction == frac)]
                    .sort_values("split_idx")[metric]
                    .to_numpy()
                )
                if len(vals) != 10:
                    raise RuntimeError(f"expected 10 split values for {ds}/{who}/frac={frac}/{metric}, got {len(vals)}")
                series[(frac, who)] = vals
        n = 10
        idx = rng.integers(0, n, size=(b, n))
        diff01_boot = series[(0.1, model)][idx].mean(axis=1) - series[(0.1, "logreg")][idx].mean(axis=1)
        diff10_boot = series[(1.0, model)][idx].mean(axis=1) - series[(1.0, "logreg")][idx].mean(axis=1)
        dod_boot = diff01_boot - diff10_boot
        boot_per_dataset.append(dod_boot)
        diff01_point = series[(0.1, model)].mean() - series[(0.1, "logreg")].mean()
        diff10_point = series[(1.0, model)].mean() - series[(1.0, "logreg")].mean()
        point_per_dataset.append(float(diff01_point - diff10_point))
    boot_avg = np.mean(np.vstack(boot_per_dataset), axis=0)
    point = float(np.mean(point_per_dataset))
    lo, hi = np.percentile(boot_avg, [2.5, 97.5])
    return point, float(lo), float(hi), point_per_dataset


def build_group8_phase4b_claims(book: NumbersBook) -> None:
    from analysis.bootstrap import bootstrap_diff_ci_averaged

    all_runs_path = PROJECT_ROOT / "results" / "all_runs.csv"
    all_runs = pd.read_csv(all_runs_path)
    selector_note = (
        "all_runs.csv, nested bootstrap seed=42 B=5000 resampling the 10 splits per dataset "
        "(same resampled indices reused across all series compared for that dataset), "
        "dataset-averaged (model-logreg diff @ f0.1) - (model-logreg diff @ f1.0), auroc"
    )

    # C3: diff-of-differences (AUROC gap to logreg at f0.1 minus at f1.0), per model.
    for model in MODELS:
        if model == "logreg":
            continue
        for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
            seed = 42  # fixed seed, per-model/per-dataset-set reproducibility is from re-deriving the rng fresh each call
            point, lo, hi, per_ds = _diff_of_diff_bootstrap(all_runs, model, "auroc", datasets, seed)
            n_total = len(datasets)
            n_agree = sum(1 for v in per_ds if (v > 0) == (point > 0)) if point != 0 else sum(1 for v in per_ds if v == 0)
            base = make_key("auroc-dod-f0.1-f1.0", model, ds_label(set_name))
            book.add(f"{base}-point", point, "diff", "", "primary", all_runs_path, selector_note + f" | model={model} dataset_set={set_name}", "4b")
            book.add(f"{base}-ci-lo", lo, "ci", "", "primary", all_runs_path, selector_note + " | ci_lo", "4b")
            book.add(f"{base}-ci-hi", hi, "ci", "", "primary", all_runs_path, selector_note + " | ci_hi", "4b")
            book.add(f"{base}-nsign-agree", n_agree, "count", "datasets", "descriptive", all_runs_path,
                      selector_note + " | count of per-dataset point estimates sharing the overall point estimate's sign", "4b")
            book.add(f"{base}-ntotal", n_total, "count", "datasets", "descriptive", all_runs_path,
                      selector_note + " | number of datasets in this dataset_set", "4b")

    # C6: paired log-loss difference, dataset-averaged leader vs. second-best, for
    # the two (fraction, ALL5) cells whose leader/second-best pair does not already
    # involve logreg (f1.0/NO_CKD and f0.1/NO_CKD are logreg-involving and already
    # covered by the existing logloss-diff-logreg-* keys from build_group2).
    new_pairs = [
        (1.0, "ALL5", "mlp", "xgb"),  # f1.0 ALL5 leader=mlp, second=xgb
        (0.1, "ALL5", "rf", "xgb"),   # f0.1 ALL5 leader=rf, second=xgb
    ]
    for frac, set_name, leader, second in new_pairs:
        datasets = DATASETS_ALL5 if set_name == "ALL5" else DATASETS_NO_CKD
        leader_by_ds = [
            all_runs[(all_runs.dataset == d) & (all_runs.model == leader) & (all_runs.fraction == frac)].sort_values("split_idx")["log_loss"].to_numpy()
            for d in datasets
        ]
        second_by_ds = [
            all_runs[(all_runs.dataset == d) & (all_runs.model == second) & (all_runs.fraction == frac)].sort_values("split_idx")["log_loss"].to_numpy()
            for d in datasets
        ]
        rng = np.random.default_rng(42)
        point, lo, hi = bootstrap_diff_ci_averaged(leader_by_ds, second_by_ds, rng, 5000)
        frac_tag = "f1.0" if frac == 1.0 else "f0.1"
        base = make_key("logloss-diff-leader-2nd", leader, second, ds_label(set_name), frac_tag)
        sel = (
            f"all_runs.csv model in {{{leader},{second}}} fraction=={frac} dataset_set=={set_name}, "
            "nested bootstrap seed=42 B=5000 (same analysis.bootstrap.bootstrap_diff_ci_averaged as Phase 3a), "
            f"dataset-averaged ({leader} - {second}) log_loss"
        )
        book.add(f"{base}-point", point, "diff", "", "primary", all_runs_path, sel, "4b")
        book.add(f"{base}-ci-lo", lo, "ci", "", "primary", all_runs_path, sel + " | ci_lo", "4b")
        book.add(f"{base}-ci-hi", hi, "ci", "", "primary", all_runs_path, sel + " | ci_hi", "4b")

    # C11: fit-time ratios to logreg, dataset-averaged ALL5, fraction 1.0.
    timing_path = PROJECT_ROOT / "results" / "phase3a" / "table5_calibration_timing.csv"
    timing = pd.read_csv(timing_path)
    logreg_time = float(_one_row(timing, timing.model == "logreg", "timing[logreg]").fit_time_s)
    for model in MODELS:
        if model == "logreg":
            continue
        model_time = float(_one_row(timing, timing.model == model, f"timing[{model}]").fit_time_s)
        ratio = model_time / logreg_time
        book.add(make_key("fit-time-ratio-logreg", model), ratio, "diff", "x", "descriptive", timing_path,
                  f"table5_calibration_timing.csv, fit_time_s[{model}] / fit_time_s[logreg] (measured with 3-4 parallel workers; rough)", "4b")


# ---------------------------------------------------------------------------
# Phase 4d: dataset facts for Table I and the drafting sheet, group "4d".
# Additions only: every value is read from dataset_summary.csv, all_runs.csv or the
# cached dataset's meta (src.data.load_dataset); existing keys are untouched.
# ---------------------------------------------------------------------------
def build_group9_phase4d(book: NumbersBook) -> None:
    summary_path = PROJECT_ROOT / "results" / "dataset_summary.csv"
    summary = pd.read_csv(summary_path)
    runs_path = PROJECT_ROOT / "results" / "all_runs.csv"
    runs = pd.read_csv(runs_path)
    for ds in DATASETS_ALL5:
        row = _one_row(summary, summary.dataset == ds, f"dataset_summary[{ds}]")
        for key, col, unit in (("n-numeric", "n_numeric", "features"), ("n-categorical", "n_categorical", "features"),
                               ("n-rows-raw", "n_rows_before_dedup", "rows"), ("n-duplicates-dropped", "n_duplicates_found", "rows")):
            book.add(make_key(key, ds), int(row[col]), "count", unit, "descriptive", summary_path,
                     f"dataset_summary.csv row dataset=={ds}, column {col}", "4d")
        sub = runs[runs.dataset == ds]
        book.add(make_key("min-train-f0.1", ds), int(sub[sub.fraction == 0.1].n_train.min()), "count", "rows", "descriptive",
                 runs_path, f"all_runs.csv dataset=={ds} & fraction==0.1, min(n_train) over the 10 splits", "4d")
        book.add(make_key("n-test-min", ds), int(sub.n_test.min()), "count", "rows", "descriptive", runs_path,
                 f"all_runs.csv dataset=={ds}, min(n_test)", "4d")
        book.add(make_key("n-test-max", ds), int(sub.n_test.max()), "count", "rows", "descriptive", runs_path,
                 f"all_runs.csv dataset=={ds}, max(n_test)", "4d")
    book.add("n-fractions", int(runs.fraction.nunique()), "count", "fractions", "descriptive", runs_path,
             "all_runs.csv, nunique(fraction)", "4d")
    book.add("n-datasets-no-ckd", len(DATASETS_NO_CKD), "count", "datasets", "descriptive", runs_path,
             "analysis.constants.DATASETS_NO_CKD (ALL5 minus ckd); all_runs.csv listed as the dataset roster source", "4d")

    # Pima zero-to-missing replacement counts and per-column missingness (cached dataset meta).
    pima_pkl = PROJECT_ROOT / "data" / "pima_diabetes_raw.pkl"
    _, _, meta = load_dataset("pima_diabetes", verbose=False)
    for col, n in meta["zeros_replaced_with_nan"].items():
        book.add(make_key("zeros-replaced", "pima_diabetes", col), int(n), "count", "values", "descriptive", pima_pkl,
                 f"load_dataset('pima_diabetes').meta['zeros_replaced_with_nan']['{col}']", "4d")
    for col in ("insu", "skin"):
        book.add(make_key("missing-col", "pima_diabetes", col), meta["missing_fraction"][col] * 100, "percentage", "%",
                 "descriptive", pima_pkl, f"load_dataset('pima_diabetes').meta['missing_fraction']['{col}']*100", "4d")
    ckd_pkl = PROJECT_ROOT / "data" / "ckd_raw.pkl"
    _, _, ckd_meta = load_dataset("ckd", verbose=False)
    # Per-(dataset, model) AUROC excess over logreg (model mean - logreg mean over the 10 splits), for the
    # claim-map C1/C2 exception cells; positive value = the model's per-dataset mean exceeds logreg's.
    for frac in (1.0, 0.1):
        means = runs[runs.fraction == frac].groupby(["dataset", "model"]).auroc.mean()
        for ds in DATASETS_ALL5:
            for model in MODELS:
                if model == "logreg":
                    continue
                book.add(make_key("auroc-excess-over-logreg", ds, model, f"f{frac}"), float(means[(ds, model)] - means[(ds, "logreg")]),
                         "diff", "", "descriptive", runs_path,
                         f"all_runs.csv dataset=={ds} fraction=={frac}: mean(auroc) of {model} minus mean(auroc) of logreg over the 10 splits", "4d")
    # Outer CV scheme (src.utils.make_splits defaults: 5 folds x 2 repeats, seed 42).
    from src.utils import make_splits
    sig = inspect.signature(make_splits).parameters
    utils_path = PROJECT_ROOT / "src" / "utils.py"
    book.add("n-cv-folds", sig["n_splits"].default, "count", "folds", "descriptive", utils_path,
             "inspect.signature(src.utils.make_splits).parameters['n_splits'].default", "4d")
    book.add("n-cv-repeats", sig["n_repeats"].default, "count", "repeats", "descriptive", utils_path,
             "inspect.signature(src.utils.make_splits).parameters['n_repeats'].default", "4d")

    # Per-dataset pooled calibration slope with 95% bootstrap CI (Phase 3b), fractions 1.0 and 0.1.
    cal_path = PROJECT_ROOT / "results" / "phase3b" / "calibration_metrics.csv"
    ci_path = PROJECT_ROOT / "results" / "phase3b" / "calibration_bootstrap_ci.csv"
    cal, cal_ci = pd.read_csv(cal_path), pd.read_csv(ci_path)
    for frac in (1.0, 0.1):
        for ds in DATASETS_ALL5:
            for model in MODELS:
                r = _one_row(cal, (cal.dataset == ds) & (cal.model == model) & (cal.fraction == frac), f"calibration_metrics[{ds},{model},{frac}]")
                c = _one_row(cal_ci, (cal_ci.dataset == ds) & (cal_ci.model == model) & (cal_ci.fraction == frac), f"calibration_bootstrap_ci[{ds},{model},{frac}]")
                base = make_key("slope-per-dataset", ds, model, f"f{frac}")
                sel = f"calibration_metrics.csv / calibration_bootstrap_ci.csv dataset=={ds} model=={model} fraction=={frac}"
                book.add(f"{base}-point", float(r.slope), "slope", "", "primary", cal_path, sel + ", slope", "4d")
                book.add(f"{base}-ci-lo", float(c.slope_lo), "slope", "", "primary", ci_path, sel + ", slope_lo", "4d")
                book.add(f"{base}-ci-hi", float(c.slope_hi), "slope", "", "primary", ci_path, sel + ", slope_hi", "4d")
    book.add("n-ordinal-as-numeric-ckd", len(ckd_meta["ordinal_as_numeric"]), "count", "columns", "descriptive", ckd_pkl,
             "len(load_dataset('ckd').meta['ordinal_as_numeric'])", "4d")


# ---------------------------------------------------------------------------
# Output writers: numbers.json / .csv / .tex / .md / numbers_index.md
# ---------------------------------------------------------------------------
_LATEX_ESCAPE = {"%": r"\%", "#": r"\#", "&": r"\&", "_": r"\_"}


def latex_safe_formatted(formatted: str) -> str:
    """Escape % # & _ and write a leading minus sign as $-$ (LaTeX text mode)."""
    if formatted is None:
        return ""
    s = "".join(_LATEX_ESCAPE.get(ch, ch) for ch in formatted)
    if s.startswith("-"):
        s = "$-$" + s[1:]
    s = s.replace("<0.001", "$<$0.001")
    return s


def write_numbers_outputs(book: NumbersBook) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = sorted(book.entries, key=lambda e: e["key"])

    # numbers.json
    (OUTPUT_DIR / "numbers.json").write_text(json.dumps(entries, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    # numbers.csv
    df = pd.DataFrame(entries)
    df.to_csv(OUTPUT_DIR / "numbers.csv", index=False)

    # numbers.tex
    tex_lines = [
        "% numbers.tex -- generated by scripts/build_numbers.py. DO NOT EDIT BY HAND.",
        "% Every number used in the paper must come from \\N{key}. An undefined key makes",
        "% LaTeX fail to compile -- this is intended: it means the key does not exist (yet)",
        "% in numbers.json, so the paper would otherwise be citing an unverified number.",
        "\\newcommand{\\N}[1]{\\csname num:#1\\endcsname}",
        "",
    ]
    for e in entries:
        if e["undefined"]:
            continue  # an undefined key is deliberately left undefined in LaTeX too
        tex_lines.append(r"\expandafter\def\csname num:" + e["key"] + r"\endcsname{" + latex_safe_formatted(e["formatted"]) + "}")
    (OUTPUT_DIR / "numbers.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    # numbers.md
    md_lines = [
        "# numbers.json / numbers.tex -- usage and rounding rules\n",
        "## Usage in the paper",
        "Write `\\N{key}` in the LaTeX source (after `\\input{paper/evidence/numbers.tex}`).",
        "An undefined key makes LaTeX fail to compile -- this is intended, not a bug: it",
        "means that number is not (yet) in the numbers book, so it must not be hand-typed.",
        "",
        "## Rounding rules",
        "- AUROC, Brier, log loss, ECE, slopes, intercepts, their differences and CI limits: 3 decimals.",
        "- p-values: 3 decimals, or `<0.001` when below 0.001.",
        "- Mean ranks: 2 decimals.",
        "- Jaccard, Spearman, precision: 2 decimals.",
        "- Percentages: 1 decimal.",
        "- Times (seconds): 2 decimals.",
        "- Counts: integers.",
        "- Negative zero (e.g. -0.0001 rounded to 3 dp) prints as `0.000`, never `-0.000`.",
        "",
        "## LaTeX safety",
        "Formatted strings are escaped for `% # & _`; a leading minus sign is written as",
        "`$-$` (so it renders correctly in text mode); `<0.001` is written as `$<$0.001`.",
        "",
        "## Key naming scheme",
        "Lowercase `a-z0-9.-` only. Dataset-set labels: `ALL5` -> `all5`, `NO_CKD` -> `nockd`.",
        "Examples:",
        "- `auroc-mean-logreg-all5-f1.0` -- dataset-averaged ALL5 mean AUROC, logreg, fraction 1.0.",
        "- `jaccard-logreg-rf-ckd` -- top-5 Jaccard between logreg and rf on ckd.",
        "- `n-holm-sig-block-pairs` -- count of Holm-significant BLOCK-unit pairs (Phase 3a).",
        "",
        f"Total entries: {len(entries)}.",
    ]
    (OUTPUT_DIR / "numbers.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # numbers_index.md: grouped by phase, then alphabetically, with formatted values.
    index_lines = ["# numbers_index.md -- all keys grouped by phase (for searching)\n"]
    by_phase: dict[str, list[dict]] = {}
    for e in entries:
        by_phase.setdefault(e["phase"], []).append(e)
    for phase in sorted(by_phase):
        group = by_phase[phase]
        index_lines.append(f"## Phase {phase} ({len(group)} keys)\n")
        for e in sorted(group, key=lambda x: x["key"]):
            value_str = "UNDEFINED" if e["undefined"] else e["formatted"]
            index_lines.append(f"- `{e['key']}` = {value_str} ({e['role']})")
        index_lines.append("")
    (OUTPUT_DIR / "numbers_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Part 4: inventory of figures and tables
# ---------------------------------------------------------------------------
_ID_RE = re.compile(r"(fig\d+[a-z]?|table\d+[a-z]?)", re.IGNORECASE)


def parse_captions_file(path: Path) -> dict[str, str]:
    """{id: first-paragraph text, verbatim (whitespace-joined)} from a captions/notes .md."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^##\s+", text)[1:]
    result: dict[str, str] = {}
    for section in sections:
        header, _, body = section.partition("\n")
        ids = [m.group(1).lower() for m in _ID_RE.finditer(header)]
        if not ids:
            continue
        first_para = body.strip().split("\n\n", 1)[0]
        body_text = " ".join(line.strip() for line in first_para.splitlines() if line.strip())
        for id_ in ids:
            id_normalized = id_.replace(" ", "")
            if id_normalized not in result:
                result[id_normalized] = body_text
    return result


def png_dimensions(path: Path) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as fh:
            header = fh.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        return width, height
    except OSError:
        return None


def latex_table_width_estimate(tex_path: Path) -> tuple[float, int]:
    """Rough physical-width estimate for a booktabs table: for each COLUMN, take the
    longest cell's character count (\\textbf{} stripped) across all rows (including
    the header), convert to inches (0.075in/char + 0.12in padding per column), and
    SUM over columns. Documented heuristic, not a real LaTeX layout engine --
    flagged as an estimate everywhere it is reported.
    """
    text = tex_path.read_text(encoding="utf-8")
    body_lines = [l.strip() for l in text.splitlines() if l.strip().endswith(r"\\") and "&" in l]
    if not body_lines:
        return 0.0, 0
    rows = [[re.sub(r"\\textbf\{(.*?)\}", r"\1", cell).strip() for cell in line.rstrip("\\").rstrip().rstrip("\\").split("&")] for line in body_lines]
    n_cols = max(len(r) for r in rows)
    col_max_chars = [0] * n_cols
    for row in rows:
        for i, cell in enumerate(row):
            col_max_chars[i] = max(col_max_chars[i], len(cell))
    width_in = sum(chars * 0.075 + 0.12 for chars in col_max_chars)
    return round(width_in, 2), n_cols


# Curated (not auto-parsed) per-ID metadata, from reading the phase scripts/notes
# this same build authored. Panel counts match the plotting code; data sources
# match each subcommand's documented inputs; known_issues are copied from the
# phase's own notes.md / caption text / CLAUDE.md decisions.
_PANEL_COUNTS = {
    "fig1": 6, "fig1b": 6, "fig1c": 6, "fig2": 1, "fig3": 1,
    "fig4": 7, "fig4b": 7, "fig4c": 35, "fig5": 2, "fig6": 4,
    "fig7": 2, "fig7b": 2, "fig7c": 2, "fig7d": 2, "fig7e": 10, "fig8": 1,
    "fig9": 2, "fig10": 5, "fig11": 2,
}
_KNOWN_ISSUES = {
    "table3": "Phase 2a/2b's own Table 3 (per-split mean ECE) is inflated by small test folds (60-154 rows) and must not be quoted (CLAUDE.md decision 11); not to be confused with Phase 3a's table3_mean_auroc_*.csv, which reports AUROC only.",
    "table4": "Only fractions 1.0 and 0.1 (Phase 3a); fractions 0.25 and 0.5 are in Phase 3b's table4b instead.",
    "table4b": "Only fractions 0.25 and 0.5 (Phase 3b); fractions 1.0 and 0.1 are in Phase 3a's table4 instead. Descriptive only (no hypothesis tests).",
    "table5": "fit_time_s / predict_time_s measured with 3 to 4 parallel workers; rough.",
    "table6": "NO_CKD slope/intercept; 0 cells separable in this run (see Phase 3b notes); CKD's own extreme slopes are not in this table.",
    "table7": "Same caveats as table6, at fraction 0.1.",
    "table8": "Post-hoc Platt/isotonic changes are descriptive and cross-fitted on test predictions; not a pipeline component.",
    "table9": "Clean + 3 missing levels only (no noise); ALL5/NO_CKD as separate files.",
    "table10": "Clean + 3 noise levels only (no missing); ALL5/NO_CKD as separate files.",
    "table11": "Separate files for missing vs. noise kinds; log loss only.",
    "table12": "The version of this table printed in the Phase 3c session's final chat message was garbled; only the CSV/TeX file is authoritative (CLAUDE.md decision 13).",
    "table13": "Continuous-only (_cont) vs. original Phase 3c noise (_orig) side by side; AUROC and log loss are separate files.",
    "table14": "Only the highest level (30% missing / noise 0.5); per-dataset breakdown is in noise_cont_diff_vs_logreg.csv / robustness_diff_vs_logreg.csv, not in this table.",
    "table15": "Top-5 by MEAN normalised importance across splits, not by per-split frequency.",
    "table16": "BLOCK-style per-split agreement is NOT computed; only per-dataset/averaged Jaccard and Spearman (10-split means) are reported, since the 10 splits overlap (repeated CV) and are not independent.",
    "table17": "Precision only; recall of the clinical list within the top-5 is not computed (not requested).",
}
_DATA_SOURCE = {
    "fig1": "results/all_runs.csv", "fig1b": "results/all_runs.csv", "fig1c": "results/all_runs.csv",
    "fig2": "results/phase3a/friedman_summary.csv, pairwise_wilcoxon.csv", "fig3": "results/phase3a/ci_diff_vs_logreg.csv",
    "fig4": "results/predictions/*.npz", "fig4b": "results/predictions/*.npz", "fig4c": "results/predictions/*.npz",
    "fig5": "results/phase3b/calibration_metrics.csv, calibration_bootstrap_ci.csv", "fig6": "results/phase3b/posthoc_summary.csv",
    "fig7": "results/phase3c/robustness_averaged.csv", "fig7b": "results/phase3c/robustness_averaged.csv",
    "fig7c": "results/phase3c/robustness_averaged.csv", "fig7d": "results/phase3c/robustness_averaged.csv",
    "fig7e": "results/phase3c/robustness_per_dataset.csv", "fig8": "results/phase3c/robustness_diff_vs_logreg.csv",
    "fig9": "results/phase3c/robustness_averaged.csv, results/phase3d/noise_continuous/noise_cont_averaged.csv",
    "fig10": "results/phase3d/top_features.csv", "fig11": "results/phase3d/agreement_per_dataset.csv, baselines.csv",
}


def build_inventory() -> list[dict]:
    phase_dirs = {
        "2c": PROJECT_ROOT / "results" / "phase2c",
        "3a": PROJECT_ROOT / "results" / "phase3a",
        "3b": PROJECT_ROOT / "results" / "phase3b",
        "3c": PROJECT_ROOT / "results" / "phase3c",
        "3d": PROJECT_ROOT / "results" / "phase3d",
        "3d-noise-cont": PROJECT_ROOT / "results" / "phase3d" / "noise_continuous",
    }
    captions_by_phase: dict[str, dict[str, str]] = {}
    for phase, d in phase_dirs.items():
        merged: dict[str, str] = {}
        for name in ("figure_captions.md", "table_captions.md"):
            merged.update(parse_captions_file(d / name))
        captions_by_phase[phase] = merged

    rows: list[dict] = []
    seen_ids: dict[tuple[str, str], dict] = {}
    for phase, d in phase_dirs.items():
        if not d.exists():
            continue
        for path in sorted(d.glob("*")):
            if not path.is_file():
                continue
            m = _ID_RE.match(path.stem)
            if not m or not (path.stem.lower().startswith("fig") or path.stem.lower().startswith("table")):
                continue
            item_id = m.group(1).lower()
            key = (phase, item_id)
            if key not in seen_ids:
                seen_ids[key] = {
                    "id": item_id, "phase": phase, "files": [], "what_it_shows": captions_by_phase[phase].get(item_id, ""),
                    "panel_count": _PANEL_COUNTS.get(item_id, ""), "known_issues": _KNOWN_ISSUES.get(item_id, ""),
                    "data_source": _DATA_SOURCE.get(item_id, ""), "pixel_size": "", "physical_size_in": "",
                    "rows": "", "cols": "", "fits_one_column_3.4in": "", "latex_width_estimate_in": "",
                }
            entry = seen_ids[key]
            rel_path = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            entry["files"].append(rel_path)
            if path.suffix == ".png" and not entry["pixel_size"]:
                dims = png_dimensions(path)
                if dims:
                    entry["pixel_size"] = f"{dims[0]}x{dims[1]}"
                    entry["physical_size_in"] = f"{dims[0]/300:.2f}x{dims[1]/300:.2f}"
            if path.suffix == ".csv" and not entry["rows"]:
                try:
                    df = pd.read_csv(path, comment="#")
                    entry["rows"] = len(df)
                    entry["cols"] = len(df.columns)
                except Exception:  # noqa: BLE001
                    pass
            if path.suffix == ".tex" and not entry["latex_width_estimate_in"]:
                width_in, n_cols = latex_table_width_estimate(path)
                entry["latex_width_estimate_in"] = width_in
                entry["fits_one_column_3.4in"] = width_in <= 3.4 if width_in else ""

    for entry in seen_ids.values():
        entry["files"] = "; ".join(sorted(entry["files"]))
        rows.append(entry)
    rows.sort(key=lambda r: (r["phase"], r["id"]))
    return rows


def write_inventory(rows: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "inventory.csv", index=False)

    lines = ["# Inventory of figures and tables (Phases 2c, 3a, 3b, 3c, 3d)\n"]
    lines.append(
        "LaTeX column-width estimate method: for each table COLUMN, the longest cell's "
        "character count (header included, \\textbf{} stripped) converted to inches "
        "(0.075in/char + 0.12in padding), summed over columns -- a documented heuristic, "
        "NOT a real LaTeX layout pass -- treat `fits_one_column_3.4in` as an estimate, not "
        "a guarantee.\n"
    )
    for phase in sorted({r["phase"] for r in rows}):
        phase_rows = [r for r in rows if r["phase"] == phase]
        lines.append(f"## Phase {phase} ({len(phase_rows)} items)\n")
        for r in phase_rows:
            lines.append(f"### {r['id']}")
            lines.append(f"- files: {r['files']}")
            lines.append(f"- what it shows: {r['what_it_shows'] or '(no caption found)'}")
            if r["panel_count"] != "":
                lines.append(f"- panel count: {r['panel_count']}")
            if r["pixel_size"]:
                lines.append(f"- pixel size: {r['pixel_size']} ({r['physical_size_in']} in at 300 dpi)")
            if r["rows"] != "":
                lines.append(f"- table rows x cols: {r['rows']} x {r['cols']}")
            if r["latex_width_estimate_in"] != "":
                lines.append(f"- LaTeX width estimate: {r['latex_width_estimate_in']} in (fits one IEEE column: {r['fits_one_column_3.4in']})")
            lines.append(f"- data source: {r['data_source'] or '(not curated)'}")
            lines.append(f"- known issues: {r['known_issues'] or 'none noted'}")
            lines.append("")

    lines.append("## Captions and notes files per phase\n")
    for phase, d in {
        "2c": PROJECT_ROOT / "results" / "phase2c", "3a": PROJECT_ROOT / "results" / "phase3a",
        "3b": PROJECT_ROOT / "results" / "phase3b", "3c": PROJECT_ROOT / "results" / "phase3c",
        "3d": PROJECT_ROOT / "results" / "phase3d", "3d-noise-cont": PROJECT_ROOT / "results" / "phase3d" / "noise_continuous",
    }.items():
        for name in ("figure_captions.md", "table_captions.md", "stats_notes.md", "calibration_notes.md", "robustness_notes.md", "explanation_notes.md"):
            p = d / name
            if p.exists():
                lines.append(f"- {str(p.relative_to(PROJECT_ROOT)).replace(chr(92), '/')}")

    (OUTPUT_DIR / "inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    print(f"sys.executable = {sys.executable}")
    book = NumbersBook()
    facts = build_protocol_facts(book)
    write_protocol_facts(facts)
    build_group2_primary_performance(book)
    build_group3_learning_curve(book)
    build_group4_calibration(book)
    build_group5_robustness(book)
    build_group6_explanations(book)
    build_group7_sensitivity(book)
    build_group8_phase4b_claims(book)
    build_group9_phase4d(book)
    write_numbers_outputs(book)
    n_undefined = sum(1 for e in book.entries if e["undefined"])
    print(f"numbers.json: {len(book.entries)} entries, {n_undefined} undefined")

    inventory_rows = build_inventory()
    write_inventory(inventory_rows)
    print(f"inventory.csv: {len(inventory_rows)} figure/table items")


if __name__ == "__main__":
    main()

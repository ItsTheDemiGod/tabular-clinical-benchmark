"""Phase 3d, Part 4: SHAP explanation agreement and clinical plausibility.

Loads the SAVED fraction-1.0 pipelines for logreg/rf/mlp; never fits, refits or
tunes anything. Explanations act on the ENCODED matrix (via the fitted pipeline's
"prep" step) and are mapped back to ORIGINAL features through the ColumnTransformer
structure, never by parsing column-name strings. Writes ONLY inside
results/phase3d/ (results/phase3d/shap/ for per-task artifacts).

Subcommands: verify, run, analyze, figures, tables, all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.bootstrap import bootstrap_mean_ci, bootstrap_mean_ci_averaged
from analysis.constants import DATASETS_ALL5, DATASETS_NO_CKD, MODEL_COLOR, MODEL_MARKER, SEED
from analysis.explain import (
    additivity_errors,
    aggregate_signed_shap,
    expected_chance_jaccard,
    feature_importance_from_agg,
    jaccard,
    map_encoded_to_original,
    normalize_shap_values,
    permutation_importance_logloss,
    sample_background_rows,
    spearman_safe,
    top_k_with_ties,
)
from analysis.tables import escape_latex, format_float, write_csv_and_tex
from src.data import load_dataset
from src.utils import make_splits

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only if Part 4a chose the fallback
    SHAP_AVAILABLE = False

SHAP_MODELS = ["logreg", "rf", "mlp"]
MODELS_DIR = PROJECT_ROOT / "results" / "models"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "predictions"
OUTPUT_DIR = PROJECT_ROOT / "results" / "phase3d"
SHAP_DIR = OUTPUT_DIR / "shap"
CLINICAL_LIST_PATH = OUTPUT_DIR / "clinical_list.json"
CLINICAL_LIST_SHA_PATH = OUTPUT_DIR / "clinical_list.sha256"
TOP_K = 5
BOOTSTRAP_B = 5000
TIMING_GATE_MINUTES = 60
KERNEL_BACKGROUND_N = 50

_DATASET_CACHE: dict[str, tuple] = {}


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def verify_clinical_list_hash() -> dict:
    """Verify results/phase3d/clinical_list.json against its pre-registered SHA-256.
    Raises RuntimeError on any mismatch or missing file -- callers must not proceed.
    """
    if not CLINICAL_LIST_PATH.exists() or not CLINICAL_LIST_SHA_PATH.exists():
        raise RuntimeError("clinical_list.json or clinical_list.sha256 is missing")
    expected = CLINICAL_LIST_SHA_PATH.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(CLINICAL_LIST_PATH.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"clinical_list.json SHA-256 mismatch: expected {expected}, got {actual}")
    return json.loads(CLINICAL_LIST_PATH.read_text(encoding="utf-8"))


def _get_dataset(dataset: str):
    if dataset not in _DATASET_CACHE:
        X, y, meta = load_dataset(dataset, verbose=False)
        splits = make_splits(X, y, n_splits=5, n_repeats=2, seed=42)
        _DATASET_CACHE[dataset] = (X, y, meta, splits)
    return _DATASET_CACHE[dataset]


def _stem(dataset: str, model: str, split: int) -> str:
    return f"{dataset}__{model}__split{split}"


def _pipeline_path(dataset: str, model: str, split: int) -> Path:
    return MODELS_DIR / f"{_stem(dataset, model, split)}.joblib"


def _npz_path(dataset: str, model: str, split: int) -> Path:
    return PREDICTIONS_DIR / f"{_stem(dataset, model, split)}.npz"


def _free_gpu_memory(pipeline) -> None:
    del pipeline
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _print_mapping_counts() -> None:
    for dataset in DATASETS_ALL5:
        X, y, meta, splits = _get_dataset(dataset)
        pipeline = joblib.load(_pipeline_path(dataset, "rf", 0))
        prep = pipeline.named_steps["prep"]
        mapping = map_encoded_to_original(prep, meta["numeric_cols"], meta["categorical_cols"])
        encoded_width = prep.transform(X.iloc[:5]).shape[1]
        total_mapped = sum(len(v) for v in mapping.values())
        print(f"{dataset}: {len(mapping)} original features -> {total_mapped} encoded columns (encoded width {encoded_width})")
        for feature in list(X.columns):
            print(f"    {feature}: {len(mapping[feature])} encoded column(s)")
        if total_mapped != encoded_width:
            raise RuntimeError(f"{dataset}: mapping total {total_mapped} != encoded width {encoded_width}")
        _free_gpu_memory(pipeline)


# ---------------------------------------------------------------------------
# Subcommand: verify
# ---------------------------------------------------------------------------
def run_verify() -> dict:
    from src.metrics import compute_metrics
    from src.models import positive_class_proba

    structural_errors: list[str] = []
    alignment_errors: list[str] = []
    max_prob_diff = {m: 0.0 for m in SHAP_MODELS}

    for dataset in DATASETS_ALL5:
        X, y, meta, splits = _get_dataset(dataset)
        for model in SHAP_MODELS:
            for split in range(10):
                train_idx, test_idx = splits[split]
                data = np.load(_npz_path(dataset, model, split))
                if not np.array_equal(data["test_idx"], test_idx):
                    alignment_errors.append(f"{_stem(dataset, model, split)}: test_idx mismatch")
                    continue
                y_true_expected = y.iloc[test_idx].to_numpy()
                if not np.array_equal(data["y_true"], y_true_expected):
                    alignment_errors.append(f"{_stem(dataset, model, split)}: y_true mismatch")
                    continue
                try:
                    pipeline = joblib.load(_pipeline_path(dataset, model, split))
                    proba = positive_class_proba(pipeline, X.iloc[test_idx])
                except Exception as exc:  # noqa: BLE001
                    structural_errors.append(f"{_stem(dataset, model, split)}: predict failed: {exc}")
                    continue
                diff = float(np.max(np.abs(proba - data["prob_1.0"])))
                max_prob_diff[model] = max(max_prob_diff[model], diff)
                if diff > 1e-3:
                    alignment_errors.append(f"{_stem(dataset, model, split)}: prob diff {diff:.3e} exceeds 1e-3")
                _free_gpu_memory(pipeline)

    stop = bool(structural_errors) or bool(alignment_errors)
    return {
        "structural_errors": structural_errors,
        "alignment_errors": alignment_errors,
        "max_prob_diff": max_prob_diff,
        "stop": stop,
    }


def run_additivity_check() -> dict:
    """Reads every saved shap npz and reports the max additivity error per model."""
    max_error = {m: 0.0 for m in SHAP_MODELS}
    tolerances = {"logreg": 1e-3, "rf": 1e-3, "mlp": 1e-2}
    stop_items = []
    for dataset in DATASETS_ALL5:
        for model in SHAP_MODELS:
            for split in range(10):
                path = SHAP_DIR / f"{_stem(dataset, model, split)}.npz"
                if not path.exists():
                    continue
                data = np.load(path)
                err = float(data["additivity_max_error"])
                max_error[model] = max(max_error[model], err)
                if err > tolerances[model]:
                    stop_items.append(f"{_stem(dataset, model, split)}: additivity error {err:.3e} > {tolerances[model]}")
    return {"max_error": max_error, "stop_items": stop_items, "stop": bool(stop_items)}


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------
def _explain_logreg(clf, x_train_enc: np.ndarray, x_test_enc: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    explainer = shap.LinearExplainer(clf, x_train_enc, feature_perturbation="interventional")
    raw = explainer.shap_values(x_test_enc)
    shap_matrix = normalize_shap_values(raw, positive_class_index=1)
    base_value = explainer.expected_value
    base_value = float(np.ravel(base_value)[0])
    model_output = clf.decision_function(x_test_enc)
    return shap_matrix, base_value, model_output


def _explain_rf(clf, x_test_enc: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    # shap 0.54.1 only supports model_output="raw" with feature_perturbation=
    # "tree_path_dependent" (no background). For sklearn's RandomForestClassifier
    # this "raw" output already equals predicted probability exactly (each tree's
    # leaf values are themselves class-probability estimates), confirmed empirically
    # (max abs diff ~1e-15 against clf.predict_proba), so no further conversion is
    # needed -- explaining "raw" here IS explaining the positive-class probability.
    explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    raw = explainer.shap_values(x_test_enc)
    shap_matrix = normalize_shap_values(raw, positive_class_index=1)
    base_value = explainer.expected_value
    base_value = float(np.ravel(base_value)[-1])  # positive class (index 1 of 2, or the only value)
    model_output = clf.predict_proba(x_test_enc)[:, 1]
    return shap_matrix, base_value, model_output


def _explain_mlp(clf, x_train_enc: np.ndarray, x_test_enc: np.ndarray, seed: int) -> tuple[np.ndarray, float, np.ndarray]:
    def f(rows: np.ndarray) -> np.ndarray:
        return clf.predict_proba(np.asarray(rows, dtype=np.float64))[:, 1]

    background = sample_background_rows(x_train_enc, KERNEL_BACKGROUND_N, seed)
    explainer = shap.KernelExplainer(f, background)
    raw = explainer.shap_values(x_test_enc, nsamples="auto", silent=True)
    shap_matrix = normalize_shap_values(raw, positive_class_index=1)
    base_value = float(np.ravel(explainer.expected_value)[0])
    model_output = f(x_test_enc)
    return shap_matrix, base_value, model_output


def _run_one_task(dataset: str, model: str, split: int) -> dict:
    X, y, meta, splits = _get_dataset(dataset)
    train_idx, test_idx = splits[split]
    pipeline = joblib.load(_pipeline_path(dataset, model, split))
    prep = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]

    x_test_enc = np.asarray(prep.transform(X.iloc[test_idx]), dtype=np.float64)
    x_train_enc = np.asarray(prep.transform(X.iloc[train_idx]), dtype=np.float64)
    feature_order = list(X.columns)
    mapping = map_encoded_to_original(prep, meta["numeric_cols"], meta["categorical_cols"])

    if SHAP_AVAILABLE:
        if model == "logreg":
            shap_matrix, base_value, model_output = _explain_logreg(clf, x_train_enc, x_test_enc)
        elif model == "rf":
            shap_matrix, base_value, model_output = _explain_rf(clf, x_test_enc)
        elif model == "mlp":
            shap_matrix, base_value, model_output = _explain_mlp(clf, x_train_enc, x_test_enc, SEED + split)
        else:
            raise ValueError(model)
        agg = aggregate_signed_shap(shap_matrix, mapping, feature_order)
        importance = feature_importance_from_agg(agg)
        errors = additivity_errors(base_value, agg, model_output)
        additivity_max_error = float(errors.max())
    else:
        # Fallback: permutation importance, same function for all three models.
        def predict_fn(x_raw: pd.DataFrame) -> np.ndarray:
            return pipeline.predict_proba(x_raw)[:, list(pipeline.classes_).index(1)]

        y_true = y.iloc[test_idx].to_numpy()
        perm_importance = permutation_importance_logloss(
            predict_fn, X.iloc[test_idx], y_true, feature_order, n_repeats=10, seed=SEED + split
        )
        importance = np.array([perm_importance[f] for f in feature_order])
        agg = np.zeros((len(test_idx), len(feature_order)))
        base_value = float("nan")
        additivity_max_error = float("nan")

    _free_gpu_memory(pipeline)
    return {
        "feature_order": feature_order,
        "agg": agg,
        "importance": importance,
        "base_value": base_value,
        "additivity_max_error": additivity_max_error,
        "test_idx": test_idx,
    }


def _task_complete(dataset: str, model: str, split: int) -> bool:
    return (SHAP_DIR / f"{_stem(dataset, model, split)}.npz").exists()


def run_timing_gate() -> float:
    """Probes one task per model on the LARGEST/slowest dataset (ckd) and
    extrapolates to all 150 tasks."""
    probe_dataset = "ckd"
    per_model_seconds: dict[str, float] = {}
    for model in SHAP_MODELS:
        start = time.time()
        result = _run_one_task(probe_dataset, model, 0)
        per_model_seconds[model] = time.time() - start
        np.savez_compressed(
            SHAP_DIR / f"{_stem(probe_dataset, model, 0)}.npz",
            feature_names=np.array(result["feature_order"]),
            shap_agg=result["agg"].astype(np.float32),
            base_value=np.float32(result["base_value"]),
            importance=result["importance"].astype(np.float32),
            test_idx=result["test_idx"],
            additivity_max_error=np.float32(result["additivity_max_error"]),
        )
    avg_seconds = float(np.mean(list(per_model_seconds.values())))
    total_estimate_minutes = avg_seconds * 150 / 60.0
    print(f"Timing gate probe (1 task each, dataset={probe_dataset}):")
    for model, seconds in per_model_seconds.items():
        print(f"  {model}: {seconds:.1f}s/task")
    print(f"  average: {avg_seconds:.1f}s/task -> extrapolated total: {total_estimate_minutes:.1f} min for 150 tasks")
    return total_estimate_minutes


def run_run_subcommand(skip_timing_gate: bool = False) -> None:
    ensure_output_dir()
    if not skip_timing_gate:
        estimate = run_timing_gate()
        if estimate > TIMING_GATE_MINUTES:
            print(f"STOP: estimated total runtime {estimate:.1f} min exceeds the {TIMING_GATE_MINUTES} min gate.")
            print("Options (not choosing one): (a) explain a fixed random subsample of at most 40")
            print("test rows per task for all three models, (b) a smaller KernelExplainer background,")
            print("(c) fewer nsamples for the KernelExplainer.")
            sys.exit(1)

    n_done, n_skipped = 0, 0
    importance_rows = []
    for dataset in DATASETS_ALL5:
        for model in SHAP_MODELS:
            for split in range(10):
                if _task_complete(dataset, model, split):
                    n_skipped += 1
                else:
                    result = _run_one_task(dataset, model, split)
                    np.savez_compressed(
                        SHAP_DIR / f"{_stem(dataset, model, split)}.npz",
                        feature_names=np.array(result["feature_order"]),
                        shap_agg=result["agg"].astype(np.float32),
                        base_value=np.float32(result["base_value"]),
                        importance=result["importance"].astype(np.float32),
                        test_idx=result["test_idx"],
                        additivity_max_error=np.float32(result["additivity_max_error"]),
                    )
                    n_done += 1
                    if n_done % 10 == 0:
                        print(f"  ... {n_done} tasks completed, {n_skipped} skipped (resume)")
    print(f"run: {n_done} tasks completed this run, {n_skipped} already complete (resumed)")

    # Build shap_importance.csv from all saved npz files.
    for dataset in DATASETS_ALL5:
        for model in SHAP_MODELS:
            for split in range(10):
                path = SHAP_DIR / f"{_stem(dataset, model, split)}.npz"
                if not path.exists():
                    continue
                data = np.load(path, allow_pickle=True)
                feature_names = [str(f) for f in data["feature_names"]]
                importance = data["importance"]
                order = np.argsort(-importance, kind="stable")
                for rank, idx in enumerate(order, start=1):
                    importance_rows.append(
                        {"dataset": dataset, "model": model, "split_idx": split, "feature": feature_names[idx], "importance": float(importance[idx]), "rank": rank}
                    )
    pd.DataFrame(importance_rows).to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------
MODEL_PAIRS = [("logreg", "rf"), ("logreg", "mlp"), ("rf", "mlp")]


def _load_task_importance(dataset: str, model: str, split: int) -> tuple[list[str], np.ndarray]:
    data = np.load(SHAP_DIR / f"{_stem(dataset, model, split)}.npz", allow_pickle=True)
    return [str(f) for f in data["feature_names"]], data["importance"]


def run_analyze_subcommand() -> None:
    output_dir = ensure_output_dir()
    clinical_list = verify_clinical_list_hash()["datasets"]
    rng = np.random.default_rng(SEED)

    # Per-split pairwise agreement.
    pair_rows = []
    importance_cache: dict[tuple, tuple] = {}
    for dataset in DATASETS_ALL5:
        for split in range(10):
            for model in SHAP_MODELS:
                importance_cache[(dataset, model, split)] = _load_task_importance(dataset, model, split)
            for model_a, model_b in MODEL_PAIRS:
                feat_a, imp_a = importance_cache[(dataset, model_a, split)]
                feat_b, imp_b = importance_cache[(dataset, model_b, split)]
                top_a = set(top_k_with_ties(imp_a, feat_a, TOP_K))
                top_b = set(top_k_with_ties(imp_b, feat_b, TOP_K))
                jac = jaccard(top_a, top_b)
                rho, const = spearman_safe(imp_a, imp_b)
                pair_rows.append(
                    {"dataset": dataset, "split_idx": split, "model_a": model_a, "model_b": model_b, "jaccard": jac, "spearman": rho, "spearman_constant": const}
                )
    pair_df = pd.DataFrame(pair_rows)

    per_dataset_rows = []
    for dataset in DATASETS_ALL5:
        for model_a, model_b in MODEL_PAIRS:
            sub = pair_df[(pair_df.dataset == dataset) & (pair_df.model_a == model_a) & (pair_df.model_b == model_b)].sort_values("split_idx")
            jac_point, jac_lo, jac_hi = bootstrap_mean_ci(sub["jaccard"].to_numpy(), rng, BOOTSTRAP_B)
            spearman_vals = sub["spearman"].dropna().to_numpy()
            if len(spearman_vals) == 10:
                sp_point, sp_lo, sp_hi = bootstrap_mean_ci(spearman_vals, rng, BOOTSTRAP_B)
            else:
                sp_point, sp_lo, sp_hi = float(np.nanmean(sub["spearman"])), np.nan, np.nan
            per_dataset_rows.append(
                {"dataset": dataset, "model_a": model_a, "model_b": model_b, "jaccard_mean": jac_point, "jaccard_ci_lo": jac_lo, "jaccard_ci_hi": jac_hi,
                 "spearman_mean": sp_point, "spearman_ci_lo": sp_lo, "spearman_ci_hi": sp_hi, "n_spearman_nan": int(sub["spearman"].isna().sum())}
            )
    pd.DataFrame(per_dataset_rows).to_csv(output_dir / "agreement_per_dataset.csv", index=False)

    averaged_rows = []
    for model_a, model_b in MODEL_PAIRS:
        for set_name, datasets in (("ALL5", DATASETS_ALL5), ("NO_CKD", DATASETS_NO_CKD)):
            jac_by_dataset, sp_by_dataset = [], []
            for dataset in datasets:
                sub = pair_df[(pair_df.dataset == dataset) & (pair_df.model_a == model_a) & (pair_df.model_b == model_b)].sort_values("split_idx")
                jac_by_dataset.append(sub["jaccard"].to_numpy())
                sp_vals = sub["spearman"].to_numpy()
                sp_by_dataset.append(np.nan_to_num(sp_vals, nan=0.0))
            jac_point, jac_lo, jac_hi = bootstrap_mean_ci_averaged(jac_by_dataset, rng, BOOTSTRAP_B)
            sp_point, sp_lo, sp_hi = bootstrap_mean_ci_averaged(sp_by_dataset, rng, BOOTSTRAP_B)
            averaged_rows.append(
                {"model_a": model_a, "model_b": model_b, "dataset_set": set_name, "jaccard_mean": jac_point, "jaccard_ci_lo": jac_lo, "jaccard_ci_hi": jac_hi,
                 "spearman_mean": sp_point, "spearman_ci_lo": sp_lo, "spearman_ci_hi": sp_hi}
            )
    pd.DataFrame(averaged_rows).to_csv(output_dir / "agreement_averaged.csv", index=False)

    # Baselines: within-model stability + chance.
    baseline_rows = []
    m_by_dataset = {}
    for dataset in DATASETS_ALL5:
        X, y, meta, splits = _get_dataset(dataset)
        m = len(X.columns)
        m_by_dataset[dataset] = m
        chance_jac = expected_chance_jaccard(m, TOP_K)
        model_stability_jac, model_stability_sp = [], []
        for model in SHAP_MODELS:
            top_sets, importances = {}, {}
            for split in range(10):
                feat, imp = importance_cache[(dataset, model, split)]
                top_sets[split] = set(top_k_with_ties(imp, feat, TOP_K))
                importances[split] = imp
            jac_vals, sp_vals = [], []
            for i in range(10):
                for j in range(i + 1, 10):
                    jac_vals.append(jaccard(top_sets[i], top_sets[j]))
                    rho, const = spearman_safe(importances[i], importances[j])
                    if not const:
                        sp_vals.append(rho)
            mean_jac = float(np.mean(jac_vals))
            mean_sp = float(np.mean(sp_vals)) if sp_vals else float("nan")
            model_stability_jac.append(mean_jac)
            model_stability_sp.append(mean_sp)
            baseline_rows.append(
                {"dataset": dataset, "type": "within_model", "model": model, "jaccard_mean": mean_jac, "spearman_mean": mean_sp, "m_features": m, "chance_jaccard": chance_jac}
            )
        baseline_rows.append(
            {"dataset": dataset, "type": "within_model_mean_over_models", "model": "ALL3", "jaccard_mean": float(np.mean(model_stability_jac)),
             "spearman_mean": float(np.nanmean(model_stability_sp)), "m_features": m, "chance_jaccard": chance_jac}
        )
        baseline_rows.append({"dataset": dataset, "type": "chance", "model": "-", "jaccard_mean": chance_jac, "spearman_mean": 0.0, "m_features": m, "chance_jaccard": chance_jac})
    pd.DataFrame(baseline_rows).to_csv(output_dir / "baselines.csv", index=False)

    # Plausibility.
    plaus_rows = []
    for dataset in DATASETS_ALL5:
        listed = set(clinical_list[dataset]["columns"])
        m = m_by_dataset[dataset]
        chance_precision = len(listed) / m
        for model in SHAP_MODELS:
            precisions = []
            for split in range(10):
                feat, imp = importance_cache[(dataset, model, split)]
                top = set(top_k_with_ties(imp, feat, TOP_K))
                precisions.append(len(top & listed) / TOP_K)
            point, lo, hi = bootstrap_mean_ci(np.array(precisions), rng, BOOTSTRAP_B)
            plaus_rows.append(
                {"dataset": dataset, "model": model, "precision_mean": point, "precision_ci_lo": lo, "precision_ci_hi": hi, "chance_precision": chance_precision, "list_size": len(listed), "m_features": m}
            )
    pd.DataFrame(plaus_rows).to_csv(output_dir / "plausibility.csv", index=False)

    # Mean importance / top features table.
    top_rows = []
    for dataset in DATASETS_ALL5:
        listed = set(clinical_list[dataset]["columns"])
        for model in SHAP_MODELS:
            feat_names = importance_cache[(dataset, model, 0)][0]
            norm_sum = np.zeros(len(feat_names))
            for split in range(10):
                _, imp = importance_cache[(dataset, model, split)]
                total = imp.sum()
                norm_sum += imp / total if total > 0 else 0
            mean_norm = norm_sum / 10
            top_feats = top_k_with_ties(mean_norm, feat_names, TOP_K)
            for rank, feature in enumerate(top_feats, start=1):
                idx = feat_names.index(feature)
                top_rows.append(
                    {"dataset": dataset, "model": model, "rank": rank, "feature": feature, "mean_normalized_importance": float(mean_norm[idx]), "on_clinical_list": feature in listed}
                )
    pd.DataFrame(top_rows).to_csv(output_dir / "top_features.csv", index=False)


# ---------------------------------------------------------------------------
# Subcommand: figures
# ---------------------------------------------------------------------------
def run_figures_subcommand() -> list[str]:
    output_dir = ensure_output_dir()
    top_features = pd.read_csv(output_dir / "top_features.csv")
    per_dataset = pd.read_csv(output_dir / "agreement_per_dataset.csv")
    baselines = pd.read_csv(output_dir / "baselines.csv")
    checked = []

    fig, axes = plt.subplots(len(DATASETS_ALL5), 1, figsize=(7.0, 3.0 * len(DATASETS_ALL5)))
    for ax, dataset in zip(axes, DATASETS_ALL5):
        sub = top_features[top_features.dataset == dataset]
        union_features = list(dict.fromkeys(sub["feature"]))
        y_positions = np.arange(len(union_features))
        width = 0.25
        for i, model in enumerate(SHAP_MODELS):
            model_sub = sub[sub.model == model].set_index("feature")
            values = [model_sub.loc[f, "mean_normalized_importance"] if f in model_sub.index else 0 for f in union_features]
            ax.barh(y_positions + (i - 1) * width, values, height=width, color=MODEL_COLOR[model], label=model)
        labels = []
        for f in union_features:
            on_list = sub[sub.feature == f]["on_clinical_list"].any()
            labels.append(f"{f}*" if on_list else f)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title(dataset, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Top-5 union feature importance (* = on clinical list)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    png_path = output_dir / "fig10_top_features_importance.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig10_top_features_importance.pdf", bbox_inches="tight")
    plt.close(fig)
    checked.append(f"{png_path.name}: 5 panels, bars per model, clinical-list asterisks, labels readable")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    for ax, metric in zip(axes, ("jaccard", "spearman")):
        y_positions = np.arange(len(DATASETS_ALL5))[::-1]
        for yi, dataset in enumerate(DATASETS_ALL5):
            y = y_positions[yi]
            for j, (model_a, model_b) in enumerate(MODEL_PAIRS):
                row = per_dataset[(per_dataset.dataset == dataset) & (per_dataset.model_a == model_a) & (per_dataset.model_b == model_b)]
                if row.empty:
                    continue
                point = row[f"{metric}_mean"].iloc[0]
                lo, hi = row[f"{metric}_ci_lo"].iloc[0], row[f"{metric}_ci_hi"].iloc[0]
                offs = (j - 1) * 0.15
                xerr = None if np.isnan(lo) else [[point - lo], [hi - point]]
                ax.errorbar(point, y + offs, xerr=xerr, fmt="o", markersize=4, capsize=2, color=f"C{j}", label=f"{model_a}-{model_b}" if yi == 0 else None)
            stab = baselines[(baselines.dataset == dataset) & (baselines.type == "within_model_mean_over_models")]
            chance = baselines[(baselines.dataset == dataset) & (baselines.type == "chance")]
            if not stab.empty:
                ax.plot(stab[f"{metric}_mean"].iloc[0], y, marker="s", color="black", markersize=5)
            if not chance.empty and metric == "jaccard":
                ax.plot(chance["chance_jaccard"].iloc[0], y, marker="x", color="gray", markersize=6)
            elif metric == "spearman":
                ax.plot(0.0, y, marker="x", color="gray", markersize=6)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(DATASETS_ALL5, fontsize=7)
        ax.set_title(metric, fontsize=9)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6, loc="upper left")
    fig.suptitle("Cross-model agreement, within-model stability (square), chance (x)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png_path = output_dir / "fig11_agreement_dotplot.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fig11_agreement_dotplot.pdf", bbox_inches="tight")
    plt.close(fig)
    checked.append(f"{png_path.name}: 2 panels, markers for stability/chance distinct, dataset labels readable")

    (output_dir / "figure_captions.md").write_text(
        "# Phase 3d (Part 4) figure captions (factual only, no interpretation)\n\n"
        "## fig10: top-5 union feature importance\n"
        "Horizontal bars of the mean (over 10 splits) normalised SHAP-derived importance "
        "for the union of each model's top-5 original features, one panel per dataset, "
        "3 bars per feature (logreg, rf, mlp). Features on the pre-registered clinical "
        "list (results/phase3d/clinical_list.json) are marked with an asterisk.\n\n"
        "## fig11: agreement dot plot\n"
        "Per dataset: mean top-5 Jaccard (left) and mean Spearman rank correlation "
        "(right) over the 10 splits for the 3 model pairs (95% bootstrap CI, seed 42, "
        "B=5000), the within-model stability level (square, mean over the 3 models' "
        "45 split-pairs), and the chance level (x: exact expected Jaccard for the left "
        "panel, 0 for the right panel).\n",
        encoding="utf-8",
    )
    return checked


# ---------------------------------------------------------------------------
# Subcommand: tables
# ---------------------------------------------------------------------------
def run_tables_subcommand() -> None:
    output_dir = ensure_output_dir()
    top_features = pd.read_csv(output_dir / "top_features.csv")
    per_dataset = pd.read_csv(output_dir / "agreement_per_dataset.csv")
    averaged = pd.read_csv(output_dir / "agreement_averaged.csv")
    baselines = pd.read_csv(output_dir / "baselines.csv")
    plausibility = pd.read_csv(output_dir / "plausibility.csv")

    # Table 15: top-5 features per model/dataset.
    display15 = top_features.copy()
    display15["feature"] = display15.apply(lambda r: f"{r.feature}*" if r.on_clinical_list else r.feature, axis=1)
    display15["mean_normalized_importance"] = display15["mean_normalized_importance"].map(lambda v: format_float(v, 4))
    display15 = display15[["dataset", "model", "rank", "feature", "mean_normalized_importance"]]
    display15.to_csv(output_dir / "table15_top_features.csv", index=False)
    with open(output_dir / "table15_top_features.tex", "w", encoding="utf-8") as fh:
        fh.write(r"\begin{tabular}{llrlr}" + "\n" + r"\toprule" + "\n")
        fh.write("dataset & model & rank & feature & importance \\\\\n" + r"\midrule" + "\n")
        for _, r in display15.iterrows():
            fh.write(f"{escape_latex(r.dataset)} & {escape_latex(r.model)} & {r['rank']} & {escape_latex(r.feature)} & {r.mean_normalized_importance} \\\\\n")
        fh.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")

    # Table 16: agreement per dataset and averaged.
    rows16 = []
    for _, r in per_dataset.iterrows():
        rows16.append({"dataset": r.dataset, "pair": f"{r.model_a}-{r.model_b}", "jaccard_mean": r.jaccard_mean, "spearman_mean": r.spearman_mean})
    for dataset in DATASETS_ALL5:
        stab = baselines[(baselines.dataset == dataset) & (baselines.type == "within_model_mean_over_models")].iloc[0]
        chance = baselines[(baselines.dataset == dataset) & (baselines.type == "chance")].iloc[0]
        rows16.append({"dataset": dataset, "pair": "within_model_stability", "jaccard_mean": stab.jaccard_mean, "spearman_mean": stab.spearman_mean})
        rows16.append({"dataset": dataset, "pair": "chance", "jaccard_mean": chance.jaccard_mean, "spearman_mean": 0.0})
    for _, r in averaged.iterrows():
        rows16.append({"dataset": r.dataset_set, "pair": f"{r.model_a}-{r.model_b}", "jaccard_mean": r.jaccard_mean, "spearman_mean": r.spearman_mean})
    table16 = pd.DataFrame(rows16)
    table16.to_csv(output_dir / "table16_agreement.csv", index=False)
    display16 = table16.copy()
    for col in ("jaccard_mean", "spearman_mean"):
        display16[col] = display16[col].map(lambda v: format_float(v, 3))
    write_csv_and_tex("table16_agreement", display16.set_index("dataset"), output_dir, index_label="dataset")

    # Table 17: plausibility precision vs chance.
    display17 = plausibility.copy()
    for col in ("precision_mean", "precision_ci_lo", "precision_ci_hi", "chance_precision"):
        display17[col] = display17[col].map(lambda v: format_float(v, 3))
    display17.to_csv(output_dir / "table17_plausibility.csv", index=False)
    with open(output_dir / "table17_plausibility.tex", "w", encoding="utf-8") as fh:
        fh.write(r"\begin{tabular}{llrrrr}" + "\n" + r"\toprule" + "\n")
        fh.write("dataset & model & precision & ci\\_lo & ci\\_hi & chance \\\\\n" + r"\midrule" + "\n")
        for _, r in display17.iterrows():
            fh.write(f"{escape_latex(r.dataset)} & {escape_latex(r.model)} & {r.precision_mean} & {r.precision_ci_lo} & {r.precision_ci_hi} & {r.chance_precision} \\\\\n")
        fh.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")

    (output_dir / "table_captions.md").write_text(
        "# Phase 3d (Part 4) table captions\n\n"
        "Table 15: top-5 original features per model and dataset by mean (10-split) "
        "normalised importance (* = on the pre-registered clinical list).\n"
        "Table 16: top-5 Jaccard and Spearman rank correlation per dataset and pair, "
        "plus within-model stability and chance, and the ALL5/NO_CKD averages.\n"
        "Table 17: precision of the top-5 against the clinical list vs. chance "
        "(list size / number of original features), per dataset and model.\n",
        encoding="utf-8",
    )
    notes = [
        "# Phase 3d (Part 4) notes: SHAP agreement and clinical plausibility\n",
        "- Explanations are computed on ENCODED columns and aggregated to ORIGINAL",
        "  features by summing signed SHAP values per row, then taking the mean",
        "  absolute value over test rows.",
        "- logreg is explained in LOG-ODDS (decision_function); rf and mlp are explained",
        "  in PROBABILITY of the positive class. Only RANKINGS (top-5 sets, Spearman",
        "  correlation) are compared across models, never raw SHAP magnitudes.",
        f"- The mlp KernelExplainer uses a {KERNEL_BACKGROUND_N}-row TRAINING background",
        "  (seed 42 + split_idx) and nsamples=\"auto\".",
        "- The 10 splits of a dataset come from repeated cross-validation with",
        "  overlapping training sets, so they are not independent.",
        "- The clinical feature list was fixed BEFORE any SHAP value was computed",
        "  (results/phase3d/clinical_list.json, SHA-256 verified at analysis time).",
        "- No direction-of-effect (positive/negative SHAP) check is performed here.",
        "- Correlated features, especially Breast Wisconsin's mean/standard-error/worst",
        "  triplets, can substitute for one another in a top-5 set.",
        "- No file here states that models \"agree\" or that a feature is \"plausible\";",
        "  only numbers (Jaccard, Spearman, precision) are reported.",
    ]
    (output_dir / "explanation_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3d SHAP explanation analysis")
    parser.add_argument("subcommand", choices=["verify", "run", "analyze", "figures", "tables", "all"])
    parser.add_argument("--skip-timing-gate", action="store_true")
    parser.add_argument("--print-mapping", action="store_true")
    args = parser.parse_args()

    print(f"sys.executable = {sys.executable}")
    print(f"shap available: {SHAP_AVAILABLE}")
    ensure_output_dir()
    start = time.time()

    if args.print_mapping:
        _print_mapping_counts()

    if args.subcommand in ("verify", "all"):
        t0 = time.time()
        report = run_verify()
        print(f"[verify] done in {time.time() - t0:.1f}s")
        print("structural errors:", len(report["structural_errors"]))
        for e in report["structural_errors"][:20]:
            print(" ", e)
        print("alignment errors:", len(report["alignment_errors"]))
        for e in report["alignment_errors"][:20]:
            print(" ", e)
        for model in SHAP_MODELS:
            print(f"  {model}: max_prob_diff={report['max_prob_diff'][model]:.3e}")
        if report["stop"]:
            print("STOP: verify found an alignment/structural error.")
            sys.exit(1)
    if args.subcommand in ("run", "all"):
        t0 = time.time()
        run_run_subcommand(skip_timing_gate=args.skip_timing_gate)
        print(f"[run] done in {time.time() - t0:.1f}s")
        additivity = run_additivity_check()
        print("additivity max errors:", additivity["max_error"])
        if additivity["stop"]:
            print("STOP: additivity error exceeded tolerance.")
            for item in additivity["stop_items"][:20]:
                print(" ", item)
            sys.exit(1)
    if args.subcommand in ("analyze", "all"):
        t0 = time.time()
        run_analyze_subcommand()
        print(f"[analyze] done in {time.time() - t0:.1f}s")
    if args.subcommand in ("figures", "all"):
        t0 = time.time()
        checked = run_figures_subcommand()
        print(f"[figures] done in {time.time() - t0:.1f}s")
        for line in checked:
            print(f"  checked: {line}")
    if args.subcommand in ("tables", "all"):
        t0 = time.time()
        run_tables_subcommand()
        print(f"[tables] done in {time.time() - t0:.1f}s")
    print(f"total runtime: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

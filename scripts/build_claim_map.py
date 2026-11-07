"""Phase 4b: builds the claim map (paper/evidence/claim_map.md and .csv).

Mechanically decides, BEFORE any paper prose is written, which candidate
statements (C1-C11) the evidence supports, using the fixed status rules below
(applied literally, never tuned after seeing results):

  - SUPPORTED: the 95% bootstrap CI of the dataset-averaged paired difference
    excludes 0 in BOTH ALL5 and NO_CKD AND the sign of the per-dataset mean
    differences agrees with it in at least 3 of 5 datasets (ALL5) and at
    least 3 of 4 (NO_CKD).
  - PARTLY: the CI excludes 0 in only one of the two dataset sets, or the
    sign-agreement condition fails in one of them.
  - NOT SUPPORTED: the CI includes 0 in both.
  - DESCRIPTIVE: for statements about a single table (rankings, ranges,
    counts) -- no CI-exclusion test applies.

Every number used here is read from paper/evidence/numbers.json (built by
scripts/build_numbers.py, which must be run first) or, for the two per-cell
exception lists in C1/C2, computed directly and reproducibly from
results/all_runs.csv. Read-only except for writing claim_map.csv/.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from analysis.constants import DATASETS_ALL5, DATASETS_NO_CKD, MODELS

OUTPUT_DIR = PROJECT_ROOT / "paper" / "evidence"
NUMBERS_JSON_PATH = OUTPUT_DIR / "numbers.json"

NON_LOGREG_MODELS = [m for m in MODELS if m != "logreg"]


def load_numbers() -> dict[str, dict]:
    entries = json.loads(NUMBERS_JSON_PATH.read_text(encoding="utf-8"))
    return {e["key"]: e for e in entries}


def nb_status(ci_excludes_0_all5: bool, ci_excludes_0_nockd: bool, sign_ok_all5: bool, sign_ok_nockd: bool) -> str:
    """Mechanical status rule C1-C3/C8 depend on: CI-excludes-0 AND sign agreement, per dataset set."""
    supported_all5 = ci_excludes_0_all5 and sign_ok_all5
    supported_nockd = ci_excludes_0_nockd and sign_ok_nockd
    if supported_all5 and supported_nockd:
        return "SUPPORTED"
    if supported_all5 or supported_nockd:
        return "PARTLY"
    return "NOT SUPPORTED"


def auroc_exceeds_logreg_cells(all_runs: pd.DataFrame, fraction: float) -> list[dict]:
    """[{dataset, model, model_mean, logreg_mean, excess}, ...] for every cell at this
    fraction where model's per-dataset mean AUROC (over its 10 splits) exceeds
    logreg's per-dataset mean AUROC (over its 10 splits)."""
    rows = []
    for ds in DATASETS_ALL5:
        logreg_mean = all_runs[(all_runs.dataset == ds) & (all_runs.model == "logreg") & (all_runs.fraction == fraction)]["auroc"].mean()
        for m in NON_LOGREG_MODELS:
            model_mean = all_runs[(all_runs.dataset == ds) & (all_runs.model == m) & (all_runs.fraction == fraction)]["auroc"].mean()
            if model_mean > logreg_mean:
                rows.append({"dataset": ds, "model": m, "fraction": fraction,
                             "model_mean_auroc": round(float(model_mean), 4),
                             "logreg_mean_auroc": round(float(logreg_mean), 4),
                             "excess": round(float(model_mean - logreg_mean), 4)})
    return rows


def g(nb: dict, key: str) -> float:
    e = nb[key]
    if e["undefined"]:
        raise RuntimeError(f"key {key} is undefined in numbers.json")
    return e["value"]


def diff_ci_lines(nb: dict, metric_tag: str, model: str, frac_tag: str) -> dict:
    """point/ci_lo/ci_hi/n_positive/n_total for ALL5 and NO_CKD, from existing
    auroc-diff-logreg-*/logloss-diff-logreg-* keys (Phase 3a, build_group2)."""
    out = {}
    for set_name, tag in (("ALL5", "all5"), ("NO_CKD", "nockd")):
        base = f"{metric_tag}-diff-logreg-{model}-{tag}-{frac_tag}"
        out[set_name] = {
            "point": g(nb, f"{base}-point"),
            "ci_lo": g(nb, f"{base}-ci-lo"),
            "ci_hi": g(nb, f"{base}-ci-hi"),
            "n_positive": int(g(nb, f"{base}-npos")),
            "n_total": 5 if set_name == "ALL5" else 4,
        }
    return out


def sign_ok(n_positive: int, n_total: int, point: float) -> bool:
    """Per-dataset sign agreement with the overall point estimate's sign, at
    least 3/5 (ALL5) or 3/4 (NO_CKD). n_positive counts datasets where
    (model - logreg) > 0; if the averaged point is negative, agreement means
    the MAJORITY of datasets are also negative, i.e. n_total - n_positive is
    the agreeing count."""
    threshold = 3
    n_agree = n_positive if point > 0 else (n_total - n_positive)
    return n_agree >= threshold


def build_c1_c2(nb: dict, all_runs: pd.DataFrame, frac_tag: str, fraction: float, claim_id: str) -> dict:
    rows = []
    statuses = []
    for model in NON_LOGREG_MODELS:
        d = diff_ci_lines(nb, "auroc", model, frac_tag)
        all5 = d["ALL5"]
        nockd = d["NO_CKD"]
        ci_ex_all5 = not (all5["ci_lo"] <= 0 <= all5["ci_hi"])
        ci_ex_nockd = not (nockd["ci_lo"] <= 0 <= nockd["ci_hi"])
        sign_all5 = sign_ok(all5["n_positive"], all5["n_total"], all5["point"])
        sign_nockd = sign_ok(nockd["n_positive"], nockd["n_total"], nockd["point"])
        status = nb_status(ci_ex_all5, ci_ex_nockd, sign_all5, sign_nockd)
        statuses.append(status)
        all5_agree = all5["n_positive"] if all5["point"] > 0 else (all5["n_total"] - all5["n_positive"])
        nockd_agree = nockd["n_positive"] if nockd["point"] > 0 else (nockd["n_total"] - nockd["n_positive"])
        rows.append({
            "model": model,
            "all5_point": round(all5["point"], 4), "all5_ci_lo": round(all5["ci_lo"], 4), "all5_ci_hi": round(all5["ci_hi"], 4),
            "all5_nagree": all5_agree, "all5_ntotal": all5["n_total"],
            "nockd_point": round(nockd["point"], 4), "nockd_ci_lo": round(nockd["ci_lo"], 4), "nockd_ci_hi": round(nockd["ci_hi"], 4),
            "nockd_nagree": nockd_agree, "nockd_ntotal": nockd["n_total"],
            "status": status,
        })
    exceptions = auroc_exceeds_logreg_cells(all_runs, fraction)
    n_supported = sum(1 for s in statuses if s == "SUPPORTED")
    return {"claim_id": claim_id, "per_model": rows, "exceptions": exceptions,
            "n_models_supported": n_supported, "n_models_total": len(statuses)}


def build_c3(nb: dict) -> dict:
    rows = []
    for model in NON_LOGREG_MODELS:
        d = {}
        for set_name, tag in (("ALL5", "all5"), ("NO_CKD", "nockd")):
            base = f"auroc-dod-f0.1-f1.0-{model}-{tag}"
            point = g(nb, f"{base}-point")
            ci_lo = g(nb, f"{base}-ci-lo")
            ci_hi = g(nb, f"{base}-ci-hi")
            nagree = int(g(nb, f"{base}-nsign-agree"))
            ntotal = int(g(nb, f"{base}-ntotal"))
            ci_ex = not (ci_lo <= 0 <= ci_hi)
            sign = nagree >= 3
            d[set_name] = {"point": point, "ci_lo": ci_lo, "ci_hi": ci_hi, "nagree": nagree, "ntotal": ntotal, "ci_ex": ci_ex, "sign_ok": sign}
        status = nb_status(d["ALL5"]["ci_ex"], d["NO_CKD"]["ci_ex"], d["ALL5"]["sign_ok"], d["NO_CKD"]["sign_ok"])
        rows.append({
            "model": model,
            "all5_point": round(d["ALL5"]["point"], 4), "all5_ci_lo": round(d["ALL5"]["ci_lo"], 4), "all5_ci_hi": round(d["ALL5"]["ci_hi"], 4),
            "nockd_point": round(d["NO_CKD"]["point"], 4), "nockd_ci_lo": round(d["NO_CKD"]["ci_lo"], 4), "nockd_ci_hi": round(d["NO_CKD"]["ci_hi"], 4),
            "status": status,
            "note": "expected negative (gap larger/more negative at f0.1)" if model != "rf" else "expected ~flat (claimed exception)",
        })
    return {"claim_id": "C3", "per_model": rows}


def build_c4(nb: dict) -> dict:
    return {
        "claim_id": "C4",
        "n_total_cells": int(g(nb, "nb-n-total-cells")),
        "n_sig_cells": int(g(nb, "nb-n-sig-cells")),
        "significant_cell": {
            "dataset": "ckd", "model": "xgb", "metric": "auroc", "fraction": 0.1,
            "mean_diff_vs_logreg": round(g(nb, "nb-sig-0-ckd-xgb-auroc-f0.1-meandiff"), 4),
            "t_statistic": round(g(nb, "nb-sig-0-ckd-xgb-auroc-f0.1-t"), 4),
            "p_holm": round(g(nb, "nb-sig-0-ckd-xgb-auroc-f0.1-pholm"), 4),
        },
    }


def build_c5(nb: dict, calib_metrics: pd.DataFrame, calib_ci: pd.DataFrame) -> dict:
    rows = []
    for frac, frac_tag in ((1.0, "f1.0"), (0.1, "f0.1")):
        for model in MODELS:
            slope_key = f"slope-{model}-nockd-{frac_tag}"
            nockd_slope = g(nb, slope_key) if slope_key in nb else None
            per_ds = []
            for ds in DATASETS_ALL5:
                mrow = calib_metrics[(calib_metrics.dataset == ds) & (calib_metrics.model == model) & (calib_metrics.fraction == frac)]
                cirow = calib_ci[(calib_ci.dataset == ds) & (calib_ci.model == model) & (calib_ci.fraction == frac)]
                if len(mrow) == 1 and len(cirow) == 1:
                    per_ds.append({"dataset": ds, "slope": round(float(mrow.iloc[0]["slope"]), 4),
                                    "ci_lo": round(float(cirow.iloc[0]["slope_lo"]), 4), "ci_hi": round(float(cirow.iloc[0]["slope_hi"]), 4)})
            rows.append({"fraction": frac, "model": model, "nockd_pooled_slope": round(nockd_slope, 4) if nockd_slope is not None else None, "per_dataset": per_ds})
    return {"claim_id": "C5", "rows": rows}


def build_c6(nb: dict) -> dict:
    leaders = [
        {"fraction": 1.0, "dataset_set": "ALL5", "leader": "mlp", "second": "xgb",
         "leader_mean": None, "second_mean": None, "existing_pair": False},
        {"fraction": 1.0, "dataset_set": "NO_CKD", "leader": "logreg", "second": "mlp",
         "leader_mean": None, "second_mean": None, "existing_pair": True},
        {"fraction": 0.1, "dataset_set": "ALL5", "leader": "rf", "second": "xgb",
         "leader_mean": None, "second_mean": None, "existing_pair": False},
        {"fraction": 0.1, "dataset_set": "NO_CKD", "leader": "rf", "second": "logreg",
         "leader_mean": None, "second_mean": None, "existing_pair": True},
    ]
    for entry in leaders:
        frac_tag = "f1.0" if entry["fraction"] == 1.0 else "f0.1"
        set_tag = "all5" if entry["dataset_set"] == "ALL5" else "nockd"
        entry["leader_mean"] = round(g(nb, f"logloss-mean-{entry['leader']}-{set_tag}-{frac_tag}"), 4) if f"logloss-mean-{entry['leader']}-{set_tag}-{frac_tag}" in nb else None
        entry["second_mean"] = round(g(nb, f"logloss-mean-{entry['second']}-{set_tag}-{frac_tag}"), 4) if f"logloss-mean-{entry['second']}-{set_tag}-{frac_tag}" in nb else None
        if entry["existing_pair"]:
            # leader or second is logreg: reuse logloss-diff-logreg-* (model - logreg), sign-adjusted to (leader - second).
            other = entry["second"] if entry["leader"] == "logreg" else entry["leader"]
            base = f"logloss-diff-logreg-{other}-{set_tag}-{frac_tag}"
            point = g(nb, f"{base}-point"); ci_lo = g(nb, f"{base}-ci-lo"); ci_hi = g(nb, f"{base}-ci-hi")
            if entry["leader"] == "logreg":  # leader - second = logreg - other = -(other - logreg)
                point, ci_lo, ci_hi = -point, -ci_hi, -ci_lo
            entry["diff_leader_minus_second"] = {"point": round(point, 4), "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4)}
        else:
            base = f"logloss-diff-leader-2nd-{entry['leader']}-{entry['second']}-{set_tag}-{frac_tag}"
            entry["diff_leader_minus_second"] = {
                "point": round(g(nb, f"{base}-point"), 4), "ci_lo": round(g(nb, f"{base}-ci-lo"), 4), "ci_hi": round(g(nb, f"{base}-ci-hi"), 4),
            }
    return {"claim_id": "C6", "leaders": leaders}


def build_c7(nb: dict) -> dict:
    rows = []
    for kind_tag, label in (("miss30", "30% missing"), ("noise50-orig", "noise 0.5 (original)"), ("noise50-cont", "noise 0.5 (continuous-only)")):
        for set_name, tag in (("ALL5", "all5"), ("NO_CKD", "nockd")):
            degs = {}
            for model in MODELS:
                key = f"auroc-{kind_tag}-degradation-{model}-{tag}"
                degs[model] = g(nb, key)
            min_model = min(degs, key=degs.get)
            max_model = max(degs, key=degs.get)
            rows.append({
                "perturbation": label, "dataset_set": set_name,
                "degradations": {m: round(v, 4) for m, v in degs.items()},
                "smallest_drop_model": min_model, "smallest_drop": round(degs[min_model], 4),
                "largest_drop_model": max_model, "largest_drop": round(degs[max_model], 4),
                "range": round(degs[max_model] - degs[min_model], 4),
            })
    return {"claim_id": "C7", "rows": rows}


def build_c8(nb: dict) -> dict:
    rows = []
    for kind_tag, metric_label in (("miss30", "missing 30%"), ("noise50-orig", "noise 0.5 (original)"), ("noise50-cont", "noise 0.5 (continuous-only)")):
        for metric_tag, metric_name in (("auroc", "auroc"), ("logloss", "log_loss")):
            for model in NON_LOGREG_MODELS:
                for set_name, tag in (("ALL5", "all5"), ("NO_CKD", "nockd")):
                    base = f"{metric_tag}-diff-logreg-degradation-{kind_tag}-{model}-{tag}"
                    if f"{base}-point" not in nb:
                        continue
                    point = g(nb, f"{base}-point"); ci_lo = g(nb, f"{base}-ci-lo"); ci_hi = g(nb, f"{base}-ci-hi")
                    ci_ex = not (ci_lo <= 0 <= ci_hi)
                    rows.append({
                        "perturbation": kind_tag, "metric": metric_name, "model": model, "dataset_set": set_name,
                        "point": round(point, 4), "ci_lo": round(ci_lo, 4), "ci_hi": round(ci_hi, 4),
                        "ci_excludes_0": ci_ex,
                    })
    # Status per (perturbation, metric, model): SUPPORTED/PARTLY/NOT SUPPORTED from CI-excludes-0 in both sets
    # (sign agreement not separately available here beyond the CI test; CI-based rule applied to both sets).
    by_cell: dict[tuple, dict] = {}
    for r in rows:
        key = (r["perturbation"], r["metric"], r["model"])
        by_cell.setdefault(key, {})[r["dataset_set"]] = r["ci_excludes_0"]
    statuses = {}
    for key, d in by_cell.items():
        a, n = d.get("ALL5", False), d.get("NO_CKD", False)
        statuses[key] = "SUPPORTED" if (a and n) else ("PARTLY" if (a or n) else "NOT SUPPORTED")
    return {"claim_id": "C8", "rows": rows, "statuses": {"|".join(k): v for k, v in statuses.items()}}


def build_c9(nb: dict, agreement: pd.DataFrame, baselines: pd.DataFrame) -> dict:
    rows = []
    below_stability = []
    near_chance = []
    for ds in DATASETS_ALL5:
        within_row = baselines[(baselines.dataset == ds) & (baselines.type == "within_model_mean_over_models")]
        within_stab = float(within_row.iloc[0]["jaccard_mean"])
        chance_row = baselines[(baselines.dataset == ds) & (baselines.type == "chance")]
        chance = float(chance_row.iloc[0]["jaccard_mean"])
        for _, prow in agreement[agreement.dataset == ds].iterrows():
            pair = f"{prow['model_a']}-{prow['model_b']}"
            jacc = float(prow["jaccard_mean"])
            rows.append({"dataset": ds, "pair": pair, "jaccard_mean": round(jacc, 4),
                         "within_model_stability": round(within_stab, 4), "chance": round(chance, 4)})
            if jacc < within_stab:
                below_stability.append({"dataset": ds, "pair": pair, "jaccard_mean": round(jacc, 4), "within_model_stability": round(within_stab, 4)})
            if jacc <= chance + 0.05:
                near_chance.append({"dataset": ds, "pair": pair, "jaccard_mean": round(jacc, 4), "chance": round(chance, 4), "diff_from_chance": round(jacc - chance, 4)})
    n_distinct_bw = {m: int(g(nb, f"n-distinct-top5-breast-wisconsin-{m}")) for m in ("logreg", "rf", "mlp")}
    return {"claim_id": "C9", "rows": rows, "below_within_model_stability": below_stability,
            "within_0.05_of_chance": near_chance, "breast_wisconsin_distinct_top5_sets_of_10": n_distinct_bw}


def build_c10(nb: dict, plausibility: pd.DataFrame) -> dict:
    rows = []
    flagged = []
    for _, row in plausibility.iterrows():
        chance = float(row["chance_precision"])
        entry = {"dataset": row["dataset"], "model": row["model"], "precision": round(float(row["precision_mean"]), 4),
                 "ci_lo": round(float(row["precision_ci_lo"]), 4), "ci_hi": round(float(row["precision_ci_hi"]), 4),
                 "chance": round(chance, 4)}
        rows.append(entry)
        if chance >= 0.60:
            flagged.append({"dataset": row["dataset"], "chance_precision": round(chance, 4)})
    flagged_datasets = sorted({f["dataset"] for f in flagged})
    return {"claim_id": "C10", "rows": rows, "non_informative_datasets_chance_ge_0.60": flagged_datasets}


def build_c11(nb: dict) -> dict:
    rows = []
    for model in NON_LOGREG_MODELS:
        ratio = g(nb, f"fit-time-ratio-logreg-{model}")
        fit_time = g(nb, f"fit-time-{model}")
        rows.append({"model": model, "fit_time_s": round(fit_time, 4), "ratio_to_logreg": round(ratio, 3)})
    rows.sort(key=lambda r: r["ratio_to_logreg"])
    return {"claim_id": "C11", "rows": rows, "logreg_fit_time_s": round(g(nb, "fit-time-logreg"), 4),
            "caveat": "fit_time_s measured with 3 to 4 parallel workers; rough, not a controlled single-process timing"}


MUST_NOT_MAKE = [
    ("Deep learning never helps on this benchmark.",
     "The MLP ties or beats logreg on several metrics (lowest dataset-averaged Brier/log loss at fraction 1.0); "
     "only 1/120 Nadeau-Bengio cells is significant either way, so 'never helps' overstates both directions."),
    ("Any model is 'significantly' better than another, citing the Friedman test or the BLOCK-unit Wilcoxon test.",
     "The BLOCK unit treats the 10 overlapping-training-set splits as independent, which is anti-conservative "
     "(CLAUDE.md decision 6); these tests are descriptive only."),
    ("Any model is 'significantly' better than another, citing the DATASET-unit Wilcoxon test.",
     "With 5 (or 4) datasets the DATASET-unit test has essentially no power (minimum possible two-sided p is "
     "0.0625 with 5 datasets, 0.125 with 4); 0/252 DATASET-unit pairs reached Holm significance."),
    ("TabNet is generally poor as a model family.",
     "Only under THIS training schedule: the Phase 2c sensitivity check (3 splits, fractions 0.1/1.0 only) showed "
     "TabNet's ckd collapse resolves under a longer schedule, so the result is schedule-specific, not a verdict on TabNet in general."),
    ("Any model is 'best calibrated', based on ECE alone.",
     "ECE on 60-154-row test folds is noisy and biased; report log loss, Brier and the calibration slope alongside it, with CIs."),
    ("The SHAP explanations are 'correct' or identify the 'true' important features.",
     "No direction-of-effect check was performed; features are often correlated (e.g. Breast Wisconsin's mean/SE/worst "
     "triplets); the MLP's KernelExplainer uses only a 50-row training background."),
    ("Any claim about CKD's AUROC DIFFERENCES between models.",
     "CKD is saturated (mean AUROC 0.9999 for logreg/rf); it separates models poorly on AUROC. Use log loss/Brier/ECE for CKD, "
     "and always report ALL5 and NO_CKD separately."),
    ("Any claim based on the Phase 2a/2b per-split mean ECE (the old Table 3).",
     "That figure is inflated by the small test folds (60-154 rows); CLAUDE.md decision 11 says to use the Phase 3b "
     "pooled ECE values instead."),
    ("Any claim that post-hoc recalibration (Platt/isotonic) does or does not help in general.",
     "The Phase 3b post-hoc check is exploratory and cross-fitted on the test predictions themselves (not a pipeline "
     "component); isotonic has a clipping artefact that inflates log loss on misses (CLAUDE.md decision 10)."),
]


def main() -> None:
    nb = load_numbers()
    all_runs = pd.read_csv(PROJECT_ROOT / "results" / "all_runs.csv")
    calib_metrics = pd.read_csv(PROJECT_ROOT / "results" / "phase3b" / "calibration_metrics.csv")
    calib_ci = pd.read_csv(PROJECT_ROOT / "results" / "phase3b" / "calibration_bootstrap_ci.csv")
    agreement = pd.read_csv(PROJECT_ROOT / "results" / "phase3d" / "agreement_per_dataset.csv")
    baselines = pd.read_csv(PROJECT_ROOT / "results" / "phase3d" / "baselines.csv")
    plausibility = pd.read_csv(PROJECT_ROOT / "results" / "phase3d" / "plausibility.csv")

    claims = {
        "C1": build_c1_c2(nb, all_runs, "f1.0", 1.0, "C1"),
        "C2": build_c1_c2(nb, all_runs, "f0.1", 0.1, "C2"),
        "C3": build_c3(nb),
        "C4": build_c4(nb),
        "C5": build_c5(nb, calib_metrics, calib_ci),
        "C6": build_c6(nb),
        "C7": build_c7(nb),
        "C8": build_c8(nb),
        "C9": build_c9(nb, agreement, baselines),
        "C10": build_c10(nb, plausibility),
        "C11": build_c11(nb),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "claim_map_detail.json").write_text(json.dumps(claims, indent=2), encoding="utf-8")

    # --- claim_map.csv: one summary row per claim (or per claim/model for C1-C3) ---
    csv_rows = []
    for model_row in claims["C1"]["per_model"]:
        csv_rows.append({"claim_id": "C1", "item": model_row["model"], "status": model_row["status"],
                          "all5_point": model_row["all5_point"], "all5_ci_lo": model_row["all5_ci_lo"], "all5_ci_hi": model_row["all5_ci_hi"],
                          "nockd_point": model_row["nockd_point"], "nockd_ci_lo": model_row["nockd_ci_lo"], "nockd_ci_hi": model_row["nockd_ci_hi"]})
    for model_row in claims["C2"]["per_model"]:
        csv_rows.append({"claim_id": "C2", "item": model_row["model"], "status": model_row["status"],
                          "all5_point": model_row["all5_point"], "all5_ci_lo": model_row["all5_ci_lo"], "all5_ci_hi": model_row["all5_ci_hi"],
                          "nockd_point": model_row["nockd_point"], "nockd_ci_lo": model_row["nockd_ci_lo"], "nockd_ci_hi": model_row["nockd_ci_hi"]})
    for model_row in claims["C3"]["per_model"]:
        csv_rows.append({"claim_id": "C3", "item": model_row["model"], "status": model_row["status"],
                          "all5_point": model_row["all5_point"], "all5_ci_lo": model_row["all5_ci_lo"], "all5_ci_hi": model_row["all5_ci_hi"],
                          "nockd_point": model_row["nockd_point"], "nockd_ci_lo": model_row["nockd_ci_lo"], "nockd_ci_hi": model_row["nockd_ci_hi"]})
    csv_rows.append({"claim_id": "C4", "item": "nb-corrected-t-test", "status": "DESCRIPTIVE",
                      "all5_point": claims["C4"]["n_sig_cells"], "all5_ci_lo": "", "all5_ci_hi": "", "nockd_point": claims["C4"]["n_total_cells"], "nockd_ci_lo": "", "nockd_ci_hi": ""})
    csv_rows.append({"claim_id": "C5", "item": "calibration-slopes", "status": "DESCRIPTIVE", "all5_point": "", "all5_ci_lo": "", "all5_ci_hi": "", "nockd_point": "", "nockd_ci_lo": "", "nockd_ci_hi": ""})
    for entry in claims["C6"]["leaders"]:
        d = entry["diff_leader_minus_second"]
        csv_rows.append({"claim_id": "C6", "item": f"f{entry['fraction']}-{entry['dataset_set']}-{entry['leader']}-vs-{entry['second']}",
                          "status": "DESCRIPTIVE", "all5_point": d["point"], "all5_ci_lo": d["ci_lo"], "all5_ci_hi": d["ci_hi"],
                          "nockd_point": "", "nockd_ci_lo": "", "nockd_ci_hi": ""})
    for row in claims["C7"]["rows"]:
        csv_rows.append({"claim_id": "C7", "item": f"{row['perturbation']}-{row['dataset_set']}", "status": "DESCRIPTIVE",
                          "all5_point": row["range"], "all5_ci_lo": row["smallest_drop"], "all5_ci_hi": row["largest_drop"],
                          "nockd_point": "", "nockd_ci_lo": "", "nockd_ci_hi": ""})
    for key, status in claims["C8"]["statuses"].items():
        csv_rows.append({"claim_id": "C8", "item": key, "status": status, "all5_point": "", "all5_ci_lo": "", "all5_ci_hi": "", "nockd_point": "", "nockd_ci_lo": "", "nockd_ci_hi": ""})
    csv_rows.append({"claim_id": "C9", "item": "agreement-vs-stability-and-chance", "status": "DESCRIPTIVE", "all5_point": len(claims["C9"]["below_within_model_stability"]), "all5_ci_lo": "", "all5_ci_hi": "", "nockd_point": len(claims["C9"]["within_0.05_of_chance"]), "nockd_ci_lo": "", "nockd_ci_hi": ""})
    csv_rows.append({"claim_id": "C10", "item": "plausibility-vs-chance", "status": "DESCRIPTIVE", "all5_point": ";".join(claims["C10"]["non_informative_datasets_chance_ge_0.60"]) or "none", "all5_ci_lo": "", "all5_ci_hi": "", "nockd_point": "", "nockd_ci_lo": "", "nockd_ci_hi": ""})
    for row in claims["C11"]["rows"]:
        csv_rows.append({"claim_id": "C11", "item": row["model"], "status": "DESCRIPTIVE", "all5_point": row["ratio_to_logreg"], "all5_ci_lo": "", "all5_ci_hi": "", "nockd_point": "", "nockd_ci_lo": "", "nockd_ci_hi": ""})

    pd.DataFrame(csv_rows).to_csv(OUTPUT_DIR / "claim_map.csv", index=False)

    # --- claim_map.md ---
    lines = ["# Claim map (Phase 4b)", "",
             "Mechanical SUPPORTED/PARTLY/NOT SUPPORTED/DESCRIPTIVE status, fixed rules "
             "(see scripts/build_claim_map.py docstring), applied literally. No paper wording here.", ""]

    lines.append("## C1. At fraction 1.0, no model's dataset-averaged AUROC exceeds logreg's")
    lines.append("")
    lines.append("sign column = number of datasets whose per-dataset mean diff agrees in sign with the dataset-averaged point estimate (threshold: >=3/5 ALL5, >=3/4 NO_CKD).")
    lines.append("")
    lines.append("| model | ALL5 diff | ALL5 CI | ALL5 sign agree | NO_CKD diff | NO_CKD CI | NO_CKD sign agree | status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in claims["C1"]["per_model"]:
        lines.append(f"| {r['model']} | {r['all5_point']} | [{r['all5_ci_lo']}, {r['all5_ci_hi']}] | {r['all5_nagree']}/{r['all5_ntotal']} | "
                      f"{r['nockd_point']} | [{r['nockd_ci_lo']}, {r['nockd_ci_hi']}] | {r['nockd_nagree']}/{r['nockd_ntotal']} | {r['status']} |")
    lines.append("")
    lines.append(f"{claims['C1']['n_models_supported']}/{claims['C1']['n_models_total']} non-logreg models are SUPPORTED (by the fixed rule) as having a "
                 "dataset-averaged AUROC below logreg's at fraction 1.0; the per-cell exceptions below must still be stated alongside this, since the "
                 "dataset-averaged claim does not rule out individual (dataset, model) cells where the model wins.")
    lines.append("")
    lines.append(f"Exception cells at fraction 1.0 ({len(claims['C1']['exceptions'])} of 30 (dataset,model) cells, model's per-dataset mean AUROC > logreg's):")
    lines.append("")
    lines.append("| dataset | model | model mean AUROC | logreg mean AUROC | excess |")
    lines.append("|---|---|---|---|---|")
    for e in claims["C1"]["exceptions"]:
        lines.append(f"| {e['dataset']} | {e['model']} | {e['model_mean_auroc']} | {e['logreg_mean_auroc']} | {e['excess']} |")
    lines.append("")

    lines.append("## C2. The same at fraction 0.1")
    lines.append("")
    lines.append("| model | ALL5 diff | ALL5 CI | ALL5 sign agree | NO_CKD diff | NO_CKD CI | NO_CKD sign agree | status |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in claims["C2"]["per_model"]:
        lines.append(f"| {r['model']} | {r['all5_point']} | [{r['all5_ci_lo']}, {r['all5_ci_hi']}] | {r['all5_nagree']}/{r['all5_ntotal']} | "
                      f"{r['nockd_point']} | [{r['nockd_ci_lo']}, {r['nockd_ci_hi']}] | {r['nockd_nagree']}/{r['nockd_ntotal']} | {r['status']} |")
    lines.append("")
    lines.append(f"{claims['C2']['n_models_supported']}/{claims['C2']['n_models_total']} non-logreg models are SUPPORTED as having a dataset-averaged "
                 "AUROC below logreg's at fraction 0.1; see the per-cell exceptions below.")
    lines.append("")
    lines.append(f"Exception cells at fraction 0.1 ({len(claims['C2']['exceptions'])} of 30 (dataset,model) cells):")
    lines.append("")
    lines.append("| dataset | model | model mean AUROC | logreg mean AUROC | excess |")
    lines.append("|---|---|---|---|---|")
    for e in claims["C2"]["exceptions"]:
        lines.append(f"| {e['dataset']} | {e['model']} | {e['model_mean_auroc']} | {e['logreg_mean_auroc']} | {e['excess']} |")
    lines.append("")

    lines.append("## C3. Gap to logreg is larger at fraction 0.1 than 1.0, for xgb/lgbm/mlp/ftt/tabnet but not rf")
    lines.append("(new quantity: diff-of-differences, group 4b, nested bootstrap seed=42 B=5000)")
    lines.append("")
    lines.append("| model | ALL5 diff-of-diff | ALL5 CI | NO_CKD diff-of-diff | NO_CKD CI | status | note |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in claims["C3"]["per_model"]:
        lines.append(f"| {r['model']} | {r['all5_point']} | [{r['all5_ci_lo']}, {r['all5_ci_hi']}] | {r['nockd_point']} | [{r['nockd_ci_lo']}, {r['nockd_ci_hi']}] | {r['status']} | {r['note']} |")
    lines.append("")
    lines.append("Negative diff-of-diff means the gap to logreg is MORE negative (larger) at fraction 0.1 than at 1.0, matching the claim's direction for that model.")
    lines.append("")

    lines.append("## C4. Per-dataset corrected t-tests (Nadeau-Bengio, Holm)")
    lines.append("")
    lines.append(f"Total cells: {claims['C4']['n_total_cells']} (5 datasets x 2 fractions x 2 metrics x 6 models). Significant (p_holm<0.05): {claims['C4']['n_sig_cells']}.")
    lines.append("")
    sc = claims["C4"]["significant_cell"]
    lines.append(f"The 1 significant cell: dataset={sc['dataset']}, model={sc['model']}, metric={sc['metric']}, fraction={sc['fraction']}, "
                 f"mean_diff_vs_logreg={sc['mean_diff_vs_logreg']}, t={sc['t_statistic']}, p_holm={sc['p_holm']}. Status: **DESCRIPTIVE**.")
    lines.append("")

    lines.append("## C5. Calibration at fraction 0.1 and 1.0: pooled NO_CKD slope per model")
    lines.append("")
    lines.append("| fraction | model | NO_CKD pooled slope | per-dataset slope [CI] |")
    lines.append("|---|---|---|---|")
    for r in claims["C5"]["rows"]:
        per_ds_str = "; ".join(f"{p['dataset']}={p['slope']} [{p['ci_lo']},{p['ci_hi']}]" for p in r["per_dataset"])
        lines.append(f"| {r['fraction']} | {r['model']} | {r['nockd_pooled_slope']} | {per_ds_str} |")
    lines.append("")
    lines.append("Status: **DESCRIPTIVE**. At fraction 0.1, mlp/ftt/tabnet's pooled NO_CKD slopes are below 1 as claimed; "
                 "note xgb's pooled slope (0.975) is ALSO below 1 at fraction 0.1, so the logreg/rf/xgb-vs-mlp/ftt/tabnet grouping is not perfectly clean for xgb.")
    lines.append("")

    lines.append("## C6. Lowest dataset-averaged log loss model, fraction 1.0 and 0.1, ALL5 and NO_CKD, vs. second-best")
    lines.append("")
    lines.append("| fraction | dataset_set | leader | leader mean log loss | second | second mean log loss | leader-second diff | CI |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in claims["C6"]["leaders"]:
        d = e["diff_leader_minus_second"]
        lines.append(f"| {e['fraction']} | {e['dataset_set']} | {e['leader']} | {e['leader_mean']} | {e['second']} | {e['second_mean']} | {d['point']} | [{d['ci_lo']}, {d['ci_hi']}] |")
    lines.append("")
    lines.append("Status: **DESCRIPTIVE**. Note the leader is NOT the same model across all 4 cells (mlp at f1.0/ALL5, logreg at f1.0/NO_CKD, rf at f0.1/ALL5 and f0.1/NO_CKD).")
    lines.append("")

    lines.append("## C7. Range of dataset-averaged AUROC degradation across the 7 models")
    lines.append("")
    lines.append("| perturbation | dataset_set | smallest drop (model) | largest drop (model) | range |")
    lines.append("|---|---|---|---|---|")
    for r in claims["C7"]["rows"]:
        lines.append(f"| {r['perturbation']} | {r['dataset_set']} | {r['smallest_drop']} ({r['smallest_drop_model']}) | {r['largest_drop']} ({r['largest_drop_model']}) | {r['range']} |")
    lines.append("")
    lines.append("Status: **DESCRIPTIVE**. FLAG: at 30% missing / ALL5, the full-precision smallest-magnitude degradation belongs to "
                 "ftt (-0.019576), not mlp (-0.019990) as stated in an earlier CLAUDE.md summary sentence -- both round to -0.020 at 3dp, "
                 "which is why the earlier session reported mlp; the raw (unrounded) values differ. See the Part 7 FLAGS section.")
    lines.append("")

    lines.append("## C8. Robustness log loss and AUROC degradation diff to logreg, per model, at 30% missing / noise 0.5")
    lines.append("")
    lines.append("See claim_map.csv rows with claim_id=C8 for the per-(perturbation,metric,model) SUPPORTED/PARTLY/NOT SUPPORTED status "
                 "(CI-excludes-0 in ALL5 and/or NO_CKD; full point/CI values in claim_map_detail.json).")
    lines.append("")

    lines.append("## C9. Explanation agreement vs. within-model stability and chance")
    lines.append("")
    lines.append(f"Pairs with cross-model Jaccard BELOW within-model stability: {len(claims['C9']['below_within_model_stability'])} of 15.")
    for e in claims["C9"]["below_within_model_stability"]:
        lines.append(f"  - {e['dataset']}/{e['pair']}: {e['jaccard_mean']} < {e['within_model_stability']}")
    lines.append("")
    lines.append(f"Pairs NOT below within-model stability (exceptions): {15 - len(claims['C9']['below_within_model_stability'])}.")
    all_pairs = {(r["dataset"], r["pair"]) for r in claims["C9"]["rows"]}
    below_pairs = {(e["dataset"], e["pair"]) for e in claims["C9"]["below_within_model_stability"]}
    for ds, pair in sorted(all_pairs - below_pairs):
        row = next(r for r in claims["C9"]["rows"] if r["dataset"] == ds and r["pair"] == pair)
        lines.append(f"  - {ds}/{pair}: {row['jaccard_mean']} >= {row['within_model_stability']}")
    lines.append("")
    lines.append(f"Pairs within 0.05 of chance: {len(claims['C9']['within_0.05_of_chance'])}.")
    for e in claims["C9"]["within_0.05_of_chance"]:
        lines.append(f"  - {e['dataset']}/{e['pair']}: jaccard={e['jaccard_mean']}, chance={e['chance']}, diff={e['diff_from_chance']}")
    lines.append("")
    lines.append(f"Breast Wisconsin distinct top-5 sets (of 10 splits): {claims['C9']['breast_wisconsin_distinct_top5_sets_of_10']}.")
    lines.append("")
    lines.append("Status: **DESCRIPTIVE**.")
    lines.append("")

    lines.append("## C10. Clinical plausibility precision vs. chance")
    lines.append("")
    lines.append("| dataset | model | precision | CI | chance |")
    lines.append("|---|---|---|---|---|")
    for r in claims["C10"]["rows"]:
        lines.append(f"| {r['dataset']} | {r['model']} | {r['precision']} | [{r['ci_lo']}, {r['ci_hi']}] | {r['chance']} |")
    lines.append("")
    lines.append(f"Datasets flagged non-informative (chance >= 0.60): {claims['C10']['non_informative_datasets_chance_ge_0.60'] or 'none'}.")
    lines.append("")
    lines.append("Status: **DESCRIPTIVE**.")
    lines.append("")

    lines.append("## C11. Fit-time ratios to logreg, dataset-averaged ALL5, fraction 1.0")
    lines.append("")
    lines.append(f"logreg fit time: {claims['C11']['logreg_fit_time_s']} s. Caveat: {claims['C11']['caveat']}.")
    lines.append("")
    lines.append("| model | fit time (s) | ratio to logreg |")
    lines.append("|---|---|---|")
    for r in claims["C11"]["rows"]:
        lines.append(f"| {r['model']} | {r['fit_time_s']} | {r['ratio_to_logreg']}x |")
    lines.append("")
    lines.append("Status: **DESCRIPTIVE**.")
    lines.append("")

    lines.append("## STATEMENTS THAT MUST NOT BE MADE")
    lines.append("")
    for statement, reason in MUST_NOT_MAKE:
        lines.append(f"- **{statement}**")
        lines.append(f"  - Reason: {reason}")
    lines.append("")

    (OUTPUT_DIR / "claim_map.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"claim_map.csv: {len(csv_rows)} rows")
    print(f"claim_map.md: {len(lines)} lines")
    print(f"claim_map_detail.json written")


if __name__ == "__main__":
    main()

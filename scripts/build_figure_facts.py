r"""Phase 4e: writes paper/evidence/figure_facts.json and .md -- FACTS for drafting the captions, no prose.

For each main figure (4) and Table I / Table II: ids and files, size, per-panel quantity, items shown, symbol
conventions, CI method (B and seed READ from the source scripts / analysis.constants, never retyped), numbers of
splits and datasets (numbers-book key references), exclusions, source CSVs with the SHA-256 copied from
paper/figures/SOURCES.txt, and caveats copied verbatim (registry CAVEATS of build_drafting_sheet.py, each asserted
to be an exact substring of CLAUDE.md / claim_map.md). Result values are never written: a panel lists the
numbers-book KEYS that it displays (matched with the FIGURE_CONTENT patterns of build_drafting_sheet.py).
Deterministic (byte-identical on rerun).
"""
from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_drafting_sheet as ds  # noqa: E402  (NUM, FIGURE_CONTENT, TABLE_KEYS, CAVEATS)

EVID = PROJECT_ROOT / "paper" / "evidence"
OUT_JSON, OUT_MD = EVID / "figure_facts.json", EVID / "figure_facts.md"
SOURCES_TXT = PROJECT_ROOT / "paper" / "figures" / "SOURCES.txt"
FIG_DIR = PROJECT_ROOT / "paper" / "figures"


# ------------------------------------------------------------------ readers (nothing typed by hand)
def _const(path: str, name: str) -> int:
    """Integer constant `NAME = 123` (optionally annotated) read from a source file."""
    m = re.search(rf"^{name}\s*(?::\s*\w+)?\s*=\s*(\d+)\s*$", (PROJECT_ROOT / path).read_text(encoding="utf-8"), re.M)
    assert m, (path, name)
    return int(m.group(1))


def _assert_in(path: str, snippet: str) -> str:
    assert snippet in (PROJECT_ROOT / path).read_text(encoding="utf-8"), (path, snippet)
    return snippet


def parse_sources() -> dict[str, dict]:
    """{figure id: {"size_in": (w, h), "files": [...], "sources": [{"file", "sha256", "filter"}]}} from SOURCES.txt."""
    out: dict[str, dict] = {}
    cur = None
    for line in SOURCES_TXT.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"\[(\w+)\]\s+size ([\d.]+) x ([\d.]+) in\s+files: (.*)", line):
            cur = out.setdefault(m.group(1), {"size_in": [float(m.group(2)), float(m.group(3))], "files": m.group(4), "sources": []})
        elif m := re.match(r"\s+source: (\S+)\s+sha256=([0-9a-f]{64})", line):
            cur["sources"].append({"file": m.group(1), "sha256": m.group(2), "filter": ""})
        elif m := re.match(r"\s+filter: (.*)", line):
            cur["sources"][-1]["filter"] = m.group(1)
        elif m := re.match(r"\s+source files \(pattern\): (.*)", line):
            cur["sources"].append({"file": m.group(1), "sha256": None, "filter": "", "manifest_lines": []})
        elif m := re.match(r"\s+manifest: ([0-9a-f]{64})\s+(\S+)", line):
            cur["sources"][-1]["manifest_lines"].append({"file": m.group(2), "sha256": m.group(1)})
    return out


def png_px(name: str) -> list[int]:
    b = (FIG_DIR / f"{name}.png").read_bytes()
    return list(struct.unpack(">II", b[16:24]))


def keys_matching(panel: str) -> list[str]:
    pat = ds.FIGURE_CONTENT[panel]["key_re"]
    return sorted(k for k in ds.NUM if re.fullmatch(pat, k))


def kv(key: str) -> dict:
    return {"key": key, "value": ds.NUM[key]["formatted"]}


def caveats(ids: list[str]) -> list[dict]:
    return [{"id": i, "text": ds.CAVEATS[i][1]} for i in ids]


def build() -> dict:
    src = parse_sources()
    b_3a, seed = _const("analysis/constants.py", "BOOTSTRAP_B"), _const("analysis/constants.py", "SEED")
    b_3b = _const("scripts/phase3b_calibration.py", "BOOTSTRAP_B")
    b_3c = _const("scripts/phase3c_robustness.py", "BOOTSTRAP_B")
    b_3d = _const("scripts/phase3d_shap.py", "BOOTSTRAP_B")
    b_3dn = _const("scripts/phase3d_noise_continuous.py", "BOOTSTRAP_B")
    _assert_in("analysis/bootstrap.py", "np.percentile(boot_means, [2.5, 97.5])")
    _assert_in("scripts/phase3a_stats.py", "resampling the 10 splits with replacement")
    _assert_in("scripts/phase3b_calibration.py", "np.percentile(valid_slope, [2.5, 97.5])")
    _assert_in("analysis/calibration.py", "def wilson_interval(successes: int, n: int, z: float = 1.96)")
    _assert_in("scripts/make_paper_figures.py", "ax.set_yscale(\"log\")")
    dag = _const_float("scripts/make_paper_figures.py", "DAGGER_THRESHOLD")
    n_splits, n5, n4 = kv("n-outer-splits"), kv("n-datasets"), kv("n-datasets-no-ckd")

    def ci(method: str, b: int, interval: str, resampling: str, **kw) -> dict:
        return {"method": method, "B": b, "seed": seed, "interval": interval, "resampling": resampling, **kw}

    boot_3a = ci("nested percentile bootstrap of the dataset-averaged mean", b_3a, "95% percentile [2.5, 97.5]",
                 "the 10 splits are resampled with replacement within each dataset, then averaged over datasets")
    boot_3c = ci("percentile bootstrap of the dataset-averaged mean change", b_3c, "95% percentile [2.5, 97.5]",
                 "the 10 splits are resampled with replacement within each dataset (between-dataset variability is ignored)")
    boot_3dn = {**boot_3c, "B": b_3dn}

    def panel(letter, panel_id, quantity, items, symbols, ci_, n_s, n_d, excl, files, cav, extra=None):
        return {"letter": letter, "panel_id": panel_id, "quantity": quantity, "items": items, "symbols": symbols, "ci": ci_,
                "n_splits": n_s, "n_datasets": n_d, "exclusions": excl, "sources": files,
                "numbers_book_keys": keys_matching(panel_id) if panel_id in ds.FIGURE_CONTENT else [], "caveats": caveats(cav),
                **(extra or {})}

    def fc(pid):  # the quantity description of FIGURE_CONTENT, minus the regex
        return {k: v for k, v in ds.FIGURE_CONTENT[pid].items() if k != "key_re"}

    def srcs(fig, *needles):
        return [s for s in src[fig]["sources"] if any(n in s["file"] for n in needles)]

    models7 = "logreg, rf, xgb, lgbm, mlp, ftt, tabnet"
    f1, f2, f3, f4 = "fig_main_learning_calibration", "fig_main_robustness", "fig_main_explanations", "fig_main_reliability"
    figures = []

    def figure(fid, label, panels):
        figures.append({"id": fid, "label": label, "files": src[fid]["files"], "size_in": src[fid]["size_in"],
                        "size_px": png_px(fid), "panels": panels, "sources_all": src[fid]["sources"]})

    figure(f1, "Fig. 1", [
        panel("a", "Fig. 1(a)", fc("Fig. 1(a)"), models7,
              "line + marker per model (colour and marker per model; solid for linear/tree models, dashed for deep models); shaded band = 95% CI; log x axis with ticks at the 4 fractions",
              boot_3a, n_splits, n5, ["none (all 5 datasets, ALL5)"], srcs(f1, "phase3a"), ["CKD", "UNIT"]),
        panel("b", "Fig. 1(b)", fc("Fig. 1(b)"), models7, "same conventions as panel (a)", boot_3a, n_splits, n5,
              ["none (all 5 datasets, ALL5)"], srcs(f1, "phase3a"), ["CKD", "UNIT"]),
        panel("c", "Fig. 1(c)", fc("Fig. 1(c)"), models7 + "; 4 datasets: pima_diabetes, heart_cleveland, ilpd, breast_wisconsin",
              "colour = model; marker SHAPE = dataset (o pima_diabetes, s heart_cleveland, ^ ilpd, D breast_wisconsin); whiskers = bootstrap CI of the pooled slope; "
              "black horizontal bar = NO_CKD mean of the 4 pooled slopes; dotted horizontal line at slope 1 (perfect calibration slope); log y axis",
              ci("percentile bootstrap of the pooled slope (per dataset)", b_3b, "95% percentile [2.5, 97.5]",
                 "the 10 splits are resampled with replacement; replicates with separable classes are dropped (column n_dropped_replicates); NO CI exists for the NO_CKD-average bar, none is drawn"),
              n_splits, n4, ["ckd is excluded from the slope panel and its NO_CKD average: saturated dataset with extreme pooled slopes (CLAUDE.md sections 6-7)"],
              srcs(f1, "phase3b"), ["CAL1", "CAL2", "CKD"]),
    ])
    figure(f2, "Fig. 2", [
        panel("a", "Fig. 2(a)", fc("Fig. 2(a)"), models7,
              "two groups on the x axis (30% missing, noise 0.5); one marker per model with vertical CI whisker (colour and marker per model); filled markers",
              boot_3c, n_splits, n5, ["none (ALL5)"], srcs(f2, "phase3c"), ["ROB1", "ROB2", "NOISE"]),
        panel("b", "Fig. 2(b)", fc("Fig. 2(b)"), models7,
              "FILLED markers = original noise; OPEN (white-filled) markers = continuous-only noise variant; one marker per model with CI whisker",
              {**boot_3c, "note": "original noise: B above; continuous-only variant: B = " + str(b_3dn)},
              n_splits, n5, ["continuous-only variant excludes ordinal columns (CKD sg/al/su, Cleveland ca) from the noise"],
              srcs(f2, "phase3c", "noise_continuous"), ["ROB1", "ROB2", "NOISE"],
              {"numbers_book_keys_note": "original-noise keys and continuous-only keys are listed together in numbers_book_keys"}),
    ])
    figure(f3, "Fig. 3", [
        panel("a", "Fig. 3(a)", fc("Fig. 3(a)"), "pairs logreg-rf, logreg-mlp, rf-mlp; within-model stability; chance level; 5 datasets on the x axis",
              "pair markers: o logreg-rf, ^ logreg-mlp, D rf-mlp (neutral greys); open black square = within-model stability (mean over models); black cross = chance level; no CI drawn",
              {"method": "none drawn in panel (a)", "note": "the CI columns jaccard_ci_lo/hi of agreement_per_dataset.csv are not plotted"},
              n_splits, n5, ["none"], srcs(f3, "agreement_per_dataset", "baselines"), ["EXPL", "CORR", "C9"]),
        panel("b", "Fig. 3(b)", fc("Fig. 3(b)"), "logreg, rf, mlp; 5 datasets on the x axis",
              f"colour and marker per model with CI whisker; black cross = chance precision; DAGGER (†) on a dataset tick label = chance precision >= {dag:.2f} (plausibility non-informative, claim map C10)",
              ci("percentile bootstrap of the mean precision over the 10 splits", b_3d, "95% percentile [2.5, 97.5]", "the 10 splits are resampled with replacement"),
              n_splits, n5, ["none"], srcs(f3, "plausibility"), ["DIR", "C10"]),
    ])
    figure(f4, "Fig. 4", [
        panel("-", "Fig. 4", fc("Fig. 4"), "logreg, rf, mlp",
              "one line + marker per model; vertical bars = Wilson 95% intervals; dashed grey diagonal = perfect calibration; equal-mass bins (10)",
              {"method": "Wilson score interval for a binomial proportion", "interval": "95% (z = 1.96)", "B": None, "seed": None, "resampling": "none"},
              n_splits, n4, ["ckd excluded (saturated); the 4 NO_CKD datasets are pooled per model, n = 2 x rows per dataset summed"],
              [s for s in src[f4]["sources"]], ["CAL1", "CAL2"])])
    tables = []
    for tid, fname in ds.TABLE_FILES.items():
        keys = sorted(ds.TABLE_KEYS[tid])
        tables.append({"id": tid, "file": f"paper/tables/{fname}", "content": ds.TABLE_CONTENT[tid],
                       "env": re.search(r"\\begin\{(table\*?)\}", (PROJECT_ROOT / "paper" / "tables" / fname).read_text(encoding="utf-8")).group(1),
                       "n_numbers_book_keys": len(keys), "numbers_book_keys": keys,
                       "rule": ("best per column in bold by the UNROUNDED value (highest AUROC, lowest log loss, slope closest to 1); cells that print identically after rounding can both look best"
                                if tid == "Table II" else "no bold; one row per dataset"),
                       "caveats": caveats(["CAL1", "CKD"] if tid == "Table II" else ["CKD"])})
    return {"generated_by": "scripts/build_figure_facts.py", "figures": figures, "tables": tables}


def _const_float(path: str, name: str) -> float:
    m = re.search(rf"^{name}\s*=\s*([\d.]+)", (PROJECT_ROOT / path).read_text(encoding="utf-8"), re.M)
    assert m, (path, name)
    return float(m.group(1))


def render_md(facts: dict) -> str:
    L = ["# Figure and table facts for the captions (generated by scripts/build_figure_facts.py; facts only, no prose)", "",
         "Result values are never written here: each panel lists the numbers-book KEYS it displays (use `\\N{key}`). Source hashes are copied from `paper/figures/SOURCES.txt`.", ""]
    for f in facts["figures"]:
        L += [f"## {f['label']} -- `{f['id']}`", f"- files: {f['files']}", f"- size: {f['size_in'][0]} x {f['size_in'][1]} in ({f['size_px'][0]} x {f['size_px'][1]} px)", ""]
        for p in f["panels"]:
            q = p["quantity"]
            L += [f"### {p['panel_id']}", f"- quantity: metric = {q['metric']}; dataset set = {q['dataset_set']}; fraction or level = {q['fraction_or_level']}; kind = {q['kind']}; statistic = {q['statistic']}",
                  f"- items shown: {p['items']}", f"- symbols: {p['symbols']}",
                  f"- CI: {json.dumps(p['ci'], ensure_ascii=False)}",
                  f"- splits: `{p['n_splits']['key']}` = {p['n_splits']['value']}; datasets: `{p['n_datasets']['key']}` = {p['n_datasets']['value']}",
                  f"- exclusions: {'; '.join(p['exclusions'])}", f"- numbers-book keys displayed: {len(p['numbers_book_keys'])} (see figure_facts.json)",
                  "- sources:"]
            L += [f"  - `{s['file']}` sha256 `{s['sha256']}`" + (f" ({len(s['manifest_lines'])} manifest lines)" if s.get("manifest_lines") else "") + f" -- {s['filter']}" for s in p["sources"]]
            L += ["- caveats (verbatim):"] + [f"  - **{c['id']}**: {c['text']}" for c in p["caveats"]] + [""]
    for t in facts["tables"]:
        L += [f"## {t['id']} -- `{t['file']}`", f"- content: {t['content']}", f"- rule: {t['rule']}", f"- numbers-book keys used: {t['n_numbers_book_keys']} (see figure_facts.json)",
              "- caveats (verbatim):"] + [f"  - **{c['id']}**: {c['text']}" for c in t["caveats"]] + [""]
    return "\n".join(L)


def main() -> None:
    facts = build()
    OUT_JSON.write_text(json.dumps(facts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    OUT_MD.write_text(render_md(facts), encoding="utf-8", newline="\n")
    for f in facts["figures"]:
        print(f"{f['id']}: {len(f['panels'])} panel(s), {sum(len(p['numbers_book_keys']) for p in f['panels'])} keys")
    for t in facts["tables"]:
        print(f"{t['id']}: {t['n_numbers_book_keys']} keys")


if __name__ == "__main__":
    main()

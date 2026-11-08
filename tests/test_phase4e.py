r"""Phase 4e tests: drafting-sheet labels and hints, C7 errata, static LaTeX check, checker config, figure facts,
IEEEtran main.tex, bundle, and the guards (no test overwrites paper/evidence, no test imports another test)."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_claim_map_errata as errata  # noqa: E402
from scripts import build_drafting_sheet as ds  # noqa: E402
from scripts import build_figure_facts as ff  # noqa: E402
from scripts import check_paper_numbers as cpn  # noqa: E402
from scripts import check_paper_static as cps  # noqa: E402
from scripts import make_overleaf_bundle as bundle  # noqa: E402
from tests.conftest import evidence_hashes  # noqa: E402  (conftest helper, not a test module)

PAPER = PROJECT_ROOT / "paper"
EVID = PAPER / "evidence"
NUM = {e["key"]: e for e in json.loads((EVID / "numbers.json").read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def sheet() -> pd.DataFrame:
    return pd.read_csv(EVID / "drafting_sheet.csv", dtype=str, keep_default_na=False)


# ---------------------------------------------------------------- Part 2: drafting-sheet labels
def _panels(label: str) -> list[str]:
    return [p for p in list(ds.FIGURE_CONTENT) + list(ds.TABLE_FILES) if p in label.split("; ")]


def test_figure_content_covers_every_figure_and_table_with_all_fields() -> None:
    assert set(ds.FIGURE_CONTENT) == {"Fig. 1(a)", "Fig. 1(b)", "Fig. 1(c)", "Fig. 2(a)", "Fig. 2(b)", "Fig. 3(a)", "Fig. 3(b)", "Fig. 4"}
    for c in ds.FIGURE_CONTENT.values():
        assert {"metric", "dataset_set", "fraction_or_level", "kind", "statistic", "key_re"} <= set(c)
    assert set(ds.TABLE_CONTENT) == set(ds.TABLE_FILES) == {"Table I", "Table II"}


def test_table_labels_use_keys_that_appear_in_that_table_file(sheet: pd.DataFrame) -> None:
    n = 0
    for r in sheet.itertuples():
        for t, fname in ds.TABLE_FILES.items():
            if t in r.shown_in.split("; "):
                tex_keys = set(re.findall(r"\\N\{([^}]+)\}", (PAPER / "tables" / fname).read_text(encoding="utf-8")))
                assert any(k in tex_keys for k in r.keys.split("|")), (r.item_id, t)
                n += 1
    assert n >= 8


def test_figure_labels_match_figure_content_exactly(sheet: pd.DataFrame) -> None:
    n = 0
    for r in sheet.itertuples():
        labelled = [p for p in _panels(r.shown_in) if p in ds.FIGURE_CONTENT]
        for p in labelled:  # the panel really displays at least one key of the row
            assert any(re.fullmatch(ds.FIGURE_CONTENT[p]["key_re"], k) for k in r.keys.split("|")), (r.item_id, p)
            n += 1
        if r.shown_in != "text only":  # and every key of the row is displayed by some labelled item
            for k in r.keys.split("|"):
                assert any(re.fullmatch(ds.FIGURE_CONTENT[p]["key_re"], k) for p in labelled) or \
                    any(k in ds.TABLE_KEYS[t] for t in ds.TABLE_KEYS if t in r.shown_in.split("; ")), (r.item_id, k)
    assert n > 40


def test_difference_rows_of_c1_c2_c3_c8_are_text_only(sheet: pd.DataFrame) -> None:
    pat = re.compile(r"-diff-|excess|-dod-|degradation-miss30|degradation-noise50|-degradation-")
    for r in sheet.itertuples():
        if re.search(r"\bC(1|2|3|8):", r.claim_status) and any(pat.search(k) for k in r.keys.split("|")):
            assert r.shown_in == "text only", (r.item_id, r.shown_in)
    for c in ds.FIGURE_CONTENT.values():  # no panel pattern can match a difference/excess key
        assert not any(re.fullmatch(c["key_re"], k) for k in NUM if re.search(r"-diff-|excess|-dod-", k))


def test_c7_rows_are_labelled_only_if_every_key_is_displayed(sheet: pd.DataFrame) -> None:
    for r in sheet[sheet.claim_status.str.contains("C7:")].itertuples():
        # C7 rows mix AUROC/log loss, NO_CKD and the continuous-only noise: some key is never displayed -> text only
        if any(not any(re.fullmatch(c["key_re"], k) for c in ds.FIGURE_CONTENT.values()) for k in r.keys.split("|")):
            assert r.shown_in == "text only", r.item_id


def test_shown_label_unit(monkeypatch) -> None:
    monkeypatch.setattr(ds, "NUM", {k: {"value": 0.1} for k in
                                    ("auroc-mean-rf-all5-f0.1", "logloss-mean-rf-all5-f0.1", "auroc-mean-rf-nockd-f0.1", "x-diff-a")})
    monkeypatch.setattr(ds, "TABLE_KEYS", {"Table II": {"auroc-mean-rf-all5-f0.1"}, "Table I": set()})
    assert ds.shown_label(["auroc-mean-rf-all5-f0.1"]) == "Fig. 1(a); Table II"
    assert ds.shown_label(["auroc-mean-rf-all5-f0.1", "logloss-mean-rf-all5-f0.1"]) == "Fig. 1(a); Fig. 1(b); Table II"
    assert ds.shown_label(["auroc-mean-rf-all5-f0.1", "auroc-mean-rf-nockd-f0.1"]) == "text only"  # NO_CKD key is shown nowhere
    assert ds.shown_label(["x-diff-a"]) == "text only" and ds.shown_label([]) == "text only"
    assert ds.shown_label(["pf:something"]) == "text only"


def _fake_book(monkeypatch, **vals: float) -> None:
    monkeypatch.setattr(ds, "NUM", {k: {"value": v} for k, v in vals.items()})


def test_display_hint_rules(monkeypatch) -> None:
    _fake_book(monkeypatch, **{
        "auroc-diff-a-b-all5-f1.0-point": 0.0004, "auroc-diff-a-b-all5-f1.0-ci-lo": -0.01, "auroc-diff-a-b-all5-f1.0-ci-hi": 0.01,
        "auroc-diff-c-d-all5-f1.0-point": -0.004, "auroc-diff-c-d-all5-f1.0-ci-lo": -0.006, "auroc-diff-c-d-all5-f1.0-ci-hi": -0.001,
        "auroc-excess-over-logreg-ckd-rf-f1.0": 0.0003, "auroc-mean-rf-all5-f1.0": 0.0001,
        "auroc-diff-a-b-all5-f1.0-npos": 0.0, "auroc-diff-e-f-all5-f1.0-point": 0.0005,
        "auroc-diff-g-h-all5-f1.0-ci-lo": 0.002, "auroc-diff-g-h-all5-f1.0-ci-hi": 0.009})
    stem = "auroc-diff-a-b-all5-f1.0"
    assert ds.display_hint([f"{stem}-point", f"{stem}-ci-lo", f"{stem}-ci-hi"]) == "TIE_AT_3_DECIMALS|CI_INCLUDES_ZERO"
    s2 = "auroc-diff-c-d-all5-f1.0"
    assert ds.display_hint([f"{s2}-point", f"{s2}-ci-lo", f"{s2}-ci-hi"]) == "NONE"  # CI excludes 0, not a tie
    assert ds.display_hint(["auroc-excess-over-logreg-ckd-rf-f1.0"]) == "TIE_AT_3_DECIMALS"
    assert ds.display_hint(["auroc-mean-rf-all5-f1.0"]) == "NONE"  # small, but not a difference/excess
    assert ds.display_hint([f"{stem}-npos"]) == "NONE"  # counts are never ties
    assert ds.display_hint(["auroc-diff-e-f-all5-f1.0-point"]) == "NONE"  # exactly 0.0005 is not below 0.0005
    s3 = "auroc-diff-g-h-all5-f1.0"
    assert ds.display_hint([f"{s3}-ci-lo", f"{s3}-ci-hi"]) == "NONE"  # CI above 0
    assert ds.display_hint([]) == "NONE"


def test_sheet_hints_agree_with_numbers(sheet: pd.DataFrame) -> None:
    assert set(sheet.display_hint) <= {"NONE", "TIE_AT_3_DECIMALS", "CI_INCLUDES_ZERO", "TIE_AT_3_DECIMALS|CI_INCLUDES_ZERO"}
    for r in sheet.itertuples():
        refs = [k for k in r.keys.split("|") if k]
        tie = any(any(w in k for w in ds._DIFF_WORDS) and not k.endswith(ds._COUNT_SUFFIX) and abs(NUM[k]["value"]) < 0.0005
                  for k in refs if k in NUM)
        assert ("TIE_AT_3_DECIMALS" in r.display_hint) == tie, r.item_id
    assert (sheet.display_hint == "NONE").sum() > 300 and sheet.display_hint.str.contains("TIE").sum() >= 10


def test_sheet_regeneration_is_byte_identical(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ds, "OUT_CSV", tmp_path / "a.csv")
    monkeypatch.setattr(ds, "OUT_MD", tmp_path / "a.md")
    ds.build(); ds.write()
    first = (tmp_path / "a.csv").read_bytes(), (tmp_path / "a.md").read_bytes()
    monkeypatch.setattr(ds, "OUT_CSV", tmp_path / "b.csv")
    monkeypatch.setattr(ds, "OUT_MD", tmp_path / "b.md")
    ds.build(); ds.write()
    assert first == ((tmp_path / "b.csv").read_bytes(), (tmp_path / "b.md").read_bytes())
    assert first[0] == (EVID / "drafting_sheet.csv").read_bytes() and first[1] == (EVID / "drafting_sheet.md").read_bytes()
    assert "TIE_AT_3_DECIMALS" in first[1].decode() and "CI_INCLUDES_ZERO" in first[1].decode()


def test_sheet_keys_exist_and_values_equal_numbers_book(sheet: pd.DataFrame) -> None:
    n = 0
    for r in sheet.itertuples():
        for k, v, raw in zip(r.keys.split("|"), r.values.split("|"), r.raw_values.split("|")):
            if k in NUM:
                assert NUM[k]["formatted"] == v and repr(NUM[k]["value"]) == raw, k
                n += 1
    assert n > 500


# ---------------------------------------------------------------- Part 3: C7 errata
def test_errata_equals_independent_recomputation() -> None:
    rows = errata.compute(NUM)
    assert len(rows) == 6
    for r in rows:
        pk = dict(errata.PERTURBATIONS)[r["perturbation"]]
        sk = dict(errata.SETS)[r["set"]]
        vals = sorted((NUM[f"auroc-{pk}-degradation-{m}-{sk}"]["value"], m) for m in ("logreg", "rf", "xgb", "lgbm", "mlp", "ftt", "tabnet"))
        assert r["largest"][0] == vals[0][1] and r["smallest"][0] == vals[-1][1]  # most negative = largest drop
        assert r["largest"][3] < r["smallest"][3]
        assert abs(r["range"] - (vals[-1][0] - vals[0][0])) < 1e-15
    text = (EVID / "claim_map_errata.md").read_text(encoding="utf-8")
    for r in rows:
        assert r["largest"][1] in text and r["smallest"][1] in text and r["largest"][2] in text


def test_errata_documents_the_swapped_labels_and_claim_map_is_untouched() -> None:
    printed = errata.claim_map_c7_table()
    by = {(r["perturbation"], r["set"]): r for r in errata.compute(NUM)}
    assert len(printed) == 6
    for p in printed:
        r = by[(p["perturbation"], p["set"])]
        assert p["smallest_label"] == r["largest"][0] and p["largest_label"] == r["smallest"][0]  # claim_map.md has them swapped
    assert "swapped? |" in (EVID / "claim_map_errata.md").read_text(encoding="utf-8")
    assert (EVID / "claim_map_errata.md").read_text(encoding="utf-8") == errata.render(errata.compute(NUM), printed, errata.check_sheet_c7(NUM))


def test_c7_sheet_rows_agree_with_numbers_book() -> None:
    assert errata.check_sheet_c7(NUM) == []


def test_c7_check_detects_a_planted_wrong_value(tmp_path: Path) -> None:
    src = (EVID / "drafting_sheet.csv").read_text(encoding="utf-8").splitlines()
    bad = [l.replace("-0.024|0.053", "-0.099|0.053", 1) if l.startswith("VC-001,") else l for l in src]
    p = tmp_path / "s.csv"
    p.write_text("\n".join(bad) + "\n", encoding="utf-8")
    assert any("VC-001" in x for x in errata.check_sheet_c7(NUM, p))


# ---------------------------------------------------------------- Part 5 / 6 / 7: guards
def test_rebuilds_never_touch_files_under_paper_evidence(rebuilt_numbers_dir: Path, tmp_path: Path, monkeypatch) -> None:
    import scripts.build_claim_map as bcm
    import scripts.build_numbers as bn
    before = evidence_hashes()
    bn.main(out_dir=tmp_path / "nb")
    monkeypatch.setattr(bcm, "OUTPUT_DIR", tmp_path / "cm")
    bcm.main()
    monkeypatch.setattr(ds, "OUT_CSV", tmp_path / "s.csv")
    monkeypatch.setattr(ds, "OUT_MD", tmp_path / "s.md")
    ds.build(); ds.write()
    assert evidence_hashes() == before
    assert (tmp_path / "nb" / "numbers.json").read_bytes() == (EVID / "numbers.json").read_bytes()
    assert (rebuilt_numbers_dir / "numbers.json").read_bytes() == (EVID / "numbers.json").read_bytes()
    st = subprocess.run(["git", "status", "--porcelain", "--", "paper/evidence"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    if st.returncode == 0:  # only files that this very phase changes on purpose may differ from HEAD; none may be the numbers book
        assert "numbers" not in st.stdout and "claim_map.md" not in st.stdout and "claim_map.csv" not in st.stdout


def test_build_numbers_out_dir_option_exists_and_default_dir_unchanged() -> None:
    import scripts.build_numbers as bn
    assert bn.OUTPUT_DIR == PAPER / "evidence"
    assert "--out-dir" in (PROJECT_ROOT / "scripts" / "build_numbers.py").read_text(encoding="utf-8")


def test_no_test_imports_another_test_module() -> None:
    bad = []
    for f in (PROJECT_ROOT / "tests").glob("*.py"):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(from\s+tests\.test_\w+|from\s+test_\w+|import\s+test_\w+|from\s+\.test_\w+)", line):
                bad.append(f"{f.name}:{i}: {line.strip()}")
    assert bad == [], bad
    assert not re.search(r"from tests\.test_", "\n".join(f.read_text(encoding="utf-8") for f in (PROJECT_ROOT / "tests").glob("test_*.py")))


def test_no_stale_last_page_names_left() -> None:
    needle = "page" + "5"
    for d in ("scripts", "tests"):
        for f in (PROJECT_ROOT / d).glob("*.py"):
            assert needle not in f.read_text(encoding="utf-8"), f
    import scripts.check_paper_layout as cpl
    assert hasattr(cpl, "check_last_page_headings") and not hasattr(cpl, "check_" + needle + "_headings")


def test_no_control_characters_in_paper_text_files() -> None:
    """Phase 4e finding: sections/abstract.tex held a backspace and an escape byte instead of \\begin and \\end."""
    for p in PAPER.rglob("*"):
        if p.is_file() and p.suffix in {".tex", ".bib", ".md", ".txt", ".csv", ".json"}:
            assert not re.search(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]", p.read_bytes()), p


# ---------------------------------------------------------------- Part 8: numbers-checker config
def test_checker_config_excludes_supplementary_tables() -> None:
    cfg = json.loads((EVID / "checker_config.json").read_text(encoding="utf-8"))
    assert cfg["excluded_globs"] == ["paper/tables/supp_*.tex"]
    globs = cpn.load_excluded_globs()
    for f in (PAPER / "tables").glob("supp_*.tex"):
        assert cpn.is_excluded(f, globs)
    for f in [PAPER / "main.tex", PAPER / "tables" / "table_main.tex", *(PAPER / "sections").glob("*.tex")]:
        assert not cpn.is_excluded(f, globs)


def test_exclusion_by_config_file(tmp_path: Path) -> None:
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"excluded_globs": ["*/odd_*.tex"]}), encoding="utf-8")
    assert cpn.is_excluded(tmp_path / "odd_table.tex", cpn.load_excluded_globs(cfg))
    assert not cpn.is_excluded(tmp_path / "even.tex", cpn.load_excluded_globs(cfg))
    assert cpn.load_excluded_globs(tmp_path / "missing.json") == []


def _run_numbers(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "check_paper_numbers.py"), *args],
                          capture_output=True, text=True, cwd=PROJECT_ROOT)


def test_exclusion_by_cli_and_excluded_file_is_still_listed(tmp_path: Path) -> None:
    f = tmp_path / "supp_x.tex"
    f.write_text("value 123456.789 typed by hand\n", encoding="utf-8")
    assert _run_numbers(str(f)).returncode == 1  # not excluded: the typed number is unmatched
    r = _run_numbers(str(f), "--exclude", "*supp_x.tex")
    assert r.returncode == 0 and "supp_x.tex" in r.stdout and "skipped (supplementary; numbers come directly from CSV files)" in r.stdout
    assert "UNMATCHED" not in r.stdout


def test_default_run_exits_zero_and_lists_skipped_supplementary_tables() -> None:
    r = _run_numbers()
    assert r.returncode == 0, r.stdout[-800:]
    assert r.stdout.count("skipped (supplementary; numbers come directly from CSV files)") == 4
    assert "main.tex" in r.stdout and "table_main.tex" in r.stdout and "sections" in r.stdout


# ---------------------------------------------------------------- Part 9: main.tex uses IEEEtran
def test_main_tex_uses_ieeetran_and_only_needed_packages() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert tex.startswith("% Compile on Overleaf. If IEEEtran reports an error, copy it into the planning chat.")
    assert "\\documentclass[conference]{IEEEtran}" in tex
    assert "article" not in tex and "providecommand" not in tex and "TEMPORARY" not in tex
    assert re.findall(r"^\\usepackage\{([^}]*)\}", tex, re.M) == ["graphicx", "booktabs"]
    assert "\\input{evidence/numbers.tex}" in tex and tex.count("[EMAIL]") == 4
    assert re.findall(r"\\IEEEauthorblockN\{([^}]*)\}", tex) == ["Demian Sharwin Xaxa", "Aaron John Langan", "Nithy Maria Mathew", "Gangavarapu V Vasu"]
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "check_paper_layout.py"), str(PAPER / "main.tex")],
                       capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert r.returncode == 0 and "Authors: 4" in r.stdout


# ---------------------------------------------------------------- Part 10: static check
def _tree(tmp_path: Path, main: str, **files: str) -> Path:
    (tmp_path / "main.tex").write_text(main, encoding="utf-8")
    for name, content in files.items():
        p = tmp_path / name.replace("__", "/")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return tmp_path / "main.tex"


def _warns(rep: cps.Report) -> list[str]:
    """Warnings except the (expected, in synthetic trees) 'no numbers book next to main.tex' one."""
    return [w for w in rep.warnings if "check skipped" not in w]


def _errors(rep: cps.Report, needle: str) -> list[str]:
    return [e for e in rep.errors if needle in e]


def test_static_clean_tree(tmp_path: Path) -> None:
    (tmp_path / "references.bib").write_text("@article{a1, title={T}}\n", encoding="utf-8")
    m = _tree(tmp_path, "\\begin{document}\n\\input{sec}\\cite{a1}\\ref{f:x}\n\\end{document}\n", sec="\\begin{figure}\n\\caption{c}\\label{f:x}\n\\end{figure}\n")
    rep = cps.check(m)
    assert rep.errors == [] and _warns(rep) == [] and len(rep.files) == 2


def test_static_unbalanced_brace_reports_file_and_line(tmp_path: Path) -> None:
    rep = cps.check(_tree(tmp_path, "a\n\\textbf{x\nb}\n}\n"))
    assert _errors(rep, "main.tex:4") and "closing brace" in rep.errors[0]
    rep2 = cps.check(_tree(tmp_path, "\\textbf{x\n"))
    assert _errors(rep2, "not balanced")


def test_static_environment_nesting_and_unclosed(tmp_path: Path) -> None:
    rep = cps.check(_tree(tmp_path, "\\begin{a}\n\\begin{b}\n\\end{a}\n\\end{b}\n"))
    assert _errors(rep, "bad nesting") and _errors(rep, "main.tex:3")
    assert _errors(cps.check(_tree(tmp_path, "\\begin{a}\n")), "never closed")
    assert _errors(cps.check(_tree(tmp_path, "\\end{a}\n")), "without \\begin")


def test_static_labels_and_refs(tmp_path: Path) -> None:
    rep = cps.check(_tree(tmp_path, "\\label{x}\n\\label{x}\n\\ref{nope}\\eqref{x}\\autoref{y}\\cref{x,z}\n"))
    assert _errors(rep, "duplicate \\label{x}")
    assert sorted(re.findall(r"label '(\w+)'", " ".join(_errors(rep, "undefined label")))) == ["nope", "y", "z"]


def test_static_missing_figure_and_input_files(tmp_path: Path) -> None:
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "ok.pdf").write_bytes(b"%PDF")
    m = _tree(tmp_path, "\\includegraphics[width=1in]{figures/ok}\\includegraphics{figures/missing.pdf}\n\\input{nosuch}\\input{sec}\n", sec="x\n")
    rep = cps.check(m)
    assert _errors(rep, "'figures/missing.pdf' not found") and _errors(rep, "'nosuch' not found")
    assert not _errors(rep, "figures/ok") and not _errors(rep, "'sec'")


def test_static_unknown_number_key(tmp_path: Path) -> None:
    ev = tmp_path / "evidence"
    ev.mkdir()
    (ev / "numbers.json").write_text(json.dumps([{"key": "good-key"}]), encoding="utf-8")
    rep = cps.check(_tree(tmp_path, "\\N{good-key} \\N{bad-key}\n"))
    assert _errors(rep, "bad-key") and not _errors(rep, "good-key")


def test_static_number_keys_fall_back_to_numbers_tex(tmp_path: Path) -> None:
    ev = tmp_path / "evidence"
    ev.mkdir()
    (ev / "numbers.tex").write_text("\\expandafter\\def\\csname num:k-1\\endcsname{0.5}\n", encoding="utf-8")
    rep = cps.check(_tree(tmp_path, "\\N{k-1} \\N{k-2}\n"))
    assert _errors(rep, "k-2") and not _errors(rep, "k-1")


def test_static_empty_bib_is_a_warning_and_cites_are_checked_otherwise(tmp_path: Path) -> None:
    rep = cps.check(_tree(tmp_path, "\\cite{a,b}\\citep[p.~2]{c}\n"))
    assert rep.errors == [] and "references.bib is empty: 3 cited keys unresolved" in _warns(rep)[0]
    (tmp_path / "references.bib").write_text("@book{a,\n title={x}}\n", encoding="utf-8")
    rep2 = cps.check(tmp_path / "main.tex")
    assert sorted(re.findall(r"key '(\w)'", " ".join(rep2.errors))) == ["b", "c"] and not _warns(rep2)


def test_static_caption_without_label_warns(tmp_path: Path) -> None:
    rep = cps.check(_tree(tmp_path, "\\begin{table}\n\\begin{center}\n\\caption{c}\n\\end{center}\n\\end{table}\n\\begin{figure}\\caption{d}\\label{f}\\end{figure}\n"))
    assert rep.errors == [] and sum("\\caption without a \\label" in w for w in rep.warnings if "empty" not in w) == 1


def test_static_commented_out_placeholders_are_ignored(tmp_path: Path) -> None:
    text = ("% \\begin{figure*}\n%   \\includegraphics{figures/nothing}\\ref{fig:todo}\\N{nokey}\\cite{x}\n"
            "text % \\begin{inline} \\label{dup}\n\\label{dup} 100\\% done \\{ and \\}\n")
    rep = cps.check(_tree(tmp_path, text))
    assert rep.errors == [], rep.errors


def test_static_cli_exit_codes_and_header(tmp_path: Path) -> None:
    ok = _tree(tmp_path, "fine\n")
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "check_paper_static.py"), str(ok)], capture_output=True, text=True)
    assert r.returncode == 0 and "NOT a compiler" in r.stdout
    bad = _tree(tmp_path, "\\textbf{x\n")
    r2 = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "check_paper_static.py"), str(bad)], capture_output=True, text=True)
    assert r2.returncode == 1 and "ERROR" in r2.stdout


def test_static_check_passes_on_the_real_paper() -> None:
    rep = cps.check(PAPER / "main.tex")
    assert rep.errors == []
    assert [w for w in rep.warnings if "references.bib is empty" not in w] == []


# ---------------------------------------------------------------- Part 11: figure facts
def test_figure_facts_sources_exist_and_hashes_current() -> None:
    facts = json.loads((EVID / "figure_facts.json").read_text(encoding="utf-8"))
    manifest = dict(reversed(l.split("  ", 1)) for l in (PROJECT_ROOT / "results" / "MANIFEST_sha256.txt").read_text(encoding="utf-8").splitlines())
    n = 0
    for fig in facts["figures"]:
        for s in fig["sources_all"]:
            if s.get("manifest_lines"):
                assert len(s["manifest_lines"]) == 120
                for m in s["manifest_lines"]:
                    assert manifest[m["file"]] == m["sha256"] == hashlib.sha256((PROJECT_ROOT / m["file"]).read_bytes()).hexdigest()
            else:
                assert hashlib.sha256((PROJECT_ROOT / s["file"]).read_bytes()).hexdigest() == s["sha256"], s["file"]
            n += 1
        for p in fig["panels"]:
            for s in p["sources"]:
                assert s in fig["sources_all"]
    assert n >= 10


def test_figure_facts_results_are_numbers_book_references() -> None:
    facts = json.loads((EVID / "figure_facts.json").read_text(encoding="utf-8"))
    assert [f["id"] for f in facts["figures"]] == ["fig_main_learning_calibration", "fig_main_robustness", "fig_main_explanations", "fig_main_reliability"]
    assert [t["id"] for t in facts["tables"]] == ["Table I", "Table II"]
    for f in facts["figures"]:
        for p in f["panels"]:
            assert set(p["numbers_book_keys"]) <= set(NUM)
            for k in ("n_splits", "n_datasets"):
                assert NUM[p[k]["key"]]["formatted"] == p[k]["value"]
            assert set(p) >= {"quantity", "items", "symbols", "ci", "exclusions", "sources", "caveats"}
            for c in p["caveats"]:  # verbatim from CLAUDE.md or claim_map.md
                assert c["text"] in (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8") or c["text"] in (EVID / "claim_map.md").read_text(encoding="utf-8")
    for t in facts["tables"]:
        assert set(t["numbers_book_keys"]) <= set(NUM)
        assert t["numbers_book_keys"] == sorted(ds.TABLE_KEYS[t["id"]])
    # no free-floating result number: the only numeric leaves are the declared structural ones
    allowed = {"size_in", "size_px", "B", "seed", "n_numbers_book_keys"}

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from walk(v, k)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v, path)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            yield path
    assert set(walk(facts)) <= allowed


def test_figure_facts_ci_parameters_are_read_from_sources() -> None:
    facts = json.loads((EVID / "figure_facts.json").read_text(encoding="utf-8"))
    f1, f2, f3, f4 = facts["figures"]
    assert f1["panels"][0]["ci"]["B"] == ff._const("analysis/constants.py", "BOOTSTRAP_B")
    assert f1["panels"][2]["ci"]["B"] == ff._const("scripts/phase3b_calibration.py", "BOOTSTRAP_B")
    assert f2["panels"][0]["ci"]["B"] == ff._const("scripts/phase3c_robustness.py", "BOOTSTRAP_B")
    assert f4["panels"][0]["ci"]["interval"].startswith("95%") and f4["panels"][0]["numbers_book_keys"] == []
    assert all(len(f["size_in"]) == 2 and len(f["size_px"]) == 2 for f in facts["figures"])


def test_figure_facts_rebuild_is_byte_identical() -> None:
    facts = ff.build()
    assert (json.dumps(facts, indent=2, ensure_ascii=False) + "\n").encode("utf-8") == (EVID / "figure_facts.json").read_bytes()
    assert ff.render_md(facts).encode("utf-8") == (EVID / "figure_facts.md").read_bytes()


# ---------------------------------------------------------------- Part 13: bundle
def test_bundle_has_numbers_tex_only_from_evidence_and_extracted_copy_passes_static_check(tmp_path: Path) -> None:
    z = bundle.build(tmp_path / "b.zip")
    out = tmp_path / "x"
    with zipfile.ZipFile(z) as zf:
        evid = [n for n in zf.namelist() if n.startswith("evidence/")]
        assert evid == ["evidence/numbers.tex"]
        zf.extractall(out)
    rep = cps.check(out / "main.tex")
    assert rep.errors == [], rep.errors
    assert any("references.bib is empty" in w for w in rep.warnings)
    assert "\\documentclass[conference]{IEEEtran}" in (out / "main.tex").read_text(encoding="utf-8")
    assert b"providecommand" not in zipfile.ZipFile(z).read("BUNDLE_README.txt")

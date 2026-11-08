"""Tests for scripts/check_paper_layout.py (Phase 4b): citation-key counting,
author counting, and comment-stripping, on synthetic .tex files. The PDF-page
check is exercised separately only if a PDF tool happens to be available on
this machine (skipped otherwise, matching the "say so, don't guess" design)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.check_paper_layout as cpl  # noqa: E402


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_citation_count_under_limit_ok(tmp_path):
    tex = "\\cite{a2020}\\cite{b2021}\\cite{c2022}\n"
    main = _write(tmp_path, "main.tex", tex)
    code, report = cpl.run_check(main, None)
    assert report["n_citation_keys"] == 3
    assert report["citation_keys_ok"] is True
    assert code == 0


def test_citation_count_exactly_20_is_ok(tmp_path):
    keys = [f"ref{i}" for i in range(20)]
    tex = "".join(f"\\cite{{{k}}}\n" for k in keys)
    main = _write(tmp_path, "main.tex", tex)
    code, report = cpl.run_check(main, None, {**cpl.DEFAULT_LIMITS, "max_references": 20})
    assert report["n_citation_keys"] == 20
    assert report["citation_keys_ok"] is True
    assert code == 0


def test_planted_21_citations_violates_limit(tmp_path):
    keys = [f"ref{i}" for i in range(21)]
    tex = "".join(f"\\cite{{{k}}}\n" for k in keys)
    main = _write(tmp_path, "main.tex", tex)
    code, report = cpl.run_check(main, None, {**cpl.DEFAULT_LIMITS, "max_references": 20})
    assert report["n_citation_keys"] == 21
    assert report["citation_keys_ok"] is False
    assert code == 1
    assert any("citation keys" in v for v in report["violations"])


def test_multiple_keys_in_one_cite_command_all_counted(tmp_path):
    tex = "\\cite{a2020,b2021,c2022}\n"
    main = _write(tmp_path, "main.tex", tex)
    _, report = cpl.run_check(main, None)
    assert report["n_citation_keys"] == 3


def test_citep_citet_parencite_all_recognized(tmp_path):
    tex = "\\citep{a2020} \\citet{b2021} \\parencite{c2022} \\citep*[p.5]{d2023}\n"
    main = _write(tmp_path, "main.tex", tex)
    _, report = cpl.run_check(main, None)
    assert report["n_citation_keys"] == 4


def test_commented_cite_is_ignored(tmp_path):
    tex = "\\cite{real2020}\n% \\cite{commented2021}\nsome text % \\cite{alsocommented2022}\n"
    main = _write(tmp_path, "main.tex", tex)
    _, report = cpl.run_check(main, None)
    assert report["n_citation_keys"] == 1
    assert report["citation_keys"] == ["real2020"]


def test_author_count_under_limit_ok(tmp_path):
    tex = "\\IEEEauthorblockN{A}\\IEEEauthorblockN{B}\n"
    main = _write(tmp_path, "main.tex", tex)
    _, report = cpl.run_check(main, None)
    assert report["n_authors"] == 2
    assert report["authors_ok"] is True


def test_planted_36_authors_violates_limit(tmp_path):
    tex = "".join(f"\\IEEEauthorblockN{{Author {i}}}\n" for i in range(36))
    main = _write(tmp_path, "main.tex", tex)
    code, report = cpl.run_check(main, None)
    assert report["n_authors"] == 36
    assert report["authors_ok"] is False
    assert code == 1
    assert any("authors" in v for v in report["violations"])


def test_no_author_markup_is_not_parsable_not_a_violation(tmp_path):
    tex = "\\author{Plain Text Author List}\n"
    main = _write(tmp_path, "main.tex", tex)
    code, report = cpl.run_check(main, None)
    assert report["n_authors"] is None
    assert report["authors_ok"] is True  # unparsable -> not flagged as a violation
    assert code == 0


def test_input_files_are_followed_and_comment_stripped(tmp_path):
    sections = tmp_path / "sections"
    sections.mkdir()
    _write(sections, "abstract.tex", "\\cite{fromsection2020}\n")
    main_tex = "\\input{sections/abstract}\n\\cite{frommain2021}\n"
    main = _write(tmp_path, "main.tex", main_tex)
    _, report = cpl.run_check(main, None)
    assert report["n_citation_keys"] == 2
    assert set(report["citation_keys"]) == {"fromsection2020", "frommain2021"}


def test_no_pdf_given_reports_string_not_crash(tmp_path):
    main = _write(tmp_path, "main.tex", "\\cite{a2020}\n")
    code, report = cpl.run_check(main, None)
    assert isinstance(report["pdf_check"], str)
    assert code == 0


def test_check_last_page_headings_flags_disallowed_heading():
    text = "some text\n\\section{Conclusion}\nmore text"
    problems = cpl.check_last_page_headings(text)
    assert len(problems) == 1
    assert "Conclusion" in problems[0]


def test_check_last_page_headings_allows_references_and_acknowledgments():
    text = "\\section*{Acknowledgments}\nthanks\n\\section*{References}\n[1] ..."
    problems = cpl.check_last_page_headings(text)
    assert problems == []


# Phase 4e: the former version was skipped whenever no PDF tool was on PATH (pdftotext is on PATH under Git Bash
# but not under PowerShell, so the suite's skipped count depended on the shell). The tool lookup is now
# tested deterministically by faking PATH and the pypdf import.
@pytest.mark.parametrize("have,expected", [
    (("pdfinfo", "pdftotext"), "pdfinfo"), (("pdftotext",), "pdftotext"), ((), None)])
def test_find_pdf_tool_lookup_order(monkeypatch, have, expected):
    monkeypatch.setitem(sys.modules, "pypdf", None)  # makes `import pypdf` raise ImportError
    monkeypatch.setattr(cpl.shutil, "which", lambda name: f"/fake/{name}" if name in have else None)
    assert cpl.find_pdf_tool() == expected


def test_find_pdf_tool_prefers_pypdf(monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", object())
    assert cpl.find_pdf_tool() == "pypdf"


# ---- Phase 4d: venue config ------------------------------------------------
import json  # noqa: E402
import subprocess  # noqa: E402

_SCRIPT = Path(cpl.__file__)


def _cfg(tmp_path: Path, **kw) -> Path:
    base = {"max_pages": 6, "max_references": None, "reference_soft_limit": 35,
            "max_authors": 35, "last_page_references_only": False}
    base.update(kw)
    p = tmp_path / "venue.json"
    p.write_text(json.dumps(base), encoding="utf-8")
    return p


def test_config_values_are_read(tmp_path):
    lim, src, warns = cpl.load_limits(_cfg(tmp_path, max_pages=8, max_authors=10))
    assert lim["max_pages"] == 8 and lim["max_authors"] == 10
    assert src["max_pages"] == "config venue.json" and warns == []


def test_cli_overrides_win(tmp_path):
    lim, src, _ = cpl.load_limits(_cfg(tmp_path, max_pages=8), max_pages=4, max_refs=12)
    assert lim["max_pages"] == 4 and lim["max_references"] == 12
    assert src["max_pages"] == "CLI --max-pages" and src["max_references"] == "CLI --max-refs"


def test_null_max_references_warns_but_exits_zero(tmp_path):
    main = _write(tmp_path, "main.tex", "".join(f"\cite{{r{i}}}\n" for i in range(36)))
    r = subprocess.run([sys.executable, str(_SCRIPT), str(main), "--config", str(_cfg(tmp_path))],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "WARNING" in r.stdout and "soft limit 35" in r.stdout


def test_numeric_max_references_fails(tmp_path):
    main = _write(tmp_path, "main.tex", "".join(f"\cite{{r{i}}}\n" for i in range(5)))
    r = subprocess.run([sys.executable, str(_SCRIPT), str(main), "--config", str(_cfg(tmp_path, max_references=4))],
                       capture_output=True, text=True)
    assert r.returncode == 1


def test_last_page_rule_toggles(tmp_path, monkeypatch):
    main = _write(tmp_path, "main.tex", "x")
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(cpl, "find_pdf_tool", lambda: "pdftotext")
    monkeypatch.setattr(cpl, "page_count_and_last_page_text", lambda p, t: (6, "\section{Conclusion}"))
    off, _ = cpl.run_check(main, pdf, {**cpl.DEFAULT_LIMITS, "last_page_references_only": False})
    on, rep = cpl.run_check(main, pdf, {**cpl.DEFAULT_LIMITS, "last_page_references_only": True})
    assert off == 0 and on == 1 and rep["pdf_check"]["last_page_heading_problems"]


def test_page_limit_uses_max_pages(tmp_path, monkeypatch):
    main = _write(tmp_path, "main.tex", "x")
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(cpl, "find_pdf_tool", lambda: "pdftotext")
    monkeypatch.setattr(cpl, "page_count_and_last_page_text", lambda p, t: (7, ""))
    assert cpl.run_check(main, pdf)[0] == 1
    assert cpl.run_check(main, pdf, {**cpl.DEFAULT_LIMITS, "max_pages": 7})[0] == 0


def test_missing_config_falls_back_with_warning(tmp_path):
    lim, _, warns = cpl.load_limits(tmp_path / "nope.json")
    assert lim == cpl.DEFAULT_LIMITS and len(warns) == 1


# ---- Phase 4e: venue config validation ----------------------------------------
def test_unknown_key_warns_and_is_named(tmp_path):
    p = _cfg(tmp_path, max_page=9)  # typo of max_pages
    _, _, warns = cpl.load_limits(p)
    assert any("max_page" in w and "ignored" in w for w in warns)


def test_documented_info_keys_do_not_warn(tmp_path):
    p = _cfg(tmp_path, venue="X", template="Y", notes="Z")
    assert cpl.load_limits(p)[2] == []


@pytest.mark.parametrize("key,bad", [
    ("max_pages", 0), ("max_pages", "6"), ("max_pages", 6.5), ("max_pages", True),
    ("max_references", "20"), ("max_references", 1.5), ("reference_soft_limit", 0),
    ("max_authors", -1), ("last_page_references_only", "yes"), ("last_page_references_only", 1)])
def test_wrong_type_is_a_config_error(tmp_path, key, bad):
    with pytest.raises(cpl.ConfigError, match=key):
        cpl.load_limits(_cfg(tmp_path, **{key: bad}))


def test_wrong_type_exits_nonzero_from_cli(tmp_path):
    main = _write(tmp_path, "main.tex", "x")
    r = subprocess.run([sys.executable, str(_SCRIPT), str(main), "--config", str(_cfg(tmp_path, max_pages="six"))],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "ERROR" in r.stdout and "max_pages" in r.stdout


def test_valid_types_and_null_max_references_accepted(tmp_path):
    lim, _, warns = cpl.load_limits(_cfg(tmp_path, max_references=None, max_pages=8))
    assert lim["max_references"] is None and warns == []


def test_report_lists_limits_and_their_sources(tmp_path):
    main = _write(tmp_path, "main.tex", "x")
    r = subprocess.run([sys.executable, str(_SCRIPT), str(main), "--config", str(_cfg(tmp_path, max_pages=8)),
                        "--max-refs", "9"], capture_output=True, text=True)
    assert "max_pages=8 (config venue.json)" in r.stdout and "max_references=9 (CLI --max-refs)" in r.stdout
    assert "max_authors=35 (default)" not in r.stdout  # max_authors is in the config here -> source is config
    assert "max_authors=35 (config venue.json)" in r.stdout


def test_printed_report_says_last_page(tmp_path, monkeypatch, capsys):
    main = _write(tmp_path, "main.tex", "x")
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    monkeypatch.setattr(cpl, "find_pdf_tool", lambda: "pdftotext")
    monkeypatch.setattr(cpl, "page_count_and_last_page_text", lambda p, t: (6, ""))
    _, rep = cpl.run_check(main, pdf)
    cpl.print_report(rep)
    out = capsys.readouterr().out
    assert "last page" in out and "page 5" not in out.lower()

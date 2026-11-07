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
    code, report = cpl.run_check(main, None)
    assert report["n_citation_keys"] == 20
    assert report["citation_keys_ok"] is True
    assert code == 0


def test_planted_21_citations_violates_limit(tmp_path):
    keys = [f"ref{i}" for i in range(21)]
    tex = "".join(f"\\cite{{{k}}}\n" for k in keys)
    main = _write(tmp_path, "main.tex", tex)
    code, report = cpl.run_check(main, None)
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


def test_check_page5_headings_flags_disallowed_heading():
    text = "some text\n\\section{Conclusion}\nmore text"
    problems = cpl.check_page5_headings(text)
    assert len(problems) == 1
    assert "Conclusion" in problems[0]


def test_check_page5_headings_allows_references_and_acknowledgments():
    text = "\\section*{Acknowledgments}\nthanks\n\\section*{References}\n[1] ..."
    problems = cpl.check_page5_headings(text)
    assert problems == []


@pytest.mark.skipif(cpl.find_pdf_tool() is None, reason="no PDF tool (pypdf/pdfinfo/pdftotext) available on this machine")
def test_find_pdf_tool_returns_a_known_value():
    tool = cpl.find_pdf_tool()
    assert tool in ("pypdf", "pdfinfo", "pdftotext")

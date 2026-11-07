"""Phase 4b: checks the paper skeleton/draft's layout against EMBC 2027-style
budget rules (CLAUDE.md section 8): at most 20 distinct citation keys, at most
35 authors, and (if a compiled PDF and a PDF tool are available) at most 5
pages, with page 5 containing no section heading other than "Acknowledgments"
(a "References" heading is allowed; any other heading on page 5 is flagged,
since the EMBC rule is 4 pages of technical content plus a 5th page reserved
for acknowledgments and references only).

Usage:
    python scripts/check_paper_layout.py path\\to\\main.tex [--pdf path\\to\\main.pdf]

Exits non-zero when any rule is violated. Never installs anything; if no PDF
tool is available it says so plainly instead of guessing a page count.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_CITE_RE = re.compile(r"\\(?:cite|citep|citet|parencite)\*?(?:\[[^\]]*\])?\{([^}]*)\}")
_AUTHOR_BLOCK_RE = re.compile(r"\\IEEEauthorblockN\{([^}]*)\}")
_SECTION_HEADING_RE = re.compile(r"\\section\*?\{([^}]*)\}")


def strip_comments(text: str) -> str:
    return "\n".join(_COMMENT_RE.sub("", line) for line in text.splitlines())


def find_tex_files(main_tex: Path) -> list[Path]:
    """main.tex plus every file referenced via \\input{...} under the same tree
    (one level deep, matching this project's paper/sections/ layout)."""
    files = [main_tex]
    text = strip_comments(main_tex.read_text(encoding="utf-8"))
    for m in re.finditer(r"\\input\{([^}]*)\}", text):
        rel = m.group(1)
        candidate = main_tex.parent / rel
        if not candidate.suffix:
            candidate = candidate.with_suffix(".tex")
        if candidate.exists():
            files.append(candidate)
    return files


def count_citation_keys(tex_files: list[Path]) -> set[str]:
    keys: set[str] = set()
    for path in tex_files:
        text = strip_comments(path.read_text(encoding="utf-8"))
        for m in _CITE_RE.finditer(text):
            for key in m.group(1).split(","):
                key = key.strip()
                if key:
                    keys.add(key)
    return keys


def count_authors(tex_files: list[Path]) -> int | None:
    """Counts \\IEEEauthorblockN{...} occurrences (one per author) across the
    given files. Returns None if the main.tex does not use this IEEEtran-style
    author markup (not parsable), rather than guessing."""
    total = 0
    found_any = False
    for path in tex_files:
        text = strip_comments(path.read_text(encoding="utf-8"))
        matches = _AUTHOR_BLOCK_RE.findall(text)
        if matches:
            found_any = True
        total += len(matches)
    return total if found_any else None


def find_pdf_tool() -> str | None:
    """Returns 'pypdf', 'pdfinfo' or 'pdftotext' (preference order), or None."""
    try:
        import pypdf  # noqa: F401
        return "pypdf"
    except ImportError:
        pass
    if shutil.which("pdfinfo"):
        return "pdfinfo"
    if shutil.which("pdftotext"):
        return "pdftotext"
    return None


def page_count_and_last_page_text(pdf_path: Path, tool: str) -> tuple[int, str]:
    if tool == "pypdf":
        import pypdf

        reader = pypdf.PdfReader(str(pdf_path))
        n = len(reader.pages)
        last_text = reader.pages[-1].extract_text() or ""
        return n, last_text
    if tool == "pdfinfo":
        out = subprocess.run(["pdfinfo", str(pdf_path)], capture_output=True, text=True, check=True).stdout
        m = re.search(r"^Pages:\s*(\d+)", out, re.MULTILINE)
        if not m:
            raise RuntimeError(f"pdfinfo output did not contain a Pages: line: {out!r}")
        n = int(m.group(1))
        text_out = subprocess.run(["pdftotext", "-f", str(n), "-l", str(n), str(pdf_path), "-"],
                                   capture_output=True, text=True, check=True).stdout
        return n, text_out
    if tool == "pdftotext":
        # xpdf's pdftotext inserts a form-feed (\x0c) after each page by default
        # (no -nopgbrk given); page count = number of non-trivial form-feed-
        # delimited chunks. This is an estimate, not as exact as pdfinfo.
        out = subprocess.run(["pdftotext", str(pdf_path), "-"], capture_output=True, text=True, check=True).stdout
        pages = out.split("\x0c")
        # pdftotext emits a trailing form-feed after the last page too, which
        # produces one extra empty chunk at the end of the split -- drop it.
        if pages and pages[-1].strip() == "":
            pages = pages[:-1]
        return len(pages), (pages[-1] if pages else "")
    raise RuntimeError(f"unknown tool {tool!r}")


def check_page5_headings(page5_text: str) -> list[str]:
    """Returns a list of problem descriptions: any \\section heading found in
    the raw page-5 text other than 'References' or 'Acknowledgments'."""
    problems = []
    for m in _SECTION_HEADING_RE.finditer(page5_text):
        heading = m.group(1).strip()
        if heading.lower() not in ("references", "acknowledgments", "acknowledgements"):
            problems.append(f"page 5 contains a disallowed heading: {heading!r}")
    return problems


def run_check(main_tex: Path, pdf_path: Path | None) -> tuple[int, dict]:
    tex_files = find_tex_files(main_tex)
    citation_keys = count_citation_keys(tex_files)
    n_authors = count_authors(tex_files)

    report: dict = {
        "n_citation_keys": len(citation_keys),
        "citation_keys": sorted(citation_keys),
        "citation_keys_ok": len(citation_keys) <= 20,
        "n_authors": n_authors,
        "authors_ok": (n_authors is None) or (n_authors <= 35),
    }

    if pdf_path is not None and pdf_path.exists():
        tool = find_pdf_tool()
        if tool is None:
            report["pdf_check"] = "no PDF tool available (pypdf, pdfinfo, pdftotext all missing) -- page count not checked"
        else:
            n_pages, last_page_text = page_count_and_last_page_text(pdf_path, tool)
            problems = check_page5_headings(last_page_text) if n_pages >= 5 else []
            report["pdf_check"] = {
                "tool": tool, "n_pages": n_pages, "pages_ok": n_pages <= 5,
                "page5_heading_problems": problems,
            }
    else:
        report["pdf_check"] = "no PDF given/found -- page count not checked"

    violations = []
    if not report["citation_keys_ok"]:
        violations.append(f"{report['n_citation_keys']} citation keys > 20")
    if not report["authors_ok"]:
        violations.append(f"{report['n_authors']} authors > 35")
    if isinstance(report["pdf_check"], dict):
        if not report["pdf_check"]["pages_ok"]:
            violations.append(f"{report['pdf_check']['n_pages']} pages > 5")
        violations.extend(report["pdf_check"]["page5_heading_problems"])
    report["violations"] = violations
    return (1 if violations else 0), report


def print_report(report: dict) -> None:
    print(f"Citation keys: {report['n_citation_keys']} (<=20: {report['citation_keys_ok']}) -- {report['citation_keys']}")
    if report["n_authors"] is None:
        print("Authors: not parsable (no \\IEEEauthorblockN{...} markup found)")
    else:
        print(f"Authors: {report['n_authors']} (<=35: {report['authors_ok']})")
    pc = report["pdf_check"]
    if isinstance(pc, str):
        print(f"PDF check: {pc}")
    else:
        print(f"PDF check (tool={pc['tool']}): {pc['n_pages']} pages (<=5: {pc['pages_ok']})")
        if pc["page5_heading_problems"]:
            for p in pc["page5_heading_problems"]:
                print(f"  {p}")
    print()
    if report["violations"]:
        print(f"VIOLATIONS ({len(report['violations'])}):")
        for v in report["violations"]:
            print(f"  {v}")
    else:
        print("No violations.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check paper layout rules (citations, authors, page count/content).")
    parser.add_argument("main_tex", type=Path)
    parser.add_argument("--pdf", type=Path, default=None, help="compiled PDF path, if one exists")
    args = parser.parse_args()

    exit_code, report = run_check(args.main_tex, args.pdf)
    print_report(report)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

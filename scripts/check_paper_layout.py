"""Phase 4b/4d: checks the paper skeleton/draft's layout against the venue
limits: distinct citation keys, authors, and (if a compiled PDF and a PDF tool
are available) the page count and, optionally, the last-page rule.

Limits come from paper/venue.json (override the path with --config);
--max-pages and --max-refs on the command line take precedence. With
max_references null the reference count only WARNS (exit 0) above
reference_soft_limit; a numeric max_references FAILS when exceeded. The
"last page holds only references/acknowledgments" rule applies only when
last_page_references_only is true. If the config is missing, the documented
defaults are used (6 pages, no hard reference cap, soft limit 35) with a
warning. The report states which limits were used and where each came from.

Usage:
    python scripts/check_paper_layout.py path\\to\\main.tex [--pdf path\\to\\main.pdf]
        [--config paper/venue.json] [--max-pages N] [--max-refs N]

Exits non-zero when any rule is violated. Never installs anything; if no PDF
tool is available it says so plainly instead of guessing a page count.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = PROJECT_ROOT / "paper" / "venue.json"
# Documented fallback when the config is missing (Phase 4d).
DEFAULT_LIMITS: dict = {"max_pages": 6, "max_references": None, "reference_soft_limit": 35,
                        "max_authors": 35, "last_page_references_only": False}


def load_limits(config: Path | None = DEFAULT_CONFIG, max_pages: int | None = None,
                max_refs: int | None = None) -> tuple[dict, dict, list[str]]:
    """Returns (limits, sources, warnings). Precedence: CLI > config > defaults."""
    limits = dict(DEFAULT_LIMITS)
    sources = {k: "default" for k in limits}
    warnings: list[str] = []
    if config is not None and Path(config).exists():
        cfg = json.loads(Path(config).read_text(encoding="utf-8"))
        for k in limits:
            if k in cfg:
                limits[k], sources[k] = cfg[k], f"config {Path(config).name}"
    else:
        warnings.append(f"config {config} not found -- using documented defaults "
                        "(6 pages, no hard reference cap, soft limit 35)")
    if max_pages is not None:
        limits["max_pages"], sources["max_pages"] = max_pages, "CLI --max-pages"
    if max_refs is not None:
        limits["max_references"], sources["max_references"] = max_refs, "CLI --max-refs"
    return limits, sources, warnings


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
            problems.append(f"last page contains a disallowed heading: {heading!r}")
    return problems


def run_check(main_tex: Path, pdf_path: Path | None, limits: dict | None = None,
              sources: dict | None = None, warnings: list[str] | None = None) -> tuple[int, dict]:
    limits = dict(DEFAULT_LIMITS) if limits is None else limits
    tex_files = find_tex_files(main_tex)
    citation_keys = count_citation_keys(tex_files)
    n_authors = count_authors(tex_files)
    max_refs, soft = limits["max_references"], limits["reference_soft_limit"]
    warnings = list(warnings or [])
    if max_refs is None and soft is not None and len(citation_keys) > soft:
        warnings.append(f"{len(citation_keys)} citation keys > soft limit {soft}")

    report: dict = {
        "limits": limits, "limit_sources": sources or {k: "caller" for k in limits},
        "warnings": warnings,
        "n_citation_keys": len(citation_keys),
        "citation_keys": sorted(citation_keys),
        "citation_keys_ok": max_refs is None or len(citation_keys) <= max_refs,
        "n_authors": n_authors,
        "authors_ok": (n_authors is None) or (n_authors <= limits["max_authors"]),
    }

    if pdf_path is not None and pdf_path.exists():
        tool = find_pdf_tool()
        if tool is None:
            report["pdf_check"] = "no PDF tool available (pypdf, pdfinfo, pdftotext all missing) -- page count not checked"
        else:
            n_pages, last_page_text = page_count_and_last_page_text(pdf_path, tool)
            problems = (check_page5_headings(last_page_text)
                        if limits["last_page_references_only"] and n_pages >= 1 else [])
            report["pdf_check"] = {
                "tool": tool, "n_pages": n_pages, "pages_ok": n_pages <= limits["max_pages"],
                "page5_heading_problems": problems,
            }
    else:
        report["pdf_check"] = "no PDF given/found -- page count not checked"

    violations = []
    if not report["citation_keys_ok"]:
        violations.append(f"{report['n_citation_keys']} citation keys > {max_refs}")
    if not report["authors_ok"]:
        violations.append(f"{report['n_authors']} authors > {limits['max_authors']}")
    if isinstance(report["pdf_check"], dict):
        if not report["pdf_check"]["pages_ok"]:
            violations.append(f"{report['pdf_check']['n_pages']} pages > {limits['max_pages']}")
        violations.extend(report["pdf_check"]["page5_heading_problems"])
    report["violations"] = violations
    return (1 if violations else 0), report


def print_report(report: dict) -> None:
    lim, src = report["limits"], report["limit_sources"]
    print("Limits used: " + "; ".join(f"{k}={lim[k]} ({src.get(k, '?')})" for k in lim))
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    print(f"Citation keys: {report['n_citation_keys']} (hard cap ok: {report['citation_keys_ok']}) -- {report['citation_keys']}")
    if report["n_authors"] is None:
        print("Authors: not parsable (no \\IEEEauthorblockN{...} markup found)")
    else:
        print(f"Authors: {report['n_authors']} (<={lim['max_authors']}: {report['authors_ok']})")
    pc = report["pdf_check"]
    if isinstance(pc, str):
        print(f"PDF check: {pc}")
    else:
        print(f"PDF check (tool={pc['tool']}): {pc['n_pages']} pages (<={lim['max_pages']}: {pc['pages_ok']})")
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
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="venue limits JSON")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-refs", type=int, default=None)
    args = parser.parse_args()

    limits, sources, warns = load_limits(args.config, args.max_pages, args.max_refs)
    exit_code, report = run_check(args.main_tex, args.pdf, limits, sources, warns)
    print_report(report)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

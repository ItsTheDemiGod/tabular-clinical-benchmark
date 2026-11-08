r"""Phase 4e: STATIC consistency check of a LaTeX paper. THIS IS NOT A COMPILER.

Nothing can be compiled on this machine (no LaTeX distribution), so this script only catches the
mistakes a text scan can see; a clean result does NOT mean the paper compiles. Compile on Overleaf.

Usage:  python scripts/check_paper_static.py [path/to/main.tex]      (default paper/main.tex)

It follows \input / \include recursively and reports each finding with file and line:
  ERROR   braces not balanced in a file; \begin/\end not balanced or not properly nested;
          duplicate \label; \ref/\eqref/\autoref/\cref target without a \label;
          \includegraphics / \input target not found (relative to main.tex's folder; the
          extension may be omitted); \cite key not in references.bib (when the bib has entries);
          \N{key} not in numbers.json (or, if absent, defined in evidence/numbers.tex).
  WARNING references.bib has no entries ("N cited keys unresolved"); a \caption without a
          \label inside the same figure/table float.
Lines starting with % and the part of a line after an unescaped % are ignored.
Exit code 1 if there is any ERROR, otherwise 0.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_TOKEN_RE = re.compile(r"\\.|[{}]", re.S)
_ENV_RE = re.compile(r"\\(begin|end)\{([^}]*)\}")
_LABEL_RE = re.compile(r"\\label\{([^}]*)\}")
_REF_RE = re.compile(r"\\(?:ref|eqref|autoref|cref|Cref|pageref)\{([^}]*)\}")
_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]*)\}")
_GRAPHICS_RE = re.compile(r"\\includegraphics\*?(?:\[[^\]]*\])?\{([^}]*)\}")
_CITE_RE = re.compile(r"\\(?:cite|citep|citet|citeauthor|citeyear|parencite|textcite)\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}")
_N_RE = re.compile(r"\\N\{([^}]*)\}")
_BIB_KEY_RE = re.compile(r"^\s*@(?!comment|string|preamble)\w+\s*\{\s*([^,\s]+)\s*,", re.I | re.M)
_NUMBERS_TEX_KEY_RE = re.compile(r"\\csname num:([^\\]+?)\\endcsname")
_FLOATS = {"figure", "figure*", "table", "table*"}
_GRAPHIC_EXT = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)


def strip_comment(line: str) -> str:
    """Remove an unescaped % comment from one line."""
    for i, ch in enumerate(line):
        if ch == "%" and (i == 0 or line[i - 1] != "\\"):
            return line[:i]
    return line


def known_number_keys(paper_dir: Path) -> set[str] | None:
    """Keys of numbers.json, else those defined in evidence/numbers.tex, else None (check skipped)."""
    nj = paper_dir / "evidence" / "numbers.json"
    if nj.exists():
        return {e["key"] for e in json.loads(nj.read_text(encoding="utf-8"))}
    nt = paper_dir / "evidence" / "numbers.tex"
    if nt.exists():
        return set(_NUMBERS_TEX_KEY_RE.findall(nt.read_text(encoding="utf-8")))
    return None


def _resolve_input(base: Path, target: str) -> Path | None:
    for cand in (base / target, base / (target + ".tex")):
        if cand.is_file():
            return cand
    return None


def _check_file(path: Path, rel: str, rep: Report, base: Path, labels: dict[str, str],
                refs: list[tuple[str, int, str]], cites: list[tuple[str, int, str]], ns: list[tuple[str, int, str]],
                queue: list[Path]) -> None:
    lines = [strip_comment(l) for l in path.read_text(encoding="utf-8").splitlines()]
    # --- braces ---
    depth, first_neg = 0, None
    for no, line in enumerate(lines, 1):
        for t in _TOKEN_RE.findall(line):
            if t == "{":
                depth += 1
            elif t == "}":
                depth -= 1
                if depth < 0 and first_neg is None:
                    first_neg = no
    if first_neg is not None:
        rep.errors.append(f"{rel}:{first_neg}: closing brace without a matching opening brace")
    elif depth != 0:
        rep.errors.append(f"{rel}: braces not balanced ({depth} unclosed '{{')")
    # --- environments, floats, labels, refs, targets ---
    stack: list[tuple[str, int, dict]] = []
    for no, line in enumerate(lines, 1):
        for m in re.finditer(_ENV_RE.pattern + r"|\\caption\b|\\label\{[^}]*\}", line):
            tok = m.group(0)
            if tok.startswith("\\begin"):
                stack.append((m.group(2), no, {"caption": None, "label": False}))
            elif tok.startswith("\\end"):
                name = m.group(2)
                if not stack:
                    rep.errors.append(f"{rel}:{no}: \\end{{{name}}} without \\begin")
                elif stack[-1][0] != name:
                    rep.errors.append(f"{rel}:{no}: \\end{{{name}}} closes \\begin{{{stack[-1][0]}}} from line {stack[-1][1]} (bad nesting)")
                    stack.pop()
                else:
                    env, bno, st = stack.pop()
                    if env in _FLOATS and st["caption"] is not None and not st["label"]:
                        rep.warnings.append(f"{rel}:{st['caption']}: \\caption without a \\label in the same {env} (begun line {bno})")
            else:  # \caption or \label: attach to the innermost enclosing float
                frame = next((f for f in reversed(stack) if f[0] in _FLOATS), None)
                if frame is not None:
                    if tok.startswith("\\caption"):
                        frame[2]["caption"] = no
                    else:
                        frame[2]["label"] = True
        for m in _LABEL_RE.finditer(line):
            key = m.group(1)
            if key in labels:
                rep.errors.append(f"{rel}:{no}: duplicate \\label{{{key}}} (first in {labels[key]})")
            else:
                labels[key] = f"{rel}:{no}"
        for m in _REF_RE.finditer(line):
            for k in m.group(1).split(","):
                refs.append((rel, no, k.strip()))
        for m in _CITE_RE.finditer(line):
            for k in m.group(1).split(","):
                if k.strip():
                    cites.append((rel, no, k.strip()))
        for m in _N_RE.finditer(line):
            ns.append((rel, no, m.group(1)))
        for m in _GRAPHICS_RE.finditer(line):
            t = m.group(1)
            if not any((base / (t + e)).is_file() for e in _GRAPHIC_EXT):
                rep.errors.append(f"{rel}:{no}: \\includegraphics target {t!r} not found")
        for m in _INPUT_RE.finditer(line):
            t = m.group(1)
            found = _resolve_input(base, t)
            if found is None:
                rep.errors.append(f"{rel}:{no}: \\input/\\include target {t!r} not found")
            else:
                queue.append(found)
    for env, bno, _ in stack:
        rep.errors.append(f"{rel}:{bno}: \\begin{{{env}}} never closed")


def check(main_tex: Path) -> Report:
    main_tex = Path(main_tex)
    base = main_tex.parent
    rep = Report()
    labels: dict[str, str] = {}
    refs: list[tuple[str, int, str]] = []
    cites: list[tuple[str, int, str]] = []
    ns: list[tuple[str, int, str]] = []
    queue, seen = [main_tex], set()
    while queue:
        path = queue.pop(0)
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        rep.files.append(path)
        rel = path.relative_to(base).as_posix() if path.is_relative_to(base) else str(path)
        _check_file(path, rel, rep, base, labels, refs, cites, ns, queue)
    for rel, no, k in refs:
        if k not in labels:
            rep.errors.append(f"{rel}:{no}: reference to undefined label {k!r}")
    bib = base / "references.bib"
    bib_keys = set(_BIB_KEY_RE.findall(bib.read_text(encoding="utf-8"))) if bib.exists() else set()
    if not bib_keys:
        rep.warnings.append(f"references.bib is empty: {len({k for _, _, k in cites})} cited keys unresolved")
    else:
        for rel, no, k in cites:
            if k not in bib_keys:
                rep.errors.append(f"{rel}:{no}: \\cite key {k!r} not in references.bib")
    keys = known_number_keys(base)
    if keys is None:
        rep.warnings.append("no evidence/numbers.json or numbers.tex next to main.tex: \\N{key} check skipped")
    else:
        for rel, no, k in ns:
            if k not in keys:
                rep.errors.append(f"{rel}:{no}: \\N{{{k}}} is not a numbers-book key")
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Static LaTeX consistency check (NOT a compiler).")
    ap.add_argument("main_tex", nargs="?", type=Path, default=Path(__file__).resolve().parent.parent / "paper" / "main.tex")
    args = ap.parse_args()
    print("check_paper_static: a static text check, NOT a compiler. A clean result does not prove the paper compiles.")
    rep = check(args.main_tex)
    print(f"Files followed ({len(rep.files)}): " + ", ".join(p.name for p in rep.files))
    for w in rep.warnings:
        print(f"WARNING: {w}")
    for e in rep.errors:
        print(f"ERROR: {e}")
    print(f"{len(rep.errors)} error(s), {len(rep.warnings)} warning(s).")
    sys.exit(1 if rep.errors else 0)


if __name__ == "__main__":
    main()

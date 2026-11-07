"""Phase 4a: checks a paper draft for numbers that do not come from the numbers
book (paper/evidence/numbers.json).

Usage: python scripts/check_paper_numbers.py path\\to\\draft.tex [--warn-only]
Also accepts .md and .txt drafts.

THIS CHECKER IS HEURISTIC, not a full LaTeX/Markdown parser. It extracts numbers
with a line-by-line regex scan and classifies each as "exact", "rounded" or
"UNMATCHED" against numbers.json's `formatted` strings. It is meant to catch
obvious hand-typed numbers, not to guarantee every number in the draft is
checked (for example a number split across a line break, or produced by LaTeX
arithmetic, will not be seen).

Exclusions (numbers inside these are never extracted):
  - whole lines starting with `%` (LaTeX comments)
  - the argument of \\N{...} (already a numbers-book reference, checked separately)
  - the argument(s) of \\cite{...}, \\citep{...}, \\citet{...}, \\parencite{...}
  - the argument of \\label{...}, \\ref{...}, \\eqref{...}, \\autoref{...}

Extraction heuristic for what counts as "a number" (see UNIT_WORDS below):
  - any token with a decimal point (optionally followed by %) is always checked;
  - any bare integer followed directly by % is always checked;
  - a bare integer of 2+ digits is always checked;
  - a bare integer of 1 digit (i.e. BELOW 10) is only checked if the very next
    word is one of UNIT_WORDS (e.g. "5 datasets"); otherwise it is ignored, since
    small bare integers in running prose ("one of the three...") are usually not
    data values. Number WORDS ("five", "ten", ...) are never extracted at all.

Matching: a token matches "exact" if its text (with a trailing % stripped, since
numbers.json never stores the % sign) equals some entry's `formatted` string
exactly (a `pvalue` entry formatted "<0.001" also matches the bare number
"0.001"). It matches "rounded" if rounding the token's value to fewer decimals
(but never all the way down to a bare integer, which collides with unrelated
count-kind entries) equals some entry's `formatted` string. Otherwise it is
UNMATCHED.

Known limitation: years, equation/footnote numbers and other large bare
integers (e.g. a 2027 in a conference deadline) are not specially exempted and
will be reported UNMATCHED like any other 2+-digit integer; add them to
numbers.json or review them by eye.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NUMBERS_JSON_PATH = PROJECT_ROOT / "paper" / "evidence" / "numbers.json"

UNIT_WORDS = {
    "trials", "folds", "splits", "tasks", "rows", "models", "datasets", "epochs",
    "steps", "features", "columns", "seeds", "draws", "cells", "pairs", "sets",
    "workers", "fits", "classes", "bins", "repeats", "levels", "splits.",
}

_N_MACRO_RE = re.compile(r"\\N\{([^}]*)\}")
_MASK_RES = [
    re.compile(r"\\N\{[^}]*\}"),
    re.compile(r"\\(?:cite|citep|citet|parencite)\*?(?:\[[^\]]*\])?\{[^}]*\}"),
    re.compile(r"\\(?:label|ref|eqref|autoref)\{[^}]*\}"),
]
_TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?%?")


def mask_line(line: str) -> str:
    """Replace excluded spans with spaces of the same length (keeps column offsets)."""
    for pattern in _MASK_RES:
        line = pattern.sub(lambda m: " " * len(m.group()), line)
    return line


def find_n_macro_usages(text: str) -> list[tuple[int, str]]:
    """[(line_no, key), ...] for every \\N{key} usage (line-commented ones excluded)."""
    usages = []
    for i, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("%"):
            continue
        for m in _N_MACRO_RE.finditer(line):
            usages.append((i, m.group(1)))
    return usages


def extract_numbers(text: str) -> list[tuple[int, str, str]]:
    """[(line_no, token_text, context), ...] for every number the heuristic keeps."""
    results = []
    for i, raw_line in enumerate(text.splitlines(), start=1):
        if raw_line.lstrip().startswith("%"):
            continue
        line = mask_line(raw_line)
        for m in _TOKEN_RE.finditer(line):
            token = m.group()
            has_decimal = "." in token
            has_percent = token.endswith("%")
            digits = token.rstrip("%").lstrip("-")
            if has_decimal or has_percent:
                keep = True
            elif len(digits) >= 2:
                keep = True
            else:
                after = line[m.end() : m.end() + 20]
                word_match = re.match(r"\s+([A-Za-z.]+)", after)
                keep = bool(word_match and word_match.group(1).lower().rstrip(".") + "s" in UNIT_WORDS
                            or word_match and word_match.group(1).lower() in UNIT_WORDS)
            if not keep:
                continue
            start, end = max(0, m.start() - 30), min(len(raw_line), m.end() + 30)
            context = raw_line[start:end].strip()
            results.append((i, token, context))
    return results


def build_formatted_index(entries: list[dict]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for e in entries:
        if e["undefined"] or e["formatted"] is None:
            continue
        index.setdefault(e["formatted"], []).append(e["key"])
        if e["formatted"].startswith("<"):
            index.setdefault(e["formatted"][1:], []).append(e["key"])
    return index


def classify_token(token: str, formatted_index: dict[str, list[str]]) -> tuple[str, list[str], int | None]:
    """Returns (status, matching_keys, rounded_decimals). status in
    {"exact", "rounded", "UNMATCHED"}."""
    stripped = token.rstrip("%")
    if stripped in formatted_index:
        return "exact", formatted_index[stripped], None
    if "." in stripped:
        decimals = len(stripped.split(".")[1])
        try:
            value = float(stripped)
        except ValueError:
            return "UNMATCHED", [], None
        # Never round all the way down to a bare integer (d=0): that routinely
        # collides with unrelated count-kind entries (e.g. 12.345 -> "12" would
        # spuriously "match" a hyperparameter count of 12), which is misleading
        # rather than a genuine over-precise citation of a numbers.json value.
        for d in range(decimals - 1, 0, -1):
            candidate = f"{value:.{d}f}"
            if candidate in formatted_index:
                return "rounded", formatted_index[candidate], d
    return "UNMATCHED", [], None


def check_stale_sources(entries: list[dict]) -> list[str]:
    warnings = []
    seen: dict[str, str] = {}
    for e in entries:
        seen.setdefault(e["source_file"], e["source_sha256"])
    for source_file, expected_sha in seen.items():
        path = PROJECT_ROOT / source_file
        if not path.exists():
            warnings.append(f"source file missing: {source_file}")
            continue
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            warnings.append(f"STALE: {source_file} (numbers.json hash {expected_sha[:12]}..., current {actual_sha[:12]}...)")
    return warnings


def run_check(draft_path: Path, numbers_json_path: Path = NUMBERS_JSON_PATH) -> tuple[int, dict]:
    entries = json.loads(numbers_json_path.read_text(encoding="utf-8"))
    known_keys = {e["key"] for e in entries}
    formatted_index = build_formatted_index(entries)
    stale = check_stale_sources(entries)

    text = draft_path.read_text(encoding="utf-8")
    numbers = extract_numbers(text)
    n_usages = find_n_macro_usages(text)

    results = []
    n_unmatched = 0
    for line_no, token, context in numbers:
        status, keys, decimals = classify_token(token, formatted_index)
        if status == "UNMATCHED":
            n_unmatched += 1
        results.append({"line": line_no, "token": token, "context": context, "status": status, "keys": keys, "rounded_decimals": decimals})

    unknown_keys = [(line_no, key) for line_no, key in n_usages if key not in known_keys]

    report = {
        "stale_sources": stale,
        "numbers": results,
        "n_unmatched": n_unmatched,
        "unknown_n_keys": unknown_keys,
        "n_numbers_checked": len(results),
        "n_usages_checked": len(n_usages),
    }
    exit_code = 1 if (n_unmatched > 0 or unknown_keys) else 0
    return exit_code, report


def print_report(report: dict) -> None:
    print(__doc__.split("\n\n")[0])  # heuristic-disclosure header line
    print()
    if report["stale_sources"]:
        print(f"STALE SOURCE WARNINGS ({len(report['stale_sources'])}):")
        for w in report["stale_sources"]:
            print(f"  {w}")
        print()
    print(f"Numbers extracted: {report['n_numbers_checked']}, \\N{{}} usages: {report['n_usages_checked']}")
    for r in report["numbers"]:
        if r["status"] == "exact":
            print(f"  line {r['line']}: {r['token']!r} [{r['context']}] -> exact match: {r['keys']}")
        elif r["status"] == "rounded":
            print(f"  line {r['line']}: {r['token']!r} [{r['context']}] -> rounded match ({r['rounded_decimals']} dp): {r['keys']}")
        else:
            print(f"  line {r['line']}: {r['token']!r} [{r['context']}] -> UNMATCHED")
    print()
    if report["unknown_n_keys"]:
        print(f"UNKNOWN \\N{{}} KEYS ({len(report['unknown_n_keys'])}):")
        for line_no, key in report["unknown_n_keys"]:
            print(f"  line {line_no}: \\N{{{key}}} -- key not in numbers.json")
    print()
    print(f"Summary: {report['n_unmatched']} UNMATCHED number(s), {len(report['unknown_n_keys'])} unknown \\N{{}} key(s).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a paper draft's numbers against the numbers book.")
    parser.add_argument("draft", type=Path)
    parser.add_argument("--warn-only", action="store_true", help="always exit 0, report only")
    args = parser.parse_args()

    exit_code, report = run_check(args.draft)
    print_report(report)
    sys.exit(0 if args.warn_only else exit_code)


if __name__ == "__main__":
    main()

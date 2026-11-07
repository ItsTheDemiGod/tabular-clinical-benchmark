"""Phase 4b: copies a given list of figure ids from the VERIFIED inventory
(paper/evidence/inventory_verified.csv) into paper/figures/, recording each
copy's source path and SHA-256 in paper/figures/SOURCES.txt.

Written in Phase 4b but NOT executed in this phase (no figure has been
selected for the paper yet -- that is a writing-phase decision, made against
the page budget in paper/evidence/inventory_verified.csv's layout flags).

Usage:
    python scripts/copy_paper_figures.py fig3 fig8 fig11 [--pdf|--png]

Only ids present in inventory_verified.csv are accepted (a KeyError-style
message lists the valid ids otherwise). Copies the PDF file by default (print
quality); pass --png to copy the PNG instead. Never invents a figure id and
never copies anything not already verified in Part 2 of Phase 4b.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INVENTORY_PATH = PROJECT_ROOT / "paper" / "evidence" / "inventory.csv"
VERIFIED_PATH = PROJECT_ROOT / "paper" / "evidence" / "inventory_verified.csv"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"
SOURCES_PATH = FIGURES_DIR / "SOURCES.txt"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_ids() -> set[str]:
    import csv

    with VERIFIED_PATH.open(encoding="utf-8") as f:
        return {row["id"] for row in csv.DictReader(f)}


def load_inventory_files() -> dict[str, list[str]]:
    """id -> list of files (relative paths) from inventory.csv's 'files' column."""
    import csv

    out: dict[str, list[str]] = {}
    with INVENTORY_PATH.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = [p.strip() for p in row["files"].split(";")]
    return out


def copy_figure(fig_id: str, files_by_id: dict[str, list[str]], extension: str, sources_lines: list[str]) -> None:
    candidates = [f for f in files_by_id[fig_id] if f.endswith(extension)]
    if not candidates:
        raise FileNotFoundError(f"no .{extension} file listed for figure id {fig_id!r} in inventory.csv")
    if len(candidates) > 1:
        # Bundled ids (e.g. fig2, fig3, fig9's PDF-variant list) list several files
        # of the same id/extension (one per fraction/dataset_set); copy all of them.
        pass
    for rel_path in candidates:
        src = PROJECT_ROOT / rel_path
        if not src.exists():
            raise FileNotFoundError(f"source file does not exist: {src}")
        dst = FIGURES_DIR / src.name
        dst.write_bytes(src.read_bytes())
        sources_lines.append(f"{dst.relative_to(PROJECT_ROOT)}\tfrom={rel_path}\tsha256={sha256(src)}")
        print(f"copied {rel_path} -> {dst.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy selected verified-inventory figures into paper/figures/.")
    parser.add_argument("fig_ids", nargs="+", help="figure ids from paper/evidence/inventory_verified.csv, e.g. fig3 fig8")
    parser.add_argument("--png", action="store_true", help="copy the PNG instead of the PDF")
    args = parser.parse_args()

    verified_ids = load_verified_ids()
    unknown = [f for f in args.fig_ids if f not in verified_ids]
    if unknown:
        print(f"ERROR: not in inventory_verified.csv: {unknown}", file=sys.stderr)
        print(f"Valid ids: {sorted(verified_ids)}", file=sys.stderr)
        sys.exit(1)

    files_by_id = load_inventory_files()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    extension = "png" if args.png else "pdf"

    sources_lines: list[str] = []
    if SOURCES_PATH.exists():
        sources_lines = SOURCES_PATH.read_text(encoding="utf-8").splitlines()
    for fig_id in args.fig_ids:
        copy_figure(fig_id, files_by_id, extension, sources_lines)
    SOURCES_PATH.write_text("\n".join(sources_lines) + "\n", encoding="utf-8")
    print(f"SOURCES.txt: {len(sources_lines)} lines")


if __name__ == "__main__":
    main()

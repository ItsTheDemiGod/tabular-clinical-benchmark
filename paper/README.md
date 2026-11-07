# Paper (Phase 4b skeleton)

This directory holds the EMBC 2027 paper itself, built from the evidence pack
in `paper/evidence/` (Phase 4a/4b). **No prose has been written yet** -- every
file under `sections/` contains only a heading and a `TODO` comment. See
`CLAUDE.md` section 8 for the writing plan and milestones.

## How to build

No LaTeX distribution was found on this machine at Phase 4b time
(2026-11-08): `pdflatex`, `xelatex`, `latexmk`, `bibtex`, `biber`, `pandoc`
and `kpsewhich` were all absent (`paper/evidence/claim_map.md`'s Part 3
tool-availability table, and CLAUDE.md section 8, have the full list). Until
one is installed and `IEEEtran.cls`/`.bst` are available:

```
# once a LaTeX distribution with IEEEtran is installed:
cd paper
latexmk -pdf main.tex      # or: pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

`main.tex` currently falls back to the plain `article` class (clearly marked
`TODO`) so the file at least parses; switch the commented
`\documentclass[conference]{IEEEtran}` line back on once the class is
available, and remove the `IEEEauthorblockN`/`IEEEauthorblockA` fallback
`\providecommand`s added for the placeholder class.

## The 4+1 page rule and the 20-reference cap

EMBC 2027 rules verified from the official 2027 pages on 2026-11-08
(CLAUDE.md section 8): **strict 4+1 pages** (up to 4 pages of technical
content, a 5th page reserved for acknowledgments and references only), **at
most 20 references**, **at most 35 authors**. `scripts/check_paper_layout.py`
checks the citation-key count, author count, and (once a PDF exists) the page
count and page-5 content against these limits.

## Every number is `\N{key}`

`main.tex` does `\input{evidence/numbers.tex}` near the top. Every result
number in the paper's prose, tables and captions must be written as
`\N{key}` (see `paper/evidence/numbers.md` for the usage and rounding
rules) -- never hand-typed. An undefined key is a deliberate hard compile
error. Run `scripts/check_paper_numbers.py <file>` after every editing
session; it must pass (or be explicitly reviewed with `--warn-only`) before
a draft is shared.

## Figures

Figures are copied into `paper/figures/` **only** by running
`scripts/copy_paper_figures.py <fig-id> [<fig-id> ...]` against the
VERIFIED inventory (`paper/evidence/inventory_verified.csv`, which records
actual panel counts, legend visibility, physical size and column fit for
every figure -- not `inventory.csv`, which only records what the figure
generation code claims). That script was written in Phase 4b but has not
been run yet: no figure has been selected for the paper, since several
figures (`fig7e`, `fig10`) are far too tall for any page at their current
size and need resizing or splitting first (see `inventory_verified.csv`'s
`visible_problem` column). Every copy is logged with its source path and
SHA-256 in `paper/figures/SOURCES.txt`.

## Claim map

`paper/evidence/claim_map.md` / `.csv` hold the mechanically-decided
SUPPORTED / PARTLY / NOT SUPPORTED / DESCRIPTIVE status for every candidate
claim (C1-C11) plus a "STATEMENTS THAT MUST NOT BE MADE" list. Draft every
results/discussion sentence against this file, not from memory of earlier
phase summaries.

## Phase 4c additions: authors, figures, table, Overleaf bundle

**Authors (names only, in this order):** Demian Sharwin Xaxa (first author,
proposed corresponding author), Aaron John Langan, Nithy Maria Mathew,
Gangavarapu V Vasu; all from [DEPARTMENT NAME], Karunya Institute of
Technology and Sciences, Coimbatore, India. **Emails are never stored in this
repository**: `main.tex` carries the placeholder `[EMAIL]`, and the real
addresses are added by hand in Overleaf at the very end.

**EMBC 2027 limits:** strict 4+1 pages, at most 20 references, at most 35 authors.

**Generated files** (re-create with `scripts/make_paper_figures.py`,
`scripts/make_paper_tables.py`, `scripts/make_overleaf_bundle.py`):

| file | final size |
|---|---|
| `figures/fig_main_learning_calibration.pdf/.png` | 7.16 x 2.4 in (double column), 3 panels |
| `figures/fig_main_robustness.pdf/.png` | 7.16 x 2.2 in (double column), 2 panels |
| `figures/SOURCES.txt` | source CSVs with SHA-256 and filters |
| `tables/table_main.tex` | `table*`, double column; every number is `\N{key}` |
| `tables/supp_slopes_per_dataset.tex/.csv`, `tables/supp_noise_comparison.tex/.csv` | supplementary (arXiv/repository only) |
| `overleaf_bundle.zip` | upload bundle (git-ignored) |

r"""Phase 4d tests: venue config, skeleton, new figures, Table I, numbers book additions, drafting sheet, bundle."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.calibration import ece_equal_mass
from analysis.constants import MODEL_COLOR, MODEL_MARKER
from scripts import build_drafting_sheet as ds
from scripts import make_overleaf_bundle as bundle
from scripts import make_paper_figures as figs
from scripts import make_paper_tables as tabs
from tests.test_phase4c import _scannable

PAPER = PROJECT_ROOT / "paper"
EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
AUTHORS = ["Demian Sharwin Xaxa", "Aaron John Langan", "Nithy Maria Mathew", "Gangavarapu V Vasu"]
AFFIL = "Department of Artificial Intelligence and Machine Learning,\n  Karunya Institute of Technology and Sciences\\\\\n  Coimbatore, India"
SECTION_ORDER = ["abstract", "introduction", "related_work", "datasets_protocol", "models_training", "results",
                 "discussion_and_limitations", "conclusion", "acknowledgments"]
NEW_FIGS = {**figs.FIGURE_SIZES, **figs.FIGURE_SIZES_4D}


# ---------------------------------------------------------------- venue config
def test_venue_json_keys_and_values() -> None:
    v = json.loads((PAPER / "venue.json").read_text(encoding="utf-8"))
    assert list(v) == ["venue", "template", "max_pages", "max_references", "reference_soft_limit", "max_authors",
                       "last_page_references_only", "notes"]
    assert v["max_pages"] == 6 and v["max_references"] is None and v["reference_soft_limit"] == 35
    assert v["max_authors"] == 35 and v["last_page_references_only"] is False


# ---------------------------------------------------------------- skeleton
def test_main_tex_inputs_in_order_and_old_files_gone() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    body = "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))
    assert re.findall(r"\\input\{(sections/[a-z_]+)\}", body) == [f"sections/{s}" for s in SECTION_ORDER]
    assert "\\input{evidence/numbers.tex}" in body
    assert "\\bibliographystyle{IEEEtran}" in body and "\\bibliography{references}" in body
    assert not (PAPER / "sections" / "methods.tex").exists()
    assert sorted(p.stem for p in (PAPER / "sections").glob("*.tex")) == sorted(SECTION_ORDER)


def test_author_block_and_affiliation() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert re.findall(r"\\IEEEauthorblockN\{([^}]*)\}", tex) == AUTHORS
    assert tex.count(AFFIL) == 4
    assert tex.count("[EMAIL]") == 4


def test_results_subsections_present() -> None:
    t = (PAPER / "sections" / "results.tex").read_text(encoding="utf-8")
    assert re.findall(r"\\subsection\{([^}]*)\}", t) == [
        "Discrimination and Learning Curves", "Calibration", "Robustness to Missing Values and Noise", "Explanation Agreement"]


def test_no_at_sign_outside_evidence_and_no_email_anywhere() -> None:
    for p in PAPER.rglob("*"):
        if not p.is_file() or p.name == "overleaf_bundle.zip" or "evidence" in p.relative_to(PAPER).parts:
            continue
        if p.suffix.lower() in (".pdf", ".png"):
            continue
        assert "@" not in p.read_text(encoding="utf-8"), p
    for p in PAPER.rglob("*"):
        if p.is_file() and p.name != "overleaf_bundle.zip":
            assert not EMAIL_RE.search(_scannable(p, p.read_bytes())), p
    z = PAPER / "overleaf_bundle.zip"
    if z.exists():
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                assert not EMAIL_RE.search(_scannable(Path(n), zf.read(n)) if n.endswith(".pdf") else zf.read(n)), n


def test_layout_checker_reports_four_authors() -> None:
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "check_paper_layout.py"), str(PAPER / "main.tex")],
                       capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Authors: 4" in r.stdout and "max_pages=6 (config venue.json)" in r.stdout


def test_number_checker_passes_on_paper_files() -> None:
    files = [PAPER / "main.tex", *(PAPER / "sections").glob("*.tex"), PAPER / "tables" / "table_main.tex", PAPER / "tables" / "table_datasets.tex"]
    for f in files:
        r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "check_paper_numbers.py"), str(f)],
                           capture_output=True, text=True, cwd=PROJECT_ROOT)
        assert r.returncode == 0, f"{f}\n{r.stdout[-800:]}"


# ---------------------------------------------------------------- figures
@pytest.fixture(scope="module")
def fig_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("figs4d")
    old = figs.OUT_DIR
    figs.OUT_DIR = out
    try:
        figs.main()
    finally:
        figs.OUT_DIR = old
    return out


@pytest.mark.parametrize("name", list(NEW_FIGS))
def test_files_pixel_size_and_no_type3(fig_dir: Path, name: str) -> None:
    w, h = NEW_FIGS[name]
    with Image.open(fig_dir / f"{name}.png") as im:
        assert abs(im.size[0] - round(w * 300)) <= 2 and abs(im.size[1] - round(h * 300)) <= 2
    assert b"/Type3" not in (fig_dir / f"{name}.pdf").read_bytes()


@pytest.mark.parametrize("name", list(figs.BUILDERS_4D))
def test_new_figure_fonts_and_models(name: str) -> None:
    fig, sources = figs.BUILDERS_4D[name]()
    fig.canvas.draw()
    for t in fig.findobj(Text):
        if t.get_text().strip() and t.get_visible():
            assert t.get_fontsize() >= figs.MIN_FONT_PT, (t.get_text(), t.get_fontsize())
    handles = {t.get_text(): h for h, t in zip(fig.legends[0].legend_handles, fig.legends[0].get_texts())}
    for m in figs.SHAP_MODELS:
        h = handles.get(f"{m} (b)") or handles[m]
        assert matplotlib.colors.to_hex(h.get_color()).upper() == MODEL_COLOR[m].upper()
        assert h.get_marker() == MODEL_MARKER[m]
    assert sources
    plt.close(fig)


def test_fig1_panel_c_uses_four_distinct_dataset_markers() -> None:
    fig, _ = figs.build_fig_learning_calibration()
    ax = fig.axes[2]
    markers = {ln.get_marker() for ln in ax.lines if ln.get_marker() not in (None, "", "None", " ")}
    assert markers == set(figs.DATASET_MARKER.values()) and len(markers) == 4
    plt.close(fig)


def test_dagger_rule_on_hand_example() -> None:
    assert figs.dagger_label("X", 0.60).endswith("\u2020")
    assert not figs.dagger_label("X", 0.599).endswith("\u2020")
    assert figs.dagger_label("X", 0.625).endswith("\u2020")


def test_reliability_helper_reproduces_equal_mass_ece_on_synthetic() -> None:
    rng = np.random.default_rng(42)
    p = rng.uniform(0, 1, 237)
    y = (rng.uniform(0, 1, 237) < p).astype(float)
    assert abs(figs.reliability_ece(y, p) - ece_equal_mass(y, p)) < 1e-12


def test_reliability_figure_reproduces_phase3b_ece(fig_dir: Path) -> None:
    assert figs.RELIABILITY_CHECK["max_abs_diff"] <= 1e-9


def test_sources_txt_lists_every_figure_with_current_hashes(fig_dir: Path) -> None:
    text = (fig_dir / "SOURCES.txt").read_text(encoding="utf-8")
    for name in NEW_FIGS:
        assert f"[{name}]" in text
    for p, h in re.findall(r"source: (\S+)\s+sha256=([0-9a-f]{64})", text):
        assert hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest() == h
    manifest = set((PROJECT_ROOT / "results" / "MANIFEST_sha256.txt").read_text(encoding="utf-8").splitlines())
    lines = re.findall(r"manifest: (.+)", text)
    assert len(lines) == 120 and all(ln in manifest for ln in lines)
    for ln in lines:  # the listed hashes equal the files on disk
        h, path = ln.split(None, 1)
        assert hashlib.sha256((PROJECT_ROOT / path.strip().lstrip("*")).read_bytes()).hexdigest() == h
    assert "\u2020" in text  # dagger definition for the caption


# ---------------------------------------------------------------- tables / numbers book
def test_table_i_macros_only_and_deterministic() -> None:
    numbers = tabs.load_numbers()
    tex = tabs.build_datasets_table(numbers)
    for k in re.findall(r"\\N\{([^}]+)\}", tex):
        assert k in numbers and not numbers[k]["undefined"]
    body = "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))
    body = re.sub(r"\\N\{[^}]+\}", "", body)
    for h in tabs.TABLE1_HEADER_TEXT:
        body = body.replace(h, "")
    assert not re.search(r"\d", body), re.findall(r".*\d.*", body)
    assert tex == tabs.build_datasets_table(numbers)
    assert (PAPER / "tables" / "table_datasets.tex").read_text(encoding="utf-8") == tex
    assert tex.count("\\N{") == 5 * 7  # per row: samples, features + numeric + categorical, positive, missing, min train


def test_table_main_is_byte_identical_to_phase4c() -> None:
    old = subprocess.run(["git", "show", "6a8d0ed:paper/tables/table_main.tex"], cwd=PROJECT_ROOT, capture_output=True)
    if old.returncode != 0:
        pytest.skip("git not available")
    assert (PAPER / "tables" / "table_main.tex").read_bytes().replace(b"\r\n", b"\n") == old.stdout.replace(b"\r\n", b"\n")


def test_supplementary_tables_exist() -> None:
    for n in ("supp_search_spaces", "supp_explanations"):
        assert (PAPER / "tables" / f"{n}.tex").exists() and (PAPER / "tables" / f"{n}.csv").exists()
    assert len(pd.read_csv(PAPER / "tables" / "supp_explanations.csv")) == 15
    assert "results/phase3d/top_features.csv" in (PAPER / "tables" / "supp_explanations.tex").read_text(encoding="utf-8")


def test_numbers_book_4d_keys_and_idempotent_rebuild() -> None:
    path = PAPER / "evidence" / "numbers.json"
    before = path.read_bytes()
    entries = json.loads(before)
    new = [e for e in entries if e["phase"] == "4d"]
    assert len(new) > 0 and all(not e["undefined"] and e["source_sha256"] for e in new)
    old = subprocess.run(["git", "show", "6a8d0ed:paper/evidence/numbers.json"], cwd=PROJECT_ROOT, capture_output=True)
    if old.returncode == 0:
        cur = {e["key"]: e for e in entries}
        for e in json.loads(old.stdout):
            assert cur[e["key"]] == e, e["key"]
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "build_numbers.py")], capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert r.returncode == 0, r.stderr[-500:]
    assert path.read_bytes() == before  # idempotent


# ---------------------------------------------------------------- drafting sheet
@pytest.fixture(scope="module")
def sheet() -> pd.DataFrame:
    return pd.read_csv(PAPER / "evidence" / "drafting_sheet.csv", dtype=str, keep_default_na=False)


def test_sheet_keys_exist_and_values_match(sheet: pd.DataFrame) -> None:
    numbers = {e["key"]: e for e in json.loads((PAPER / "evidence" / "numbers.json").read_text(encoding="utf-8"))}
    n_checked = 0
    for r in sheet.itertuples():
        if not r.keys:
            continue
        refs, vals = r.keys.split("|"), r.values.split("|")
        # values containing '|' cannot occur for numeric keys; text facts are resolved separately
        for ref in refs:
            if ref.startswith(("pf:", "meta:", "claude:")):
                assert ds._resolve(ref)[0] in r.values
            else:
                assert ref in numbers, ref
                assert numbers[ref]["formatted"] in vals and repr(numbers[ref]["value"]) in r.raw_values.split("|")
                n_checked += 1
    assert n_checked > 500


def test_sheet_claim_status_matches_claim_map(sheet: pd.DataFrame) -> None:
    cm = pd.read_csv(PAPER / "evidence" / "claim_map.csv", dtype=str, keep_default_na=False)
    truth = {(r.claim_id, r.item): r.status for r in cm.itertuples()}
    seen = 0
    for cell in sheet.claim_status:
        if cell == "n/a":
            continue
        for part in cell.split("; "):
            m = re.fullmatch(r"(C\d+):(.+)=(SUPPORTED|NOT SUPPORTED|PARTLY|DESCRIPTIVE)", part)
            assert m, part
            assert truth[(m.group(1), m.group(2))] == m.group(3)
            seen += 1
    assert seen > 100


def test_sheet_limitations_and_must_not_blocks_verbatim(sheet: pd.DataFrame) -> None:
    claude = (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    sec7 = claude.split("## 7.", 1)[1].split("## 8.", 1)[0].split("\n", 1)[1].strip("\n")
    lim = sheet[sheet.item_id.str.fullmatch(r"LIM-\d{3}")]
    assert "\n".join(lim["values"]) == sec7
    cm = (PAPER / "evidence" / "claim_map.md").read_text(encoding="utf-8")
    mn = cm.split("## STATEMENTS THAT MUST NOT BE MADE", 1)[1].strip("\n")
    must = sheet[sheet.item_id.str.fullmatch(r"LIM-M\d{2}")]
    assert "\n".join(must["values"]) == mn


# Digit rule for `fact_label`: after removing ids (C#, L#, M#, ALL5, top-5), every numeric token must be a
# fixed design constant (fractions, perturbation levels, split counts), never a result value.
DESIGN_NUMBERS = {"0.1", "0.25", "0.5", "1.0", "0.60", "30", "10", "50", "15", "1", "2", "5"}


def test_sheet_labels_contain_no_result_numbers(sheet: pd.DataFrame) -> None:
    for label in sheet.fact_label:
        label = re.sub(r"^[a-z]+\d{4}:", "", label)  # bibtex-key prefix of the related-work role list
        s = re.sub(r"Fig\. \d|section \d|ALL5|top-5|\b[CLM]\d+\b|\bf(?=\d)", " ", label)
        for tok in re.findall(r"\d+(?:\.\d+)?", s):
            assert tok in DESIGN_NUMBERS, (label, tok)


def test_sheet_is_deterministic_and_has_all_sections(sheet: pd.DataFrame) -> None:
    ds.build()
    csv_now = pd.DataFrame(ds.ROWS)[ds.COLUMNS]
    assert csv_now.astype(str).equals(sheet.astype(str).reset_index(drop=True))
    assert set(sheet.section) == set(ds.SECTIONS.values())
    assert sum(sheet.section == ds.SECTIONS["REL"]) == 17


# ---------------------------------------------------------------- bundle
def test_bundle_exact_file_set_and_ignored(tmp_path: Path) -> None:
    members = bundle.bundle_members()
    expected = {"main.tex", "references.bib", "evidence/numbers.tex", "tables/table_main.tex", "tables/table_datasets.tex",
                "BUNDLE_README.txt", *(f"sections/{s}.tex" for s in SECTION_ORDER),
                *(f"figures/{n}.pdf" for n in ("fig_main_learning_calibration", "fig_main_robustness", "fig_main_explanations", "fig_main_reliability"))}
    assert set(members) == expected
    assert bundle.check(members) == []
    for _tgt in bundle.referenced_targets(members):
        assert _tgt[2] in members
    for data in members.values():
        assert not EMAIL_RE.search(data)
    r = subprocess.run(["git", "check-ignore", "paper/overleaf_bundle.zip"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert r.returncode == 0
    r2 = subprocess.run(["git", "check-ignore", "paper/build/main.pdf"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert r2.returncode == 0

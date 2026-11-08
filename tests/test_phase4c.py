r"""Phase 4c tests: paper figures, main table, author block, Overleaf bundle."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.constants import MODEL_COLOR, MODEL_MARKER, MODELS
from scripts import make_overleaf_bundle as bundle
from scripts import make_paper_figures as figs
from scripts import make_paper_tables as tabs

PAPER = PROJECT_ROOT / "paper"
EMAIL_RE = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
AUTHORS = ["Demian Sharwin Xaxa", "Aaron John Langan", "Nithy Maria Mathew", "Gangavarapu V Vasu"]


# ---------------------------------------------------------------- figures
@pytest.fixture(scope="module")
def fig_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("figs")
    old = figs.OUT_DIR
    figs.OUT_DIR = out
    try:
        figs.main()
    finally:
        figs.OUT_DIR = old
    return out


@pytest.mark.parametrize("name", list(figs.FIGURE_SIZES))
def test_figure_files_and_pixel_size(fig_dir: Path, name: str) -> None:
    w, h = figs.FIGURE_SIZES[name]
    assert (fig_dir / f"{name}.pdf").exists()
    with Image.open(fig_dir / f"{name}.png") as im:
        assert abs(im.size[0] - round(w * 300)) <= 2
        assert abs(im.size[1] - round(h * 300)) <= 2


@pytest.mark.parametrize("name", list(figs.FIGURE_SIZES))
def test_pdf_has_no_type3(fig_dir: Path, name: str) -> None:
    assert b"/Type3" not in (fig_dir / f"{name}.pdf").read_bytes()


@pytest.mark.parametrize("name", list(figs.BUILDERS))
def test_font_sizes_and_legend_mapping(name: str) -> None:
    fig, sources = figs.BUILDERS[name]()
    fig.canvas.draw()
    for t in fig.findobj(Text):
        if t.get_text().strip() and t.get_visible():
            assert t.get_fontsize() >= figs.MIN_FONT_PT, (t.get_text(), t.get_fontsize())
    leg = fig.legends[0]
    handles = leg.legend_handles[: len(MODELS)]
    labels = [t.get_text() for t in leg.get_texts()][: len(MODELS)]
    assert labels == MODELS  # fixed model order
    for h, m in zip(handles, MODELS):
        assert matplotlib.colors.to_hex(h.get_color()).upper() == MODEL_COLOR[m].upper()
        assert h.get_marker() == MODEL_MARKER[m]
    assert sources
    plt.close(fig)


def test_sources_txt_hashes_current(fig_dir: Path) -> None:
    text = (fig_dir / "SOURCES.txt").read_text(encoding="utf-8")
    for name in figs.FIGURE_SIZES:
        assert f"[{name}]" in text
    pairs = re.findall(r"source: (\S+)\s+sha256=([0-9a-f]{64})", text)
    assert pairs
    used = {p for p, _ in pairs}
    for p, h in pairs:
        assert hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest() == h
    for needed in (figs.P3A_CI, figs.P3B_METRICS, figs.P3B_CI, figs.P3B_AVG, figs.P3C_AVG, figs.P3D_AVG):
        assert needed.as_posix() in used


# ---------------------------------------------------------------- table
def test_best_indices_hand_example() -> None:
    assert tabs.best_indices([0.80, 0.90, 0.85], "max") == [1]
    assert tabs.best_indices([0.30, 0.25, 0.40], "min") == [1]
    assert tabs.best_indices([1.30, 0.80, 1.10], "one") == [2]
    assert tabs.best_indices([0.9, 0.9, 0.8], "max") == [0, 1]  # exact ties all bold
    assert tabs.best_indices([0.9, 1.1], "one")[0] in (0, 1)


def test_main_table_cells_are_macros_only() -> None:
    numbers = tabs.load_numbers()
    tex, report = tabs.build_main_table(numbers)
    for k in re.findall(r"\\N\{([^}]+)\}", tex):
        assert k in numbers and not numbers[k]["undefined"]
    body = "\n".join(ln for ln in tex.splitlines() if not ln.lstrip().startswith("%"))
    body = re.sub(r"\\N\{[^}]+\}", "", body)
    for h in tabs.HEADER_TEXT:
        body = body.replace(h, "")
    body = re.sub(r"\\cmidrule\(lr\)\{[\d-]+\}", "", body)
    body = re.sub(r"\\multicolumn\{\d+\}", "", body)
    assert not re.search(r"\d", body), re.findall(r".*\d.*", body)
    assert len(report) == len(tabs.COLUMNS)
    assert tex.count("\\textbf{") == sum(len(r["best_models"]) for r in report)


def test_main_table_deterministic_and_matches_file() -> None:
    numbers = tabs.load_numbers()
    a, _ = tabs.build_main_table(numbers)
    b, _ = tabs.build_main_table(numbers)
    assert a == b
    assert (PAPER / "tables" / "table_main.tex").read_text(encoding="utf-8") == a


def test_missing_key_is_reported() -> None:
    numbers = {k: v for k, v in tabs.load_numbers().items() if k != "slope-xgb-nockd-f0.1"}
    with pytest.raises(KeyError, match="slope-xgb-nockd-f0.1"):
        tabs.build_main_table(numbers)


def test_numbers_book_untouched_by_phase4c() -> None:
    """Phase 4d follows a changed requirement: group "4d" legitimately ADDS keys, so the check is now
    'every key of the Phase 4c commit (6a8d0ed) is still present with an identical entry'."""
    try:
        head = subprocess.run(["git", "show", "6a8d0ed:paper/evidence/numbers.json"], cwd=PROJECT_ROOT,
                              capture_output=True, check=True).stdout
    except Exception:
        pytest.skip("git not available")
    cur = {e["key"]: e for e in json.loads((PAPER / "evidence" / "numbers.json").read_bytes())}
    for e in json.loads(head):
        assert cur[e["key"]] == e, e["key"]


# ---------------------------------------------------------------- authors / emails
def test_author_block_order_and_layout_checker() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    names = re.findall(r"\\IEEEauthorblockN\{([^}]*)\}", tex)
    assert names == AUTHORS
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "check_paper_layout.py"), str(PAPER / "main.tex")],
                       capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Authors: 4" in r.stdout


def _scannable(p: Path, data: bytes) -> bytes:
    """Bytes worth scanning for an email: compressed PNG/PDF payloads can match the regex by chance, so
    scan only PNG text metadata and PDF bytes outside stream...endstream (Phase 4d fix of a false positive)."""
    if p.suffix.lower() == ".png":
        with Image.open(p) as im:
            return " ".join(str(v) for v in im.info.values()).encode("utf-8", "replace")
    if p.suffix.lower() == ".pdf":
        return re.sub(rb"stream.*?endstream", b"", data, flags=re.S)
    return data


def test_no_at_sign_or_email_under_paper() -> None:
    for p in PAPER.rglob("*"):
        if not p.is_file() or p.name == "overleaf_bundle.zip":
            continue
        data = p.read_bytes()
        scan = _scannable(p, data)
        assert not EMAIL_RE.search(scan), f"email-like pattern in {p}"
        # Frozen Phase 4a/4b evidence files use '@' as "at fraction" in selector text; everything
        # else under paper/ (text files) must not contain '@' at all.
        if "evidence" not in p.relative_to(PAPER).parts and p.suffix in {".tex", ".bib", ".md", ".txt", ".csv"}:
            assert b"@" not in data, f"'@' in {p}"


# ---------------------------------------------------------------- bundle
def test_bundle_contents_and_paths(tmp_path: Path) -> None:
    z = bundle.build(tmp_path / "b.zip")
    with zipfile.ZipFile(z) as zf:
        names = set(zf.namelist())
        allowed = {"main.tex", "references.bib", "evidence/numbers.tex", "tables/table_main.tex", "BUNDLE_README.txt",
                   "tables/table_datasets.tex"}  # Phase 4d added Table I to the bundle
        allowed |= {f"sections/{p.name}" for p in (PAPER / "sections").glob("*.tex")}
        allowed |= {f"figures/{p.name}" for p in (PAPER / "figures").glob("*.pdf")}
        assert names == allowed
        assert not any(n.endswith(".png") for n in names)
        for n in names:
            assert not EMAIL_RE.search(zf.read(n)), n
            if not n.endswith(".pdf"):
                assert b"@" not in zf.read(n), n
        members = {n: zf.read(n) for n in names}
    targets = bundle.referenced_targets(members)
    assert any(k == "includegraphics" for _, k, _ in targets) and any(k == "input" for _, k, _ in targets)
    assert all(t in names for _, _, t in targets)


def test_bundle_refuses_email_and_missing_target() -> None:
    members = bundle.bundle_members()
    bad = dict(members)
    bad["main.tex"] = members["main.tex"] + b"\n% x@example.org\n"
    assert any("'@'" in p for p in bundle.check(bad))
    bad2 = dict(members)
    bad2["main.tex"] = members["main.tex"] + b"\n\\input{sections/nope}\n"
    assert any("nope" in p for p in bundle.check(bad2))


def test_zip_is_gitignored() -> None:
    assert "paper/overleaf_bundle.zip" in (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------- number checker
@pytest.mark.parametrize("rel", ["main.tex", "tables/table_main.tex"])
def test_check_paper_numbers_passes(rel: str) -> None:
    r = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "check_paper_numbers.py"), str(PAPER / rel)],
                       capture_output=True, text=True, cwd=PROJECT_ROOT)
    assert r.returncode == 0, r.stdout + r.stderr

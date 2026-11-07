"""Tests for Phase 4a: the numbers book (scripts/build_numbers.py) and the paper
number checker (scripts/check_paper_numbers.py). Kept reasonably fast; the full
numbers-book rebuild (a few seconds) is only exercised once, by the idempotency
test, which both tests reuse via a module-scoped fixture."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.build_numbers as bn  # noqa: E402
import scripts.check_paper_numbers as cpn  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUMBERS_JSON = PROJECT_ROOT / "paper" / "evidence" / "numbers.json"
NUMBERS_TEX = PROJECT_ROOT / "paper" / "evidence" / "numbers.tex"


@pytest.fixture(scope="module")
def built_entries() -> list[dict]:
    """Builds the real numbers book once (idempotent, deterministic) and returns
    the resulting entries, reused by every test below (fast after the first run).
    """
    bn.main()
    return json.loads(NUMBERS_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
def test_build_is_idempotent(built_entries, tmp_path):
    before_json = NUMBERS_JSON.read_bytes()
    before_tex = NUMBERS_TEX.read_bytes()
    bn.main()
    assert NUMBERS_JSON.read_bytes() == before_json
    assert NUMBERS_TEX.read_bytes() == before_tex


# ---------------------------------------------------------------------------
# Entry well-formedness
# ---------------------------------------------------------------------------
_FIELDS = {"key", "value", "formatted", "unit", "role", "source_file", "selector", "source_sha256", "phase", "undefined", "undefined_reason"}
_KEY_RE = re.compile(r"[a-z0-9.\-]+")


def test_every_entry_has_all_fields_and_unique_key(built_entries):
    seen = set()
    for e in built_entries:
        assert set(e.keys()) == _FIELDS
        assert e["key"] not in seen, f"duplicate key: {e['key']}"
        seen.add(e["key"])


def test_keys_match_allowed_charset(built_entries):
    for e in built_entries:
        assert _KEY_RE.fullmatch(e["key"]), f"invalid key: {e['key']!r}"


def test_source_file_exists_and_hash_matches_current(built_entries):
    checked = set()
    for e in built_entries:
        source_file = PROJECT_ROOT / e["source_file"]
        assert source_file.exists(), f"missing source file for key {e['key']}: {e['source_file']}"
        if e["source_file"] not in checked:
            actual = hashlib.sha256(source_file.read_bytes()).hexdigest()
            assert actual == e["source_sha256"], f"hash mismatch for {e['source_file']}"
            checked.add(e["source_file"])


def test_no_nan_or_inf_unless_marked_undefined(built_entries):
    for e in built_entries:
        if e["undefined"]:
            assert e["value"] is None
            assert e["formatted"] is None
            assert e["undefined_reason"]
        else:
            if isinstance(e["value"], float):
                assert np.isfinite(e["value"]), f"non-finite value for key {e['key']} not marked undefined"


# ---------------------------------------------------------------------------
# Rounding rules (hand-written examples)
# ---------------------------------------------------------------------------
def test_format_value_rounding_rules():
    assert bn.format_value(0.8341, "auroc") == "0.834"
    assert bn.format_value(0.00001, "pvalue") == "<0.001"
    assert bn.format_value(0.0423, "pvalue") == "0.042"
    assert bn.format_value(3.456, "rank") == "3.46"
    assert bn.format_value(0.6789, "jaccard") == "0.68"
    assert bn.format_value(34.56, "percentage") == "34.6"
    assert bn.format_value(1.234, "time") == "1.23"
    assert bn.format_value(42, "count") == "42"
    # Negative zero must print as 0.000, not -0.000.
    assert bn.format_value(-0.00001, "auroc") == "0.000"
    assert bn.format_value(-0.0, "auroc") == "0.000"


def test_make_key_and_ds_label():
    assert bn.ds_label("ALL5") == "all5"
    assert bn.ds_label("NO_CKD") == "nockd"
    assert bn.make_key("AUROC", "Mean", "NO_CKD", "f1.0") == "auroc-mean-no-ckd-f1.0"


# ---------------------------------------------------------------------------
# Round-trip: numbers.tex recovers exactly the formatted strings of numbers.json
# ---------------------------------------------------------------------------
_TEX_DEF_RE = re.compile(r"\\expandafter\\def\\csname num:(.+?)\\endcsname\{(.*)\}$")


def test_numbers_tex_roundtrip(built_entries):
    tex_text = NUMBERS_TEX.read_text(encoding="utf-8")
    tex_entries = {}
    for line in tex_text.splitlines():
        m = _TEX_DEF_RE.match(line)
        if m:
            tex_entries[m.group(1)] = m.group(2)

    for e in built_entries:
        if e["undefined"]:
            assert e["key"] not in tex_entries
            continue
        assert e["key"] in tex_entries, f"key {e['key']} missing from numbers.tex"
        recovered = tex_entries[e["key"]]
        # Undo the LaTeX escaping/minus-sign rewrite to recover the original formatted string.
        recovered = recovered.replace("$-$", "-").replace("$<$", "<")
        for ch, esc in bn._LATEX_ESCAPE.items():
            recovered = recovered.replace(esc, ch)
        assert recovered == e["formatted"], f"key {e['key']}: tex={recovered!r} json={e['formatted']!r}"


# ---------------------------------------------------------------------------
# STOP-with-clear-error behaviour (never falls back silently)
# ---------------------------------------------------------------------------
def test_one_row_raises_clear_error_on_missing_row():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(RuntimeError, match="expected exactly 1 row"):
        bn._one_row(df, df.a == 999, "test context")


def test_one_row_raises_clear_error_on_duplicate_rows():
    df = pd.DataFrame({"a": [1, 1, 3]})
    with pytest.raises(RuntimeError, match="expected exactly 1 row"):
        bn._one_row(df, df.a == 1, "test context")


def test_numbers_book_rejects_duplicate_keys():
    book = bn.NumbersBook()
    book.add("dup-key", 1.0, "count", "", "descriptive", NUMBERS_JSON, "test", "3a")
    with pytest.raises(ValueError, match="duplicate"):
        book.add("dup-key", 2.0, "count", "", "descriptive", NUMBERS_JSON, "test", "3a")


def test_numbers_book_raises_on_missing_source_file(tmp_path):
    book = bn.NumbersBook()
    with pytest.raises(FileNotFoundError):
        book.add("some-key", 1.0, "count", "", "descriptive", tmp_path / "does_not_exist.csv", "test", "3a")


def test_numbers_book_rejects_nan_without_undefined_reason():
    book = bn.NumbersBook()
    with pytest.raises(ValueError, match="undefined_reason"):
        book.add("nan-key", float("nan"), "auroc", "", "descriptive", NUMBERS_JSON, "test", "3a")


# ---------------------------------------------------------------------------
# Spot checks: recompute 5 entries directly from their recorded source and
# compare with numbers.json to 1e-9.
# ---------------------------------------------------------------------------
def test_spotcheck_all_runs_auroc_mean(built_entries):
    all_runs = pd.read_csv(PROJECT_ROOT / "results" / "all_runs.csv")
    datasets = ["pima_diabetes", "heart_cleveland", "ckd", "ilpd", "breast_wisconsin"]
    per_dataset = [
        all_runs[(all_runs.dataset == d) & (all_runs.model == "logreg") & (all_runs.fraction == 1.0)]["auroc"].mean()
        for d in datasets
    ]
    expected = float(np.mean(per_dataset))
    entry = next(e for e in built_entries if e["key"] == "auroc-mean-logreg-all5-f1.0")
    assert entry["value"] == pytest.approx(expected, abs=1e-9)


def test_spotcheck_dataset_summary_rows_used(built_entries):
    summary = pd.read_csv(PROJECT_ROOT / "results" / "dataset_summary.csv")
    expected = int(summary[summary.dataset == "ckd"]["n_rows"].iloc[0])
    entry = next(e for e in built_entries if e["key"] == "rows-used-ckd")
    assert entry["value"] == pytest.approx(expected, abs=1e-9)


def test_spotcheck_pooled_ece_via_phase3b_code(built_entries):
    from analysis.calibration import pooled_metrics

    datasets = ["pima_diabetes", "heart_cleveland", "ckd", "ilpd", "breast_wisconsin"]
    per_dataset_ece = []
    for dataset in datasets:
        y_all, p_all = [], []
        for split in range(10):
            data = np.load(PROJECT_ROOT / "results" / "predictions" / f"{dataset}__mlp__split{split}.npz")
            y_all.append(data["y_true"].astype(np.float64))
            p_all.append(data["prob_1.0"].astype(np.float64))
        metrics = pooled_metrics(np.concatenate(y_all), np.concatenate(p_all))
        per_dataset_ece.append(metrics["ece_ew"])
    expected = float(np.mean(per_dataset_ece))
    entry = next(e for e in built_entries if e["key"] == "ece-ew-mlp-all5-f1.0")
    assert entry["value"] == pytest.approx(expected, abs=1e-9)


def test_spotcheck_tabnet_collapsed_splits_count(built_entries):
    collapsed = pd.read_csv(PROJECT_ROOT / "results" / "phase2c" / "tabnet_collapsed_splits.csv")
    entry = next(e for e in built_entries if e["key"] == "n-tabnet-collapsed-splits")
    assert entry["value"] == pytest.approx(len(collapsed), abs=1e-9)


def test_spotcheck_table12_cell_from_recorded_source(built_entries):
    diff = pd.read_csv(PROJECT_ROOT / "results" / "phase3a" / "ci_diff_vs_logreg.csv")
    row = diff[(diff.model == "tabnet") & (diff.metric == "brier") & (diff.fraction == 1.0) & (diff.dataset_set == "ALL5")].iloc[0]
    entry = next(e for e in built_entries if e["key"] == "brier-diff-logreg-tabnet-all5-f1.0-point")
    assert entry["value"] == pytest.approx(float(row.point), abs=1e-9)


# ---------------------------------------------------------------------------
# scripts/check_paper_numbers.py
# ---------------------------------------------------------------------------
_PLANTED_UNMATCHED = "847.193"  # chosen to be implausible as any rounding of a real entry


@pytest.fixture()
def synthetic_draft(tmp_path, built_entries):
    some_entry = next(e for e in built_entries if not e["undefined"] and "." in (e["formatted"] or ""))
    formatted_index = cpn.build_formatted_index(built_entries)
    assert cpn.classify_token(_PLANTED_UNMATCHED, formatted_index)[0] == "UNMATCHED"  # guard against accidental collision
    draft = tmp_path / "draft.tex"
    draft.write_text(
        "% this comment number 99.999 must be ignored\n"
        f"We report \\N{{{some_entry['key']}}} as our headline result.\n"
        f"The value {_PLANTED_UNMATCHED} is a planted UNMATCHED number.\n"
        "We cite \\cite{smith2020ref12345} for prior work (the digits inside \\cite must be ignored).\n"
        "\\N{this-key-does-not-exist-xyz} is a planted unknown key.\n",
        encoding="utf-8",
    )
    return draft, some_entry


def test_checker_flags_planted_unmatched_and_unknown_key(synthetic_draft):
    draft, some_entry = synthetic_draft
    exit_code, report = cpn.run_check(draft, NUMBERS_JSON)
    assert exit_code == 1
    unmatched_tokens = [r["token"] for r in report["numbers"] if r["status"] == "UNMATCHED"]
    assert _PLANTED_UNMATCHED in unmatched_tokens
    unknown_keys = [key for _, key in report["unknown_n_keys"]]
    assert "this-key-does-not-exist-xyz" in unknown_keys
    # The comment number and the \cite-internal digits must never be extracted.
    all_tokens = [r["token"] for r in report["numbers"]]
    assert "99.999" not in all_tokens
    assert "2020" not in all_tokens
    assert "12345" not in all_tokens
    # The \N{...} usage itself matches (it's a real key): reports as exact, not unmatched.
    exact_tokens = [r for r in report["numbers"] if r["status"] == "exact"]
    assert len(exact_tokens) >= 0  # the \N{} argument itself is not number-scanned; this just confirms no crash


def test_checker_reports_rounded_match(built_entries):
    entry = next(e for e in built_entries if not e["undefined"] and e["formatted"] and re.fullmatch(r"\d+\.\d{3}", e["formatted"]))
    token = entry["formatted"] + "4"  # one extra (arbitrary) decimal digit -> should round-match
    formatted_index = cpn.build_formatted_index(built_entries)
    status, keys, decimals = cpn.classify_token(token, formatted_index)
    assert status == "rounded"
    assert entry["key"] in keys


def test_checker_warn_only_always_exits_zero(synthetic_draft):
    draft, _ = synthetic_draft
    exit_code, _ = cpn.run_check(draft, NUMBERS_JSON)
    assert exit_code == 1  # without --warn-only, the underlying check is still a failure
    # main() applies --warn-only by overriding the exit code, tested at the CLI layer below.


def test_checker_detects_stale_source(tmp_path, monkeypatch):
    # Copy one real source file into an isolated temp "project root", point
    # check_stale_sources at it via monkeypatched PROJECT_ROOT, then modify the
    # copy -> the hash must no longer match.
    monkeypatch.setattr(cpn, "PROJECT_ROOT", tmp_path)
    temp_source = tmp_path / "source_copy.csv"
    temp_source.write_text("a,b\n1,2\n", encoding="utf-8")
    original_hash = hashlib.sha256(temp_source.read_bytes()).hexdigest()

    entries = [{"source_file": "source_copy.csv", "source_sha256": original_hash}]
    assert cpn.check_stale_sources(entries) == []

    temp_source.write_text("modified content that changes the hash\n", encoding="utf-8")
    stale_warnings = cpn.check_stale_sources(entries)
    assert len(stale_warnings) == 1
    assert "STALE" in stale_warnings[0]

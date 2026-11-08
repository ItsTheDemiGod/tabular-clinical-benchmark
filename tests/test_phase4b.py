"""Tests for Phase 4b: the new group-8 numbers-book quantities
(scripts/build_numbers.py::build_group8_phase4b_claims) and the claim map
(scripts/build_claim_map.py). Kept reasonably fast: the full numbers-book
rebuild is exercised once via a module-scoped fixture, reused by every test.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.build_numbers as bn  # noqa: E402
import scripts.build_claim_map as bcm  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUMBERS_JSON = PROJECT_ROOT / "paper" / "evidence" / "numbers.json"
CLAIM_MAP_CSV = PROJECT_ROOT / "paper" / "evidence" / "claim_map.csv"
CLAIM_MAP_DETAIL = PROJECT_ROOT / "paper" / "evidence" / "claim_map_detail.json"


@pytest.fixture(scope="module")
def pre_existing_entries() -> dict[str, dict]:
    """numbers.json content BEFORE this module's rebuild (whatever is on disk at
    collection time), keyed by key, for the key-preservation check below."""
    return {e["key"]: e for e in json.loads(NUMBERS_JSON.read_text(encoding="utf-8"))}


@pytest.fixture(scope="module")
def built_entries(pre_existing_entries, rebuilt_numbers_dir, tmp_path_factory) -> dict[str, dict]:
    """The numbers book (includes group 8) rebuilt into a temp dir, then the claim map built into another
    temp dir (Phase 4e: nothing under paper/evidence is overwritten); returns the entries keyed by key."""
    entries = json.loads((rebuilt_numbers_dir / "numbers.json").read_text(encoding="utf-8"))
    out = tmp_path_factory.mktemp("claim_map")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bcm, "OUTPUT_DIR", out)  # NUMBERS_JSON_PATH stays the committed numbers.json
        bcm.main()
    for name in ("claim_map.csv", "claim_map.md", "claim_map_detail.json"):
        assert (out / name).read_bytes() == (PROJECT_ROOT / "paper" / "evidence" / name).read_bytes(), name
    return {e["key"]: e for e in entries}


# ---------------------------------------------------------------------------
# Numbers-book rebuild: idempotence + pre-existing-key preservation
# ---------------------------------------------------------------------------
def test_rebuild_is_idempotent(built_entries, tmp_path):
    before = NUMBERS_JSON.read_bytes()  # committed
    bn.main(out_dir=tmp_path)
    assert (tmp_path / "numbers.json").read_bytes() == before


def test_rebuild_preserves_all_pre_existing_keys_and_values(pre_existing_entries, built_entries):
    missing = set(pre_existing_entries) - set(built_entries)
    assert not missing, f"pre-existing keys dropped by the rebuild: {sorted(missing)[:10]}"
    changed = [k for k in pre_existing_entries if built_entries[k] != pre_existing_entries[k]]
    assert not changed, f"pre-existing keys changed value by the rebuild: {changed[:10]}"


def test_rebuild_group8_entries_are_exactly_phase_4b(built_entries):
    """Structural check, independent of what was already on disk at collection
    time (bn.main() may have been run earlier in this session too): every
    entry tagged phase=="4b" was produced by build_group8_phase4b_claims, and
    there is at least one such entry."""
    phase_4b_entries = [k for k, e in built_entries.items() if e["phase"] == "4b"]
    assert len(phase_4b_entries) > 0, "expected build_group8_phase4b_claims to add phase==4b keys"
    assert all(k.startswith(("auroc-dod-", "fit-time-ratio-", "logloss-diff-leader-2nd-")) for k in phase_4b_entries)


def test_group8_keys_have_valid_provenance(built_entries):
    new_keys = [k for k, e in built_entries.items() if e["phase"] == "4b"]
    assert len(new_keys) > 0
    for k in new_keys:
        e = built_entries[k]
        assert not e["undefined"]
        assert e["formatted"] is not None
        assert len(e["source_sha256"]) == 64
        assert (PROJECT_ROOT / e["source_file"]).exists()


# ---------------------------------------------------------------------------
# Status rule (nb_status): every branch
# ---------------------------------------------------------------------------
def test_status_supported_both_sets():
    assert bcm.nb_status(True, True, True, True) == "SUPPORTED"


def test_status_not_supported_both_ci_include_0():
    assert bcm.nb_status(False, False, True, True) == "NOT SUPPORTED"


def test_status_partly_ci_excludes_0_only_in_all5():
    assert bcm.nb_status(True, False, True, True) == "PARTLY"


def test_status_partly_ci_excludes_0_only_in_nockd():
    assert bcm.nb_status(False, True, True, True) == "PARTLY"


def test_status_partly_sign_agreement_fails_in_all5_only():
    # CI excludes 0 in both, but sign agreement fails in ALL5 -> only NO_CKD "supported" side -> PARTLY.
    assert bcm.nb_status(True, True, False, True) == "PARTLY"


def test_status_partly_sign_agreement_fails_in_nockd_only():
    assert bcm.nb_status(True, True, True, False) == "PARTLY"


def test_status_not_supported_ci_excludes_0_but_both_sign_fail():
    # CI excludes 0 everywhere but sign agreement fails in both -> neither side counts as supported.
    assert bcm.nb_status(True, True, False, False) == "NOT SUPPORTED"


def test_sign_ok_threshold_boundaries():
    # ALL5: need >=3/5 agreeing with the point estimate's sign.
    assert bcm.sign_ok(n_positive=1, n_total=5, point=-0.01) is True   # negative point: 5-1=4 agree -> True
    assert bcm.sign_ok(n_positive=3, n_total=5, point=-0.01) is False  # 5-3=2 agree -> False
    assert bcm.sign_ok(n_positive=3, n_total=5, point=0.01) is True    # positive point: 3 agree -> True
    assert bcm.sign_ok(n_positive=2, n_total=5, point=0.01) is False   # 2 agree -> False
    # NO_CKD: need >=3/4.
    assert bcm.sign_ok(n_positive=1, n_total=4, point=-0.01) is True   # 4-1=3 agree -> True
    assert bcm.sign_ok(n_positive=2, n_total=4, point=-0.01) is False  # 4-2=2 agree -> False


# ---------------------------------------------------------------------------
# Diff-of-differences bootstrap: reproducibility + a hand-computed synthetic case
# ---------------------------------------------------------------------------
def _make_synthetic_all_runs() -> pd.DataFrame:
    """A tiny synthetic all_runs-shaped frame: 1 dataset ('toy'), 2 models
    ('m', 'logreg'), 2 fractions (0.1, 1.0), 10 splits each, fixed values so the
    diff-of-differences can be hand-computed."""
    rows = []
    # At f0.1: m = 0.60 constant, logreg = 0.50 constant -> diff = +0.10
    # At f1.0: m = 0.55 constant, logreg = 0.55 constant -> diff = 0.00
    # diff-of-diff = 0.10 - 0.00 = +0.10 exactly, with ZERO bootstrap variance
    # (every split has the identical value, so every resample gives the same mean).
    for split in range(10):
        rows.append({"dataset": "toy", "model": "m", "fraction": 0.1, "split_idx": split, "metric": 0.60})
        rows.append({"dataset": "toy", "model": "logreg", "fraction": 0.1, "split_idx": split, "metric": 0.50})
        rows.append({"dataset": "toy", "model": "m", "fraction": 1.0, "split_idx": split, "metric": 0.55})
        rows.append({"dataset": "toy", "model": "logreg", "fraction": 1.0, "split_idx": split, "metric": 0.55})
    df = pd.DataFrame(rows)
    df = df.rename(columns={"metric": "auroc"})
    return df


def test_diff_of_diff_bootstrap_matches_hand_computation():
    df = _make_synthetic_all_runs()
    point, lo, hi, per_ds = _diff_of_diff_via_build_numbers(df)
    assert point == pytest.approx(0.10, abs=1e-9)
    assert lo == pytest.approx(0.10, abs=1e-9)
    assert hi == pytest.approx(0.10, abs=1e-9)
    assert per_ds == [pytest.approx(0.10, abs=1e-9)]


def _diff_of_diff_via_build_numbers(df: pd.DataFrame):
    return bn._diff_of_diff_bootstrap(df, "m", "auroc", ["toy"], seed=42, b=200)


def test_diff_of_diff_bootstrap_is_reproducible_with_same_seed():
    df = _make_synthetic_all_runs()
    # Make the series non-degenerate (varying per split) so the bootstrap CI has width.
    rng = np.random.default_rng(0)
    df = df.copy()
    df["auroc"] = df["auroc"] + rng.normal(0, 0.01, size=len(df))
    r1 = bn._diff_of_diff_bootstrap(df, "m", "auroc", ["toy"], seed=42, b=500)
    r2 = bn._diff_of_diff_bootstrap(df, "m", "auroc", ["toy"], seed=42, b=500)
    assert r1 == r2  # identical point/ci_lo/ci_hi/per_ds for the same seed


def test_diff_of_diff_bootstrap_different_seed_can_differ_in_ci():
    df = _make_synthetic_all_runs()
    rng = np.random.default_rng(0)
    df = df.copy()
    df["auroc"] = df["auroc"] + rng.normal(0, 0.05, size=len(df))
    _, lo1, hi1, _ = bn._diff_of_diff_bootstrap(df, "m", "auroc", ["toy"], seed=42, b=500)
    _, lo2, hi2, _ = bn._diff_of_diff_bootstrap(df, "m", "auroc", ["toy"], seed=7, b=500)
    # Not asserting they MUST differ (could coincide), just that both are finite and ci_lo<=ci_hi.
    assert lo1 <= hi1
    assert lo2 <= hi2


# ---------------------------------------------------------------------------
# Claim map: every entry references only existing numbers.json keys
# ---------------------------------------------------------------------------
def test_claim_map_files_exist(built_entries):
    assert CLAIM_MAP_CSV.exists()
    assert CLAIM_MAP_DETAIL.exists()
    assert (PROJECT_ROOT / "paper" / "evidence" / "claim_map.md").exists()


def test_claim_map_g_helper_raises_on_unknown_key(built_entries):
    with pytest.raises(KeyError):
        bcm.g(built_entries, "this-key-does-not-exist-xyz")


def test_claim_map_detail_c1_c2_c3_keys_exist_in_numbers_book(built_entries):
    """Re-derives every numbers.json key referenced by C1/C2/C3's construction
    and confirms each exists (the functions would already have raised if not,
    since the fixture ran build_claim_map.main() successfully; this test makes
    the key set explicit and re-checks it independently)."""
    for frac_tag in ("f1.0", "f0.1"):
        for model in bcm.NON_LOGREG_MODELS:
            for set_name, tag in (("ALL5", "all5"), ("NO_CKD", "nockd")):
                base = f"auroc-diff-logreg-{model}-{tag}-{frac_tag}"
                for suffix in ("-point", "-ci-lo", "-ci-hi", "-npos"):
                    assert f"{base}{suffix}" in built_entries, f"missing {base}{suffix}"
    for model in bcm.NON_LOGREG_MODELS:
        for set_name, tag in (("ALL5", "all5"), ("NO_CKD", "nockd")):
            base = f"auroc-dod-f0.1-f1.0-{model}-{tag}"
            for suffix in ("-point", "-ci-lo", "-ci-hi", "-nsign-agree", "-ntotal"):
                assert f"{base}{suffix}" in built_entries, f"missing {base}{suffix}"


def test_claim_map_csv_has_all_11_claim_ids(built_entries):
    df = pd.read_csv(CLAIM_MAP_CSV)
    claim_ids = set(df["claim_id"].unique())
    expected = {f"C{i}" for i in range(1, 12)}
    assert expected <= claim_ids, f"missing claim ids: {expected - claim_ids}"


def test_claim_map_csv_statuses_are_valid(built_entries):
    df = pd.read_csv(CLAIM_MAP_CSV)
    valid = {"SUPPORTED", "PARTLY", "NOT SUPPORTED", "DESCRIPTIVE"}
    assert set(df["status"].unique()) <= valid


def test_claim_map_detail_exceptions_present_for_c1_and_c2(built_entries):
    detail = json.loads(CLAIM_MAP_DETAIL.read_text(encoding="utf-8"))
    assert "exceptions" in detail["C1"]
    assert "exceptions" in detail["C2"]
    for e in detail["C1"]["exceptions"] + detail["C2"]["exceptions"]:
        assert e["excess"] > 0
        assert e["model_mean_auroc"] > e["logreg_mean_auroc"]


def test_auroc_exceeds_logreg_cells_matches_direct_recomputation():
    all_runs = pd.read_csv(PROJECT_ROOT / "results" / "all_runs.csv")
    found = bcm.auroc_exceeds_logreg_cells(all_runs, 1.0)
    # Independent recomputation using a different (non-vectorized) loop style.
    expected = []
    for ds in ["pima_diabetes", "heart_cleveland", "ckd", "ilpd", "breast_wisconsin"]:
        lr = all_runs[(all_runs.dataset == ds) & (all_runs.model == "logreg") & (all_runs.fraction == 1.0)]["auroc"].mean()
        for m in ["rf", "xgb", "lgbm", "mlp", "ftt", "tabnet"]:
            mv = all_runs[(all_runs.dataset == ds) & (all_runs.model == m) & (all_runs.fraction == 1.0)]["auroc"].mean()
            if mv > lr:
                expected.append((ds, m))
    found_pairs = [(e["dataset"], e["model"]) for e in found]
    assert sorted(found_pairs) == sorted(expected)


def test_claim_map_c9_distinct_top5_matches_numbers_book(built_entries):
    detail = json.loads(CLAIM_MAP_DETAIL.read_text(encoding="utf-8"))
    bw = detail["C9"]["breast_wisconsin_distinct_top5_sets_of_10"]
    for model in ("logreg", "rf", "mlp"):
        key = f"n-distinct-top5-breast-wisconsin-{model}"
        assert bw[model] == int(built_entries[key]["value"])


def test_claim_map_c4_matches_corrected_ttest_csv(built_entries):
    detail = json.loads(CLAIM_MAP_DETAIL.read_text(encoding="utf-8"))
    corrected = pd.read_csv(PROJECT_ROOT / "results" / "phase3a" / "corrected_ttest.csv")
    assert detail["C4"]["n_total_cells"] == len(corrected)
    assert detail["C4"]["n_sig_cells"] == int((corrected.p_holm < 0.05).sum())

"""Shared pytest helpers and guards (Phase 4e).

* `scannable`      -- bytes of a file worth scanning for an email address (was a helper imported from
                      another test file; no test file may import another test file).
* `rebuilt_numbers_dir` -- the numbers book rebuilt ONCE per session into a temporary directory
                      (never into paper/evidence); tests compare it with the committed files.
* `_evidence_untouched_guard` -- autouse session guard: SHA-256 of every file under paper/evidence is
                      taken at session start and compared at session end; a test that overwrites a
                      committed evidence file makes the session end with an error.
"""
from __future__ import annotations

import hashlib
import re
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
EVIDENCE = PROJECT_ROOT / "paper" / "evidence"


def evidence_hashes(root: Path = EVIDENCE) -> dict[str, str]:
    """{relative path: sha256} for every file under `root`."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def scannable_bytes(p: Path, data: bytes) -> bytes:
    """Bytes worth scanning for an email: compressed PNG/PDF payloads can match the regex by chance, so
    scan only PNG text metadata and PDF bytes outside stream...endstream."""
    if p.suffix.lower() == ".png":
        from PIL import Image
        with Image.open(p) as im:
            return " ".join(str(v) for v in im.info.values()).encode("utf-8", "replace")
    if p.suffix.lower() == ".pdf":
        return re.sub(rb"stream.*?endstream", b"", data, flags=re.S)
    return data


@pytest.fixture(scope="session")
def scannable():
    return scannable_bytes


@pytest.fixture(scope="session")
def rebuilt_numbers_dir(tmp_path_factory) -> Path:
    """The whole numbers book (numbers.*, protocol_facts.*, inventory.*) built into a temp dir."""
    import scripts.build_numbers as bn
    out = tmp_path_factory.mktemp("numbers_book")
    t0 = time.time()
    bn.main(out_dir=out)
    print(f"[conftest] numbers book rebuilt into a temp dir in {time.time() - t0:.1f}s")
    return out


@pytest.fixture(scope="session", autouse=True)
def _evidence_untouched_guard():
    before = evidence_hashes()
    yield
    after = evidence_hashes()
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert not changed, f"tests modified committed files under paper/evidence: {changed}"


def git_show(rev_path: str) -> bytes:
    """`git show <rev>:<path>`; when it cannot run, SKIP with the exact reason (Phase 4e: a bare
    'git not available' message hid why a test was skipped)."""
    import subprocess
    try:
        r = subprocess.run(["git", "show", rev_path], cwd=PROJECT_ROOT, capture_output=True)
    except OSError as exc:
        pytest.skip(f"git executable not runnable ({exc}); needs `git show {rev_path}` (repository history)")
    if r.returncode != 0:
        pytest.skip(f"`git show {rev_path}` failed with exit code {r.returncode}: {r.stderr.decode(errors='replace').strip()[:200]} "
                    "(needs the repository history, e.g. not a shallow clone)")
    return r.stdout


@pytest.fixture(scope="session")
def git_show_fixture():
    return git_show

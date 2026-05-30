"""Unit tests for the data-download infrastructure.

Tests focus on the SHA-256 verification helper and the CLI dry-run path.
Real HTTP downloads are NOT exercised; those require operator-supplied
URLs and are gated by the ``--accept-license`` flag.
"""

from __future__ import annotations

import hashlib
import json
import sys
import textwrap
from pathlib import Path

import pytest

# scripts/ is not a regular package import; add the repo root to sys.path so
# the ``scripts.data_download`` import works inside the test process.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.data_download import _hash_utils  # noqa: E402
from scripts.data_download import download_deepmimo  # noqa: E402


def test_hash_verifier_accepts_match_when_digest_correct(tmp_path: Path) -> None:
    """verify_sha256 returns True for a matching digest.

    Requirement R11.3 (manifest verification): downloads must be matched
    against committed hashes before being trusted.
    """
    payload = b"mv-patp-mimo test payload"
    artefact = tmp_path / "fixture.bin"
    artefact.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    assert _hash_utils.verify_sha256(artefact, digest) is True


def test_hash_verifier_rejects_mismatch_when_digest_wrong(tmp_path: Path) -> None:
    """verify_sha256 returns False on any digest divergence.

    Requirement R11.3 (manifest verification): a mismatched download must
    not pass.
    """
    artefact = tmp_path / "fixture.bin"
    artefact.write_bytes(b"original")
    wrong_digest = "0" * 64  # canonical lower-case hex but obviously wrong
    assert _hash_utils.verify_sha256(artefact, wrong_digest) is False


def test_hash_verifier_passes_tbd_placeholder_when_hash_unknown(
    tmp_path: Path,
) -> None:
    """A 'TBD' placeholder lets the dry-run / skeleton flow succeed.

    Requirement R10 (real-data pipeline): until the operator runs the
    download once and records the canonical hash, downstream tools should
    not fail just because the hash is unknown. The TBD path emits a warning
    instead of an error.
    """
    artefact = tmp_path / "fixture.bin"
    artefact.write_bytes(b"placeholder")
    assert _hash_utils.verify_sha256(artefact, "TBD") is True


def test_load_expected_hashes_has_all_dataset_keys() -> None:
    """expected_hashes.json must declare keys for every supported dataset.

    Requirement R10 (real-data benchmarks): the manifest is the index that
    download scripts consult.
    """
    hashes = _hash_utils.load_expected_hashes()
    for key in ("deepmimo", "waird", "radioml"):
        assert key in hashes, f"missing dataset key '{key}'"


def test_deepmimo_dry_run_prints_url_when_no_artefact(
    capsys, tmp_path: Path
) -> None:
    """download_deepmimo --dry-run validates flags without hitting the network.

    Requirement R10.1 (DeepMIMO pipeline): the dry-run path must succeed
    even when no URL is supplied, so CI can verify the script imports
    cleanly.
    """
    rc = download_deepmimo.main([
        "--scenario", "asu_campus_3p5",
        "--output-dir", str(tmp_path),
        "--accept-license",
        "--dry-run",
    ])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "DRY-RUN" in captured
    assert "asu_campus_3p5" in captured


def test_deepmimo_requires_license_acknowledgement(tmp_path: Path) -> None:
    """Without --accept-license the script exits with code 2.

    Requirement R10.1 (licence compliance): research datasets carry their
    own terms of use; we surface them explicitly.
    """
    rc = download_deepmimo.main([
        "--scenario", "asu_campus_3p5",
        "--output-dir", str(tmp_path),
        "--dry-run",
    ])
    assert rc == 2

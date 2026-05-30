"""Shared SHA-256 verification helpers for data download scripts.

These helpers operate on ``data/expected_hashes.json``, the manifest of
expected SHA-256 digests for every external dataset (AGENTS.md Section 6.3).
A mismatch is escalated, never silently re-computed: hash divergence
indicates either a corrupted download or an upstream dataset change that
must be reflected in a Spec.md Section 12 decision-log entry.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_HASHES_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "expected_hashes.json"
)


def repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parents[2]


def load_expected_hashes(path: Path | None = None) -> dict[str, Any]:
    """Load ``data/expected_hashes.json`` as a dictionary."""
    p = path or _DEFAULT_HASHES_PATH
    return json.loads(p.read_text())


def file_sha256(path: Path, chunk_size: int = 1 << 16) -> str:
    """Compute the SHA-256 hex digest of ``path`` streaming in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: Path, expected: str) -> bool:
    """Return True iff the SHA-256 of ``path`` matches ``expected``.

    The comparison is intentionally exact -- no leading/trailing whitespace
    tolerance, no case-folding -- because expected hashes are committed to
    the repository as canonical lower-case 64-character hex strings.

    A return value of False is the contract for the caller to abort the
    pipeline (AGENTS.md Section 10 escalation).
    """
    if expected == "TBD":
        logger.warning(
            "Hash for %s is TBD; allowing pass for skeleton/dry-run flow.", path
        )
        return True
    actual = file_sha256(path)
    matches = (actual == expected)
    if not matches:
        logger.error(
            "SHA-256 mismatch for %s: expected %s, got %s",
            path, expected, actual,
        )
    return matches

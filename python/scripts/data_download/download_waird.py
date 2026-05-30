"""Download the WAIR-D dataset for cross-environment MIMO generalisation tests.

WAIR-D (Huangfu et al. 2022) is the secondary real-data MIMO benchmark per
Spec.md Section 6.1 / Requirement R10.3. It contains multiple wireless
environments; we require at least three for the cross-environment
generalisation claim.

Same operator-supplies-URL pattern as :mod:`download_deepmimo`.

Satisfies Requirement R10.3.
Task: pipeline support for Section 17.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

from scripts.data_download._hash_utils import (
    load_expected_hashes,
    repo_root,
    verify_sha256,
)

logger = logging.getLogger(__name__)

LICENCE_NOTICE = (
    "WAIR-D is distributed under a research licence by Huangfu et al. "
    "Pass --accept-license to acknowledge."
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download WAIR-D environments with SHA-256 verification.",
    )
    parser.add_argument("--environment", required=True,
                        help="Environment identifier (operator-supplied).")
    parser.add_argument("--url", default=None)
    parser.add_argument("--output-dir", type=Path,
                        default=repo_root() / "data" / "waird")
    parser.add_argument("--accept-license", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.accept_license:
        logger.error(LICENCE_NOTICE)
        return 2

    target_filename = f"{args.environment}.h5"
    target_path = args.output_dir / target_filename

    expected = load_expected_hashes().get("waird", {}).get(target_filename)
    if expected is None:
        logger.error(
            "No entry for waird/%s in data/expected_hashes.json.",
            target_filename,
        )
        return 3

    if args.dry_run:
        print(f"DRY-RUN: would download {args.environment} to {target_path}")
        print(f"DRY-RUN: expected SHA-256 = {expected}")
        print(f"DRY-RUN: URL = {args.url or '<not provided>'}")
        return 0

    if not args.url:
        logger.error("--url is required when not in --dry-run mode.")
        return 4

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", args.url, target_path)
    urllib.request.urlretrieve(args.url, target_path)  # noqa: S310

    if not verify_sha256(target_path, expected):
        logger.error("Hash verification failed; deleting %s", target_path)
        target_path.unlink(missing_ok=True)
        return 5

    logger.info("Verified SHA-256 for %s", target_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Download the DeepMIMO dataset for primary real-data MIMO benchmarks.

DeepMIMO (Alkhateeb 2019) is the primary real-data MIMO benchmark per
Spec.md Section 6.1 / Requirement R10.1, R10.2. We use the O1 (28 GHz
outdoor street canyon) and I1 (2.5 GHz indoor) ray-tracing scenarios.

Superseded for the v4 scenarios (checked 2026-11-04): no registration is
needed; ``deepmimo.download(name)`` (deepmimo 4.0.5) fetches
``https://deepmimo.net/api/download/secure?filename=<name>.zip``, and
``export_deepmimo_channels.py`` writes the channel tables the loader reads
(see ``$RESEARCH_DATA_ROOT/store/deepmimo-v4/SOURCE.md``). Legacy URL mode:
the operator provides the resolved URL via ``--url``. The script verifies the SHA-256 of
the downloaded file against ``data/expected_hashes.json`` and, on success,
writes it under ``data/deepmimo/``.

Usage
-----
``python scripts/data_download/download_deepmimo.py
   --scenario O1 --url <resolved-url> --accept-license``

``--dry-run`` prints what would happen without performing any HTTP request.

Satisfies Requirements R10.1, R10.5, R11.3.
Task: pipeline support for Section 16.
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

# Spec.md §12 PIVOT 17 (2026-05-06): DeepMIMO v3 names O1/I1 are deprecated.
# v4 uses programmatic API; this script is retained as a thin manifest-friendly
# wrapper. Production downloads call ``deepmimo.download(name)`` directly.
VALID_SCENARIOS = ("asu_campus_3p5", "city_0_newyork_3p5")
LICENCE_NOTICE = (
    "DeepMIMO is distributed under a research licence by Alkhateeb et al. "
    "Pass --accept-license to acknowledge."
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download DeepMIMO scenarios with SHA-256 verification.",
    )
    parser.add_argument("--scenario", choices=VALID_SCENARIOS, required=True)
    parser.add_argument("--url", default=None,
                        help="Resolved download URL (DeepMIMO requires "
                             "registration; operator supplies the URL).")
    parser.add_argument("--output-dir", type=Path,
                        default=repo_root() / "data" / "deepmimo")
    parser.add_argument("--accept-license", action="store_true",
                        help="Acknowledge the DeepMIMO research licence.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without downloading.")
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

    # PIVOT 17: DeepMIMO v4 ships scenarios as directory trees, not .mat
    # files. The hash lookup key is now the bare scenario name; production
    # downloads should call ``deepmimo.download(name)`` directly. This script
    # remains a manifest-friendly entry point for the URL-based legacy mode.
    target_filename = args.scenario
    target_path = args.output_dir / target_filename

    expected = load_expected_hashes().get("deepmimo", {}).get(target_filename)
    if expected is None:
        logger.error(
            "No entry for deepmimo/%s in data/expected_hashes.json. "
            "Add it before downloading.", target_filename,
        )
        return 3

    if args.dry_run:
        print(f"DRY-RUN: would download {args.scenario} to {target_path}")
        print(f"DRY-RUN: expected SHA-256 = {expected}")
        print(f"DRY-RUN: URL = {args.url or '<not provided>'}")
        return 0

    if not args.url:
        logger.error("--url is required when not in --dry-run mode.")
        return 4

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", args.url, target_path)
    urllib.request.urlretrieve(args.url, target_path)  # noqa: S310 (whitelisted)

    if not verify_sha256(target_path, expected):
        logger.error("Hash verification failed; deleting %s", target_path)
        target_path.unlink(missing_ok=True)
        return 5

    logger.info("Verified SHA-256 for %s", target_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

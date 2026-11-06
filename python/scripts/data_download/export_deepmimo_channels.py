"""Export the DeepMIMO v4 channel tables used by ``deepmimo_loader``.

The ``deepmimo`` package (4.0.5) pins ``numpy<2.3``, while the analysis
environment runs a newer numpy. This script therefore runs once, in a
separate environment that has ``deepmimo`` installed, and writes the exact
array that ``deepmimo_loader._extract_complex_channel`` would otherwise build
in-process::

    ch = deepmimo.load(scenario).channel     # default channel parameters
    ch = ch[0] if multi-TX (MacroDataset)     # first TX
    ch2d = ch[:, 0, :, 0]                     # (num_rx, k_ant) complex

to ``<data_dir>/<scenario>_ch2d.npz`` (key ``ch2d`` plus provenance keys).
``deepmimo_loader`` reads this file when present and falls back to the
``deepmimo`` package otherwise, so both paths give the same channels.

Usage (deepmimo environment)::

    python -m venv /path/to/dmvenv && /path/to/dmvenv/bin/pip install deepmimo==4.0.5
    DEEPMIMO_DATA_DIR=$RESEARCH_DATA_ROOT/store/deepmimo-v4/data \\
      /path/to/dmvenv/bin/python scripts/data_download/export_deepmimo_channels.py

Scenarios are downloaded beforehand with ``deepmimo.download(name)``
(public, no account; https://deepmimo.net/api/download/secure?filename=<name>.zip).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SCENARIOS = ("asu_campus_3p5", "city_0_newyork_3p5")


def export(scenario: str, data_dir: Path) -> Path:
    import deepmimo

    deepmimo.config.set("scenarios_folder", str(data_dir))
    ch = deepmimo.load(scenario).channel
    n_tx = 1
    if isinstance(ch, (list, tuple)):
        n_tx = len(ch)
        ch = ch[0]
    ch = np.asarray(ch)
    if ch.ndim != 4:
        raise ValueError(f"{scenario}: channel ndim={ch.ndim}, expected 4")
    ch2d = np.ascontiguousarray(ch[:, 0, :, 0])
    if not np.all(np.isfinite(ch2d)):
        raise ValueError(f"{scenario}: non-finite channel entries")
    out = data_dir / f"{scenario}_ch2d.npz"
    np.savez(
        out,
        ch2d=ch2d,
        full_shape=np.asarray(ch.shape),
        n_tx_sets=np.asarray(n_tx),
        deepmimo_version=np.asarray(deepmimo.__version__),
        numpy_version=np.asarray(np.__version__),
    )
    print(f"{scenario}: ch2d {ch2d.shape} {ch2d.dtype}, full {ch.shape}, "
          f"n_tx_sets {n_tx} -> {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path,
                        default=os.environ.get("DEEPMIMO_DATA_DIR"))
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    args = parser.parse_args(argv)
    if args.data_dir is None:
        parser.error("--data-dir or DEEPMIMO_DATA_DIR is required")
    for s in args.scenario or SCENARIOS:
        export(s, Path(args.data_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())

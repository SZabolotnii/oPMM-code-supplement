"""End-to-end smoke test for the synthetic Monte Carlo harness.

Drives ``experiments.mv_patp_mimo.run`` programmatically on a tiny grid
(one estimator, one cell, two replicates) to confirm the orchestration
path from config -> grid expansion -> estimator dispatch -> per-cell
DataFrame -> manifest is wired correctly.

The full synthetic grid lives in
``experiments/mv_patp_mimo/config_synthetic.yaml``; this test exercises
a one-cell version so the unit-suite stays fast.

Task: Section 15.1.
Satisfies Requirements R11.1, R11.3, R11.4.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


def test_run_smoke_writes_cells_and_manifest_when_called(tmp_path: Path) -> None:
    """run.main() on a 1-cell config produces cells.csv + manifest.json.

    Requirement R11.1, R11.3.
    """
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "smoke",
                "replicates": 3,
                "grid": {
                    "k": [2],
                    "n": [128],
                    "p": [2],
                    "alpha_MAL": [1.0],
                    "M_norm": [0.0],
                    "sigma_structure": ["I"],
                },
                "estimators": ["ols"],
            }
        )
    )

    from experiments.mv_patp_mimo import run

    output_dir = tmp_path / "out"
    rc = run.main([
        "--config", str(config_path),
        "--output-dir", str(output_dir),
        "--seed", "42",
    ])
    assert rc == 0

    cells_path = output_dir / "cells.csv"
    manifest_path = output_dir / "manifest.json"
    assert cells_path.exists()
    assert manifest_path.exists()

    import pandas as pd
    df = pd.read_csv(cells_path)
    assert len(df) == 1, f"Expected 1 cell, got {len(df)}"
    row = df.iloc[0]
    assert row["estimator"] == "ols"
    assert row["status"] == "ok"
    assert row["n_replicates"] == 3
    assert np.isfinite(row["mse_mean"])

    manifest = json.loads(manifest_path.read_text())
    assert "git_commit" in manifest
    assert "package_versions" in manifest
    assert manifest["seed"] == 42
    assert manifest["pass_fail_summary"]

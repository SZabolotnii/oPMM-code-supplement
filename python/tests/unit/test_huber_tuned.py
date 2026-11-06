"""R1 (Rv1-C11): Huber IRLS with fixed and data-driven tuning constants.

* ``huber_c1p345`` reproduces ``huber_irls`` bit for bit.
* The other fixed-c variants change only ``c_efficiency``.
* ``huber_adaptive_c`` picks ``c`` by minimum plug-in asymptotic variance:
  the upper grid edge on Gaussian residuals, an interior value on
  Student-t(3); the final fit equals the fixed-c fit at ``c_hat``.
* ``reps.csv`` carries ``c_hat`` for the adaptive fit only.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from experiments.mv_patp_mimo import run
from mv_patp.baseline import huber_adaptive_c, huber_irls


def _data(noise: str, seed: int = 0, k: int = 3, n: int = 256, p: int = 2):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    if noise == "gaussian":
        N = rng.standard_normal((k, n))
    else:
        df = float(noise[1:])
        N = rng.standard_normal((k, n)) / np.sqrt(rng.chisquare(df, (k, n)) / df)
    return H @ S + N, S, H


@pytest.mark.parametrize("noise", ["gaussian", "t3"])
def test_c1p345_is_huber_irls_bit_for_bit(noise: str) -> None:
    Y, S, _ = _data(noise, seed=5)
    a = run._ESTIMATOR_REGISTRY["huber_irls"]()(Y, S)
    b = run._ESTIMATOR_REGISTRY["huber_c1p345"]()(Y, S)
    assert np.array_equal(a.H, b.H)


def test_fixed_c_variants_registered_and_distinct() -> None:
    Y, S, _ = _data("t3", seed=1)
    fits = {}
    for name, c in run.HUBER_FIXED_C.items():
        res = run._ESTIMATOR_REGISTRY[name]()(Y, S)
        ref = huber_irls.fit(Y, S, c_efficiency=c)
        assert np.array_equal(res.H, ref.H)
        fits[name] = res.H
    assert not np.array_equal(fits["huber_c0p7"], fits["huber_c2p5"])
    assert set(run.HUBER_FIXED_C) == {
        "huber_c0p7", "huber_c1p0", "huber_c1p345", "huber_c1p7",
        "huber_c2p0", "huber_c2p5"}


def test_avar_formula_matches_hand_value() -> None:
    u = np.array([-3.0, -0.5, 0.2, 1.0, 4.0])
    v = huber_adaptive_c.huber_avar(u, (1.0,))[0]
    # psi_1(u) = [-1, -0.5, 0.2, 1, 1]; psi' = [0, 1, 1, 1, 0]
    num = (1 + 0.25 + 0.04 + 1 + 1) / 5
    den = 3 / 5
    assert np.isclose(v, num / den**2)
    assert np.isinf(huber_adaptive_c.huber_avar(np.array([5.0, 6.0]), (1.0,))[0])


def test_default_grid() -> None:
    g = huber_adaptive_c.C_GRID_DEFAULT
    assert len(g) == 26 and g[0] == 0.5 and g[-1] == 3.0
    assert np.allclose(np.diff(g), 0.1)


def test_selection_gaussian_vs_heavy_tail() -> None:
    rng = np.random.default_rng(0)
    z = rng.standard_normal(20000)
    assert huber_adaptive_c.select_c(z) == 3.0          # V(c) decreasing for N(0,1)
    t3 = rng.standard_normal(20000) / np.sqrt(rng.chisquare(3, 20000) / 3)
    t3 = t3 / (np.median(np.abs(t3)) / huber_adaptive_c._MAD_SCALE)
    c3 = huber_adaptive_c.select_c(t3)
    assert 0.5 < c3 < 3.0


def test_adaptive_fit_equals_fixed_fit_at_c_hat() -> None:
    Y, S, _ = _data("t3", seed=2)
    res = huber_adaptive_c.fit(Y, S)
    c_hat = res.log_history["c_hat"]
    assert c_hat in huber_adaptive_c.C_GRID_DEFAULT
    pilot = huber_irls.fit(Y, S)
    u = huber_adaptive_c.standardised_residuals(pilot.residuals)
    assert c_hat == huber_adaptive_c.select_c(u)
    ref = huber_irls.fit(Y, S, c_efficiency=c_hat)
    assert np.array_equal(res.H, ref.H)
    assert res.converged


def test_reps_rows_carry_c_hat() -> None:
    cell = run.CellSpec(k=2, n=128, alpha_MAL=float("nan"), p=2, M_norm=0.0,
                        sigma_structure="student_t_3", replicates=2,
                        estimator="huber_adaptive_c")
    row = run.run_mc_cell(cell, seed=4, per_replicate=True)
    reps = pd.DataFrame(row["_rep_rows"])
    assert row["status"] == "ok"
    assert reps["c_hat"].notna().all()
    assert "c_hat" in run.PER_REP_COLUMNS
    cell_h = dataclasses.replace(cell, estimator="huber_irls")
    row_h = run.run_mc_cell(cell_h, seed=4, per_replicate=True)
    assert pd.DataFrame(row_h["_rep_rows"])["c_hat"].isna().all()

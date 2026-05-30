"""Unit tests for huber_irls.py baseline.

Requirement R9: robust-estimator competitor must beat OLS on heavy tails
and must NOT use cross-row Gamma coupling (row independence property).
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.baseline import huber_irls, ols


def _make_data(
    k: int, n: int, p: int, noise_type: str, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    if noise_type == "gaussian":
        N = rng.standard_normal((k, n))
    elif noise_type.startswith("t"):
        df = float(noise_type[1:])
        z = rng.standard_normal((k, n))
        chi2 = rng.chisquare(df=df, size=(k, n))
        N = z / np.sqrt(chi2 / df)
    else:
        raise ValueError(noise_type)
    Y = H_truth @ S + N
    return Y, S, H_truth


def test_gaussian_near_ols():
    """Requirement R9: on Gaussian noise, huber_irls ARE should be ∈ [0.90, 1.10].

    Huber (c=1.345) achieves 95% Gaussian asymptotic efficiency, so ARE ≈ 0.95
    at finite n. We allow ±10% around 1.0 to be generous at n=256.
    """
    n_reps = 200
    sq_huber = []
    sq_ols = []
    for rep in range(n_reps):
        Y, S, H_truth = _make_data(2, 256, 2, "gaussian", seed=rep)
        res_h = huber_irls.fit(Y, S)
        res_o = ols.fit(Y, S)
        sq_huber.append(np.linalg.norm(res_h.H - H_truth, ord="fro") ** 2)
        sq_ols.append(np.linalg.norm(res_o.H - H_truth, ord="fro") ** 2)
    are = float(np.mean(sq_ols) / np.mean(sq_huber))
    assert 0.90 <= are <= 1.10, f"Gaussian ARE={are:.4f} outside [0.90, 1.10]"


def test_no_cross_row_coupling():
    """Requirement R9: row 0 solution must not depend on row 1 data.

    We perturb Y[1,:] by a large constant and check that H[0,:] is unchanged.
    """
    rng = np.random.default_rng(42)
    k, n, p = 3, 128, 2
    Y = rng.standard_normal((k, n))
    S = rng.standard_normal((p, n))

    res1 = huber_irls.fit(Y, S)
    Y_perturbed = Y.copy()
    Y_perturbed[1, :] += 1e6  # huge perturbation to row 1
    res2 = huber_irls.fit(Y_perturbed, S)

    # Row 0 must be unchanged (independent rows)
    assert np.allclose(res1.H[0, :], res2.H[0, :], atol=1e-9), (
        "huber_irls row 0 changed when only row 1 data was perturbed — "
        "unexpected cross-row coupling."
    )
    # Row 2 must also be unchanged
    assert np.allclose(res1.H[2, :], res2.H[2, :], atol=1e-9), (
        "huber_irls row 2 changed when only row 1 data was perturbed."
    )


def test_beats_ols_at_df4():
    """Requirement R9: huber_irls ARE > 1.05 over OLS at Student-t df=4."""
    n_reps = 200
    sq_huber = []
    sq_ols = []
    for rep in range(n_reps):
        Y, S, H_truth = _make_data(2, 512, 2, "t4", seed=rep + 100)
        res_h = huber_irls.fit(Y, S)
        res_o = ols.fit(Y, S)
        sq_huber.append(np.linalg.norm(res_h.H - H_truth, ord="fro") ** 2)
        sq_ols.append(np.linalg.norm(res_o.H - H_truth, ord="fro") ** 2)
    are = float(np.mean(sq_ols) / np.mean(sq_huber))
    assert are > 1.05, (
        f"huber_irls ARE={are:.4f} over OLS at df=4 not > 1.05; "
        "huber should exploit heavy tails."
    )


def test_converged_flag_gaussian():
    """converged should be True for well-conditioned Gaussian data."""
    rng = np.random.default_rng(7)
    Y = rng.standard_normal((2, 256))
    S = rng.standard_normal((2, 256))
    res = huber_irls.fit(Y, S)
    assert res.converged, "Expected converged=True for Gaussian data."


def test_output_shape():
    """Output H must have shape (k, p)."""
    Y, S, _ = _make_data(3, 64, 2, "gaussian", seed=0)
    res = huber_irls.fit(Y, S)
    assert res.H.shape == (3, 2)
    assert res.residuals.shape == (3, 64)

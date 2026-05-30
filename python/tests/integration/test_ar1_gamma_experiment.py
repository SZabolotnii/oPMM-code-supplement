"""Integration tests for AR(1) Gamma correlated-noise experiment (Exp 6).

Requirement R9 / Spec.md §revision.A1: AR(1) Gamma noise structure must be
correctly generated, and whitening-based estimators must outperform row-
independent Huber when row correlation is high (rho=0.9).
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.baseline import huber_irls, whitened_huber

try:
    from mv_patp.core.em_patp import em_patp_estimate as _opmm_fit
    _HAS_OPMM = True
except ImportError:
    _HAS_OPMM = False


def _draw_ar1_gamma_noise(k: int, n: int, rho: float, seed: int) -> np.ndarray:
    """Replicate the run.py AR(1) Gamma noise generator."""
    rng = np.random.default_rng(seed)
    i_idx, j_idx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    Gamma_ar1 = rho ** np.abs(i_idx - j_idx)
    L = np.linalg.cholesky(Gamma_ar1 + 1e-9 * np.eye(k))
    return L @ rng.standard_normal((k, n))


def _make_ar1_data(
    k: int, n: int, p: int, rho: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 1000)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    N = _draw_ar1_gamma_noise(k, n, rho, seed=seed)
    return H_truth @ S + N, S, H_truth


def test_ar1_noise_covariance_structure():
    """Requirement R9/A1: AR(1) noise off-diagonal / diagonal ≈ rho."""
    k, n, rho = 4, 10_000, 0.9
    N = _draw_ar1_gamma_noise(k, n, rho, seed=0)
    cov = N @ N.T / n  # empirical covariance
    # Normalise to correlation matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    # First off-diagonal should be ≈ rho
    for i in range(k - 1):
        measured_rho = float(corr[i, i + 1])
        assert abs(measured_rho - rho) < 0.05, (
            f"AR(1) correlation[{i},{i+1}]={measured_rho:.4f}, expected ≈{rho}. "
            "AR(1) Gamma noise DGP may be incorrect."
        )


def test_huber_degrades_under_correlation():
    """Requirement R9/Exp6: MSE(huber_irls) / MSE(opmm_alpha) > 1.03 at rho=0.9.

    huber_irls does not use Gamma whitening; when rows are correlated it
    cannot exploit the structure, so it should be materially worse than
    oPMM_alpha which jointly whitens.
    """
    if not _HAS_OPMM:
        pytest.skip("em_patp not importable")
    n_reps = 100
    sq_huber = []
    sq_opmm = []
    for rep in range(n_reps):
        Y, S, H_truth = _make_ar1_data(4, 256, 2, rho=0.9, seed=rep)
        res_h = huber_irls.fit(Y, S)
        res_o = _opmm_fit(Y, S)
        sq_huber.append(np.linalg.norm(res_h.H - H_truth, ord="fro") ** 2)
        sq_opmm.append(np.linalg.norm(res_o.H - H_truth, ord="fro") ** 2)
    ratio = float(np.mean(sq_huber) / np.mean(sq_opmm))
    assert ratio > 1.03, (
        f"MSE(huber_irls)/MSE(opmm_alpha)={ratio:.4f} at AR(1) rho=0.9; "
        "expected > 1.03 (huber should degrade without whitening)."
    )


def test_whitened_huber_bridges_gap():
    """Requirement R9/Exp6: MSE(whitened_huber) < MSE(huber_irls) at rho=0.9.

    whitened_huber uses the same EM scaffold as oPMM_alpha but a Huber
    score.  At high row correlation the whitening benefit alone should
    lift it above row-independent huber_irls.
    """
    n_reps = 100
    sq_wh = []
    sq_hi = []
    for rep in range(n_reps):
        Y, S, H_truth = _make_ar1_data(4, 256, 2, rho=0.9, seed=rep + 500)
        res_wh = whitened_huber.fit(Y, S)
        res_hi = huber_irls.fit(Y, S)
        sq_wh.append(np.linalg.norm(res_wh.H - H_truth, ord="fro") ** 2)
        sq_hi.append(np.linalg.norm(res_hi.H - H_truth, ord="fro") ** 2)
    mse_wh = float(np.mean(sq_wh))
    mse_hi = float(np.mean(sq_hi))
    assert mse_wh < mse_hi, (
        f"MSE(whitened_huber)={mse_wh:.5f} >= MSE(huber_irls)={mse_hi:.5f} "
        "at AR(1) rho=0.9; expected whitening to help."
    )

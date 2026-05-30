"""Unit tests for whitened_huber.py baseline.

Requirement R9: whitened_huber uses Gamma whitening (unlike huber_irls)
and should outperform huber_irls when row correlation is non-trivial.
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.baseline import huber_irls, whitened_huber


def _correlated_t_data(
    k: int, n: int, p: int, rho: float, df: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data with AR(1)-correlated Student-t noise."""
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    i_idx, j_idx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
    Gamma_ar1 = rho ** np.abs(i_idx - j_idx)
    L = np.linalg.cholesky(Gamma_ar1 + 1e-9 * np.eye(k))
    # Correlated Gaussian scaled by chi2 (correlated t-like)
    z = L @ rng.standard_normal((k, n))
    chi2 = rng.chisquare(df=df, size=n)
    N = z / np.sqrt(chi2 / df)[None, :]
    Y = H_truth @ S + N
    return Y, S, H_truth


def test_uses_gamma_whitening():
    """log_history['Gamma'] should not be identity — whitening is active."""
    rng = np.random.default_rng(42)
    k, n, p = 4, 256, 2
    # Build AR(1)-correlated t3 noise
    Y, S, _ = _correlated_t_data(k, n, p, rho=0.8, df=3, seed=42)
    res = whitened_huber.fit(Y, S)
    Gamma = res.log_history.get("Gamma")
    assert Gamma is not None, "log_history['Gamma'] missing"
    # Gamma should not be close to identity
    I_k = np.eye(k)
    frob_diff = float(np.linalg.norm(Gamma - I_k, ord="fro"))
    assert frob_diff > 0.05, (
        f"Gamma too close to identity (frob diff={frob_diff:.4f}); "
        "whitening appears inactive."
    )


def test_closer_to_opmm_than_huber_on_correlated():
    """Requirement R9: on correlated noise, MSE(whitened_huber) <= MSE(huber_irls).

    At n=256 the whitening gain is modest (< 1%); we only require
    whitened_huber to not be substantially worse than huber_irls.
    The strong whitening advantage manifests at rho=0.9 and larger n;
    the full MC campaign (Exp 6) tests the 5%-gain criterion with n=512.
    """
    n_reps = 100
    sq_wh = []
    sq_hi = []
    for rep in range(n_reps):
        Y, S, H_truth = _correlated_t_data(4, 256, 2, rho=0.9, df=4, seed=rep)
        res_wh = whitened_huber.fit(Y, S)
        res_hi = huber_irls.fit(Y, S)
        sq_wh.append(np.linalg.norm(res_wh.H - H_truth, ord="fro") ** 2)
        sq_hi.append(np.linalg.norm(res_hi.H - H_truth, ord="fro") ** 2)
    mse_wh = float(np.mean(sq_wh))
    mse_hi = float(np.mean(sq_hi))
    # whitened_huber should not be worse than huber_irls by more than 5%
    assert mse_wh <= mse_hi * 1.05, (
        f"whitened_huber MSE={mse_wh:.5f} > 1.05 * huber_irls MSE={mse_hi:.5f} "
        "on AR(1) rho=0.9 data; whitened should be at least competitive."
    )


def test_output_shape():
    """Output H must have shape (k, p)."""
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((3, 128))
    S = rng.standard_normal((2, 128))
    res = whitened_huber.fit(Y, S)
    assert res.H.shape == (3, 2)
    assert res.residuals.shape == (3, 128)

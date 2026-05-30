"""Unit tests for core/irls_matrix.py::solve_weighted_matrix_LS.

Satisfies Requirement R6.1 (M-step sub-component).
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.core.irls_matrix import solve_weighted_matrix_LS
from mv_patp.utils.errors import NumericalInstabilityError


def _spd(m: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, m))
    return A @ A.T + 0.1 * np.eye(m)


def test_solve_weighted_matrix_ls_recovers_truth_when_no_noise() -> None:
    """With zero noise and unit weights, the solver returns H_truth exactly.

    Requirement R6.1.
    """
    rng = np.random.default_rng(42)
    k, p, n = 2, 2, 64
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    Y = H_truth @ S
    Gamma_hat = np.eye(k)
    Sigma_hat = np.eye(n)
    H = solve_weighted_matrix_LS(Y, S, W=None, Gamma_hat=Gamma_hat, Sigma_hat=Sigma_hat)
    assert np.allclose(H, H_truth, atol=1e-8), (
        f"||H - H_truth||_F = {np.linalg.norm(H - H_truth):.3e}"
    )


def test_solve_weighted_matrix_ls_matches_ols_when_identity_covariances() -> None:
    """Unit weights + identity covariances reduce to OLS Y S^T (S S^T)^{-1}.

    Requirement R6.1.
    """
    rng = np.random.default_rng(0)
    k, p, n = 3, 2, 128
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    N = 0.05 * rng.standard_normal((k, n))
    Y = H_truth @ S + N

    H_irls = solve_weighted_matrix_LS(
        Y, S, W=None, Gamma_hat=np.eye(k), Sigma_hat=np.eye(n)
    )
    H_ols, *_ = np.linalg.lstsq(S.T, Y.T, rcond=None)
    H_ols = H_ols.T

    assert np.allclose(H_irls, H_ols, atol=1e-8), (
        f"IRLS deviates from OLS by {np.linalg.norm(H_irls - H_ols):.3e} "
        f"on identity-covariance data"
    )


def test_solve_weighted_matrix_ls_uses_weights_when_outliers_downweighted() -> None:
    """Heavily down-weighting outlier columns yields a cleaner H estimate.

    Requirement R6.1. We inject one massive outlier into the residuals
    of a single column and verify that setting that column's weights
    near zero recovers H much closer to truth than uniform weights.
    """
    rng = np.random.default_rng(2026)
    k, p, n = 2, 2, 128
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    N = 0.01 * rng.standard_normal((k, n))
    N[:, 0] += 100.0  # huge outlier on column 0
    Y = H_truth @ S + N

    H_uniform = solve_weighted_matrix_LS(
        Y, S, W=None, Gamma_hat=np.eye(k), Sigma_hat=np.eye(n)
    )
    W = np.ones((k, n))
    W[:, 0] = 1e-8
    H_robust = solve_weighted_matrix_LS(
        Y, S, W=W, Gamma_hat=np.eye(k), Sigma_hat=np.eye(n)
    )

    err_uniform = np.linalg.norm(H_uniform - H_truth, ord="fro")
    err_robust = np.linalg.norm(H_robust - H_truth, ord="fro")
    assert err_robust < 0.5 * err_uniform, (
        f"Robust err {err_robust:.3f} did not improve on uniform err {err_uniform:.3f}"
    )


def test_solve_weighted_matrix_ls_handles_non_identity_covariances() -> None:
    """Solver succeeds when Gamma_hat and Sigma_hat are non-trivial SPD.

    Requirement R6.1.
    """
    rng = np.random.default_rng(7)
    k, p, n = 2, 2, 96
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    Y = H_truth @ S + 0.05 * rng.standard_normal((k, n))
    Gamma_hat = _spd(k, seed=10)
    Sigma_hat = _spd(n, seed=11)
    H = solve_weighted_matrix_LS(Y, S, W=None, Gamma_hat=Gamma_hat, Sigma_hat=Sigma_hat)
    assert H.shape == (k, p)
    assert np.all(np.isfinite(H))


def test_solve_weighted_matrix_ls_rejects_shape_mismatch_when_S_n_disagrees() -> None:
    """Shape mismatch surfaces a typed ValueError.

    Requirement R6.1 (input validation).
    """
    Y = np.zeros((2, 64))
    S = np.zeros((2, 32))
    with pytest.raises(ValueError, match="shapes must agree"):
        solve_weighted_matrix_LS(Y, S, W=None, Gamma_hat=np.eye(2), Sigma_hat=np.eye(64))

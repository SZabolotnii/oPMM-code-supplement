"""Jointly whitened Huber IRLS baseline for MIMO channel identification.

Uses the same EM outer scaffold as oPMM_alpha (Gamma E-step via sample
covariance, Kronecker-whitened M-step via solve_weighted_matrix_LS) but
replaces the fractional oPMM_alpha score with a Huber score:

    W_ij = min(c / max(|R_tilde_ij|, 1e-14), 1.0)   (Huber, per-element)

Comparing this estimator against huber_irls decomposes the total gain of
oPMM_alpha over OLS into:

    whitening gain  = MSE(huber_irls) - MSE(whitened_huber)
    score gain      = MSE(whitened_huber) - MSE(opmm_alpha)

Expected ranking on iid heavy-tailed data:
    OLS > huber_irls > whitened_huber > oPMM_alpha   (lower MSE = better)

Satisfies Requirement R9 (ablation for gain decomposition).
Task: Section revision.C1 (reviewer experiment 2).
"""

from __future__ import annotations

import logging

import numpy as np

from mv_patp.baseline._types import BaselineResult
from mv_patp.baseline.ols import fit as _ols_fit
from mv_patp.core.irls_matrix import solve_weighted_matrix_LS
from mv_patp.core.whitening import matrix_inverse_sqrt
from mv_patp.utils.errors import EMDivergence, NumericalInstabilityError

logger = logging.getLogger(__name__)

_MAD_SCALE = 0.6745  # median(|N(0,1)|)


def fit(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    c_efficiency: float = 1.345,
    max_iter: int = 30,
    irls_max_iter: int = 15,
    tol: float = 1e-6,
    eigenvalue_floor: float = 1e-6,
    raise_on_divergence: bool = False,
) -> BaselineResult:
    """Estimate H via Huber IRLS with joint Gamma-whitening EM scaffold.

    The Gamma E-step is identical to oPMM_alpha (sample covariance of
    residuals + eigenvalue floor). The M-step uses element-wise Huber
    weights via the Kronecker solve, so cross-row correlation is exploited
    through whitening even though the score function is classical Huber.

    Parameters
    ----------
    Y : np.ndarray
        Received signal of shape ``(k, n)``.
    S : np.ndarray
        Known training matrix of shape ``(p, n)``, full row rank.
    c_efficiency : float, default 1.345
        Huber threshold in MAD scale units.
    max_iter : int, default 30
        Maximum EM outer iterations.
    irls_max_iter : int, default 15
        Maximum inner IRLS iterations per outer step.
    tol : float, default 1e-6
        Relative Frobenius convergence tolerance.
    eigenvalue_floor : float, default 1e-6
        Floor for matrix_inverse_sqrt.
    raise_on_divergence : bool, default False

    Returns
    -------
    BaselineResult
        ``H`` of shape ``(k, p)`` plus convergence diagnostics including
        ``log_history["Gamma"]`` (the final Gamma estimate) and
        ``log_history["gamma_frob_delta"]`` per outer iteration.
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    k, n = Y.shape

    H = _ols_fit(Y, S).H.copy()
    R = Y - H @ S
    Gamma = R @ R.T / n + eigenvalue_floor * np.eye(k)
    Sigma_hat = np.eye(n)

    history_frob: list[float] = []
    history_gamma_frob: list[float] = []
    converged = False
    outer = 0

    for outer in range(max_iter):
        H_prev = H.copy()

        # E-step: sample covariance update (identical to oPMM_alpha)
        R = Y - H @ S
        Gamma_new = R @ R.T / n + eigenvalue_floor * np.eye(k)
        history_gamma_frob.append(
            float(np.linalg.norm(Gamma_new - Gamma, ord="fro"))
        )
        Gamma = Gamma_new

        G_inv_sqrt = matrix_inverse_sqrt(Gamma, eps=eigenvalue_floor)

        # Inner Huber IRLS M-step with Kronecker whitening
        for _inner in range(irls_max_iter):
            R_white = G_inv_sqrt @ (Y - H @ S)  # (k, n)
            sigma_hat = np.median(np.abs(R_white)) / _MAD_SCALE
            if sigma_hat < 1e-14:
                break
            c = c_efficiency * sigma_hat
            W = np.minimum(c / np.maximum(np.abs(R_white), 1e-14), 1.0)
            H_new = solve_weighted_matrix_LS(
                Y, S, W, Gamma, Sigma_hat, eigenvalue_floor=eigenvalue_floor
            )
            inner_delta = float(np.linalg.norm(H_new - H, ord="fro"))
            H = H_new
            if inner_delta < tol:
                break

        outer_delta = float(np.linalg.norm(H - H_prev, ord="fro"))
        norm_prev = max(float(np.linalg.norm(H_prev, ord="fro")), 1e-14)
        history_frob.append(outer_delta)
        if outer_delta / norm_prev < tol:
            converged = True
            break

    if not converged and raise_on_divergence:
        raise EMDivergence(
            f"whitened_huber: did not converge in {max_iter} outer iterations."
        )

    if not np.all(np.isfinite(H)):
        raise NumericalInstabilityError("whitened_huber: NaN/Inf in H.")

    return BaselineResult(
        H=H,
        residuals=Y - H @ S,
        converged=converged,
        n_iter=outer + 1,
        log_history={
            "frob_delta": history_frob,
            "gamma_frob_delta": history_gamma_frob,
            "Gamma": Gamma,
        },
    )

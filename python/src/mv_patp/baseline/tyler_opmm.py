"""Tyler-type scatter E-step combined with oPMM_alpha M-step.

Replaces the sample-covariance E-step of oPMM_alpha with the Tyler
M-estimator of scatter (Tyler, 1987).  Tyler's estimator is maximally
robust to heavy-tailed row distributions and recovers the true Gamma
direction (shape) without any moment assumptions -- at the cost of
requiring trace normalisation (tr(Gamma) = k) for identifiability.

Expected behaviour:
- On iid Student-t noise:  MSE(tyler_opmm) ≈ MSE(opmm_alpha) within 2%.
- On AR(1)-correlated noise (rho=0.9): MSE(tyler_opmm) <= MSE(opmm_alpha)
  because Tyler better recovers the off-diagonal structure of Gamma.

Tyler fixed-point convergence is guaranteed for n > k+1
(Kent & Tyler, 1988 J. Roy. Stat. Soc. B).

Satisfies Requirement R9 (ablation for E-step robustness).
Task: Section revision.C2 (reviewer experiment 4).
"""

from __future__ import annotations

import logging

import numpy as np

from mv_patp.baseline._types import BaselineResult
from mv_patp.baseline.ols import fit as _ols_fit
from mv_patp.core.alpha_optimizer import find_alpha_opt
from mv_patp.core.irls_matrix import solve_weighted_matrix_LS
from mv_patp.core.score import compute_optimal_weights
from mv_patp.core.whitening import matrix_inverse_sqrt
from mv_patp.utils.cumulants import empirical_cumulants
from mv_patp.utils.errors import EMDivergence, NumericalInstabilityError

logger = logging.getLogger(__name__)

_ALPHA_DEGEN_LO = 0.92
_ALPHA_DEGEN_HI = 1.08
_GAUSS_KURT_THRESHOLD = 0.5


def _alpha_heuristic(R_white: np.ndarray, lo: float = 0.3, hi: float = 3.0) -> float:
    """Kurtosis-driven alpha update (mirrors em_patp._alpha_heuristic)."""
    eps = R_white.flatten()
    var = float(np.var(eps, ddof=1))
    if var <= 0.0:
        return 1.5
    fourth = float(np.mean(eps ** 4))
    gamma4 = max(fourth / max(var ** 2, 1e-30) - 3.0, 0.0)
    if gamma4 < _GAUSS_KURT_THRESHOLD:
        return float(np.clip(2.0, lo, hi))
    alpha = 1.5 / (1.0 + 0.3 * gamma4)
    if _ALPHA_DEGEN_LO < alpha < _ALPHA_DEGEN_HI:
        alpha = 1.5
    return float(np.clip(alpha, lo, hi))


def _tyler_scatter(
    R: np.ndarray,
    Gamma_init: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    eigenvalue_floor: float = 1e-6,
) -> np.ndarray:
    """Fixed-point Tyler scatter estimator, trace-normalised to tr(Gamma)=k.

    Parameters
    ----------
    R : np.ndarray
        Residual matrix of shape ``(k, n)``.
    Gamma_init : np.ndarray
        Initial scatter estimate of shape ``(k, k)``, positive definite.
    max_iter : int
        Maximum fixed-point iterations.
    tol : float
        Relative Frobenius convergence tolerance.
    eigenvalue_floor : float
        Added to diagonal before inversion for numerical stability.

    Returns
    -------
    np.ndarray
        Tyler scatter estimate of shape ``(k, k)``, trace-normalised.
    """
    k, n = R.shape
    Gamma = Gamma_init.copy()
    I_k = np.eye(k)

    for _ in range(max_iter):
        Gamma_inv = np.linalg.inv(Gamma + 1e-9 * I_k)
        # Mahalanobis distance per observation column: r_j^T Gamma^{-1} r_j
        mahal = np.einsum("ki,kl,li->i", R, Gamma_inv, R)  # (n,)
        mahal = np.maximum(mahal, 1e-10)
        # Tyler update: (k/n) sum_j r_j r_j^T / mahal_j
        Gamma_new = (k / n) * (R / mahal[None, :]) @ R.T   # (k, k)
        Gamma_new = (Gamma_new + Gamma_new.T) / 2.0 + eigenvalue_floor * I_k
        trace = float(np.trace(Gamma_new))
        if trace < 1e-14:
            break
        Gamma_new = Gamma_new * (k / trace)  # enforce tr(Gamma) = k
        delta = float(np.linalg.norm(Gamma_new - Gamma, ord="fro")) / max(
            float(np.linalg.norm(Gamma, ord="fro")), 1e-14
        )
        Gamma = Gamma_new
        if delta < tol:
            break

    return Gamma


def fit(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    max_iter: int = 50,
    tyler_inner_iter: int = 50,
    tyler_tol: float = 1e-6,
    tol: float = 1e-6,
    irls_max_iter: int = 25,
    irls_tol: float = 1e-8,
    eps_min: float = 1e-10,
    eigenvalue_floor: float = 1e-6,
    raise_on_divergence: bool = False,
) -> BaselineResult:
    """Estimate H with Tyler scatter E-step and oPMM_alpha M-step.

    Parameters
    ----------
    Y : np.ndarray
        Received signal of shape ``(k, n)``.
    S : np.ndarray
        Known training matrix of shape ``(p, n)``, full row rank.
    max_iter : int, default 50
        Maximum EM outer iterations.
    tyler_inner_iter : int, default 50
        Maximum Tyler fixed-point iterations per outer step.
    tyler_tol : float, default 1e-6
        Relative Frobenius convergence tolerance for Tyler inner loop.
    tol : float, default 1e-6
        Relative Frobenius convergence tolerance for H (outer loop).
    irls_max_iter : int, default 25
        Maximum inner IRLS iterations per outer step.
    irls_tol : float, default 1e-8
        Absolute Frobenius tolerance for IRLS inner loop.
    eps_min : float, default 1e-10
        Floor on residual magnitude for the fractional score to prevent
        blow-up at alpha < 1.
    eigenvalue_floor : float, default 1e-6
        Floor for matrix_inverse_sqrt and Tyler regularisation.
    raise_on_divergence : bool, default False

    Returns
    -------
    BaselineResult
        ``H`` of shape ``(k, p)`` plus convergence diagnostics.
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    k, n = Y.shape

    H = _ols_fit(Y, S).H.copy()
    R = Y - H @ S
    Gamma = R @ R.T / n + eigenvalue_floor * np.eye(k)
    Sigma_hat = np.eye(n)

    # Initial alpha from full grid search on OLS residuals
    G_inv_sqrt = matrix_inverse_sqrt(Gamma, eps=eigenvalue_floor)
    R_white_init = G_inv_sqrt @ R
    try:
        alpha, _, _ = find_alpha_opt({"eps_white": R_white_init.flatten()})
        alpha = float(alpha)
    except Exception:
        alpha = 1.5

    history_frob: list[float] = []
    converged = False
    outer = 0

    for outer in range(max_iter):
        H_prev = H.copy()

        R = Y - H @ S
        # Tyler scatter E-step (replaces sample covariance)
        Gamma = _tyler_scatter(
            R, Gamma,
            max_iter=tyler_inner_iter,
            tol=tyler_tol,
            eigenvalue_floor=eigenvalue_floor,
        )

        G_inv_sqrt = matrix_inverse_sqrt(Gamma, eps=eigenvalue_floor)
        R_white = G_inv_sqrt @ R
        # Adaptive alpha: kurtosis heuristic (same as oPMM_alpha in-loop)
        alpha = _alpha_heuristic(R_white)

        # oPMM_alpha inner IRLS M-step
        for _inner in range(irls_max_iter):
            R_white = G_inv_sqrt @ (Y - H @ S)
            try:
                C = empirical_cumulants(R_white.flatten(), alpha)
                k1, k2 = compute_optimal_weights(C, alpha)
            except Exception:
                k1, k2 = 1.0, 0.0
            abs_R = np.maximum(np.abs(R_white), eps_min)
            W = np.maximum(k1 + k2 * abs_R ** (alpha - 1.0), 1e-6)
            H_new = solve_weighted_matrix_LS(
                Y, S, W, Gamma, Sigma_hat, eigenvalue_floor=eigenvalue_floor
            )
            inner_delta = float(np.linalg.norm(H_new - H, ord="fro"))
            H = H_new
            if inner_delta < irls_tol:
                break

        outer_delta = float(np.linalg.norm(H - H_prev, ord="fro"))
        norm_prev = max(float(np.linalg.norm(H_prev, ord="fro")), 1e-14)
        history_frob.append(outer_delta)
        if outer_delta / norm_prev < tol:
            converged = True
            break

    if not converged and raise_on_divergence:
        raise EMDivergence(
            f"tyler_opmm: did not converge in {max_iter} outer iterations."
        )

    if not np.all(np.isfinite(H)):
        raise NumericalInstabilityError("tyler_opmm: NaN/Inf in H.")

    return BaselineResult(
        H=H,
        residuals=Y - H @ S,
        converged=converged,
        n_iter=outer + 1,
        log_history={
            "frob_delta": history_frob,
            "alpha_final": alpha,
            "Gamma": Gamma,
        },
    )

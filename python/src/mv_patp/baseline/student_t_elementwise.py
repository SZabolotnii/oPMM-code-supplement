"""Student-t element-wise MLE baseline for MIMO channel identification.

Estimates H via EM with per-element IRLS weights

    W_ij = (df + 1) / (df + epsilon_tilde_ij^2)

applied to the whitened residuals, unlike ``matrix_t_mle.py`` which uses
a single scalar Mahalanobis weight per observation column.  On iid
heavy-tailed data the scalar Mahalanobis weight converges to a constant
(collapsing to OLS); the per-element weight retains residual-level
robustness and does NOT collapse.

When ``df=None`` (default) the degrees of freedom are estimated from the
empirical excess kurtosis of the whitened residuals via the moment formula:

    df_hat = max(6 / gamma4 + 4, df_min)

which gives df=4 at gamma4=3 (Laplace), df≈5.5 at gamma4=2, df→∞ for
near-Gaussian residuals.

Satisfies Requirement R9 (robust-estimator competitor for ARE table).
Task: Section revision.B2 (reviewer experiment 3).
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


def _excess_kurtosis(eps: np.ndarray) -> float:
    """Unbiased excess kurtosis estimator (Fisher)."""
    n = len(eps)
    if n < 4:
        return 0.0
    mean = float(np.mean(eps))
    var = float(np.var(eps, ddof=1))
    if var < 1e-14:
        return 0.0
    m4 = float(np.mean((eps - mean) ** 4))
    return m4 / var ** 2 - 3.0


def _estimate_df(R_white: np.ndarray, df_min: float = 3.0) -> float:
    """Moment estimate df = 6/gamma4 + 4, floored at df_min."""
    gamma4 = max(_excess_kurtosis(R_white.flatten()), 0.0)
    if gamma4 < 0.5:
        return 30.0  # near-Gaussian; effectively infinite df
    return float(max(6.0 / gamma4 + 4.0, df_min))


def fit(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    df: float | None = None,
    df_min: float = 3.0,
    max_iter: int = 30,
    irls_max_iter: int = 15,
    tol: float = 1e-6,
    eigenvalue_floor: float = 1e-6,
    raise_on_divergence: bool = False,
) -> BaselineResult:
    """Estimate H via per-element Student-t IRLS with EM Gamma update.

    Parameters
    ----------
    Y : np.ndarray
        Received signal of shape ``(k, n)``.
    S : np.ndarray
        Known training matrix of shape ``(p, n)``, full row rank.
    df : float or None, default None
        Degrees of freedom.  When None, estimated from empirical kurtosis
        at each outer EM iteration.
    df_min : float, default 3.0
        Minimum allowed df (prevents unbounded per-element weights).
    max_iter : int, default 30
        Maximum EM outer iterations.
    irls_max_iter : int, default 15
        Maximum inner IRLS iterations per outer step.
    tol : float, default 1e-6
        Relative Frobenius convergence tolerance.
    eigenvalue_floor : float, default 1e-6
        Eigenvalue floor for matrix_inverse_sqrt.
    raise_on_divergence : bool, default False

    Returns
    -------
    BaselineResult
        ``H`` of shape ``(k, p)`` plus convergence diagnostics including
        ``log_history["df_estimated"]`` per outer iteration.
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    k, n = Y.shape

    H = _ols_fit(Y, S).H.copy()
    R = Y - H @ S
    Gamma = R @ R.T / n + eigenvalue_floor * np.eye(k)
    Sigma_hat = np.eye(n)

    history_frob: list[float] = []
    df_history: list[float] = []
    converged = False
    outer = 0

    for outer in range(max_iter):
        H_prev = H.copy()

        # E-step: sample covariance update of Gamma
        R = Y - H @ S
        Gamma = R @ R.T / n + eigenvalue_floor * np.eye(k)

        G_inv_sqrt = matrix_inverse_sqrt(Gamma, eps=eigenvalue_floor)
        R_white = G_inv_sqrt @ R  # (k, n)

        # Estimate df from whitened residuals (or use fixed df)
        df_hat = df if df is not None else _estimate_df(R_white, df_min)
        df_history.append(df_hat)

        # Inner IRLS: per-element Student-t weights
        for _inner in range(irls_max_iter):
            R_white = G_inv_sqrt @ (Y - H @ S)  # refresh whitened residuals
            W = (df_hat + 1.0) / (df_hat + R_white ** 2)
            W = np.maximum(W, 1e-6)
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
            f"student_t_elementwise: did not converge in {max_iter} outer iterations."
        )

    if not np.all(np.isfinite(H)):
        raise NumericalInstabilityError("student_t_elementwise: NaN/Inf in H.")

    return BaselineResult(
        H=H,
        residuals=Y - H @ S,
        converged=converged,
        n_iter=outer + 1,
        log_history={
            "frob_delta": history_frob,
            "df_estimated": df_history,
            "Gamma": Gamma,
        },
    )

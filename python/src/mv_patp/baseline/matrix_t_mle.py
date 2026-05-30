"""Matrix-variate Student-t MLE baseline (heavy-tailed competitor).

Implements the ECM iteration for the matrix-t model

    vec(N) ~ matrix-t(0, Gamma kron Sigma, df)

following the EM derivation of Thompson et al. (2020). This is a robust
competitor to oPMM_alpha that explicitly models heavy tails through the
degrees-of-freedom parameter ``df``.

Algorithm
---------
Define the Mahalanobis statistic per epoch (column of ``N``)

    delta_j^2 = R[:, j]^T Gamma^{-1} R[:, j],   sum_d = sum_j delta_j^2.

The conditional weight from the Gamma latent in the t-mixture representation
is

    w = (df + k * n) / (df + sum_d).

The (H, Gamma, Sigma) updates are the weighted analogues of
:mod:`mv_patp.baseline.wishart_mle`. We do not estimate ``df``; it is a
hyper-parameter typically taken in ``[3, 8]`` to reflect heavy tails.
Trace normalisation ``tr(Gamma) = k`` is enforced per iteration.

Satisfies Requirement R9 (used as heavy-tailed ARE comparator).
Task: Section 11.3.

References
----------
Thompson, Maitra, Meeker, Bastawros (2020). Classification with the
matrix-variate-t distribution. *J. Comput. Graph. Statist.* 29(3).
"""

from __future__ import annotations

import logging

import numpy as np

from mv_patp.baseline._types import BaselineResult
from mv_patp.baseline.ols import fit as _ols_fit
from mv_patp.utils.errors import (
    EMDivergence,
    NumericalInstabilityError,
)

logger = logging.getLogger(__name__)


def _spd_inv(A: np.ndarray, eps: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals_floored = np.maximum(eigvals, eps)
    return (eigvecs * (1.0 / eigvals_floored)) @ eigvecs.T


def fit(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    df: float = 5.0,
    max_iter: int = 50,
    tol: float = 1e-6,
    eigenvalue_floor: float = 1e-9,
    raise_on_divergence: bool = False,
) -> BaselineResult:
    """Fit ``(H, Gamma, Sigma)`` for the matrix-t noise model.

    Parameters
    ----------
    Y : np.ndarray
        Received signal of shape ``(k, n)``.
    S : np.ndarray
        Known training matrix of shape ``(p, n)``, full row rank.
    df : float, default 5.0
        Degrees of freedom of the matrix-t distribution. Smaller values give
        heavier tails. ``df > 2`` is required for finite second moments.
    max_iter : int, default 50
    tol : float, default 1e-6
    eigenvalue_floor : float, default 1e-9
    raise_on_divergence : bool, default False
        If True, raise :class:`EMDivergence` instead of returning a
        non-converged result.

    Returns
    -------
    BaselineResult
        ``H`` of shape ``(k, p)`` plus convergence diagnostics.

    Raises
    ------
    ValueError
        If ``df <= 2``.
    """
    if df <= 2.0:
        raise ValueError(f"df must be > 2 for finite second moments; got {df}.")

    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    k, n = Y.shape

    # Initialise from OLS + sample covariances.
    H = _ols_fit(Y, S).H
    R = Y - H @ S
    Gamma = (R @ R.T) / max(n, 1)
    Gamma = Gamma * (k / np.trace(Gamma))
    Sigma = (R.T @ R) / max(k, 1)
    Sigma = Sigma * (n / np.trace(Sigma))

    history_frob: list[float] = []
    history_w: list[float] = []
    converged = False

    for it in range(1, max_iter + 1):
        Gamma_inv = _spd_inv(Gamma, eigenvalue_floor)
        Sigma_inv = _spd_inv(Sigma, eigenvalue_floor)

        # E-step: weight w from the Gamma latent.
        # Mahalanobis sum: trace(Gamma_inv R Sigma_inv R^T).
        whitened = Gamma_inv @ R @ Sigma_inv  # (k, n)
        sum_d = float(np.einsum("ij,ij->", whitened, R))
        w = (df + k * n) / (df + max(sum_d, eigenvalue_floor))
        history_w.append(w)

        # Weighted M-step for H. Because the weight is scalar, the closed-form
        # collapses to the same expression as Wishart MLE (the scalar weight
        # cancels in the H solve).
        SSig = S @ Sigma_inv
        gram = SSig @ S.T
        gram_inv = _spd_inv(gram, eigenvalue_floor)
        H_new = Y @ Sigma_inv @ S.T @ gram_inv

        if not np.all(np.isfinite(H_new)):
            raise NumericalInstabilityError(
                f"matrix-t H update at iter {it} contains NaN/Inf."
            )

        R_new = Y - H_new @ S

        # Weighted (Gamma, Sigma) update: scaled by w.
        Gamma_new = (w / n) * (R_new @ Sigma_inv @ R_new.T)
        Gamma_new = Gamma_new * (k / max(np.trace(Gamma_new), eigenvalue_floor))
        Gamma_new_inv = _spd_inv(Gamma_new, eigenvalue_floor)
        Sigma_new = (w / k) * (R_new.T @ Gamma_new_inv @ R_new)

        delta = float(np.linalg.norm(H_new - H, ord="fro"))
        history_frob.append(delta)
        H, Gamma, Sigma, R = H_new, Gamma_new, Sigma_new, R_new
        if delta < tol:
            converged = True
            break

    if not converged and raise_on_divergence:
        raise EMDivergence(
            f"matrix-t MLE did not converge in {max_iter} iterations; "
            f"last delta={history_frob[-1] if history_frob else float('nan'):.3e}."
        )

    return BaselineResult(
        H=H,
        residuals=Y - H @ S,
        converged=converged,
        n_iter=it,
        log_history={
            "frob_delta": history_frob,
            "weight_history": history_w,
            "Gamma": Gamma,
            "Sigma": Sigma,
            "df": df,
        },
    )

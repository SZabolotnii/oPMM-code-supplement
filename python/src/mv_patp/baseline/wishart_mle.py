"""Wishart MLE baseline assuming separable Gaussian noise covariance.

Under the model ``Y = H S + N`` with
``vec(N) ~ N(0, Sigma_col kron Gamma_row)``, the MLE alternates:

* M-step (H): given ``Gamma`` and ``Sigma``, solve a generalised LS problem
  using whitened residuals.
* M-step (Gamma, Sigma): given ``H``, update the row covariance from the
  whitened sample covariance and the column covariance from the
  reciprocally-whitened version.

The trace-normalisation ``tr(Gamma) = k`` (Assumption 1 (b) in
``system_model.tex``) is enforced after each (Gamma, Sigma) update so the
factor pair is identifiable.

This estimator is the natural Gaussian baseline for the symmetric
heavy-tailed evaluation of Requirement R9: it shares its mean structure with
oPMM_alpha but replaces the fractional score with an L2 score, so any
empirical advantage of oPMM_alpha over Wishart MLE is attributable to the
score function rather than to the EM scaffold.

Satisfies Requirement R9 (used as ARE comparator).
Task: Section 11.1.
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


def _matrix_inv_with_floor(A: np.ndarray, eps: float) -> np.ndarray:
    """Symmetric positive-definite inverse with eigenvalue floor."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals_floored = np.maximum(eigvals, eps)
    return (eigvecs * (1.0 / eigvals_floored)) @ eigvecs.T


def fit(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    eigenvalue_floor: float = 1e-9,
    raise_on_divergence: bool = False,
) -> BaselineResult:
    """Fit ``(H, Gamma, Sigma)`` by separable-covariance Gaussian MLE.

    Parameters
    ----------
    Y : np.ndarray
        Received signal of shape ``(k, n)``.
    S : np.ndarray
        Known training matrix of shape ``(p, n)``, full row rank.
    max_iter : int, default 50
        Maximum number of outer alternating iterations.
    tol : float, default 1e-6
        Stop when ``||H_t - H_{t-1}||_F`` drops below this threshold.
    eigenvalue_floor : float, default 1e-9
        Symmetric PD floor applied to eigenvalues of ``Gamma`` and ``Sigma``
        when inverting them.
    raise_on_divergence : bool, default False
        If True, raise :class:`EMDivergence` when ``max_iter`` is reached
        without satisfying ``tol``. Otherwise the last iterate is returned
        with ``converged=False``.

    Returns
    -------
    BaselineResult
        ``H`` of shape ``(k, p)`` plus convergence diagnostics.

    Raises
    ------
    EMDivergence
        Only when ``raise_on_divergence=True`` and convergence is not
        achieved by ``max_iter``.
    NumericalInstabilityError
        If any iterate contains NaN/Inf.
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    k, n = Y.shape
    p = S.shape[0]

    # Initialise H by OLS, Gamma and Sigma by sample covariances.
    H = _ols_fit(Y, S).H
    R = Y - H @ S
    Gamma = (R @ R.T) / max(n, 1)
    # Trace-normalise once before iterating.
    Gamma = Gamma * (k / np.trace(Gamma))
    Sigma = (R.T @ R) / max(k, 1)
    Sigma = Sigma * (n / np.trace(Sigma))

    history_frob: list[float] = []
    converged = False

    for it in range(1, max_iter + 1):
        Gamma_inv = _matrix_inv_with_floor(Gamma, eigenvalue_floor)
        Sigma_inv = _matrix_inv_with_floor(Sigma, eigenvalue_floor)

        # M-step for H given (Gamma, Sigma):
        #   vec(Y - H S) Gamma_inv kron Sigma_inv (Y - H S) is minimised by
        #     H = Gamma_inv^{-1} ... — but with Gamma kron Sigma separable
        #   structure, the closed form is H = Y Sigma_inv S^T (S Sigma_inv S^T)^{-1}.
        # (The Gamma weighting drops out for H because the loss in H is
        # quadratic and decoupled from Gamma in the Kronecker decomposition.)
        SSig = S @ Sigma_inv
        gram = SSig @ S.T  # (p, p)
        gram_inv = _matrix_inv_with_floor(gram, eigenvalue_floor)
        H_new = Y @ Sigma_inv @ S.T @ gram_inv  # (k, p)

        if not np.all(np.isfinite(H_new)):
            raise NumericalInstabilityError(
                f"Wishart-MLE H update at iter {it} contains NaN/Inf."
            )

        # M-step for Gamma, Sigma given H_new.
        R = Y - H_new @ S
        Gamma_new = (R @ Sigma_inv @ R.T) / n
        Gamma_new = Gamma_new * (k / max(np.trace(Gamma_new), eigenvalue_floor))

        Gamma_new_inv = _matrix_inv_with_floor(Gamma_new, eigenvalue_floor)
        Sigma_new = (R.T @ Gamma_new_inv @ R) / k

        delta = float(np.linalg.norm(H_new - H, ord="fro"))
        history_frob.append(delta)
        H, Gamma, Sigma = H_new, Gamma_new, Sigma_new
        if delta < tol:
            converged = True
            break

    if not converged and raise_on_divergence:
        raise EMDivergence(
            f"Wishart MLE did not converge in {max_iter} iterations; "
            f"last delta={history_frob[-1] if history_frob else float('nan'):.3e}."
        )

    residuals = Y - H @ S
    return BaselineResult(
        H=H,
        residuals=residuals,
        converged=converged,
        n_iter=it,
        log_history={"frob_delta": history_frob, "Gamma": Gamma, "Sigma": Sigma},
    )

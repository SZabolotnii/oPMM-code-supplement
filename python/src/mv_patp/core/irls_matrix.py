"""Weighted matrix least-squares solver for the EM-PATP M-step.

Solves the weighted normal-equations problem
``vec(H) = [B^T diag(vec(W)) B]^{-1} B^T diag(vec(W)) vec(Y_tilde)``
where ``B = (S Sigma_hat^{-1/2})^T kron Gamma_hat^{-1/2}`` and
``Y_tilde = Gamma_hat^{-1/2} Y Sigma_hat^{-1/2}``. The construction
turns the matrix MAL channel-identification problem into a standard
weighted GLS in vec form, after row-side and column-side whitening.

The Kronecker-structured operator allows a (kn x kp) explicit build,
followed by a (kp x kp) linear solve. With the typical scale of
this paper (k <= 8, p <= 4, n <= 1024) the (kp x kp) matrix has
at most 32 rows and the dense solve is trivially fast.

Satisfies Requirement R6.1 (M-step sub-component).
Task: Section 6.1.

References
----------
Spec.md Section 5.1 EM-PATP pseudocode and Section 4.3 (whitening).
design.md Section L1 (function signature).
"""

from __future__ import annotations

import logging

import numpy as np

from mv_patp.core.whitening import matrix_inverse_sqrt
from mv_patp.utils.errors import NumericalInstabilityError

logger = logging.getLogger(__name__)


def solve_weighted_matrix_LS(
    Y: np.ndarray,
    S: np.ndarray,
    W: np.ndarray | None,
    Gamma_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    *,
    eigenvalue_floor: float = 1e-6,
    regularisation: float = 1e-12,
    _Sigma_inv_sqrt: np.ndarray | None = None,
) -> np.ndarray:
    """Solve the weighted-LS normal equations for the channel matrix ``H``.

    Parameters
    ----------
    Y : np.ndarray
        Received signal of shape ``(k, n)``.
    S : np.ndarray
        Known training matrix of shape ``(p, n)``.
    W : np.ndarray or None
        Weight matrix of shape ``(k, n)`` applied element-wise on the
        whitened residuals. ``None`` means unit weights.
    Gamma_hat : np.ndarray
        Current estimate of the row covariance, shape ``(k, k)``.
    Sigma_hat : np.ndarray
        Current estimate of the column covariance, shape ``(n, n)``.
    eigenvalue_floor : float, default 1e-6
        SPD eigenvalue floor used when forming Gamma_hat^{-1/2} and
        Sigma_hat^{-1/2}.
    regularisation : float, default 1e-12
        Tikhonov ridge added to the (kp, kp) Gram matrix for numerical
        stability when conditioning is poor.

    Returns
    -------
    np.ndarray
        Estimated ``H`` of shape ``(k, p)``.

    Raises
    ------
    ValueError
        If shapes are incompatible.
    NumericalInstabilityError
        If the solution contains NaN/Inf.
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    Gamma_hat = np.asarray(Gamma_hat, dtype=np.float64)
    Sigma_hat = np.asarray(Sigma_hat, dtype=np.float64)

    if Y.ndim != 2 or S.ndim != 2:
        raise ValueError(f"Y and S must be 2-D; got Y.ndim={Y.ndim}, S.ndim={S.ndim}.")
    k, n = Y.shape
    p, n_S = S.shape
    if n_S != n:
        raise ValueError(f"Y has n={n} but S has n={n_S}; shapes must agree.")
    if Gamma_hat.shape != (k, k):
        raise ValueError(f"Gamma_hat must be ({k},{k}); got {Gamma_hat.shape}.")
    if Sigma_hat.shape != (n, n):
        raise ValueError(f"Sigma_hat must be ({n},{n}); got {Sigma_hat.shape}.")
    if W is None:
        W = np.ones((k, n), dtype=np.float64)
    else:
        W = np.asarray(W, dtype=np.float64)
        if W.shape != (k, n):
            raise ValueError(f"W must be ({k},{n}); got {W.shape}.")

    # --- Step 1: whitening factors -----------------------------------------
    G_inv_sqrt = matrix_inverse_sqrt(Gamma_hat, eps=eigenvalue_floor)  # (k, k)
    # Accept a pre-computed Sigma inverse sqrt to avoid repeated O(n³) eigh
    # when Sigma is constant across calls (e.g. "white" identifiability → I_n).
    S_inv_sqrt = (
        _Sigma_inv_sqrt if _Sigma_inv_sqrt is not None
        else matrix_inverse_sqrt(Sigma_hat, eps=eigenvalue_floor)
    )  # (n, n)

    # --- Step 2: whitened observation Y_tilde -----------------------------
    Y_tilde = G_inv_sqrt @ Y @ S_inv_sqrt  # (k, n)

    # --- Step 3: Kronecker operator B = (S Sigma^{-1/2})^T kron Gamma^{-1/2}
    # (S Sigma^{-1/2}) has shape (p, n); its transpose is (n, p).
    A = (S @ S_inv_sqrt).T                    # (n, p)
    B = np.kron(A, G_inv_sqrt)                # (k*n, k*p)

    # --- Step 4: weighted normal equations --------------------------------
    # Column-major vec to match Kronecker convention vec(X M Y) = (Y^T kron X) vec(M)
    d = W.flatten(order="F")                  # (k*n,)
    y_vec = Y_tilde.flatten(order="F")        # (k*n,)

    # B^T D B  and  B^T D y, where D = diag(d)
    Bd = B * d[:, None]                       # broadcast: row-scale
    gram = B.T @ Bd                           # (k*p, k*p)
    rhs = B.T @ (d * y_vec)                   # (k*p,)

    # Tikhonov ridge for numerical stability of poorly conditioned grams.
    gram_reg = gram + regularisation * np.eye(gram.shape[0])

    # --- Step 5: solve and reshape ----------------------------------------
    try:
        vec_H = np.linalg.solve(gram_reg, rhs)
    except np.linalg.LinAlgError:
        # Fall back to least-squares pseudo-inverse if singular.
        logger.warning("Normal-equations gram is singular; using lstsq fallback.")
        vec_H, *_ = np.linalg.lstsq(gram_reg, rhs, rcond=None)

    H = vec_H.reshape((k, p), order="F")

    if not np.all(np.isfinite(H)):
        raise NumericalInstabilityError(
            "solve_weighted_matrix_LS produced NaN/Inf in H."
        )
    return H

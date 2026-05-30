"""Whitening utilities for the EM-PATP estimator.

Provides matrix inverse square root and residual whitening operations used
in the E-step of the EM-PATP algorithm (Spec.md §5.2, §5.3).

Satisfies Requirement R6.1 (whitening sub-step).
"""

from __future__ import annotations

import numpy as np

from mv_patp.utils.errors import NumericalInstabilityError


def matrix_inverse_sqrt(A: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute the symmetric matrix inverse square root A^{-1/2}.

    Uses eigendecomposition with a floor on small eigenvalues to handle
    near-singular (positive semi-definite) inputs:

        A = V diag(λ) V^T  →  A^{-1/2} = V diag(1 / sqrt(max(λ, eps))) V^T

    Eigenvalues below ``eps`` are floored before inversion, preventing
    numerical blow-up on rank-deficient or ill-conditioned matrices.

    Satisfies Requirement R6.1 (whitening sub-step).

    Parameters
    ----------
    A : np.ndarray
        Square symmetric positive semi-definite matrix of shape ``(m, m)``.
    eps : float, optional
        Floor value applied to eigenvalues before taking the inverse square
        root.  Default is ``1e-6``.

    Returns
    -------
    np.ndarray
        Matrix A^{-1/2} of shape ``(m, m)``.

    Raises
    ------
    NumericalInstabilityError
        If the result contains NaN or Inf values.

    References
    ----------
    Spec.md §5.2 — whitening transform definition.
    """
    A = np.asarray(A, dtype=np.float64)

    # Eigendecomposition: A = V diag(λ) V^T
    # np.linalg.eigh assumes symmetric input and returns real eigenvalues
    # in ascending order.
    eigenvalues, eigenvectors = np.linalg.eigh(A)  # (m,), (m, m)

    # Floor eigenvalues at eps to avoid division by zero / blow-up
    eigenvalues_floored = np.maximum(eigenvalues, eps)

    # Inverse square root of floored eigenvalues
    inv_sqrt_eigenvalues = 1.0 / np.sqrt(eigenvalues_floored)  # (m,)

    # Reconstruct: V @ diag(1/sqrt(λ)) @ V^T
    # Equivalent to (V * inv_sqrt_λ) @ V^T
    A_inv_sqrt = (eigenvectors * inv_sqrt_eigenvalues) @ eigenvectors.T  # (m, m)

    if not np.isfinite(A_inv_sqrt).all():
        raise NumericalInstabilityError(
            "matrix_inverse_sqrt: NaN or Inf detected in A^{-1/2}. "
            f"Input eigenvalue range: [{eigenvalues.min():.3e}, {eigenvalues.max():.3e}], "
            f"eps={eps:.3e}."
        )

    return A_inv_sqrt


def whiten_residuals(
    R: np.ndarray,
    Gamma_hat: np.ndarray,
    Sigma_hat: np.ndarray,
) -> np.ndarray:
    """Whiten residual matrix R using estimated row and column covariances.

    Computes the doubly-whitened residual matrix:

        R̃ = Γ̂^{-1/2} R Σ̂^{-1/2}

    where Γ̂^{-1/2} and Σ̂^{-1/2} are obtained via :func:`matrix_inverse_sqrt`.
    This operation removes the second-order structure from both the row
    (antenna) and column (time) dimensions, yielding approximately i.i.d.
    entries under the MAL model.

    Satisfies Requirement R6.1 (whitening sub-step).

    Parameters
    ----------
    R : np.ndarray
        Residual matrix of shape ``(k, n)``.
    Gamma_hat : np.ndarray
        Estimated row covariance matrix of shape ``(k, k)``, symmetric
        positive semi-definite.
    Sigma_hat : np.ndarray
        Estimated column covariance matrix of shape ``(n, n)``, symmetric
        positive semi-definite.

    Returns
    -------
    np.ndarray
        Whitened residual matrix Γ̂^{-1/2} R Σ̂^{-1/2} of shape ``(k, n)``.

    Raises
    ------
    NumericalInstabilityError
        If the result contains NaN or Inf values.

    References
    ----------
    Spec.md §5.2 — whitening transform definition.
    """
    R = np.asarray(R, dtype=np.float64)
    Gamma_hat = np.asarray(Gamma_hat, dtype=np.float64)
    Sigma_hat = np.asarray(Sigma_hat, dtype=np.float64)

    # Compute inverse square roots
    Gamma_inv_sqrt = matrix_inverse_sqrt(Gamma_hat)   # (k, k)
    Sigma_inv_sqrt = matrix_inverse_sqrt(Sigma_hat)   # (n, n)

    # R̃ = Γ̂^{-1/2} R Σ̂^{-1/2}
    R_whitened = Gamma_inv_sqrt @ R @ Sigma_inv_sqrt   # (k, n)

    if not np.isfinite(R_whitened).all():
        raise NumericalInstabilityError(
            "whiten_residuals: NaN or Inf detected in whitened residuals."
        )

    return R_whitened

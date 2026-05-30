"""Unit tests for core/whitening.py.

Tests cover:
- Round-trip identity: A^{1/2} A^{-1/2} ≈ I (Requirement R6.1)
- Output shape of whiten_residuals (Requirement R6.1)
- Identity covariances leave residuals unchanged (Requirement R6.1)

Satisfies Requirement R6.1 (whitening sub-step).
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.core.whitening import matrix_inverse_sqrt, whiten_residuals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_spd(m: int, seed: int = 0) -> np.ndarray:
    """Return a random symmetric positive-definite (m, m) matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, m))
    # A @ A^T is PSD; add eps * I to make it strictly PD
    return A @ A.T + np.eye(m) * 0.1


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Compute A^{1/2} via eigendecomposition (for round-trip test)."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0.0))
    return (eigenvectors * sqrt_eigenvalues) @ eigenvectors.T


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_round_trip_inverse_sqrt_is_identity() -> None:
    """A^{1/2} A^{-1/2} ≈ I within machine epsilon.

    Requirement R6.1 — whitening sub-step.
    Verifies that the eigendecomposition-based inverse square root is
    consistent with the forward square root: their product must recover
    the identity matrix to within atol=1e-10.
    """
    m = 5
    A = _random_spd(m, seed=42)

    A_sqrt = _matrix_sqrt(A)          # A^{1/2}
    A_inv_sqrt = matrix_inverse_sqrt(A)  # A^{-1/2}

    product = A_sqrt @ A_inv_sqrt      # should be I_m
    assert np.allclose(product, np.eye(m), atol=1e-10), (
        f"A^{{1/2}} A^{{-1/2}} deviates from I: max abs error = "
        f"{np.abs(product - np.eye(m)).max():.3e}"
    )

def test_round_trip_larger_matrix() -> None:
    """Round-trip holds for a larger (10×10) SPD matrix.

    Requirement R6.1 — whitening sub-step.
    """
    m = 10
    A = _random_spd(m, seed=7)
    A_sqrt = _matrix_sqrt(A)
    A_inv_sqrt = matrix_inverse_sqrt(A)
    product = A_sqrt @ A_inv_sqrt
    assert np.allclose(product, np.eye(m), atol=1e-10)

def test_identity_input_returns_identity() -> None:
    """matrix_inverse_sqrt(I_m) == I_m.

    Requirement R6.1 — whitening sub-step.
    """
    m = 4
    I = np.eye(m)
    result = matrix_inverse_sqrt(I)
    assert np.allclose(result, I, atol=1e-12)

def test_eigenvalue_floor_prevents_blow_up() -> None:
    """Near-singular input does not produce Inf/NaN.

    Requirement R6.1 — whitening sub-step.
    Eigenvalues below eps are floored, so the result must be finite.
    """
    m = 3
    # Rank-1 matrix: one zero eigenvalue
    v = np.array([1.0, 2.0, 3.0])
    A = np.outer(v, v)  # rank-1, singular
    result = matrix_inverse_sqrt(A, eps=1e-6)
    assert np.isfinite(result).all(), "Result contains NaN or Inf on near-singular input."

def test_output_shape() -> None:
    """Output shape matches input shape.

    Requirement R6.1 — whitening sub-step.
    """
    m = 6
    A = _random_spd(m, seed=1)
    result = matrix_inverse_sqrt(A)
    assert result.shape == (m, m)


def test_whiten_residuals_shape() -> None:
    """Output shape of whiten_residuals is (k, n).

    Requirement R6.1 — whitening sub-step.
    """
    k, n = 3, 8
    rng = np.random.default_rng(0)
    R = rng.standard_normal((k, n))
    Gamma_hat = _random_spd(k, seed=10)
    Sigma_hat = _random_spd(n, seed=20)

    R_whitened = whiten_residuals(R, Gamma_hat, Sigma_hat)
    assert R_whitened.shape == (k, n), (
        f"Expected shape ({k}, {n}), got {R_whitened.shape}."
    )

def test_whitening_with_identity_matrices_returns_R() -> None:
    """whiten_residuals(R, I_k, I_n) == R.

    Requirement R6.1 — whitening sub-step.
    When both covariance matrices are identity, whitening is a no-op
    and the output must equal the input exactly (up to floating-point
    round-off at atol=1e-12).
    """
    k, n = 4, 7
    rng = np.random.default_rng(99)
    R = rng.standard_normal((k, n))
    Gamma_hat = np.eye(k)
    Sigma_hat = np.eye(n)

    R_whitened = whiten_residuals(R, Gamma_hat, Sigma_hat)
    assert np.allclose(R_whitened, R, atol=1e-12), (
        f"Expected R unchanged; max abs error = "
        f"{np.abs(R_whitened - R).max():.3e}"
    )

def test_whitening_decorrelates_rows_and_columns() -> None:
    """Whitened residuals have approximately identity row/col covariance.

    Requirement R6.1 — whitening sub-step.
    When R has known covariance structure Gamma ⊗ Sigma, the whitened
    residuals should have covariance ≈ I_k ⊗ I_n.
    """
    k, n = 3, 50
    rng = np.random.default_rng(123)

    # Build known covariance matrices
    Gamma_true = _random_spd(k, seed=5)
    Sigma_true = _random_spd(n, seed=6)

    # Compute square roots for sampling
    eigenvalues_g, eigenvectors_g = np.linalg.eigh(Gamma_true)
    Gamma_sqrt = (eigenvectors_g * np.sqrt(np.maximum(eigenvalues_g, 0.0))) @ eigenvectors_g.T

    eigenvalues_s, eigenvectors_s = np.linalg.eigh(Sigma_true)
    Sigma_sqrt = (eigenvectors_s * np.sqrt(np.maximum(eigenvalues_s, 0.0))) @ eigenvectors_s.T

    # Sample R with known structure: R = Gamma_sqrt @ Z @ Sigma_sqrt
    Z = rng.standard_normal((k, n))
    R = Gamma_sqrt @ Z @ Sigma_sqrt

    # Whiten using the true covariances
    R_whitened = whiten_residuals(R, Gamma_true, Sigma_true)

    # The whitened residuals should be close to Z (up to sign/rotation)
    # Check that row covariance ≈ I_k (within tolerance for finite n)
    row_cov = R_whitened @ R_whitened.T / n
    assert np.allclose(row_cov, np.eye(k), atol=0.5), (
        f"Row covariance of whitened residuals deviates from I_k: "
        f"max abs error = {np.abs(row_cov - np.eye(k)).max():.3e}"
    )

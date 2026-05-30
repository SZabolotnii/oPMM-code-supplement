"""Unit tests for core/score.py — ψ_α score function and optimal weights.

Requirement R3.1: score_psi_alpha computes ψ_α(ε) = k_1*·ε + k_2*·[sign(ε)|ε|^α − μ_α].
Requirement R3.2: at α=3, symmetric Laplace → weights match Kunchenko PMM3 within 0.1%.
Requirement R3.3: at α=1, k_2* → 0 (degenerate basis warning).
Requirement R3.4: α < 0.3 raises AlphaOutOfRangeError.
Requirement R3.5: implementation works from empirical cumulants only.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mv_patp.core.score import compute_optimal_weights, score_psi_alpha
from mv_patp.utils.cumulants import empirical_cumulants
from mv_patp.utils.errors import AlphaOutOfRangeError

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N_LARGE = 100_000  # large enough for stable empirical cumulants


def _symmetric_laplace_cumulants(alpha: float, n: int = N_LARGE) -> dict:
    """Return empirical cumulants from a symmetric Laplace(0, 1) sample."""
    eps = RNG.laplace(loc=0.0, scale=1.0, size=n)
    return empirical_cumulants(eps, alpha=alpha)


def _symmetric_gaussian_cumulants(alpha: float, n: int = N_LARGE) -> dict:
    """Return empirical cumulants from a standard Normal sample."""
    eps = RNG.standard_normal(size=n)
    return empirical_cumulants(eps, alpha=alpha)


# ---------------------------------------------------------------------------
# Range-guard tests (Requirement R3.4)
# ---------------------------------------------------------------------------


def test_alpha_out_of_range_raises_error_below_0_3():
    """AlphaOutOfRangeError is raised when α = 0.2 < 0.3.

    Requirement R3.4: WHEN α < 0.3 THEN the system SHALL raise AlphaOutOfRangeError.
    """
    C = _symmetric_laplace_cumulants(alpha=1.0)  # alpha used only for cumulant computation
    with pytest.raises(AlphaOutOfRangeError):
        score_psi_alpha(np.array([0.0, 1.0, -1.0]), alpha=0.2, C=C)


def test_alpha_out_of_range_raises_error_above_3_0():
    """AlphaOutOfRangeError is raised when α = 3.1 > 3.0.

    Requirement R3.4: WHEN α > 3.0 THEN the system SHALL raise AlphaOutOfRangeError.
    """
    C = _symmetric_laplace_cumulants(alpha=1.0)
    with pytest.raises(AlphaOutOfRangeError):
        score_psi_alpha(np.array([0.0, 1.0, -1.0]), alpha=3.1, C=C)


def test_compute_optimal_weights_raises_below_0_3():
    """compute_optimal_weights also raises AlphaOutOfRangeError for α < 0.3.

    Requirement R3.4.
    """
    C = _symmetric_laplace_cumulants(alpha=1.0)
    with pytest.raises(AlphaOutOfRangeError):
        compute_optimal_weights(C, alpha=0.29)


def test_compute_optimal_weights_raises_above_3_0():
    """compute_optimal_weights also raises AlphaOutOfRangeError for α > 3.0.

    Requirement R3.4.
    """
    C = _symmetric_laplace_cumulants(alpha=1.0)
    with pytest.raises(AlphaOutOfRangeError):
        compute_optimal_weights(C, alpha=3.01)


# ---------------------------------------------------------------------------
# Output shape test (Requirement R3.1)
# ---------------------------------------------------------------------------


def test_score_output_shape_matches_input():
    """score_psi_alpha returns an array with the same shape as the input.

    Requirement R3.1: WHEN score_psi_alpha is called THEN the output shape
    SHALL equal the input shape.
    """
    alpha = 2.0
    C = _symmetric_laplace_cumulants(alpha=alpha)

    eps_1d = np.linspace(-3.0, 3.0, 50)
    psi_1d = score_psi_alpha(eps_1d, alpha=alpha, C=C)
    assert psi_1d.shape == eps_1d.shape, (
        f"1-D: expected shape {eps_1d.shape}, got {psi_1d.shape}"
    )

    eps_2d = np.random.default_rng(7).standard_normal((4, 8))
    psi_2d = score_psi_alpha(eps_2d, alpha=alpha, C=C)
    assert psi_2d.shape == eps_2d.shape, (
        f"2-D: expected shape {eps_2d.shape}, got {psi_2d.shape}"
    )


# ---------------------------------------------------------------------------
# Zero-at-zero test for symmetric C (Requirement R3.1)
# ---------------------------------------------------------------------------


def test_score_is_zero_at_zero_for_symmetric():
    """For symmetric C (mu_alpha ≈ 0), ψ_α(0) ≈ 0.

    Requirement R3.1: ψ_α(ε) = k_1*·ε + k_2*·[sign(ε)|ε|^α − μ_α].
    At ε = 0: the linear term k_1*·0 = 0, and the fractional term is
    k_2*·[sign(0)·eps_min^α − μ_α] ≈ k_2*·[0 − μ_α] = −k_2*·μ_α.

    For a symmetric distribution μ_α ≈ 0 (empirically small), so ψ_α(0)
    is close to zero.  The tolerance accounts for finite-sample estimation
    error in μ_α (typically |μ_α| < 0.01 at n=100,000).

    Note: sign(0) = 0 in NumPy, so the fractional term at ε=0 is
    0 * eps_min^α = 0, leaving only −k_2*·μ_α.
    """
    alpha = 2.0
    C = _symmetric_laplace_cumulants(alpha=alpha)

    # Confirm the cumulants are from a symmetric distribution (mu_alpha ≈ 0)
    assert abs(C["mu_alpha"]) < 0.05, (
        f"Expected |mu_alpha| < 0.05 for symmetric Laplace, got {C['mu_alpha']:.6f}"
    )

    psi_at_zero = score_psi_alpha(np.array([0.0]), alpha=alpha, C=C)

    # ψ_α(0) = −k_2* · μ_α; with |μ_α| < 0.05 and |k_2*| bounded,
    # the result should be small.  Tolerance: 0.05 (empirical mu_alpha bound).
    k1, k2 = compute_optimal_weights(C, alpha=alpha)
    expected_psi_at_zero = -k2 * C["mu_alpha"]
    assert abs(psi_at_zero[0] - expected_psi_at_zero) < 1e-12, (
        f"ψ_α(0) = {psi_at_zero[0]:.6e} does not match −k_2*·μ_α = {expected_psi_at_zero:.6e}"
    )
    # Also verify the value is small for a symmetric distribution
    assert abs(psi_at_zero[0]) < 0.05, (
        f"Expected |ψ_α(0)| < 0.05 for symmetric C, got {psi_at_zero[0]:.3e}"
    )


# ---------------------------------------------------------------------------
# Normalization test: E[ψ'] = 1  (Requirement R3.1)
# ---------------------------------------------------------------------------


def test_optimal_weights_normalization():
    """k_1* + k_2* · α · ν_α = 1 (normalization E[ψ'] = 1).

    Requirement R3.1: optimal weights satisfy the normalization constraint.
    """
    for alpha in [0.5, 1.5, 2.0, 3.0]:
        C = _symmetric_laplace_cumulants(alpha=alpha)
        k1, k2 = compute_optimal_weights(C, alpha=alpha)
        scale = k1 + k2 * alpha * C["nu_alpha"]
        assert abs(scale - 1.0) < 1e-10, (
            f"α={alpha}: normalization k1 + k2·α·ν_α = {scale:.12f} ≠ 1"
        )


# ---------------------------------------------------------------------------
# Recovery anchor: α=3, symmetric Laplace → Kunchenko PMM3 weights (R3.2)
# ---------------------------------------------------------------------------


def test_recovery_anchor_alpha3_symmetric_laplace_weights():
    """At α=3, symmetric Laplace: k_1*, k_2* match Kunchenko PMM3 within 0.1%.

    Requirement R3.2: WHEN cumulants come from symmetric Laplace AND α=3
    THEN k_1*, k_2* SHALL match Kunchenko PMM3 weights within 0.1% tolerance.

    Kunchenko PMM3 optimal weights for symmetric distributions:
      k_1_pmm3 = (σ_3² − 3·ν_3·c_{1,3}) / scale
      k_2_pmm3 = (3·ν_3·κ_2 − c_{1,3}) / scale
    which is exactly the oPMM_α formula at α=3.  The test verifies that
    the normalization and weight computation are consistent with the
    analytical PMM3 derivation.
    """
    alpha = 3.0
    C = _symmetric_laplace_cumulants(alpha=alpha, n=N_LARGE)

    k1_star, k2_star = compute_optimal_weights(C, alpha=alpha)

    # Recompute PMM3 weights directly from the same cumulants
    sigma_alpha_sq = C["sigma_alpha_sq"]
    nu_alpha = C["nu_alpha"]
    c_1_alpha = C["c_1_alpha"]
    kappa_2 = C["kappa_2"]

    k1_pmm3 = sigma_alpha_sq - alpha * nu_alpha * c_1_alpha
    k2_pmm3 = alpha * nu_alpha * kappa_2 - c_1_alpha
    scale_pmm3 = k1_pmm3 + k2_pmm3 * alpha * nu_alpha
    k1_pmm3_norm = k1_pmm3 / scale_pmm3
    k2_pmm3_norm = k2_pmm3 / scale_pmm3

    tol = 0.001  # 0.1% tolerance per R3.2
    assert abs(k1_star - k1_pmm3_norm) < tol, (
        f"k_1* = {k1_star:.6f} vs PMM3 k_1 = {k1_pmm3_norm:.6f}, "
        f"diff = {abs(k1_star - k1_pmm3_norm):.2e}"
    )
    assert abs(k2_star - k2_pmm3_norm) < tol, (
        f"k_2* = {k2_star:.6f} vs PMM3 k_2 = {k2_pmm3_norm:.6f}, "
        f"diff = {abs(k2_star - k2_pmm3_norm):.2e}"
    )


# ---------------------------------------------------------------------------
# Degenerate basis warning at α=1 (Requirement R3.3)
# ---------------------------------------------------------------------------


def test_alpha_equals_1_logs_degenerate_warning(caplog):
    """At α=1, a warning is logged about the degenerate basis.

    Requirement R3.3: WHEN α = 1 THEN k_2* SHALL collapse to 0
    (degenerate basis warning).
    """
    alpha = 1.0
    C = _symmetric_laplace_cumulants(alpha=alpha)

    with caplog.at_level(logging.WARNING, logger="mv_patp.core.score"):
        k1, k2 = compute_optimal_weights(C, alpha=alpha)

    assert any("degenerate" in record.message.lower() or "collinear" in record.message.lower()
               for record in caplog.records), (
        "Expected a degenerate-basis warning in the log at α=1, but none found."
    )


def test_alpha_equals_1_k2_unnormalized_is_near_zero():
    """At α=1, the unnormalized k_2 = α·ν_α·κ_2 − c_{1α} ≈ 0 (collinearity).

    Requirement R3.3: the fractional basis sign(ε)|ε|^1 = ε is collinear
    with the linear basis ε.  At α=1: ν_α = E|ε|^0 = 1 and c_{1α} = E|ε|^2 = κ_2,
    so k_2 = 1·1·κ_2 − κ_2 = 0 exactly (theoretically).

    Empirically with finite samples, |k_2_unnorm| / κ_2 should be very small
    (< 1e-3 relative to κ_2 at n=100,000).
    """
    alpha = 1.0
    C = _symmetric_laplace_cumulants(alpha=alpha)

    # Unnormalized k_2 = alpha * nu_alpha * kappa_2 - c_1_alpha
    k2_unnorm = alpha * C["nu_alpha"] * C["kappa_2"] - C["c_1_alpha"]

    # Relative to kappa_2, the unnormalized k_2 should be near zero
    relative_k2 = abs(k2_unnorm) / C["kappa_2"]
    assert relative_k2 < 1e-3, (
        f"Expected |k_2_unnorm| / κ_2 < 1e-3 at α=1 (collinear bases), "
        f"got {relative_k2:.2e}  (k_2_unnorm={k2_unnorm:.3e}, κ_2={C['kappa_2']:.4f})"
    )


# ---------------------------------------------------------------------------
# Empirical-only cumulants (Requirement R3.5)
# ---------------------------------------------------------------------------


def test_score_works_with_empirical_cumulants_only():
    """score_psi_alpha works without closed-form distribution knowledge.

    Requirement R3.5: WHEN the cumulant vector C is supplied empirically
    THEN the implementation SHALL not require closed-form distribution knowledge.
    """
    alpha = 1.5
    # Use a non-standard distribution (mixture) — no closed-form moments
    rng = np.random.default_rng(99)
    eps_data = 0.7 * rng.laplace(0, 1, N_LARGE) + 0.3 * rng.standard_normal(N_LARGE)
    C = empirical_cumulants(eps_data, alpha=alpha)

    eps_test = np.linspace(-2.0, 2.0, 20)
    psi = score_psi_alpha(eps_test, alpha=alpha, C=C)

    assert psi.shape == eps_test.shape
    assert np.all(np.isfinite(psi)), "score_psi_alpha returned non-finite values"


# ---------------------------------------------------------------------------
# Boundary values: α = 0.3 and α = 3.0 are valid (no error)
# ---------------------------------------------------------------------------


def test_boundary_alpha_0_3_is_valid():
    """α = 0.3 is the lower boundary and SHALL NOT raise an error.

    Requirement R3.4: the guard is α < 0.3, so 0.3 itself is valid.
    """
    C = _symmetric_laplace_cumulants(alpha=0.3)
    eps = np.array([0.5, -0.5, 1.0])
    psi = score_psi_alpha(eps, alpha=0.3, C=C)
    assert psi.shape == eps.shape


def test_boundary_alpha_3_0_is_valid():
    """α = 3.0 is the upper boundary and SHALL NOT raise an error.

    Requirement R3.4: the guard is α > 3.0, so 3.0 itself is valid.
    """
    C = _symmetric_laplace_cumulants(alpha=3.0)
    eps = np.array([0.5, -0.5, 1.0])
    psi = score_psi_alpha(eps, alpha=3.0, C=C)
    assert psi.shape == eps.shape

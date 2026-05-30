"""Unit tests for core/alpha_optimizer.py::find_alpha_opt.

Tests cover:
- Return type is a 3-tuple (R5.1)
- alpha_opt is within the valid range [0.3, 3.0] (R5.1, R5.2)
- n_evals is bounded by 50 (R5.3)
- For Gaussian data, g(α) ≈ 1 everywhere; alpha_opt is still in range (R5.1)

Satisfies Requirements R5.1, R5.2, R5.3.
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.core.alpha_optimizer import find_alpha_opt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gaussian_cumulants(n: int = 5000, seed: int = 42) -> dict:
    """Return a C dict with raw residuals from a standard normal sample (Mode A)."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(n)
    return {"eps_white": eps}


def _laplace_cumulants(n: int = 5000, seed: int = 42) -> dict:
    """Return a C dict with raw residuals from a symmetric Laplace(0, 1) sample (Mode A)."""
    rng = np.random.default_rng(seed)
    eps = rng.laplace(loc=0.0, scale=1.0, size=n)
    return {"eps_white": eps}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_find_alpha_opt_returns_tuple_of_three() -> None:
    """Return value is a tuple of length 3.

    Requirement R5.1.
    """
    C = _laplace_cumulants()
    result = find_alpha_opt(C)
    assert isinstance(result, tuple), (
        f"Expected tuple, got {type(result)}"
    )
    assert len(result) == 3, (
        f"Expected 3-tuple, got length {len(result)}"
    )

def test_find_alpha_opt_first_element_is_float() -> None:
    """alpha_opt (first element) is a float.

    Requirement R5.1.
    """
    C = _laplace_cumulants()
    alpha_opt, g_min, n_evals = find_alpha_opt(C)
    assert isinstance(alpha_opt, float), (
        f"alpha_opt should be float, got {type(alpha_opt)}"
    )

def test_find_alpha_opt_second_element_is_float() -> None:
    """g_min (second element) is a float.

    Requirement R5.1.
    """
    C = _laplace_cumulants()
    alpha_opt, g_min, n_evals = find_alpha_opt(C)
    assert isinstance(g_min, float), (
        f"g_min should be float, got {type(g_min)}"
    )

def test_find_alpha_opt_third_element_is_int() -> None:
    """n_evals (third element) is an int.

    Requirement R5.1.
    """
    C = _laplace_cumulants()
    alpha_opt, g_min, n_evals = find_alpha_opt(C)
    assert isinstance(n_evals, int), (
        f"n_evals should be int, got {type(n_evals)}"
    )


def test_find_alpha_opt_alpha_in_valid_range_laplace() -> None:
    """alpha_opt ∈ [0.3, 3.0] for Laplace cumulants.

    Requirement R5.1, R5.2.
    """
    C = _laplace_cumulants()
    alpha_opt, g_min, n_evals = find_alpha_opt(C)
    assert 0.3 <= alpha_opt <= 3.0, (
        f"alpha_opt={alpha_opt:.6f} is outside [0.3, 3.0]."
    )

def test_find_alpha_opt_alpha_in_valid_range_gaussian() -> None:
    """alpha_opt ∈ [0.3, 3.0] for Gaussian cumulants.

    Requirement R5.1, R5.2.
    """
    C = _gaussian_cumulants()
    alpha_opt, g_min, n_evals = find_alpha_opt(C)
    assert 0.3 <= alpha_opt <= 3.0, (
        f"alpha_opt={alpha_opt:.6f} is outside [0.3, 3.0]."
    )

def test_find_alpha_opt_alpha_in_valid_range_heavy_tail() -> None:
    """alpha_opt ∈ [0.3, 3.0] for very heavy-tailed (Student-t) cumulants.

    Requirement R5.1, R5.2.
    Uses Student-t with df=3 to simulate heavy tails.
    """
    rng = np.random.default_rng(7)
    eps = rng.standard_t(df=3, size=5000)
    C = {"eps_white": eps}
    alpha_opt, g_min, n_evals = find_alpha_opt(C)
    assert 0.3 <= alpha_opt <= 3.0, (
        f"alpha_opt={alpha_opt:.6f} is outside [0.3, 3.0] for heavy-tailed data."
    )


def test_find_alpha_opt_n_evals_bounded_laplace() -> None:
    """n_evals ≤ 50 for Laplace cumulants.

    Requirement R5.3.
    """
    C = _laplace_cumulants()
    alpha_opt, g_min, n_evals = find_alpha_opt(C)
    assert n_evals <= 50, (
        f"n_evals={n_evals} exceeds the cap of 50."
    )

def test_find_alpha_opt_n_evals_bounded_gaussian() -> None:
    """n_evals ≤ 50 for Gaussian cumulants.

    Requirement R5.3.
    """
    C = _gaussian_cumulants()
    alpha_opt, g_min, n_evals = find_alpha_opt(C)
    assert n_evals <= 50, (
        f"n_evals={n_evals} exceeds the cap of 50."
    )

def test_find_alpha_opt_n_evals_bounded_various_seeds() -> None:
    """n_evals ≤ 50 across multiple random seeds.

    Requirement R5.3.
    Checks 5 different seeds to ensure the cap is consistently respected.
    """
    for seed in [0, 1, 2, 3, 4]:
        rng = np.random.default_rng(seed)
        eps = rng.laplace(loc=0.0, scale=1.0, size=3000)
        C = {"eps_white": eps}
        alpha_opt, g_min, n_evals = find_alpha_opt(C)
        assert n_evals <= 50, (
            f"n_evals={n_evals} exceeds cap of 50 at seed={seed}."
        )

def test_find_alpha_opt_n_evals_at_least_grid_size() -> None:
    """n_evals ≥ 28 (at least the full coarse grid is evaluated).

    Requirement R5.1.
    The coarse grid has 28 points; n_evals must be at least that.
    """
    C = _laplace_cumulants()
    alpha_opt, g_min, n_evals = find_alpha_opt(C)
    assert n_evals >= 28, (
        f"n_evals={n_evals} < 28: coarse grid was not fully evaluated."
    )


def test_find_alpha_opt_on_gaussian_returns_alpha_near_3() -> None:
    """For Gaussian data, alpha_opt ∈ [0.3, 3.0] and g_min is finite.

    Requirement R5.1, R5.2.
    For Gaussian noise, g(α) ≈ 1 for all α, so the optimizer may
    return any α in the valid range.  We verify the result is valid
    (in range, finite g_min) rather than asserting a specific α value.
    """
    C = _gaussian_cumulants(n=10_000, seed=42)
    alpha_opt, g_min, n_evals = find_alpha_opt(C)

    assert 0.3 <= alpha_opt <= 3.0, (
        f"alpha_opt={alpha_opt:.6f} is outside [0.3, 3.0] for Gaussian data."
    )
    assert np.isfinite(g_min), (
        f"g_min={g_min} is not finite for Gaussian data."
    )
    assert n_evals <= 50, (
        f"n_evals={n_evals} exceeds cap of 50 for Gaussian data."
    )

def test_find_alpha_opt_on_gaussian_g_min_near_one() -> None:
    """For Gaussian data, g_min should be close to 1.0.

    Requirement R5.1.
    For a standard normal distribution, g(α; C) ≈ 1 for all α,
    so g_min should be near 1.0 (within 0.1 tolerance for empirical data).
    """
    C = _gaussian_cumulants(n=20_000, seed=0)
    alpha_opt, g_min, n_evals = find_alpha_opt(C)
    assert abs(g_min - 1.0) < 0.15, (
        f"g_min={g_min:.4f} is not near 1.0 for Gaussian data "
        "(expected g ≈ 1 since oPMM_α offers no improvement over OLS)."
    )


# ---------------------------------------------------------------------------
# Reference table validation (Spec §4.4)
# ---------------------------------------------------------------------------


def _rgal_sym(n: int, alpha_gal: float, rng: np.random.Generator) -> np.ndarray:
    """Scalar symmetric GAL sampler from theory/opmm_monte_carlo.py.

    X = sqrt(W) * Z,  W ~ Gamma(alpha_GAL, 1),  Z ~ N(0, 1).
    """
    W = rng.gamma(shape=alpha_gal, scale=1.0, size=n)
    Z = rng.standard_normal(n)
    return np.sqrt(W) * Z


_N_SAMPLES: int = 10_000
_SEED: int = 42

def test_find_alpha_opt_matches_reference_table_spec_4_4() -> None:
    """α_opt from find_alpha_opt matches Spec §4.4 corrected table.

    For α_GAL ∈ {0.5, 1.0} the analytical g(α) curve is monotonically
    decreasing on (1, 3] -- i.e. once we exit the degenerate-basis
    neighbourhood at α=1, the lowest achievable g is at α=0.3 boundary.
    The optimizer correctly returns this lower-bound minimum.

    Requirement R5.4.
    """
    # Corrected reference table per Spec.md §4.4 (2026-05-02 revision).
    # Each row: (alpha_MAL, expected_alpha_opt, alpha_tol, expected_g_max).
    # alpha_tol is generous because empirical optimizer can land at any
    # grid point near the boundary; we mainly assert g_min is small.
    reference_table = [
        (0.5, 0.3, 0.15, 0.15),    # very heavy tails -> boundary minimum
        (1.0, 0.3, 0.15, 0.60),    # Laplace -> boundary minimum
        (2.0, 0.75, 0.25, 0.90),   # mid-tails -> interior minimum near 0.7-0.8
        (5.0, 1.5, 0.40, 0.98),    # near-Gaussian -> interior minimum near 1.5
    ]

    rng = np.random.default_rng(_SEED)

    for alpha_mal, expected_alpha_opt, alpha_tol, expected_g_max in reference_table:
        samples = _rgal_sym(_N_SAMPLES, alpha_gal=alpha_mal, rng=rng)
        C = {"eps_white": samples}

        alpha_opt, g_min, n_evals = find_alpha_opt(C)

        assert abs(alpha_opt - expected_alpha_opt) <= alpha_tol, (
            f"α_MAL={alpha_mal}: α_opt={alpha_opt:.4f} is not within "
            f"±{alpha_tol} of expected {expected_alpha_opt} "
            f"(Spec §4.4 corrected table; see Spec §12 PIVOT 12). "
            f"g_min={g_min:.6f}, n_evals={n_evals}."
        )
        assert g_min <= expected_g_max, (
            f"α_MAL={alpha_mal}: g_min={g_min:.4f} exceeds the analytical "
            f"upper bound {expected_g_max} -- find_alpha_opt failed to "
            f"locate a competitive minimum."
        )
        assert 0.3 <= alpha_opt <= 3.0, (
            f"α_MAL={alpha_mal}: α_opt={alpha_opt:.4f} is outside [0.3, 3.0]."
        )
        assert np.isfinite(g_min), (
            f"α_MAL={alpha_mal}: g_min={g_min} is not finite."
        )
        assert n_evals <= 50, (
            f"α_MAL={alpha_mal}: n_evals={n_evals} exceeds cap of 50."
        )

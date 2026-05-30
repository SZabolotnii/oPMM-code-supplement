"""Unit tests for utils/cumulants.py::empirical_cumulants.

Tests cover:
- All 8 expected keys are present in the returned dict (R3.5, R4.1)
- kappa_2 matches np.var(x, ddof=1) (R3.5)
- mu_alpha is near zero for symmetric distributions (R3.5)
- sigma_alpha_sq is non-negative (R3.5)

Satisfies Requirements R3.5, R4.1.
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.utils.cumulants import empirical_cumulants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_KEYS = frozenset(
    {"kappa_2", "kappa_3", "gamma_4", "gamma_6", "c_1_alpha", "nu_alpha",
     "sigma_alpha_sq", "mu_alpha"}
)


def _symmetric_sample(n: int = 2000, seed: int = 42) -> np.ndarray:
    """Return a symmetric (zero-mean) sample from a standard normal."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def _laplace_sample(n: int = 2000, seed: int = 42) -> np.ndarray:
    """Return a symmetric Laplace(0, 1) sample."""
    rng = np.random.default_rng(seed)
    return rng.laplace(loc=0.0, scale=1.0, size=n)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_empirical_cumulants_returns_correct_keys() -> None:
    """All 8 cumulant keys are present in the returned dict.

    Requirement R3.5, R4.1.
    """
    x = _symmetric_sample()
    result = empirical_cumulants(x, alpha=2.0)
    assert set(result.keys()) == _EXPECTED_KEYS, (
        f"Missing or extra keys.  Got: {set(result.keys())}"
    )

def test_all_values_are_finite_floats() -> None:
    """All returned values are finite Python floats.

    Requirement R3.5, R4.1.
    """
    x = _laplace_sample()
    result = empirical_cumulants(x, alpha=1.5)
    for key, val in result.items():
        assert isinstance(val, float), f"Key '{key}' is not a float: {type(val)}"
        assert np.isfinite(val), f"Key '{key}' is not finite: {val}"


def test_kappa_2_matches_sample_variance() -> None:
    """kappa_2 == np.var(x, ddof=1) for a standard normal sample.

    Requirement R3.5.
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal(500)
    result = empirical_cumulants(x, alpha=2.0)
    expected = np.var(x, ddof=1)
    assert abs(result["kappa_2"] - expected) < 1e-14, (
        f"kappa_2={result['kappa_2']:.15f} != np.var(x, ddof=1)={expected:.15f}"
    )

def test_kappa_2_matches_sample_variance_laplace() -> None:
    """kappa_2 == np.var(x, ddof=1) for a Laplace sample.

    Requirement R3.5.
    """
    x = _laplace_sample(n=1000, seed=13)
    result = empirical_cumulants(x, alpha=1.5)
    expected = np.var(x, ddof=1)
    assert abs(result["kappa_2"] - expected) < 1e-14


def test_mu_alpha_near_zero_for_symmetric_distribution() -> None:
    """For symmetric data, |mu_alpha| < 0.05.

    Requirement R3.5.
    Uses a large standard normal sample to ensure the empirical signed
    fractional moment is close to its theoretical value of 0.
    """
    rng = np.random.default_rng(99)
    x = rng.standard_normal(10_000)
    result = empirical_cumulants(x, alpha=2.0)
    assert abs(result["mu_alpha"]) < 0.05, (
        f"|mu_alpha| = {abs(result['mu_alpha']):.4f} >= 0.05 for symmetric data."
    )

def test_mu_alpha_near_zero_for_symmetric_laplace() -> None:
    """For symmetric Laplace data, |mu_alpha| < 0.05.

    Requirement R3.5.
    """
    rng = np.random.default_rng(17)
    x = rng.laplace(loc=0.0, scale=1.0, size=10_000)
    result = empirical_cumulants(x, alpha=1.5)
    assert abs(result["mu_alpha"]) < 0.05, (
        f"|mu_alpha| = {abs(result['mu_alpha']):.4f} >= 0.05 for symmetric Laplace."
    )

def test_mu_alpha_nonzero_for_asymmetric() -> None:
    """For strongly asymmetric data, |mu_alpha| should be noticeably > 0.

    Requirement R3.5.
    This is a sanity check that mu_alpha is sensitive to asymmetry.
    """
    rng = np.random.default_rng(55)
    # Exponential distribution is strongly right-skewed
    x = rng.exponential(scale=1.0, size=10_000)
    result = empirical_cumulants(x, alpha=2.0)
    assert abs(result["mu_alpha"]) > 0.1, (
        f"|mu_alpha| = {abs(result['mu_alpha']):.4f} unexpectedly small for "
        "asymmetric (exponential) data."
    )


def test_sigma_alpha_sq_is_nonnegative_normal() -> None:
    """sigma_alpha_sq >= 0 for standard normal data.

    Requirement R3.5.
    """
    x = _symmetric_sample(n=1000, seed=0)
    result = empirical_cumulants(x, alpha=2.0)
    assert result["sigma_alpha_sq"] >= 0.0, (
        f"sigma_alpha_sq = {result['sigma_alpha_sq']:.6f} < 0."
    )

def test_sigma_alpha_sq_is_nonnegative_laplace() -> None:
    """sigma_alpha_sq >= 0 for Laplace data.

    Requirement R3.5.
    """
    x = _laplace_sample(n=1000, seed=3)
    result = empirical_cumulants(x, alpha=1.5)
    assert result["sigma_alpha_sq"] >= 0.0, (
        f"sigma_alpha_sq = {result['sigma_alpha_sq']:.6f} < 0."
    )

def test_sigma_alpha_sq_is_nonnegative_various_alpha() -> None:
    """sigma_alpha_sq >= 0 across a range of alpha values.

    Requirement R3.5.
    Checks alpha ∈ {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}.
    """
    rng = np.random.default_rng(42)
    x = rng.standard_normal(2000)
    for alpha in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        result = empirical_cumulants(x, alpha=alpha)
        assert result["sigma_alpha_sq"] >= 0.0, (
            f"sigma_alpha_sq = {result['sigma_alpha_sq']:.6f} < 0 at alpha={alpha}."
        )

def test_sigma_alpha_sq_is_nonnegative_asymmetric() -> None:
    """sigma_alpha_sq >= 0 for asymmetric (exponential) data.

    Requirement R3.5.
    """
    rng = np.random.default_rng(11)
    x = rng.exponential(scale=2.0, size=1000)
    result = empirical_cumulants(x, alpha=2.0)
    assert result["sigma_alpha_sq"] >= 0.0, (
        f"sigma_alpha_sq = {result['sigma_alpha_sq']:.6f} < 0 for exponential data."
    )


def test_raises_on_2d_input() -> None:
    """ValueError raised when eps_white is 2-D.

    Requirement R3.5.
    """
    x = np.ones((10, 5))
    with pytest.raises(ValueError, match="1-D"):
        empirical_cumulants(x, alpha=2.0)

def test_raises_on_nan_input() -> None:
    """ValueError raised when eps_white contains NaN.

    Requirement R3.5.
    """
    x = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.raises(ValueError, match="non-finite"):
        empirical_cumulants(x, alpha=2.0)

def test_raises_on_inf_input() -> None:
    """ValueError raised when eps_white contains Inf.

    Requirement R3.5.
    """
    x = np.array([1.0, np.inf, 3.0])
    with pytest.raises(ValueError, match="non-finite"):
        empirical_cumulants(x, alpha=2.0)


def test_small_sample_emits_warning() -> None:
    """UserWarning emitted when n < 30.

    Requirement R3.5.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(20)
    with pytest.warns(UserWarning, match="n=20 < 30"):
        empirical_cumulants(x, alpha=2.0)

def test_small_sample_still_returns_correct_keys() -> None:
    """All 8 keys present even in small-sample fallback mode.

    Requirement R3.5.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10)
    with pytest.warns(UserWarning):
        result = empirical_cumulants(x, alpha=2.0)
    assert set(result.keys()) == _EXPECTED_KEYS

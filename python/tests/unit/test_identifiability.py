"""Unit tests for core/identifiability.py::project_identifiability.

Each test references the Requirement ID it validates, per AGENTS.md §5.4.

Satisfies Requirements R2.1, R2.2, R2.3.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mv_patp.core.identifiability import project_identifiability


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_spd(size: int, seed: int = 0) -> np.ndarray:
    """Return a random symmetric positive-definite matrix of shape (size, size)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((size, size))
    return A @ A.T + np.eye(size)  # guaranteed SPD


# ---------------------------------------------------------------------------
# Scenario "white" — Requirement R2.1
# ---------------------------------------------------------------------------


def test_white_scenario_fixes_sigma_to_identity():
    """Requirement R2.1: 'white' scenario must return Σ = I_n.

    The returned Sigma_hat_new must be the identity matrix of the same size
    as the input Sigma_hat, regardless of the input value.
    """
    k, n = 3, 5
    Gamma_hat = _random_spd(k, seed=1)
    Sigma_hat = _random_spd(n, seed=2)  # arbitrary non-identity input

    _, Sigma_new = project_identifiability(Gamma_hat, Sigma_hat, scenario="white")

    np.testing.assert_array_equal(
        Sigma_new,
        np.eye(n),
        err_msg="'white' scenario must return Sigma = I_n",
    )


def test_white_scenario_preserves_gamma():
    """Requirement R2.1: 'white' scenario must leave Γ̂ unchanged.

    The returned Gamma_hat_new must be numerically identical to the input.
    """
    k, n = 4, 6
    Gamma_hat = _random_spd(k, seed=3)
    Sigma_hat = _random_spd(n, seed=4)

    Gamma_new, _ = project_identifiability(Gamma_hat, Sigma_hat, scenario="white")

    np.testing.assert_array_equal(
        Gamma_new,
        Gamma_hat,
        err_msg="'white' scenario must return Gamma unchanged",
    )


# ---------------------------------------------------------------------------
# Scenario "ar1" — Requirements R2.2, R2.3
# ---------------------------------------------------------------------------


def test_ar1_scenario_rescales_trace_to_k():
    """Requirement R2.2: 'ar1' scenario must enforce tr(Γ̂_new) = k.

    After projection, the trace of the returned Gamma_hat_new must equal k
    (the number of rows / antennas) within floating-point precision.
    """
    k, n = 4, 6
    # Build Gamma_hat with tr ≠ k deliberately
    Gamma_hat = 3.0 * _random_spd(k, seed=5)  # tr will be >> k in general
    Sigma_hat = _random_spd(n, seed=6)

    Gamma_new, _ = project_identifiability(Gamma_hat, Sigma_hat, scenario="ar1")

    np.testing.assert_allclose(
        np.trace(Gamma_new),
        k,
        rtol=1e-12,
        err_msg="'ar1' scenario must enforce tr(Gamma_hat_new) == k",
    )


def test_ar1_scenario_reciprocal_scaling_on_sigma():
    """Requirement R2.2: 'ar1' scenario must apply reciprocal scale to Σ̂.

    The scale factor is tr(Γ̂) / k.  The returned Sigma_hat_new must equal
    Sigma_hat * scale.
    """
    k, n = 3, 5
    Gamma_hat = 2.5 * _random_spd(k, seed=7)
    Sigma_hat = _random_spd(n, seed=8)

    scale = np.trace(Gamma_hat) / k
    _, Sigma_new = project_identifiability(Gamma_hat, Sigma_hat, scenario="ar1")

    np.testing.assert_allclose(
        Sigma_new,
        Sigma_hat * scale,
        rtol=1e-12,
        err_msg="'ar1' scenario must return Sigma_hat_new = Sigma_hat * scale",
    )


def test_ar1_scenario_logs_warning_when_trace_deviates(caplog):
    """Requirement R2.3: 'ar1' scenario must log a warning when |tr(Γ̂) − k| > 1e-6.

    Uses pytest's caplog fixture to capture log records emitted by the
    module-level logger in identifiability.py.
    """
    k, n = 3, 4
    # Construct Gamma_hat whose trace is far from k
    Gamma_hat = 10.0 * np.eye(k)  # tr = 30, k = 3 → deviation = 27
    Sigma_hat = np.eye(n)

    with caplog.at_level(logging.WARNING, logger="mv_patp.core.identifiability"):
        project_identifiability(Gamma_hat, Sigma_hat, scenario="ar1")

    assert len(caplog.records) >= 1, (
        "Expected at least one WARNING log when |tr(Γ̂) − k| > 1e-6"
    )
    assert any("Re-projecting" in r.message for r in caplog.records), (
        "Warning message should mention 'Re-projecting'"
    )


def test_ar1_scenario_no_warning_when_trace_equals_k(caplog):
    """Requirement R2.3: no warning should be emitted when tr(Γ̂) == k exactly.

    Constructs Gamma_hat = I_k so that tr(Γ̂) = k exactly.
    """
    k, n = 4, 5
    Gamma_hat = np.eye(k)  # tr = k exactly
    Sigma_hat = np.eye(n)

    with caplog.at_level(logging.WARNING, logger="mv_patp.core.identifiability"):
        project_identifiability(Gamma_hat, Sigma_hat, scenario="ar1")

    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warning_records) == 0, (
        "No warning should be emitted when tr(Γ̂) == k"
    )


# ---------------------------------------------------------------------------
# Unknown scenario — ValueError
# ---------------------------------------------------------------------------


def test_unknown_scenario_raises_value_error():
    """Raises ValueError for an unrecognised scenario string.

    Satisfies the explicit error-handling contract in project_identifiability.
    """
    k, n = 2, 3
    Gamma_hat = np.eye(k)
    Sigma_hat = np.eye(n)

    with pytest.raises(ValueError, match="unknown scenario"):
        project_identifiability(Gamma_hat, Sigma_hat, scenario="bogus")

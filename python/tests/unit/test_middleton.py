"""Unit tests for the Middleton Class A DGP.

Requirement R9 (heavy-tailed misspecification scope): the canonical impulsive
noise generator must be a valid symmetric, unit-variance, heavy-tailed family
with a known closed-form excess kurtosis, so that the §V-G Middleton ARE
experiment (Spec.md §12 PIVOT 19) is reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.data_gen.middleton import (
    middleton_excess_kurtosis,
    sample_middleton_class_a,
)


@pytest.mark.parametrize("A,Gp", [(1.0, 0.1), (0.35, 0.1), (0.1, 0.1)])
def test_middleton_unit_variance_and_symmetry(A: float, Gp: float) -> None:
    """Requirement R9: Class A is symmetric and unit-variance by construction."""
    rng = np.random.default_rng(42)
    x = sample_middleton_class_a(200, 5000, A=A, Gamma_prime=Gp, rng=rng).ravel()
    assert abs(x.var() - 1.0) < 0.05          # unit variance
    assert abs(float(np.mean(x**3) / x.var() ** 1.5)) < 0.1  # symmetric (skew ~ 0)


@pytest.mark.parametrize("A,Gp", [(1.0, 0.1), (0.35, 0.1), (0.1, 0.1)])
def test_middleton_kurtosis_matches_closed_form(A: float, Gp: float) -> None:
    """Requirement R9: empirical excess kurtosis matches gamma_4 = 3/(A(1+G')^2)."""
    rng = np.random.default_rng(7)
    x = sample_middleton_class_a(200, 20000, A=A, Gamma_prime=Gp, rng=rng).ravel()
    emp = float(np.mean(x**4) / x.var() ** 2 - 3.0)
    closed = middleton_excess_kurtosis(A, Gp)
    assert abs(emp - closed) / closed < 0.1


def test_middleton_deterministic() -> None:
    """Requirement R1.5: identical seed yields identical output."""
    a = sample_middleton_class_a(4, 64, A=0.5, Gamma_prime=0.1, rng=np.random.default_rng(1))
    b = sample_middleton_class_a(4, 64, A=0.5, Gamma_prime=0.1, rng=np.random.default_rng(1))
    assert np.array_equal(a, b)


def test_middleton_rejects_invalid_params() -> None:
    """Requirement R9: invalid A or Gamma_prime raise ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_middleton_class_a(2, 8, A=0.0, Gamma_prime=0.1, rng=rng)
    with pytest.raises(ValueError):
        sample_middleton_class_a(2, 8, A=1.0, Gamma_prime=0.0, rng=rng)

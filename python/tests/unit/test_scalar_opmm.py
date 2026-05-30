"""Unit tests for core/scalar_opmm.py::scalar_opmm_estimate.

Satisfies Requirement R6.5 (k=1 reduction sanity).
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.core.scalar_opmm import ScalarOPMMResult, scalar_opmm_estimate
from mv_patp.utils.errors import EMDivergence


def test_scalar_opmm_recovers_truth_when_gaussian_n_large() -> None:
    """scalar_opmm_estimate recovers h_truth on Gaussian noise.

    Requirement R6.5.
    """
    rng = np.random.default_rng(42)
    n = 512
    h_true = 1.5
    s = rng.standard_normal(n)
    eps = rng.standard_normal(n)
    y = h_true * s + eps
    result = scalar_opmm_estimate(y, s, alpha_p_init=2.0, alpha_p_fixed=True)
    assert isinstance(result, ScalarOPMMResult)
    assert abs(result.h - h_true) < 0.1, (
        f"|h_hat - h_true| = {abs(result.h - h_true):.4f}"
    )


def test_scalar_opmm_converges_within_max_iter_when_default() -> None:
    """Outer loop converges within max_iter=50 on Laplace noise.

    Requirement R6.5.
    """
    rng = np.random.default_rng(0)
    n = 512
    s = rng.standard_normal(n)
    eps = rng.laplace(size=n)
    y = 2.0 * s + eps
    result = scalar_opmm_estimate(y, s, alpha_p_init=1.5, alpha_p_fixed=True)
    assert result.converged


def test_scalar_opmm_rejects_shape_mismatch_when_lengths_differ() -> None:
    """Mismatched (y, s) lengths raise ValueError.

    Requirement R6.5 (input validation).
    """
    with pytest.raises(ValueError, match="same shape"):
        scalar_opmm_estimate(np.zeros(10), np.zeros(20))


def test_scalar_opmm_raises_on_divergence_when_flag_set() -> None:
    """raise_on_divergence=True surfaces EMDivergence on tight tol.

    Requirement R6.5.
    """
    rng = np.random.default_rng(7)
    s = rng.standard_normal(64)
    y = 1.0 * s + rng.standard_normal(64)
    with pytest.raises(EMDivergence):
        scalar_opmm_estimate(
            y, s, max_iter=1, tol=1e-30, alpha_p_init=2.0,
            alpha_p_fixed=True, raise_on_divergence=True,
        )

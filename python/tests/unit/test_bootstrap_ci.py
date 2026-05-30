"""Unit tests for core/bootstrap_ci.py::parametric_bootstrap_ci.

Satisfies Requirement R8.1, R8.5 (bootstrap returns valid CI shape +
bracket on synthetic data). Coverage validation against the [0.93, 0.97]
band lives in tests/integration/test_bootstrap_ci_coverage.py (slow).
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.core.bootstrap_ci import (
    _bartlett_matrix_gamma,
    parametric_bootstrap_ci,
)
from mv_patp.core.em_patp import em_patp_estimate


def _make_problem(k: int, p: int, n: int, seed: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    eps = rng.standard_normal((k, n))
    Y = H_truth @ S + eps
    return Y, S, H_truth


def test_bootstrap_ci_returns_correct_shape_when_called() -> None:
    """parametric_bootstrap_ci returns (lower, upper) of H shape.

    Requirement R8.1 (bootstrap CI primary).
    """
    Y, S, _ = _make_problem(k=2, p=2, n=128, seed=42)
    fit = em_patp_estimate(Y, S, max_iter=15, alpha_p_init=2.0, alpha_p_fixed=True)
    lower, upper = parametric_bootstrap_ci(Y, S, fit, B=20, seed=0)
    assert lower.shape == fit.H.shape
    assert upper.shape == fit.H.shape


def test_bootstrap_ci_brackets_h_truth_when_n_large() -> None:
    """The 95% CI brackets H_truth most of the time at n=256.

    Requirement R8.1, R8.2.
    Single-run smoke check: at least half of the four (k, p) entries lie
    inside the CI.
    """
    Y, S, H_truth = _make_problem(k=2, p=2, n=256, seed=42)
    fit = em_patp_estimate(Y, S, max_iter=15, alpha_p_init=2.0, alpha_p_fixed=True)
    lower, upper = parametric_bootstrap_ci(Y, S, fit, B=40, seed=0,
                                            alpha_mal_override=5.0)
    inside = (lower <= H_truth) & (H_truth <= upper)
    assert inside.sum() >= H_truth.size // 2, (
        f"Bootstrap CI failed to bracket {H_truth.size - inside.sum()}/"
        f"{H_truth.size} entries of H_truth"
    )


def test_bootstrap_ci_lower_below_upper_when_called() -> None:
    """The 2.5% percentile is element-wise <= the 97.5% percentile.

    Requirement R8.1.
    """
    Y, S, _ = _make_problem(k=2, p=2, n=128, seed=42)
    fit = em_patp_estimate(Y, S, max_iter=15, alpha_p_init=2.0, alpha_p_fixed=True)
    lower, upper = parametric_bootstrap_ci(Y, S, fit, B=20, seed=0)
    assert np.all(lower <= upper)


def test_bartlett_matrix_gamma_has_expected_mean_when_n_large() -> None:
    """Bartlett MG_k(alpha) has E[Gamma] = I_k empirically (n=2000).

    Requirement R8.5 (resampler matches the original DGP).
    """
    rng = np.random.default_rng(42)
    samples = np.array(
        [_bartlett_matrix_gamma(k=3, alpha_mg=2.0, rng=rng) for _ in range(2000)]
    )
    mean_gamma = samples.mean(axis=0)
    assert np.allclose(mean_gamma, np.eye(3), atol=0.1), (
        f"Mean Gamma deviates from I_3:\n{mean_gamma}"
    )


def test_bartlett_matrix_gamma_rejects_invalid_alpha() -> None:
    """alpha_mg <= (k-1)/2 surfaces a typed ValueError.

    Requirement R8.5.
    """
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="alpha_mg"):
        _bartlett_matrix_gamma(k=4, alpha_mg=1.0, rng=rng)

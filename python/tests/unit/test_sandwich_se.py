"""Unit tests for core/sandwich_se.py::sandwich_se.

Satisfies Requirement R8.3 (sandwich SE shape + finiteness). Coverage
validation against the [0.90, 0.98] band lives in
tests/integration/test_sandwich_coverage.py (slow MC).
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.core.em_patp import em_patp_estimate
from mv_patp.core.sandwich_se import sandwich_se


def _make_problem(k: int, p: int, n: int, seed: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    eps = rng.standard_normal((k, n))
    Y = H_truth @ S + eps
    return Y, S, H_truth


def test_sandwich_se_returns_correct_shape_when_called() -> None:
    """sandwich_se returns SE matrix matching H shape.

    Requirement R8.3.
    """
    Y, S, _ = _make_problem(k=2, p=2, n=128, seed=42)
    fit = em_patp_estimate(Y, S, max_iter=15, alpha_p_init=2.0, alpha_p_fixed=True)
    se = sandwich_se(Y, S, fit)
    assert se.shape == fit.H.shape


def test_sandwich_se_entries_are_positive_when_called() -> None:
    """All SE entries are strictly positive on non-degenerate data.

    Requirement R8.3.
    """
    Y, S, _ = _make_problem(k=2, p=2, n=128, seed=42)
    fit = em_patp_estimate(Y, S, max_iter=15, alpha_p_init=2.0, alpha_p_fixed=True)
    se = sandwich_se(Y, S, fit)
    assert np.all(se > 0)
    assert np.all(np.isfinite(se))


def test_sandwich_se_decreases_with_n_when_n_doubled() -> None:
    """SE scales as roughly 1/sqrt(n) when n doubles.

    Requirement R8.3 (asymptotic scaling).
    """
    Y_small, S_small, _ = _make_problem(k=2, p=2, n=128, seed=42)
    fit_small = em_patp_estimate(Y_small, S_small, max_iter=15,
                                 alpha_p_init=2.0, alpha_p_fixed=True)
    se_small = sandwich_se(Y_small, S_small, fit_small)

    Y_large, S_large, _ = _make_problem(k=2, p=2, n=512, seed=42)
    fit_large = em_patp_estimate(Y_large, S_large, max_iter=15,
                                 alpha_p_init=2.0, alpha_p_fixed=True)
    se_large = sandwich_se(Y_large, S_large, fit_large)

    ratio = float(se_small.mean()) / float(se_large.mean())
    # n quadruples (128 -> 512) so SE should drop by roughly factor 2.
    assert 1.5 < ratio < 3.0, (
        f"SE ratio {ratio:.2f} out of expected band [1.5, 3.0] for 4x n change"
    )


def test_sandwich_se_rejects_unknown_options() -> None:
    """Unknown bread/meat options raise ValueError."""
    Y, S, _ = _make_problem(k=2, p=2, n=128, seed=42)
    fit = em_patp_estimate(Y, S, max_iter=15, alpha_p_init=2.0, alpha_p_fixed=True)
    with pytest.raises(ValueError):
        sandwich_se(Y, S, fit, bread="diag_w")
    with pytest.raises(ValueError):
        sandwich_se(Y, S, fit, meat="hac")


def test_sandwich_se_psi_prime_bread_exceeds_weight_bread_on_heavy_tails() -> None:
    """At alpha < 1 on heavy tails psi' < w pointwise, so the corrected SE is larger.

    psi'(e) - w(e) = k2* (alpha - 1) |e|^(alpha - 1) <= 0 when k2* >= 0, alpha < 1
    (R1, Rv4-C2b).
    """
    rng = np.random.default_rng(3)
    k, p, n = 2, 2, 512
    H = rng.standard_normal((k, p))
    S = rng.choice([-1.0, 1.0], size=(p, n))
    Y = H @ S + rng.standard_t(4, size=(k, n)) / np.sqrt(2.0)
    fit = em_patp_estimate(Y, S, max_iter=20, alpha_p_init=0.5, alpha_p_fixed=True)
    se_new = sandwich_se(Y, S, fit)
    se_old = sandwich_se(Y, S, fit, bread="weight")
    assert np.all(se_new > 1.3 * se_old)


def test_sandwich_se_cluster_meat_matches_shape_and_is_close() -> None:
    """Column-cluster meat returns the same shape and, on iid noise, similar SEs."""
    Y, S, _ = _make_problem(k=2, p=2, n=512, seed=11)
    fit = em_patp_estimate(Y, S, max_iter=15, alpha_p_init=2.0, alpha_p_fixed=True)
    se_e = sandwich_se(Y, S, fit)
    se_c = sandwich_se(Y, S, fit, meat="cluster")
    assert se_c.shape == se_e.shape
    assert np.all(np.isfinite(se_c)) and np.all(se_c > 0)
    assert np.allclose(se_c, se_e, rtol=0.25)

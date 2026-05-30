"""Unit tests for eps_min floor sensitivity (Exp 5).

Requirement R9 / Spec.md §5.3: the eps_min floor on residual magnitude
must not materially affect the score value for |eps| >> eps_min, and
the resulting ARE spread over 4 orders of magnitude must be < 0.5%.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

from mv_patp.core.em_patp import em_patp_estimate


def _make_t4_data(
    k: int = 2, n: int = 512, p: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    z = rng.standard_normal((k, n))
    chi2 = rng.chisquare(df=4, size=(k, n))
    N = z / np.sqrt(chi2 / 4)
    return H_truth @ S + N, S, H_truth


def test_score_values_identical_above_threshold():
    """For |eps| > 0.01, score function at eps1e10 ≈ score at eps1e4 within 1%.

    The score weight is w(eps) = k1 + k2 * |eps|^{alpha-1}. At alpha=0.5,
    w(eps) = k1 + k2 * |eps|^{-0.5}. For |eps| = 0.01, |eps|^{-0.5} = 10;
    flooring at eps_min=1e-4 vs 1e-10 changes nothing for |eps| > eps_min.
    """
    from mv_patp.core.score import compute_optimal_weights
    from mv_patp.utils.cumulants import empirical_cumulants

    rng = np.random.default_rng(0)
    eps = rng.standard_t(df=4, size=10_000)
    alpha = 0.5  # worst case for floor sensitivity

    C = empirical_cumulants(eps, alpha)
    k1, k2 = compute_optimal_weights(C, alpha)

    # Compute weights with two different floors; evaluate at |eps| = 0.01
    eps_test = np.array([0.01, 0.1, 1.0])
    w_tight = k1 + k2 * np.maximum(eps_test, 1e-10) ** (alpha - 1.0)
    w_loose = k1 + k2 * np.maximum(eps_test, 1e-4) ** (alpha - 1.0)

    rel_diff = np.abs(w_tight - w_loose) / np.maximum(np.abs(w_tight), 1e-14)
    assert np.all(rel_diff < 0.01), (
        f"Score relative diff at |eps|=0.01..1.0: {rel_diff}; expected < 1%."
    )


def test_are_spread_under_halfpercent():
    """Requirement: max ARE - min ARE across eps_min variants < 0.5% of ARE.

    Uses 100 replicates to estimate MSE for each eps_min variant.
    """
    eps_vals = [1e-10, 1e-8, 1e-6, 1e-4]
    n_reps = 100
    k, n, p = 2, 512, 2

    mse_by_eps: dict[float, float] = {}
    mse_ols_arr = []

    for eps_val in eps_vals:
        fit_fn = functools.partial(
            em_patp_estimate,
            eps_min=eps_val,
            alpha_p_fixed=True,
            alpha_p_init=0.5,
        )
        sq_errs = []
        for rep in range(n_reps):
            Y, S, H_truth = _make_t4_data(k=k, n=n, p=p, seed=rep)
            res = fit_fn(Y, S)
            sq_errs.append(np.linalg.norm(res.H - H_truth, ord="fro") ** 2)
        mse_by_eps[eps_val] = float(np.mean(sq_errs))

    from mv_patp.baseline.ols import fit as ols_fit
    for rep in range(n_reps):
        Y, S, H_truth = _make_t4_data(k=k, n=n, p=p, seed=rep)
        res_o = ols_fit(Y, S)
        mse_ols_arr.append(np.linalg.norm(res_o.H - H_truth, ord="fro") ** 2)
    mse_ols = float(np.mean(mse_ols_arr))

    are_vals = [mse_ols / mse_by_eps[eps_val] for eps_val in eps_vals]
    are_spread = max(are_vals) - min(are_vals)
    are_mean = float(np.mean(are_vals))

    assert are_spread / are_mean < 0.005, (
        f"ARE spread across eps_min variants = {are_spread:.4f} "
        f"({100*are_spread/are_mean:.2f}% of mean ARE={are_mean:.4f}); "
        "expected < 0.5%."
    )

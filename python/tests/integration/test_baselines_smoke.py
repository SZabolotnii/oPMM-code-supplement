"""Integration smoke tests for the channel-estimation baselines.

These tests confirm that ``ols``, ``wishart_mle``, and ``matrix_t_mle`` all
run end-to-end on a small heavy-tailed instance, return finite ``H``
matrices of the right shape, and -- when the same Y is fed in -- agree to
within a sane Frobenius radius. A failure here usually points at a shared
infrastructure bug (e.g. ``BaselineResult`` shape drift, an SPD-floor
regression in the Kronecker-covariance updates) rather than at any
single estimator.

All three estimators take only ``(Y, S)``; they do not import from
``src/mv_patp/core/`` and therefore can be smoke-tested before the
EM-PATP pipeline lands.

Task: Section 11.5.
Satisfies Requirement R9 (used as ARE comparators).
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.baseline import (
    BaselineResult,
    matrix_t_mle,
    ols,
    wishart_mle,
)


_K = 2
_P = 2
_N = 256


def _make_heavy_tailed_problem(
    seed: int, df: float = 5.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(Y, S, H_truth)`` with Student-t noise, df=5."""
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((_K, _P))
    S = rng.standard_normal((_P, _N))
    z = rng.standard_normal((_K, _N))
    chi2 = rng.chisquare(df=df, size=(_K, _N))
    N = z / np.sqrt(chi2 / df)
    Y = H_truth @ S + N
    return Y, S, H_truth


@pytest.mark.parametrize("estimator_name", ["ols", "wishart_mle", "matrix_t_mle"])
def test_baseline_runs_without_error_when_heavy_tailed(estimator_name: str) -> None:
    """Each baseline returns a finite, correct-shape H on Student-t noise.

    Requirement R9 (baselines must run end-to-end on the DGP we use in
    Phase C).
    """
    Y, S, H_truth = _make_heavy_tailed_problem(seed=42)
    fit = {
        "ols": ols.fit,
        "wishart_mle": wishart_mle.fit,
        "matrix_t_mle": matrix_t_mle.fit,
    }[estimator_name]

    result = fit(Y, S)

    assert isinstance(result, BaselineResult)
    assert result.H.shape == (_K, _P)
    assert np.all(np.isfinite(result.H)), f"{estimator_name} returned non-finite H"
    assert result.residuals.shape == (_K, _N)


def test_baselines_agree_within_bounded_radius_when_seeded_identically() -> None:
    """All three baselines land within a bounded Frobenius radius of OLS.

    Requirement R9 (baselines must produce comparable estimates on the
    same data so that ARE comparisons are meaningful). The radius is
    deliberately generous because matrix-t MLE down-weights heavy-tail
    epochs differently from the Gaussian estimators.
    """
    Y, S, _ = _make_heavy_tailed_problem(seed=42)

    h_ols = ols.fit(Y, S).H
    h_wishart = wishart_mle.fit(Y, S).H
    h_matrix_t = matrix_t_mle.fit(Y, S, df=5.0).H

    delta_wishart = np.linalg.norm(h_wishart - h_ols, ord="fro")
    delta_matrix_t = np.linalg.norm(h_matrix_t - h_ols, ord="fro")

    assert delta_wishart < 1.0, (
        f"Wishart MLE differs from OLS by {delta_wishart:.3f} on Student-t data"
    )
    assert delta_matrix_t < 1.0, (
        f"matrix-t MLE differs from OLS by {delta_matrix_t:.3f} on Student-t data"
    )


def test_baselines_residual_norm_does_not_exceed_naive_zero_estimate() -> None:
    """Each baseline yields a strictly better fit than ``H = 0``.

    Requirement R9 (sanity check: baselines must extract signal). If a
    baseline's residual norm is no smaller than the original Y norm,
    something is fundamentally wrong with the fit.
    """
    Y, S, _ = _make_heavy_tailed_problem(seed=42)
    naive_residual = float(np.linalg.norm(Y, ord="fro"))

    for fit in (ols.fit, wishart_mle.fit, matrix_t_mle.fit):
        result = fit(Y, S)
        residual_norm = float(np.linalg.norm(result.residuals, ord="fro"))
        assert residual_norm < naive_residual, (
            f"{fit.__module__}: residual norm {residual_norm:.3f} not better "
            f"than naive {naive_residual:.3f}"
        )

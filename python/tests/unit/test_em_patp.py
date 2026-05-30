"""Unit tests for core/em_patp.py::em_patp_estimate.

Satisfies Requirements R6.1, R6.2, R6.3, R6.4.
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.core.em_patp import EMPATPResult, em_patp_estimate
from mv_patp.utils.errors import EMDivergence


def _make_gaussian(k: int, p: int, n: int, seed: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    N = rng.standard_normal((k, n))
    Y = H_truth @ S + N
    return Y, S, H_truth


def _make_heavy_tailed(
    k: int, p: int, n: int, df: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    z = rng.standard_normal((k, n))
    chi2 = rng.chisquare(df=df, size=(k, n))
    N = z / np.sqrt(chi2 / df)
    Y = H_truth @ S + N
    return Y, S, H_truth


def test_em_patp_returns_expected_result_type() -> None:
    """em_patp_estimate returns an EMPATPResult with all expected fields.

    Requirement R6.1 (M-step + outer-EM contract).
    """
    Y, S, _ = _make_gaussian(k=2, p=2, n=128, seed=42)
    result = em_patp_estimate(Y, S, max_iter=10, alpha_p_init=2.0,
                              alpha_p_fixed=True)
    assert isinstance(result, EMPATPResult)
    assert result.H.shape == (2, 2)
    assert result.Gamma.shape == (2, 2)
    assert result.Sigma.shape == (128, 128)
    assert isinstance(result.alpha_p, float)
    assert isinstance(result.converged, bool)
    assert result.n_iter_outer >= 1
    assert result.n_iter_irls_total >= 1
    assert "outer_delta" in result.log_history


def test_em_patp_converges_when_gaussian_noise() -> None:
    """EM-PATP converges within max_iter=50 on Gaussian noise.

    Requirement R6.2 (convergence handling).
    """
    Y, S, _ = _make_gaussian(k=2, p=2, n=512, seed=42)
    result = em_patp_estimate(Y, S, max_iter=50, tol=1e-6,
                              alpha_p_init=2.0, alpha_p_fixed=True)
    assert result.converged, (
        f"EM-PATP failed to converge on Gaussian data; "
        f"last outer delta={result.log_history['outer_delta'][-1]:.3e}"
    )


def test_em_patp_recovers_truth_when_n_large() -> None:
    """EM-PATP returns H close to H_truth on large-sample Gaussian.

    Requirement R6.4 (H estimation accuracy under benign noise).
    """
    Y, S, H_truth = _make_gaussian(k=2, p=2, n=512, seed=42)
    result = em_patp_estimate(Y, S, max_iter=50, alpha_p_init=2.0,
                              alpha_p_fixed=True)
    err = np.linalg.norm(result.H - H_truth, ord="fro")
    assert err < 0.5, f"EM-PATP error {err:.3f} exceeds sanity threshold 0.5"


def test_em_patp_handles_heavy_tail_when_df_5() -> None:
    """EM-PATP returns finite H on Student-t (df=5) noise.

    Requirement R6.1 (heavy-tail robustness).
    """
    Y, S, H_truth = _make_heavy_tailed(k=2, p=2, n=256, df=5.0, seed=2026)
    result = em_patp_estimate(Y, S, max_iter=30, alpha_p_init=1.5,
                              alpha_p_fixed=True)
    assert np.all(np.isfinite(result.H))
    err = np.linalg.norm(result.H - H_truth, ord="fro")
    assert err < 1.5, f"EM-PATP heavy-tail error {err:.3f}"


def test_em_patp_records_history_when_iterating() -> None:
    """log_history captures per-iteration diagnostics.

    Requirement R6.3 (log per-iteration Frobenius delta).
    """
    Y, S, _ = _make_gaussian(k=2, p=2, n=128, seed=1)
    result = em_patp_estimate(Y, S, max_iter=20, alpha_p_init=2.0,
                              alpha_p_fixed=True)
    assert len(result.log_history["outer_delta"]) == result.n_iter_outer
    assert len(result.log_history["irls_iters"]) == result.n_iter_outer
    assert all(d >= 0.0 for d in result.log_history["outer_delta"])


def test_em_patp_raises_on_divergence_when_flag_set() -> None:
    """raise_on_divergence=True surfaces EMDivergence on max_iter exhaustion.

    Requirement R6.3 (divergence handling).
    """
    Y, S, _ = _make_gaussian(k=2, p=2, n=64, seed=0)
    # Force divergence with extremely tight tolerance and 1 iteration.
    with pytest.raises(EMDivergence):
        em_patp_estimate(Y, S, max_iter=1, tol=1e-30, alpha_p_init=2.0,
                         alpha_p_fixed=True, raise_on_divergence=True)


def test_em_patp_rejects_shape_mismatch_when_S_n_disagrees() -> None:
    """Y and S column-count mismatch surfaces ValueError.

    Requirement R6.1 (input validation).
    """
    Y = np.zeros((2, 64))
    S = np.zeros((2, 32))
    with pytest.raises(ValueError, match="shapes must agree"):
        em_patp_estimate(Y, S)

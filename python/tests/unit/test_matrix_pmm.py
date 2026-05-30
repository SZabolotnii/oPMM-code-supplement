"""Unit tests for matrix-PMM2 and matrix-PMM3 baselines.

Satisfies Requirement R9.5 (PMM2 collapses to OLS on symmetric data) and
generic R9 robustness checks for both estimators.
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.baseline import matrix_pmm2, matrix_pmm3, ols


def _make_problem(k: int, p: int, n: int, seed: int, df: float | None = None) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    if df is None:
        N = rng.standard_normal((k, n))
    else:
        z = rng.standard_normal((k, n))
        chi2 = rng.chisquare(df=df, size=(k, n))
        N = z / np.sqrt(chi2 / df)
    Y = H_truth @ S + N
    return Y, S, H_truth


def test_matrix_pmm2_collapses_to_ols_when_symmetric_gaussian() -> None:
    """Matrix-PMM2 on symmetric Gaussian data is within 5% of OLS.

    Requirement R9.5 (PMM2 ARE near 1.0 on symmetric -> kappa_3 = 0
    forces k_2 = 0 forces score = OLS).
    """
    Y, S, H_truth = _make_problem(k=2, p=2, n=512, seed=42)
    h_pmm2 = matrix_pmm2.fit(Y, S).H
    h_ols = ols.fit(Y, S).H
    delta = np.linalg.norm(h_pmm2 - h_ols, ord="fro")
    h_norm = np.linalg.norm(h_ols, ord="fro")
    assert delta / h_norm < 0.05, (
        f"PMM2 deviates from OLS by {delta/h_norm:.4f} on symmetric Gaussian "
        f"data (should be < 5% per R9.5)"
    )


def test_matrix_pmm3_handles_heavy_tail_when_df_4() -> None:
    """Matrix-PMM3 returns finite H on Student-t df=4 noise.

    Requirement R9 (heavy-tailed baseline must run).
    """
    Y, S, H_truth = _make_problem(k=2, p=2, n=512, seed=42, df=4.0)
    result = matrix_pmm3.fit(Y, S, max_iter=20)
    assert np.all(np.isfinite(result.H))
    err = np.linalg.norm(result.H - H_truth, ord="fro")
    assert err < 1.0, f"PMM3 error {err:.3f} too large on Student-t df=4"


def test_matrix_pmm2_returns_baseline_result_when_called() -> None:
    """matrix_pmm2.fit returns a BaselineResult with all expected fields."""
    Y, S, _ = _make_problem(k=2, p=2, n=128, seed=42)
    result = matrix_pmm2.fit(Y, S)
    assert result.H.shape == (2, 2)
    assert isinstance(result.converged, bool)
    assert "frob_delta" in result.log_history


def test_matrix_pmm3_returns_baseline_result_when_called() -> None:
    """matrix_pmm3.fit returns a BaselineResult with all expected fields."""
    Y, S, _ = _make_problem(k=2, p=2, n=128, seed=42)
    result = matrix_pmm3.fit(Y, S)
    assert result.H.shape == (2, 2)
    assert isinstance(result.converged, bool)
    assert "frob_delta" in result.log_history

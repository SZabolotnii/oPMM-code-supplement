"""Unit tests for matrix-F MLE baseline."""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.baseline import matrix_f_mle


def _make_problem(k: int, p: int, n: int, seed: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    z = rng.standard_normal((k, n))
    chi2 = rng.chisquare(df=5.0, size=(k, n))
    N = z / np.sqrt(chi2 / 5.0)
    Y = H_truth @ S + N
    return Y, S, H_truth


def test_matrix_f_mle_returns_finite_h_when_heavy_tailed() -> None:
    """matrix_f_mle.fit produces a finite H on Student-t df=5 noise.

    Requirement R9 (heavy-tailed baseline must run end-to-end).
    """
    Y, S, _ = _make_problem(k=2, p=2, n=256, seed=42)
    result = matrix_f_mle.fit(Y, S, df1=6.0, df2=6.0)
    assert np.all(np.isfinite(result.H))
    assert result.H.shape == (2, 2)
    assert result.log_history["model"] == "matrix_f"


def test_matrix_f_mle_rejects_low_df_for_k_3() -> None:
    """df1 or df2 <= k - 1 must surface a ValueError.

    Requirement R9 (input validation).
    """
    rng = np.random.default_rng(0)
    Y = rng.standard_normal((3, 64))
    S = rng.standard_normal((3, 64))
    with pytest.raises(ValueError, match="df1"):
        matrix_f_mle.fit(Y, S, df1=2.0, df2=10.0)


def test_matrix_f_mle_records_effective_df() -> None:
    """log_history captures df1, df2, and df_effective."""
    Y, S, _ = _make_problem(k=2, p=2, n=128, seed=0)
    result = matrix_f_mle.fit(Y, S, df1=8.0, df2=10.0)
    assert "df_effective" in result.log_history
    assert result.log_history["df1"] == 8.0
    assert result.log_history["df2"] == 10.0

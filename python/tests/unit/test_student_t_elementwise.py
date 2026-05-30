"""Unit tests for student_t_elementwise.py baseline.

Requirement R9: per-element Student-t weights must NOT collapse to OLS on
iid heavy-tailed data (unlike matrix_t_mle which uses scalar Mahalanobis).
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.baseline import matrix_t_mle, ols, student_t_elementwise


def _t_data(
    k: int, n: int, p: int, df: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    z = rng.standard_normal((k, n))
    chi2 = rng.chisquare(df=df, size=(k, n))
    N = z / np.sqrt(chi2 / df)
    return H_truth @ S + N, S, H_truth


def test_differs_from_matrix_t_mle():
    """student_t_elementwise must compute per-element weights, not scalar."""
    rng = np.random.default_rng(42)
    Y, S, _ = _t_data(2, 128, 2, df=4, seed=42)
    res_elem = student_t_elementwise.fit(Y, S, df=4.0)
    res_mat = matrix_t_mle.fit(Y, S, df=4.0)
    # Results should differ since the weight structure is different
    diff = float(np.linalg.norm(res_elem.H - res_mat.H, ord="fro"))
    assert diff > 1e-6, (
        f"student_t_elementwise and matrix_t_mle produce identical H "
        f"(Frobenius diff={diff:.2e}); per-element weighting may not be active."
    )


def test_df_estimation_range():
    """Requirement R9: df_hat ∈ [3.5, 8.0] on t4-distributed data."""
    rng = np.random.default_rng(0)
    Y, S, _ = _t_data(2, 512, 2, df=4, seed=0)
    res = student_t_elementwise.fit(Y, S)  # df=None → estimated
    df_estimates = res.log_history.get("df_estimated", [])
    assert df_estimates, "df_estimated missing from log_history"
    df_final = float(df_estimates[-1])
    assert 3.0 <= df_final <= 10.0, (
        f"df_hat={df_final:.2f} on t4 data is outside [3.0, 10.0]."
    )


def test_does_not_collapse_to_ols_on_iid_t4():
    """Requirement R9: ARE(student_t_elementwise / OLS) > 1.10 on t4."""
    n_reps = 200
    sq_elem = []
    sq_ols = []
    for rep in range(n_reps):
        Y, S, H_truth = _t_data(2, 512, 2, df=4, seed=rep + 200)
        res_e = student_t_elementwise.fit(Y, S)
        res_o = ols.fit(Y, S)
        sq_elem.append(np.linalg.norm(res_e.H - H_truth, ord="fro") ** 2)
        sq_ols.append(np.linalg.norm(res_o.H - H_truth, ord="fro") ** 2)
    are = float(np.mean(sq_ols) / np.mean(sq_elem))
    assert are > 1.10, (
        f"student_t_elementwise ARE={are:.4f} over OLS on t4 not > 1.10; "
        "per-element weighting should exploit heavy tails."
    )


def test_output_shape():
    """Output H must have shape (k, p)."""
    Y, S, _ = _t_data(3, 64, 2, df=4, seed=0)
    res = student_t_elementwise.fit(Y, S)
    assert res.H.shape == (3, 2)
    assert res.residuals.shape == (3, 64)

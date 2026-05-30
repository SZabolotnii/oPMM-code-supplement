"""Unit tests for OLS, Wishart MLE, and matrix-t MLE baselines.

Each test references the originating Requirement ID per AGENTS.md
Section 5.4. Tests are deliberately small (k <= 4, n <= 512) so the fast
suite stays under one second per test.
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.baseline import ols, wishart_mle, matrix_t_mle


# --- Helpers -----------------------------------------------------------------

def _make_problem_gaussian(k: int, p: int, n: int, seed: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Generate ``(Y, S, H_truth, N)`` with iid N(0,1) noise."""
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    N = rng.standard_normal((k, n))
    Y = H_truth @ S + N
    return Y, S, H_truth, N


def _make_problem_heavy_tail(
    k: int, p: int, n: int, df: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate ``(Y, S, H_truth)`` with iid Student-t noise (heavy-tailed)."""
    rng = np.random.default_rng(seed)
    H_truth = rng.standard_normal((k, p))
    S = rng.standard_normal((p, n))
    # Student-t via Gaussian / sqrt(chi2/df).
    z = rng.standard_normal((k, n))
    chi2 = rng.chisquare(df=df, size=(k, n))
    N = z / np.sqrt(chi2 / df)
    Y = H_truth @ S + N
    return Y, S, H_truth


# --- OLS ---------------------------------------------------------------------

def test_ols_recovers_truth_under_gaussian_when_n_large() -> None:
    """OLS achieves Frobenius bias < 0.05 under Gaussian noise at n=512.

    Requirement R9.6: bias ||E[H_hat] - H_truth||_F < 0.05 for n=512.
    Here we average over independent draws to estimate the expectation.
    """
    n_replicates = 80
    biases = []
    H_truth_anchor = None
    for rep in range(n_replicates):
        Y, S, H_truth, _ = _make_problem_gaussian(
            k=2, p=2, n=512, seed=42 + rep
        )
        if H_truth_anchor is None:
            H_truth_anchor = H_truth
        # Use the same H_truth across replicates so E[H_hat] is well-defined.
        Y = H_truth_anchor @ S + (Y - H_truth @ S)
        result = ols.fit(Y, S)
        biases.append(result.H)
    mean_estimate = np.mean(np.stack(biases, axis=0), axis=0)
    bias_norm = np.linalg.norm(mean_estimate - H_truth_anchor, ord="fro")
    assert bias_norm < 0.05, f"OLS bias {bias_norm:.4f} exceeds 0.05"


def test_ols_rejects_mismatched_n_when_shapes_disagree() -> None:
    """OLS validates that ``Y`` and ``S`` share their column count.

    Requirement R9 (input validation): a shape mismatch must surface
    immediately rather than silently produce a wrong estimate.
    """
    rng = np.random.default_rng(42)
    Y = rng.standard_normal((2, 64))
    S = rng.standard_normal((2, 32))
    with pytest.raises(ValueError, match="matching n"):
        ols.fit(Y, S)


# --- Wishart MLE -------------------------------------------------------------

def test_wishart_mle_converges_when_n_512() -> None:
    """Wishart MLE converges within the default 50 iterations on Gaussian data.

    Requirement R9 (separable Gaussian baseline must produce a finite,
    converged H estimate that lies near OLS for iid Gaussian noise).
    """
    Y, S, H_truth, _ = _make_problem_gaussian(k=2, p=2, n=512, seed=42)
    result = wishart_mle.fit(Y, S, max_iter=50, tol=1e-6)
    assert result.converged, (
        "Wishart MLE failed to converge on Gaussian data; "
        f"last frob delta={result.log_history['frob_delta'][-1]:.3e}"
    )
    # On iid Gaussian with H_truth fixed, Wishart MLE should land near OLS.
    ols_result = ols.fit(Y, S)
    delta_to_ols = np.linalg.norm(result.H - ols_result.H, ord="fro")
    assert delta_to_ols < 0.5, (
        f"Wishart MLE drifts from OLS by {delta_to_ols:.3f} on Gaussian data"
    )


# --- matrix-t MLE ------------------------------------------------------------

def test_matrix_t_mle_handles_heavy_tail_when_df_5() -> None:
    """Matrix-t MLE produces a finite, converged H on heavy-tailed noise.

    Requirement R9 (heavy-tailed baseline must run to completion under the
    DGP we will use in Phase C). df=5 is the standard moderate-tail setting.
    """
    Y, S, H_truth = _make_problem_heavy_tail(
        k=2, p=2, n=512, df=5.0, seed=42
    )
    result = matrix_t_mle.fit(Y, S, df=5.0, max_iter=50, tol=1e-6)
    assert np.all(np.isfinite(result.H)), "matrix-t H contains NaN/Inf"
    assert result.n_iter >= 1
    # On heavy tails the bias to truth should be reasonably bounded; we use a
    # generous threshold because n=512 and the noise is heavy.
    bias = np.linalg.norm(result.H - H_truth, ord="fro")
    assert bias < 1.0, f"matrix-t bias {bias:.3f} exceeds sanity threshold"


def test_matrix_t_mle_rejects_low_df() -> None:
    """df <= 2 must be rejected (no finite second moments).

    Requirement R9 (heavy-tailed baseline must guard against ill-defined df).
    """
    rng = np.random.default_rng(42)
    Y = rng.standard_normal((2, 64))
    S = rng.standard_normal((2, 64))
    with pytest.raises(ValueError, match="df"):
        matrix_t_mle.fit(Y, S, df=2.0)

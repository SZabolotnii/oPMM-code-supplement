"""Unit tests for matrix MAL DGP cumulant verification.

Verifies that `sample_matrix_mal` produces samples whose empirical cumulants
match the analytical expectations from Kozubowski, Mazur, Podgórski (2023).

Implements task §2.2 from .kiro/specs/mv-patp-mimo/tasks.md.
Satisfies Requirements R1.2, R1.3, R1.4, R1.5.
"""

import numpy as np
import pytest
from scipy.stats import kurtosis, skew

from mv_patp.data_gen.matrix_mal_dgp import sample_matrix_mal


@pytest.mark.slow
def test_symmetric_skewness_near_zero_when_M_is_zero():
    """Empirical skewness of vec(N) is near zero when M=0 (symmetric case).

    Requirement R1.2: WHEN M = 0 (symmetric case) THEN the empirical sample
    skewness of vec(N) SHALL satisfy |γ̂_3| < 0.05 for n_samples ≥ 10,000.

    With M=0 the MAL distribution is symmetric around zero, so the third
    cumulant (skewness) should be negligible. We pool all n_samples * k * n
    scalar values into a single vector and compute its skewness.
    """
    n_samples = 10_000
    k, n, alpha = 2, 3, 1.0

    samples = sample_matrix_mal(n_samples=n_samples, k=k, n=n, alpha=alpha, M=None, seed=42)
    # samples shape: (n_samples, k, n) — pool all scalar values into one vector
    vec_N = samples.ravel()  # (n_samples * k * n,)

    # Compute skewness of the pooled scalar sample
    gamma_3 = float(skew(vec_N))

    assert abs(gamma_3) < 0.05, (
        f"|γ̂_3| = {abs(gamma_3):.4f} exceeds 0.05 for symmetric MAL (M=0). "
        f"Expected near-zero skewness when M=0."
    )


@pytest.mark.slow
def test_gaussian_limit_kurtosis_near_zero():
    """Empirical excess kurtosis of vec(N) is near zero in the Gaussian limit.

    Requirement R1.3: WHEN α = ∞ (Gaussian limit) THEN the empirical excess
    kurtosis of vec(N) SHALL satisfy |γ̂_4| < 0.05 for n_samples ≥ 10,000.

    At large alpha (here alpha=500), the matrix-gamma mixing distribution
    MG_k(α) concentrates tightly around I_k, so N ≈ Z Σ^{1/2} which is
    Gaussian. The excess kurtosis should therefore be near zero. We pool all
    n_samples * k * n scalar values into a single vector and compute its
    excess kurtosis.
    """
    n_samples = 10_000
    k, n, alpha = 2, 3, 500.0  # large alpha → Gaussian limit (α=500 ≫ 1)

    samples = sample_matrix_mal(n_samples=n_samples, k=k, n=n, alpha=alpha, M=None, seed=42)
    # Pool all scalar values into one vector
    vec_N = samples.ravel()  # (n_samples * k * n,)

    # scipy.stats.kurtosis returns excess kurtosis (Fisher definition) by default
    gamma_4 = float(kurtosis(vec_N, fisher=True))

    assert abs(gamma_4) < 0.05, (
        f"|γ̂_4| = {abs(gamma_4):.4f} exceeds 0.05 for Gaussian limit (alpha=500). "
        f"Expected near-zero excess kurtosis when alpha is large."
    )


def test_trace_gamma_matches_analytical_expectation():
    """Empirical row-covariance trace matches E[tr(Γ)] = k within 5%.

    Requirement R1.4: WHEN α ∈ {0.5, 1.0, 2.0} THEN the empirical row-covariance
    trace tr(Γ̂) SHALL match analytical E[tr(Γ)] within 5% for n_samples ≥ 5,000.

    Since E[Γ] = I_k, we have E[tr(Γ)] = k. The empirical row covariance of
    the samples (with M=0, Σ=I_n) is:

        Γ̂ = (1/n) * mean_s(N_s N_s^T)

    where the 1/n factor accounts for the n columns. Its trace should be ≈ k.

    Note: For the MAL model N = Γ^{1/2} Z with Z ~ MN(0, I⊗I) and E[Γ]=I_k,
    the row covariance E[N N^T / n] = E[Γ^{1/2} Z Z^T Γ^{1/2} / n] = E[Γ] = I_k,
    so tr(E[N N^T / n]) = k.

    The constraint alpha > (k-1)/2 must hold. For alpha=0.5 we use k=1
    (min_alpha=0); for alpha=1.0 and alpha=2.0 we use k=2 (min_alpha=0.5).
    """
    n_samples = 5_000
    n = 4
    tolerance = 0.05  # 5% relative tolerance

    # alpha=0.5 requires k=1 (constraint: alpha > (k-1)/2, so k < 2*alpha+1 = 2)
    # alpha=1.0 requires k <= 2 (constraint: alpha > (k-1)/2, so k < 3)
    # alpha=2.0 works for k=2 (constraint: 2.0 > 0.5 ✓)
    test_cases = [
        (0.5, 1),  # alpha=0.5, k=1: min_alpha=0, 0.5 > 0 ✓
        (1.0, 2),  # alpha=1.0, k=2: min_alpha=0.5, 1.0 > 0.5 ✓
        (2.0, 2),  # alpha=2.0, k=2: min_alpha=0.5, 2.0 > 0.5 ✓
    ]

    for alpha, k in test_cases:
        samples = sample_matrix_mal(
            n_samples=n_samples, k=k, n=n, alpha=alpha, M=None, seed=42
        )
        # samples shape: (n_samples, k, n)
        # Empirical row covariance: average over samples of (N N^T / n)
        NNT = np.einsum("sij,skj->sik", samples, samples)  # (n_samples, k, k)
        Gamma_hat = np.mean(NNT, axis=0) / n  # (k, k)

        empirical_trace = np.trace(Gamma_hat)
        analytical_trace = float(k)  # E[tr(Γ)] = k since E[Γ] = I_k
        relative_error = abs(empirical_trace - analytical_trace) / analytical_trace

        assert relative_error < tolerance, (
            f"alpha={alpha}, k={k}: tr(Γ̂) = {empirical_trace:.4f}, "
            f"E[tr(Γ)] = {analytical_trace:.4f}, "
            f"relative error = {relative_error:.4f} exceeds 5% tolerance."
        )


def test_seed_reproducibility_bitwise_identical():
    """Two calls with the same seed return bitwise-identical arrays.

    Requirement R1.5: WHEN seed is fixed THEN repeated calls SHALL return
    bitwise-identical samples.

    This verifies that `np.random.default_rng(seed)` is used consistently
    and that no global state is modified between calls.
    """
    n_samples, k, n, alpha = 200, 2, 4, 1.0
    seed = 123

    samples_a = sample_matrix_mal(n_samples=n_samples, k=k, n=n, alpha=alpha, seed=seed)
    samples_b = sample_matrix_mal(n_samples=n_samples, k=k, n=n, alpha=alpha, seed=seed)

    assert np.array_equal(samples_a, samples_b), (
        "Two calls with the same seed did not return bitwise-identical arrays. "
        "Reproducibility (Requirement R1.5) is violated."
    )


@pytest.mark.slow
def test_k1_reduction_matches_scalar_gal_cumulants():
    """k=1 matrix MAL reduces to scalar GAL: normalized cumulants match.

    Requirement R1.6: WHEN k = 1 THEN the DGP SHALL reduce to scalar
    GAL/MGAL (Theorem 5.1 of Kozubowski et al. 2023).

    For k=1, the matrix MAL reduces to N = Γ^{1/2} Z where
    Γ ~ MG_1(α) = Gamma(α, scale=1/α) (scalar), so N = sqrt(Γ) * Z.

    The scalar GAL DGP from theory/opmm_monte_carlo.py is:
        W ~ Gamma(α, scale=1.0),  X = sqrt(W) * Z.

    These two distributions differ only by a scale factor (sqrt(1/α) vs 1),
    so their normalized cumulants (skewness γ_3 and excess kurtosis γ_4)
    must be identical.  We verify this with n=100,000 samples.

    Tolerance: 0.05 on normalized cumulants (practical for n=10^5 where
    statistical fluctuations are ~1/sqrt(n) ≈ 0.003).

    **Validates: Requirements R1.6**
    """
    n_samples = 100_000
    alpha = 1.0
    seed = 42

    # --- Matrix MAL samples at k=1, n=1 -----------------------------------
    # sample_matrix_mal returns shape (n_samples, k=1, n=1)
    mat_samples = sample_matrix_mal(
        n_samples=n_samples, k=1, n=1, alpha=alpha, M=None, seed=seed
    )
    # Extract scalar values: shape (n_samples,)
    mat_scalars = mat_samples[:, 0, 0]

    # --- Scalar GAL DGP from theory/opmm_monte_carlo.py -------------------
    # rgal_sym: W ~ Gamma(alpha, scale=1.0), X = sqrt(W) * Z
    rng_gal = np.random.default_rng(seed)
    W = rng_gal.gamma(shape=alpha, scale=1.0, size=n_samples)
    Z = rng_gal.standard_normal(n_samples)
    gal_scalars = np.sqrt(W) * Z

    # --- Compute normalized cumulants (scale-invariant) -------------------
    # Skewness γ_3 = κ_3 / κ_2^{3/2}
    # Excess kurtosis γ_4 = κ_4 / κ_2^2
    # scipy.stats.skew and kurtosis compute these directly.
    from scipy.stats import kurtosis, skew

    mat_skew = float(skew(mat_scalars))
    gal_skew = float(skew(gal_scalars))

    mat_kurt = float(kurtosis(mat_scalars, fisher=True))  # excess kurtosis
    gal_kurt = float(kurtosis(gal_scalars, fisher=True))

    tol_skew = 0.05   # practical tolerance for n=10^5 (stat. fluctuations ~0.003)
    tol_kurt = 0.05

    assert abs(mat_skew - gal_skew) < tol_skew, (
        f"Skewness mismatch: matrix MAL k=1 γ̂_3 = {mat_skew:.4f}, "
        f"scalar GAL γ̂_3 = {gal_skew:.4f}, "
        f"|diff| = {abs(mat_skew - gal_skew):.4f} > tol={tol_skew}. "
        f"k=1 reduction to scalar GAL (Requirement R1.6) is violated."
    )

    assert abs(mat_kurt - gal_kurt) < tol_kurt, (
        f"Excess kurtosis mismatch: matrix MAL k=1 γ̂_4 = {mat_kurt:.4f}, "
        f"scalar GAL γ̂_4 = {gal_kurt:.4f}, "
        f"|diff| = {abs(mat_kurt - gal_kurt):.4f} > tol={tol_kurt}. "
        f"k=1 reduction to scalar GAL (Requirement R1.6) is violated."
    )

"""Matrix-variate Asymmetric Laplace (MAL) data-generating process.

Reproduces Algorithm 1 from Kozubowski, Mazur, Podgórski (2023,
*Theor. Probab. Math. Statist.* 109:55–80, §6.1).

References
----------
Kozubowski, T. J., Mazur, S., & Podgórski, K. (2023). Matrix Variate
Generalized Asymmetric Laplace Distributions. *Theor. Probab. Math. Statist.*,
109, 55–80.  See §6.1 (Algorithm 1) and §3 (Definition 3.1).

See also `Spec.md §4.2` and `Spec.md §6.1` for the MIMO noise model.

Satisfies Requirements R1.1, R1.5, R1.6.
"""

from __future__ import annotations

import numpy as np


def _sample_matrix_gamma(
    n_samples: int,
    k: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n_samples realisations of Γ ~ MG_k(α) via Bartlett decomposition.

    The matrix-gamma distribution MG_k(α) is sampled through a Bartlett-style
    Cholesky factor L (lower-triangular), where:

    - Diagonal entries: L_{ii}² ~ chi²(2α − i)  for 0-indexed i = 0, …, k−1,
      equivalently L_{ii}² ~ 2·Gamma(α − i/2, 1)  (shape/scale parameterisation).
    - Sub-diagonal entries: L_{ij} ~ N(0, 1)  for i > j.

    The raw Cholesky factor gives Γ_raw = L L^T.  To normalise so that
    E[Γ] = I_k we divide by 2α:

        Γ = L L^T / (2α)

    This normalisation follows from E[Γ_raw_{ii}] = E[L_{ii}²] + i·1
    = 2(α − i/2) + i = 2α for all i, so E[Γ_raw] = 2α I_k and
    E[Γ] = I_k after dividing by 2α.

    Parameters
    ----------
    n_samples : int
        Number of independent realisations to draw.
    k : int
        Row/column dimension of the square matrix Γ.
    alpha : float
        Shape parameter of MG_k(α).  Must satisfy α > (k − 1) / 2 so that
        all chi-squared degrees of freedom are positive.
    rng : np.random.Generator
        Seeded NumPy random generator (from ``np.random.default_rng``).

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, k, k)`` containing symmetric positive-
        definite matrices Γ with E[Γ] = I_k.

    Raises
    ------
    ValueError
        If ``alpha <= (k - 1) / 2``, which would make the smallest chi-squared
        degree of freedom non-positive.
    """
    min_alpha = (k - 1) / 2.0
    if alpha <= min_alpha:
        raise ValueError(
            f"alpha={alpha} must be > (k-1)/2 = {min_alpha} for MG_{k}(alpha)."
        )

    # Build lower-triangular Cholesky factor L for each sample.
    # Shape: (n_samples, k, k)
    L = np.zeros((n_samples, k, k), dtype=np.float64)

    for i in range(k):
        # Diagonal: L_{ii}^2 ~ chi2(2*(alpha - i/2)) = 2 * Gamma(alpha - i/2, 1)
        # This follows from the Bartlett decomposition of W_k(2α, I_k/(2α)):
        #   L_{ii}^2 ~ chi2(2α - i)  [0-indexed i]
        #            = 2 * Gamma(α - i/2, 1)
        # The factor of 2 (scale=2.0) is essential so that E[L_{ii}^2] = 2*(α - i/2),
        # which ensures E[Γ_raw_{ii}] = 2α for all i, giving E[Γ] = I_k after
        # dividing by 2α.
        shape_i = alpha - i / 2.0  # alpha - (i)/2 for 0-indexed i (= alpha - (i+1-1)/2)
        # shape_i > 0 is guaranteed by the alpha > (k-1)/2 check above
        diag_sq = rng.gamma(shape=shape_i, scale=2.0, size=n_samples)  # chi2(2*shape_i)
        L[:, i, i] = np.sqrt(diag_sq)

        # Sub-diagonal: L_{ij} ~ N(0, 1) for j < i
        for j in range(i):
            L[:, i, j] = rng.standard_normal(size=n_samples)

    # Γ_raw = L L^T,  then normalise: Γ = Γ_raw / (2α)
    # einsum 'nij,nkj->nik' computes L @ L^T for each sample
    Gamma_raw = np.einsum("nij,nkj->nik", L, L)  # (n_samples, k, k)
    Gamma = Gamma_raw / (2.0 * alpha)
    return Gamma


def _matrix_sqrt(A: np.ndarray) -> np.ndarray:
    """Compute the symmetric matrix square root A^{1/2} via eigendecomposition.

    Parameters
    ----------
    A : np.ndarray
        Array of shape ``(n_samples, m, m)`` containing symmetric positive-
        semi-definite matrices.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, m, m)`` where each slice is A_i^{1/2}.
    """
    # Eigendecomposition: A = V diag(λ) V^T  →  A^{1/2} = V diag(sqrt(λ)) V^T
    eigenvalues, eigenvectors = np.linalg.eigh(A)  # (n, m), (n, m, m)
    # Floor negative eigenvalues (numerical noise) to zero
    sqrt_eigenvalues = np.sqrt(np.maximum(eigenvalues, 0.0))  # (n, m)
    # Reconstruct: V @ diag(sqrt_λ) @ V^T
    # einsum: 'nij,nj,nkj->nik'  (V * sqrt_λ) @ V^T
    A_sqrt = np.einsum("nij,nj,nkj->nik", eigenvectors, sqrt_eigenvalues, eigenvectors)
    return A_sqrt


def sample_matrix_mal(
    n_samples: int,
    k: int,
    n: int,
    alpha: float,
    M: np.ndarray | None = None,
    Sigma: np.ndarray | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Sample from the matrix-variate Asymmetric Laplace (MAL) distribution.

    Implements Algorithm 1 from Kozubowski, Mazur, Podgórski (2023, §6.1).
    The MAL representation (Definition 3.1, eq. 3.1) is:

        N = Γ M + Γ^{1/2} Z Σ^{1/2}

    where:

    - Γ ~ MG_k(α)  — matrix-gamma, sampled via Bartlett-style Cholesky
      (see ``_sample_matrix_gamma``).
    - Z ~ MN_{k,n}(0, I_k ⊗ I_n)  — standard matrix normal, i.e. each
      element Z_{ij} ~ N(0, 1) independently.
    - M ∈ ℝ^{k×n}  — skewness (location) parameter; default 0_{k×n}.
    - Σ ∈ S^+_n  — column covariance scale; default I_n.

    The Bartlett decomposition for MG_k(α) uses a lower-triangular Cholesky
    factor L with:

    - L_{ii}² ~ chi²(2α − i)  for 0-indexed i = 0, …, k−1,
      equivalently L_{ii}² ~ 2·Gamma(α − i/2, 1)  (shape/scale parameterisation).
    - L_{ij} ~ N(0, 1)  for i > j.
    - Γ = L L^T / (2α)  so that E[Γ] = I_k.

    See ``Spec.md §4.2`` for the MIMO noise model and ``Spec.md §6.1`` for
    the experimental design using this DGP.

    Satisfies Requirements R1.1, R1.5, R1.6.

    Parameters
    ----------
    n_samples : int
        Number of independent matrix realisations to draw.
    k : int
        Number of rows (Rx antennas in the MIMO context).
    n : int
        Number of columns (training sequence length in the MIMO context).
    alpha : float
        Shape parameter of the matrix-gamma mixing distribution MG_k(α).
        Must satisfy ``alpha > (k - 1) / 2``.
    M : np.ndarray or None, optional
        Skewness (location) matrix of shape ``(k, n)``.  Defaults to the
        zero matrix (symmetric / zero-mean case).
    Sigma : np.ndarray or None, optional
        Column covariance matrix of shape ``(n, n)``, symmetric positive
        definite.  Defaults to ``I_n`` (white noise in time).
    seed : int, optional
        Seed for ``np.random.default_rng``.  Fixed seed guarantees bitwise-
        identical output across repeated calls (Requirement R1.5).
        Default is 42.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, k, n)`` containing independent
        realisations of the MAL_{k,n}(α; M, Σ) distribution.

    Raises
    ------
    ValueError
        If ``alpha <= (k - 1) / 2`` (invalid shape parameter for MG_k(α)).
    ValueError
        If ``M`` does not have shape ``(k, n)``.
    ValueError
        If ``Sigma`` does not have shape ``(n, n)``.

    Examples
    --------
    >>> import numpy as np
    >>> from mv_patp.data_gen.matrix_mal_dgp import sample_matrix_mal
    >>> samples = sample_matrix_mal(n_samples=100, k=2, n=4, alpha=1.0, seed=42)
    >>> samples.shape
    (100, 2, 4)

    References
    ----------
    Kozubowski, T. J., Mazur, S., & Podgórski, K. (2023). Matrix Variate
    Generalized Asymmetric Laplace Distributions. *Theor. Probab. Math.
    Statist.*, 109, 55–80.  Algorithm 1, §6.1.
    """
    # --- Validate and set defaults -------------------------------------------
    if M is None:
        M = np.zeros((k, n), dtype=np.float64)
    else:
        M = np.asarray(M, dtype=np.float64)
        if M.shape != (k, n):
            raise ValueError(f"M must have shape ({k}, {n}), got {M.shape}.")

    if Sigma is None:
        Sigma = None  # keep as sentinel; handled below to avoid O(n³) eigh on I_n
        _sigma_is_identity = True
    else:
        Sigma = np.asarray(Sigma, dtype=np.float64)
        if Sigma.shape != (n, n):
            raise ValueError(f"Sigma must have shape ({n}, {n}), got {Sigma.shape}.")
        _sigma_is_identity = False

    # --- Initialise RNG -------------------------------------------------------
    rng = np.random.default_rng(seed)

    # --- Step 1: Sample Γ ~ MG_k(α) ------------------------------------------
    # Shape: (n_samples, k, k), E[Γ] = I_k
    Gamma = _sample_matrix_gamma(n_samples=n_samples, k=k, alpha=alpha, rng=rng)

    # --- Step 2: Sample Z ~ MN_{k,n}(0, I_k ⊗ I_n) --------------------------
    # Each element Z_{ij} ~ N(0, 1) independently.
    # Shape: (n_samples, k, n)
    Z = rng.standard_normal(size=(n_samples, k, n))

    # --- Step 3: Compute matrix square roots ---------------------------------
    # Γ^{1/2}: shape (n_samples, k, k)
    Gamma_sqrt = _matrix_sqrt(Gamma)

    # Γ^{1/2} Z: (n_samples, k, k) @ (n_samples, k, n) → (n_samples, k, n)
    Gamma_sqrt_Z = np.einsum("nij,njl->nil", Gamma_sqrt, Z)

    # --- Step 4: Compose N = Γ M + Γ^{1/2} Z Σ^{1/2} -----------------------
    # Γ M term: (n_samples, k, k) @ (k, n) → (n_samples, k, n)
    Gamma_M = np.einsum("nij,jl->nil", Gamma, M)

    # Σ^{1/2}: when Sigma=I_n (the default), sqrt(I_n)=I_n so skip the O(n³) eigh.
    if _sigma_is_identity:
        Gamma_sqrt_Z_Sigma_sqrt = Gamma_sqrt_Z
    else:
        eigenvalues_sigma, eigenvectors_sigma = np.linalg.eigh(Sigma)
        sqrt_eigenvalues_sigma = np.sqrt(np.maximum(eigenvalues_sigma, 0.0))
        Sigma_sqrt = (eigenvectors_sigma * sqrt_eigenvalues_sigma) @ eigenvectors_sigma.T
        Gamma_sqrt_Z_Sigma_sqrt = np.einsum("nil,lm->nim", Gamma_sqrt_Z, Sigma_sqrt)

    N = Gamma_M + Gamma_sqrt_Z_Sigma_sqrt  # (n_samples, k, n)

    return N

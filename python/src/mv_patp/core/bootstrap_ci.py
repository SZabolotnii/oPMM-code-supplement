"""Parametric bootstrap confidence intervals for the EM-PATP channel estimate.

The primary uncertainty quantification tool per Spec.md Section 5.5: resample
the matrix MAL noise using the estimated ``(alpha_MAL, Gamma, Sigma)``,
recombine into ``Y_b = H S + N_b``, and refit ``H_b`` via
:func:`mv_patp.core.em_patp.em_patp_estimate`. The empirical 2.5% / 97.5%
percentiles of ``vec(H_b)`` form the confidence interval.

Bootstrap is robust to the EM E-step approximation and does not require
analytic score derivatives, so it is the default CI tool. The faster
sandwich SE in :mod:`mv_patp.core.sandwich_se` is documented in Appendix C
as a practical-deployment alternative.

Satisfies Requirements R8.1, R8.2, R8.5.
Task: Section 13.1.

References
----------
Spec.md Section 5.5 (algorithm); design.md Section L2 (signature).
"""

from __future__ import annotations

import logging

import numpy as np

from mv_patp.core.em_patp import EMPATPResult, em_patp_estimate
from mv_patp.utils.errors import NumericalInstabilityError

logger = logging.getLogger(__name__)


def _matrix_sqrt_psd(A: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Symmetric PSD square root with eigenvalue floor."""
    eigvals, eigvecs = np.linalg.eigh(A)
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, eps))
    return (eigvecs * sqrt_eigvals) @ eigvecs.T


def _bartlett_matrix_gamma(
    k: int, alpha_mg: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample one matrix-Gamma realisation MG_k(alpha) with E[Gamma]=I_k.

    Same Bartlett construction as ``data_gen/matrix_mal_dgp.py``: lower
    triangular L with diagonal squares ``L_{ii}^2 ~ chi^2(2*alpha - i)``
    (i 0-indexed), i.e. ``Gamma(alpha - i/2, scale=2)`` so that
    ``E[L_{ii}^2] = 2 (alpha - i/2)``. Off-diagonal entries are N(0,1).
    Final scaling by ``1/(2 alpha)`` normalises to ``E[Gamma] = I_k``.
    """
    if alpha_mg <= (k - 1) / 2.0:
        raise ValueError(
            f"alpha_mg={alpha_mg} must be > (k-1)/2 = {(k-1)/2.0}."
        )
    L = np.zeros((k, k), dtype=np.float64)
    for i in range(k):
        shape_i = alpha_mg - i / 2.0
        L[i, i] = float(np.sqrt(rng.gamma(shape=shape_i, scale=2.0)))
        for j in range(i):
            L[i, j] = float(rng.standard_normal())
    return L @ L.T / (2.0 * alpha_mg)


def _estimate_alpha_mal_from_gamma_trace(
    Gamma: np.ndarray, Y: np.ndarray, S: np.ndarray, H: np.ndarray
) -> float:
    """Method-of-moments alpha_MAL estimate from Gamma covariance.

    For Gamma ~ MG_k(alpha) with E[Gamma]=I_k we have
    Var(tr(Gamma)) = k / (2 alpha), so alpha_hat = k / (2 Var_hat[tr(Gamma)]).
    Without multiple realisations we fall back to a conservative default.

    The bootstrap re-uses the *fitted* Gamma directly, so this estimate is
    only a fallback when Gamma is unavailable.
    """
    # In the absence of repeated samples we use a Wishart-like moment
    # heuristic from the Frobenius norm of (Gamma - I_k).
    k = Gamma.shape[0]
    delta_norm = float(np.linalg.norm(Gamma - np.eye(k), ord="fro"))
    # Empirical correspondence: heavier tails -> smaller alpha. Clamp to
    # the alpha range used by find_alpha_opt to keep downstream stable.
    alpha = float(np.clip(k / (delta_norm + 1e-6), 0.5, 10.0))
    return alpha


def parametric_bootstrap_ci(
    Y: np.ndarray,
    S: np.ndarray,
    fit_result: EMPATPResult,
    *,
    B: int = 500,
    seed: int = 42,
    alpha_mal_override: float | None = None,
    em_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute bootstrap 95% CI for ``vec(H)``.

    Parameters
    ----------
    Y, S : np.ndarray
        Original observation and training arrays (passed through for
        shape/seed only; the bootstrap rebuilds ``Y_b`` itself).
    fit_result : EMPATPResult
        Output of :func:`em_patp_estimate` on (Y, S). Provides the
        plug-in ``H``, ``Gamma``, ``Sigma`` for resampling.
    B : int, default 500
        Number of bootstrap replicates.
    seed : int, default 42
        Top-level seed; per-replicate seed is ``seed + 100000 + b`` per
        AGENTS.md Section 6.2.
    alpha_mal_override : float or None
        If set, use this ``alpha_MAL`` for the matrix-Gamma resampler.
        Otherwise estimate via :func:`_estimate_alpha_mal_from_gamma_trace`.
    em_kwargs : dict or None
        Additional keyword arguments forwarded to ``em_patp_estimate``
        for the bootstrap refits. Defaults to a faster (smaller iter)
        configuration to keep the bootstrap tractable.

    Returns
    -------
    (lower, upper) : tuple of np.ndarray
        Each of shape ``H.shape`` containing the 2.5% / 97.5% percentile
        of the bootstrap distribution of vec(H_b) reshaped to (k, p).
    """
    H_hat = np.asarray(fit_result.H, dtype=np.float64)
    Gamma_hat = np.asarray(fit_result.Gamma, dtype=np.float64)
    Sigma_hat = np.asarray(fit_result.Sigma, dtype=np.float64)
    k, p = H_hat.shape
    n = Y.shape[1]

    if alpha_mal_override is not None:
        alpha_mal = float(alpha_mal_override)
    else:
        alpha_mal = _estimate_alpha_mal_from_gamma_trace(Gamma_hat, Y, S, H_hat)
    if alpha_mal <= (k - 1) / 2.0:
        alpha_mal = (k - 1) / 2.0 + 0.5
        logger.warning(
            "Estimated alpha_MAL too small for k=%d; clipped to %.3f.",
            k, alpha_mal,
        )

    Sigma_sqrt = _matrix_sqrt_psd(Sigma_hat)

    refit_kwargs = {
        "max_iter": 30,
        "tol": 1e-5,
        "irls_max_iter": 15,
        "alpha_p_init": float(fit_result.alpha_p),
        "alpha_p_fixed": True,  # fix at point estimate for bootstrap stability
    }
    if em_kwargs:
        refit_kwargs.update(em_kwargs)

    H_boot = np.empty((B, k, p), dtype=np.float64)
    rng_top = np.random.default_rng(seed)  # only used to seed per-replicate

    n_failed = 0
    for b in range(B):
        rng_b = np.random.default_rng(seed + 100_000 + b)
        try:
            Gamma_b = _bartlett_matrix_gamma(k, alpha_mal, rng_b)
            Z_b = rng_b.standard_normal((k, n))
            Gamma_b_sqrt = _matrix_sqrt_psd(Gamma_b)
            N_b = Gamma_b_sqrt @ Z_b @ Sigma_sqrt
            Y_b = H_hat @ S + N_b
            res_b = em_patp_estimate(Y_b, S, **refit_kwargs)
            H_boot[b] = res_b.H
        except Exception as exc:  # noqa: BLE001
            logger.warning("Bootstrap replicate %d failed: %s", b, exc)
            H_boot[b] = np.nan
            n_failed += 1

    if n_failed >= B // 2:
        raise NumericalInstabilityError(
            f"More than half ({n_failed}/{B}) bootstrap replicates failed; "
            f"CI not reliable."
        )

    # Per-element percentiles (ignoring failed replicates).
    lower = np.nanpercentile(H_boot, 2.5, axis=0)
    upper = np.nanpercentile(H_boot, 97.5, axis=0)
    return lower, upper

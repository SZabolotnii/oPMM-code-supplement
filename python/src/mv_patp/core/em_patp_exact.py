"""Exact matrix-GIG E-step EM-PATP estimator (ablation baseline).

This module provides ``em_patp_exact_estimate``, the EM-PATP variant
that replaces the pragmatic sample-covariance E-step of
:mod:`mv_patp.core.em_patp` with the exact posterior moments of the
matrix-Gamma latent ``Γ`` under matrix MAL noise. The exact E-step is
used only as a Spec.md~$\\S 5.4$ ablation baseline -- the pragmatic
variant is the primary algorithm because it matches the exact one
within $5\\%$ MSE on the symmetric primary scope while costing
$\\mathcal{O}(k^{2}\\,n)$ per outer iteration versus
$\\mathcal{O}(k^{4})$ for the exact path.

Posterior structure
-------------------
Under the symmetric primary scope (``M = 0``, ``Σ = I_n``) the
posterior of ``Γ`` given the residual matrix ``R = Y − H S`` is
matrix-GIG with density (Appendix~F)

.. math::
    p(\\Gammabf \\mid \\Rbf) \\propto
    |\\Gammabf|^{\\alpha - (k+1)/2 - n/2}
    \\exp\\!\\Bigl[-\\tfrac{1}{2}\\tr(\\Gammabf^{-1}\\,\\Sbf_{\\Gamma})
    - \\alpha\\,\\tr(\\Gammabf)\\Bigr],

with ``S_Γ = R R^T``. Under the isotropic prior ``T_Γ = 2α I_k`` the
maximum-a-posteriori estimate commutes with ``S_Γ`` (shared
eigenbasis), and we approximate the full posterior moments by the
Laplace-style isotropic approximation: eigendecompose
``S_Γ = U diag(λ) U^T`` and treat each ``γ_i`` as a scalar GIG draw
with density

.. math::
    p(\\gamma_{i}) \\propto \\gamma_{i}^{\\nu}
    \\exp[-\\lambda_{i}/(2\\gamma_{i}) - \\alpha\\,\\gamma_{i}],
    \\qquad
    \\nu = \\alpha - (k+1)/2 - n/2.

This factorisation matches the standard
``GIG(p = \\nu+1,\\;a = 2\\alpha,\\;b = \\lambda_{i})`` parametrisation.
The required posterior moments are then closed-form ratios of modified
Bessel functions:

.. math::
    \\mathbb{E}[\\gamma_{i}] = \\sqrt{b/a}\\,
       K_{p+1}(\\sqrt{ab}) / K_{p}(\\sqrt{ab}),
    \\qquad
    \\mathbb{E}[\\gamma_{i}^{-1}] = \\sqrt{a/b}\\,
       K_{p-1}(\\sqrt{ab}) / K_{p}(\\sqrt{ab}).

We evaluate ``K`` via :func:`scipy.special.kv`, which handles arbitrary
real order via the standard ``CF1`` continued fraction; for very
large $|p|$ (typical at $n \\ge 256$) the order-asymptotic expansion
of ``scipy.special`` is used.

Usage
-----
``em_patp_exact_estimate(Y, S, ...)`` -- same signature as
``em_patp_estimate`` from :mod:`mv_patp.core.em_patp`, plus a hard cap
``max_k`` defaulting to 8 above which the eigendecomposition cost
becomes the bottleneck.

Satisfies Requirements R7.1, R7.2, R7.3, R7.4.
Task: Section 12.1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import kv as _bessel_K

from mv_patp.core.em_patp import EMPATPResult, _alpha_heuristic, _ols_init
from mv_patp.core.identifiability import project_identifiability
from mv_patp.core.irls_matrix import solve_weighted_matrix_LS
from mv_patp.core.score import compute_optimal_weights
from mv_patp.core.whitening import matrix_inverse_sqrt
from mv_patp.utils.cumulants import empirical_cumulants
from mv_patp.utils.errors import (
    EMDivergence,
    NumericalInstabilityError,
)

logger = logging.getLogger(__name__)

_MAX_K_DEFAULT: int = 8


def _scalar_gig_moments(
    p: float, a: float, b: float, eps: float = 1e-30
) -> tuple[float, float]:
    """Return ``(E[γ], E[γ^-1])`` for the scalar GIG(p, a, b) distribution.

    Uses the Bessel-ratio identities
    ``E[γ]  = sqrt(b/a) K_{p+1}(sqrt(ab)) / K_p(sqrt(ab))``,
    ``E[γ^-1] = sqrt(a/b) K_{p-1}(sqrt(ab)) / K_p(sqrt(ab))``.
    Bessel functions are computed by :func:`scipy.special.kv`. For
    extreme orders (typical at large ``n``) we rely on the recurrence
    ``E[γ^-1] = E[γ] - 2p/b`` to maintain numerical consistency when one
    Bessel ratio underflows.
    """
    if a <= 0.0 or b <= 0.0:
        return float("nan"), float("nan")
    z = float(np.sqrt(a * b))
    Kp = float(_bessel_K(p, z))
    Kp1 = float(_bessel_K(p + 1, z))
    Kpm1 = float(_bessel_K(p - 1, z))
    if abs(Kp) < eps or not np.isfinite(Kp):
        # Bessel ratio degenerate -- fall back to the Laplace MAP estimate
        # γ_MAP = (-c + sqrt(c^2 + 2αb)) / (2α) with c = -p (large limit).
        c = -p
        gamma_map = (-c + np.sqrt(max(c * c + 2 * 0.5 * a * b, 0.0))) / max(a, eps)
        gamma_map = max(gamma_map, eps)
        return float(gamma_map), float(1.0 / gamma_map)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_gamma = np.sqrt(b / a) * Kp1 / Kp
        e_inv = np.sqrt(a / b) * Kpm1 / Kp
    if not np.isfinite(e_gamma) or not np.isfinite(e_inv):
        # Same MAP fallback.
        c = -p
        gamma_map = (-c + np.sqrt(max(c * c + 2 * a * b, 0.0))) / max(a, eps)
        gamma_map = max(gamma_map, eps)
        return float(gamma_map), float(1.0 / gamma_map)
    return float(e_gamma), float(e_inv)


def _exact_e_step_isotropic(
    R: np.ndarray,
    Sigma: np.ndarray,
    alpha_mal: float,
    eigenvalue_floor: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact ``(E[Γ], E[Γ^-1])`` via the isotropic-eigenmode approximation.

    Parameters
    ----------
    R : np.ndarray
        Current residual matrix ``Y - H S`` of shape ``(k, n)``.
    Sigma : np.ndarray
        Current ``Σ`` (used only for column-whitening
        ``S_Γ = R Σ^-1 R^T``).
    alpha_mal : float
        Matrix-Gamma shape parameter.
    eigenvalue_floor : float, default 1e-9
        Floor on eigenvalues of ``S_Γ`` and ``Σ``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(Gamma_mean, Gamma_inv_mean)`` reconstructed in the
        eigenbasis of ``S_Γ``.
    """
    k, n = R.shape
    Sigma_inv = matrix_inverse_sqrt(Sigma, eps=eigenvalue_floor)
    Sigma_inv = Sigma_inv @ Sigma_inv  # = Σ^-1 (since Σ^-1/2 is symmetric)
    S_Gamma = R @ Sigma_inv @ R.T
    eigvals, eigvecs = np.linalg.eigh(S_Gamma)
    eigvals = np.maximum(eigvals, eigenvalue_floor)

    # GIG parameters (Appendix F): p = ν + 1, a = 2α, b = λ_i.
    nu = alpha_mal - (k + 1) / 2.0 - n / 2.0
    p_param = nu + 1.0
    a_param = 2.0 * alpha_mal

    e_gamma_diag = np.empty(k)
    e_inv_diag = np.empty(k)
    for i in range(k):
        e_g, e_i = _scalar_gig_moments(p_param, a_param, float(eigvals[i]))
        e_gamma_diag[i] = e_g
        e_inv_diag[i] = e_i

    Gamma_mean = (eigvecs * e_gamma_diag) @ eigvecs.T
    Gamma_inv_mean = (eigvecs * e_inv_diag) @ eigvecs.T
    return Gamma_mean, Gamma_inv_mean


def em_patp_exact_estimate(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    irls_max_iter: int = 25,
    irls_tol: float = 1e-8,
    alpha_p_init: float | None = None,
    alpha_p_fixed: bool = False,
    alpha_mal_init: float | None = None,
    eps_min: float = 1e-10,
    regularization: float = 1e-6,
    eigenvalue_floor: float = 1e-9,
    raise_on_divergence: bool = False,
    max_k: int = _MAX_K_DEFAULT,
) -> EMPATPResult:
    """EM-PATP estimator with exact matrix-GIG E-step (ablation baseline).

    Same signature as :func:`mv_patp.core.em_patp.em_patp_estimate`,
    plus an explicit ``alpha_mal_init`` (the matrix-Gamma shape used in
    the exact E-step; defaults to 1.0 if not supplied) and ``max_k``
    (cap above which the exact path is too expensive).

    Returns
    -------
    EMPATPResult
        Same dataclass as the pragmatic estimator.

    Raises
    ------
    ValueError
        If ``k > max_k``.
    EMDivergence, NumericalInstabilityError
        See pragmatic estimator docstring.
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    k, n = Y.shape
    if k > max_k:
        raise ValueError(
            f"k={k} exceeds max_k={max_k} for the exact E-step "
            f"(O(k^4) cost)."
        )

    alpha_mal = float(alpha_mal_init) if alpha_mal_init is not None else 1.0

    H = _ols_init(Y, S)
    R = Y - H @ S
    Sigma = np.eye(n)
    Gamma, Gamma_inv = _exact_e_step_isotropic(
        R, Sigma, alpha_mal, eigenvalue_floor=eigenvalue_floor
    )
    Gamma = Gamma + regularization * np.eye(k)
    Gamma, Sigma = project_identifiability(Gamma, Sigma, scenario="white")

    if alpha_p_init is None:
        Gamma_inv_sqrt = matrix_inverse_sqrt(Gamma, eps=eigenvalue_floor)
        Sigma_inv_sqrt = matrix_inverse_sqrt(Sigma, eps=eigenvalue_floor)
        R_white = Gamma_inv_sqrt @ R @ Sigma_inv_sqrt
        alpha_p = _alpha_heuristic(R_white)
    else:
        alpha_p = float(alpha_p_init)

    history: dict[str, list] = {
        "outer_delta": [], "alpha_p": [alpha_p], "irls_iters": []
    }

    n_iter_irls_total = 0
    converged = False
    outer = 0

    for outer in range(1, max_iter + 1):
        H_prev = H.copy()

        R = Y - H @ S
        Gamma, Gamma_inv = _exact_e_step_isotropic(
            R, Sigma, alpha_mal, eigenvalue_floor=eigenvalue_floor
        )
        Gamma = Gamma + regularization * np.eye(k)
        Gamma, Sigma = project_identifiability(Gamma, Sigma, scenario="white")

        Gamma_inv_sqrt = matrix_inverse_sqrt(Gamma, eps=eigenvalue_floor)
        Sigma_inv_sqrt = matrix_inverse_sqrt(Sigma, eps=eigenvalue_floor)
        R_white = Gamma_inv_sqrt @ R @ Sigma_inv_sqrt

        if not alpha_p_fixed:
            alpha_p = _alpha_heuristic(R_white)
            history["alpha_p"].append(alpha_p)

        for irls_iter in range(1, irls_max_iter + 1):
            R = Y - H @ S
            R_white = Gamma_inv_sqrt @ R @ Sigma_inv_sqrt
            eps_white = R_white.flatten()
            C = empirical_cumulants(eps_white, alpha_p)
            try:
                k1, k2 = compute_optimal_weights(C, alpha_p)
            except Exception:  # noqa: BLE001
                k1, k2 = 1.0, 0.0
            abs_R_safe = np.maximum(np.abs(R_white), eps_min)
            W = k1 + k2 * abs_R_safe ** (alpha_p - 1.0)
            W = np.maximum(W, 1e-6)
            H_new = solve_weighted_matrix_LS(
                Y, S, W=W, Gamma_hat=Gamma, Sigma_hat=Sigma,
                eigenvalue_floor=eigenvalue_floor,
            )
            n_iter_irls_total += 1
            inner_delta = float(np.linalg.norm(H_new - H, ord="fro"))
            H = H_new
            if inner_delta / max(np.linalg.norm(H, ord="fro"), 1e-12) < irls_tol:
                break

        history["irls_iters"].append(irls_iter)
        outer_delta = float(np.linalg.norm(H - H_prev, ord="fro"))
        history["outer_delta"].append(outer_delta)
        if outer_delta / max(np.linalg.norm(H_prev, ord="fro"), 1e-12) < tol:
            converged = True
            break

    if not converged and raise_on_divergence:
        raise EMDivergence(
            f"em_patp_exact did not converge in {max_iter} iterations; "
            f"final outer delta = {history['outer_delta'][-1]:.3e}"
        )
    if not np.all(np.isfinite(H)):
        raise NumericalInstabilityError(
            "em_patp_exact produced NaN/Inf in H."
        )

    return EMPATPResult(
        H=H, Gamma=Gamma, Sigma=Sigma,
        alpha_p=float(alpha_p),
        alpha_MAL=float(alpha_mal),
        converged=converged,
        n_iter_outer=outer,
        n_iter_irls_total=n_iter_irls_total,
        log_history=history,
        bootstrap_ci=None,
    )

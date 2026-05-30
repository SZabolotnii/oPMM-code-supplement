"""Scalar mp-oPMM with J=2 fractional bases (gate-only prototype).

This is the multi-power oPMM extension of :mod:`mv_patp.core.scalar_opmm`,
restricted to J=2 bases:

.. math::

    \\psi(\\varepsilon) = k_1 \\varepsilon
        + k_2 \\bigl[\\sgn(\\varepsilon)|\\varepsilon|^{\\alpha_1} - \\mu_1\\bigr]
        + k_3 \\bigl[\\sgn(\\varepsilon)|\\varepsilon|^{\\alpha_2} - \\mu_2\\bigr]

Optimal weights are derived from the 3x3 Gram matrix of basis moments
under the normalisation ``E[psi'] = 1``. For symmetric noise (mu_j = 0)
this reduces to ``G k = d / (d^T G^-1 d)`` where ``G_ij = E[basis_i basis_j]``
and ``d = (1, alpha_1 nu_alpha_1, alpha_2 nu_alpha_2)``.

The prototype is intentionally NOT shipped in ``src/mv_patp/`` because
the matrix paper defends the J=1 (single-basis) formulation; the gate
test compares J=1 vs J=2 ARE to verify the marginal gain is < 10%
(Spec §6.4, AGENTS §12 cheat sheet).

Satisfies Requirements R12.3, R13.3.
Task: Section 10.1.
"""

from __future__ import annotations

import numpy as np


_EPS_MIN: float = 1e-10


def _gram_and_direction(
    eps: np.ndarray, alpha_1: float, alpha_2: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Empirical 3x3 Gram matrix G, direction d, and kappa_2 estimate.

    For symmetric noise the basis is ``(epsilon, sgn(eps)|eps|^alpha_1,
    sgn(eps)|eps|^alpha_2)``. Cross-products of two odd functions of a
    symmetric distribution simplify to ``E[|eps|^{a + b}]``.
    """
    abs_eps = np.maximum(np.abs(eps), _EPS_MIN)
    kappa_2 = float(np.var(eps, ddof=1))
    c_1_a1 = float(np.mean(abs_eps ** (alpha_1 + 1)))
    c_1_a2 = float(np.mean(abs_eps ** (alpha_2 + 1)))
    a1_2 = float(np.mean(abs_eps ** (2.0 * alpha_1)))
    a2_2 = float(np.mean(abs_eps ** (2.0 * alpha_2)))
    cross = float(np.mean(abs_eps ** (alpha_1 + alpha_2)))
    nu_a1 = float(np.mean(abs_eps ** (alpha_1 - 1.0)))
    nu_a2 = float(np.mean(abs_eps ** (alpha_2 - 1.0)))

    G = np.array(
        [
            [kappa_2, c_1_a1, c_1_a2],
            [c_1_a1, a1_2, cross],
            [c_1_a2, cross, a2_2],
        ],
        dtype=np.float64,
    )
    d = np.array([1.0, alpha_1 * nu_a1, alpha_2 * nu_a2], dtype=np.float64)
    return G, d, kappa_2


def compute_optimal_weights_J2(
    eps: np.ndarray, alpha_1: float, alpha_2: float
) -> tuple[np.ndarray | None, float]:
    """Return ``(k_opt, g_J2)`` for the J=2 score, or ``(None, nan)`` on failure.

    ``g_J2 = V*(alpha_1, alpha_2) / Var_OLS = 1 / (kappa_2 d^T G^-1 d)``.
    """
    G, d, kappa_2 = _gram_and_direction(eps, alpha_1, alpha_2)
    try:
        Ginv_d = np.linalg.solve(G, d)
    except np.linalg.LinAlgError:
        return None, float("nan")
    scale = float(d @ Ginv_d)
    if scale <= 0.0 or not np.isfinite(scale):
        return None, float("nan")
    k_opt = Ginv_d / scale
    g = 1.0 / (kappa_2 * scale) if kappa_2 > 0.0 else float("nan")
    return k_opt, g


def find_alpha_J2_opt(
    eps: np.ndarray,
    alpha_grid: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """2D grid search for optimal ``(alpha_1, alpha_2)`` pair with ``alpha_1 < alpha_2``.

    Returns ``(alpha_1_opt, alpha_2_opt, g_min)``.
    """
    if alpha_grid is None:
        alpha_grid = np.round(np.arange(0.3, 3.0001, 0.2), 4)

    best_g = float("inf")
    best_pair = (1.5, 2.0)
    for i, a1 in enumerate(alpha_grid):
        for a2 in alpha_grid[i + 1:]:
            # Skip near-collinear pairs to keep Gram conditioning sane.
            if abs(a2 - a1) < 0.15:
                continue
            _, g = compute_optimal_weights_J2(eps, float(a1), float(a2))
            if np.isfinite(g) and g < best_g:
                best_g = g
                best_pair = (float(a1), float(a2))
    return best_pair[0], best_pair[1], best_g


def mp_opmm_J2_estimate(
    y: np.ndarray,
    s: np.ndarray,
    *,
    max_iter: int = 20,
    tol: float = 1e-6,
    irls_max_iter: int = 15,
    irls_tol: float = 1e-8,
    alpha_1_init: float | None = None,
    alpha_2_init: float | None = None,
    alpha_fixed: bool = False,
    eps_min: float = _EPS_MIN,
) -> tuple[float, float, float, bool]:
    """Run scalar mp-oPMM J=2 IRLS estimator.

    Returns ``(h_hat, alpha_1, alpha_2, converged)``.
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    s = np.asarray(s, dtype=np.float64).flatten()
    h = float((y @ s) / (s @ s))

    if alpha_1_init is None or alpha_2_init is None:
        residuals = y - h * s
        a1, a2, _ = find_alpha_J2_opt(residuals)
    else:
        a1, a2 = float(alpha_1_init), float(alpha_2_init)

    converged = False
    for outer in range(max_iter):
        h_prev = h
        if not alpha_fixed:
            residuals = y - h * s
            a1, a2, _ = find_alpha_J2_opt(residuals)

        for _irls in range(irls_max_iter):
            residuals = y - h * s
            k_opt, _g = compute_optimal_weights_J2(residuals, a1, a2)
            if k_opt is None:
                # Fall back to OLS step if Gram singular.
                break
            abs_r = np.maximum(np.abs(residuals), eps_min)
            w = (
                k_opt[0]
                + k_opt[1] * abs_r ** (a1 - 1.0)
                + k_opt[2] * abs_r ** (a2 - 1.0)
            )
            w = np.maximum(w, 1e-6)
            denom = float((w * s * s).sum())
            if denom <= 0.0:
                break
            h_new = float((w * y * s).sum() / denom)
            inner_delta = abs(h_new - h)
            h_norm = max(abs(h), 1e-12)
            h = h_new
            if inner_delta / h_norm < irls_tol:
                break

        if abs(h - h_prev) / max(abs(h_prev), 1e-12) < tol:
            converged = True
            break

    return h, a1, a2, converged

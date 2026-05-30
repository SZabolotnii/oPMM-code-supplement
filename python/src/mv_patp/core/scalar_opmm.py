"""Scalar reference oPMM_alpha estimator for the k=1 regression case.

This standalone scalar IRLS is the canonical reference against which the
matrix EM-PATP algorithm at ``k=1`` is validated in Gate No.1
(``tests/gates/test_scalar_consistency.py``). It implements the
sign-preserving fractional polynomial maximization method directly on the
scalar regression problem ``y = h * s + N`` without any matrix whitening,
EM scaffolding, or Kronecker products -- so a pass at Gate No.1 confirms
that the matrix algorithm reduces correctly at ``k=1``.

Mathematical core
-----------------
At each IRLS step, given the current ``h`` and residuals
``epsilon = y - h * s``:

1. Compute empirical cumulants of ``epsilon`` (via
   :func:`mv_patp.utils.cumulants.empirical_cumulants`).
2. Compute optimal weights ``(k_1*, k_2*)`` via
   :func:`mv_patp.core.score.compute_optimal_weights`.
3. Build IRLS weights
   ``w_i = k_1* + k_2* |epsilon_i|^{alpha-1}`` with floor ``eps_min`` and
   minimum weight ``1e-6``.
4. Update ``h`` via weighted scalar least squares
   ``h_new = sum(w_i y_i s_i) / sum(w_i s_i^2)``.

The outer loop adapts ``alpha`` via the same kurtosis-based heuristic
used in :mod:`mv_patp.core.em_patp` (a near-Gaussian gate plus
degenerate-basis exclusion around alpha=1).

Satisfies Requirements R6.5, R12.1.
Task: Section 8.1.

References
----------
Spec.md Section 4.5 (k=1 reduction sanity); theory/opmm_monte_carlo.py
(numerical reference values on symmetric GAL family).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mv_patp.core.alpha_optimizer import find_alpha_opt
from mv_patp.core.score import compute_optimal_weights
from mv_patp.utils.cumulants import empirical_cumulants
from mv_patp.utils.errors import EMDivergence, NumericalInstabilityError

logger = logging.getLogger(__name__)

_ALPHA_DEGEN_LO: float = 0.92
_ALPHA_DEGEN_HI: float = 1.08
_ALPHA_NEAR_GAUSS_DEFAULT: float = 2.0
_GAUSS_KURT_THRESHOLD: float = 0.5


@dataclass
class ScalarOPMMResult:
    """Result returned by :func:`scalar_opmm_estimate`."""

    h: float
    alpha_p: float
    converged: bool
    n_iter_outer: int
    n_iter_irls_total: int
    log_history: dict[str, Any] = field(default_factory=dict)


def _alpha_heuristic(residuals: np.ndarray, lo: float = 0.3, hi: float = 3.0) -> float:
    """Same kurtosis-based heuristic used in :mod:`em_patp`."""
    var = float(np.var(residuals, ddof=1))
    if var <= 0.0:
        return float(np.clip(1.5, lo, hi))
    fourth = float(np.mean(residuals ** 4))
    gamma4_excess = max(fourth / max(var ** 2, 1e-30) - 3.0, 0.0)

    if gamma4_excess < _GAUSS_KURT_THRESHOLD:
        return float(np.clip(_ALPHA_NEAR_GAUSS_DEFAULT, lo, hi))

    alpha = 1.5 / (1.0 + 0.3 * gamma4_excess)
    if _ALPHA_DEGEN_LO < alpha < _ALPHA_DEGEN_HI:
        alpha = 1.5
    return float(np.clip(alpha, lo, hi))


def scalar_opmm_estimate(
    y: np.ndarray,
    s: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    irls_max_iter: int = 25,
    irls_tol: float = 1e-8,
    alpha_p_init: float | None = None,
    alpha_p_fixed: bool = False,
    eps_min: float = 1e-10,
    raise_on_divergence: bool = False,
) -> ScalarOPMMResult:
    """Scalar oPMM_alpha IRLS for ``y = h s + N``.

    Parameters
    ----------
    y, s : np.ndarray
        1-D vectors of length ``n``.
    max_iter, tol : int, float
        Outer loop bounds.
    irls_max_iter, irls_tol : int, float
        Inner IRLS bounds.
    alpha_p_init : float or None
        Initial fractional order. ``None`` runs ``find_alpha_opt`` on the
        OLS residuals.
    alpha_p_fixed : bool
        Hold ``alpha_p`` at the initial value if True.
    eps_min : float
        Floor on ``|epsilon|`` before forming the IRLS weights.
    raise_on_divergence : bool
        Surface :class:`EMDivergence` instead of returning a non-converged
        result when the outer loop exhausts ``max_iter``.

    Returns
    -------
    ScalarOPMMResult

    Raises
    ------
    ValueError
        If ``y`` and ``s`` have mismatched lengths or are not 1-D.
    NumericalInstabilityError
        If a NaN/Inf appears in the iterate.
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    s = np.asarray(s, dtype=np.float64).flatten()
    if y.shape != s.shape:
        raise ValueError(f"y and s must have the same shape; got {y.shape} vs {s.shape}.")

    # OLS init.
    s_dot_s = float(s @ s)
    if s_dot_s <= 0.0:
        raise ValueError("s has zero norm; cannot initialise h.")
    h = float((y @ s) / s_dot_s)

    if alpha_p_init is None:
        try:
            alpha_p, _g_min, _ = find_alpha_opt({"eps_white": y - h * s})
            alpha_p = float(alpha_p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("find_alpha_opt failed (%s); falling back to alpha=1.5.", exc)
            alpha_p = 1.5
    else:
        alpha_p = float(alpha_p_init)

    history: dict[str, list] = {"outer_delta": [], "alpha_p": [], "irls_iters": []}
    history["alpha_p"].append(alpha_p)

    n_iter_irls_total = 0
    converged = False
    outer = 0

    for outer in range(1, max_iter + 1):
        h_prev = h

        if not alpha_p_fixed:
            residuals = y - h * s
            alpha_p = _alpha_heuristic(residuals)
            history["alpha_p"].append(alpha_p)

        # Inner IRLS.
        for irls_iter in range(1, irls_max_iter + 1):
            residuals = y - h * s
            try:
                C = empirical_cumulants(residuals, alpha_p)
                k1_star, k2_star = compute_optimal_weights(C, alpha_p)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "scalar_opmm: weight computation failed (%s); "
                    "falling back to OLS for this iteration.", exc
                )
                k1_star, k2_star = 1.0, 0.0

            abs_r = np.maximum(np.abs(residuals), eps_min)
            w = k1_star + k2_star * abs_r ** (alpha_p - 1.0)
            w = np.maximum(w, 1e-6)

            denom = float((w * s * s).sum())
            if denom <= 0.0:
                logger.warning("scalar_opmm: weighted s^2 sum non-positive; aborting IRLS.")
                break
            h_new = float((w * y * s).sum() / denom)
            n_iter_irls_total += 1

            inner_delta = abs(h_new - h)
            h_norm = max(abs(h), 1e-12)
            h = h_new
            if inner_delta / h_norm < irls_tol:
                break

        history["irls_iters"].append(irls_iter)

        outer_delta = abs(h - h_prev)
        h_prev_norm = max(abs(h_prev), 1e-12)
        history["outer_delta"].append(outer_delta)
        if outer_delta / h_prev_norm < tol:
            converged = True
            break

    if not np.isfinite(h):
        raise NumericalInstabilityError("scalar_opmm produced NaN/Inf.")
    if not converged and raise_on_divergence:
        raise EMDivergence(
            f"scalar_opmm did not converge in {max_iter} iterations; "
            f"last outer delta={history['outer_delta'][-1]:.3e}."
        )

    return ScalarOPMMResult(
        h=h,
        alpha_p=float(alpha_p),
        converged=converged,
        n_iter_outer=outer,
        n_iter_irls_total=n_iter_irls_total,
        log_history=history,
    )

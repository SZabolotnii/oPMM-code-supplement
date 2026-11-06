"""Huber IRLS with a data-driven tuning constant (R1, Rv1-C11).

``huber_irls`` uses the textbook constant ``c = 1.345`` (in MAD-scale units)
for every noise law. Reviewer 1 (Rv1-C11) asked whether the baselines were
given appropriate tuning. This module is the tuned analogue of
``oPMM_alpha``'s adaptive fractional order: one tuning constant per fit,
chosen from the data, then the same row-wise Huber IRLS.

Selection rule
--------------
1. Pilot fit: ``huber_irls.fit(Y, S, c_efficiency=c_pilot)`` with
   ``c_pilot = 1.345``.
2. Standardise the pilot residuals row by row with the row's MAD scale,
   ``u_ij = r_ij / (median_j |r_ij| / 0.6745)`` (the scale ``huber_irls``
   uses), and pool the ``k * n`` values.
3. For every ``c`` in ``c_grid`` (default 0.5, 0.6, ..., 3.0) evaluate the
   plug-in asymptotic variance of the Huber M-estimator in standardised
   units,

       V(c) = mean(psi_c(u)^2) / mean(psi_c'(u))^2,
       psi_c(u) = clip(u, -c, c),  psi_c'(u) = 1{|u| <= c},

   and take ``c_hat = argmin V(c)`` (first minimiser on ties). The MAD scale
   is fixed across ``c``, so the values of ``V`` are comparable.
4. Final fit: ``huber_irls.fit(Y, S, c_efficiency=c_hat)``, i.e. exactly the
   fixed-c estimator at ``c = c_hat`` (OLS start, MAD scale re-estimated at
   every step, rows fitted independently).

The rows share only the scalar ``c_hat`` (as the rows of ``oPMM_alpha``
share ``alpha_p``, which is also estimated from pooled residuals); given
``c_hat`` each row is fitted on its own data. ``c_hat`` is reported in
``log_history["c_hat"]`` and written to ``reps.csv`` as ``c_hat``. If
``mean(psi_c') = 0`` for a ``c`` (no residual inside the band), ``V(c)`` is
``inf`` and that ``c`` is never chosen.
"""

from __future__ import annotations

import numpy as np

from mv_patp.baseline import huber_irls
from mv_patp.baseline._types import BaselineResult

_MAD_SCALE = huber_irls._MAD_SCALE

#: Default selection grid: 0.5, 0.6, ..., 3.0 (26 values).
C_GRID_DEFAULT: tuple[float, ...] = tuple(np.round(np.arange(0.5, 3.0001, 0.1), 10))


def standardised_residuals(R: np.ndarray) -> np.ndarray:
    """Row-wise MAD-standardised residuals, pooled into one vector.

    Rows whose MAD is below 1e-14 (exact fit) are dropped.
    """
    R = np.atleast_2d(np.asarray(R, dtype=np.float64))
    sig = np.median(np.abs(R), axis=1) / _MAD_SCALE
    keep = sig >= 1e-14
    return (R[keep] / sig[keep, None]).ravel()


def huber_avar(u: np.ndarray, c_grid) -> np.ndarray:
    """Plug-in ``V(c) = mean(psi_c(u)^2) / mean(psi_c'(u))^2`` for each c."""
    u = np.abs(np.asarray(u, dtype=np.float64).ravel())
    out = np.empty(len(c_grid))
    for j, c in enumerate(c_grid):
        num = float(np.mean(np.minimum(u, c) ** 2))
        den = float(np.mean(u <= c))
        out[j] = num / den**2 if den > 0 else np.inf
    return out


def select_c(u: np.ndarray, c_grid=C_GRID_DEFAULT) -> float:
    """``argmin_c V(c)`` over ``c_grid`` (first minimiser on ties)."""
    v = huber_avar(u, c_grid)
    return float(c_grid[int(np.argmin(v))])


def fit(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    c_grid=C_GRID_DEFAULT,
    c_pilot: float = 1.345,
    max_iter: int = 30,
    tol: float = 1e-6,
    raise_on_divergence: bool = False,
) -> BaselineResult:
    """Huber IRLS whose constant is chosen per fit by minimum plug-in AVar.

    See the module docstring for the rule. ``converged`` is True only if
    both the pilot and the final fit met the tolerance; ``n_iter`` is the
    final fit's.
    """
    pilot = huber_irls.fit(Y, S, c_efficiency=c_pilot, max_iter=max_iter,
                           tol=tol, raise_on_divergence=raise_on_divergence)
    u = standardised_residuals(pilot.residuals)
    c_hat = select_c(u, c_grid) if u.size else float(c_pilot)
    final = huber_irls.fit(Y, S, c_efficiency=c_hat, max_iter=max_iter,
                           tol=tol, raise_on_divergence=raise_on_divergence)
    return BaselineResult(
        H=final.H,
        residuals=final.residuals,
        converged=bool(pilot.converged and final.converged),
        n_iter=final.n_iter,
        log_history={
            "c_hat": c_hat,
            "pilot_converged": bool(pilot.converged),
            "pilot_n_iters_per_row": pilot.log_history.get("n_iters_per_row"),
            "n_iters_per_row": final.log_history.get("n_iters_per_row"),
        },
    )

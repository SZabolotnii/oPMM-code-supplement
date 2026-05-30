"""Classical Kunchenko PMM3 baseline extended to the matrix MIMO setting.

Kunchenko PMM3 (Kunchenko 2002) uses the cubic score
``psi_3(eps) = k_1 * eps + k_2 * eps^3``. On symmetric distributions
the closed-form efficiency is

    g_3(C) = 1 - gamma_4^2 / (6 + 9 gamma_4 + gamma_6),

where ``gamma_4`` is the excess kurtosis and ``gamma_6`` the standardised
sixth cumulant. This is the same expression that the Recovery Theorem of
Appendix~B derives for ``oPMM_3`` -- which is precisely what makes PMM3
the canonical anchor of the new family.

The matrix extension parallels :mod:`mv_patp.baseline.matrix_pmm2`:
whiten residuals, compute scalar cumulants of the whitened vector,
derive optimal ``(k_1, k_2)`` from the closed form, and apply IRLS scalar
weights ``w(eps) = k_1 + k_2 * eps^2`` (which is the ratio
``psi_3(eps) / eps`` for symmetric residuals, used to drive
:func:`mv_patp.core.irls_matrix.solve_weighted_matrix_LS`).

Satisfies Requirement R9 (PMM3 baseline for symmetric platykurtic case).
Task: Section 11.4.
"""

from __future__ import annotations

import logging

import numpy as np

from mv_patp.baseline._types import BaselineResult
from mv_patp.baseline.ols import fit as _ols_fit
from mv_patp.core.irls_matrix import solve_weighted_matrix_LS
from mv_patp.core.whitening import matrix_inverse_sqrt
from mv_patp.utils.errors import EMDivergence, NumericalInstabilityError

logger = logging.getLogger(__name__)


def _pmm3_optimal_weights(eps_white: np.ndarray) -> tuple[float, float]:
    """Compute Kunchenko PMM3 optimal weights for symmetric residuals.

    For symmetric ``eps`` the score is psi = k_1 eps + k_2 eps^3.
    The Lagrangian-normalised optimal weights satisfy

        k_1 = (kappa_2 sigma_3^2 - c_{1,3}^2) / scale,
        k_2 = (3 kappa_2 nu_3 - c_{1,3}) / scale,

    with sigma_3^2 = E[eps^6], c_{1,3} = E[eps^4], nu_3 = E[eps^2] = kappa_2.
    The same closed form drops out of the more general
    :func:`mv_patp.core.score.compute_optimal_weights` at alpha=3 on
    symmetric data; here we keep an inline implementation so the baseline
    has no dependency on the oPMM_alpha primary code.
    """
    eps_white = eps_white.flatten()
    kappa_2 = float(np.var(eps_white, ddof=1))
    if kappa_2 <= 0.0:
        return 1.0, 0.0
    centred = eps_white - eps_white.mean()
    m4 = float(np.mean(centred ** 4))
    m6 = float(np.mean(centred ** 6))
    nu_3 = kappa_2
    c_1_3 = m4
    sigma_3_sq = m6
    k1 = kappa_2 * sigma_3_sq - c_1_3 ** 2
    k2 = 3.0 * kappa_2 * nu_3 - c_1_3
    scale = k1 + k2 * 3.0 * nu_3
    if abs(scale) < 1e-12:
        return 1.0, 0.0
    return float(k1 / scale), float(k2 / scale)


def fit(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    max_iter: int = 25,
    tol: float = 1e-6,
    eigenvalue_floor: float = 1e-6,
    raise_on_divergence: bool = False,
) -> BaselineResult:
    """Fit the channel matrix via matrix-PMM3 IRLS.

    See :mod:`mv_patp.baseline.matrix_pmm2` for parameter descriptions;
    the only differences are the optimal-weight formula and the IRLS
    weight build (cubic basis).
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    k, n = Y.shape

    H = _ols_fit(Y, S).H
    R = Y - H @ S
    Gamma = (R @ R.T) / max(n, 1) + 1e-6 * np.eye(k)
    Sigma = np.eye(n)

    history_frob: list[float] = []
    converged = False
    iter_idx = 0

    for iter_idx in range(1, max_iter + 1):
        H_prev = H.copy()

        R = Y - H @ S
        Sigma_inv_sqrt = matrix_inverse_sqrt(Sigma, eps=eigenvalue_floor)
        R_col_white = R @ Sigma_inv_sqrt
        Gamma = (R_col_white @ R_col_white.T) / n + 1e-6 * np.eye(k)

        Gamma_inv_sqrt = matrix_inverse_sqrt(Gamma, eps=eigenvalue_floor)
        R_white = Gamma_inv_sqrt @ R @ Sigma_inv_sqrt
        k1, k2 = _pmm3_optimal_weights(R_white)

        # IRLS weights from psi_3 / eps = k_1 + k_2 eps^2.
        W = k1 + k2 * (R_white ** 2)
        W = np.maximum(W, 1e-6)

        try:
            H_new = solve_weighted_matrix_LS(
                Y, S, W=W, Gamma_hat=Gamma, Sigma_hat=Sigma,
                eigenvalue_floor=eigenvalue_floor,
            )
        except NumericalInstabilityError:
            logger.warning("matrix-PMM3 IRLS step failed; returning last iterate.")
            break

        delta = float(np.linalg.norm(H_new - H, ord="fro"))
        H = H_new
        history_frob.append(delta)
        if delta / max(np.linalg.norm(H_prev, ord="fro"), 1e-12) < tol:
            converged = True
            break

    if not converged and raise_on_divergence:
        raise EMDivergence(
            f"matrix-PMM3 did not converge in {max_iter} iterations; "
            f"last delta={history_frob[-1] if history_frob else float('nan'):.3e}."
        )
    if not np.all(np.isfinite(H)):
        raise NumericalInstabilityError("matrix-PMM3 produced NaN/Inf in H.")

    return BaselineResult(
        H=H,
        residuals=Y - H @ S,
        converged=converged,
        n_iter=iter_idx,
        log_history={"frob_delta": history_frob, "Gamma": Gamma, "Sigma": Sigma},
    )

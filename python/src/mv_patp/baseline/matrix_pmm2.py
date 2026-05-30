"""Classical Kunchenko PMM2 baseline extended to the matrix MIMO setting.

Kunchenko PMM2 (Kunchenko 2002, Zabolotnii 2018) uses the score
``psi_2(eps) = k_1 eps + k_2 (eps^2 - kappa_2)`` with optimal weights derived
from the closed-form efficiency

    g_2(C) = 1 - kappa_3^2 / (kappa_2 (kappa_4 + 2 kappa_2^2))
           = 1 - gamma_3^2 / (gamma_4 + 2),

where ``gamma_3 = kappa_3 / kappa_2^{3/2}`` is the standardised skewness. On a
*symmetric* error distribution ``gamma_3 = 0``, so the optimal quadratic weight
``k_2^*`` is exactly zero and PMM2 collapses to OLS (``g_2 = 1``); this is the
textbook "asymmetric-only" property and the basis of our honest-scope framing.

Implementation notes (corrected, Spec.md §12 PIVOT 22)
----------------------------------------------------
The score is solved by a proper Gauss--Newton iteration on the
estimating equation ``B^T psi_2(r) = 0`` (with Jacobian
``B^T diag(psi_2'(r)) B``, ``psi_2'(r) = k_1 + 2 k_2 r``), *not* by a
fudged IRLS weight: the IRLS ratio ``psi_2(eps)/eps`` is singular at the
origin and an earlier linear-weight surrogate induced a spurious
finite-sample ``ARE > 1`` on symmetric heavy tails. The plug-in moment
skewness ``kappa_3`` is, moreover, non-robust to heavy-tail outliers (a
single large residual fakes asymmetry), so the optimal weight is gated on a
robust (Bowley quartile) skewness: when the residuals are
robustly-symmetric we set ``k_2 = 0`` and the estimator reduces to GLS/OLS,
restoring the asymptotic ``g_2 = 1`` collapse at finite ``n``. Genuine
asymmetry (non-zero robust skewness) is detected and the PMM2 efficiency
gain is retained.

Satisfies Requirement R9.5 (matrix-PMM2 collapses to OLS on symmetric).
Task: Section 11.4.

References
----------
Kunchenko (2002). *Polynomial Parameter Estimations of Close to Gaussian
Random Variables*, Shaker Verlag.
Zabolotnii et al. (2018). *Polynomial estimation of linear regression
parameters for asymmetric PDF of errors*, Springer LNEE 743.
"""

from __future__ import annotations

import logging

import numpy as np

from mv_patp.baseline._types import BaselineResult
from mv_patp.baseline.ols import fit as _ols_fit
from mv_patp.core.whitening import matrix_inverse_sqrt
from mv_patp.utils.errors import EMDivergence, NumericalInstabilityError

logger = logging.getLogger(__name__)


def _bowley_skewness(eps: np.ndarray) -> float:
    """Robust (Bowley quartile) skewness; ~0 for any symmetric distribution."""
    q1, q2, q3 = np.percentile(eps, [25.0, 50.0, 75.0])
    iqr = q3 - q1
    if iqr <= 1e-12:
        return 0.0
    return float((q3 + q1 - 2.0 * q2) / iqr)


def _pmm2_optimal_weights(
    eps_white: np.ndarray, robust_skew_tol: float
) -> tuple[float, float, float]:
    """Compute Kunchenko PMM2 optimal weights ``(k_1, k_2, kappa_2)``.

    The quadratic weight is gated on a robust skewness estimate: if the
    whitened residuals are robustly symmetric (``|Bowley| < tol``), the
    optimal ``k_2`` is zero and PMM2 reduces to OLS (``g_2 = 1``), as theory
    requires for symmetric errors.
    """
    eps = eps_white.flatten()
    kappa_2 = float(np.var(eps, ddof=1))
    if kappa_2 <= 0.0:
        return 1.0, 0.0, 0.0
    if abs(_bowley_skewness(eps)) < robust_skew_tol:
        return 1.0, 0.0, kappa_2  # robustly symmetric -> collapse to OLS/GLS
    centred = eps - eps.mean()
    kappa_3 = float(np.mean(centred ** 3))
    kappa_4 = float(np.mean(centred ** 4)) - 3.0 * kappa_2 ** 2
    denom = kappa_4 + 2.0 * kappa_2 ** 2
    if abs(denom) < 1e-12:
        return 1.0, 0.0, kappa_2
    return 1.0, float(-kappa_3 / denom), kappa_2


def fit(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    max_iter: int = 25,
    newton_iter: int = 15,
    tol: float = 1e-6,
    robust_skew_tol: float = 0.04,
    eigenvalue_floor: float = 1e-6,
    regularisation: float = 1e-12,
    raise_on_divergence: bool = False,
) -> BaselineResult:
    """Fit the channel matrix via matrix-PMM2 (Gauss--Newton on the PMM2 score).

    Parameters
    ----------
    Y, S : np.ndarray
        Received ``(k, n)`` and training ``(p, n)`` matrices.
    max_iter : int, default 25
        Maximum number of outer alternating iterations (Gamma update + score
        solve).
    newton_iter : int, default 15
        Maximum inner Gauss--Newton iterations per outer step.
    tol : float, default 1e-6
        Relative Frobenius convergence tolerance.
    robust_skew_tol : float, default 0.04
        Bowley-skewness threshold below which the residuals are treated as
        symmetric and ``k_2`` is set to zero (collapse to OLS).
    eigenvalue_floor : float, default 1e-6
        Floor on eigenvalues of ``Gamma_hat`` during whitening.
    regularisation : float, default 1e-12
        Tikhonov ridge on the ``(kp x kp)`` Gauss--Newton matrix.
    raise_on_divergence : bool, default False

    Returns
    -------
    BaselineResult
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    k, n = Y.shape
    p = S.shape[0]

    H = _ols_fit(Y, S).H
    A = S.T  # (n, p); white-time scenario Sigma = I_n so S_inv_sqrt = I.

    history_frob: list[float] = []
    converged = False
    iter_idx = 0

    for iter_idx in range(1, max_iter + 1):
        H_prev = H.copy()

        # E-step-like Gamma update from current residuals.
        R = Y - H @ S
        Gamma = (R @ R.T) / n + eigenvalue_floor * np.eye(k)
        G_inv_sqrt = matrix_inverse_sqrt(Gamma, eps=eigenvalue_floor)

        # Whitened residuals and (robustly-gated) optimal PMM2 weights.
        R_white = G_inv_sqrt @ R
        k1, k2, kappa2 = _pmm2_optimal_weights(R_white, robust_skew_tol)

        # Gauss--Newton on  B^T psi_2(r) = 0,  r = vec(G^{-1/2}(Y - H S)).
        B = np.kron(A, G_inv_sqrt)                      # (k n, k p)
        y_tilde = (G_inv_sqrt @ Y).flatten(order="F")   # (k n,)
        h = H.flatten(order="F")
        ridge = regularisation * np.eye(k * p)
        steps = newton_iter if k2 != 0.0 else 1
        for _ in range(steps):
            r = y_tilde - B @ h
            psi = k1 * r + k2 * (r ** 2 - kappa2)
            psip = k1 + 2.0 * k2 * r
            jac = (B * psip[:, None]).T @ B + ridge
            try:
                delta = np.linalg.solve(jac, B.T @ psi)
            except np.linalg.LinAlgError:
                logger.warning("matrix-PMM2 Gauss-Newton solve singular; lstsq fallback.")
                delta, *_ = np.linalg.lstsq(jac, B.T @ psi, rcond=None)
            h = h + delta
            if np.linalg.norm(delta) < tol * max(np.linalg.norm(h), 1e-12):
                break

        H = h.reshape((k, p), order="F")
        delta_frob = float(np.linalg.norm(H - H_prev, ord="fro"))
        history_frob.append(delta_frob)
        if delta_frob / max(np.linalg.norm(H_prev, ord="fro"), 1e-12) < tol:
            converged = True
            break

    if not converged and raise_on_divergence:
        raise EMDivergence(
            f"matrix-PMM2 did not converge in {max_iter} iterations; "
            f"last delta={history_frob[-1] if history_frob else float('nan'):.3e}."
        )
    if not np.all(np.isfinite(H)):
        raise NumericalInstabilityError("matrix-PMM2 produced NaN/Inf in H.")

    return BaselineResult(
        H=H,
        residuals=Y - H @ S,
        converged=converged,
        n_iter=iter_idx,
        log_history={"frob_delta": history_frob, "Gamma": Gamma},
    )

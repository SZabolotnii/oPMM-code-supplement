"""Row-independent Huber IRLS baseline for MIMO channel identification.

Each row of H is estimated independently via Huber-score IRLS with a
MAD scale estimate, deliberately omitting the Gamma-Kronecker whitening of
oPMM_alpha. Isolates the per-residual robust gain from the whitening gain:

    MSE(huber_irls)     < MSE(OLS)          on heavy tails
    MSE(whitened_huber) < MSE(huber_irls)   when row correlation is non-trivial

This establishes the lower bound "any robust method beats OLS" and
the baseline "whitening gain = whitened_huber - huber_irls gap".

Satisfies Requirement R9 (robust-estimator competitor for ARE table).
Task: Section revision.B1 (reviewer experiment 1).
"""

from __future__ import annotations

import logging

import numpy as np

from mv_patp.baseline._types import BaselineResult
from mv_patp.baseline.ols import fit as _ols_fit
from mv_patp.utils.errors import EMDivergence, NumericalInstabilityError

logger = logging.getLogger(__name__)

_MAD_SCALE = 0.6745  # median(|N(0,1)|) normalisation constant


def fit(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    c_efficiency: float = 1.345,
    max_iter: int = 30,
    tol: float = 1e-6,
    raise_on_divergence: bool = False,
) -> BaselineResult:
    """Estimate the channel matrix H row-by-row via Huber IRLS (no Gamma whitening).

    Parameters
    ----------
    Y : np.ndarray
        Received signal of shape ``(k, n)``.
    S : np.ndarray
        Known training matrix of shape ``(p, n)``, full row rank.
    c_efficiency : float, default 1.345
        Huber threshold in units of the MAD scale estimate.
        1.345 gives 95% Gaussian asymptotic efficiency.
    max_iter : int, default 30
        Maximum IRLS iterations per row.
    tol : float, default 1e-6
        Relative Frobenius convergence tolerance per row.
    raise_on_divergence : bool, default False
        If True, raise :class:`EMDivergence` if any row fails to converge.

    Returns
    -------
    BaselineResult
        ``H`` of shape ``(k, p)`` plus convergence diagnostics.

    Raises
    ------
    NumericalInstabilityError
        If the solution contains NaN/Inf.
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    k, n = Y.shape
    p = S.shape[0]

    H = _ols_fit(Y, S).H.copy()  # (k, p) — OLS initialisation

    converged_rows = np.zeros(k, dtype=bool)
    n_iters = np.zeros(k, dtype=int)

    for i in range(k):
        h_i = H[i, :].copy()  # (p,)
        for it in range(1, max_iter + 1):
            r_i = Y[i, :] - h_i @ S  # (n,)
            sigma_hat = np.median(np.abs(r_i)) / _MAD_SCALE
            if sigma_hat < 1e-14:
                # Residuals are essentially zero — already converged.
                converged_rows[i] = True
                n_iters[i] = it
                break
            c = c_efficiency * sigma_hat
            w_i = np.minimum(c / np.maximum(np.abs(r_i), 1e-14), 1.0)  # (n,)
            # Weighted normal equations: (S diag(w) S^T + reg I) h = S diag(w) Y[i,:]
            Sw = S * w_i[None, :]  # (p, n)
            gram = Sw @ S.T + 1e-10 * np.eye(p)
            rhs = Sw @ Y[i, :]
            h_new = np.linalg.solve(gram, rhs)

            delta = np.linalg.norm(h_new - h_i)
            h_ref = max(float(np.linalg.norm(h_i)), 1e-14)
            h_i = h_new
            n_iters[i] = it
            if delta / h_ref < tol:
                converged_rows[i] = True
                break

        H[i, :] = h_i

    all_converged = bool(np.all(converged_rows))
    if not all_converged and raise_on_divergence:
        bad = int((~converged_rows).sum())
        raise EMDivergence(
            f"huber_irls: {bad} of {k} rows did not converge in {max_iter} iterations."
        )

    if not np.all(np.isfinite(H)):
        raise NumericalInstabilityError("huber_irls: NaN/Inf in H.")

    return BaselineResult(
        H=H,
        residuals=Y - H @ S,
        converged=all_converged,
        n_iter=int(n_iters.max()),
        log_history={"n_iters_per_row": n_iters.tolist()},
    )

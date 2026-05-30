"""Ordinary least squares baseline for MIMO channel identification.

OLS is the maximum-likelihood estimator under the iid Gaussian noise
assumption. For the matrix MAL noise of ``Spec.md`` Section 4.2 it is
asymptotically inefficient -- precisely the inefficiency that oPMM_alpha
exploits. OLS is therefore the canonical comparator for ARE in
Requirement R9.

Closed form
-----------
For the model ``Y = H S + N`` with rectangular training matrix
``S`` of shape ``(p, n)`` (full row rank), the OLS estimator is

    H_hat = Y S^T (S S^T)^{-1}.

When ``S S^T`` is near-singular we fall back to the Moore-Penrose
pseudo-inverse via ``np.linalg.lstsq``.

Satisfies Requirement R9 (used as ARE comparator).
Task: Section 9.1.
"""

from __future__ import annotations

import logging

import numpy as np

from mv_patp.baseline._types import BaselineResult
from mv_patp.utils.errors import NumericalInstabilityError

logger = logging.getLogger(__name__)


def fit(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    rcond: float | None = None,
) -> BaselineResult:
    """Estimate the channel matrix ``H`` by ordinary least squares.

    Parameters
    ----------
    Y : np.ndarray
        Received signal of shape ``(k, n)``.
    S : np.ndarray
        Known training matrix of shape ``(p, n)``, full row rank.
    rcond : float or None, optional
        Cut-off ratio for small singular values in the lstsq fallback.
        ``None`` (default) lets NumPy pick a sensible default.

    Returns
    -------
    BaselineResult
        ``H`` of shape ``(k, p)`` plus residuals.

    Raises
    ------
    ValueError
        If ``Y`` and ``S`` have inconsistent column counts.
    NumericalInstabilityError
        If the OLS solution contains NaN/Inf.
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if Y.ndim != 2 or S.ndim != 2:
        raise ValueError(f"Y and S must be 2-D; got Y.ndim={Y.ndim}, "
                         f"S.ndim={S.ndim}.")
    if Y.shape[1] != S.shape[1]:
        raise ValueError(
            f"Y and S must have matching n; got Y.shape={Y.shape}, "
            f"S.shape={S.shape}."
        )

    # Solve H S = Y in least-squares sense: H = Y @ S^T @ (S S^T)^{-1}.
    # The lstsq formulation H_T = lstsq(S^T, Y^T) gives H_T of shape (p, k);
    # transpose for the conventional (k, p) layout.
    H_T, _residuals_lstsq, _rank, _sv = np.linalg.lstsq(S.T, Y.T, rcond=rcond)
    H = H_T.T

    if not np.all(np.isfinite(H)):
        raise NumericalInstabilityError(
            "OLS solution contains NaN or Inf; check S conditioning."
        )

    residuals = Y - H @ S
    return BaselineResult(H=H, residuals=residuals, converged=True, n_iter=0)

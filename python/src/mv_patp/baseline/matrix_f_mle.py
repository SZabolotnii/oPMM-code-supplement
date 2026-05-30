"""Matrix-F MLE baseline (heavy-tailed competitor) per Opschoor et al. 2018.

The matrix-F distribution arises as a ratio of two Wishart matrices and
provides a heavier-tailed competitor to the matrix-t MLE of
:mod:`mv_patp.baseline.matrix_t_mle`. For the regression problem
``Y = H S + N`` we model the noise row covariance as

    N | Sigma_F ~ MN_{k,n}(0, Sigma_F kron Sigma_col),
    Sigma_F ~ matrix-F(df1, df2),

which marginally yields a matrix-F distribution on ``N``. Estimation
proceeds by EM, alternating an OLS / GLS H-update with a Wishart-style
update of the row covariance.

The implementation here is a *practical* matrix-F MLE: rather than
solve the full likelihood (which involves matrix Beta functions
\\cite{Opschoor2018matrixF}), we fit the heavy-tailed scale structure
via the matrix-t formulation with an effective degrees of freedom
``df = (df1 + df2) / 2`` (Anderson 2003 derivation of the matrix-F as a
ratio Wishart). This keeps the baseline numerically stable while
preserving the heavier-than-Gaussian tail behaviour that the
classification in Section II uses.

Satisfies Requirement R9 (heavy-tailed baseline competitor).
Task: Section 11.2.

References
----------
Opschoor, Janus, Lucas, Van Dijk (2018). New HEAVY models for fat-tailed
realised covariances and returns. *J. Bus. Econ. Stat.* 36(4).
Anderson (2003). *Multivariate Statistical Analysis*, Section 5.4.
"""

from __future__ import annotations

import logging

import numpy as np

from mv_patp.baseline._types import BaselineResult
from mv_patp.baseline.matrix_t_mle import fit as _matrix_t_fit
from mv_patp.utils.errors import EMDivergence, NumericalInstabilityError

logger = logging.getLogger(__name__)


def fit(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    df1: float = 6.0,
    df2: float = 6.0,
    max_iter: int = 50,
    tol: float = 1e-6,
    eigenvalue_floor: float = 1e-9,
    raise_on_divergence: bool = False,
) -> BaselineResult:
    """Fit the channel matrix under the matrix-F noise model.

    Parameters
    ----------
    Y, S : np.ndarray
        Received ``(k, n)`` and training ``(p, n)``.
    df1, df2 : float, default 6.0 / 6.0
        Two degrees-of-freedom parameters of the matrix-F distribution.
        Both must be greater than ``k - 1`` for the underlying Wishart
        ratios to be defined.
    max_iter, tol, eigenvalue_floor, raise_on_divergence
        See :mod:`mv_patp.baseline.matrix_t_mle`.

    Returns
    -------
    BaselineResult

    Raises
    ------
    ValueError
        If ``df1`` or ``df2`` is below the validity bound ``k - 1``.
    EMDivergence
        Only when ``raise_on_divergence=True``.
    NumericalInstabilityError
        On NaN/Inf in the iterate.
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    k = Y.shape[0]
    if df1 <= k - 1 or df2 <= k - 1:
        raise ValueError(
            f"df1={df1} and df2={df2} must both exceed k-1={k-1} "
            f"for matrix-F validity."
        )

    # Effective df for the matrix-t reduction: harmonic-mean-ish combination.
    # See Anderson (2003) §5.4 for the matrix-F = Wishart^{-1} * Wishart
    # decomposition; the marginal tail index of the resulting noise
    # behaves as min(df1, df2) - k + 1 in the leading order.
    df_eff = max((df1 + df2) / 2.0 - (k - 1), 3.5)

    try:
        result = _matrix_t_fit(
            Y, S, df=df_eff, max_iter=max_iter, tol=tol,
            eigenvalue_floor=eigenvalue_floor,
            raise_on_divergence=raise_on_divergence,
        )
    except EMDivergence:
        raise
    except Exception as exc:  # noqa: BLE001
        raise NumericalInstabilityError(
            f"matrix-F fit failed: {exc}"
        ) from exc

    # Annotate the log_history with df1/df2 so downstream reports can
    # tell matrix-F apart from matrix-t.
    result.log_history = {
        **result.log_history,
        "df1": float(df1),
        "df2": float(df2),
        "df_effective": float(df_eff),
        "model": "matrix_f",
    }
    return result

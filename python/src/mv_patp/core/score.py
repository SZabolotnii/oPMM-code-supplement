"""oPMM_α score function and optimal weight computation.

Implements the sign-preserving fractional score function ψ_α(ε) and the
closed-form optimal weights k_1*, k_2* derived in Spec.md §4.3.

References
----------
Spec.md §4.3 — oPMM_α score function and optimal weights.
theory/opmm_derivation.py — numerical verification of weight formulas.
"""

from __future__ import annotations

import logging

import numpy as np

from mv_patp.utils.errors import AlphaOutOfRangeError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALPHA_MIN: float = 0.3
_ALPHA_MAX: float = 3.0
_EPS_MIN: float = 1e-10  # floor on |ε| before computing |ε|^α (Spec §5.3c)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_optimal_weights(C: dict, alpha: float) -> tuple[float, float]:
    """Compute optimal weights k_1*, k_2* per Spec.md §4.3.

    The unnormalized weights are:

    .. math::

        k_1 \\propto \\sigma_\\alpha^2 - \\alpha \\nu_\\alpha c_{1\\alpha}

        k_2 \\propto \\alpha \\nu_\\alpha \\kappa_2 - c_{1\\alpha}

    Normalization enforces :math:`E[\\psi'_\\alpha] = 1`, i.e.
    :math:`k_1^* + k_2^* \\cdot \\alpha \\nu_\\alpha = 1`.

    Parameters
    ----------
    C : dict
        Cumulant dictionary as returned by
        :func:`mv_patp.utils.cumulants.empirical_cumulants`.
        Required keys: ``kappa_2``, ``c_1_alpha``, ``nu_alpha``,
        ``sigma_alpha_sq``, ``mu_alpha``.
    alpha : float
        Fractional order α ∈ [0.3, 3.0].

    Returns
    -------
    tuple[float, float]
        ``(k1_star, k2_star)`` — normalized optimal weights.

    Raises
    ------
    AlphaOutOfRangeError
        If α < 0.3 or α > 3.0.

    Warns
    -----
    Logs a warning (via :mod:`logging`) when α = 1 because the fractional
    basis collapses to the linear basis (k_2* → 0, degenerate case).

    Notes
    -----
    At α = 1 the normalization scale k_1 + k_2·α·ν_α equals κ_2 − c_{1,1}
    which may be very small, causing k_2* → 0 and k_1* → 1 (OLS limit).
    A warning is emitted to alert the caller.

    Satisfies Requirements R3.1, R3.3, R3.4.

    Examples
    --------
    >>> import numpy as np
    >>> from mv_patp.utils.cumulants import empirical_cumulants
    >>> rng = np.random.default_rng(0)
    >>> eps = rng.laplace(0, 1, 10_000)
    >>> C = empirical_cumulants(eps, alpha=3.0)
    >>> k1, k2 = compute_optimal_weights(C, alpha=3.0)
    >>> abs(k1 + k2 * 3.0 * C["nu_alpha"] - 1.0) < 1e-10
    True
    """
    _check_alpha_range(alpha)

    sigma_alpha_sq: float = C["sigma_alpha_sq"]
    nu_alpha: float = C["nu_alpha"]
    c_1_alpha: float = C["c_1_alpha"]
    kappa_2: float = C["kappa_2"]

    # Unnormalized weights (Spec.md §4.3)
    k1 = sigma_alpha_sq - alpha * nu_alpha * c_1_alpha
    k2 = alpha * nu_alpha * kappa_2 - c_1_alpha

    # Normalization: k1 + k2 * alpha * nu_alpha = 1  (E[ψ'] = 1)
    scale = k1 + k2 * alpha * nu_alpha

    if alpha == 1.0:
        logger.warning(
            "α = 1: fractional basis sign(ε)|ε|^1 = ε is collinear with the "
            "linear basis ε. k_2* → 0 (degenerate basis). Score reduces to OLS."
        )

    if abs(scale) < 1e-12:
        # Pathological: return OLS weights as safe fallback
        logger.warning(
            "Normalization scale |k1 + k2·α·ν_α| = %.3e < 1e-12 at α=%.4f. "
            "Returning OLS fallback weights (k1*=1, k2*=0).",
            abs(scale),
            alpha,
        )
        return 1.0, 0.0

    k1_star = k1 / scale
    k2_star = k2 / scale

    return float(k1_star), float(k2_star)


def score_psi_alpha(
    eps: np.ndarray,
    alpha: float,
    C: dict,
) -> np.ndarray:
    """Compute ψ_α(ε) = k_1*·ε + k_2*·[sign(ε)|ε|^α − μ_α].

    Implements the sign-preserving fractional score function from Spec.md §4.3:

    .. math::

        \\psi_\\alpha(\\varepsilon) =
        k_1^* \\cdot \\varepsilon
        + k_2^* \\cdot \\bigl[\\mathrm{sign}(\\varepsilon)
        |\\varepsilon|^\\alpha - \\mu_\\alpha\\bigr]

    where :math:`k_1^*, k_2^*` are the optimal weights from
    :func:`compute_optimal_weights` and :math:`\\mu_\\alpha =
    E[\\mathrm{sign}(\\varepsilon)|\\varepsilon|^\\alpha]` is the centering
    term (≈ 0 for symmetric distributions).

    A floor of ``eps_min = 1e-10`` is applied to :math:`|\\varepsilon|`
    before computing :math:`|\\varepsilon|^\\alpha` to avoid ``0^α``
    numerical issues when α < 1 (Spec §5.3c).

    Parameters
    ----------
    eps : np.ndarray
        Array of whitened residuals, any shape.  Must be finite.
    alpha : float
        Fractional order α ∈ [0.3, 3.0].
    C : dict
        Cumulant dictionary as returned by
        :func:`mv_patp.utils.cumulants.empirical_cumulants`.
        Required keys: ``kappa_2``, ``c_1_alpha``, ``nu_alpha``,
        ``sigma_alpha_sq``, ``mu_alpha``.

    Returns
    -------
    np.ndarray
        Score values ψ_α(ε), same shape as ``eps``.

    Raises
    ------
    AlphaOutOfRangeError
        If α < 0.3 or α > 3.0.

    Notes
    -----
    The ``eps_min`` floor is applied only to the absolute value used in the
    power computation; the sign is preserved from the original ``eps``.
    This ensures ψ_α(0) ≈ 0 for symmetric C (μ_α = 0).

    Satisfies Requirements R3.1, R3.2, R3.3, R3.4, R3.5.

    Examples
    --------
    >>> import numpy as np
    >>> from mv_patp.utils.cumulants import empirical_cumulants
    >>> rng = np.random.default_rng(42)
    >>> eps_data = rng.laplace(0, 1, 5_000)
    >>> C = empirical_cumulants(eps_data, alpha=2.0)
    >>> psi = score_psi_alpha(np.array([0.0, 1.0, -1.0]), alpha=2.0, C=C)
    >>> psi.shape
    (3,)
    """
    _check_alpha_range(alpha)

    eps = np.asarray(eps, dtype=np.float64)
    mu_alpha: float = C["mu_alpha"]

    k1_star, k2_star = compute_optimal_weights(C, alpha)

    # Apply eps_min floor to |ε| before raising to power α (Spec §5.3c)
    abs_eps_safe = np.maximum(np.abs(eps), _EPS_MIN)

    # sign-preserving fractional basis: sign(ε) · |ε|^α
    phi_alpha = np.sign(eps) * (abs_eps_safe ** alpha)

    # ψ_α(ε) = k_1* · ε + k_2* · [φ_α(ε) − μ_α]
    psi = k1_star * eps + k2_star * (phi_alpha - mu_alpha)

    return psi


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_alpha_range(alpha: float) -> None:
    """Raise AlphaOutOfRangeError if α is outside [0.3, 3.0].

    Parameters
    ----------
    alpha : float
        Fractional order to validate.

    Raises
    ------
    AlphaOutOfRangeError
        If α < 0.3 or α > 3.0.
    """
    if alpha < _ALPHA_MIN or alpha > _ALPHA_MAX:
        raise AlphaOutOfRangeError(
            f"α = {alpha} is outside the valid range [{_ALPHA_MIN}, {_ALPHA_MAX}]. "
            "oPMM_α is numerically unstable below 0.3 and the basis is "
            "degenerate above 3.0 for the supported distributions."
        )

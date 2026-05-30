"""Empirical cumulant estimation for whitened residuals.

Provides :func:`empirical_cumulants` and :func:`closed_form_g_alpha` used
throughout the oPMM_α pipeline.

References
----------
Spec.md §4.4 — closed-form g(α; C) and cumulant vector definition.
Spec.md §4.3 — optimal weights k₁*, k₂* and fractional moment definitions.
"""

from __future__ import annotations

import warnings

import numpy as np


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def empirical_cumulants(eps_white: np.ndarray, alpha: float) -> dict:
    """Compute empirical cumulants of whitened residuals.

    Returns the cumulant vector **C** = (κ₂, κ₃, γ₄, γ₆, c_{1α}, ν_α,
    σ_α², μ_α) required by the oPMM_α score function and the closed-form
    efficiency g(α; C) (Spec.md §4.4).

    Fisher k-statistic (unbiased) adjustments are used for κ₂, κ₃, κ₄.
    For κ₆ and γ₆ the plug-in estimator is used (no bias correction needed
    for our purposes).  When n < 30 the function falls back to plug-in
    moments for all cumulants and emits a :class:`UserWarning`.

    Parameters
    ----------
    eps_white : np.ndarray
        1-D array of whitened residuals, shape (n,).  Must be finite.
    alpha : float
        Fractional order α ∈ (0, ∞).  Typically α ∈ [0.3, 3.0] for oPMM_α.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``kappa_2`` : float
            Second cumulant (variance), Fisher unbiased: ``np.var(eps, ddof=1)``.
        ``kappa_3`` : float
            Third cumulant, Fisher unbiased.
        ``gamma_4`` : float
            Excess kurtosis (fourth standardised cumulant): κ₄ / κ₂².
        ``gamma_6`` : float
            Sixth standardised cumulant: κ₆ / κ₂³ (plug-in estimator).
        ``c_1_alpha`` : float
            E[|ε|^{α+1}] — empirical absolute moment.
        ``nu_alpha`` : float
            E[|ε|^{α-1}] — empirical absolute moment.
        ``sigma_alpha_sq`` : float
            E[|ε|^{2α}] − μ_α² — variance of sign(ε)|ε|^α.
        ``mu_alpha`` : float
            E[sign(ε)|ε|^α] — signed fractional moment (≈ 0 for symmetric).

    Raises
    ------
    ValueError
        If ``eps_white`` is not 1-D or contains non-finite values.

    Notes
    -----
    **Small-sample fallback (n < 30):** Fisher k-statistic corrections for
    κ₃ and κ₄ require n ≥ 3 and n ≥ 4 respectively; for n < 30 the
    corrections can be numerically unstable, so plug-in central moments are
    used instead and a warning is emitted.

    **Spec.md §4.4 reference:**

    .. math::

        g(\\alpha; \\mathbf{C}) =
        \\frac{\\kappa_2 \\sigma_\\alpha^2 - c_{1\\alpha}^2}
        {\\kappa_2 (\\sigma_\\alpha^2 - 2\\alpha\\nu_\\alpha c_{1\\alpha}
        + \\alpha^2 \\nu_\\alpha^2 \\kappa_2)}

    Satisfies Requirements R3.5, R4.1.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> eps = rng.standard_normal(1000)
    >>> C = empirical_cumulants(eps, alpha=2.0)
    >>> abs(C["kappa_2"] - 1.0) < 0.1
    True
    """
    eps_white = np.asarray(eps_white, dtype=np.float64)
    if eps_white.ndim != 1:
        raise ValueError(
            f"eps_white must be 1-D, got shape {eps_white.shape}."
        )
    if not np.isfinite(eps_white).all():
        raise ValueError("eps_white contains non-finite values (NaN or Inf).")

    n = eps_white.size
    small_sample = n < 30

    if small_sample:
        warnings.warn(
            f"n={n} < 30: falling back to plug-in (biased) moment estimators "
            "for all cumulants.  Fisher k-statistic corrections are unreliable "
            "at this sample size.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Central moments (plug-in, used as building blocks)
    # ------------------------------------------------------------------
    mean = np.mean(eps_white)
    centered = eps_white - mean

    m2 = np.mean(centered ** 2)          # biased variance
    m3 = np.mean(centered ** 3)
    m4 = np.mean(centered ** 4)
    m6 = np.mean(centered ** 6)

    # ------------------------------------------------------------------
    # Fisher k-statistic (unbiased) cumulants
    # ------------------------------------------------------------------
    if small_sample:
        # Plug-in fallback
        kappa_2 = m2
        kappa_3 = m3
        kappa_4 = m4 - 3.0 * m2 ** 2
    else:
        # κ₂ = sample variance with ddof=1
        kappa_2 = np.var(eps_white, ddof=1)

        # κ₃ = n / ((n-1)(n-2)) * Σ(x - x̄)³  — Fisher unbiased third cumulant
        kappa_3 = (n / ((n - 1) * (n - 2))) * np.sum(centered ** 3)

        # κ₄ = ((n+1)*m4 - 3*(n-1)*m2²) * n*(n+1) / ((n-1)*(n-2)*(n-3))
        #      — Fisher unbiased fourth cumulant
        kappa_4 = (
            ((n + 1) * m4 - 3.0 * (n - 1) * m2 ** 2)
            * n * (n + 1)
            / ((n - 1) * (n - 2) * (n - 3))
        )

    # ------------------------------------------------------------------
    # Standardised cumulants
    # ------------------------------------------------------------------
    # γ₄ = κ₄ / κ₂²  (excess kurtosis)
    if small_sample:
        gamma_4 = kappa_4 / (kappa_2 ** 2) if kappa_2 > 0.0 else 0.0
    else:
        gamma_4 = kappa_4 / (kappa_2 ** 2) if kappa_2 > 0.0 else 0.0

    # γ₆ = κ₆ / κ₂³  — plug-in estimator (sixth standardised cumulant)
    # Using the cumulant-moment relation:
    #   κ₆ = m₆ − 15 m₄ m₂ + 30 m₂³ − 10 m₃² + ... (for centred variable)
    # For the standardised form we use:
    #   γ₆ = m₆/m₂³ − 15 m₄/m₂² + 30 − 15
    # (derived from the sixth cumulant formula for centred distributions)
    if m2 > 0.0:
        gamma_6 = m6 / (m2 ** 3) - 15.0 * m4 / (m2 ** 2) + 30.0 - 15.0
    else:
        gamma_6 = 0.0

    # ------------------------------------------------------------------
    # Fractional moments
    # ------------------------------------------------------------------
    abs_eps = np.abs(eps_white)

    # c_{1α} = E[|ε|^{α+1}]
    c_1_alpha = np.mean(abs_eps ** (alpha + 1.0))

    # ν_α = E[|ε|^{α-1}]
    nu_alpha = np.mean(abs_eps ** (alpha - 1.0))

    # μ_α = E[sign(ε)|ε|^α]  (≈ 0 for symmetric distributions)
    mu_alpha = np.mean(np.sign(eps_white) * abs_eps ** alpha)

    # σ_α² = E[|ε|^{2α}] − μ_α²
    sigma_alpha_sq = np.mean(abs_eps ** (2.0 * alpha)) - mu_alpha ** 2

    return {
        "kappa_2": float(kappa_2),
        "kappa_3": float(kappa_3),
        "gamma_4": float(gamma_4),
        "gamma_6": float(gamma_6),
        "c_1_alpha": float(c_1_alpha),
        "nu_alpha": float(nu_alpha),
        "sigma_alpha_sq": float(sigma_alpha_sq),
        "mu_alpha": float(mu_alpha),
    }


def closed_form_g_alpha(alpha: float, C: dict) -> float:
    """Evaluate the closed-form efficiency g(α; C) per Spec.md §4.4.

    Implements:

    .. math::

        g(\\alpha; \\mathbf{C}) =
        \\frac{\\kappa_2 \\sigma_\\alpha^2 - c_{1\\alpha}^2}
        {\\kappa_2 (\\sigma_\\alpha^2 - 2\\alpha\\nu_\\alpha c_{1\\alpha}
        + \\alpha^2 \\nu_\\alpha^2 \\kappa_2)}

    where **C** = (κ₂, c_{1α}, ν_α, σ_α², μ_α) is the cumulant vector of
    whitened residuals.

    g(α; C) < 1 indicates that oPMM_α is more efficient than OLS.
    g(α; C) ≥ 1 means oPMM_α offers no improvement; a warning is emitted.

    Parameters
    ----------
    alpha : float
        Fractional order α.
    C : dict
        Cumulant dictionary as returned by :func:`empirical_cumulants`.
        Required keys: ``kappa_2``, ``c_1_alpha``, ``nu_alpha``,
        ``sigma_alpha_sq``.

    Returns
    -------
    float
        g(α; C).  Returns ``np.nan`` if the denominator is pathological
        (|denom| < 1e-15).

    Warns
    -----
    UserWarning
        When g(α; C) ≥ 1 (oPMM_α offers no improvement over OLS).

    References
    ----------
    Spec.md §4.4 — Theorem 1 (oPMM efficiency).

    Satisfies Requirements R4.1, R4.4.
    """
    kappa_2: float = C["kappa_2"]
    c_1_alpha: float = C["c_1_alpha"]
    nu_alpha: float = C["nu_alpha"]
    sigma_alpha_sq: float = C["sigma_alpha_sq"]

    numerator = kappa_2 * sigma_alpha_sq - c_1_alpha ** 2
    inner = (
        sigma_alpha_sq
        - 2.0 * alpha * nu_alpha * c_1_alpha
        + alpha ** 2 * nu_alpha ** 2 * kappa_2
    )
    denominator = kappa_2 * inner

    if abs(denominator) < 1e-15:
        return float(np.nan)

    g = numerator / denominator

    if g >= 1.0:
        warnings.warn(
            f"g(α={alpha}; C) = {g:.6f} ≥ 1.0: oPMM_α offers no improvement "
            "over OLS for this α and cumulant vector.",
            UserWarning,
            stacklevel=2,
        )

    return float(g)

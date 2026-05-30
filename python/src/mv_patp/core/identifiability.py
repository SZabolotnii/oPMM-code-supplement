"""Identifiability constraint projection for the EM-PATP estimator.

Enforces the (Γ, Σ) scaling constraint that prevents the EM iteration from
collapsing or oscillating due to the inherent scaling ambiguity between the
row covariance Γ and the column covariance Σ.

Two scenarios are supported:

- ``"white"`` (default): fixes Σ = I_n and leaves Γ̂ unchanged.  This is the
  primary scenario for MIMO RF channels where noise is white in time.
- ``"ar1"``: enforces tr(Γ̂) = k by rescaling Γ̂ and applying the reciprocal
  scale to Σ̂.  Used when temporal covariance follows an AR(1) or
  exponential-decay structure.

Satisfies Requirements R2.1, R2.2, R2.3.

References
----------
Spec.md §III assumption — identifiability constraint definition.
design.md DD-2 — rationale for compile-time identifiability choice.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def project_identifiability(
    Gamma_hat: np.ndarray,
    Sigma_hat: np.ndarray,
    scenario: str = "white",
) -> tuple[np.ndarray, np.ndarray]:
    """Project (Γ̂, Σ̂) onto the identifiability constraint surface.

    Enforces the scaling constraint that resolves the (Γ, Σ) ambiguity in
    the matrix MAL model.  The two supported scenarios correspond to the
    primary (white noise) and secondary (AR(1) temporal covariance) settings
    described in Spec.md §III.

    Scenario ``"white"`` (Requirement R2.1):
        Returns ``(Gamma_hat, I_n)`` unchanged.  Σ is fixed to the identity
        matrix of size n × n, where n = ``Sigma_hat.shape[0]``.

    Scenario ``"ar1"`` (Requirements R2.2, R2.3):
        Rescales so that tr(Γ̂_new) = k, where k = ``Gamma_hat.shape[0]``,
        with a reciprocal scaling applied to Σ̂::

            scale         = tr(Gamma_hat) / k
            Gamma_hat_new = Gamma_hat / scale
            Sigma_hat_new = Sigma_hat * scale

        If ``|tr(Gamma_hat) − k| > 1e-6`` a warning is logged via the
        module-level logger before re-projection.

    Parameters
    ----------
    Gamma_hat : np.ndarray
        Estimated row covariance matrix of shape ``(k, k)``, symmetric
        positive semi-definite.
    Sigma_hat : np.ndarray
        Estimated column covariance matrix of shape ``(n, n)``, symmetric
        positive semi-definite.
    scenario : str, optional
        Identifiability scenario.  One of ``{"white", "ar1"}``.
        Default is ``"white"``.

    Returns
    -------
    Gamma_hat_new : np.ndarray
        Projected row covariance of shape ``(k, k)``.
    Sigma_hat_new : np.ndarray
        Projected column covariance of shape ``(n, n)``.

    Raises
    ------
    ValueError
        If ``scenario`` is not one of the recognised strings.

    References
    ----------
    Spec.md §III — identifiability constraint.
    Requirements R2.1, R2.2, R2.3.
    """
    Gamma_hat = np.asarray(Gamma_hat, dtype=np.float64)
    Sigma_hat = np.asarray(Sigma_hat, dtype=np.float64)

    if scenario == "white":
        # R2.1: fix Σ = I_n; Γ̂ is returned unchanged.
        n = Sigma_hat.shape[0]
        Sigma_new = np.eye(n, dtype=np.float64)
        return Gamma_hat.copy(), Sigma_new

    if scenario == "ar1":
        # R2.2 / R2.3: enforce tr(Γ̂) = k via proportional rescaling.
        k = Gamma_hat.shape[0]
        trace_gamma = np.trace(Gamma_hat)

        # R2.3: warn if the current iterate already deviates from the constraint.
        if abs(trace_gamma - k) > 1e-6:
            logger.warning(
                "project_identifiability: |tr(Γ̂) − k| = %.3e > 1e-6 "
                "(tr=%.6f, k=%d).  Re-projecting.",
                abs(trace_gamma - k),
                trace_gamma,
                k,
            )

        scale = trace_gamma / k
        Gamma_new = Gamma_hat / scale
        Sigma_new = Sigma_hat * scale
        return Gamma_new, Sigma_new

    raise ValueError(
        f"project_identifiability: unknown scenario {scenario!r}. "
        "Expected one of {'white', 'ar1'}."
    )

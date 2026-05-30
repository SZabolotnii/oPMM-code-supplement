"""Matrix sandwich standard errors for the EM-PATP channel estimate.

Secondary CI tool per Spec.md Section 5.5 (Appendix C). For a Z-estimator
defined by the score equation ``sum_i psi(epsilon_i; theta) = 0``, the
asymptotic variance of ``vec(theta_hat)`` is the matrix sandwich

.. math::
    \\widehat{\\mathrm{Var}}\\bigl(\\vec(\\widehat{H})\\bigr)
    \\approx \\widehat{B}^{-1}\\,\\widehat{C}\\,\\widehat{B}^{-1}

with the bread ``B = -E[\\nabla psi]`` and the meat ``C = E[psi psi^T]``.
For the matrix oPMM_alpha score on a vec'ed regression, the bread is the
standard weighted-LS Hessian ``B = (S Sigma^{-1/2})^T kron (Gamma^{-1/2})``
projected onto the IRLS weights at convergence; the meat is the empirical
outer product of per-element scores.

In the symmetric primary regime (``mu_alpha = 0``) the score factorises
as ``psi(eps) = w(eps) * eps`` with the same IRLS weights used in
``solve_weighted_matrix_LS``, so the sandwich reuses those weights
directly.

Satisfies Requirements R8.3, R8.4.
Task: Section 14.1.

References
----------
Spec.md Section 5.5; design.md Section L2; Maronna et al. (2019)
*Robust Statistics: Theory and Methods*, Section 4.3 for the M-estimator
sandwich form.
"""

from __future__ import annotations

import logging

import numpy as np

from mv_patp.core.em_patp import EMPATPResult
from mv_patp.core.score import compute_optimal_weights
from mv_patp.core.whitening import matrix_inverse_sqrt
from mv_patp.utils.cumulants import empirical_cumulants
from mv_patp.utils.errors import NumericalInstabilityError

logger = logging.getLogger(__name__)


def sandwich_se(
    Y: np.ndarray,
    S: np.ndarray,
    fit_result: EMPATPResult,
    *,
    eigenvalue_floor: float = 1e-6,
    regularisation: float = 1e-12,
) -> np.ndarray:
    """Asymptotic standard errors for ``vec(H)`` via the sandwich formula.

    Parameters
    ----------
    Y, S : np.ndarray
        Observation ``(k, n)`` and training matrix ``(p, n)``.
    fit_result : EMPATPResult
        Output of :func:`mv_patp.core.em_patp.em_patp_estimate`.
    eigenvalue_floor : float, default 1e-6
        SPD floor used when forming Gamma_hat^{-1/2} and Sigma_hat^{-1/2}.
    regularisation : float, default 1e-12
        Tikhonov ridge for the bread-matrix solve.

    Returns
    -------
    np.ndarray
        Standard error per element of ``vec(H_hat)``, reshaped to ``H``'s
        ``(k, p)`` for convenience. ``SE_ij`` corresponds to ``H_hat_ij``.

    Raises
    ------
    NumericalInstabilityError
        If the sandwich matrix is non-finite or yields negative diagonal.
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    H_hat = np.asarray(fit_result.H, dtype=np.float64)
    Gamma = np.asarray(fit_result.Gamma, dtype=np.float64)
    Sigma = np.asarray(fit_result.Sigma, dtype=np.float64)
    alpha_p = float(fit_result.alpha_p)

    k, n = Y.shape
    p = S.shape[0]

    G_inv_sqrt = matrix_inverse_sqrt(Gamma, eps=eigenvalue_floor)
    S_inv_sqrt = matrix_inverse_sqrt(Sigma, eps=eigenvalue_floor)

    R = Y - H_hat @ S
    R_white = G_inv_sqrt @ R @ S_inv_sqrt

    # Recompute IRLS weights at the converged residuals.
    eps_white = R_white.flatten()
    C_cumulants = empirical_cumulants(eps_white, alpha_p)
    try:
        k1_star, k2_star = compute_optimal_weights(C_cumulants, alpha_p)
    except Exception:  # noqa: BLE001
        k1_star, k2_star = 1.0, 0.0
    abs_R_safe = np.maximum(np.abs(R_white), 1e-10)
    W = k1_star + k2_star * abs_R_safe ** (alpha_p - 1.0)
    W = np.maximum(W, 1e-6)

    # Build the Kronecker design B = (S Sigma^{-1/2})^T kron Gamma^{-1/2}
    # used in solve_weighted_matrix_LS, then form bread and meat.
    A = (S @ S_inv_sqrt).T                    # (n, p)
    B_kron = np.kron(A, G_inv_sqrt)           # (k*n, k*p)

    d = W.flatten(order="F")                  # (k*n,)
    psi_vec = (W * R_white).flatten(order="F")  # element-wise score values

    # Bread: -E[nabla psi] approximated by sum w_i * basis_i basis_i^T = B^T diag(w) B
    bread = B_kron.T @ (d[:, None] * B_kron)  # (k*p, k*p)
    bread = bread + regularisation * np.eye(bread.shape[0])

    # Meat: empirical outer product of per-element score-weighted basis rows.
    # score_i = psi_i * basis_i (column-major vec).
    score_rows = psi_vec[:, None] * B_kron    # (k*n, k*p)
    meat = score_rows.T @ score_rows           # (k*p, k*p)

    try:
        bread_inv = np.linalg.solve(bread, np.eye(bread.shape[0]))
    except np.linalg.LinAlgError as exc:
        raise NumericalInstabilityError(
            "sandwich_se bread matrix singular; fit may be degenerate."
        ) from exc

    var_vec = bread_inv @ meat @ bread_inv
    diag_var = np.diag(var_vec)
    if np.any(diag_var < -1e-9):
        raise NumericalInstabilityError(
            f"sandwich_se produced negative variance entries "
            f"(min = {diag_var.min():.3e})."
        )
    se_flat = np.sqrt(np.maximum(diag_var, 0.0))
    return se_flat.reshape((k, p), order="F")

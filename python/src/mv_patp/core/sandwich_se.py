"""Matrix sandwich standard errors for the EM-PATP channel estimate.

Secondary CI tool per Spec.md Section 5.5 (Appendix C). For a Z-estimator
defined by the score equation ``sum_i psi(epsilon_i; theta) = 0``, the
asymptotic variance of ``vec(theta_hat)`` is the matrix sandwich

.. math::
    \\widehat{\\mathrm{Var}}\\bigl(\\vec(\\widehat{H})\\bigr)
    \\approx \\widehat{B}^{-1}\\,\\widehat{C}\\,\\widehat{B}^{-1}

with the bread ``B = -E[\\nabla psi]`` and the meat ``C = E[psi psi^T]``.

In the symmetric primary regime (``mu_alpha = 0``) the score factorises as
``psi(e) = w(e) * e`` with the IRLS weight ``w(e) = k1* + k2* |e|^(alpha-1)``
used in ``solve_weighted_matrix_LS``. The bread needs the derivative of the
*whole* score, including the residual-dependence of the weight:

.. math::
    \\psi'(e) = w(e) + w'(e)\\,e
              = k_1^\\ast + \\alpha k_2^\\ast |e|^{\\alpha-1},

so ``B = Kron^T diag(psi'(vec R_white)) Kron`` with
``Kron = (S Sigma^{-1/2})^T kron Gamma^{-1/2}``. Where the IRLS weight floor
``w_min`` is active the implemented score is ``w_min * e`` and its
derivative is ``w_min``; the same floor on ``|e|`` (``eps_min``) is applied
as in the EM-IRLS loop.

R1 correction (DSP-D-26-03654, Rv4-C2b): the submitted version used
``diag(w)`` in place of ``diag(psi')``. That drops the term
``k2* (alpha - 1) |e|^(alpha-1)`` and makes the SE too small by the factor
``E[psi'] / E[w]`` (1 at the Gaussian, 0.27-0.77 on Student-t df 3-8).
The legacy bread is kept behind ``bread="weight"`` only so the comparison
can be reproduced; it must not be used for inference.

The meat is the empirical outer product of the per-element scores
(``meat="element"``, the form in Appendix C), or of the per-column cluster
sums (``meat="cluster"``), which stays valid when entries within a column
are not jointly sign-symmetric.

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

# Same floors as the EM-IRLS loop in core/em_patp.py.
_EPS_MIN: float = 1e-10
_W_MIN: float = 1e-6


def sandwich_se(
    Y: np.ndarray,
    S: np.ndarray,
    fit_result: EMPATPResult,
    *,
    eigenvalue_floor: float = 1e-6,
    regularisation: float = 1e-12,
    bread: str = "psi_prime",
    meat: str = "element",
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
    bread : {"psi_prime", "weight"}, default "psi_prime"
        ``"psi_prime"`` is the correct bread ``diag(psi')``. ``"weight"``
        reproduces the pre-R1 bread ``diag(w)``, which underestimates the
        SE on heavy tails; it exists only for the old-vs-new comparison.
    meat : {"element", "cluster"}, default "element"
        Per-element outer products, or per-column cluster sums.

    Returns
    -------
    np.ndarray
        Standard error per element of ``vec(H_hat)``, reshaped to ``H``'s
        ``(k, p)`` for convenience. ``SE_ij`` corresponds to ``H_hat_ij``.

    Raises
    ------
    ValueError
        If ``bread`` or ``meat`` is not a recognised option.
    NumericalInstabilityError
        If the sandwich matrix is non-finite or yields negative diagonal.
    """
    if bread not in ("psi_prime", "weight"):
        raise ValueError(f"bread must be 'psi_prime' or 'weight', got {bread!r}")
    if meat not in ("element", "cluster"):
        raise ValueError(f"meat must be 'element' or 'cluster', got {meat!r}")

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
    abs_R_safe = np.maximum(np.abs(R_white), _EPS_MIN)
    W_raw = k1_star + k2_star * abs_R_safe ** (alpha_p - 1.0)
    floor_active = W_raw <= _W_MIN
    W = np.where(floor_active, _W_MIN, W_raw)

    if bread == "psi_prime":
        # psi'(e) = k1* + alpha k2* |e|^(alpha-1); w_min where the floor is active.
        dpsi = k1_star + alpha_p * k2_star * abs_R_safe ** (alpha_p - 1.0)
        dpsi = np.where(floor_active, _W_MIN, dpsi)
    else:
        dpsi = W

    # Kronecker design Kron = (S Sigma^{-1/2})^T kron Gamma^{-1/2} used in
    # solve_weighted_matrix_LS; rows ordered as vec_F(R_white).
    A = (S @ S_inv_sqrt).T                    # (n, p)
    B_kron = np.kron(A, G_inv_sqrt)           # (k*n, k*p)

    d = dpsi.flatten(order="F")               # (k*n,)
    psi_vec = (W * R_white).flatten(order="F")  # element-wise score values

    # Bread: -E[nabla psi] = Kron^T diag(psi') Kron.
    bread_mat = B_kron.T @ (d[:, None] * B_kron)  # (k*p, k*p)
    bread_mat = bread_mat + regularisation * np.eye(bread_mat.shape[0])

    # Meat: score_i = psi_i * basis_i (column-major vec).
    score_rows = psi_vec[:, None] * B_kron    # (k*n, k*p)
    if meat == "cluster":
        # Rows i + k*j belong to column j; sum within each column.
        score_rows = score_rows.reshape(n, k, k * p).sum(axis=1)  # (n, k*p)
    meat_mat = score_rows.T @ score_rows      # (k*p, k*p)

    try:
        bread_inv = np.linalg.solve(bread_mat, np.eye(bread_mat.shape[0]))
    except np.linalg.LinAlgError as exc:
        raise NumericalInstabilityError(
            "sandwich_se bread matrix singular; fit may be degenerate."
        ) from exc

    var_vec = bread_inv @ meat_mat @ bread_inv
    diag_var = np.diag(var_vec)
    if not np.all(np.isfinite(diag_var)):
        raise NumericalInstabilityError("sandwich_se produced non-finite variances.")
    if np.any(diag_var < -1e-9):
        raise NumericalInstabilityError(
            f"sandwich_se produced negative variance entries "
            f"(min = {diag_var.min():.3e})."
        )
    se_flat = np.sqrt(np.maximum(diag_var, 0.0))
    return se_flat.reshape((k, p), order="F")

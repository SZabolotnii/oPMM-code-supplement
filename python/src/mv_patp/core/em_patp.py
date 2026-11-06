"""Primary EM-PATP estimator for matrix MAL channel identification.

Implements ``em_patp_estimate`` per Spec.md Section 5.1 pseudocode and
design.md ``L2`` contract. The estimator alternates a pragmatic
sample-covariance E-step (Spec.md Section 5.4 ablation: matches the exact
matrix-GIG E-step within ~5% MSE delta on heavy-tailed scenarios) with a
matrix-IRLS M-step that uses the closed-form optimal oPMM_alpha weights
from ``mv_patp.core.score.compute_optimal_weights``.

Algorithm sketch
----------------
0. Initialise H by OLS, Gamma by sample row covariance, Sigma by the
   identifiability scenario (white -> I_n; ar1 -> trace-normalised).
1. Outer loop, t = 1..max_iter:
   1a. E-step: update Gamma from the column-whitened sample covariance
       of the current residuals; project identifiability.
   1b. Adaptive alpha selection (skipped if ``alpha_p_fixed=True``):
       call ``find_alpha_opt`` on the whitened residuals.
   1c. Inner IRLS, irls_iter = 1..irls_max_iter:
       compute optimal weights k1*, k2* from empirical cumulants of the
       whitened residuals; build IRLS scalar weights
       ``W = k1* + k2* |R_white|^{alpha-1}``; call
       ``solve_weighted_matrix_LS``; check inner convergence.
   1d. Check outer convergence on H Frobenius delta.
2. Return ``EMPATPResult`` with full diagnostics.

Satisfies Requirements R6.1, R6.2, R6.3, R6.4.
Task: Section 7.1, Section 7.2.

References
----------
Spec.md Section 5.1 (pseudocode), Section 5.2 (identifiability),
Section 5.3 (numerical safeguards). theory_revision.md Section 3 (corrected
optimal-weight formulas).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mv_patp.core.alpha_optimizer import find_alpha_opt
from mv_patp.core.identifiability import project_identifiability
from mv_patp.core.irls_matrix import solve_weighted_matrix_LS
from mv_patp.core.score import compute_optimal_weights
from mv_patp.core.whitening import matrix_inverse_sqrt
from mv_patp.utils.cumulants import empirical_cumulants
from mv_patp.utils.errors import EMDivergence, NumericalInstabilityError

logger = logging.getLogger(__name__)


@dataclass
class EMPATPResult:
    """Result returned by :func:`em_patp_estimate`.

    Mirrors the data-model contract in design.md (L2 section).
    """

    H: np.ndarray
    Gamma: np.ndarray
    Sigma: np.ndarray
    alpha_p: float
    alpha_MAL: float | None
    converged: bool
    n_iter_outer: int
    n_iter_irls_total: int
    log_history: dict[str, Any] = field(default_factory=dict)
    bootstrap_ci: tuple[np.ndarray, np.ndarray] | None = None
    # R1 safeguards (DSP-D-26-03654). Defaults keep other constructors
    # (e.g. em_patp_exact) valid; em_patp_estimate always sets them.
    guard_triggered: bool = False
    n_guard_backtracks: int = 0
    alpha_frozen_at: int | None = None
    stop_reason: str = ""
    huber_ref_ols: float = float("nan")
    huber_ref_final: float = float("nan")
    # Return-only guard (R1 v2 default): outer index of the returned iterate
    # (0 = OLS start) when the guard replaced the final iterate, else None.
    guard_return_iter: int | None = None


# --- R1 safeguard: fixed reference criterion --------------------------------
# A Huber loss of the residuals whitened with the OLS-start (Gamma_0, Sigma_0)
# and scaled by the OLS-start MAD. Every quantity that defines it is fixed at
# the OLS start, so it is comparable across all iterates and all alpha values.
# It is used only to accept/reject H updates; it never enters the weights.
_GUARD_HUBER_C: float = 1.345


def _huber_sum(z: np.ndarray, c: float = _GUARD_HUBER_C) -> float:
    a = np.abs(z)
    return float(np.sum(np.where(a <= c, 0.5 * a * a, c * a - 0.5 * c * c)))


class _ReferenceCriterion:
    """Fixed-reference Huber criterion ``L(H)`` for the divergence guard."""

    def __init__(self, Y: np.ndarray, S: np.ndarray, H0: np.ndarray,
                 G_inv_sqrt0: np.ndarray, S_inv_sqrt0: np.ndarray) -> None:
        self.Y, self.S = Y, S
        self.G, self.Sg = G_inv_sqrt0, S_inv_sqrt0
        z0 = self._white(H0)
        mad = float(np.median(np.abs(z0 - np.median(z0))))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0.0:
            scale = float(np.std(z0)) or 1.0
        self.scale = scale

    def _white(self, H: np.ndarray) -> np.ndarray:
        return self.G @ (self.Y - H @ self.S) @ self.Sg

    def __call__(self, H: np.ndarray) -> float:
        return _huber_sum(self._white(H) / self.scale)


def _ols_init(Y: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Closed-form OLS initialiser for H."""
    H_T, *_ = np.linalg.lstsq(S.T, Y.T, rcond=None)
    return H_T.T


def _safe_alpha_opt(
    R_white: np.ndarray, fallback: float = 1.5
) -> tuple[float, float]:
    """Find optimal alpha via full grid+Brent; on failure fall back."""
    try:
        eps_white = R_white.flatten()
        alpha, g_min, _ = find_alpha_opt({"eps_white": eps_white})
        return float(alpha), float(g_min)
    except Exception as exc:  # noqa: BLE001
        logger.warning("find_alpha_opt failed (%s); falling back to alpha=%.2f",
                       exc, fallback)
        return float(fallback), float("nan")


_ALPHA_DEGEN_LO: float = 0.92  # avoid degenerate basis around alpha=1
_ALPHA_DEGEN_HI: float = 1.08
_ALPHA_NEAR_GAUSS_DEFAULT: float = 2.0
_GAUSS_KURT_THRESHOLD: float = 0.5  # if gamma_4_excess < 0.5 -> near-Gaussian


def _alpha_heuristic(R_white: np.ndarray, lo: float = 0.3, hi: float = 3.0) -> float:
    """Stable kurtosis-based alpha update for adaptive EM.

    Adapted from Spec.md Section 5.1 pseudocode with two practical guards:

    1. **Near-Gaussian gate** -- when ``gamma_4_excess < 0.5`` the
       distribution is essentially Gaussian (``g(alpha; C) ~ 1`` for any
       ``alpha``), so we set ``alpha = 2.0`` for stability rather than
       letting the heuristic drift toward the degenerate alpha=1 region.

    2. **Degenerate-basis exclusion** -- the fractional basis
       sign(eps)|eps|^alpha collapses onto the linear basis at alpha=1
       (``k_2_star -> 0``). To avoid IRLS oscillation and tiny weights, the
       heuristic skips the [0.92, 1.08] band by snapping to alpha=1.5.

    The full ``find_alpha_opt`` search is reserved for the initial
    ``alpha_p`` estimate; in-loop adaptive updates use this damped form to
    avoid limit cycles between the cumulant estimate and the H update.
    """
    eps = R_white.flatten()
    var = float(np.var(eps, ddof=1))
    if var <= 0.0:
        return float(np.clip(1.5, lo, hi))
    fourth = float(np.mean(eps ** 4))
    gamma4_excess = max(fourth / max(var ** 2, 1e-30) - 3.0, 0.0)

    if gamma4_excess < _GAUSS_KURT_THRESHOLD:
        # Near-Gaussian: any alpha gives g~1; pick a stable default away
        # from the degenerate basis at alpha=1.
        return float(np.clip(_ALPHA_NEAR_GAUSS_DEFAULT, lo, hi))

    alpha = 1.5 / (1.0 + 0.3 * gamma4_excess)
    if _ALPHA_DEGEN_LO < alpha < _ALPHA_DEGEN_HI:
        alpha = 1.5
    return float(np.clip(alpha, lo, hi))


def em_patp_estimate(
    Y: np.ndarray,
    S: np.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    irls_max_iter: int = 25,
    irls_tol: float = 1e-8,
    alpha_p_init: float | None = None,
    alpha_p_fixed: bool = False,
    eps_min: float = 1e-10,
    regularization: float = 1e-6,
    identifiability: str = "white",
    raise_on_divergence: bool = False,
    compute_bootstrap_ci: bool = False,
    bootstrap_B: int = 500,
    return_guard: bool = True,
    return_guard_tol: float = 0.05,
    divergence_guard: bool = False,
    guard_tol: float = 1e-2,
    guard_mode: str = "level",
    guard_max_halvings: int = 5,
    alpha_freeze_after: int | None = None,
) -> EMPATPResult:
    """Estimate the channel matrix ``H`` under matrix-MAL noise.

    Parameters
    ----------
    Y, S : np.ndarray
        Observation ``(k, n)`` and known training matrix ``(p, n)``.
    max_iter : int, default 50
        Maximum number of outer EM iterations.
    tol : float, default 1e-6
        Outer convergence tolerance on relative Frobenius delta of ``H``.
    irls_max_iter : int, default 25
        Maximum number of inner IRLS iterations per outer iteration.
    irls_tol : float, default 1e-8
        Inner IRLS convergence tolerance.
    alpha_p_init : float or None, default None
        Initial fractional order. ``None`` uses ``find_alpha_opt`` on the
        OLS-residual whitened sample.
    alpha_p_fixed : bool, default False
        If True, ``alpha_p`` is held at ``alpha_p_init`` throughout.
    eps_min : float, default 1e-10
        Floor on ``|epsilon|`` when forming IRLS weights (Spec Section 5.3c).
    regularization : float, default 1e-6
        Eigenvalue ridge added to ``Gamma_hat`` after the E-step
        (Spec Section 5.3b).
    identifiability : {"white", "ar1"}, default "white"
        Passed straight to :func:`project_identifiability`.
    raise_on_divergence : bool, default False
        If True and convergence is not reached, raise :class:`EMDivergence`.
        Default behaviour returns the last iterate with ``converged=False``.
    compute_bootstrap_ci : bool, default False
        If True, attaches a parametric bootstrap CI (deferred to
        ``mv_patp.core.bootstrap_ci``; not yet implemented in Phase A).
    bootstrap_B : int, default 500
        Number of bootstrap replicates (used only when ``compute_bootstrap_ci``).
    return_guard : bool, default True
        R1 safeguard (v2, default). The iteration runs exactly as submitted;
        only the *returned* iterate is checked. A fixed reference criterion
        ``L(H)`` (Huber loss, c=1.345, of the residuals whitened with the
        OLS-start Gamma_0, Sigma_0 and scaled by their MAD) is recorded at
        the OLS start and after every outer iteration. If
        ``L(H_final) <= (1 + return_guard_tol) L(H_OLS)`` the final iterate
        is returned unchanged (bit-identical to the as-submitted fit).
        Otherwise the recorded iterate with the smallest ``L`` (OLS start
        included) is returned together with its Gamma, Sigma and alpha,
        ``guard_triggered=True``, ``stop_reason="guard_return"``,
        ``converged=False``; ``guard_return_iter`` gives its outer index.
    return_guard_tol : float, default 0.05
        Relative tolerance ``tau`` of the return-only guard. Chosen from the
        separation in safeguards.md: regular fits end within +0.32 % of
        ``L(H_OLS)``, the blow-ups 10-26 % above it.
    divergence_guard : bool, default False
        R1 v1 safeguard (superseded, off by default). A fixed reference
        criterion ``L(H)`` (as above) is evaluated after every outer
        step. A step whose ``L`` exceeds the bound is halved toward the
        previous iterate up to ``guard_max_halvings`` times; if no halved
        step meets the bound, the step is rejected and the loop stops at
        the previous iterate (``stop_reason="guard_reject"``,
        ``converged=False``). Every returned H meets the bound.
    guard_tol : float, default 1e-2
        Bound for ``guard_mode="level"``: ``L(H) <= (1 + guard_tol) L(H_OLS)``.
    guard_mode : {"level", "monotone"}, default "level"
        ``"level"``: the bound above (default). ``"monotone"``: every step
        must satisfy ``L(H_new) <= L(H_prev)`` (kept for the R1 record; it
        stops most regular fits at outer iteration 2 because L is not the
        oPMM objective, see safeguards.md).
    guard_max_halvings : int, default 5
        Maximum number of step halvings per outer step (smallest trial
        step 1/32 of the IRLS update).
    alpha_freeze_after : int or None, default None
        R1 v1 safeguard (superseded, off by default). In adaptive mode
        ``alpha`` is updated by the heuristic only in outer iterations
        ``1..alpha_freeze_after`` and is held fixed afterwards. ``None``
        (default) is the as-submitted, unfrozen behaviour. Ignored when
        ``alpha_p_fixed=True``.

    ``return_guard=False`` (with the other two safeguards at their default
    off) reproduces the as-submitted estimator bit for bit; with the
    defaults the result is bit-identical to it whenever the return-only
    guard does not trigger.

    Returns
    -------
    EMPATPResult

    Raises
    ------
    ValueError
        If ``Y`` and ``S`` have inconsistent column counts.
    EMDivergence
        Only when ``raise_on_divergence=True`` and the EM loop does not
        meet ``tol`` within ``max_iter``.
    NumericalInstabilityError
        On NaN/Inf in any iterate.
    """
    Y = np.asarray(Y, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if Y.ndim != 2 or S.ndim != 2:
        raise ValueError(f"Y and S must be 2-D; got Y.ndim={Y.ndim}, S.ndim={S.ndim}.")
    k, n = Y.shape
    p, n_S = S.shape
    if n_S != n:
        raise ValueError(f"Y has n={n}, S has n={n_S}; shapes must agree.")
    if guard_mode not in ("level", "monotone"):
        raise ValueError(f"guard_mode must be 'level' or 'monotone'; got {guard_mode!r}.")
    if not return_guard_tol >= 0.0:
        raise ValueError(f"return_guard_tol must be >= 0; got {return_guard_tol!r}.")

    # --- Step 0: initialisation ---------------------------------------------
    H = _ols_init(Y, S)
    R = Y - H @ S
    Gamma = (R @ R.T) / n + regularization * np.eye(k)
    Sigma = np.eye(n) if identifiability == "white" else (R.T @ R) / k
    Gamma, Sigma = project_identifiability(Gamma, Sigma, scenario=identifiability)

    # For "white" identifiability Sigma = I_n is fixed throughout; precompute
    # its inverse sqrt once to avoid O(n³) eigh on every E-step and IRLS call.
    _sigma_inv_sqrt_cached: np.ndarray | None = (
        np.eye(n) if identifiability == "white" else None
    )

    if alpha_p_init is None:
        Gamma_inv_sqrt = matrix_inverse_sqrt(Gamma)
        Sigma_inv_sqrt = (
            _sigma_inv_sqrt_cached
            if _sigma_inv_sqrt_cached is not None
            else matrix_inverse_sqrt(Sigma)
        )
        R_white = Gamma_inv_sqrt @ R @ Sigma_inv_sqrt
        alpha_p, g_min_init = _safe_alpha_opt(R_white)
    else:
        alpha_p = float(alpha_p_init)
        g_min_init = float("nan")

    history: dict[str, list] = {
        "outer_delta": [], "inner_delta_last": [], "alpha_p": [],
        "g_min": [], "irls_iters": [], "huber_ref": [], "guard_step": [],
    }
    history["alpha_p"].append(alpha_p)
    history["g_min"].append(g_min_init)

    n_iter_irls_total = 0
    converged = False

    # R1 safeguard: reference criterion fixed at the OLS start. Always
    # recorded (cheap, O(kn)); it only acts when divergence_guard=True.
    ref = _ReferenceCriterion(
        Y, S, H,
        matrix_inverse_sqrt(Gamma),
        _sigma_inv_sqrt_cached if _sigma_inv_sqrt_cached is not None
        else matrix_inverse_sqrt(Sigma),
    )
    L_ols = ref(H)
    L_cur = L_ols
    history["huber_ref"].append(L_ols)
    guard_triggered = False
    n_guard_backtracks = 0
    alpha_frozen_at: int | None = None
    stop_reason = "max_iter"
    # Return-only guard: best recorded iterate by L (OLS start = iterate 0).
    # Only references are kept; no iterate is modified in place.
    best = (L_ols, 0, H, Gamma, Sigma, alpha_p)

    # --- Outer EM loop ------------------------------------------------------
    for outer in range(1, max_iter + 1):
        H_prev = H.copy()

        # E-step: update Gamma from column-whitened residuals.
        R = Y - H @ S
        Sigma_inv_sqrt = (
            _sigma_inv_sqrt_cached
            if _sigma_inv_sqrt_cached is not None
            else matrix_inverse_sqrt(Sigma)
        )
        R_col_white = R @ Sigma_inv_sqrt
        Gamma = (R_col_white @ R_col_white.T) / n + regularization * np.eye(k)
        Gamma, Sigma = project_identifiability(Gamma, Sigma, scenario=identifiability)

        # Adaptive alpha_p update (skipped if fixed).
        # Use the kurtosis-based heuristic (Spec.md Section 5.1) inside the
        # outer EM loop because the full find_alpha_opt search creates
        # oscillating limit cycles when interleaved with the H update.
        # find_alpha_opt is used only for the initial alpha estimate.
        Gamma_inv_sqrt = matrix_inverse_sqrt(Gamma)
        R_white = Gamma_inv_sqrt @ R @ Sigma_inv_sqrt
        if not alpha_p_fixed:
            if alpha_freeze_after is None or outer <= alpha_freeze_after:
                alpha_p = _alpha_heuristic(R_white)
            elif alpha_frozen_at is None:
                # R1 safeguard: alpha held at its iteration-K value.
                alpha_frozen_at = int(alpha_freeze_after)
            history["alpha_p"].append(alpha_p)

        # Inner IRLS for H given (Gamma, Sigma, alpha_p).
        inner_delta_last = float("nan")
        for irls_iter in range(1, irls_max_iter + 1):
            R = Y - H @ S
            R_white = Gamma_inv_sqrt @ R @ Sigma_inv_sqrt

            # Optimal weights k1*, k2* from empirical cumulants on whitened residuals.
            eps_white = R_white.flatten()
            C = empirical_cumulants(eps_white, alpha_p)
            try:
                k1_star, k2_star = compute_optimal_weights(C, alpha_p)
            except Exception as exc:  # noqa: BLE001
                logger.warning("compute_optimal_weights failed (%s); using OLS fallback.", exc)
                k1_star, k2_star = 1.0, 0.0

            # IRLS scalar weights w(eps) = k1* + k2* |eps|^{alpha-1}.
            abs_R_safe = np.maximum(np.abs(R_white), eps_min)
            W = k1_star + k2_star * abs_R_safe ** (alpha_p - 1.0)
            # Guard against degenerate negative or zero weights from sampling noise.
            W = np.maximum(W, 1e-6)

            H_new = solve_weighted_matrix_LS(
                Y, S, W=W, Gamma_hat=Gamma, Sigma_hat=Sigma,
                _Sigma_inv_sqrt=_sigma_inv_sqrt_cached,
            )
            n_iter_irls_total += 1

            inner_delta = float(np.linalg.norm(H_new - H, ord="fro"))
            H_norm = max(float(np.linalg.norm(H, ord="fro")), 1e-12)
            H = H_new
            inner_delta_last = inner_delta
            if inner_delta / H_norm < irls_tol:
                break

        history["inner_delta_last"].append(inner_delta_last)
        history["irls_iters"].append(irls_iter)

        # R1 divergence guard: accept the outer step only if the fixed
        # reference criterion meets the bound; otherwise halve the step
        # toward H_prev; if no halving helps, reject it and stop.
        L_new = ref(H) if np.all(np.isfinite(H)) else float("inf")
        L_bound = L_cur if guard_mode == "monotone" else (1.0 + guard_tol) * L_ols
        guard_step = 1.0
        rejected = False
        if divergence_guard and not L_new <= L_bound:
            guard_triggered = True
            H_full = H
            accepted = False
            for m in range(1, guard_max_halvings + 1):
                n_guard_backtracks += 1
                guard_step = 0.5 ** m
                H_try = H_prev + guard_step * (H_full - H_prev)
                L_try = ref(H_try)
                if L_try <= L_bound:
                    H, L_new, accepted = H_try, L_try, True
                    break
            if not accepted:
                H, L_new, guard_step, rejected = H_prev, L_cur, 0.0, True
        L_cur = L_new
        history["huber_ref"].append(L_cur)
        history["guard_step"].append(guard_step)
        if L_cur < best[0]:
            best = (L_cur, outer, H, Gamma, Sigma, alpha_p)

        # Outer convergence on relative Frobenius delta of H.
        outer_delta = float(np.linalg.norm(H - H_prev, ord="fro"))
        H_prev_norm = max(float(np.linalg.norm(H_prev, ord="fro")), 1e-12)
        history["outer_delta"].append(outer_delta)
        if rejected:
            # The next outer step would start from the same H with the same
            # alpha and repeat the same rejected update; stop here.
            stop_reason = "guard_reject"
            break
        if outer_delta / H_prev_norm < tol:
            converged = True
            stop_reason = "converged"
            break

    if not converged and raise_on_divergence:
        raise EMDivergence(
            f"EM-PATP did not converge in {max_iter} iterations; "
            f"final outer delta={history['outer_delta'][-1]:.3e}."
        )

    # R1 return-only guard (v2 default): the iteration above is untouched;
    # if the final iterate's fixed criterion exceeds (1 + tau) L(H_OLS),
    # return the recorded iterate with the smallest L instead.
    guard_return_iter: int | None = None
    if return_guard and not L_cur <= (1.0 + return_guard_tol) * L_ols:
        L_cur, guard_return_iter, H, Gamma, Sigma, alpha_p = best
        guard_triggered = True
        converged = False
        stop_reason = "guard_return"

    if not np.all(np.isfinite(H)):
        raise NumericalInstabilityError("EM-PATP final H contains NaN/Inf.")

    bootstrap_ci = None
    if compute_bootstrap_ci:
        # Phase C task §13.1 — implementation deferred.
        logger.warning(
            "compute_bootstrap_ci=True but core/bootstrap_ci is not yet "
            "implemented; returning bootstrap_ci=None."
        )

    return EMPATPResult(
        H=H,
        Gamma=Gamma,
        Sigma=Sigma,
        alpha_p=float(alpha_p),
        alpha_MAL=None,  # method-of-moments alpha_MAL deferred (Spec §5 mentions but
                        # not required for primary H estimation pipeline).
        converged=converged,
        n_iter_outer=outer,
        n_iter_irls_total=n_iter_irls_total,
        log_history=history,
        bootstrap_ci=bootstrap_ci,
        guard_triggered=guard_triggered,
        n_guard_backtracks=n_guard_backtracks,
        alpha_frozen_at=alpha_frozen_at,
        stop_reason=stop_reason,
        huber_ref_ols=float(L_ols),
        huber_ref_final=float(L_cur),
        guard_return_iter=guard_return_iter,
    )

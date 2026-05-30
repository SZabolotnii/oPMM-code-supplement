"""α-optimizer for oPMM_α: coarse grid + Brent refinement.

Implements :func:`find_alpha_opt` which finds the optimal fractional order
α* = argmin_{α ∈ [0.3, 3.0]} g(α; C) using a two-stage strategy:

1. Coarse grid of 28 points (α ∈ {0.3, 0.4, …, 3.0}, step=0.1).
2. Brent refinement on the bracket around the grid minimum.

The two-stage approach guards against non-unimodality of g(α; C) on
heavy-tailed distributions (α_MAL ≤ 0.5), where a pure Brent search
could land in a local minimum.  See design.md DD-5 and Spec.md §4.4.

References
----------
Spec.md §4.4 — closed-form g(α; C) and α-optimizer specification.
design.md — Components and Interfaces, ``core/alpha_optimizer.py``.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.optimize import minimize_scalar

from mv_patp.utils.cumulants import closed_form_g_alpha, empirical_cumulants
from mv_patp.utils.errors import BrentNoConvergence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALPHA_MIN: float = 0.3
_ALPHA_MAX: float = 3.0
_GRID_STEP: float = 0.1
_MAX_EVALS: int = 50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def find_alpha_opt(
    C: dict,
    alpha_range: tuple[float, float] = (_ALPHA_MIN, _ALPHA_MAX),
    grid_step: float = _GRID_STEP,
) -> tuple[float, float, int]:
    """Find the optimal fractional order α* = argmin g(α; C).

    Two-stage strategy: coarse grid (28 points) followed by Brent
    refinement on the bracket around the grid minimum.  Total function
    evaluations are capped at 50; if the cap is exceeded the grid minimum
    is returned and a :class:`~mv_patp.utils.errors.BrentNoConvergence`
    warning is logged (not raised).

    Parameters
    ----------
    C : dict
        Cumulant dictionary.  Two modes are supported:

        **Mode A — raw residuals (recommended):** ``C`` contains the key
        ``"eps_white"`` (a 1-D ``np.ndarray`` of whitened residuals).  In
        this mode the α-dependent moments (``c_1_alpha``, ``nu_alpha``,
        ``sigma_alpha_sq``, ``mu_alpha``) are recomputed at each grid point
        via :func:`~mv_patp.utils.cumulants.empirical_cumulants`, which is
        the correct behaviour for the optimizer.

        **Mode B — fixed cumulant dict:** ``C`` does NOT contain
        ``"eps_white"`` but contains the standard keys returned by
        :func:`~mv_patp.utils.cumulants.empirical_cumulants`
        (``kappa_2``, ``c_1_alpha``, ``nu_alpha``, ``sigma_alpha_sq``).
        In this mode the same α-dependent moments are reused at every grid
        point.  This is only correct when the moments were computed at the
        same α as the grid point being evaluated (e.g., a single-α call).
        Use Mode A whenever possible.

    alpha_range : tuple[float, float], optional
        (alpha_min, alpha_max) search interval.  Defaults to (0.3, 3.0).
    grid_step : float, optional
        Step size for the coarse grid.  Defaults to 0.1.

    Returns
    -------
    alpha_opt : float
        Optimal α* ∈ [alpha_min, alpha_max].
    g_min : float
        Minimum value g(α*; C).
    n_evals : int
        Total number of g(α; C) evaluations (grid + Brent).

    Notes
    -----
    **Grid construction:** ``np.arange(alpha_min, alpha_max + grid_step/2,
    grid_step)`` rounded to 10 decimal places to avoid floating-point
    accumulation.  For the default parameters this yields exactly 28 points.

    **Brent bracket:** [α_{j*-1}, α_{j*+1}] clipped to [alpha_min, alpha_max].
    If the grid minimum is at the boundary (j*=0 or j*=N-1) the bracket
    degenerates to a one-sided interval; Brent still converges in this case.

    **Evaluation cap:** The counter includes all grid evaluations plus all
    evaluations performed by ``scipy.optimize.minimize_scalar`` during
    Brent refinement.  If the cap is reached, the function logs a warning
    at level WARNING using :data:`logging.getLogger(__name__)` and returns
    the grid minimum.

    **NaN handling:** Grid points where g(α; C) evaluates to ``np.nan``
    are ignored when selecting the grid minimum.

    Satisfies Requirements R5.1, R5.2, R5.3.

    Examples
    --------
    >>> import numpy as np
    >>> from mv_patp.utils.cumulants import empirical_cumulants
    >>> rng = np.random.default_rng(0)
    >>> eps = rng.laplace(loc=0.0, scale=1.0, size=5000)
    >>> C = {"eps_white": eps}
    >>> alpha_opt, g_min, n_evals = find_alpha_opt(C)
    >>> 0.3 <= alpha_opt <= 3.0
    True
    >>> n_evals <= 50
    True
    """
    alpha_min, alpha_max = float(alpha_range[0]), float(alpha_range[1])

    # Determine whether we have raw residuals (Mode A) or a fixed dict (Mode B).
    eps_white: np.ndarray | None = C.get("eps_white", None)

    def _build_C_at_alpha(a: float) -> dict:
        """Return a cumulant dict appropriate for evaluating g(a; C)."""
        if eps_white is not None:
            # Mode A: recompute α-dependent moments at each grid point.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return empirical_cumulants(eps_white, alpha=a)
        # Mode B: use the fixed dict as-is (caller's responsibility).
        return C

    # ------------------------------------------------------------------
    # Stage 1: Coarse grid
    # ------------------------------------------------------------------
    # Build grid with rounding to avoid floating-point accumulation.
    grid = np.round(
        np.arange(alpha_min, alpha_max + grid_step / 2.0, grid_step),
        decimals=10,
    )
    # Clip to [alpha_min, alpha_max] in case of floating-point overshoot.
    grid = grid[(grid >= alpha_min - 1e-12) & (grid <= alpha_max + 1e-12)]
    grid = np.clip(grid, alpha_min, alpha_max)

    n_grid = len(grid)
    g_grid = np.full(n_grid, np.nan)

    for i, a in enumerate(grid):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            g_grid[i] = closed_form_g_alpha(a, _build_C_at_alpha(a))

    n_evals: int = n_grid

    # Find grid minimum ignoring NaN values.
    valid_mask = np.isfinite(g_grid)
    if not valid_mask.any():
        # All NaN — return midpoint as fallback.
        logger.warning(
            "find_alpha_opt: all g(α; C) values are NaN on the coarse grid. "
            "Returning midpoint α=%.2f as fallback.",
            (alpha_min + alpha_max) / 2.0,
        )
        alpha_grid_opt = (alpha_min + alpha_max) / 2.0
        return float(alpha_grid_opt), float(np.nan), n_evals

    j_star = int(np.nanargmin(g_grid))
    alpha_grid_opt = float(grid[j_star])
    g_grid_min = float(g_grid[j_star])

    # ------------------------------------------------------------------
    # Stage 2: Brent refinement
    # ------------------------------------------------------------------
    # Bracket: [α_{j*-1}, α_{j*+1}] clipped to [alpha_min, alpha_max].
    bracket_lo = float(grid[max(0, j_star - 1)])
    bracket_hi = float(grid[min(n_grid - 1, j_star + 1)])

    # If bracket is degenerate (boundary case), skip Brent.
    if bracket_lo >= bracket_hi:
        return alpha_grid_opt, g_grid_min, n_evals

    # Remaining evaluation budget for Brent.
    remaining_budget = _MAX_EVALS - n_evals

    if remaining_budget <= 0:
        logger.warning(
            "find_alpha_opt: evaluation budget exhausted after coarse grid "
            "(%d evals). Returning grid minimum α*=%.4f, g=%.6f. "
            "BrentNoConvergence.",
            n_evals,
            alpha_grid_opt,
            g_grid_min,
        )
        return alpha_grid_opt, g_grid_min, n_evals

    # Wrap g(α; C) with an evaluation counter.
    brent_evals: list[int] = [0]
    budget_exceeded: list[bool] = [False]

    def _g_counted(a: float) -> float:
        brent_evals[0] += 1
        if brent_evals[0] > remaining_budget:
            budget_exceeded[0] = True
            # Return a large value so Brent stops exploring.
            return 1e10
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            val = closed_form_g_alpha(a, _build_C_at_alpha(a))
        return val if np.isfinite(val) else 1e10

    try:
        result = minimize_scalar(
            _g_counted,
            bounds=(bracket_lo, bracket_hi),
            method="bounded",
            options={"xatol": 1e-6, "maxiter": remaining_budget},
        )
        n_evals += brent_evals[0]

        if budget_exceeded[0] or n_evals > _MAX_EVALS:
            logger.warning(
                "find_alpha_opt: Brent refinement exceeded %d total evaluations "
                "(used %d). Returning grid minimum α*=%.4f, g=%.6f. "
                "BrentNoConvergence.",
                _MAX_EVALS,
                n_evals,
                alpha_grid_opt,
                g_grid_min,
            )
            n_evals = min(n_evals, _MAX_EVALS)
            return alpha_grid_opt, g_grid_min, n_evals

        alpha_brent = float(np.clip(result.x, alpha_min, alpha_max))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            g_brent = closed_form_g_alpha(alpha_brent, _build_C_at_alpha(alpha_brent))

        if np.isfinite(g_brent) and g_brent < g_grid_min:
            return alpha_brent, float(g_brent), n_evals
        else:
            return alpha_grid_opt, g_grid_min, n_evals

    except Exception as exc:  # noqa: BLE001
        n_evals += brent_evals[0]
        logger.warning(
            "find_alpha_opt: Brent refinement failed with %s. "
            "Returning grid minimum α*=%.4f, g=%.6f. BrentNoConvergence.",
            exc,
            alpha_grid_opt,
            g_grid_min,
        )
        return alpha_grid_opt, g_grid_min, n_evals

"""Sandwich SE coverage validation for §14.2 (Requirement R8.4).

Empirically validates that nominal 95% intervals built from the sandwich
SE bracket the true ``vec(H)`` near 0.95 on synthetic matrix-MAL data.

Reduced to ``M=200`` replicates (no inner bootstrap loop, so this is fast);
acceptance band ``[0.85, 0.99]`` absorbs Monte Carlo error at M=200
(2σ MC band at p=0.95 is ±0.031, so [0.92, 0.98]; we widen further to
account for sandwich's known asymptotic-bias under finite ``n``).

Task: §14.2.
Satisfies Requirement R8.4.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from mv_patp.core.em_patp import em_patp_estimate
from mv_patp.core.sandwich_se import sandwich_se
from mv_patp.data_gen.matrix_mal_dgp import sample_matrix_mal

_K: int = 2
_P: int = 2
_N: int = 256
_M_REPLICATES: int = 200
_ALPHA_MAL: float = 1.0
_COVERAGE_BAND: tuple[float, float] = (0.85, 0.99)
_Z_975: float = float(stats.norm.ppf(0.975))


@pytest.mark.slow
def test_sandwich_se_achieves_nominal_coverage_on_matrix_mal() -> None:
    """Empirical 95% sandwich-SE coverage falls inside [0.85, 0.99].

    Requirement R8.4: Wald-style intervals using the sandwich SE achieve
    nominal coverage on synthetic matrix MAL data (alpha_MAL=1.0, n=256).
    Sandwich is the secondary CI tool; band looser than bootstrap by spec.
    """
    rng = np.random.default_rng(42)
    H_truth = rng.standard_normal((_K, _P))

    inside_count = np.zeros((_K, _P), dtype=np.int64)

    for rep in range(_M_REPLICATES):
        seed_rep = 42 + rep
        rng_rep = np.random.default_rng(seed_rep)
        S = rng_rep.standard_normal((_P, _N))
        N = sample_matrix_mal(
            n_samples=1, k=_K, n=_N, alpha=_ALPHA_MAL, seed=seed_rep + 100,
        )[0]
        Y = H_truth @ S + N

        fit = em_patp_estimate(
            Y, S, max_iter=20, alpha_p_init=1.5, alpha_p_fixed=True,
        )
        se = sandwich_se(Y, S, fit)
        lower = fit.H - _Z_975 * se
        upper = fit.H + _Z_975 * se
        inside_count += ((lower <= H_truth) & (H_truth <= upper)).astype(np.int64)

    coverage = inside_count.mean() / _M_REPLICATES
    lo, hi = _COVERAGE_BAND
    assert lo <= coverage <= hi, (
        f"Empirical coverage {coverage:.3f} outside [{lo}, {hi}] "
        f"(target ~0.95, M={_M_REPLICATES})."
    )

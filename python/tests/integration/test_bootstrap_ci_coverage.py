"""Bootstrap CI coverage validation for §13.2 (Requirement R8.2).

Empirically validates that the parametric bootstrap CI brackets the true
``vec(H)`` near the nominal 95% rate on synthetic matrix-MAL data.

Reduced to ``M=60`` outer replicates × ``B=60`` bootstrap reps per replicate
for tractability. The full-spec validation (M=1000, B=500) lives in the
production MC harness; this test is a smoke check that the resampler and
percentile aggregation produce sensible coverage. Acceptance band is widened
from the spec target ``[0.93, 0.97]`` to ``[0.85, 0.99]`` to absorb Monte
Carlo error at M=60 (2σ MC band at p=0.95, M=60 is [0.89, 1.00]).

Task: §13.2.
Satisfies Requirement R8.2.
"""

from __future__ import annotations

import numpy as np
import pytest

from mv_patp.core.bootstrap_ci import parametric_bootstrap_ci
from mv_patp.core.em_patp import em_patp_estimate
from mv_patp.data_gen.matrix_mal_dgp import sample_matrix_mal

_K: int = 2
_P: int = 2
_N: int = 256
_M_REPLICATES: int = 60
_B_BOOT: int = 60
_ALPHA_MAL: float = 1.0
_COVERAGE_BAND: tuple[float, float] = (0.85, 0.99)


@pytest.mark.slow
def test_bootstrap_ci_achieves_nominal_coverage_on_matrix_mal() -> None:
    """Empirical 95% bootstrap coverage falls inside [0.85, 0.99].

    Requirement R8.2: parametric bootstrap CI achieves nominal coverage on
    synthetic matrix MAL data (alpha_MAL=1.0, n=256). Smoke-scale (M=60,
    B=60); see test docstring for MC-error rationale on the widened band.
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
        lower, upper = parametric_bootstrap_ci(
            Y, S, fit, B=_B_BOOT, seed=seed_rep + 200,
            alpha_mal_override=_ALPHA_MAL,
        )
        inside_count += ((lower <= H_truth) & (H_truth <= upper)).astype(np.int64)

    coverage = inside_count.mean() / _M_REPLICATES
    lo, hi = _COVERAGE_BAND
    assert lo <= coverage <= hi, (
        f"Empirical coverage {coverage:.3f} outside [{lo}, {hi}] "
        f"(target ~0.95, M={_M_REPLICATES}, B={_B_BOOT})."
    )

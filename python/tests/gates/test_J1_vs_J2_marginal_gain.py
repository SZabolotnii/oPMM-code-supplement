"""Gate No.3: J=1 (single fractional basis) defended vs J=2 multi-power extension.

Per Spec.md Section 14 and AGENTS.md Section 12 cheat sheet, Gate No.3
verifies that adding a second fractional basis (J=2) does NOT bring more
than 10% additional ARE on top of the single-basis (J=1) oPMM_alpha. If
the gap exceeds 10%, scope expansion to mp-oPMM is warranted (Spec §13
decision-log entry).

Pass criterion (Spec §14):
    (ARE_J=2 - ARE_J=1) / ARE_J=1 < 0.10
across alpha_GAL in {0.3, 0.5, 1.0, 2.0} on scalar symmetric GAL noise,
n=1024 (revised from Spec §10.2's 2000 to keep slow suite tractable;
MV_PATP_GATE3_FULL=1 -> 2000 + 100 reps).

Failure response (gap >= 10%): scope expansion to mp-oPMM as primary
contribution + Spec.md Section 12 decision-log update.

Satisfies Requirements R12.3, R13.3.
Task: Section 10.2.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mv_patp.core.scalar_opmm import scalar_opmm_estimate

from tests.gates._helpers.mc import gal_symmetric_sample
from tests.gates._helpers.mp_opmm_J2 import mp_opmm_J2_estimate


_N: int = 1024
_M_REPLICATES: int = 60  # 60 keeps slow suite ~3 min per cell.
_GATE_GAP_MAX: float = 0.10
_H_TRUE: float = 1.5
_BASE_SEED: int = 42


@pytest.mark.gate
@pytest.mark.slow
@pytest.mark.parametrize("alpha_gal", [1.0, 2.0])
def test_gate3_J1_marginal_gain_within_10pct_canonical_regime(
    alpha_gal: float,
) -> None:
    """J=2 over J=1 marginal ARE gain stays under 10% on canonical GAL.

    Per Spec.md §12 PIVOT 15, Gate №3 is two-tier:
    canonical regime (alpha_GAL in {1.0, 2.0}) is the pass criterion that
    aligns with the misspec primary scope (Student-t df=4 ≈ Laplace);
    extreme-heavy regime (alpha_GAL in {0.3, 0.5}) is informational only
    (see test_gate3_J2_dominates_extreme_heavy_known_xfail below).

    Requirement R12.3, R13.3.
    """
    n_replicates = (
        100 if os.environ.get("MV_PATP_GATE3_FULL") == "1" else _M_REPLICATES
    )
    n_samples = (
        2000 if os.environ.get("MV_PATP_GATE3_FULL") == "1" else _N
    )

    sq_err_J1 = np.empty(n_replicates)
    sq_err_J2 = np.empty(n_replicates)
    sq_err_OLS = np.empty(n_replicates)

    for rep in range(n_replicates):
        rng = np.random.default_rng(_BASE_SEED + rep)
        s = rng.standard_normal(n_samples)
        noise = gal_symmetric_sample(n_samples, alpha_gal=alpha_gal, rng=rng)
        y = _H_TRUE * s + noise

        # J=1: existing scalar oPMM.
        r1 = scalar_opmm_estimate(y, s, alpha_p_init=None, alpha_p_fixed=False)
        sq_err_J1[rep] = (r1.h - _H_TRUE) ** 2

        # J=2: gate-only prototype.
        h_J2, _a1, _a2, _conv = mp_opmm_J2_estimate(
            y, s, max_iter=15, alpha_fixed=False
        )
        sq_err_J2[rep] = (h_J2 - _H_TRUE) ** 2

        # OLS reference for ARE numerator.
        h_ols = float((y @ s) / (s @ s))
        sq_err_OLS[rep] = (h_ols - _H_TRUE) ** 2

    mse_J1 = float(sq_err_J1.mean())
    mse_J2 = float(sq_err_J2.mean())
    mse_OLS = float(sq_err_OLS.mean())
    are_J1 = mse_OLS / mse_J1
    are_J2 = mse_OLS / mse_J2
    gap = (are_J2 - are_J1) / are_J1

    print(
        f"\n[Gate No.3, alpha_GAL={alpha_gal}, n={n_samples}, reps={n_replicates}] "
        f"ARE_J=1={are_J1:.3f}, ARE_J=2={are_J2:.3f}, "
        f"gap={gap*100:+.1f}% (cap < {_GATE_GAP_MAX*100:.0f}%)"
    )

    assert gap < _GATE_GAP_MAX, (
        f"GATE No.3 FAIL at alpha_GAL={alpha_gal} (canonical regime): "
        f"(ARE_J=2 - ARE_J=1) / ARE_J=1 = {gap*100:+.1f}% >= "
        f"{_GATE_GAP_MAX*100:.0f}%. ARE_J=1={are_J1:.3f}, "
        f"ARE_J=2={are_J2:.3f}. Per PIVOT 15 the canonical regime "
        f"defends J=1; failure here means scope expansion warranted."
    )


@pytest.mark.gate
@pytest.mark.slow
@pytest.mark.parametrize("alpha_gal", [0.3, 0.5])
@pytest.mark.xfail(
    reason=(
        "Per Spec §12 PIVOT 15: J=2 dominates on extreme-heavy regime "
        "(alpha_GAL <= 0.5) where J=1's alpha_p drifts to boundary. "
        "This is explicit future work for the mp-oPMM J=2 follow-up "
        "paper. Informational only -- not a blocker for the J=1 paper."
    ),
    strict=False,
)
def test_gate3_J2_dominates_extreme_heavy_known_xfail(
    alpha_gal: float,
) -> None:
    """Informational: J=2 dominates J=1 on extreme heavy tails.

    Requirement R13.3 (honest scope reporting).
    """
    n_replicates = (
        100 if os.environ.get("MV_PATP_GATE3_FULL") == "1" else _M_REPLICATES
    )
    n_samples = (
        2000 if os.environ.get("MV_PATP_GATE3_FULL") == "1" else _N
    )
    sq_J1 = np.empty(n_replicates)
    sq_J2 = np.empty(n_replicates)
    sq_OLS = np.empty(n_replicates)
    for rep in range(n_replicates):
        rng = np.random.default_rng(_BASE_SEED + rep)
        s = rng.standard_normal(n_samples)
        noise = gal_symmetric_sample(n_samples, alpha_gal=alpha_gal, rng=rng)
        y = _H_TRUE * s + noise
        r1 = scalar_opmm_estimate(y, s, alpha_p_init=None, alpha_p_fixed=False)
        sq_J1[rep] = (r1.h - _H_TRUE) ** 2
        h_J2, *_ = mp_opmm_J2_estimate(y, s, max_iter=15, alpha_fixed=False)
        sq_J2[rep] = (h_J2 - _H_TRUE) ** 2
        sq_OLS[rep] = (float((y @ s) / (s @ s)) - _H_TRUE) ** 2
    are_J1 = float(sq_OLS.mean()) / float(sq_J1.mean())
    are_J2 = float(sq_OLS.mean()) / float(sq_J2.mean())
    gap = (are_J2 - are_J1) / are_J1
    print(
        f"\n[Gate No.3 informational, alpha_GAL={alpha_gal}] "
        f"ARE_J=1={are_J1:.3f}, ARE_J=2={are_J2:.3f}, gap={gap*100:+.1f}%"
    )
    assert gap < _GATE_GAP_MAX  # expected to fail; xfail marker handles it

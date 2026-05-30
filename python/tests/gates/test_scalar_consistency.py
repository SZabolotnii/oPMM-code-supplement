"""Gate No.1: k=1 reduction of matrix EM-PATP matches scalar oPMM reference.

This is the first of three mandatory gates per AGENTS.md Section 12 cheat
sheet. The matrix algorithm at k=1 must reduce -- mathematically and
empirically -- to the standalone scalar oPMM_alpha estimator. A failure
here points at a whitening, weight, or vec-shape bug in the matrix
implementation.

Pass criterion (Spec.md Section 14, Gate No.1):
    |MSE_matrix(k=1) - MSE_scalar| / MSE_scalar < 0.05
on 500 Monte Carlo replicates with n=512 across alpha_MAL in {0.5, 1.0}.

Failure response: halt manuscript drafting; debug the whitening or weight
construction in src/mv_patp/core/em_patp.py.

Satisfies Requirements R6.5, R12.1.
Task: Section 8.2.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mv_patp.baseline import ols
from mv_patp.core.em_patp import em_patp_estimate
from mv_patp.core.scalar_opmm import scalar_opmm_estimate

from tests.gates._helpers.mc import gal_symmetric_sample


_N: int = 512
_M_REPLICATES: int = 200  # 200 keeps slow-suite under ~5 min; final
                          # gate run uses 500 (set MV_PATP_GATE1_FULL=1).
_GATE_DELTA_TOLERANCE: float = 0.05
_H_TRUE: float = 1.5
_BASE_SEED: int = 42


def _run_one_replicate(
    alpha_gal: float, rep: int, n: int = _N, h_true: float = _H_TRUE
) -> tuple[float, float, float]:
    """Run one MC replicate; return (mse_matrix, mse_scalar, mse_ols)."""
    rng = np.random.default_rng(_BASE_SEED + rep)
    s = rng.standard_normal(n)
    noise = gal_symmetric_sample(n, alpha_gal=alpha_gal, rng=rng)
    y = h_true * s + noise

    # Scalar reference
    sr = scalar_opmm_estimate(y, s, alpha_p_init=None, alpha_p_fixed=False)
    h_scalar = sr.h

    # Matrix EM-PATP at k=1
    Y = y[None, :]
    S = s[None, :]
    mr = em_patp_estimate(Y, S, alpha_p_init=None, alpha_p_fixed=False)
    h_matrix = float(mr.H[0, 0])

    # OLS sanity
    h_ols = float(ols.fit(Y, S).H[0, 0])

    return (h_matrix - h_true) ** 2, (h_scalar - h_true) ** 2, (h_ols - h_true) ** 2


@pytest.mark.gate
@pytest.mark.slow
@pytest.mark.parametrize("alpha_gal", [0.5, 1.0])
def test_gate1_matrix_k1_matches_scalar_reference_when_symmetric_gal(
    alpha_gal: float,
) -> None:
    """k=1 matrix EM-PATP matches scalar oPMM reference within 5% MSE delta.

    Requirement R6.5, R12.1.
    """
    n_replicates = (
        500 if os.environ.get("MV_PATP_GATE1_FULL") == "1" else _M_REPLICATES
    )
    mse_matrix = np.empty(n_replicates)
    mse_scalar = np.empty(n_replicates)
    mse_ols = np.empty(n_replicates)

    for rep in range(n_replicates):
        m, s, o = _run_one_replicate(alpha_gal=alpha_gal, rep=rep)
        mse_matrix[rep] = m
        mse_scalar[rep] = s
        mse_ols[rep] = o

    avg_matrix = float(mse_matrix.mean())
    avg_scalar = float(mse_scalar.mean())
    avg_ols = float(mse_ols.mean())
    rel_delta = abs(avg_matrix - avg_scalar) / avg_scalar

    print(
        f"\n[Gate No.1, alpha_GAL={alpha_gal}] "
        f"MSE_matrix={avg_matrix:.5f}, MSE_scalar={avg_scalar:.5f}, "
        f"MSE_ols={avg_ols:.5f}, "
        f"rel_delta={rel_delta * 100:.2f}%, "
        f"ARE_matrix={avg_ols/avg_matrix:.3f}, "
        f"ARE_scalar={avg_ols/avg_scalar:.3f}"
    )

    assert rel_delta < _GATE_DELTA_TOLERANCE, (
        f"GATE No.1 FAIL at alpha_GAL={alpha_gal}: rel_delta={rel_delta*100:.2f}% "
        f">= {_GATE_DELTA_TOLERANCE*100:.0f}%. "
        f"MSE_matrix={avg_matrix:.5f}, MSE_scalar={avg_scalar:.5f}. "
        f"Halt manuscript drafting; debug whitening / weights "
        f"in src/mv_patp/core/em_patp.py."
    )

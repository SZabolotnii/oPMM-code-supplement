"""Pragmatic vs exact E-step ablation for §V-E of the manuscript.

Per Spec.md §5.4 the exact matrix-GIG E-step (Appendix F) and the
pragmatic sample-covariance E-step (Section IV-B) should agree within
5% MSE on the symmetric primary scope. This integration test runs
both variants on identical synthetic matrix MAL draws across
``alpha_MAL ∈ {0.5, 1.0, 2.0}`` and asserts the pass criterion.

Task: Section 12.2.
Satisfies Requirement R7.1 (exact matrix-GIG E-step matches pragmatic
within 5% MSE on the primary scope).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mv_patp.core.em_patp import em_patp_estimate
from mv_patp.core.em_patp_exact import em_patp_exact_estimate
from mv_patp.data_gen.matrix_mal_dgp import sample_matrix_mal


_K: int = 2
_P: int = 2
_N: int = 256
_M_REPLICATES: int = 30  # 30 reps keeps the ablation under ~30s.
_DELTA_TOL: float = 0.05
_BASE_SEED: int = 42


def _gen_replicate(rep: int, alpha_mal: float) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(_BASE_SEED + rep)
    H_truth = rng.standard_normal((_K, _P))
    S = rng.standard_normal((_P, _N))
    if alpha_mal <= (_K - 1) / 2.0:
        # Constraint: alpha > (k-1)/2 for matrix-Gamma validity.
        alpha_mal = (_K - 1) / 2.0 + 0.05
    N_arr = sample_matrix_mal(
        n_samples=1, k=_K, n=_N, alpha=alpha_mal,
        seed=_BASE_SEED + rep + 10_000,
    )
    Y = H_truth @ S + N_arr[0]
    return Y, S, H_truth


@pytest.mark.parametrize("alpha_mal", [0.6, 1.0, 2.0])
def test_pragmatic_within_5pct_of_exact_when_matrix_mal(
    alpha_mal: float,
) -> None:
    """Pragmatic and exact E-step variants agree within 5% MSE.

    Requirement R7.1.
    """
    n_replicates = (
        100 if os.environ.get("MV_PATP_ABLATION_FULL") == "1" else _M_REPLICATES
    )
    sq_err_prag = np.empty(n_replicates)
    sq_err_exact = np.empty(n_replicates)

    for rep in range(n_replicates):
        Y, S, H_truth = _gen_replicate(rep, alpha_mal)

        prag = em_patp_estimate(
            Y, S, alpha_p_init=2.0, alpha_p_fixed=True, max_iter=20
        )
        sq_err_prag[rep] = float(np.linalg.norm(prag.H - H_truth, ord="fro") ** 2)

        exact = em_patp_exact_estimate(
            Y, S, alpha_p_init=2.0, alpha_p_fixed=True, max_iter=20,
            alpha_mal_init=float(alpha_mal),
        )
        sq_err_exact[rep] = float(np.linalg.norm(exact.H - H_truth, ord="fro") ** 2)

    mse_prag = float(sq_err_prag.mean())
    mse_exact = float(sq_err_exact.mean())
    rel_delta = abs(mse_prag - mse_exact) / max(mse_exact, 1e-12)

    print(
        f"\n[Ablation alpha_MAL={alpha_mal}, k={_K}, p={_P}, n={_N}, "
        f"reps={n_replicates}] "
        f"MSE_pragmatic={mse_prag:.5f}, MSE_exact={mse_exact:.5f}, "
        f"|delta|/MSE_exact={rel_delta * 100:.2f}%"
    )

    assert rel_delta < _DELTA_TOL, (
        f"Pragmatic vs exact ablation FAIL at alpha_MAL={alpha_mal}: "
        f"{rel_delta * 100:.2f}% >= {_DELTA_TOL * 100:.0f}%. "
        f"MSE_pragmatic={mse_prag:.5f}, MSE_exact={mse_exact:.5f}. "
        f"Spec §5.4 promises < 5% delta on symmetric primary scope."
    )

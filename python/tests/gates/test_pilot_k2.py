"""Gate No.2: k=2 pilot ARE(matrix-oPMM vs OLS) >= 1.20 on misspec heavy-tail.

Phase B Gate №2 per Spec §12 PIVOT 13/14: the matrix MAL DGP at k>=2 is
conditionally Gaussian, so EM correctly reduces to GLS and oPMM offers
no extra gain there. The retrofitted gate (PIVOT 14) tests the
**misspecification regime**: iid Student-t (df=4) noise -- a canonical
heavy-tailed model that does NOT factorize as Γ⊗Σ -- is the primary
real-world target (Middleton Class A/B impulsive RF noise has the same
qualitative tail behaviour).

Pass criterion (revised Spec §6.4 table B):
    ARE >= 1.20 at df=4 (canonical heavy-tailed iid)
on 60 Monte Carlo replicates with k=2, p=2, n=512.

Failure response: halt manuscript drafting; document scope reduction in
Spec.md Section 12 decision log.

Satisfies Requirement R12.2 (revised post-PIVOT 14).
Task: Section 9.2 (retrofit).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from mv_patp.baseline import ols
from mv_patp.core.em_patp import em_patp_estimate


_K: int = 2
_P: int = 2
_N: int = 512
_M_REPLICATES: int = 60  # 60 keeps the slow suite under ~3 min;
                         # set MV_PATP_GATE2_FULL=1 for 200 replicates.
_DF: float = 4.0
_GATE_ARE_MIN: float = 1.20
_BASE_SEED: int = 42


def _student_t_problem(rep: int, df: float = _DF) -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """One MC replicate: synthetic Y = H S + N with iid Student-t noise.

    Student-t representation: ``N_{ij} = Z_{ij} / sqrt(chi2_{ij} / df)``
    with independent Z ~ N(0, 1) and chi2 ~ chi^2(df). Tail index df
    controls the heaviness; df=4 gives finite second moment but heavy
    excess kurtosis (gamma_4 = 6 / (df - 4) -> infinite at df=4 limit).
    """
    rng = np.random.default_rng(_BASE_SEED + rep)
    H_truth = rng.standard_normal((_K, _P))
    S = rng.standard_normal((_P, _N))
    z = rng.standard_normal((_K, _N))
    chi2 = rng.chisquare(df=df, size=(_K, _N))
    N = z / np.sqrt(chi2 / df)
    Y = H_truth @ S + N
    return Y, S, H_truth


@pytest.mark.gate
@pytest.mark.slow
def test_gate2_matrix_em_patp_beats_ols_on_k2_pilot_misspec() -> None:
    """k=2 pilot Monte Carlo: ARE(matrix-oPMM vs OLS) >= 1.20 on Student-t df=4.

    Requirement R12.2 (revised). Misspecification regime per Spec §12 PIVOT 14.
    """
    n_replicates = (
        200 if os.environ.get("MV_PATP_GATE2_FULL") == "1" else _M_REPLICATES
    )
    sq_err_em = np.empty(n_replicates)
    sq_err_ols = np.empty(n_replicates)

    for rep in range(n_replicates):
        Y, S, H_truth = _student_t_problem(rep)

        em = em_patp_estimate(Y, S, alpha_p_init=None, alpha_p_fixed=False,
                              max_iter=30)
        sq_err_em[rep] = float(np.linalg.norm(em.H - H_truth, ord="fro") ** 2)

        ols_fit = ols.fit(Y, S)
        sq_err_ols[rep] = float(np.linalg.norm(ols_fit.H - H_truth, ord="fro") ** 2)

    mse_em = float(sq_err_em.mean())
    mse_ols = float(sq_err_ols.mean())
    are = mse_ols / mse_em

    print(
        f"\n[Gate No.2 (Student-t df={_DF}), k={_K}, p={_P}, n={_N}, "
        f"reps={n_replicates}] "
        f"MSE_EM={mse_em:.5f}, MSE_OLS={mse_ols:.5f}, ARE={are:.3f} "
        f"(target >= {_GATE_ARE_MIN})"
    )

    assert are >= _GATE_ARE_MIN, (
        f"GATE No.2 FAIL: ARE={are:.3f} < {_GATE_ARE_MIN} on Student-t df={_DF}. "
        f"MSE_EM={mse_em:.5f}, MSE_OLS={mse_ols:.5f}. "
        f"Halt manuscript drafting; consider scope further restriction "
        f"(Spec.md Section 12 next PIVOT)."
    )


@pytest.mark.gate
@pytest.mark.slow
def test_gate2_matrix_em_patp_non_degradation_on_matrix_mal() -> None:
    """Non-degradation cell: matrix MAL DGP gives ARE in [0.85, 1.05].

    Per Spec §6.4 table A: estimator must not degrade Gaussian-conditional
    optimal efficiency on parametrically-correct matrix MAL data.
    Requirement R9.4 (revised).
    """
    from mv_patp.data_gen.matrix_mal_dgp import sample_matrix_mal

    n_replicates = (
        100 if os.environ.get("MV_PATP_GATE2_FULL") == "1" else _M_REPLICATES
    )
    alpha_mal = 1.0
    sq_err_em = np.empty(n_replicates)
    sq_err_ols = np.empty(n_replicates)

    for rep in range(n_replicates):
        rng = np.random.default_rng(_BASE_SEED + rep)
        H_truth = rng.standard_normal((_K, _P))
        S = rng.standard_normal((_P, _N))
        N_arr = sample_matrix_mal(
            n_samples=1, k=_K, n=_N, alpha=alpha_mal,
            seed=_BASE_SEED + rep + 10_000,
        )
        N = N_arr[0]
        Y = H_truth @ S + N
        em = em_patp_estimate(Y, S, alpha_p_init=None, alpha_p_fixed=False,
                              max_iter=30)
        sq_err_em[rep] = float(np.linalg.norm(em.H - H_truth, ord="fro") ** 2)
        sq_err_ols[rep] = float(
            np.linalg.norm(ols.fit(Y, S).H - H_truth, ord="fro") ** 2
        )

    are = float(sq_err_ols.mean()) / float(sq_err_em.mean())
    print(
        f"\n[Gate No.2 non-degradation (matrix MAL alpha_MAL={alpha_mal})] "
        f"ARE={are:.3f} (target in [0.85, 1.05])"
    )

    assert 0.85 <= are <= 1.10, (
        f"Non-degradation FAIL: ARE={are:.3f} on matrix MAL is outside "
        f"the [0.85, 1.10] band. Either EM is doing something weird, or "
        f"matrix MAL DGP is not behaving as conditional Gaussian."
    )

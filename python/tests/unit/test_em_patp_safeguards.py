"""R1 safeguards of oPMM_alpha (DSP-D-26-03654).

v2 (default): return-only guard. The iteration is the as-submitted one; a
fixed reference criterion L (Huber loss of the residuals whitened at the OLS
start) is recorded at the OLS start and after every outer iteration. If the
final iterate has L > (1 + tau) L(H_OLS), tau = 0.05, the recorded iterate
with the smallest L is returned instead (``stop_reason="guard_return"``).
Otherwise the fit is bit-identical to the as-submitted one.

v1 (superseded, options off by default): per-step divergence guard (step
halving, reject-and-stop) and alpha freeze after K outer iterations.

The regression replicate is (k, p, n) = (2, 2, 512), Student-t df=4,
seed scheme of ``synthetic_full_v1`` (``default_rng(42 + b)``), b = 460:
the as-submitted fit returns an error 216x that of OLS.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.mv_patp_mimo import run
from mv_patp.core.em_patp import _ols_init, em_patp_estimate
from tests.gates._helpers.mc import gal_symmetric_sample

# As-submitted estimator: every safeguard off.
UNGUARDED = dict(return_guard=False, divergence_guard=False, alpha_freeze_after=None)
# v1 safeguards (superseded): per-step guard + freeze, no return guard.
V1 = dict(return_guard=False, divergence_guard=True, alpha_freeze_after=5)


def _cell(structure: str, k: int = 2, n: int = 512, alpha_mal: float = float("nan")):
    return run.CellSpec(k=k, n=n, alpha_MAL=alpha_mal, p=2, M_norm=0.0,
                        sigma_structure=structure, replicates=1,
                        estimator="opmm_alpha")


def _replicate(df: int, b: int):
    return run.draw_replicate(_cell(f"student_t_{df}"), seed=42, rep=b)


def _replicate_460():
    return _replicate(4, 460)


def _sq(H_hat, H):
    return float(np.linalg.norm(H_hat - H, ord="fro") ** 2)


def _assert_identical(a, b) -> None:
    np.testing.assert_array_equal(a.H, b.H)
    np.testing.assert_array_equal(a.Gamma, b.Gamma)
    np.testing.assert_array_equal(a.Sigma, b.Sigma)
    assert a.alpha_p == b.alpha_p
    assert a.converged == b.converged
    assert a.n_iter_outer == b.n_iter_outer
    assert a.n_iter_irls_total == b.n_iter_irls_total
    assert a.stop_reason == b.stop_reason
    assert a.huber_ref_final == b.huber_ref_final


# --- v2: return-only guard (default) ----------------------------------------

def test_defaults_are_return_only_guard() -> None:
    import inspect
    sig = inspect.signature(em_patp_estimate).parameters
    assert sig["return_guard"].default is True
    assert sig["return_guard_tol"].default == 0.05
    assert sig["divergence_guard"].default is False
    assert sig["alpha_freeze_after"].default is None


_IDENTITY_CASES = [
    ("student_t_3", 2, float("nan"), 0), ("student_t_3", 2, float("nan"), 7),
    ("student_t_4", 2, float("nan"), 3), ("student_t_4", 4, float("nan"), 11),
    ("student_t_5", 2, float("nan"), 342), ("student_t_5", 2, float("nan"), 369),
    ("middleton_A0p35_G0p1", 2, float("nan"), 1),
    ("middleton_A0p1_G0p1", 2, float("nan"), 4),
    ("middleton_A1p0_G0p1", 2, float("nan"), 9),
    ("I", 2, 1.0, 2), ("I", 4, 2.0, 5),
]


@pytest.mark.parametrize("structure,k,alpha_mal,b", _IDENTITY_CASES)
def test_default_is_bit_identical_to_unguarded_when_guard_idle(
        structure, k, alpha_mal, b) -> None:
    Y, S, _ = run.draw_replicate(_cell(structure, k=k, alpha_mal=alpha_mal),
                                 seed=42, rep=b)
    new = em_patp_estimate(Y, S)
    legacy = em_patp_estimate(Y, S, **UNGUARDED)
    assert not new.guard_triggered and new.guard_return_iter is None
    _assert_identical(new, legacy)
    assert new.log_history == legacy.log_history


def test_run_unguarded_estimator_matches_legacy_call() -> None:
    fit = run._ESTIMATOR_REGISTRY["opmm_alpha_unguarded"]()
    Y, S, _ = _replicate_460()
    _assert_identical(fit(Y, S), em_patp_estimate(Y, S, **UNGUARDED))


def test_replicate_460_diverges_without_safeguards() -> None:
    Y, S, H = _replicate_460()
    ols = _sq(_ols_init(Y, S), H)
    res = em_patp_estimate(Y, S, **UNGUARDED)
    assert _sq(res.H, H) / ols > 100.0          # documented: 216x
    assert not res.converged
    assert not res.guard_triggered
    assert res.huber_ref_final > 1.05 * res.huber_ref_ols


def test_replicate_460_return_guard_keeps_error_near_ols() -> None:
    Y, S, H = _replicate_460()
    ols = _sq(_ols_init(Y, S), H)
    legacy = em_patp_estimate(Y, S, **UNGUARDED)
    res = em_patp_estimate(Y, S)
    assert res.guard_triggered and res.stop_reason == "guard_return"
    assert not res.converged
    # The iteration itself is unchanged: same number of outer/inner steps.
    assert res.n_iter_outer == legacy.n_iter_outer
    assert res.n_iter_irls_total == legacy.n_iter_irls_total
    assert _sq(res.H, H) / ols < 5.0            # observed 0.44
    # The returned iterate is the recorded minimum of L.
    L_hist = res.log_history["huber_ref"]
    assert res.huber_ref_final == min(L_hist)
    assert res.huber_ref_final == L_hist[res.guard_return_iter]
    assert res.alpha_p == res.log_history["alpha_p"][res.guard_return_iter]


def test_return_guard_returns_ols_start_when_it_is_best() -> None:
    # b=460: the first outer step raises L by 14 % (safeguards.md §1), so a
    # one-iteration fit triggers the guard and the OLS start is the minimum.
    Y, S, _ = _replicate_460()
    res = em_patp_estimate(Y, S, max_iter=1)
    assert res.guard_triggered and res.guard_return_iter == 0
    np.testing.assert_array_equal(res.H, _ols_init(Y, S))
    assert res.huber_ref_final == res.huber_ref_ols
    with pytest.raises(ValueError, match="return_guard_tol"):
        em_patp_estimate(Y, S, return_guard_tol=-0.1)


def test_df5_rep_342_not_degraded() -> None:
    # v1 (freeze + per-step guard) took this fit from 0.57x to 4.39x OLS.
    Y, S, H = _replicate(5, 342)
    new = em_patp_estimate(Y, S)
    legacy = em_patp_estimate(Y, S, **UNGUARDED)
    assert not new.guard_triggered
    assert _sq(new.H, H) == _sq(legacy.H, H)


@pytest.mark.parametrize("rep", [63, 115, 122])
def test_gal_k1_transient_excursions_are_not_truncated(rep) -> None:
    # Gate No. 1 design (k=1, symmetric GAL alpha_GAL=1.0): L rises
    # transiently in these fits and comes back; v1 truncated them.
    rng = np.random.default_rng(42 + rep)
    s = rng.standard_normal(512)
    y = 1.5 * s + gal_symmetric_sample(512, alpha_gal=1.0, rng=rng)
    new = em_patp_estimate(y[None, :], s[None, :])
    legacy = em_patp_estimate(y[None, :], s[None, :], **UNGUARDED)
    assert not new.guard_triggered
    _assert_identical(new, legacy)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_returned_iterate_meets_bound_or_is_the_minimum(seed) -> None:
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((2, 2))
    S = rng.standard_normal((2, 256))
    Y = H @ S + rng.standard_t(3, size=(2, 256))
    res = em_patp_estimate(Y, S)
    assert res.stop_reason in {"converged", "max_iter", "guard_return"}
    if res.guard_triggered:
        assert res.huber_ref_final == min(res.log_history["huber_ref"])
    else:
        assert res.huber_ref_final <= 1.05 * res.huber_ref_ols


# --- v1 options (superseded, kept available) --------------------------------

def test_v1_replicate_460_guard_keeps_error_near_ols() -> None:
    Y, S, H = _replicate_460()
    ols = _sq(_ols_init(Y, S), H)
    res = em_patp_estimate(Y, S, **V1)
    assert res.guard_triggered and res.n_guard_backtracks >= 1
    assert _sq(res.H, H) / ols < 5.0            # observed 2.57
    assert res.huber_ref_final <= 1.01 * res.huber_ref_ols + 1e-12


def test_v1_monotone_mode_returns_ols_on_replicate_460() -> None:
    Y, S, _ = _replicate_460()
    res = em_patp_estimate(Y, S, return_guard=False, divergence_guard=True,
                           alpha_freeze_after=5, guard_mode="monotone")
    assert res.stop_reason == "guard_reject"
    assert np.allclose(res.H, _ols_init(Y, S))


def test_v1_alpha_is_frozen_after_K_outer_iterations() -> None:
    Y, S, _ = _replicate_460()          # runs all 50 outer iterations
    for K in (2, 5):
        res = em_patp_estimate(Y, S, alpha_freeze_after=K)
        hist = res.log_history["alpha_p"]   # [init, iter 1, iter 2, ...]
        assert res.n_iter_outer > K
        assert res.alpha_frozen_at == K
        assert all(a == hist[K] for a in hist[K + 1:])
    free = em_patp_estimate(Y, S, **UNGUARDED)
    assert len(set(free.log_history["alpha_p"][6:])) > 1
    assert free.alpha_frozen_at is None


def test_v1_freeze_does_not_act_when_alpha_is_fixed() -> None:
    Y, S, _ = _replicate_460()
    res = em_patp_estimate(Y, S, alpha_p_init=0.5, alpha_p_fixed=True,
                           max_iter=8, alpha_freeze_after=5)
    assert res.alpha_frozen_at is None
    assert set(res.log_history["alpha_p"]) == {0.5}


def test_invalid_guard_mode_raises() -> None:
    Y, S, _ = _replicate_460()
    with pytest.raises(ValueError, match="guard_mode"):
        em_patp_estimate(Y, S, guard_mode="strict")


# --- run.py diagnostics ------------------------------------------------------

def test_run_records_fit_diagnostics_per_replicate(tmp_path) -> None:
    cell = run.CellSpec(k=2, n=64, alpha_MAL=float("nan"), p=2, M_norm=0.0,
                        sigma_structure="student_t_4", replicates=3,
                        estimator="opmm_alpha")
    row = run.run_mc_cell(cell, seed=42, per_replicate=True)
    assert row["status"] == "ok"
    assert 0.0 <= row["converged_rate"] <= 1.0
    assert 0.0 <= row["guard_rate"] <= 1.0
    reps = row["_rep_rows"]
    assert len(reps) == 3
    for r in reps:
        for col in ("converged", "n_outer", "n_inner", "alpha_hat",
                    "guard_triggered", "stop_reason"):
            assert col in r
        assert r["n_outer"] >= 1 and r["n_inner"] >= r["n_outer"]
        assert np.isfinite(r["alpha_hat"])
    path = run.write_per_replicate(reps, tmp_path / "reps.csv")
    header = path.read_text().splitlines()[0].split(",")
    for col in ("converged", "n_outer", "n_inner", "alpha_hat",
                "guard_triggered", "stop_reason"):
        assert col in header


def test_ols_baseline_diagnostics_are_neutral() -> None:
    cell = run.CellSpec(k=2, n=64, alpha_MAL=float("nan"), p=2, M_norm=0.0,
                        sigma_structure="student_t_4", replicates=2,
                        estimator="ols")
    row = run.run_mc_cell(cell, seed=1, per_replicate=True)
    assert row["guard_rate"] == 0.0
    assert all(r["guard_triggered"] is False for r in row["_rep_rows"])

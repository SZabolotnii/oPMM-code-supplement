"""R1 (DSP-D-26-03654) changes to the Monte Carlo harness.

* MAL alpha is never clipped: an invalid cell reports ``invalid_alpha``;
  every row records ``alpha_used``.
* ``alpha_MAL_offset`` gives a per-k valid grid.
* Non-MAL cells are not replicated over the inert alpha axis.
* Per-replicate output and the paired design (identical data per rep
  across estimators).
* New noise kind ``ar1t_rho<r>_df<v>`` (AR(1) rows, Student-t entries).
* Paired-bootstrap script.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from experiments.mv_patp_mimo import run

ROOT = Path(__file__).resolve().parents[2]


def _cell(**kw) -> run.CellSpec:
    base = dict(k=2, n=64, alpha_MAL=1.0, p=2, M_norm=0.0,
                sigma_structure="I", replicates=3, estimator="ols")
    base.update(kw)
    return run.CellSpec(**base)


# --- alpha validation and alpha_used ---------------------------------------

def test_invalid_mal_alpha_is_reported_not_clipped() -> None:
    cell = _cell(k=4, alpha_MAL=1.0)          # needs alpha > 1.5
    assert not cell.alpha_valid
    row = run.run_mc_cell(cell, seed=1, per_replicate=True)
    assert row["status"] == "invalid_alpha"
    assert row["alpha_used"] == 1.0
    assert row["n_replicates"] == 0
    assert row["_rep_rows"] == []


def test_draw_noise_raises_on_invalid_alpha() -> None:
    with pytest.raises(run.InvalidAlphaError):
        run.draw_replicate(_cell(k=8, alpha_MAL=2.0), seed=0, rep=0)


def test_boundary_alpha_is_invalid() -> None:
    assert not _cell(k=4, alpha_MAL=1.5).alpha_valid
    assert _cell(k=4, alpha_MAL=1.5000001).alpha_valid


def test_alpha_used_recorded_for_valid_mal_and_nan_for_student_t() -> None:
    row = run.run_mc_cell(_cell(k=2, alpha_MAL=1.0), seed=1)
    assert row["status"] == "ok" and row["alpha_used"] == 1.0
    row_t = run.run_mc_cell(
        _cell(sigma_structure="student_t_4", alpha_MAL=float("nan")), seed=1
    )
    assert row_t["status"] == "ok" and np.isnan(row_t["alpha_used"])


def test_alpha_offset_grid_is_valid_per_k() -> None:
    cfg = {"replicates": 2, "estimators": ["ols"],
           "grid": {"k": [2, 4, 8], "n": [64], "p": [2],
                    "alpha_MAL_offset": [0.5, 1.0, 3.0], "sigma_structure": ["I"]}}
    cells = run.expand_grid(cfg)
    assert len(cells) == 9
    assert all(c.alpha_valid for c in cells)
    got = sorted((c.k, c.alpha_used, c.alpha_offset) for c in cells)
    assert (8, 3.5 + 0.5, 0.5) in got and (4, 1.5 + 3.0, 3.0) in got


def test_absolute_alpha_still_supported() -> None:
    cfg = {"estimators": ["ols"],
           "grid": {"k": [2], "alpha_MAL": [0.6, 2.0], "sigma_structure": ["I"]}}
    cells = run.expand_grid(cfg)
    assert sorted(c.alpha_used for c in cells) == [0.6, 2.0]


def test_inert_alpha_axis_not_replicated_for_non_mal() -> None:
    cfg = {"estimators": ["ols", "huber_irls"],
           "grid": {"k": [2], "n": [64], "p": [2], "alpha_MAL": [0.6, 1.0, 2.0],
                    "sigma_structure": ["I", "student_t_4", "ar1t_rho0p5_df4",
                                        "gaussian"]}}
    cells = run.expand_grid(cfg)
    by_sigma = pd.Series([c.sigma_structure for c in cells]).value_counts()
    assert by_sigma["I"] == 3 * 2
    assert by_sigma["student_t_4"] == 2
    assert by_sigma["ar1t_rho0p5_df4"] == 2
    assert by_sigma["gaussian"] == 2
    assert len({c.cell_id() for c in cells}) == len(cells)


# --- per-replicate output and paired design --------------------------------

@pytest.mark.parametrize("sigma", ["I", "student_t_3", "ar1t_rho0p9_df4", "gaussian"])
def test_paired_design_identical_data_across_estimators(sigma: str) -> None:
    a = _cell(sigma_structure=sigma, estimator="ols")
    b = _cell(sigma_structure=sigma, estimator="opmm_alpha")
    assert a.data_id() == b.data_id()
    for rep in range(3):
        Ya, Sa, Ha = run.draw_replicate(a, seed=7, rep=rep)
        Yb, Sb, Hb = run.draw_replicate(b, seed=7, rep=rep)
        assert np.array_equal(Ya, Yb) and np.array_equal(Sa, Sb)
        assert np.array_equal(Ha, Hb)
    Y0, *_ = run.draw_replicate(a, seed=7, rep=0)
    Y1, *_ = run.draw_replicate(a, seed=7, rep=1)
    assert not np.array_equal(Y0, Y1)


def test_per_replicate_rows_match_aggregate() -> None:
    row = run.run_mc_cell(_cell(estimator="opmm_alpha", replicates=4), seed=3,
                          per_replicate=True)
    reps = pd.DataFrame(row["_rep_rows"])
    assert list(reps["rep"]) == [0, 1, 2, 3]
    assert list(reps["seed"]) == [3, 4, 5, 6]
    assert set(run.PER_REP_COLUMNS) <= set(reps.columns)
    assert np.isclose(reps["sq_err"].mean(), row["mse_mean"])
    assert reps["alpha_hat"].notna().all()          # oPMM exposes alpha_p
    ols = run.run_mc_cell(_cell(replicates=2), seed=3, per_replicate=True)
    assert pd.DataFrame(ols["_rep_rows"])["alpha_hat"].isna().all()


def test_main_writes_reps_csv(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({
        "run_id": "t", "replicates": 3, "estimators": ["ols", "huber_irls"],
        "grid": {"k": [2, 4], "n": [64], "p": [2], "alpha_MAL": [1.0],
                 "sigma_structure": ["I", "student_t_4"]},
    }))
    out = tmp_path / "out"
    rc = run.main(["--config", str(cfg), "--output-dir", str(out), "--per-replicate"])
    assert rc == 0
    cells = pd.read_csv(out / "cells.csv")
    reps = pd.read_csv(out / "reps.csv")
    # k=4 MAL at alpha=1.0 is invalid -> reported, no replicate rows
    inv = cells[cells["status"] == "invalid_alpha"]
    assert set(inv["k"]) == {4} and set(inv["sigma_structure"]) == {"I"}
    assert "alpha_used" in cells.columns
    assert len(reps) == 3 * (len(cells) - len(inv))
    assert list(reps.columns) == list(run.PER_REP_COLUMNS)
    # results identical with 2 workers (data depend only on data_id, seed, rep)
    out2 = tmp_path / "out2"
    run.main(["--config", str(cfg), "--output-dir", str(out2), "--per-replicate",
              "--workers", "2"])
    reps2 = pd.read_csv(out2 / "reps.csv")
    key = ["cell_id", "rep"]
    m = reps.merge(reps2, on=key, suffixes=("", "_2"))
    assert len(m) == len(reps)
    assert np.allclose(m["sq_err"], m["sq_err_2"])


# --- ar1t noise kind ---------------------------------------------------------

def test_parse_ar1t() -> None:
    assert run.parse_ar1t("ar1t_rho0p5_df4") == (0.5, 4.0)
    assert run.parse_ar1t("ar1t_rho0p9_df3") == (0.9, 3.0)
    with pytest.raises(ValueError):
        run.parse_ar1t("ar1t_rho1p0_df4")
    with pytest.raises(ValueError):
        run.parse_ar1t("ar1t_df4")


def test_ar1t_covariance_matches_documented_formula() -> None:
    k, n, rho, df = 4, 400_000, 0.9, 6.0     # df=6: finite 4th moment, stable check
    cell = _cell(k=k, n=n, sigma_structure="ar1t_rho0p9_df6")
    N = run._draw_noise(cell, np.random.default_rng(0))
    emp = N @ N.T / n
    expected = df / (df - 2.0) * run.ar1_corr(k, rho)
    assert np.allclose(emp, expected, atol=0.03)


def test_ar1t_is_heavy_tailed() -> None:
    cell = _cell(k=2, n=200_000, sigma_structure="ar1t_rho0p5_df5")
    N = run._draw_noise(cell, np.random.default_rng(1))
    x = N[0]                       # first row = iid t_5 (L[0,0] = 1)
    kurt = np.mean((x - x.mean()) ** 4) / np.var(x) ** 2 - 3.0
    assert kurt > 2.0              # t_5 excess kurtosis = 6


# --- paired bootstrap script ------------------------------------------------

def _load_bootstrap():
    path = ROOT / "scripts" / "r1_paired_bootstrap.py"
    spec = importlib.util.spec_from_file_location("r1_paired_bootstrap", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_paired_bootstrap_ratio_ci() -> None:
    mod = _load_bootstrap()
    rng = np.random.default_rng(0)
    base = rng.exponential(size=500)
    num, den = 2.0 * base, base               # perfectly paired: ratio exactly 2
    point, lo, hi = mod.paired_ratio_ci(num, den, 500, np.random.default_rng(1))
    assert point == pytest.approx(2.0) and lo == pytest.approx(2.0)
    assert hi == pytest.approx(2.0)


def test_paired_bootstrap_summary_from_reps(tmp_path: Path) -> None:
    mod = _load_bootstrap()
    rows = []
    for rep in range(50):
        e = 1.0 + 0.1 * rep
        for est, f in [("ols", 2.0), ("huber_irls", 1.0), ("opmm_alpha", 1.25)]:
            rows.append(dict(data_id="d", k=2, n=64, p=2, sigma_structure="s",
                             alpha_used=np.nan, estimator=est, rep=rep,
                             status="ok", sq_err=f * e))
    res = mod.summarise(pd.DataFrame(rows), B=200, seed=1)
    are = res[(res.comparison == "are_vs_ols") & (res.estimator == "opmm_alpha")]
    assert are["ratio"].iloc[0] == pytest.approx(1.6)
    hub = res[res.comparison == "mse_ratio_opmm_alpha_over_huber_irls"]
    assert hub["ratio"].iloc[0] == pytest.approx(1.25)
    assert (res["n_pairs"] == 50).all()

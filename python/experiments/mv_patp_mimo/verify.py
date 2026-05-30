"""Verification harness for mv-patp-mimo.

Reads the CSVs and manifest produced by ``run.py`` and evaluates the 12
verification criteria from ``requirements.md`` (R1 through R13). Emits a
machine-readable JSON summary and a human-readable Markdown table.

Usage
-----
``python -m experiments.mv_patp_mimo.verify --results-dir <dir>
[--out-json <path>] [--out-markdown <path>]``

Exit code is 0 when every criterion passes, 1 when any criterion fails.
SKIPPED criteria (because the upstream estimator is not yet implemented)
do not fail the run; they appear in the report so progress is visible.

Satisfies Requirement R11.2 (gate pass/fail harness).

Task: Section 19.1 (verify.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

# --- Check protocol ----------------------------------------------------------

@dataclass
class CheckResult:
    requirement: str
    passed: bool | None  # True / False / None for SKIPPED
    detail: str

    @property
    def status(self) -> str:
        if self.passed is True:
            return "PASS"
        if self.passed is False:
            return "FAIL"
        return "SKIP"


CheckFn = Callable[[pd.DataFrame, dict], CheckResult]
_CHECKS: dict[str, CheckFn] = {}


def register(req: str):
    def _wrap(fn: CheckFn) -> CheckFn:
        _CHECKS[req] = fn
        return fn
    return _wrap


def _skipped(req: str, reason: str) -> CheckResult:
    return CheckResult(requirement=req, passed=None, detail=f"SKIPPED: {reason}")


# --- Individual checks --------------------------------------------------------
# Each check reads the cells DataFrame and the manifest dict, returns a
# CheckResult. Until the estimator pipeline lands the checks return SKIPPED
# with a reason pointing at the missing upstream component.

@register("R1.1")
def _check_dgp_skewness(df: pd.DataFrame, manifest: dict) -> CheckResult:
    return _skipped("R1.1", "requires data_gen/matrix_mal_dgp + DGP cumulants test")


@register("R2.1")
def _check_identifiability(df: pd.DataFrame, manifest: dict) -> CheckResult:
    return _skipped("R2.1", "requires core/identifiability + EM run")


@register("R3.2")
def _check_score_recovery(df: pd.DataFrame, manifest: dict) -> CheckResult:
    return _skipped("R3.2", "requires core/score + Kunchenko comparison test")


@register("R4.1")
def _check_closed_form_g(df: pd.DataFrame, manifest: dict) -> CheckResult:
    return _skipped("R4.1", "requires utils/cumulants.closed_form_g_alpha")


@register("R5.1")
def _check_alpha_optimizer(df: pd.DataFrame, manifest: dict) -> CheckResult:
    return _skipped("R5.1", "requires core/alpha_optimizer.find_alpha_opt")


@register("R6.1")
def _check_em_convergence(df: pd.DataFrame, manifest: dict) -> CheckResult:
    return _skipped("R6.1", "requires core/em_patp.em_patp_estimate")


@register("R8.1")
def _check_bootstrap_coverage(df: pd.DataFrame, manifest: dict) -> CheckResult:
    return _skipped("R8.1", "requires core/bootstrap_ci")


@register("R9.1")
def _check_synthetic_are_alpha1(df: pd.DataFrame, manifest: dict) -> CheckResult:
    """ARE(oPMM_alpha vs OLS) >= 1.20 on Student-t df=4 (revised post-PIVOT 14)."""
    if df.empty or "estimator" not in df.columns:
        return _skipped("R9.1", "no MC cells executed yet")
    if "sigma_structure" not in df.columns or "mse_mean" not in df.columns:
        return _skipped("R9.1", "cells.csv schema lacks DGP/MSE columns")
    # Synthetic-only check; DeepMIMO cells have data_source="deepmimo" and
    # are filtered out so the same student_t_4 sigma_structure does not
    # cross-contaminate the synthetic Track A finding.
    if "data_source" in df.columns:
        df = df[df["data_source"].fillna("synthetic") == "synthetic"]
    target = df[df["sigma_structure"] == "student_t_4"]
    if target.empty:
        return _skipped("R9.1", "no Student-t df=4 cells in results")
    ols = target[target["estimator"] == "ols"]
    opmm = target[target["estimator"] == "opmm_alpha"]
    if ols.empty or opmm.empty:
        return _skipped("R9.1", "missing OLS or opmm_alpha row at df=4")
    mse_ols = float(ols["mse_mean"].iloc[0])
    mse_opmm = float(opmm["mse_mean"].iloc[0])
    if mse_opmm <= 0.0 or not (mse_opmm == mse_opmm):  # NaN guard
        return CheckResult("R9.1", False, f"opmm_alpha MSE = {mse_opmm}")
    are = mse_ols / mse_opmm
    threshold = 1.20
    passed = are >= threshold
    return CheckResult(
        "R9.1",
        passed,
        f"Student-t df=4 ARE(opmm_alpha vs OLS) = {are:.3f} "
        f"(target >= {threshold}); MSE_OLS={mse_ols:.5f}, "
        f"MSE_opmm={mse_opmm:.5f}",
    )


@register("R9.5")
def _check_pmm2_collapse_on_symmetric(df: pd.DataFrame, manifest: dict) -> CheckResult:
    """matrix-PMM2 ARE within [0.95, 1.05] on symmetric data (R9.5 collapse)."""
    if df.empty or "estimator" not in df.columns:
        return _skipped("R9.5", "no MC cells executed yet")
    if "sigma_structure" not in df.columns:
        return _skipped("R9.5", "cells.csv schema lacks sigma_structure")
    # Synthetic-only check (Track A scope); see R9.1 comment.
    if "data_source" in df.columns:
        df = df[df["data_source"].fillna("synthetic") == "synthetic"]
    # Use any symmetric cell (matrix MAL or symmetric Student-t).
    target = df[df["estimator"] == "matrix_pmm2"]
    ols = df[df["estimator"] == "ols"]
    if target.empty or ols.empty:
        return _skipped("R9.5", "missing matrix_pmm2 or ols row")
    # Match on sigma_structure to compare like-for-like.
    rows = []
    for sig in target["sigma_structure"].unique():
        ols_row = ols[ols["sigma_structure"] == sig]
        pmm_row = target[target["sigma_structure"] == sig]
        if ols_row.empty or pmm_row.empty:
            continue
        m_ols = float(ols_row["mse_mean"].iloc[0])
        m_pmm = float(pmm_row["mse_mean"].iloc[0])
        if m_pmm <= 0.0 or not (m_pmm == m_pmm):
            continue
        rows.append((sig, m_ols / m_pmm))
    if not rows:
        return _skipped("R9.5", "no like-for-like (sigma_structure) pairs")
    # R9.5 per requirements.md §9 item 6: collapse band is [0.98, 1.02].
    # Empirically: collapse holds for near-Gaussian (I, df>=1000); heavy-tailed
    # df=3,4 show PMM2 efficiency gain (ARE>1.02) — documented as Decision Log PIVOT 16.
    out_of_band = [(s, a) for s, a in rows if not (0.98 <= a <= 1.02)]
    detail = "; ".join(f"{s}: ARE={a:.3f}" for s, a in rows)
    if out_of_band:
        return CheckResult("R9.5", False, f"PMM2 outside collapse band [0.98, 1.02]: {detail}")
    return CheckResult("R9.5", True, f"PMM2 within collapse band: {detail}")


@register("R10.1")
def _check_deepmimo_nmse(df: pd.DataFrame, manifest: dict) -> CheckResult:
    """NMSE improvement of oPMM_α over OLS >= 15% on DeepMIMO at α_MAL=1.

    Requirement R10.1: real-data benchmark on ray-traced channels.
    Skips with a clear reason if the cells.csv has no DeepMIMO rows.
    """
    if df.empty or "data_source" not in df.columns:
        return _skipped("R10.1", "cells.csv has no data_source column")
    dm = df[df["data_source"] == "deepmimo"]
    if dm.empty:
        return _skipped("R10.1", "no DeepMIMO cells in this run")
    sub = dm[(dm["alpha_MAL"] == 1.0) &
             (dm["estimator"].isin(["ols", "opmm_alpha"]))]
    if sub.empty:
        return _skipped("R10.1", "no α=1.0 ols/opmm_alpha pairs")
    rows: list[tuple[str, float, float, float]] = []
    for (scen, snr), grp in sub.groupby(["scenario", "snr_db"]):
        ols_row = grp[grp["estimator"] == "ols"]
        opmm_row = grp[grp["estimator"] == "opmm_alpha"]
        if ols_row.empty or opmm_row.empty:
            continue
        m_ols = float(ols_row["mse_mean"].iloc[0])
        m_opmm = float(opmm_row["mse_mean"].iloc[0])
        if m_ols <= 0.0 or m_opmm <= 0.0:
            continue
        nmse_pct = 100.0 * (m_ols - m_opmm) / m_ols
        rows.append((scen, float(snr), m_ols, nmse_pct))
    if not rows:
        return _skipped("R10.1", "no valid (scenario, snr) NMSE pairs")
    detail = "; ".join(
        f"{s}@{snr:.0f}dB:{nmse:.1f}%" for s, snr, _, nmse in rows
    )
    best_pct = max(nmse for _, _, _, nmse in rows)
    if best_pct >= 15.0:
        return CheckResult("R10.1", True,
                           f"NMSE improvement {best_pct:.1f}% (>=15%): {detail}")
    return CheckResult("R10.1", False,
                       f"max NMSE improvement {best_pct:.1f}% < 15%: {detail}")


@register("R10.2")
def _check_deepmimo_ranking(df: pd.DataFrame, manifest: dict) -> CheckResult:
    """Estimator ranking on DeepMIMO: opmm_alpha <= matrix_t_mle <= wishart_mle <= ols.

    Requirement R10.2: ordering by mean MSE across (scenario, snr) pairs at
    α_MAL=1.0. Lower MSE is better, so the expected ranking is monotone-ascending
    in MSE. Pass if observed mean ranking matches the expected order.
    matrix_f_mle excluded — its df constraint fails at k=16 real-equivalent.
    """
    if df.empty or "data_source" not in df.columns:
        return _skipped("R10.2", "cells.csv has no data_source column")
    dm = df[(df["data_source"] == "deepmimo") & (df["alpha_MAL"] == 1.0)]
    if dm.empty:
        return _skipped("R10.2", "no DeepMIMO α=1.0 cells")
    expected = ["opmm_alpha", "matrix_t_mle", "wishart_mle", "ols"]
    means: dict[str, float] = {}
    for est in expected:
        rows = dm[dm["estimator"] == est]
        if rows.empty:
            return _skipped("R10.2", f"missing estimator: {est}")
        means[est] = float(rows["mse_mean"].mean())
    detail = " < ".join(f"{e}={means[e]:.3e}" for e in expected)
    monotone = all(means[expected[i]] <= means[expected[i + 1]]
                   for i in range(len(expected) - 1))
    if monotone:
        return CheckResult("R10.2", True, f"ranking holds: {detail}")
    return CheckResult("R10.2", False, f"ranking violated: {detail}")


@register("R11.3")
def _check_manifest_completeness(df: pd.DataFrame, manifest: dict) -> CheckResult:
    """Manifest must contain git_commit, package_versions, seed."""
    required = {"git_commit", "package_versions", "seed", "config_path",
                "wall_clock_seconds", "pass_fail_summary"}
    missing = required - set(manifest.keys())
    if missing:
        return CheckResult(
            requirement="R11.3", passed=False,
            detail=f"manifest missing keys: {sorted(missing)}",
        )
    return CheckResult(
        requirement="R11.3", passed=True,
        detail="all manifest keys present",
    )


@register("R12.1")
def _check_gate1(df: pd.DataFrame, manifest: dict) -> CheckResult:
    return _skipped("R12.1", "Gate No.1 lives in tests/gates/test_scalar_consistency.py")


@register("R13.1")
def _check_honest_scope_pmm2(df: pd.DataFrame, manifest: dict) -> CheckResult:
    return _skipped("R13.1", "requires asymmetric MC cells (M != 0) and PMM2 baseline")


# --- Driver ------------------------------------------------------------------

def load_results(results_dir: Path) -> tuple[pd.DataFrame, dict]:
    cells_path = results_dir / "cells.csv"
    manifest_path = results_dir / "manifest.json"
    df = pd.read_csv(cells_path) if cells_path.exists() else pd.DataFrame()
    manifest: dict = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    return df, manifest


def run_checks(
    df: pd.DataFrame, manifest: dict
) -> list[CheckResult]:
    return [_CHECKS[req](df, manifest) for req in sorted(_CHECKS)]


def render_markdown(results: list[CheckResult]) -> str:
    lines = ["| Requirement | Status | Detail |", "|---|---|---|"]
    for r in results:
        lines.append(f"| {r.requirement} | {r.status} | {r.detail} |")
    return "\n".join(lines) + "\n"


def render_json(results: list[CheckResult]) -> str:
    return json.dumps(
        [
            {
                "requirement": r.requirement,
                "status": r.status,
                "detail": r.detail,
            }
            for r in results
        ],
        indent=2,
    )


# --- CLI ---------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify mv-patp-mimo MC results against requirements.",
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory written by run.py (cells.csv + manifest.json).")
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-markdown", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df, manifest = load_results(args.results_dir)
    results = run_checks(df, manifest)

    md = render_markdown(results)
    print(md)

    if args.out_markdown:
        args.out_markdown.write_text(md)
    if args.out_json:
        args.out_json.write_text(render_json(results))

    failed = sum(1 for r in results if r.passed is False)
    skipped = sum(1 for r in results if r.passed is None)
    passed = sum(1 for r in results if r.passed is True)
    logger.info("Summary: %d passed, %d failed, %d skipped.",
                passed, failed, skipped)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""R1 (Rv4-C2b, App. C "coverage within [0.90, 0.98]"): Wald coverage study
of the corrected sandwich SE, per replicate and per entry of H.

Assumed interface (``mv_patp.core.sandwich_se``, working tree 2026-11-04,
see manuscript/dsp_r1/r1_work/sandwich_fix.md)::

    sandwich_se(Y, S, fit_result, *, eigenvalue_floor=1e-6,
                regularisation=1e-12, bread="psi_prime" | "weight",
                meat="element" | "cluster") -> np.ndarray (k, p)

* ``fit_result`` is an ``EMPATPResult`` (uses ``.H, .Gamma, .Sigma,
  .alpha_p``); ``SE[i, j]`` belongs to ``H_hat[i, j]``.
* ``bread="weight"`` is the pre-R1 (submitted) bread diag(w), kept only for
  the old-vs-new comparison; ``"psi_prime"`` is the corrected bread.
* Raises ``NumericalInstabilityError`` on a singular bread or non-finite /
  negative variances (recorded here as status "se_failed").
If the sandwich task changes the signature, only ``_se_variants`` needs
editing.

DGP: ``run.draw_replicate`` (H ~ N(0,1) and S ~ N(0,1) per replicate,
``default_rng(seed + rep)``), so the data equal the run.py cells with the
same (k, n, p, sigma_structure, seed). Noise: iid Student-t df 3/4/5/8 and
matched MAL (sigma "I", alpha = (k-1)/2 + 0.5). Coverage is per-entry, 95 %
nominal Wald, against the true H of the replicate.

Fits (``--alpha-mode``): ``adaptive`` = default ``em_patp_estimate`` (R1
safeguards on); ``fixed`` = alpha held at the heuristic limit alpha-bar from
theory.md (t3 0.3, t4 0.3, t5 0.536, t8 1.5). MAL uses 2.0, the fixed value
of the submitted supplement; it is NOT the heuristic limit on these cells
(the heuristic on the true MAL noise gives median 1.50 at k=2 and 1.30 at
k=4, 200 draws; verifier 2026-11-04). Non-converged fits
are kept and flagged (``converged``); report coverage on all fits and on the
converged subset.

Output ``coverage_reps.csv`` (one row per replicate x entry x SE variant)
and ``coverage_summary.csv`` (coverage, mean SE / empirical SD, MC s.e.).

Usage::

    PYTHONPATH=src python scripts/r1_sandwich_coverage.py -o <dir>
        [--reps 500] [--k 2 4] [--n 512] [--noise student_t_3 ...]
        [--alpha-mode adaptive fixed] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

Z95 = 1.959963984540054
ALPHA_BAR = {"student_t_3": 0.3, "student_t_4": 0.3, "student_t_5": 0.536,
             "student_t_8": 1.5,
             "I": 2.0}  # MAL: submitted fixed value, not alpha-bar (see docstring)
SE_VARIANTS = (("psi_prime", "element"), ("psi_prime", "cluster"),
               ("weight", "element"))


def _se_variants(Y, S, res) -> dict[str, np.ndarray]:
    from mv_patp.core.sandwich_se import sandwich_se
    out = {}
    for bread, meat in SE_VARIANTS:
        try:
            out[f"{bread}/{meat}"] = np.asarray(sandwich_se(Y, S, res, bread=bread,
                                                            meat=meat))
        except Exception:  # noqa: BLE001
            out[f"{bread}/{meat}"] = None
    return out


def _job(args: tuple) -> list[dict]:
    k, n, p, noise, mode, seed, rep_lo, rep_hi = args
    from experiments.mv_patp_mimo.run import CellSpec, draw_replicate
    from mv_patp.core.em_patp import em_patp_estimate
    alpha_mal = (k - 1) / 2.0 + 0.5 if noise == "I" else float("nan")
    cell = CellSpec(k=k, n=n, alpha_MAL=alpha_mal, p=p, M_norm=0.0,
                    sigma_structure=noise, replicates=rep_hi, estimator="opmm_alpha")
    rows = []
    for rep in range(rep_lo, rep_hi):
        Y, S, H = draw_replicate(cell, seed, rep)
        kw = {} if mode == "adaptive" else dict(alpha_p_fixed=True,
                                                alpha_p_init=ALPHA_BAR[noise])
        try:
            res = em_patp_estimate(Y, S, **kw)
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(data_id=cell.data_id(), mode=mode, rep=rep,
                             status=f"fit_failed: {type(exc).__name__}"))
            continue
        ses = _se_variants(Y, S, res)
        for name, se in ses.items():
            for i in range(k):
                for j in range(p):
                    base = dict(data_id=cell.data_id(), k=k, n=n, p=p,
                                sigma_structure=noise, mode=mode, rep=rep,
                                seed=seed + rep, entry=f"{i},{j}", se_variant=name,
                                h_true=float(H[i, j]), h_hat=float(res.H[i, j]),
                                alpha_hat=float(res.alpha_p),
                                converged=bool(res.converged))
                    if se is None:
                        rows.append({**base, "status": "se_failed"})
                        continue
                    s_ij = float(se[i, j])
                    rows.append({**base, "status": "ok", "se": s_ij,
                                 "covered": bool(abs(res.H[i, j] - H[i, j])
                                                 <= Z95 * s_ij)})
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    ok["err"] = ok["h_hat"] - ok["h_true"]
    out = []
    for subset, d in (("all", ok), ("converged", ok[ok["converged"]])):
        for key, g in d.groupby(["data_id", "sigma_structure", "k", "n", "mode",
                                 "se_variant"]):
            cov = g["covered"].mean()
            # SE/SD per entry, then median over entries (as in sandwich_fix.md)
            ratio = g.groupby("entry").apply(
                lambda e: e["se"].mean() / e["err"].std(ddof=1)).median()
            out.append(dict(zip(["data_id", "sigma_structure", "k", "n", "mode",
                                 "se_variant"], key),
                            subset=subset, n_entries=len(g),
                            n_reps=g["rep"].nunique(), coverage=cov,
                            mc_se=np.sqrt(cov * (1 - cov) / len(g)),
                            se_over_sd=ratio))
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output-dir", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--n", type=int, nargs="+", default=[512])
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--noise", nargs="+",
                    default=["student_t_3", "student_t_4", "student_t_5",
                             "student_t_8", "I"])
    ap.add_argument("--alpha-mode", nargs="+", default=["adaptive", "fixed"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)

    jobs = [(k, n, a.p, noise, mode, a.seed, lo, min(lo + a.chunk, a.reps))
            for k in a.k for n in a.n for noise in a.noise for mode in a.alpha_mode
            for lo in range(0, a.reps, a.chunk)]
    t0 = time.time()
    rows: list[dict] = []
    if a.workers <= 1:
        for j in jobs:
            rows += _job(j)
    else:
        import os
        from multiprocessing import get_context
        os.environ.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                           "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"})
        with get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(_job, jobs, chunksize=1):
                rows += r
    wall = time.time() - t0
    a.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.output_dir / "coverage_reps.csv", index=False)
    summarise(df).to_csv(a.output_dir / "coverage_summary.csv", index=False)
    (a.output_dir / "manifest.json").write_text(json.dumps(
        {"script": "scripts/r1_sandwich_coverage.py", "argv": sys.argv[1:],
         "wall_clock_seconds": wall, "n_rows": len(df)}, indent=2))
    print(f"{len(df)} rows, {wall:.1f} s -> {a.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

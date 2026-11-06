"""R1: k = 1 reduction on symmetric scalar GAL, per replicate (Gate No. 1 data).

The submitted Table I GAL rows (ARE 4.50 at alpha_GAL = 0.5, 1.46 at
alpha_GAL = 1.0 / Laplace; "matrix k = 1 matches scalar within 0.26 %") come
from ``tests/gates/test_scalar_consistency.py`` and were never stored. This
script stores them, with the same DGP and seeds as the gate:
``rng = default_rng(42 + rep)``; ``s ~ N(0, 1)^n``; noise = sqrt(W) Z with
W ~ Gamma(alpha_GAL, 1) (``tests/gates/_helpers/mc.gal_symmetric_sample``);
y = 1.5 s + noise; n = 512.

Estimators (identical data per replicate):

* ``ols``
* ``opmm_alpha``            matrix EM-PATP at k = 1, R1 safeguards (default)
* ``opmm_alpha_unguarded``  matrix EM-PATP at k = 1, as submitted
* ``scalar_opmm``           ``mv_patp.core.scalar_opmm.scalar_opmm_estimate``

Note (safeguards.md, verification): with the R1 defaults Gate No. 1 FAILS at
alpha_GAL = 1.0 (matrix ARE 1.276 vs legacy 1.473 on 200 reps). This run
documents both columns; which one the paper prints is the author's call.

Output ``reps.csv`` in run.py column names (sigma_structure = "gal_<a>"), so
``scripts/r1_paired_bootstrap.py`` applies (``--target opmm_alpha --refs
scalar_opmm opmm_alpha_unguarded``).

Usage::

    PYTHONPATH=src python scripts/r1_gal_k1.py -o <dir> [--reps 500]
        [--alpha-gal 0.5 1.0] [--n 512] [--workers 8]
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

H_TRUE = 1.5
BASE_SEED = 42
ESTIMATORS = ("ols", "opmm_alpha", "opmm_alpha_unguarded", "scalar_opmm")


def _job(args: tuple) -> list[dict]:
    alpha_gal, n, est, rep_lo, rep_hi = args
    from mv_patp.core.em_patp import em_patp_estimate
    from mv_patp.core.scalar_opmm import scalar_opmm_estimate
    from tests.gates._helpers.mc import gal_symmetric_sample
    rows = []
    for rep in range(rep_lo, rep_hi):
        rng = np.random.default_rng(BASE_SEED + rep)
        s = rng.standard_normal(n)
        y = H_TRUE * s + gal_symmetric_sample(n, alpha_gal=alpha_gal, rng=rng)
        row = dict(data_id=f"gal_k1_alpha{alpha_gal}_n{n}", estimator=est, k=1, n=n,
                   p=1, sigma_structure=f"gal_{alpha_gal}", alpha_used=np.nan,
                   rep=rep, seed=BASE_SEED + rep)
        t0 = time.perf_counter()
        try:
            alpha_hat, conv, n_outer = np.nan, True, np.nan
            if est == "ols":
                h = float((y @ s) / (s @ s))
            elif est == "scalar_opmm":
                r = scalar_opmm_estimate(y, s, alpha_p_init=None, alpha_p_fixed=False)
                h = float(r.h)
                alpha_hat = float(getattr(r, "alpha_p", np.nan))
                conv = bool(getattr(r, "converged", True))
            else:
                kw = {} if est == "opmm_alpha" else dict(return_guard=False,
                                                         divergence_guard=False,
                                                         alpha_freeze_after=None)
                r = em_patp_estimate(y[None, :], s[None, :], alpha_p_init=None,
                                     alpha_p_fixed=False, **kw)
                h = float(r.H[0, 0])
                alpha_hat, conv = float(r.alpha_p), bool(r.converged)
                n_outer = float(r.n_iter_outer)
            row.update(status="ok", sq_err=(h - H_TRUE) ** 2, alpha_hat=alpha_hat,
                       converged=conv, n_outer=n_outer)
        except Exception as exc:  # noqa: BLE001
            row.update(status=f"failed: {type(exc).__name__}", sq_err=np.nan,
                       alpha_hat=np.nan, converged=False, n_outer=np.nan)
        row["wall_clock_s"] = time.perf_counter() - t0
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--output-dir", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--alpha-gal", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=50, help="replicates per job")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)

    jobs = [(ag, a.n, est, lo, min(lo + a.chunk, a.reps))
            for ag in a.alpha_gal for est in ESTIMATORS
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
    df = pd.DataFrame(rows).sort_values(["data_id", "estimator", "rep"])
    df.to_csv(a.output_dir / "reps.csv", index=False)
    (a.output_dir / "manifest.json").write_text(json.dumps(
        {"script": "scripts/r1_gal_k1.py", "argv": sys.argv[1:],
         "wall_clock_seconds": wall, "n_rows": len(df)}, indent=2))
    print(f"{len(df)} rows, {wall:.1f} s -> {a.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

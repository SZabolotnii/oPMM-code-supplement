"""R1 (Rv3-C2): pragmatic vs exact matrix-GIG E-step, per replicate.

The submitted supplement claims the pragmatic E-step is within 5 % MSE of
the exact one, with two inconsistent descriptions (alpha_MAL in {0.5, 1, 2},
k = 2, n = 256, 30 reps; App. F: k in {2, 4}, n in {256, 512}) and no stored
output. This script produces that output on a valid MAL grid.

Data: ``run.draw_replicate`` on a synthetic MAL ``CellSpec`` (sigma "I"),
alpha_MAL = (k-1)/2 + offset, offset in {0.5, 1.0, 3.0} — the same per-k grid
and seeds as ``config_r1_mal_valid.yaml``, so the rows pair with that run.

Estimators (identical (Y, S, H) per replicate):

* ``ols``
* ``opmm_alpha``            pragmatic E-step, R1 safeguards (default fit)
* ``opmm_alpha_unguarded``  pragmatic E-step, as submitted (no guard/freeze)
* ``opmm_alpha_exact``      exact E-step, ``alpha_mal_init`` = the TRUE
  alpha_MAL (oracle). ``em_patp_exact`` has no R1 safeguards, so the
  like-for-like comparison is exact vs ``opmm_alpha_unguarded``.

(The registry name ``opmm_alpha_exact`` in run.py uses alpha_mal_init = 1.0
for every cell, which at k = 4 is outside alpha > (k-1)/2; that is why this
ablation is a script and not a run.py config.)

Output ``reps.csv`` uses the run.py column names, so
``scripts/r1_paired_bootstrap.py reps.csv --target opmm_alpha_exact
--refs opmm_alpha_unguarded opmm_alpha`` gives the paired CIs.

Usage::

    PYTHONPATH=src python scripts/r1_estep_ablation.py -o <dir>
        [--reps 500] [--k 2 4] [--n 256 512] [--offsets 0.5 1 3]
        [--seed 42] [--workers 8]
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
sys.path.insert(0, str(ROOT))  # experiments.*
sys.path.insert(0, str(ROOT / "src"))  # mv_patp.*

ESTIMATORS = ("ols", "opmm_alpha", "opmm_alpha_unguarded", "opmm_alpha_exact")


def _fit(name: str, Y: np.ndarray, S: np.ndarray, alpha_mal: float):
    from mv_patp.baseline import ols
    from mv_patp.core.em_patp import em_patp_estimate
    from mv_patp.core.em_patp_exact import em_patp_exact_estimate
    if name == "ols":
        return ols.fit(Y, S)
    if name == "opmm_alpha":
        return em_patp_estimate(Y, S)
    if name == "opmm_alpha_unguarded":
        return em_patp_estimate(Y, S, return_guard=False, divergence_guard=False,
                                alpha_freeze_after=None)
    if name == "opmm_alpha_exact":
        return em_patp_exact_estimate(Y, S, alpha_mal_init=alpha_mal)
    raise ValueError(name)


def _job(args: tuple) -> list[dict]:
    k, n, p, offset, est, seed, reps = args
    from experiments.mv_patp_mimo.run import CellSpec, draw_replicate
    alpha = (k - 1) / 2.0 + offset
    cell = CellSpec(k=k, n=n, alpha_MAL=alpha, p=p, M_norm=0.0,
                    sigma_structure="I", replicates=reps, estimator=est,
                    alpha_offset=offset)
    rows = []
    for rep in range(reps):
        Y, S, H = draw_replicate(cell, seed, rep)
        row = dict(data_id=cell.data_id(), estimator=est, k=k, n=n, p=p,
                   sigma_structure="I", alpha_MAL=alpha, alpha_offset=offset,
                   alpha_used=alpha, rep=rep, seed=seed + rep)
        t0 = time.perf_counter()
        try:
            res = _fit(est, Y, S, alpha)
            row.update(status="ok",
                       sq_err=float(np.linalg.norm(np.asarray(res.H) - H) ** 2),
                       alpha_hat=float(getattr(res, "alpha_p", np.nan) or np.nan),
                       converged=bool(getattr(res, "converged", True)),
                       n_outer=float(getattr(res, "n_iter_outer",
                                             getattr(res, "n_iter", np.nan)) or np.nan))
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
    ap.add_argument("--k", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--n", type=int, nargs="+", default=[256, 512])
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--offsets", type=float, nargs="+", default=[0.5, 1.0, 3.0])
    ap.add_argument("--estimators", nargs="+", default=list(ESTIMATORS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args(argv)

    jobs = [(k, n, a.p, off, est, a.seed, a.reps)
            for k in a.k for n in a.n for off in a.offsets for est in a.estimators]
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
        {"script": "scripts/r1_estep_ablation.py", "argv": sys.argv[1:],
         "wall_clock_seconds": wall, "n_rows": len(df)}, indent=2))
    print(f"{len(df)} rows, {wall:.1f} s -> {a.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

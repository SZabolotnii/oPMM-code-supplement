"""Complex-baseband proof-of-concept for oPMM_alpha (reviewer Q7).

Demonstrates that the estimator extends smoothly to the complex case via the
standard *real-equivalent* (widely-linear) embedding already used by the
DeepMIMO / WAIR-D loaders of this paper -- a native Wirtinger-calculus
formulation is deferred to future work (§VI-E).

Complex model
-------------
    Y_c = H_c S_c + N_c,   H_c in C^{k x p}, S_c in C^{p x n}, N_c in C^{k x n},

with circularly-symmetric heavy-tailed noise (complex Student-t, common
real/imag scale per element). The complex problem maps to the real-equivalent
linear model

    Y_r = H_r S_r + N_r,
    Y_r = [[Re Y]; [Im Y]]  (2k x n),
    H_r = [[Re H, -Im H]; [Im H, Re H]]  (2k x 2p),
    S_r = [[Re S]; [Im S]]  (2p x n),
    N_r = [[Re N]; [Im N]]  (2k x n),

on which the unmodified real-valued ``em_patp_estimate`` is run and compared to
OLS and the per-residual robust baselines. ARE = MSE_OLS / MSE_method is taken
on the full real-equivalent channel matrix (both estimators are unconstrained,
so the comparison is fair).

Usage
-----
    PYTHONPATH=src python -m experiments.mv_patp_mimo.run_complex_demo \
        [--k 4] [--p 2] [--n 512] [--df 4] [--reps 200] [--seed 42]

Writes ``results/complex_demo_v1/{cells.csv,manifest.json}``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np


def _real_equiv(H: np.ndarray, S: np.ndarray, N: np.ndarray):
    """Map a complex (H, S, N) triple to its real-equivalent (H_r, S_r, N_r)."""
    H_r = np.block([[H.real, -H.imag], [H.imag, H.real]])         # (2k, 2p)
    S_r = np.concatenate([S.real, S.imag], axis=0)                # (2p, n)
    N_r = np.concatenate([N.real, N.imag], axis=0)                # (2k, n)
    return H_r, S_r, N_r


def _complex_circular_t(k: int, n: int, df: float, rng: np.random.Generator) -> np.ndarray:
    """Circularly-symmetric complex Student-t noise, per-element common scale."""
    w = rng.chisquare(df=df, size=(k, n)) / df                   # shared Re/Im scale
    gr = rng.standard_normal((k, n))
    gi = rng.standard_normal((k, n))
    return (gr + 1j * gi) / np.sqrt(w)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=4, help="complex receive antennas")
    ap.add_argument("--p", type=int, default=2, help="complex transmit streams")
    ap.add_argument("--n", type=int, default=512, help="training length")
    ap.add_argument("--df", type=float, default=4.0, help="complex Student-t dof")
    ap.add_argument("--reps", type=int, default=200, help="MC replicates")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    from mv_patp.baseline import huber_irls, student_t_elementwise, whitened_huber
    from mv_patp.baseline.ols import fit as ols_fit
    from mv_patp.core.em_patp import em_patp_estimate

    methods = {
        "ols": lambda Y, S: ols_fit(Y, S).H,
        "huber_irls": lambda Y, S: huber_irls.fit(Y, S).H,
        "whitened_huber": lambda Y, S: whitened_huber.fit(Y, S).H,
        "student_t_elementwise": lambda Y, S: student_t_elementwise.fit(Y, S).H,
        "opmm_alpha": lambda Y, S: em_patp_estimate(Y, S).H,
    }
    sq_err: dict[str, list[float]] = {m: [] for m in methods}

    k, p, n = args.k, args.p, args.n
    for rep in range(args.reps):
        rng = np.random.default_rng(args.seed + rep)
        H_c = (rng.standard_normal((k, p)) + 1j * rng.standard_normal((k, p))) / np.sqrt(2)
        S_c = (rng.standard_normal((p, n)) + 1j * rng.standard_normal((p, n))) / np.sqrt(2)
        N_c = _complex_circular_t(k, n, args.df, rng) * 0.3
        H_r, S_r, N_r = _real_equiv(H_c, S_c, N_c)
        Y_r = H_r @ S_r + N_r
        for name, fit in methods.items():
            try:
                H_hat = np.asarray(fit(Y_r, S_r))
                sq_err[name].append(float(np.linalg.norm(H_hat - H_r, "fro") ** 2))
            except Exception as exc:  # noqa: BLE001
                print(f"  rep {rep} method {name} failed: {exc}", file=sys.stderr)

    mse = {m: float(np.mean(v)) if v else float("nan") for m, v in sq_err.items()}
    mse_ols = mse["ols"]
    rows = []
    print(f"\nComplex-baseband real-equivalent demo  (k={k}, p={p}, n={n}, "
          f"complex Student-t df={args.df}, M={args.reps})")
    print(f"{'method':<24}{'MSE':>14}{'ARE vs OLS':>14}")
    for m in methods:
        are = mse_ols / mse[m] if mse[m] > 0 else float("nan")
        print(f"{m:<24}{mse[m]:>14.6f}{are:>14.3f}")
        rows.append({"estimator": m, "mse_mean": mse[m], "are_vs_ols": are,
                     "n_replicates": len(sq_err[m])})

    out_dir = args.output_dir or (Path(__file__).resolve().parent / "results" / "complex_demo_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with (out_dir / "cells.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["estimator", "mse_mean", "are_vs_ols", "n_replicates"])
        w.writeheader()
        w.writerows(rows)
    manifest = {
        "run_id": "complex_demo_v1",
        "description": "Complex-baseband real-equivalent embedding proof-of-concept (reviewer Q7)",
        "seed": args.seed, "k": k, "p": p, "n": n, "df": args.df, "replicates": args.reps,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "noise": "circularly-symmetric complex Student-t, common Re/Im scale",
        "embedding": "real-equivalent widely-linear (H_r 2k x 2p, S_r 2p x n)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {out_dir}/cells.csv and manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

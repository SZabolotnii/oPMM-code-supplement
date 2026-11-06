"""Paired-bootstrap CIs of MSE ratios from a per-replicate Monte Carlo table.

Reads ``reps.csv`` written by ``experiments/mv_patp_mimo/run.py
--per-replicate`` (one row per cell x replicate) and, for every data
configuration (``data_id``), reports

* ``ARE`` of each estimator vs OLS, ``MSE(ols) / MSE(est)``, and
* the ratio ``MSE(opmm_alpha) / MSE(<ref>)`` for each robust reference
  (default ``huber_irls`` and ``whitened_huber``),

each with a 95 % percentile CI from a paired bootstrap: replicates are
resampled jointly (the same replicate indices for numerator and
denominator), which is valid because every estimator in a ``data_id`` sees
identical data per replicate. Only replicates where both estimators
succeeded enter a ratio; the count is reported as ``n_pairs``.

Usage::

    python scripts/r1_paired_bootstrap.py <reps.csv> [-o out.csv]
        [--B 2000] [--seed 20261104] [--level 0.95]
        [--refs huber_irls whitened_huber] [--target opmm_alpha]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY_COLS = ("data_id", "k", "n", "p", "sigma_structure", "alpha_used")


def paired_ratio_ci(
    num: np.ndarray, den: np.ndarray, B: int, rng: np.random.Generator,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI of ``mean(num) / mean(den)``.

    ``num`` and ``den`` are aligned per replicate; resampling draws the same
    indices for both (paired bootstrap).
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    if num.shape != den.shape or num.size == 0:
        raise ValueError("num and den must be non-empty and aligned.")
    point = float(num.mean() / den.mean())
    idx = rng.integers(0, num.size, size=(B, num.size))
    boot = num[idx].mean(axis=1) / den[idx].mean(axis=1)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [a, 1.0 - a])
    return point, float(lo), float(hi)


def _pairs(wide: pd.DataFrame, a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    sub = wide[[a, b]].dropna()
    return sub[a].to_numpy(), sub[b].to_numpy()


def summarise(
    reps: pd.DataFrame, B: int = 2000, seed: int = 20261104, level: float = 0.95,
    target: str = "opmm_alpha", refs: tuple[str, ...] = ("huber_irls", "whitened_huber"),
) -> pd.DataFrame:
    """One row per (data_id, comparison)."""
    ok = reps[reps["status"] == "ok"]
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    keys = [c for c in KEY_COLS if c in ok.columns]
    for key_vals, grp in ok.groupby(keys, dropna=False, sort=True):
        meta = dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,)))
        wide = grp.pivot_table(index="rep", columns="estimator", values="sq_err",
                               aggfunc="first")
        comparisons: list[tuple[str, str, str]] = []
        if "ols" in wide.columns:
            # ARE vs OLS = MSE(ols) / MSE(est)
            comparisons += [("are_vs_ols", est, "ols") for est in wide.columns
                            if est != "ols"]
        if target in wide.columns:
            comparisons += [(f"mse_ratio_{target}_over_{r}", target, r)
                            for r in refs if r in wide.columns]
        for kind, est, ref in comparisons:
            if kind == "are_vs_ols":
                num, den = _pairs(wide, "ols", est)
            else:
                num, den = _pairs(wide, est, ref)
            if num.size < 2:
                continue
            point, lo, hi = paired_ratio_ci(num, den, B, rng, level)
            out.append({**meta, "comparison": kind, "estimator": est,
                        "reference": ref, "n_pairs": int(num.size),
                        "ratio": point, "ci_lo": lo, "ci_hi": hi,
                        "B": B, "level": level})
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reps_csv", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output CSV (default: <reps_csv dir>/paired_bootstrap.csv).")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--target", default="opmm_alpha")
    ap.add_argument("--refs", nargs="+", default=["huber_irls", "whitened_huber"])
    args = ap.parse_args(argv)

    reps = pd.read_csv(args.reps_csv)
    res = summarise(reps, B=args.B, seed=args.seed, level=args.level,
                    target=args.target, refs=tuple(args.refs))
    out = args.output or args.reps_csv.with_name("paired_bootstrap.csv")
    res.to_csv(out, index=False)
    print(f"{len(res)} comparisons over {res['data_id'].nunique() if len(res) else 0} "
          f"data configurations -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

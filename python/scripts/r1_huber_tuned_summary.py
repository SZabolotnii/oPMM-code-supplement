"""Summarise r1_huber_tuned (Rv1-C11) from reps.csv + paired_bootstrap.csv.

Prints the markdown tables used in huber_tuned.md and writes the source rows
next to the run's outputs:
  r1_huber_tuned/huber_tuned_source_rows.csv   (rows of paired_bootstrap.csv quoted)
  r1_huber_tuned/huber_tuned_adapt_vs_1p345.csv (paired CI of
      MSE(huber_adaptive_c)/MSE(huber_irls), computed here with the same
      paired_ratio_ci, B = 2000, seed 20261104)

Usage (from python/):
  python scripts/r1_huber_tuned_summary.py ../data/results
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_paired_bootstrap import paired_ratio_ci  # noqa: E402


def reps_path(d):
    """reps.csv as written by run.py, or the gzipped copy in the published snapshot."""
    p = d / "reps.csv"
    return p if p.exists() else d / "reps.csv.gz"


def md(df, index=True):
    """Markdown table without the optional tabulate dependency."""
    d = df.reset_index() if index else df
    d = d.astype(object).where(d.notna(), "")
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in d.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(out)


root = Path(sys.argv[1])
run_dir = root / "r1_huber_tuned"
new = pd.read_csv(reps_path(run_dir))
old = pd.read_csv(reps_path(root / "r1_consolidated"))
pb = pd.read_csv(run_dir / "paired_bootstrap.csv")

FIXED = ["huber_c0p7", "huber_c1p0", "huber_c1p345", "huber_c1p7", "huber_c2p0", "huber_c2p5"]
CVAL = {"huber_c0p7": 0.7, "huber_c1p0": 1.0, "huber_c1p345": 1.345, "huber_c1p7": 1.7,
        "huber_c2p0": 2.0, "huber_c2p5": 2.5}
NOISE = ["student_t_3", "student_t_4", "student_t_5", "student_t_1000", "gaussian"]
NL = {"student_t_3": "t3", "student_t_4": "t4", "student_t_5": "t5",
      "student_t_1000": "t1000", "gaussian": "Gauss"}
CELLS = [(s, k, n) for s in NOISE for k in (2, 4, 8) for n in (256, 512, 1024)]


def lab(s, k, n):
    return f"{NL[s]} k{k} n{n}"


print("## status / convergence")
print(md(new.groupby("estimator").agg(n=("status", "size"),
                                      ok=("status", lambda x: int((x == "ok").sum())),
                                      converged=("converged", "mean"))))

print("\n## reproduction vs r1_consolidated")
key = ["data_id", "estimator", "rep"]
for est in ["ols", "opmm_alpha", "huber_irls"]:
    a = new[new.estimator == est][key + ["sq_err"]]
    b = old[old.estimator == est][key + ["sq_err"]]
    m = a.merge(b, on=key, suffixes=("_new", "_old"))
    eq = m.sq_err_new == m.sq_err_old
    print(f"{est}: rows new={len(a)} old={len(b)} matched={len(m)} "
          f"bit-identical={int(eq.sum())} max|diff|={np.abs(m.sq_err_new - m.sq_err_old).max():.3e}")
a = new[new.estimator == "huber_irls"].set_index(["data_id", "rep"]).sq_err
b = new[new.estimator == "huber_c1p345"].set_index(["data_id", "rep"]).sq_err
j = pd.concat([a, b], axis=1, keys=["irls", "c1p345"]).dropna()
print(f"huber_c1p345 vs huber_irls (this run): pairs={len(j)} "
      f"bit-identical={int((j.irls == j.c1p345).sum())}")

are = pb[pb.comparison == "are_vs_ols"]
pt = are.pivot_table(index=["sigma_structure", "k", "n"], columns="estimator", values="ratio")
pt = pt.reindex(pd.MultiIndex.from_tuples(CELLS, names=["noise", "k", "n"]))
best = pt[FIXED].idxmax(axis=1)
show = pt[["opmm_alpha", "huber_irls", "huber_c0p7", "huber_c1p0", "huber_c1p7", "huber_c2p0",
           "huber_c2p5", "huber_adaptive_c"]].round(3)
show["best fixed c"] = best.map(CVAL)
show["ARE best fixed"] = pt[FIXED].max(axis=1).round(3)
show.index = [lab(*c) for c in show.index]
show.columns = ["oPMM", "c=1.345", "c=0.7", "c=1.0", "c=1.7", "c=2.0", "c=2.5", "adaptive",
                "best fixed c", "ARE best fixed"]
print("\n## ARE vs OLS per cell (point; MSE(ols)/MSE(est))")
print(md(show))

rat = pb[pb.comparison.str.startswith("mse_ratio_opmm_alpha_over_")]
rows = []
for (s, k, n), bc in best.items():
    sub = rat[(rat.sigma_structure == s) & (rat.k == k) & (rat.n == n)].set_index("reference")
    f = lambda r: f"{r.ratio:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"  # noqa: E731
    rows.append({"cell": lab(s, k, n), "oPMM / c=1.345": f(sub.loc["huber_irls"]),
                 "oPMM / adaptive": f(sub.loc["huber_adaptive_c"]), "best fixed c": CVAL[bc],
                 "oPMM / best fixed c": f(sub.loc[bc])})
print("\n## MSE(oPMM)/MSE(ref), paired 95% CI (>1: oPMM worse)")
print(md(pd.DataFrame(rows), index=False))

print("\n## range and CI counts by noise")
cnt = []
for ref in ["huber_irls", "huber_adaptive_c"]:
    for s in NOISE:
        g = rat[(rat.reference == ref) & (rat.sigma_structure == s)]
        cnt.append({"reference": ref, "noise": NL[s], "min": round(g.ratio.min(), 4),
                    "max": round(g.ratio.max(), 4), "CI above 1": int((g.ci_lo > 1).sum()),
                    "CI below 1": int((g.ci_hi < 1).sum()), "cells": len(g)})
print(md(pd.DataFrame(cnt), index=False))

rng = np.random.default_rng(20261104)
rows = []
ok = new[new.status == "ok"]
for s, k, n in CELLS:
    g = ok[(ok.sigma_structure == s) & (ok.k == k) & (ok.n == n)]
    w = g.pivot_table(index="rep", columns="estimator", values="sq_err",
                      aggfunc="first")[["huber_adaptive_c", "huber_irls"]].dropna()
    p, lo, hi = paired_ratio_ci(w.huber_adaptive_c.to_numpy(), w.huber_irls.to_numpy(), 2000, rng)
    rows.append({"sigma_structure": s, "k": k, "n": n, "n_pairs": len(w),
                 "ratio": p, "ci_lo": lo, "ci_hi": hi})
adv = pd.DataFrame(rows)
disp = pd.DataFrame({"cell": [lab(r.sigma_structure, r.k, r.n) for r in adv.itertuples()],
                     "MSE(adaptive)/MSE(c=1.345)": [f"{r.ratio:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
                                                    for r in adv.itertuples()]})
print("\n## MSE(huber_adaptive_c)/MSE(huber_irls), paired 95% CI (B=2000, seed 20261104)")
print(md(disp, index=False))

ad = ok[ok.estimator == "huber_adaptive_c"]
d = ad.groupby("sigma_structure").c_hat.describe(percentiles=[.1, .25, .5, .75, .9]).reindex(NOISE)
d["share at 3.0"] = ad.groupby("sigma_structure").c_hat.apply(lambda x: (x >= 2.999).mean())
d["share at 0.5"] = ad.groupby("sigma_structure").c_hat.apply(lambda x: (x <= 0.501).mean())
d["mode"] = ad.groupby("sigma_structure").c_hat.agg(lambda x: x.round(1).mode().iloc[0])
print("\n## chosen c by noise (pooled over k, n)")
print(md(d.round(3)))
print("\n## median [IQR] of chosen c by noise x n (pooled over k)")
print(md(ad.pivot_table(index="sigma_structure", columns="n", values="c_hat",
                        aggfunc=lambda x: f"{x.median():.1f} [{x.quantile(.25):.1f}, "
                                          f"{x.quantile(.75):.1f}]").reindex(NOISE)))
print("\n## median chosen c by noise x k (pooled over n)")
print(md(ad.pivot_table(index="sigma_structure", columns="k", values="c_hat",
                        aggfunc="median").reindex(NOISE)))

src = pb[(pb.comparison == "are_vs_ols") | pb.comparison.str.startswith("mse_ratio_opmm_alpha_over_")]
src.to_csv(run_dir / "huber_tuned_source_rows.csv", index=False)
adv.to_csv(run_dir / "huber_tuned_adapt_vs_1p345.csv", index=False)
print(f"\nsource rows written: {len(src)} + {len(adv)}")

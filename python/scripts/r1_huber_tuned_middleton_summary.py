"""Summarise r1_middleton_tuned (Rv1-C11 on Middleton Class A).

Reads r1_middleton_tuned/{reps.csv, paired_bootstrap.csv,
paired_bootstrap_adapt_vs_irls.csv} and r1_middleton/{reps.csv,
paired_bootstrap.csv}; prints the checks and the markdown tables used in
huber_tuned_middleton.md and writes the quoted rows to
  r1_middleton_tuned/huber_tuned_middleton_source_rows.csv

Usage (from python/):
  python scripts/r1_huber_tuned_middleton_summary.py ../data/results
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd


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
run_dir = root / "r1_middleton_tuned"
new = pd.read_csv(reps_path(run_dir))
old = pd.read_csv(reps_path(root / "r1_middleton"))
pb = pd.read_csv(run_dir / "paired_bootstrap.csv")
pba = pd.read_csv(run_dir / "paired_bootstrap_adapt_vs_irls.csv")
pb_old = pd.read_csv(root / "r1_middleton/paired_bootstrap.csv")

FIXED = ["huber_c0p7", "huber_c1p0", "huber_c1p345", "huber_c1p7", "huber_c2p0", "huber_c2p5"]
CVAL = {"huber_c0p7": 0.7, "huber_c1p0": 1.0, "huber_c1p345": 1.345, "huber_c1p7": 1.7,
        "huber_c2p0": 2.0, "huber_c2p5": 2.5}
NOISE = ["middleton_A1p0_G0p1", "middleton_A0p35_G0p1", "middleton_A0p1_G0p1"]
NL = {"middleton_A1p0_G0p1": "A=1.0", "middleton_A0p35_G0p1": "A=0.35",
      "middleton_A0p1_G0p1": "A=0.1"}

print("## status / convergence")
print(md(new.groupby("estimator").agg(n=("status", "size"),
                                      ok=("status", lambda x: int((x == "ok").sum())),
                                      converged=("converged", "mean"))))

print("\n## reproduction vs r1_middleton (sq_err per data_id x rep)")
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

print("\n## point ratios vs r1_middleton/paired_bootstrap.csv (same pairs -> same point)")
for est in ["opmm_alpha", "huber_irls"]:
    for s in NOISE:
        r_new = pb[(pb.comparison == "are_vs_ols") & (pb.estimator == est) & (pb.sigma_structure == s)]
        r_old = pb_old[(pb_old.comparison == "are_vs_ols") & (pb_old.estimator == est)
                       & (pb_old.sigma_structure == s)]
        print(f"ARE {est} {NL[s]}: new {r_new.ratio.iloc[0]!r} [{r_new.ci_lo.iloc[0]:.4f}, "
              f"{r_new.ci_hi.iloc[0]:.4f}] old {r_old.ratio.iloc[0]!r} "
              f"[{r_old.ci_lo.iloc[0]:.4f}, {r_old.ci_hi.iloc[0]:.4f}]")
for s in NOISE:
    r_new = pb[(pb.comparison == "mse_ratio_opmm_alpha_over_huber_irls") & (pb.sigma_structure == s)]
    r_old = pb_old[(pb_old.comparison == "mse_ratio_opmm_alpha_over_huber_irls")
                   & (pb_old.sigma_structure == s)]
    print(f"oPMM/Huber1.345 {NL[s]}: new {r_new.ratio.iloc[0]!r} [{r_new.ci_lo.iloc[0]:.4f}, "
          f"{r_new.ci_hi.iloc[0]:.4f}] old {r_old.ratio.iloc[0]!r} "
          f"[{r_old.ci_lo.iloc[0]:.4f}, {r_old.ci_hi.iloc[0]:.4f}]")

are = pb[pb.comparison == "are_vs_ols"]
fmt = lambda r: f"{r.ratio:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]"  # noqa: E731
ests = ["opmm_alpha", "huber_irls", "huber_adaptive_c"] + [e for e in FIXED if e != "huber_c1p345"]
rows = []
for s in NOISE:
    g = are[are.sigma_structure == s].set_index("estimator")
    rows.append({"A": NL[s], **{e: fmt(g.loc[e]) for e in ests}})
print("\n## ARE vs OLS, paired 95% CI (MSE(ols)/MSE(est), >1 beats OLS)")
print(md(pd.DataFrame(rows), index=False))

rat = pb[pb.comparison.str.startswith("mse_ratio_opmm_alpha_over_")]
rows = []
for s in NOISE:
    g = are[are.sigma_structure == s].set_index("estimator")
    bc = g.loc[FIXED].ratio.idxmax()
    sub = rat[rat.sigma_structure == s].set_index("reference")
    ad = pba[(pba.comparison == "mse_ratio_huber_adaptive_c_over_huber_irls")
             & (pba.sigma_structure == s)].iloc[0]
    rows.append({"A": NL[s],
                 "ARE oPMM": f"{g.loc['opmm_alpha'].ratio:.3f}",
                 "ARE c=1.345": f"{g.loc['huber_irls'].ratio:.3f}",
                 "ARE c_hat": f"{g.loc['huber_adaptive_c'].ratio:.3f}",
                 "oracle c": CVAL[bc], "ARE oracle": f"{g.loc[bc].ratio:.3f}",
                 "oPMM / c=1.345": fmt(sub.loc["huber_irls"]),
                 "oPMM / c_hat": fmt(sub.loc["huber_adaptive_c"]),
                 "oPMM / oracle": fmt(sub.loc[bc]),
                 "c_hat / c=1.345": fmt(ad)})
print("\n## per A summary (>1 in MSE ratios: numerator worse)")
print(md(pd.DataFrame(rows), index=False))

print("\n## all MSE(oPMM)/MSE(ref) rows")
rows = []
for s in NOISE:
    sub = rat[rat.sigma_structure == s].set_index("reference")
    rows.append({"A": NL[s], **{r: fmt(sub.loc[r]) for r in
                                ["huber_irls", "huber_adaptive_c"] + [e for e in FIXED
                                                                      if e != "huber_c1p345"]}})
print(md(pd.DataFrame(rows), index=False))

ok = new[new.status == "ok"]
adc = ok[ok.estimator == "huber_adaptive_c"]
print("\n## chosen c (c_hat of huber_adaptive_c), 500 fits per A")
d = adc.groupby("sigma_structure").c_hat.describe(percentiles=[.1, .25, .5, .75, .9]).reindex(NOISE)
d["IQR"] = d["75%"] - d["25%"]
d["share at 0.5"] = adc.groupby("sigma_structure").c_hat.apply(lambda x: (x <= 0.501).mean())
d["share at 3.0"] = adc.groupby("sigma_structure").c_hat.apply(lambda x: (x >= 2.999).mean())
d["mode"] = adc.groupby("sigma_structure").c_hat.agg(lambda x: x.round(1).mode().iloc[0])
d.index = [NL[s] for s in d.index]
print(md(d.round(3)))
print("\n## c_hat frequency table")
ft = adc.assign(c=adc.c_hat.round(1)).pivot_table(index="c", columns="sigma_structure",
                                                   values="rep", aggfunc="size",
                                                   fill_value=0)[NOISE]
ft.columns = [NL[s] for s in ft.columns]
print(md(ft))

src = pd.concat([
    pb[(pb.comparison == "are_vs_ols") | pb.comparison.str.startswith("mse_ratio_opmm_alpha_over_")]
    .assign(source="r1_middleton_tuned/paired_bootstrap.csv"),
    pba[pba.comparison == "mse_ratio_huber_adaptive_c_over_huber_irls"]
    .assign(source="r1_middleton_tuned/paired_bootstrap_adapt_vs_irls.csv"),
])
src.to_csv(run_dir / "huber_tuned_middleton_source_rows.csv", index=False)
print(f"\nsource rows written: {len(src)}")
